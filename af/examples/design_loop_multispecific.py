import argparse
import copy
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.af.prep import get_multi_id, prep_pdb

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


DEFAULT_WEIGHTS = {
    "seq_ent": 0.0,
    "plddt": 0.1,
    "pae": 0.0,
    "exp_res": 0.0,
    "helix": 0.0,
    "con": 0.0,
    "i_con": 1.0,
    "i_pae": 0.0,
    "ptmenergy": 0.0,
}

WEIGHT_ALIASES = {
    "pae_intra": "pae",
    "pae_inter": "i_pae",
    "con_intra": "con",
    "con_inter": "i_con",
}


def load_settings(settings_path):
    with open(settings_path, encoding="utf-8") as handle:
        return json.load(handle)


def get_target_lengths(targets):
    lengths = []
    for target in targets:
        pdb_info = prep_pdb(
            target["pdb_filename"],
            chain=target["target_chain"],
            ignore_missing=True,
        )
        lengths.append(int(pdb_info["residue_index"].shape[0]))
    return lengths


def validate_settings(settings):
    required_top_level = [
        "data_dir",
        "num_designs",
        "binder_length",
        "targets",
        "iterations",
        "contact",
        "weights",
        "output",
    ]
    missing = [key for key in required_top_level if key not in settings]
    if missing:
        raise ValueError(f"Missing required settings keys: {missing}")

    targets = settings["targets"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("'targets' must be a non-empty list")

    required_target_keys = {"pdb_filename", "target_chain"}
    for index, target in enumerate(targets):
        missing_target_keys = sorted(required_target_keys.difference(target))
        if missing_target_keys:
            raise ValueError(
                f"Target {index} is missing required keys: {missing_target_keys}"
            )

    lengths = get_target_lengths(targets)
    if len(set(lengths)) != 1:
        raise ValueError(
            "Multispecific binder design currently expects all targets to have "
            f"the same parsed length. Got lengths: {lengths}"
        )

    binder_sequence_init = settings.get("binder_sequence_init")
    if binder_sequence_init in {"", None}:
        settings["binder_sequence_init"] = None
    else:
        if not isinstance(binder_sequence_init, str):
            raise ValueError("'binder_sequence_init' must be a string when provided")
        binder_sequence_init = binder_sequence_init.strip().upper()
        invalid_amino_acids = sorted(
            {amino_acid for amino_acid in binder_sequence_init if amino_acid not in VALID_AMINO_ACIDS}
        )
        if invalid_amino_acids:
            raise ValueError(
                "'binder_sequence_init' contains invalid amino acids: "
                f"{invalid_amino_acids}"
            )
        if len(binder_sequence_init) != settings["binder_length"]:
            raise ValueError(
                "'binder_sequence_init' length must match 'binder_length'. "
                f"Got {len(binder_sequence_init)} and {settings['binder_length']}"
            )
        settings["binder_sequence_init"] = binder_sequence_init

    fix_pos = settings.get("fix_pos")
    if fix_pos == "":
        settings["fix_pos"] = None
    elif fix_pos is not None and settings.get("binder_sequence_init") is None:
        raise ValueError(
            "'fix_pos' requires 'binder_sequence_init' so fixed residues have "
            "a reference sequence"
        )


def normalize_weights(weight_settings):
    weights = DEFAULT_WEIGHTS.copy()
    for key, value in weight_settings.items():
        canonical_key = WEIGHT_ALIASES.get(key, key)
        if canonical_key in weights:
            weights[canonical_key] = value
    return weights


def initialize_model(settings):
    clear_mem()
    model_settings = settings.get("model", {})
    optimizer_settings = settings.get("optimizer", {})

    af_model = mk_afdesign_model(
        protocol="multispecific",
        use_multimer=settings.get("use_multimer", False),
        data_dir=settings["data_dir"],
        recycle_mode=model_settings.get("recycle_mode", "sample"),
        model_names=model_settings.get("model_names"),
    )

    targets = copy.deepcopy(settings["targets"])
    af_model.prep_inputs(
        targets=targets,
        binder_len=settings["binder_length"],
        binder_seq=settings.get("binder_sequence_init"),
        binder_chain=settings.get("binder_chain"),
        fix_pos=settings.get("fix_pos"),
    )

    contact_settings = settings["contact"]
    af_model.opt["con"].update(
        {
            "num": contact_settings["intra_contact_number"],
            "cutoff": contact_settings["intra_contact_distance"],
            "binary": False,
            "seqsep": 9,
        }
    )
    af_model.opt["i_con"].update(
        {
            "num": contact_settings["inter_contact_number"],
            "cutoff": contact_settings["inter_contact_distance"],
            "binary": False,
        }
    )

    af_model.opt["weights"].update(normalize_weights(settings["weights"]))

    af_model.set_optimizer(
        optimizer=optimizer_settings.get("method", "sgd"),
        learning_rate=optimizer_settings.get("learning_rate", 0.1),
        norm_seq_grad=optimizer_settings.get("norm_seq_grad", True),
    )
    return af_model


def get_best_aux(af_model):
    if "best" in af_model._tmp and "aux" in af_model._tmp["best"]:
        return af_model._tmp["best"]["aux"]
    return af_model.aux


def summarize_metrics(af_model, aux):
    binder_plddt = float(np.mean(aux["plddt"][-af_model._binder_len:]))
    metrics = {
        "loss": float(aux["log"]["loss"]),
        "binder_plddt": binder_plddt,
        "ptm": float(aux.get("ptm", np.nan)),
        "i_ptm": float(aux.get("i_ptm", np.nan)),
        "ptmenergy": float(aux.get("ptmenergy", np.nan)),
    }
    for key in ["con", "i_con", "pae", "i_pae", "seq_ent", "exp_res", "helix"]:
        if key in aux["log"]:
            metrics[key] = float(aux["log"][key])
    return metrics


def get_target_label(target, index):
    raw_label = target.get("name", f"target_{index + 1}")
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in raw_label
    )


def collect_per_target_metrics(af_model, settings):
    model_settings = settings.get("model", {})
    af_model.set_seq(seq=af_model.get_seqs(get_best=True)[0])

    model_nums = af_model._get_model_nums(
        num_models=model_settings.get("num_models", 1),
        sample_models=False,
    )
    num_recycles = model_settings.get("num_recycles", 0)

    per_target_metrics = []
    for index, (target_settings, target_data) in enumerate(
        zip(settings["targets"], af_model._targets)
    ):
        af_model._target_len = target_data["target_len"]
        af_model._binder_len = target_data["binder_len"]
        af_model._lengths = target_data["lengths"]

        af_model._inputs.update(
            {
                "batch": target_data["batch"],
                "residue_index": target_data["residue_index"],
                "rm_template_seq": target_data["rm_template_seq"],
                "rm_template_sc": target_data["rm_template_sc"],
            }
        )
        af_model._inputs.update(get_multi_id(af_model._lengths))

        if target_data["hotspot"] is not None:
            af_model.opt["hotspot"] = target_data["hotspot"]
        else:
            af_model.opt.pop("hotspot", None)

        aux_by_model = []
        for model_num in model_nums:
            model_params = af_model._model_params[model_num]
            aux_by_model.append(
                af_model._recycle(
                    model_params,
                    num_recycles=num_recycles,
                    backprop=False,
                )
            )

        avg_aux = jax.tree_util.tree_map(
            lambda *xs: xs[0]
            if np.issubdtype(np.asarray(xs[0]).dtype, np.integer)
            else np.mean(np.stack(xs), axis=0),
            *aux_by_model,
        )

        per_target_metrics.append(
            {
                "label": get_target_label(target_settings, index),
                "weight": float(target_settings.get("weight", 1.0)),
                "i_ptm": float(avg_aux["i_ptm"]),
                "ptmenergy": float(avg_aux["ptmenergy"]),
            }
        )

    return per_target_metrics


def run_optimization(af_model, settings, seed):
    model_settings = settings.get("model", {})
    optimizer_settings = settings.get("optimizer", {})
    iteration_settings = settings["iterations"]

    af_model.restart(
        seed=seed,
        seq=settings.get("binder_sequence_init"),
        reset_opt=False,
    )
    af_model.set_optimizer(
        optimizer=optimizer_settings.get("method", "sgd"),
        learning_rate=optimizer_settings.get("learning_rate", 0.1),
        norm_seq_grad=optimizer_settings.get("norm_seq_grad", True),
    )

    flags = {
        "num_recycles": model_settings.get("num_recycles", 0),
        "num_models": model_settings.get("num_models", 1),
        "sample_models": model_settings.get("sample_models", True),
        "dropout": optimizer_settings.get("dropout", True),
    }

    print("Stage 1: Test Logits")
    af_model.design_logits(
        iters=iteration_settings["initial_logit"],
        e_soft=0.9,
        ramp_recycles=False,
        save_best=True,
        **flags,
    )
    initial_aux = get_best_aux(af_model)
    initial_plddt = float(np.mean(initial_aux["plddt"][-af_model._binder_len:]))

    threshold = settings.get("initial_plddt_threshold", 0.65)
    if initial_plddt < threshold:
        print(f"Initial trajectory pLDDT bad: {initial_plddt:.3f}")
        return None

    print(f"Initial trajectory pLDDT good, continuing: {initial_plddt:.3f}")

    print("Stage 2: Additional Logits Optimisation")
    af_model.design_logits(
        iters=iteration_settings["logit"],
        soft=0.9,
        e_soft=1.0,
        ramp_recycles=False,
        save_best=True,
        **flags,
    )

    print("Stage 3: Soft iterations")
    af_model.design_soft(
        iters=iteration_settings["soft"],
        e_temp=1e-2,
        ramp_recycles=False,
        save_best=True,
        **flags,
    )

    af_model.clear_best()
    print("Stage 4: Hard iterations")
    af_model.design_hard(
        iters=iteration_settings["hard"],
        temp=1e-2,
        dropout=False,
        ramp_recycles=False,
        save_best=True,
        num_recycles=flags["num_recycles"],
        num_models=flags["num_models"],
        sample_models=flags["sample_models"],
    )

    print("Stage 5: Semigreedy")
    af_model.design_pssm_semigreedy(
        soft_iters=0,
        hard_iters=iteration_settings["semigreedy"],
        tries=iteration_settings.get("semigreedy_tries", 10),
        dropout=False,
        num_models=1,
        ramp_models=False,
        save_best=True,
        num_recycles=flags["num_recycles"],
    )

    return af_model


def save_outputs(af_model, settings, seed, design_index):
    output_settings = settings["output"]
    output_dir = Path(output_settings["colabdesign_output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_aux = get_best_aux(af_model)
    metrics = summarize_metrics(af_model, best_aux)
    per_target_metrics = collect_per_target_metrics(af_model, settings)
    seq = af_model.get_seqs(get_best=True)[0]

    stem = (
        f"{seed}_{metrics['loss']:.2f}_{metrics['binder_plddt']:.2f}_"
        f"{metrics['i_ptm']:.2f}_{metrics['ptmenergy']:.2f}"
    )

    pdb_path = output_dir / f"{stem}.pdb"
    af_model.save_pdb(str(pdb_path), get_best=True)

    fasta_path = Path(output_settings["fasta_file"])
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    target_header = "|".join(
        (
            f"{target_metric['label']}_i_ptm={target_metric['i_ptm']:.2f}|"
            f"{target_metric['label']}_ptmenergy={target_metric['ptmenergy']:.2f}"
        )
        for target_metric in per_target_metrics
    )
    header = (
        f">{seed}|design_{design_index}|loss={metrics['loss']:.2f}|"
        f"binder_plddt={metrics['binder_plddt']:.2f}|{target_header}"
    )
    with open(fasta_path, "a", encoding="utf-8") as handle:
        handle.write(f"{header}\n{seq}\n")

    traj_path = output_dir / f"{stem}_traj.pkl"
    with open(traj_path, "wb") as handle:
        pickle.dump(af_model._tmp["traj"], handle)

    log_path = output_dir / f"{stem}_log.pkl"
    with open(log_path, "wb") as handle:
        pickle.dump(af_model._tmp["log"], handle)

    summary_path = output_dir / f"{stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": seed,
                "sequence": seq,
                "metrics": metrics,
                "per_target_metrics": per_target_metrics,
                "targets": settings["targets"],
            },
            handle,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Multispecific binder design loop")
    parser.add_argument(
        "--settings",
        type=str,
        required=True,
        help="Path to a JSON settings file",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    validate_settings(settings)

    successful_designs = 0
    while successful_designs < settings["num_designs"]:
        seed = int(np.random.randint(0, 999999))
        af_model = initialize_model(settings)
        result = run_optimization(af_model, settings, seed)
        if result is None:
            continue

        successful_designs += 1
        save_outputs(af_model, settings, seed, successful_designs)


if __name__ == "__main__":
    main()
