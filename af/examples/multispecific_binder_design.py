import os
import re
from scipy.special import softmax
import numpy as np
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict

def get_pdb(pdb_code=""):
    """
    Fetches a PDB file from a given code or local path.
    - If pdb_code is a 4-character string, fetches from RCSB.
    - If pdb_code is a UniProt ID, fetches from AlphaFold DB.
    - If pdb_code is a file path, uses the local file.
    """
    if os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        print(f"Fetching PDB {pdb_code} from RCSB...")
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"
    else:
        print(f"Fetching AlphaFoldDB model for {pdb_code}...")
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
        return f"AF-{pdb_code}-F1-model_v3.pdb"

def run_design():
    # -------------------
    # --- Input Params ---
    # -------------------
    
    # Target Info
    pdb_code = "1YVA"
    target_chain = "A"
    target_hotspot = None  # Example: "1-10,12,15"
    
    # Binder Info
    binder_len = 4
    binder_seq_init = None # Example: "ACE..."
    
    # Model Config
    use_multimer = False
    num_recycles = 0
    num_models = 1
    # MODIFICATION: Use only one model for faster testing
    num_models = 1
    # MODIFICATION: Explicitly define the smallest model to use
    model_name_to_use = "model_1_ptm"
    
    # Optimizer Config
    optimizer = "pssm_semigreedy" # "pssm_semigreedy", "3stage", "semigreedy", etc.
    
    # Advanced GD settings
    GD_method = "sgd"
    learning_rate = 0.1
    norm_seq_grad = True
    dropout = True

    # ------------------------
    # --- Multispecific Setup ---
    # ------------------------
    
    # For this example, we duplicate the same target to simulate a bispecific design.
    # For a true bispecific design, define two different targets here.
    # The 'weight' can be negative to penalize binding to a specific target.
    targets = [
        {
            "pdb_filename": get_pdb(pdb_code),
            "target_chain": target_chain,
            #"hotspot": target_hotspot,
            "weight": 1.0
        },
        {
            "pdb_filename": get_pdb('2V9B'),
            "target_chain": target_chain,
            #"hotspot": target_hotspot,
            "weight": -0.1
        }
    ]

    # -------------------
    # --- Model Setup ---
    # -------------------
    clear_mem()
    model = mk_afdesign_model(protocol="multispecific",
                              use_multimer=use_multimer,
                              num_recycles=num_recycles,
                              recycle_mode="sample",
                              model_names = [model_name_to_use],
                              )

    model.prep_inputs(targets=targets, binder_len=binder_len)
    
    print(f"Designing a binder of length {model._binder_len} against {len(targets)} targets.")

    # ----------------
    # --- Run Design ---
    # ----------------
    model.restart(seq=binder_seq_init)
    model.set_optimizer(optimizer=GD_method,
                        learning_rate=learning_rate,
                        norm_seq_grad=norm_seq_grad)
    
    models_to_use = model._model_names[:num_models]

    flags = {"num_recycles": num_recycles,
             "models": models_to_use,
             "dropout": dropout}

    if optimizer == "3stage":
        model.design_3stage(120, 60, 10, **flags)
    elif optimizer == "pssm_semigreedy":
        model.design_pssm_semigreedy(4, 2, **flags)
    elif optimizer == "semigreedy":
        model.design_pssm_semigreedy(0, 32, **flags)
    elif optimizer == "pssm":
        model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
        model.design_soft(32, num_models=1, **flags)
        flags.update({"dropout": False, "save_best": True})
        model.design_soft(10, num_models=num_models, **flags)
    else:
        design_function = {
            "logits": model.design_logits,
            "soft": model.design_soft,
            "hard": model.design_hard
        }.get(optimizer)
        
        if design_function:
            design_function(120, num_models=1, ramp_recycles=True, **flags)
            flags.update({"dropout": False, "save_best": True})
            design_function(10, num_models=num_models, **flags)
        else:
            print(f"ERROR: Optimizer '{optimizer}' not recognized.")
            return

    # --------------
    # --- Save & Print ---
    # --------------
    output_pdb_filename = f"{model.protocol}_design.pdb"
    model.save_pdb(output_pdb_filename)
    print(f"\nFinished! Designed structure saved to: {output_pdb_filename}")
    
    # Print final sequence and metrics
    final_sequences = model.get_seqs()
    print("\nDesigned Sequences:")
    for i, seq in enumerate(final_sequences):
        print(f"  Sequence {i+1}: {seq}")

    if "best" in model._tmp and "aux" in model._tmp["best"]:
        print("\nBest Model Metrics:")
        print(model._tmp["best"]["aux"]["log"])
    else:
        print("\nFinal Model Metrics:")
        print(model.aux["log"])


if __name__ == "__main__":
    run_design()
