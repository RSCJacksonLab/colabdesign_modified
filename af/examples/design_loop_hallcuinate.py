import argparse
import os
import json
import numpy as np
from colabdesign import mk_afdesign_model, clear_mem
import pickle

def load_settings(settings_path):
    with open(settings_path) as f:
        return json.load(f)

def initialize_model(settings):
    clear_mem()

    samples = np.arange(min(settings["lengths"]), max(settings["lengths"]) + 1)
    length = np.random.choice(samples)
  
    af_model = mk_afdesign_model(protocol="binder", 
                                use_multimer=True,
                                data_dir=settings["data_dir"],
                                )
    
    af_model.prep_inputs(pdb_filename=settings["input_pdb"],
                        chain=settings["chains"],
                        hotspot=settings["hotspot_residues"],
                        binder_len=length
                        )
    # update contact distances
    af_model.opt["con"].update({"num":settings["intra_contact_number"],"cutoff":settings["intra_contact_distance"],"binary":False,"seqsep":9})
    af_model.opt["i_con"].update({"num":settings["inter_contact_number"],"cutoff":settings["inter_contact_distance"],"binary":False})

    af_model.opt["weights"].update({
        "pae": settings["weights"]["pae_intra"],
        "plddt": settings["weights"]["plddt"],
        "i_pae": settings["weights"]["pae_inter"],
        "con": settings["weights"]["con_intra"],
        "i_con": settings["weights"]["con_inter"],
        "rg": settings["weights"]["rg"],
        "i_ptm": settings["weights"]["iptm"],
        "ptmenergy": settings["weights"]["ptmenergy"],
        "dgram_cce": settings["weights"]["dgram_cce"]
    })
    return af_model

def run_optimization(af_model, settings, seed):    
    # Stage 1: Initial Logits
    print("Stage 1: Test Logits")
    af_model.design_logits(iters=settings["iterations"]["initial_logit"], e_soft=0.9, ramp_recycles=False, save_best=True)
    initial_plddt = round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-af_model._binder_len:]),2)
    
    if initial_plddt < 0.65:
        print("Initial trajectory pLDDT bad: "+str(initial_plddt))
        return None, None, None
    else:
        print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
    # Stage 2: Additional Logits
    print("Stage 1: Additional Logits Optimisation")
    af_model.design_logits(iters=settings["iterations"]["logit"], soft=0.9, e_soft=1.0, ramp_recycles=False, save_best=True)
    
    # Stage 3: Soft iterations
    print('Starting soft iterations')
    af_model.design_soft(iters=settings["iterations"]["soft"], e_temp=1e-2, ramp_recycles=False, save_best=True)
    
    # Stage 4: Hard iterations
    af_model.clear_best()
    print('Starting hard iterations:')
    af_model.design_hard(iters=settings["iterations"]["hard"], temp=1e-2, dropout=False, ramp_recycles=False, save_best=True)
    
    # Stage 5: Semigreedy
    print('Starting semigreedy')
    af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=settings["iterations"]["semigreedy"], tries=10, dropout=False, num_models=1, ramp_models=False, save_best=True)
    
    return af_model

def main():
    parser = argparse.ArgumentParser(description='colabdesign')
    parser.add_argument('--settings', type=str, required=True,
                        help='Settings json location')
    args = parser.parse_args()
    settings = load_settings(args.settings)

    os.makedirs(settings["output"]["colabdesign_output_directory"], exist_ok=True)
    successful_designs = 0

    while successful_designs < settings["num_designs"]:
        seed = np.random.randint(0, 999999)
        af_model = initialize_model(settings)
        result = run_optimization(af_model, settings, seed)

        if result is not None:
            # Save final PDB
            loss = af_model._tmp['best']['aux']["log"]["loss"]
            plddt = np.mean(af_model._tmp['best']['aux']["log"]["plddt"])
            iptm = af_model._tmp['best']['aux']["log"]["i_ptm"]
            ptmenergy= af_model._tmp['best']['aux']["log"]["ptmenergy"]
            
            final_filename = f"{seed}_{loss:.2f}_{plddt:.2f}_{iptm:.2f}_{ptmenergy:.2f}.pdb"
            final_path = os.path.join(settings["output"]["colabdesign_output_directory"], final_filename)
            af_model.save_pdb(final_path, get_best=True)
            
            # Write to FASTA
            seq = af_model.get_seqs(get_best=True)[0]
            header = f">{seed}|design_{successful_designs+1}|"\
                    f"{loss:.2f}|"\
                    f"{plddt:.2f}|"\
                    f"{iptm:.2f}|"\
                    f"{ptmenergy:.2f}"
            
            with open(settings["output"]["fasta_file"], "a") as fasta:
                fasta.write(f"{header}\n{seq}\n")
            successful_designs += 1

if __name__ == "__main__":
    main()
