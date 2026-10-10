import json
import shutil

notebook_path = "modelling.ipynb"
backup_path = "modelling_backup.ipynb"
shutil.copy(notebook_path, backup_path)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

idea1_code = """\
# --- IDEA 1: Data-Driven Transition Matrices via Expectation-Maximization ---
import numpy as np
import pandas as pd

try:
    from hmmlearn import hmm
    from scipy.linalg import logm
    
    print("Training HMM with Baum-Welch (Idea 1)...")
    obs_map_local = {name: i for i, name in enumerate(OBS_ORDER)}
    model_dials_hmm = model_dials.copy()
    model_dials_hmm['obs_idx'] = model_dials_hmm['obs'].map(obs_map_local)

    sequences = []
    sequence_lengths = []
    for cp_id, g in model_dials_hmm.groupby("cp_id"):
        daily_obs = g.set_index('attempt_ts').resample('D')['obs_idx'].first().dropna()
        if len(daily_obs) > 1:
            sequences.append(daily_obs.values.reshape(-1, 1))
            sequence_lengths.append(len(daily_obs))
            
    X = np.concatenate(sequences)
    em_model = hmm.CategoricalHMM(n_components=len(STATES), n_iter=20, random_state=42, init_params="te")
    
    PSEUDO_local = pd.DataFrame([
        [40, 2, 1, 8, 20, 3, 4, 0.2, 3], [4, 2, 1, 25, 30, 4, 6, 0.2, 2],
        [2, 1, 1, 6, 20, 8, 25, 0.3, 1], [1, 1, 2, 3, 8, 40, 20, 0.5, 0.5],
        [2, 40, 4, 6, 15, 3, 3, 0.2, 5], [1, 4, 40, 4, 12, 5, 5, 2, 2],
        [0.2, 0.5, 3, 1, 4, 3, 3, 40, 0.3]
    ], index=STATES, columns=OBS_ORDER, dtype=float)
    
    B_initial = PSEUDO_local.div(PSEUDO_local.sum(axis=1), axis=0).to_numpy()
    em_model.emissionprob_ = B_initial
    em_model.fit(X, sequence_lengths)
    
    P_daily = em_model.transmat_
    B = em_model.emissionprob_
    Q = np.real(logm(P_daily))
    for i in range(len(STATES)):
        Q[i, i] = 0
        Q[i, i] = -np.sum(Q[i, :])
    
    obs_index = {name: i for i, name in enumerate(OBS_ORDER)}
    print("Successfully trained HMM with Baum-Welch!")
except ImportError:
    print("hmmlearn not installed! Falling back to heuristic matrices.")
"""

idea2_code = """\
# --- IDEA 2: Hierarchical Intent (Fixing Independence across contact points) ---
GAMMA = 0.30
account_intent = phones.groupby('account_id').agg(
    global_p_avoiding=('p_state_valid_avoiding', 'max'),
    global_p_reachable=('p_state_valid_reachable', 'max')
).reset_index()

phones = phones.merge(account_intent, on='account_id', how='left')

phones['p_state_valid_avoiding'] = (1 - GAMMA) * phones['p_state_valid_avoiding'] + GAMMA * phones['global_p_avoiding']
phones['p_state_valid_reachable'] = (1 - GAMMA) * phones['p_state_valid_reachable'] + GAMMA * phones['global_p_reachable']

state_cols = [f"p_state_{s}" for s in STATES]
"""

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", [])
    source_str = "".join(source)
    
    if "RATE_TO =" in source_str and "PSEUDO = pd.DataFrame" in source_str:
        new_source_str = source_str.replace("obs_index = {name: i for i, name in enumerate(OBS_ORDER)}", 
        "obs_index = {name: i for i, name in enumerate(OBS_ORDER)}\n" + idea1_code)
        cell["source"] = [line + "\n" for line in new_source_str.split("\n")][:-1]
        
    if "phones = phones.merge(posteriors, on=\"cp_id\", how=\"left\", validate=\"one_to_one\")" in source_str:
        new_source_str = source_str.replace(
            "state_cols = [f\"p_state_{s}\" for s in STATES]",
            idea2_code
        )
        cell["source"] = [line + "\n" for line in new_source_str.split("\n")][:-1]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook modified successfully.")
