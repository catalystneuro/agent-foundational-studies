"""Run the pre-stimulus bias analysis on every session that passed the scan."""
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Apple's Accelerate BLAS raises spurious FP-flag warnings on plain float64 matmuls
# (reproducible with numpy 2.0.2 on random matrices); the results are finite.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

import session_analysis as sa

paths = json.load(open("selected_sessions.json"))
print(f"{len(paths)} sessions")

results = []
for i, p in enumerate(tqdm(paths, desc="sessions")):
    t0 = time.time()
    r = sa.analyze_session(p, seed=i, keep_arrays=(i == 0))
    r["runtime_s"] = time.time() - t0
    results.append(r)
    print(f"{r['subject']:18s} units={r['n_units']:4d} trials={r['n_trials']:4d} "
          f"AUC_block={r['auc_block']:.3f} (p={r['p_block']:.3f})  "
          f"AUC_cross->choice={r['auc_crossdecode_choice']:.3f} (p={r['p_crossdecode']:.3f})  "
          f"{r['runtime_s']:.0f}s", flush=True)

pickle.dump(results, open("all_results.pkl", "wb"))

summary_cols = [
    "subject", "n_units", "n_trials", "n_zero", "bias_zero_contrast", "bias_chi2_p",
    "auc_block", "p_block", "auc_block_move", "p_block_move",
    "auc_block_resid", "p_block_resid", "auc_block_hist", "p_block_hist",
    "auc_block_resid_full", "p_block_resid_full", "wheel_absvel_p",
    "auc_choice_zero", "p_choice_zero", "auc_choice_zero_move", "p_choice_zero_move",
    "auc_crossdecode_choice", "p_crossdecode", "auc_crossdecode_block",
    "auc_crossdecode_choice_resid", "p_crossdecode_resid",
    "block_coef", "block_p", "resid_coef", "resid_p", "frac_units_sig",
]
summary = pd.DataFrame([{c: r[c] for c in summary_cols} for r in results])
summary.to_csv("session_summary.csv", index=False)
print(summary.to_string(index=False))
