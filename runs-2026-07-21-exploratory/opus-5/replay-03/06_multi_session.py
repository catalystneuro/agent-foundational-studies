"""Stage 6: run the full pipeline on every straight-track session and aggregate."""

import os
import warnings

import numpy as np
import pandas as pd

import multisummary as M
import pipeline as P
import replaylib as R

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("figures", exist_ok=True)
os.makedirs("cache", exist_ok=True)

summaries = []
for i, session in enumerate(R.LINEAR_SESSIONS):
    print(f"\n===== {session} ({i + 1}/{len(R.LINEAR_SESSIONS)}) =====", flush=True)
    summary, frames = P.run_session(session, seed=i)
    for k, df in frames.items():
        df.to_csv(f"cache/events_{session}_{k}.csv", index=False)
    summaries.append(summary)
    print({k: (round(v, 4) if isinstance(v, float) else v)
           for k, v in summary.items()}, flush=True)

pd.DataFrame(summaries).to_csv("cache/multi_session_summary.csv", index=False)

S, pooled, stats = M.summarize()
M.figure(S, pooled, stats)
np.savez("cache/multi_session_stats.npz", n_sessions=len(S), **stats)
