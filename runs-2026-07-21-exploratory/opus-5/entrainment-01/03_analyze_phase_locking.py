"""Run the phase-locking pipeline on one session and report the per-state summary.

The heavy lifting lives in `compute_session.compute`, which caches its result to
`cache_<session>.pkl`; `04_make_figures.py` reuses those caches.
"""

import sys

import numpy as np

import compute_session as cs
import hc11_io as io

session = io.SESSIONS[0] if len(sys.argv) < 2 else io.SESSIONS[int(sys.argv[1])]
res = cs.compute(session)
df = res["df"]

print("\nmedian MRL by state and cell type")
print(df.groupby(["state", "cell_type"])[["mrl", "n_spikes"]].median().round(3))

for state in ["run", "REM", "nonREM"]:
    sub = df[(df.state == state) & (df.n_spikes >= cs.MIN_SPIKES)]
    if not len(sub):
        continue
    print(f"\n{state}: {len(sub)} units with >={cs.MIN_SPIKES} spikes, "
          f"{(sub.shuffle_p < 0.05).sum()} significantly modulated "
          f"(circular-shift p<0.05)")
    top = sub.sort_values("mrl", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"   unit {int(r.unit):3d} {r.cell_type:<10s} MRL={r.mrl:.3f} "
              f"pref={np.degrees(r.pref_phase):5.0f}° n={int(r.n_spikes)}")
