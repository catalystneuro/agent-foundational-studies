"""Recompute single-trial decoding using every unit, not the tuned subset.

Selecting units by a tuning test computed on all trials and then decoding those
same trials biases the accuracy upward, because the selection has already seen
the held-out folds.  Decoding from the full, unselected population removes that
circularity.  This script overwrites `conf` / `acc` in results_all_sessions.npy
with the unselected version and keeps the selected one as `conf_tuned` /
`acc_tuned` for comparison.
"""

import numpy as np
from tqdm import tqdm

import audlib as A

S = list(np.load("results_all_sessions.npy", allow_pickle=True))
assets = dict(A.list_assets())

for s in tqdm(S, desc="sessions"):
    nwbfile, nap_nwb = A.load_session(assets[s["path"]])
    trials = A.trial_table(nwbfile)
    counts = A.window_counts(nap_nwb["units"], trials["onset"], A.EVOKED_WIN)
    flab, freqs = trials["frequency"], s["freqs"]

    s["conf_tuned"], s["acc_tuned"] = s["conf"], s["acc"]
    s["conf"], s["acc"] = A.decode_frequency(counts, flab, freqs)
    print(f"{s['path']}: all units acc {s['acc']:.3f} "
          f"(tuned-subset acc was {s['acc_tuned']:.3f})", flush=True)

np.save("results_all_sessions.npy", np.array(S, dtype=object), allow_pickle=True)
a = np.array([s["acc"] for s in S])
at = np.array([s["acc_tuned"] for s in S])
print("\nall units:    %.3f +- %.3f" % (a.mean(), a.std()))
print("tuned subset: %.3f +- %.3f" % (at.mean(), at.std()))
