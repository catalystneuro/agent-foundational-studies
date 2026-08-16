"""Run the cortical tuning analysis over all DANDI:000986 sessions and cache the results."""
import pickle
import sys

import numpy as np
from tqdm import tqdm

import cortex_analysis as cx
import dandi_io as dio

N_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
OUT = "results_cortex.pkl"

files = dio.list_assets("000986")[:N_SESSIONS]
print(f"analysing {len(files)} sessions of DANDI:000986")

all_units, per_session = [], []
for path, url in tqdm(files, desc="sessions"):
    sess = dio.load_tone_session_000986(url)
    units, freqs, spikes, _ = cx.analyze_session(sess, rng=np.random.default_rng(1))
    conf, acc, _ = cx.decode_frequency(spikes, sess, rng=np.random.default_rng(2))
    _, acc_shuf, _ = cx.decode_frequency(
        spikes, sess, rng=np.random.default_rng(3), shuffle=True
    )
    for u in units:
        u["path"] = path
    all_units += units
    per_session.append(dict(
        path=path, session=sess["session"], subject=sess["subject"],
        n_units=len(units), n_trials=len(sess["tone_onset"]),
        confusion=conf, accuracy=acc, accuracy_shuffled=acc_shuf, freqs=freqs,
    ))
    tuned = sum(u["p_tuned"] < 0.01 for u in units)
    tqdm.write(f"  {path}: {len(units)} units, {tuned} frequency-tuned, "
               f"decoding accuracy {acc:.3f} (shuffled {acc_shuf:.3f})")

with open(OUT, "wb") as fh:
    pickle.dump(dict(units=all_units, sessions=per_session, freqs=per_session[0]["freqs"]), fh)

n_tuned = sum(u["p_tuned"] < 0.01 for u in all_units)
print(f"\n{len(all_units)} units total, {n_tuned} ({100*n_tuned/len(all_units):.0f}%) "
      f"frequency-tuned at p < 0.01")
print("mean decoding accuracy: %.3f  shuffled control %.3f  chance %.3f" % (
    np.mean([s["accuracy"] for s in per_session]),
    np.mean([s["accuracy_shuffled"] for s in per_session]),
    1 / len(per_session[0]["freqs"])))
print("wrote", OUT)
