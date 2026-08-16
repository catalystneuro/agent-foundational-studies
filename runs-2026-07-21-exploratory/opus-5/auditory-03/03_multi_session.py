"""Step 3: run the tuning analysis on every session and decode tone identity.

Writes results/units_all.csv, results/session_summary.csv, results/decoding.npz
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import FREQS, RESDIR, SESSIONS, load_session
from decoding import accuracy_vs_units, decode_frequency, population_features
from tuning import analyze_session

os.makedirs(RESDIR, exist_ok=True)

SUBSET_SIZES = [1, 2, 5, 10, 20, 50, 100, 200]

all_units, summary, confusions = [], [], {}
acc_curve = None

for name, asset in tqdm(SESSIONS, desc="sessions"):
    sess = load_session(name, asset, with_behavior=False)
    units, tmean, _, _ = analyze_session(name, asset, sess=sess)
    for i, f in enumerate(FREQS):
        units[f"tune_{int(f)}"] = tmean[i]
    all_units.append(units)

    X = population_features(sess["spikes"], sess["trials"])
    y = sess["trials"].stim_frequency.values
    acc, conf = decode_frequency(X, y)
    acc_shuf, _ = decode_frequency(X, y, shuffle_label=True)
    confusions[name] = conf
    if name == SESSIONS[0][0]:
        acc_curve = accuracy_vs_units(X, y, SUBSET_SIZES)

    tuned = units[units.freq_tuned]
    summary.append(dict(
        session=name, subject=units.subject.iloc[0], n_units=len(units),
        n_trials=len(y),
        n_responsive=int(units.responsive.sum()),
        n_enhanced=int(units.enhanced.sum()),
        n_suppressed=int(units.suppressed.sum()),
        n_tuned=int(units.freq_tuned.sum()),
        frac_tuned=float(units.freq_tuned.mean()),
        median_si=float(tuned.selectivity_index.median()),
        median_sparseness=float(tuned.sparseness.median()),
        median_latency_ms=float(1e3 * np.nanmedian(tuned.latency_s)),
        bf_match=float(tuned.bf_split_match.mean()),
        decode_acc=acc, decode_acc_shuffled=acc_shuf,
    ))
    tqdm.write(f"{name}: {len(units)} units, {units.freq_tuned.sum()} tuned, "
               f"decoding {100 * acc:.1f}% (shuffled {100 * acc_shuf:.1f}%)")

units_all = pd.concat(all_units, ignore_index=True)
units_all.to_csv(f"{RESDIR}/units_all.csv", index=False)
summary = pd.DataFrame(summary)
summary.to_csv(f"{RESDIR}/session_summary.csv", index=False)
np.savez(f"{RESDIR}/decoding.npz",
         names=np.array(list(confusions)),
         confusions=np.stack([confusions[k] for k in confusions]),
         acc_curve=acc_curve, subset_sizes=np.array(SUBSET_SIZES))

print(f"\n{len(units_all)} units from {len(summary)} sessions, "
      f"{summary.subject.nunique()} mice")
print(f"tone-responsive: {units_all.responsive.sum()} "
      f"({100 * units_all.responsive.mean():.0f}%)")
print(f"frequency-tuned: {units_all.freq_tuned.sum()} "
      f"({100 * units_all.freq_tuned.mean():.0f}%)")
print(summary[["session", "n_units", "n_tuned", "frac_tuned", "decode_acc",
               "decode_acc_shuffled"]].to_string(index=False))
