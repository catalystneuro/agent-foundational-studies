"""Run the tuning analysis over every session of dandiset 000986 and pool.

Results per session are cached to `session_results.npz` so that the figure and
decoding scripts do not have to re-stream the data.
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import aft_analysis as aa
import aft_io

CACHE = "session_results.pkl"

manifest = aft_io.list_assets()
paths = sorted(manifest, key=aft_io.session_label)

if os.path.exists(CACHE):
    with open(CACHE, "rb") as fh:
        sessions = pickle.load(fh)
    print(f"loaded cached results for {len(sessions)} sessions")
else:
    sessions = {}
    for path in tqdm(paths, desc="sessions"):
        label = aft_io.session_label(path)
        nwbfile, nwb, h5 = aft_io.open_session(manifest[path])
        units = aft_io.load_units(nwbfile)
        trials = aft_io.load_trials(nwbfile)
        res = aa.analyse_session(units, trials, n_perm=1000)
        res["label"] = label
        res["subject"] = nwbfile.subject.subject_id
        res["n_trials"] = len(trials)
        res["bf_interp"] = aa.gaussian_bf(res["tuning_evoked"], res["freqs"])
        sessions[label] = res
        h5.close()
    with open(CACHE, "wb") as fh:
        pickle.dump(sessions, fh)
    print(f"cached results for {len(sessions)} sessions -> {CACHE}")

freqs = sessions[next(iter(sessions))]["freqs"]
n_freq = len(freqs)
oct_axis = np.log2(freqs / 1000.0)

# ---------------------------------------------------------------- summary
rows = []
for label, r in sessions.items():
    rows.append(dict(
        session=label, subject=r["subject"], units=len(r["unit_ids"]),
        trials=r["n_trials"], driven=int(r["driven"].sum()),
        freq_tuned=int(r["freq_tuned"].sum()),
        pct_driven=100 * r["driven"].mean(),
        pct_tuned=100 * r["freq_tuned"].mean(),
        median_selectivity=float(np.nanmedian(r["selectivity"][r["freq_tuned"]])),
    ))
summary = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
summary.to_csv("session_summary.csv", index=False)
print(summary.to_string(index=False))

tot_units = summary.units.sum()
tot_driven = summary.driven.sum()
tot_tuned = summary.freq_tuned.sum()
print(f"\nTOTAL: {tot_units} units across {len(summary)} sessions "
      f"from {summary.subject.nunique()} mice")
print(f"  tone-driven      {tot_driven} ({100 * tot_driven / tot_units:.1f}%)")
print(f"  frequency-tuned  {tot_tuned} ({100 * tot_tuned / tot_units:.1f}%, "
      f"{100 * tot_tuned / tot_driven:.1f}% of driven)")

# --------------------------------------------------------- pooled arrays
pool = {k: np.concatenate([r[k] for r in sessions.values()])
        for k in ["driven", "freq_tuned", "best_freq", "bf_idx", "selectivity",
                  "centroid", "depth", "depth_z", "p_freq", "baseline_rate",
                  "bf_interp"]}
pool["tuning_evoked"] = np.concatenate([r["tuning_evoked"] for r in sessions.values()])
pool["session"] = np.concatenate([[r["label"]] * len(r["unit_ids"])
                                  for r in sessions.values()])
pool["subject"] = np.concatenate([[r["subject"]] * len(r["unit_ids"])
                                  for r in sessions.values()])
np.savez("pooled_units.npz", **pool, freqs=freqs)

tuned = pool["freq_tuned"]
tc = pool["tuning_evoked"][tuned]
# Peak-normalised curves for the population heat map.
norm = tc / np.maximum(tc.max(axis=1, keepdims=True), 1e-9)
# Order by the centre of mass of each curve so the population heat map shows a
# continuous progression rather than five discrete blocks.
order = np.argsort(pool["centroid"][tuned])

# ------------------------------------------------------------------ figure 4
fig = plt.figure(figsize=(13, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.36)

ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(norm[order], aspect="auto", cmap="magma", vmin=-0.3, vmax=1,
               extent=[-0.5, n_freq - 0.5, len(order), 0], interpolation="nearest")
ax.set_xticks(range(n_freq))
ax.set_xticklabels([f"{f / 1000:g}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("frequency-tuned unit (sorted by tuning centroid)")
ax.set_title(f"Tuning curves of all {tuned.sum()} frequency-tuned units")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("evoked rate / peak evoked rate")

# Average tuning curve aligned on each unit's best frequency: the classic
# demonstration that individual units are band-pass in log frequency.
ax = fig.add_subplot(gs[0, 1])
rel = np.full((tuned.sum(), 2 * n_freq - 1), np.nan)
bf = pool["bf_idx"][tuned]
for i in range(tuned.sum()):
    start = n_freq - 1 - bf[i]
    rel[i, start:start + n_freq] = norm[i]
mean_rel = np.nanmean(rel, axis=0)
sem_rel = np.nanstd(rel, axis=0) / np.sqrt(np.sum(~np.isnan(rel), axis=0))
off = np.arange(-(n_freq - 1), n_freq) * 1.0  # octaves (tones are octave spaced)
ax.fill_between(off, mean_rel - sem_rel, mean_rel + sem_rel, color="tab:blue",
                alpha=0.3, lw=0)
ax.plot(off, mean_rel, "o-", color="tab:blue")
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("octaves from best frequency")
ax.set_ylabel("normalised evoked rate")
ax.set_title("Population-average tuning curve\naligned on each unit's BF")
n_at = np.sum(~np.isnan(rel), axis=0)
for x, y, n in zip(off, mean_rel, n_at):
    if n < tuned.sum() * 0.25:
        ax.annotate(f"n={n}", (x, y), fontsize=6, ha="center",
                    xytext=(0, 7), textcoords="offset points", color="0.4")

ax = fig.add_subplot(gs[0, 2])
counts = np.array([(pool["best_freq"][tuned] == f).sum() for f in freqs])
cmap = plt.get_cmap("viridis")
ax.bar(range(n_freq), 100 * counts / counts.sum(),
       color=[cmap(i / (n_freq - 1)) for i in range(n_freq)])
ax.set_xticks(range(n_freq))
ax.set_xticklabels([f"{f / 1000:g}" for f in freqs])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("% of frequency-tuned units")
ax.set_title("Distribution of best frequencies")
for i, c in enumerate(counts):
    ax.text(i, 100 * c / counts.sum() + 0.6, str(c), ha="center", fontsize=8)
ax.set_ylim(0, 100 * counts.max() / counts.sum() * 1.18)

fig.suptitle("Auditory-cortex frequency tuning pooled over "
             f"{len(summary)} sessions / {summary.subject.nunique()} mice "
             "(DANDI 000986)", y=1.02)
fig.savefig("fig04_population_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ figure 5
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), gridspec_kw={"hspace": 0.42,
                                                              "wspace": 0.32})

ax = axes[0, 0]
x = np.arange(len(summary))
ax.bar(x - 0.2, summary.pct_driven, width=0.4, label="tone-driven", color="0.55")
ax.bar(x + 0.2, summary.pct_tuned, width=0.4, label="frequency-tuned",
       color="tab:red")
ax.set_xticks(x)
ax.set_xticklabels(summary.session, rotation=90, fontsize=7)
ax.set_ylabel("% of units")
ax.set_title("Responsive units per session")
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
ax.hist(pool["selectivity"][tuned], bins=np.linspace(0, 1, 26), color="tab:red")
ax.axvline(np.nanmedian(pool["selectivity"][tuned]), color="k", ls="--",
           label=f"median {np.nanmedian(pool['selectivity'][tuned]):.2f}")
ax.set_xlabel("frequency selectivity index")
ax.set_ylabel("units")
ax.set_title("Sharpness of tuning\n(0 = flat, 1 = one frequency only)")
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 2]
ax.hist(pool["depth_z"][pool["driven"]], bins=40, color="0.55",
        label="tone-driven units")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("tuning depth (z re frequency-shuffled null)")
ax.set_ylabel("units")
ax.set_title("Permutation test on tuning depth")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
only_driven = pool["driven"] & ~tuned
ax.scatter(pool["baseline_rate"][only_driven], pool["selectivity"][only_driven],
           s=12, alpha=0.7, color="0.45", label="driven, not freq. tuned")
ax.scatter(pool["baseline_rate"][tuned], pool["selectivity"][tuned], s=8,
           alpha=0.6, color="tab:red", label="frequency-tuned")
ax.legend(frameon=False, fontsize=7, loc="lower left")
ax.set_xscale("log")
ax.set_xlabel("spontaneous rate (Hz)")
ax.set_ylabel("selectivity index")
ax.set_title("Selectivity vs spontaneous rate")

ax = axes[1, 1]
sub_bf = pd.DataFrame({"subject": pool["subject"][tuned],
                       "bf": pool["best_freq"][tuned] / 1000})
subj = sorted(sub_bf.subject.unique())
mat = np.array([[100 * ((sub_bf.subject == s) & (sub_bf.bf == f / 1000)).sum()
                 / (sub_bf.subject == s).sum() for f in freqs] for s in subj])
im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0)
ax.set_xticks(range(n_freq))
ax.set_xticklabels([f"{f / 1000:g}" for f in freqs])
ax.set_yticks(range(len(subj)))
ax.set_yticklabels(subj, fontsize=8)
ax.set_xlabel("best frequency (kHz)")
ax.set_title("Best-frequency distribution per mouse")
fig.colorbar(im, ax=ax, pad=0.02).set_label("% of that mouse's tuned units")

ax = axes[1, 2]
# Split-half reliability: tuning curves computed on odd vs even trials.
rel_r = []
for r in sessions.values():
    resp = r["resp_counts"]
    base = r["base_counts"]
    ev = resp / (aa.RESPONSE_WIN[1] - aa.RESPONSE_WIN[0]) - \
        base / (aa.BASELINE_WIN[1] - aa.BASELINE_WIN[0])
    fi = r["fidx"]
    half = np.arange(len(fi)) % 2
    a = np.array([ev[(fi == k) & (half == 0)].mean(axis=0) for k in range(n_freq)]).T
    b = np.array([ev[(fi == k) & (half == 1)].mean(axis=0) for k in range(n_freq)]).T
    for j in np.nonzero(r["freq_tuned"])[0]:
        rel_r.append(np.corrcoef(a[j], b[j])[0, 1])
rel_r = np.array(rel_r)
ax.hist(rel_r, bins=np.linspace(-1, 1, 41), color="tab:green")
ax.axvline(np.nanmedian(rel_r), color="k", ls="--",
           label=f"median r = {np.nanmedian(rel_r):.2f}")
ax.set_xlabel("odd- vs even-trial tuning-curve correlation")
ax.set_ylabel("units")
ax.set_title("Split-half reliability of tuning")
ax.legend(frameon=False, fontsize=8)

fig.suptitle("Population statistics of frequency tuning (DANDI 000986)", y=0.98)
fig.savefig("fig05_population_statistics.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nmedian selectivity of tuned units: "
      f"{np.nanmedian(pool['selectivity'][tuned]):.3f}")
print(f"median split-half tuning reliability: {np.nanmedian(rel_r):.3f}")
print("wrote fig04_population_tuning.png, fig05_population_statistics.png, "
      "session_summary.csv, pooled_units.npz")
