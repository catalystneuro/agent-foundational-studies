"""Single-session prototype: tone-evoked responses and frequency tuning."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy import stats

import dandi_auditory as da

assets = da.list_assets()
asset_id = assets.query("subject == 'LA11' and session == '1'")["asset_id"].iloc[0]
s = da.load_session(asset_id)
units, freqs, freq = s["units"], s["freqs"], s["frequency"]
onsets = s["trials"].start
print(f"sub-{s['subject']} ses-{s['session_id']}: {len(units)} units, {len(onsets)} tones")

# ------------------------------------------------------- per-trial spike counts
base = da.trial_spike_counts(units, onsets, da.BASELINE_WINDOW)
evok = da.trial_spike_counts(units, onsets, da.EVOKED_WINDOW)
base_rate = base / (da.BASELINE_WINDOW[1] - da.BASELINE_WINDOW[0])
evok_rate = evok / (da.EVOKED_WINDOW[1] - da.EVOKED_WINDOW[0])
delta = evok_rate - base_rate
print("mean baseline %.2f Hz, mean evoked %.2f Hz" % (base_rate.mean(), evok_rate.mean()))

# ------------------------------------------------------------------- PSTHs
psth = da.psth_by_frequency(units, onsets, freq, freqs)  # (n_freq, n_bins, n_units)
bins = da.psth_bin_centers()

# ------------------------------------------------------------ tuning statistics
tuning = np.stack([delta[freq == f].mean(axis=0) for f in freqs])         # (n_freq, n_units)
sem = np.stack([stats.sem(delta[freq == f], axis=0) for f in freqs])

responsive_p = np.array(
    [stats.wilcoxon(evok[:, i], base[:, i])[1] if evok[:, i].sum() + base[:, i].sum() > 0 else 1.0
     for i in range(delta.shape[1])]
)
tuned_p = np.array(
    [stats.kruskal(*[delta[freq == f, i] for f in freqs])[1] for i in range(delta.shape[1])]
)
responsive = responsive_p < 0.01
tuned = tuned_p < 0.01
print(f"sound-responsive: {responsive.sum()}/{len(responsive)}")
print(f"frequency-tuned:  {tuned.sum()}/{len(tuned)}")

bf_idx = np.argmax(tuning, axis=0)
print("best-frequency counts:", {f"{f/1000:g}kHz": int(np.sum(bf_idx[tuned] == i))
                                 for i, f in enumerate(freqs)})

# ------------------------------------------------------- example unit figure
score = np.where(responsive & tuned, tuning.max(axis=0) - tuning.min(axis=0), -np.inf)
# pick the strongest example for each best frequency, so all BFs are represented
examples = []
for i in range(len(freqs)):
    cand = np.where((bf_idx == i) & np.isfinite(score))[0]
    if len(cand):
        examples.append(cand[np.argmax(score[cand])])
print("example units:", examples)

unit_ids = np.array(list(units.keys()))
fig, axes = plt.subplots(3, len(examples), figsize=(3.1 * len(examples), 8.5),
                         gridspec_kw={"height_ratios": [1.5, 1, 1], "hspace": 0.45})
for col, ui in enumerate(examples):
    uid = unit_ids[ui]
    # raster: 40 trials per frequency
    ax = axes[0, col]
    y = 0
    for fi, f in enumerate(freqs):
        ons = onsets[freq == f][:40]
        pe = nap.compute_perievent(units[uid], nap.Ts(ons), window=da.PSTH_WINDOW)
        for k in pe.keys():
            t = pe[k].t
            ax.plot(t, np.full_like(t, y), "|", color=da.FREQ_COLORS[fi], ms=3, mew=0.7)
            y += 1
    ax.axvspan(0, s["tone_duration"], color="0.85", zorder=0)
    ax.set_xlim(*da.PSTH_WINDOW)
    ax.set_ylim(0, y)
    ax.set_title(f"unit {uid}  (BF {freqs[bf_idx[ui]]/1000:g} kHz)", fontsize=10)
    if col == 0:
        ax.set_ylabel("trial (grouped by freq.)")

    ax = axes[1, col]
    for fi, f in enumerate(freqs):
        ax.plot(bins, psth[fi, :, ui], color=da.FREQ_COLORS[fi], lw=1.2,
                label=f"{f/1000:g} kHz")
    ax.axvspan(0, s["tone_duration"], color="0.85", zorder=0)
    ax.set_xlim(*da.PSTH_WINDOW)
    ax.set_xlabel("time from tone onset (s)")
    if col == 0:
        ax.set_ylabel("firing rate (Hz)")
    if col == len(examples) - 1:
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax = axes[2, col]
    ax.errorbar(freqs / 1000, tuning[:, ui], yerr=sem[:, ui], marker="o", color="k",
                capsize=3, lw=1.5)
    ax.axhline(0, color="0.6", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(freqs / 1000)
    ax.set_xticklabels([f"{f/1000:g}" for f in freqs])
    ax.set_xlabel("frequency (kHz)")
    if col == 0:
        ax.set_ylabel("evoked rate\n(Hz, baseline-subtracted)")
fig.suptitle(f"Tone-evoked responses, sub-{s['subject']} ses-{s['session_id']}", y=0.94)
fig.savefig("fig03_example_units.png", dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez("session_prototype.npz", tuning=tuning, sem=sem, psth=psth, bins=bins,
         freqs=freqs, responsive=responsive, tuned=tuned, bf_idx=bf_idx)
print("wrote fig03_example_units.png")
