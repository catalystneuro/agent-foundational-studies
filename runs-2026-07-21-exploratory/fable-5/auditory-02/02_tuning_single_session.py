"""Single-session frequency tuning: example units and the population picture."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aft_analysis as aa
import aft_io

PROTOTYPE = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
RASTER_TRIALS = 120  # trials drawn per frequency for the raster panels

manifest = aft_io.list_assets()
nwbfile, nwb, h5 = aft_io.open_session(manifest[PROTOTYPE])
units = aft_io.load_units(nwbfile)
trials = aft_io.load_trials(nwbfile)
res = aa.analyse_session(units, trials, n_perm=1000)

freqs = res["freqs"]
n_freq = len(freqs)
cmap = plt.get_cmap("viridis")
fcol = [cmap(i / (n_freq - 1)) for i in range(n_freq)]

print(f"{len(units)} units | tone-driven {res['driven'].sum()} "
      f"| frequency-tuned {res['freq_tuned'].sum()}")

# One example unit per best frequency: the most strongly tuned unit preferring
# each of the five tones.
examples = []
for k in range(n_freq):
    cand = np.nonzero(res["freq_tuned"] & (res["bf_idx"] == k))[0]
    if len(cand):
        examples.append(cand[np.argmax(res["depth_z"][cand])])
print("example units:", res["unit_ids"][examples],
      "BF:", res["best_freq"][examples])

edges, psth = aa.psth_by_frequency(units, trials, bin_size=0.005,
                                   window=(-0.10, 0.20),
                                   unit_ids=res["unit_ids"][examples])
centers = (edges[:-1] + edges[1:]) / 2 * 1000

rng = np.random.default_rng(1)
onsets = trials.start_time.values
fidx = res["fidx"]

# ------------------------------------------------------------------ figure 3
n_ex = len(examples)
fig, axes = plt.subplots(3, n_ex, figsize=(3.1 * n_ex, 8.6),
                         gridspec_kw={"height_ratios": [1.5, 1, 1],
                                      "hspace": 0.42, "wspace": 0.38})
if n_ex == 1:
    axes = axes[:, None]

for c, j in enumerate(examples):
    uid = int(res["unit_ids"][j])
    spikes = units[uid].times()

    # --- raster, trials grouped by tone frequency
    ax = axes[0, c]
    row = 0
    total_rows = n_freq * RASTER_TRIALS
    for k in range(n_freq):
        ons = onsets[fidx == k]
        ons = rng.choice(ons, size=min(RASTER_TRIALS, len(ons)), replace=False)
        ons.sort()
        lo = np.searchsorted(spikes, ons - 0.10)
        hi = np.searchsorted(spikes, ons + 0.20)
        xs, ys = [], []
        for a, b, o in zip(lo, hi, ons):
            rel = spikes[a:b] - o
            xs.append(rel * 1000)
            ys.append(np.full(rel.size, row))
            row += 1
        if xs:
            ax.plot(np.concatenate(xs), np.concatenate(ys), "|", ms=1.6,
                    mew=0.5, color=fcol[k])
        # The y-axis is inverted, so convert the row index to an axes fraction.
        ax.text(1.01, 1 - (row - RASTER_TRIALS / 2) / total_rows,
                f"{freqs[k] / 1000:g}", transform=ax.transAxes,
                color=fcol[k], fontsize=8, va="center")
    ax.axvspan(0, 25, color="0.85", zorder=0)
    ax.set_ylim(row, 0)
    ax.set_xlim(-100, 200)
    ax.set_title(f"unit {uid}  (BF {res['best_freq'][j] / 1000:g} kHz)", pad=8)
    if c == 0:
        ax.set_ylabel("trial (grouped by frequency)")

    # --- PSTH per frequency
    ax = axes[1, c]
    for k in range(n_freq):
        ax.plot(centers, psth[c, k], color=fcol[k], lw=1.2,
                label=f"{freqs[k] / 1000:g} kHz")
    ax.axvspan(0, 25, color="0.85", zorder=0)
    ax.set_xlim(-100, 200)
    ax.set_xlabel("time from tone onset (ms)")
    if c == 0:
        ax.set_ylabel("firing rate (Hz)")
        ax.legend(fontsize=7, frameon=False, ncol=1, loc="upper left")

    # --- tuning curve
    ax = axes[2, c]
    tc = res["tuning_evoked"][j]
    ax.errorbar(freqs / 1000, tc, yerr=res["tuning_sem"][j], marker="o",
                color="k", ms=5, lw=1.5, capsize=3)
    for k in range(n_freq):
        ax.plot(freqs[k] / 1000, tc[k], "o", color=fcol[k], ms=6, zorder=5)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(freqs / 1000)
    ax.set_xticklabels([f"{f / 1000:g}" for f in freqs])
    ax.set_xlabel("tone frequency (kHz)")
    if c == 0:
        ax.set_ylabel("evoked rate (Hz)\n(5-55 ms, baseline subtracted)")
    ax.set_title(f"selectivity {res['selectivity'][j]:.2f}", fontsize=9, pad=4)

fig.suptitle(f"Frequency tuning of single auditory-cortex units "
             f"({aft_io.session_label(PROTOTYPE)}, dandiset 000986)", y=0.995)
fig.savefig("fig03_example_units.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig03_example_units.png")

np.save("_res_prototype.npy", res, allow_pickle=True)
