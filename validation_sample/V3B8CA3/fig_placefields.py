"""Figures 3-6: place fields, population maps, spatial information, field properties."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pf_core as pf

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160})

SESSION = "Achilles_10252013"
res = pf.analyze_session(SESSION, n_shuffles=1000)
meta = res["meta"]
tables = {d: res["per_dir"][d]["table"] for d in pf.DIRECTIONS}
TRACK_LEN = res["meta"]["track_length_cm"]
N_BINS = res["meta"]["n_pos_bins"]
maps = {d: res["per_dir"][d]["maps"] for d in pf.DIRECTIONS}
nulls = {d: res["per_dir"][d]["null"] for d in pf.DIRECTIONS}
bin_centers = res["per_dir"]["rightward"]["engine"].bin_centers
pyr = res["pyr"]
pd.concat(tables.values()).to_csv("place_cell_table_Achilles_10252013.csv", index=False)

for d in pf.DIRECTIONS:
    t = tables[d]
    print(f"{d}: {t.is_place_cell.sum()}/{len(t)} place cells "
          f"({100 * t.is_place_cell.mean():.0f} %)")

# ---------------------------------------------------------------- Figure 3 ---
# Example cells: spike position on every traversal, plus the resulting rate map.
tr = tables["rightward"]
good = tr.is_place_cell.values & (tr.field_width_cm.values < 55) \
    & (tr.peak_rate.values > 3.0) & (tr.stability.values > 0.7)
cand = np.flatnonzero(good)
peaks = tr.peak_pos_cm.values[cand]
examples = list(dict.fromkeys(
    [cand[np.argmin(np.abs(peaks - t))] for t in np.linspace(8, 152, 6)]))
print("example units:", [int(tr.unit.values[i]) for i in examples])

fig = plt.figure(figsize=(12, 7.0))
outer = fig.add_gridspec(2, 1, hspace=0.42, left=0.07, right=0.98, top=0.86, bottom=0.08)
for block in range(2):
    inner = outer[block].subgridspec(2, 3, height_ratios=[2.3, 1.0], hspace=0.10, wspace=0.24)
    for c in range(3):
        n = block * 3 + c
        if n >= len(examples):
            continue
        idx = examples[n]
        ax_r = fig.add_subplot(inner[0, c])
        ax_t = fig.add_subplot(inner[1, c], sharex=ax_r)
        unit = int(tr.unit.values[idx])

        offset = 0
        for d in pf.DIRECTIONS:
            trials = res["trial_eps"][d]
            sp = pf.spike_positions_by_trial(pyr[unit], res["position"], trials)
            for j, p in enumerate(sp):
                ax_r.plot(p, np.full_like(p, offset + j), "|", ms=3.0,
                          color=pf.DIR_COLORS[d], mew=0.8)
            offset += len(trials) + 3
        ax_r.set_ylim(-1, offset - 2)
        ax_r.set_title(f"unit {unit}   ({tr.info_bits_per_spike.values[idx]:.2f} bits/spike)",
                       fontsize=8.5, pad=4)
        ax_r.tick_params(labelbottom=False)
        if c == 0:
            ax_r.set_ylabel("traversal")

        for d in pf.DIRECTIONS:
            ax_t.plot(bin_centers, maps[d][idx], color=pf.DIR_COLORS[d], lw=1.6,
                      label=d if n == 0 else None)
        ax_t.set_xlim(0, TRACK_LEN)
        ax_t.set_xlabel("position (cm)")
        if c == 0:
            ax_t.set_ylabel("rate (Hz)")
        if n == 0:
            ax_t.legend(frameon=False, fontsize=7.5, loc="upper right")

fig.suptitle("Individual CA1 place cells fire at a reproducible track location\n"
             "top of each pair: spike positions on every traversal (blue = rightward, "
             "red = leftward);   bottom: occupancy-normalised rate map",
             fontweight="bold", fontsize=10)
fig.savefig("fig03_example_place_cells.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig03_example_place_cells.png")

# ---------------------------------------------------------------- Figure 4 ---
# Population: peak-normalised rate maps sorted by field location.
fig = plt.figure(figsize=(11.5, 5.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.06, 1.05], wspace=0.42,
                      left=0.065, right=0.98, top=0.82, bottom=0.13)

pc_r = tables["rightward"].is_place_cell.values
sort_ref = np.argsort(tables["rightward"].peak_pos_cm.values[pc_r])

for i, d in enumerate(pf.DIRECTIONS):
    ax = fig.add_subplot(gs[0, i])
    m = maps[d][pc_r][sort_ref]
    m = m / np.maximum(m.max(axis=1, keepdims=True), 1e-9)
    im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, TRACK_LEN, 0, m.shape[0]], vmin=0, vmax=1)
    ax.set_xlabel("position (cm)")
    ax.set_title(f"{'AB'[i]}  {d} runs", loc="left", fontweight="bold")
    if i == 0:
        ax.set_ylabel("place cell (sorted by rightward peak)")

cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(im, cax=cax)
cb.set_label("normalised firing rate")

ax = fig.add_subplot(gs[0, 3])
for d in pf.DIRECTIONS:
    pk = tables[d].peak_pos_cm.values[tables[d].is_place_cell.values]
    ax.hist(pk, bins=np.linspace(0, TRACK_LEN, 17), histtype="step", lw=1.8,
            color=pf.DIR_COLORS[d], label=d)
ax.set_xlabel("place-field peak position (cm)")
ax.set_ylabel("number of place cells")
ax.legend(frameon=False, fontsize=8)
ax.set_title("C  Fields tile the track", loc="left", fontweight="bold")

fig.suptitle(f"Population place-field map, {SESSION}.  Sorting by rightward peak (A) produces a "
             "continuous diagonal;\nthe same sorting applied to leftward runs (B) is scrambled, "
             "so the fields are direction-specific.", fontweight="bold", fontsize=9.5)
fig.savefig("fig04_population_place_fields.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig04_population_place_fields.png")

# ---------------------------------------------------------------- Figure 5 ---
# Spatial information against the circular-shift null.
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
fig.subplots_adjust(wspace=0.34, top=0.80, bottom=0.19, left=0.07, right=0.98)

d = "rightward"
t = tables[d]
ax = axes[0]
ax.hist(nulls[d].ravel(), bins=60, density=True, color="0.75", label="circular-shift null")
ax.hist(t.info_bits_per_spike.values, bins=30, density=True, histtype="step", lw=1.8,
        color=pf.DIR_COLORS[d], label="observed")
ax.set_xlabel("spatial information (bits/spike)"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8)
ax.set_title("A  Observed vs null, rightward", loc="left", fontweight="bold")

ax = axes[1]
p99 = np.percentile(nulls[d], 99, axis=0)
sig = t.shuffle_p.values < pf.SHUFFLE_ALPHA
ax.scatter(p99[~sig], t.info_bits_per_spike.values[~sig], s=16, color="0.6",
           label=f"n.s. (n={(~sig).sum()})")
ax.scatter(p99[sig], t.info_bits_per_spike.values[sig], s=16, color=pf.DIR_COLORS[d],
           label=f"p < 0.01 (n={sig.sum()})")
lim = [0, max(p99.max(), t.info_bits_per_spike.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("99th percentile of that unit's null (bits/spike)")
ax.set_ylabel("observed (bits/spike)")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("B  Per-unit significance", loc="left", fontweight="bold")

ax = axes[2]
for dd in pf.DIRECTIONS:
    ax.hist(tables[dd].stability.values, bins=np.linspace(-1, 1, 25), histtype="step",
            lw=1.8, color=pf.DIR_COLORS[dd], label=dd)
ax.axvline(0.5, color="k", ls="--", lw=1.0)
ax.set_xlabel("odd/even traversal map correlation"); ax.set_ylabel("units")
ax.legend(frameon=False, fontsize=8)
ax.set_title("C  Within-session stability", loc="left", fontweight="bold")

n_pc = int((tables["rightward"].is_place_cell | tables["leftward"].is_place_cell).sum())
fig.suptitle(f"Place-cell criteria, {SESSION}: {n_pc}/{len(t)} pyramidal cells qualify in at "
             "least one direction\n(spatial information above the 99th shuffle percentile, "
             "peak > 1 Hz, odd/even map correlation > 0.5)",
             fontweight="bold", fontsize=10)
fig.savefig("fig05_spatial_information.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig05_spatial_information.png")

# ---------------------------------------------------------------- Figure 6 ---
# Field properties and directionality.
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
fig.subplots_adjust(wspace=0.38, top=0.79, bottom=0.20, left=0.055, right=0.99)

pc_mask = {d: tables[d].is_place_cell.values for d in pf.DIRECTIONS}

ax = axes[0]
w = np.concatenate([tables[d].field_width_cm.values[pc_mask[d]] for d in pf.DIRECTIONS])
ax.hist(w, bins=np.arange(0, 100, 6), color="#4c72b0")
ax.axvline(np.median(w), color="crimson", ls="--")
ax.set_xlabel("field width at 50 % of peak (cm)"); ax.set_ylabel("place fields")
ax.set_title(f"A  Width, median {np.median(w):.0f} cm", loc="left", fontweight="bold")

ax = axes[1]
pk = np.concatenate([tables[d].peak_rate.values[pc_mask[d]] for d in pf.DIRECTIONS])
ax.hist(pk, bins=np.logspace(0, 2, 24), color="#4c72b0")
ax.set_xscale("log")
ax.axvline(np.median(pk), color="crimson", ls="--")
ax.set_xlabel("in-field peak rate (Hz)"); ax.set_ylabel("place fields")
ax.set_title(f"B  Peak rate, median {np.median(pk):.1f} Hz", loc="left", fontweight="bold")

ax = axes[2]
both_pc = pc_mask["rightward"] & pc_mask["leftward"]
ax.scatter(tables["rightward"].peak_pos_cm.values[both_pc],
           tables["leftward"].peak_pos_cm.values[both_pc], s=16, color="0.3")
ax.plot([0, TRACK_LEN], [0, TRACK_LEN], "k--", lw=1)
rho = np.corrcoef(tables["rightward"].peak_pos_cm.values[both_pc],
                  tables["leftward"].peak_pos_cm.values[both_pc])[0, 1]
ax.set_xlabel("peak position, rightward (cm)")
ax.set_ylabel("peak position, leftward (cm)")
ax.set_title(f"C  Peak location, r = {rho:.2f}", loc="left", fontweight="bold")

ax = axes[3]
rr = tables["rightward"].peak_rate.values
ll = tables["leftward"].peak_rate.values
sel = pc_mask["rightward"] | pc_mask["leftward"]
di = (rr - ll) / (rr + ll)
ax.hist(di[sel], bins=np.linspace(-1, 1, 25), color="#4c72b0")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("directionality index\n(right - left) / (right + left)")
ax.set_ylabel("place cells")
frac = np.mean(np.abs(di[sel]) > 0.3)
ax.set_title(f"D  {100 * frac:.0f} % strongly directional", loc="left", fontweight="bold")

fig.suptitle(f"Place-field properties, {SESSION}", fontweight="bold", fontsize=10)
fig.savefig("fig06_field_properties.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig06_field_properties.png")
