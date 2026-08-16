"""Figures for orientation selectivity prototype (session 715093703)."""
import numpy as np
import matplotlib.pyplot as plt
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

ASSET_ID = "58703c97-c0a9-4736-b684-73c85c1a444a"
LINDI_URL = f"https://lindi.neurosift.org/dandi/dandisets/000021/assets/{ASSET_ID}/nwb.lindi.json"
DIRECTIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
DIR_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]

R = np.load("prototype_results.npz", allow_pickle=True)
tuning, gosi, gdsi = R["tuning"], R["gosi"], R["gdsi"]
pref_ori, pref_dir, sig = R["pref_ori"], R["pref_dir"], R["sig"]
unit_loc, kept_indices = R["unit_loc"], R["kept_indices"]
blank_rate = R["blank_rate"]
n_units = tuning.shape[0]
print(f"loaded results: {n_units} units, {sig.sum()} significant")

# reload session for spike times of example units + sweep table
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
nwbfile = NWBHDF5IO(file=f).read()
nwb = nap.NWBFile(nwbfile)
tab = nwbfile.intervals["drifting_gratings_presentations"]
starts = tab["start_time"].data[:]
stops = tab["stop_time"].data[:]
direction = tab["orientation"].data[:]
temp_freq = tab["temporal_frequency"].data[:]
unit_ids = nwbfile.units["id"].data[:] if hasattr(nwbfile.units["id"], "data") else nwbfile.units.id[:]
ts_group = nwb["units"]
kept_unit_ids = unit_ids[kept_indices]

# pick example unit: highest-gOSI significant VISp unit
visp_mask = (unit_loc == "VISp") & sig
ex_local = np.where(visp_mask)[0][np.argmax(gosi[visp_mask])]
ex_uid = kept_unit_ids[ex_local]
ex_spikes = ts_group[ex_uid].t
print(f"example unit id {ex_uid}, gOSI={gosi[ex_local]:.3f}, pref_ori={pref_ori[ex_local]:.0f}")

# ---------------- Figure 1: raster + PSTH ----------------
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

# raster: all non-blank sweeps, grouped by direction
ax = fig.add_subplot(gs[0, :])
window = (-0.5, 2.5)
row = 0
yticks, ylabels = [], []
for di, d in enumerate(DIRECTIONS):
    m = np.where((direction == d))[0]
    for s in m:
        t0 = starts[s]
        spk = ex_spikes[(ex_spikes >= t0 + window[0]) & (ex_spikes <= t0 + window[1])] - t0
        ax.plot(spk, np.full(len(spk), row), "|", color=DIR_COLORS[di], ms=3, mew=0.6)
        row += 1
    yticks.append(row - len(m) / 2)
    ylabels.append(f"{int(d)}°")
    ax.axhline(row - 0.5, color="k", lw=0.3, alpha=0.3)
ax.axvspan(0, 2, color="gray", alpha=0.15)
ax.set_yticks(yticks, ylabels)
ax.set_xlim(window)
ax.set_xlabel("Time from grating onset (s)")
ax.set_ylabel("Drift direction")
ax.set_title(f"Example VISp unit {ex_uid}: spike raster across drifting-grating sweeps (gray = 2 s stimulus)")

# PSTH per direction
ax = fig.add_subplot(gs[1, 0])
bin_size = 0.05
edges = np.arange(window[0], window[1] + bin_size, bin_size)
centers = edges[:-1] + bin_size / 2
for di, d in enumerate(DIRECTIONS):
    m = np.where(direction == d)[0]
    all_spk = []
    for s in m:
        t0 = starts[s]
        all_spk.append(ex_spikes[(ex_spikes >= t0 + window[0]) & (ex_spikes <= t0 + window[1])] - t0)
    all_spk = np.concatenate(all_spk)
    h, _ = np.histogram(all_spk, bins=edges)
    ax.plot(centers, h / (len(m) * bin_size), color=DIR_COLORS[di], lw=1.2, label=f"{int(d)}°")
ax.axvspan(0, 2, color="gray", alpha=0.15)
ax.set_xlabel("Time from grating onset (s)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("PSTH per drift direction")
ax.legend(fontsize=7, ncol=2, frameon=False)

# direction tuning curve of example unit
ax = fig.add_subplot(gs[1, 1])
tc = tuning[ex_local]
ax.plot(DIRECTIONS, tc, "o-", color="k")
ax.axhline(blank_rate[ex_local], color="gray", ls="--", lw=1, label="blank (spont.)")
ax.set_xlabel("Drift direction (°)")
ax.set_ylabel("Mean rate (Hz)")
ax.set_xticks(DIRECTIONS)
ax.set_title(f"Direction tuning (gOSI={gosi[ex_local]:.2f}, gDSI={gdsi[ex_local]:.2f})")
ax.legend(fontsize=8, frameon=False)
fig.savefig("fig1_example_raster_psth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig1_example_raster_psth.png")

# ---------------- Figure 2: example tuning curves (6 units, cartesian + polar) ----------------
sig_idx = np.where(sig)[0]
top = sig_idx[np.argsort(gosi[sig_idx])[::-1]]
# spread examples across areas
examples, seen_areas = [], set()
for i in top:
    a = unit_loc[i]
    if a not in seen_areas:
        examples.append(i)
        seen_areas.add(a)
    if len(examples) == 5:
        break
examples = examples[:5]

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 5, wspace=0.55, hspace=0.5)
for k, i in enumerate(examples):
    tc = tuning[i]
    ax = fig.add_subplot(gs[0, k])
    ax.plot(DIRECTIONS, tc, "o-", color="C0", ms=4)
    ax.axhline(blank_rate[i], color="gray", ls="--", lw=0.8)
    ax.set_xticks([0, 180])
    ax.set_xlabel("Direction (°)", fontsize=8)
    if k == 0:
        ax.set_ylabel("Rate (Hz)")
    ax.set_title(f"{unit_loc[i]} unit\ngOSI={gosi[i]:.2f}", fontsize=9)
    axp = fig.add_subplot(gs[1, k], projection="polar")
    th = np.deg2rad(np.append(DIRECTIONS, 0))
    axp.plot(th, np.append(tc, tc[0]), color="C0")
    axp.fill(th, np.append(tc, tc[0]), color="C0", alpha=0.25)
    axp.set_theta_zero_location("N")
    axp.set_rticks([])
    axp.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"], fontsize=7)
    axp.tick_params(pad=2)
fig.suptitle("Example orientation/direction tuning curves (top: cartesian; bottom: polar)", y=1.0)
fig.savefig("fig2_example_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig2_example_tuning_curves.png")

# ---------------- Figure 3: population stats ----------------
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
ax = axes[0]
bins = np.linspace(0, 1, 41)
ax.hist(gosi[~sig], bins=bins, color="gray", alpha=0.8, label=f"not sig. ({(~sig).sum()})")
ax.hist(gosi[sig], bins=bins, color="C0", alpha=0.8, label=f"sig. p<0.01 ({sig.sum()})")
ax.set_xlabel("gOSI")
ax.set_ylabel("Units")
ax.set_title("Orientation selectivity (gOSI)")
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
ax.hist(pref_ori[sig], bins=np.linspace(0, 180, 19), color="C0")
ax.set_xlabel("Preferred orientation (°)")
ax.set_ylabel("Units")
ax.set_title(f"Preferred orientation (n={sig.sum()} sig.)")

ax = axes[2]
ax.scatter(gosi[~sig], gdsi[~sig], s=6, color="gray", alpha=0.5, label="not sig.")
ax.scatter(gosi[sig], gdsi[sig], s=6, color="C0", alpha=0.6, label="sig. OS")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("gOSI")
ax.set_ylabel("gDSI")
ax.set_title("Orientation vs direction selectivity")
ax.legend(fontsize=8, frameon=False)

ax = axes[3]
areas = ["VISp", "VISl", "VISpm", "VISam", "VISrl"]
fracs = [sig[unit_loc == a].mean() for a in areas]
counts_a = [(unit_loc == a).sum() for a in areas]
ax.bar(areas, fracs, color="C0")
for x, (fr, c) in enumerate(zip(fracs, counts_a)):
    ax.text(x, fr + 0.02, f"n={c}", ha="center", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction significant")
ax.set_title("Orientation selectivity by area")
fig.tight_layout()
fig.savefig("fig3_population_stats.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig3_population_stats.png")

# ---------------- Figure 4: population tuning heatmap ----------------
sig_tuning = tuning[sig]
norm = sig_tuning / np.maximum(sig_tuning.max(axis=1, keepdims=True), 1e-9)
order = np.argsort(pref_ori[sig])
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(norm[order], aspect="auto", cmap="viridis",
               extent=[0, 360, sig.sum(), 0])
ax.set_xticks(DIRECTIONS)
ax.set_xlabel("Drift direction (°)")
ax.set_ylabel("Units (sorted by preferred orientation)")
ax.set_title(f"Normalized direction tuning, {sig.sum()} significantly orientation-selective units")
fig.colorbar(im, ax=ax, label="Normalized rate", shrink=0.8)
fig.savefig("fig4_population_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig4_population_heatmap.png")
