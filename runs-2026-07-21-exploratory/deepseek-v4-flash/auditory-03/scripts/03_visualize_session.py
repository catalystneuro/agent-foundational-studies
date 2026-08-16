"""Single-session figures for the auditory tuning prototype.

Produces:
  fig_raw_activity.png   - raw spike raster, behavior traces, population PSTH
  fig_example_units.png  - rasters + PSTHs per frequency for 3 example units
  fig_tuning_curves.png  - tuning curves with SEM for 12 example units
  fig_population.png     - population heatmap of normalized tuning curves
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap

sys.path.insert(0, os.path.dirname(__file__))
from tuning_analysis import compute_session_tuning, RESP_WIN

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(os.path.dirname(__file__), "..")

ASSET_ID = "aacd1c8a-73f7-469e-bf08-0afd5c1052f9"  # sub-LA11_ses-1_behavior.nwb
URL = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{ASSET_ID}/nwb.lindi.json"

local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)

st_all = f["units/spike_times"][()]
st_idx = f["units/spike_times_index"][()]
spike_times_list = []
for k in range(len(st_idx)):
    a = 0 if k == 0 else st_idx[k - 1]
    spike_times_list.append(np.sort(st_all[a:st_idx[k]]))

trials = nwb["trials"]
onsets = np.asarray(trials["start"])
freqs_per_trial = np.asarray(trials["stim_frequency"])
order = np.argsort(onsets)
onsets = onsets[order]
freqs_per_trial = freqs_per_trial[order]
frqs = np.sort(np.unique(freqs_per_trial))
n_units = len(spike_times_list)

res = compute_session_tuning(
    spike_times_list, onsets, freqs_per_trial, frqs,
    subject_id="LA11", session_id="LA11_ses-1",
)
metrics = res["metrics"]
tuned = metrics.tuned.values
resp = metrics.responsive.values


def bin_around(spikes, onsets_sub, t_edges):
    """Count spikes per event into histogram windows (vectorized).

    spikes: sorted spikes of one unit; onsets_sub: sorted trial onset times;
    t_edges: (n_bins+1,) relative time edges.
    Returns (n_events, n_bins) counts.
    """
    n_ev, n_bin = len(onsets_sub), len(t_edges) - 1
    out = np.zeros((n_ev, n_bin), dtype=int)
    for j in range(n_bin):
        out[:, j] = (
            np.searchsorted(spikes, onsets_sub + t_edges[j + 1], side="left")
            - np.searchsorted(spikes, onsets_sub + t_edges[j], side="left")
        )
    return out


DT = 0.002
T_EDGES = np.arange(-0.05, 0.08 + DT, DT)
T_CENTER = (T_EDGES[:-1] + T_EDGES[1:]) / 2

# %% ---------------------------------------------------------------- fig 1: raw
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.12})
t0, t1 = 900.0, 905.0  # 5-s window in the middle of the session
# (a) raster
ax = axes[0]
for k, st in enumerate(spike_times_list):
    i0, i1 = np.searchsorted(st, [t0, t1])
    ax.plot(st[i0:i1], np.full(i1 - i0, k), "k,", rasterized=True, ms=1.5)
mask_t = (onsets >= t0) & (onsets < t1)
for o in onsets[mask_t]:
    ax.axvline(o, color="crimson", lw=0.8, alpha=0.6)
ax.set_ylim(-2, n_units + 1)
ax.set_ylabel("unit #")
ax.set_title(f"Ses: LA11_ses-1 | spike raster (n={n_units} units), tone onsets in red")
# (b) running speed & pupil
ax = axes[1]
rs_time = np.asarray(nwb["running_speed"].t)
rs_data = np.asarray(nwb["running_speed"].d)
ax.plot(rs_time, rs_data, lw=0.8, color="tab:blue")
ax.set_ylabel("speed\n(cm/s)")
ax = axes[2]
pp_time = np.asarray(nwb["pupil_diameter"].t)
pp_data = np.asarray(nwb["pupil_diameter"].d)
ax.plot(pp_time, pp_data, lw=0.8, color="tab:orange")
ax.set_ylabel("pupil\n(a.u.)")
ax.set_xlabel("time (s)")
fig.align_ylabels(axes)
fig.savefig(os.path.join(FIGDIR, "fig1_raw_activity.png"), dpi=150)
plt.close(fig)
print("saved fig1_raw_activity.png")

# =------------------------------------------------ fig 2: example units rasta+PST multi (=6 panels each)
bf_freqs = [2000.0, 8000.0, 32000.0]
exc_units = []
for bf in bf_freqs:
    cand = np.where(metrics.bf_freq.values == bf)[0]
    cand = cand[metrics.tuned.values[cand]]
    if len(cand) == 0:
        continue
    peak = res["rates_net"][cand, metrics.bf_idx.values[cand]]
    exc_units.append(cand[np.argmax(peak)])
print("example units:", exc_units)

fig, axes = plt.subplots(len(exc_units), 6, figsize=(16, 3.2 * len(exc_units)), squeeze=False)
for ri, u in enumerate(exc_units):
    for j, frq in enumerate(frqs):
        ax = axes[ri, j]
        on = onsets[freqs_per_trial == frq]
        on = np.sort(on)
        bins = bin_around(spike_times_list[u], on, T_EDGES)
        sdf = bins.mean(axis=0) / DT  # spikes per bin -> rate (Hz)
        ax.plot(T_CENTER, sdf, color="steelblue", lw=1.4)
        ax.axvline(0, color="crimson", lw=0.8)
        # raster: every 5th trial up to 100
        keep = on[::5][:100]
        for kk, o in enumerate(keep):
            m0 = np.searchsorted(spike_times_list[u], o + T_EDGES[0])
            m1 = np.searchsorted(spike_times_list[u], o + T_EDGES[-1])
            w = spike_times_list[u][m0:m1]
            w = w[(w >= o + T_EDGES[0]) & (w <= o + T_EDGES[-1])]
            ax.plot(w - o, np.full(len(w), kk), "k.", ms=1.1, rasterized=True)
        ax.set_xlim(-0.05, 0.08)
        if ri == 0:
            ax.set_title(f"{frq/1000:g} kHz")
        if j == 0:
            ax.set_ylabel(f"unit {u}\nrate (Hz)")
        if ri == len(exc_units) - 1:
            ax.set_xlabel("time rel. onset (s)")
        ax.tick_params(labelsize=8)
fig.suptitle("Tuning of three example units: PSTH (per frequency) + spike raster", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig2_example_units.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig2_example_units.png")

# ============================================= fig 3: tuning curves of 12 units
from scipy.stats import sem
sinds = np.where(metrics.tuned.values)[0]
# pick spread across BF, highest peak per BF bucket
order = np.argsort(metrics.bf_freq.values[sinds])
picked = []
seen_bf = set()
for i in order:
    bf = metrics.bf_freq.values[sinds[i]]
    if bf not in seen_bf:
        seen_bf.add(bf)
        picked.append(sinds[i])
    if len(picked) >= 12:
        break
# ensure variety in peak rate
picked = sorted(picked, key=lambda u: metrics.peak_rate.values[u], reverse=True)[:12]
picked = sorted(picked)

fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=False)
for a, u in enumerate(picked):
    ax = axes.flat[a]
    rc = res["rates_net"][u]  # net (baseline-subtracted)
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in rc]
    ax.bar(np.arange(len(frqs)), rc, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(frqs)))
    ax.set_xticklabels([f"{frq/1000:g}" for frq in frqs])
    ax.tick_params(labelsize=7)
    if a % 4 == 0:
        ax.set_ylabel("net rate (Hz)")
    ax.set_title(f"unit {u} | BF={metrics.bf_freq.values[u]/1000:.0f} kHz", fontsize=9)
fig.suptitle("Net tuning curves (baseline-subtracted evoked rate) for 12 tuned units", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig3_tuning_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig3_tuning_curves.png")

# ===================================================== fig 4: population heatmap
tuned_idx = np.where(metrics.tuned.values)[0]
if len(tuned_idx) == 0:
    raise RuntimeError("no tuned units")
dim = res["rates_net"][tuned_idx]
# min-max normalize each row
dmin = dim.min(axis=1, keepdims=True)
dmax = dim.max(axis=1, keepdims=True)
nrm = (dim - dmin) / (dmax - dmin + 1e-9)
# sort by BF then peak
sbf = metrics.bf_idx.values[tuned_idx]
keep = np.lexsort((nrm[:, 0], sbf))  # stable by bf, then peak
nrm_s = nrm[keep]
fig, axes = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [5, 1.3]})
ax = axes[0]
im = ax.imshow(nrm_s, aspect="auto", interpolation="nearest", cmap="viridis",
               vmin=0, vmax=1, origin="lower")
ax.set_xticks(range(len(frqs)))
ax.set_xticklabels([f"{frq/1000:g}" for frq in frqs])
ax.set_xlabel("frequency (kHz)")
ax.set_ylabel("unit (sorted by best frequency)")
ax.set_title(f"Normalized tuning curves, n={len(tuned_idx)} tuned units")
cbar = fig.colorbar(im, ax=ax, label="norm. net rate")
ax = axes[1]
ax.hist(sbf, bins=np.arange(len(frqs) + 1), orientation="horizontal", color="steelblue")
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("BF", fontsize=9) if False else None
ax.text(0.1, 0.9, "BF count", transform=ax.transAxes, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig4_population_heatmap.png"), dpi=150)
plt.close(fig)
print("saved fig4_population_heatmap.png")
print("all single-session figures done")