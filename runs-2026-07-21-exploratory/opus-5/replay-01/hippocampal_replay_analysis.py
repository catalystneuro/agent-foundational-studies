# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decoding hippocampal replay during sharp-wave ripples
#
# Hippocampal place cells fire in a fixed sequence as an animal traverses a track.
# During sharp-wave ripples (SWRs, 150-250 Hz oscillations in CA1 that occur during
# immobility and slow-wave sleep) the same cells fire again in compressed sequences
# that sweep across the track in roughly 100 ms. This is replay. Demonstrating it
# requires three things: place fields good enough to support decoding, SWRs detected
# from the LFP, and a statistical test that the position decoded inside an SWR moves
# coherently in time rather than jumping around.
#
# ## Dataset
#
# [DANDI:000044](https://dandiarchive.org/dandiset/000044), Grosmark, Long and Buzsáki
# (2016), *Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences*. Bilateral silicon-probe recordings from dorsal CA1 in
# freely moving Long-Evans rats. Each session has the structure
#
# * **PRE**: ~4 h of rest/sleep in the home cage, in a familiar room,
# * **MAZE**: ~35 min running on a **novel** 1.6 m linear platform in a novel room,
# * **POST**: ~4 h of rest/sleep back in the home cage.
#
# The PRE epoch is the control that makes this dataset unusually good for a replay
# demonstration: during PRE the animal has never seen the maze, so any apparent
# replay of the maze during PRE is a false positive of the pipeline. The same
# analysis run on PRE and on POST gives a within-session estimate of the noise floor.
#
# This notebook analyses session `sub-Achilles_ses-Achilles-10252013`. The NWB file
# is 8.7 GB and is **streamed** with `remfile` plus a local disk cache; only the
# spike times, the tracking, and one LFP channel are ever transferred.
#
# ## What the analysis does
#
# 1. Load spikes, position, sleep scoring and LFP by streaming.
# 2. Build direction-specific place fields from the maze traversals.
# 3. Validate the decoder by cross-validated decoding of real running position.
# 4. Detect SWRs on the CA1 pyramidal-layer channel with the strongest ripple band.
# 5. Define candidate events as place-cell population bursts containing a ripple.
# 6. Decode a posterior over position in 20 ms bins inside each event.
# 7. Score each event by the posterior-weighted correlation between decoded
#    position and time, and test it against two within-event shuffles plus a
#    cell-identity shuffle of the whole pipeline.

# %% [markdown]
# ## Setup

# %%
import os

import h5py
import matplotlib
matplotlib.use("Agg")          # headless: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import scipy.signal as sig
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- data location ---------------------------------------------------------
# Blob URL for sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb, resolved
# from https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/
S3_URL = ("https://dandiarchive.s3.amazonaws.com/blobs/"
          "763/2d8/7632d81b-2819-473d-8946-34dc939e6028")
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000044")

# --- analysis parameters ---------------------------------------------------
LFP_RATE = 1250.0
TRACK_LEN = 1.6                 # m
NB_BINS = 50                    # 3.2 cm position bins
SMOOTH_BINS = 1.5               # Gaussian sigma for tuning curves, in bins
MIN_RUN_DISPLACEMENT = 0.8      # m, a traversal must cover half the track
MIN_PEAK_RATE = 1.0             # Hz
MIN_RUN_SPIKES = 50

RIPPLE_BAND = (150.0, 250.0)
PEAK_Z, EDGE_Z = 5.0, 2.0
MIN_DUR, MAX_DUR, MERGE_GAP = 0.020, 0.250, 0.030

BIN = 0.020                     # decoding bin inside events, s
MUA_SIGMA = 0.015
MUA_Z_PEAK = 3.0
MIN_EVENT_DUR, MAX_EVENT_DUR = 0.080, 0.500
MIN_ACTIVE_CELLS, MIN_BINS = 5, 4
RATE_FLOOR = 0.01               # Hz floor so log-likelihoods stay finite
N_SHUFFLE, N_IDENTITY_SHUFFLE, ALPHA = 1000, 15, 0.05
N_CONTROL_EVENTS = 400          # per epoch, for the cell-identity control

RNG = np.random.default_rng(0)
RNG_SCORE = np.random.default_rng(1)     # separate streams keep each stage
RNG_CTRL = np.random.default_rng(2)      # reproducible on its own
EPOCH_ORDER = ["PRE_wake", "PRE_NREM", "MAZE", "POST_wake", "POST_NREM"]
EPOCH_COLOR = {"PRE_wake": "#9CB4D8", "PRE_NREM": "#4C72B0", "MAZE": "#DD8452",
               "POST_wake": "#9FD1B0", "POST_NREM": "#55A868"}


def mask_in(t, ivs):
    """Boolean mask: which sorted timestamps fall inside an IntervalSet."""
    i = np.searchsorted(ivs.start, t, side="right") - 1
    ok = i >= 0
    m = np.zeros(np.size(t), bool)
    m[ok] = t[ok] < ivs.end[i[ok]]
    return m


# %% [markdown]
# ## 1. Streaming load and first look at every data stream
#
# `remfile` with a `DiskCache` turns the remote NWB file into a file-like object;
# `h5py` then reads only the chunks that are actually touched.

# %%
rem = remfile.File(S3_URL, disk_cache=remfile.DiskCache(CACHE_DIR))
nwb = h5py.File(rem, "r")

# --- experimental epochs ---------------------------------------------------
_ep = nwb["intervals/epochs"]
_labels = [s.decode() for s in _ep["label"][:]]
_key = {"PREEpoch": "PRE", "MazeEpoch": "MAZE", "POSTEpoch": "POST"}
epochs = {_key[l]: nap.IntervalSet(start=s, end=e)
          for l, s, e in zip(_labels, _ep["start_time"][:], _ep["stop_time"][:])}

# --- author sleep scoring --------------------------------------------------
_st = nwb["processing/behavior/states"]
_slab = np.array([s.decode() for s in _st["label"][:]])
states = {lab: nap.IntervalSet(start=_st["start_time"][:][_slab == lab].astype(float),
                               end=_st["stop_time"][:][_slab == lab].astype(float))
          for lab in np.unique(_slab)}
nrem, awake = states["Non-REM"], states["Awake"]

# --- spike trains ----------------------------------------------------------
_u = nwb["units"]
_idx = _u["spike_times_index"][:]
_times = _u["spike_times"][:]
_starts = np.concatenate([[0], _idx[:-1]])
units = nap.TsGroup(
    {i: nap.Ts(t=_times[a:b]) for i, (a, b) in enumerate(zip(_starts, _idx))},
    metadata={
        "cell_type": np.array([s.decode() for s in _u["cell_type"][:]]),
        "location": np.array([s.decode() for s in _u["location"][:]]),
        "shank_id": _u["shank_id"][:],
    },
)

# --- position --------------------------------------------------------------
_lg = nwb["processing/behavior/1.6mLinearMazeLinearizedPosition/"
          "1.6mLinearMazeLinearizedTimeSeries"]
_xg = nwb["processing/behavior/1.6mLinearMazePosition/1.6mLinearMazeSpatialSeries"]
_t0, _rate = float(_lg["starting_time"][()]), float(_lg["starting_time"].attrs["rate"])
_lin, _xy = _lg["data"][:, 0], _xg["data"][:]
_tt = _t0 + np.arange(_lin.size) / _rate
lin = nap.Tsd(t=_tt[~np.isnan(_lin)], d=_lin[~np.isnan(_lin)])
xy = nap.TsdFrame(t=_tt[~np.isnan(_xy).any(1)], d=_xy[~np.isnan(_xy).any(1)],
                  columns=["x", "y"])

print("epochs (s):", {k: (float(v.start[0]), float(v.end[-1])) for k, v in epochs.items()})
print(f"{len(units)} units: "
      f"{(units.cell_type == 'excitatory').sum()} excitatory, "
      f"{(units.cell_type == 'inhibitory').sum()} inhibitory")
print(f"sleep scoring: " +
      ", ".join(f"{k} {float(v.tot_length())/60:.0f} min" for k, v in states.items()))
print(f"tracking: {len(xy)} 2-D samples, "
      f"{len(lin)} samples with a curated linearised position")

# %% [markdown]
# The NWB file supplies a curated linearised position that is defined only while the
# animal is actually on the track, so it segments cleanly into individual traversals.
# The 2-D tracking is shown alongside it as a check that the linearisation is sensible.

# %%
pyr = units[units.cell_type == "excitatory"]

fig = plt.figure(figsize=(13, 11))
_gs1 = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.4], hspace=0.45, wspace=0.22)
axes = [fig.add_subplot(_gs1[0, :]), fig.add_subplot(_gs1[1, :]),
        fig.add_subplot(_gs1[2, 0]), fig.add_subplot(_gs1[2, 1])]

ax = axes[0]
ecol = {"PRE": "#4C72B0", "MAZE": "#DD8452", "POST": "#55A868"}
for name, ep in epochs.items():
    ax.axvspan(ep.start[0] / 60, ep.end[-1] / 60, color=ecol[name], alpha=0.35)
    ax.text((ep.start[0] + ep.end[-1]) / 120, 0.78, name, ha="center", fontsize=11)
for lab, c, y in [("Non-REM", "#8172B3", 0.35), ("REM", "#C44E52", 0.2),
                  ("Awake", "#937860", 0.05)]:
    for s, e in zip(states[lab].start, states[lab].end):
        ax.plot([s / 60, e / 60], [y, y], color=c, lw=4, solid_capstyle="butt")
    ax.text(-8, y, lab, ha="right", va="center", fontsize=9, color=c)
ax.set_ylim(-0.05, 1.0)
ax.set_yticks([])
ax.set_xlim(-25, epochs["POST"].end[-1] / 60)
ax.set_xlabel("time in session (min)")
ax.set_title("Session structure: PRE sleep, novel MAZE, POST sleep, with sleep scoring")

ax = axes[1]
for i, u in enumerate(pyr.keys()):
    t = pyr[u].index.values[::7]
    ax.plot(t / 60, np.full(t.size, i), "|", ms=1.5, color="k", alpha=0.25)
ax.set_ylabel("pyramidal unit #")
ax.set_xlabel("time in session (min)")
ax.set_xlim(0, epochs["POST"].end[-1] / 60)
ax.set_title("Spike raster, all excitatory units (every 7th spike drawn)")

ax = axes[2]
ax.plot(xy["x"].values, xy["y"].values, ".", ms=0.8, alpha=0.15, color="#4C72B0")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
ax.set_title("Raw 2-D tracking on the 1.6 m linear platform")

ax = axes[3]
seg = lin.get(epochs["MAZE"].start[0] + 100, epochs["MAZE"].start[0] + 400)
ax.plot(seg.index.values, seg.values, ".", ms=2.5, color="#4C72B0")
ax.set_xlabel("time (s)")
ax.set_ylabel("position (m)")
ax.set_title("Curated linearised position, 5 min of the maze epoch")

plt.savefig("fig01_session_overview.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote fig01_session_overview.png")

# %% [markdown]
# ## 2. Direction-specific place fields
#
# The linearised position splits into contiguous runs. On a linear track the animal
# shuttles back and forth and CA1 place fields are strongly directional, so fields
# are estimated separately for rightward and leftward traversals. The two field sets
# are then stacked into a single state space of `2 x NB_BINS` states, which lets a
# single decoder represent both the position and the direction of a replayed
# trajectory.

# %%
_t, _d = lin.index.values, lin.values
_segments = np.split(np.arange(_t.size), np.flatnonzero(np.diff(_t) > 0.5) + 1)
run_starts, run_ends, run_dir = [], [], []
for s in _segments:
    if s.size < 10 or abs(_d[s[-1]] - _d[s[0]]) < MIN_RUN_DISPLACEMENT:
        continue
    run_starts.append(_t[s[0]])
    run_ends.append(_t[s[-1]])
    run_dir.append(1 if _d[s[-1]] > _d[s[0]] else -1)
run_starts, run_ends = np.array(run_starts), np.array(run_ends)
run_dir = np.array(run_dir)
all_runs = nap.IntervalSet(start=run_starts, end=run_ends)

runs = {"rightward": nap.IntervalSet(start=run_starts[run_dir > 0],
                                     end=run_ends[run_dir > 0]),
        "leftward": nap.IntervalSet(start=run_starts[run_dir < 0],
                                    end=run_ends[run_dir < 0])}
run_speed = TRACK_LEN / np.mean(run_ends - run_starts)
print(f"{len(run_dir)} traversals ({(run_dir > 0).sum()} rightward, "
      f"{(run_dir < 0).sum()} leftward); mean traversal speed {run_speed:.2f} m/s")

_g = np.exp(-0.5 * (np.arange(-4, 5) / SMOOTH_BINS) ** 2)
_g /= _g.sum()


def tuning_curves(cells, run_idx):
    """Stacked [rightward; leftward] rate maps, shape (2*NB_BINS, n_cells)."""
    out = []
    for d in (1, -1):
        sub = [i for i in run_idx if run_dir[i] == d]
        ep = nap.IntervalSet(start=run_starts[sub], end=run_ends[sub])
        tc = nap.compute_1d_tuning_curves(cells, lin, nb_bins=NB_BINS, ep=ep,
                                          minmax=(0, TRACK_LEN)).fillna(0.0)
        out.append(np.apply_along_axis(lambda c: np.convolve(c, _g, "same"), 0,
                                       tc.values))
    return np.vstack(out)


_all_idx = np.arange(len(run_starts))
_tc_all = tuning_curves(pyr, _all_idx)
tc_right, tc_left = _tc_all[:NB_BINS], _tc_all[NB_BINS:]

_n_run_spikes = np.array([len(pyr[u].restrict(all_runs)) for u in pyr.keys()])
_peak = np.maximum(tc_right.max(0), tc_left.max(0))
is_place = (_peak >= MIN_PEAK_RATE) & (_n_run_spikes >= MIN_RUN_SPIKES)
place_ids = np.array(pyr.keys())[is_place]
place_cells = units[list(place_ids)]
tc_right, tc_left = tc_right[:, is_place], tc_left[:, is_place]
combined = np.vstack([tc_right, tc_left])
bin_centers = (np.arange(NB_BINS) + 0.5) * TRACK_LEN / NB_BINS
print(f"{len(place_ids)} of {len(pyr)} pyramidal cells pass the place-cell criteria")


def spatial_info(rate, occ):
    p = occ / occ.sum()
    mean_r = (p * rate).sum()
    nz = (rate > 0) & (p > 0)
    return float((p[nz] * rate[nz] / mean_r * np.log2(rate[nz] / mean_r)).sum())


_edges = np.linspace(0, TRACK_LEN, NB_BINS + 1)
_occ = {k: np.histogram(lin.restrict(v).values, bins=_edges)[0].astype(float)
        for k, v in runs.items()}
si = np.array([max(spatial_info(tc_right[:, j], _occ["rightward"]),
                   spatial_info(tc_left[:, j], _occ["leftward"]))
               for j in range(len(place_ids))])
print(f"spatial information: median {np.median(si):.2f} bits/spike "
      f"(range {si.min():.2f}-{si.max():.2f})")

# %%
_order = np.argsort(np.argmax(tc_right, axis=0))
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for j, (name, m) in enumerate([("rightward", tc_right), ("leftward", tc_left)]):
    mm = m[:, _order].T
    axes[0, j].imshow(mm / np.maximum(mm.max(1, keepdims=True), 1e-9), aspect="auto",
                      origin="lower", cmap="viridis", extent=[0, TRACK_LEN, 0, mm.shape[0]])
    axes[0, j].set_title(f"{name} runs (peak-normalised)")
    axes[0, j].set_xlabel("position on track (m)")
    axes[0, j].set_ylabel("place cell (sorted by rightward peak)")
axes[0, 2].hist(si, bins=25, color="#4C72B0", edgecolor="w")
axes[0, 2].set_xlabel("spatial information (bits/spike)")
axes[0, 2].set_ylabel("# cells")
axes[0, 2].set_title("Spatial information")

_ex = np.argsort(si)[::-1][:6]
for j, ax in enumerate(axes[1]):
    for c in _ex[j * 2:(j + 1) * 2]:
        ax.plot(bin_centers, tc_right[:, c], lw=2, label=f"unit {place_ids[c]} →")
        ax.plot(bin_centers, tc_left[:, c], lw=2, ls="--",
                label=f"unit {place_ids[c]} ←")
    ax.set_xlabel("position on track (m)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Example fields (solid →, dashed ←)", fontsize=10)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("fig02_place_fields.png", dpi=140)
plt.close(fig)
print("wrote fig02_place_fields.png")

# %% [markdown]
# ## 3. Does the decoder work? Cross-validated decoding of real running
#
# Before decoding anything inside a ripple, the decoder has to be shown to recover
# position when the answer is known. Fields are rebuilt from half of the traversals
# and used to decode the other half, so no traversal is decoded with a tuning curve
# it helped to build. Because the animal shuttles back and forth, traversal index
# parity is identical to running direction; folds are therefore assigned by the rank
# of each traversal *within* its direction.

# %%
def make_tc(rate_map):
    """Wrap a rate map as the xarray input expected by nap.decode_bayes."""
    return xr.DataArray(
        data=np.maximum(rate_map, RATE_FLOOR).T,
        coords={"unit": np.asarray(place_ids), "0": np.arange(rate_map.shape[0])},
        attrs={"occupancy": np.ones(rate_map.shape[0])},
    )


_rank = np.zeros(len(run_starts), int)
for d in (1, -1):
    _sel = np.flatnonzero(run_dir == d)
    _rank[_sel] = np.arange(_sel.size)

_true, _dec, _decdir, _truedir, _vt = [], [], [], [], []
for fold in (0, 1):
    train, test = _all_idx[_rank % 2 == fold], _all_idx[_rank % 2 != fold]
    ep = nap.IntervalSet(start=run_starts[test], end=run_ends[test])
    _, proba = nap.decode_bayes(make_tc(tuning_curves(place_cells, train)),
                                place_cells, ep, 0.25)
    P, t = np.asarray(proba.values), proba.index.values
    est = bin_centers[np.argmax(P[:, :NB_BINS] + P[:, NB_BINS:], axis=1)]
    truth = lin.interpolate(nap.Tsd(t=t, d=np.zeros(t.size))).values
    tdir = np.zeros(t.size)
    for i in test:
        tdir[(t >= run_starts[i]) & (t <= run_ends[i])] = run_dir[i]
    ok = ~np.isnan(truth)
    _true.append(truth[ok]); _dec.append(est[ok]); _vt.append(t[ok])
    _decdir.append(np.sign(P[:, :NB_BINS].sum(1) - P[:, NB_BINS:].sum(1))[ok])
    _truedir.append(tdir[ok])

true_pos, dec_pos = np.concatenate(_true), np.concatenate(_dec)
val_t = np.concatenate(_vt)
dir_acc = float(np.mean(np.concatenate(_decdir) == np.concatenate(_truedir)))
err = np.abs(dec_pos - true_pos)
chance_err = np.median(np.abs(RNG.permutation(dec_pos) - true_pos))
print(f"{true_pos.size} cross-validated 250 ms bins during running")
print(f"median decoding error {np.median(err)*100:.1f} cm "
      f"(chance {chance_err*100:.1f} cm); running direction correct in "
      f"{100*dir_acc:.1f}% of bins")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
_h = axes[0].hist2d(true_pos, dec_pos, bins=[np.linspace(0, TRACK_LEN, 33)] * 2,
                    cmap="magma")
axes[0].plot([0, TRACK_LEN], [0, TRACK_LEN], "w--", lw=1)
axes[0].set_xlabel("true position (m)")
axes[0].set_ylabel("decoded position (m)")
axes[0].set_title("Cross-validated decoding during running")
plt.colorbar(_h[3], ax=axes[0], label="# time bins")

axes[1].hist(err * 100, bins=40, color="#4C72B0", edgecolor="w")
axes[1].axvline(np.median(err) * 100, color="#C44E52", ls="--",
                label=f"median {np.median(err)*100:.1f} cm")
axes[1].set_xlabel("absolute decoding error (cm)")
axes[1].set_ylabel("# time bins")
axes[1].set_title("Decoding error")
axes[1].legend()

_s = val_t < val_t[0] + 120
_tp, _dp, _ts = true_pos[_s].copy(), dec_pos[_s], val_t[_s]
_tp[np.append(np.diff(_ts) > 1.0, False)] = np.nan     # break the line across gaps
axes[2].plot(_ts - _ts[0], _tp, "k.-", lw=1, ms=3, label="true")
axes[2].plot(_ts - _ts[0], _dp, ".", ms=6, color="#DD8452", label="decoded")
axes[2].set_xlabel("time (s)")
axes[2].set_ylabel("position (m)")
axes[2].set_title("Example stretch of running")
axes[2].legend(fontsize=8)
fig.suptitle("Decoder validation: these place fields recover real position during running",
             fontsize=13)
plt.tight_layout()
plt.savefig("fig03_decoder_validation.png", dpi=140)
plt.close(fig)
print("wrote fig03_decoder_validation.png")

# %% [markdown]
# ## 4. Sharp-wave ripple detection
#
# The recording has 128 LFP channels. Ripples are largest in the CA1 pyramidal layer,
# so the channel is chosen automatically: one storage chunk (136 s) of non-REM POST
# sleep is read for all channels and the channel with the highest ratio of extreme to
# typical ripple-band envelope is selected. Only that one channel is then streamed for
# the whole 9.7 h session.
#
# Detection follows the standard recipe: band-pass 150-250 Hz, Hilbert envelope,
# z-scored against non-REM statistics, events where the envelope crosses 2 SD and
# peaks above 5 SD, 20-250 ms long, merging events less than 30 ms apart.

# %%
_dset = nwb["processing/ecephys/LFP/LFP/data"]
_conv = float(_dset.attrs["conversion"])
_chunk = _dset.chunks[0]
_cands = [c for c in range(_dset.shape[0] // _chunk)
          if c * _chunk / LFP_RATE >= epochs["POST"].start[0]]
_frac = [mask_in(np.arange(c * _chunk, (c + 1) * _chunk, 25) / LFP_RATE, nrem).mean()
         for c in _cands]
_c0 = _cands[int(np.argmax(_frac))]
_i0, _i1 = _c0 * _chunk, min((_c0 + 1) * _chunk, _dset.shape[0])
block = _dset[_i0:_i1, :].astype(np.float32) * _conv
block_t = np.arange(_i0, _i1) / LFP_RATE
_m = mask_in(block_t, nrem)
print(f"channel-selection window {block_t[0]:.0f}-{block_t[-1]:.0f} s, "
      f"{100*_m.mean():.0f}% non-REM")

_b, _a = sig.butter(4, RIPPLE_BAND, btype="bandpass", fs=LFP_RATE)
prominence = np.zeros(block.shape[1])
for ch in tqdm(range(block.shape[1]), desc="ripple band per channel"):
    _e = np.abs(sig.hilbert(sig.filtfilt(_b, _a, block[:, ch])))[_m]
    prominence[ch] = np.percentile(_e, 99.9) / np.median(_e)
best_ch = int(np.argmax(prominence))
print(f"selected LFP channel {best_ch} (prominence {prominence[best_ch]:.1f})")

# %%
print("streaming the full session on that one channel ...")
_lt = np.arange(_dset.shape[0]) / LFP_RATE
lfp = nap.Tsd(t=_lt, d=_dset[:, best_ch].astype(np.float32) * _conv)
filt = nap.Tsd(t=_lt, d=sig.filtfilt(_b, _a, lfp.values))
_amp = np.abs(sig.hilbert(filt.values))
_w = int(round(0.008 * LFP_RATE)) * 6 + 1
_gk = sig.windows.gaussian(_w, std=0.008 * LFP_RATE)
env = nap.Tsd(t=_lt, d=np.convolve(_amp, _gk / _gk.sum(), mode="same"))

_base = env.restrict(nrem).values
z = nap.Tsd(t=_lt, d=(env.values - _base.mean()) / _base.std())

_cand = z.threshold(EDGE_Z, "above").time_support.merge_close_intervals(MERGE_GAP)
_si = np.searchsorted(_lt, _cand.start)
_ei = np.searchsorted(_lt, _cand.end)
_pz = np.array([z.values[s:e].max() if e > s else -np.inf for s, e in zip(_si, _ei)])
_pt = np.array([_lt[s + int(np.argmax(z.values[s:e]))] if e > s else np.nan
                for s, e in zip(_si, _ei)])
_dur = _cand.end - _cand.start
_keep = (_pz >= PEAK_Z) & (_dur >= MIN_DUR) & (_dur <= MAX_DUR)
ripples = nap.IntervalSet(start=_cand.start[_keep], end=_cand.end[_keep])
rip_peak_t, rip_peak_z = _pt[_keep], _pz[_keep]
print(f"{len(ripples)} ripples in the session, "
      f"median duration {np.median(_dur[_keep])*1000:.0f} ms")
for k in ["PRE", "POST"]:
    m = mask_in(rip_peak_t, epochs[k].intersect(nrem))
    print(f"  {k} non-REM: {m.sum()} ripples, "
          f"{m.sum()/float(epochs[k].intersect(nrem).tot_length()):.3f} Hz")

# %%
mua_all = nap.Ts(t=np.sort(np.concatenate([pyr[u].index.values for u in pyr.keys()])))
_is_post = mask_in(rip_peak_t, epochs["POST"].intersect(nrem))
post_peaks = nap.Ts(t=rip_peak_t[_is_post])

fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.plot(prominence, ".-", lw=0.8, ms=3, color="#4C72B0")
ax.axvline(best_ch, color="#C44E52", ls="--", label=f"selected ch {best_ch}")
ax.set_xlabel("LFP channel")
ax.set_ylabel("99.9th pct / median\nripple-band envelope")
ax.set_title("Ripple prominence across the probe")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 1:])
_ei4 = np.flatnonzero(_is_post)[:4]
for n, ei in enumerate(_ei4):
    c = rip_peak_t[ei]
    raw, flt = lfp.get(c - 0.15, c + 0.15), filt.get(c - 0.15, c + 0.15)
    tloc = (raw.index.values - c) * 1000
    ax.plot(tloc + n * 320, raw.values * 1e3 + 0.6, lw=0.7, color="k")
    ax.plot(tloc + n * 320, flt.values * 1e3 - 0.4, lw=0.7, color="#C44E52")
ax.set_xticks([n * 320 for n in range(len(_ei4))])
ax.set_xticklabels([f"{rip_peak_t[ei]:.1f} s" for ei in _ei4])
ax.set_ylabel("mV (raw above,\n150-250 Hz below)")
ax.set_title("Four example POST-sleep ripples (300 ms windows)")

_n_side = int(0.25 * LFP_RATE)
_idxs = np.searchsorted(_lt, post_peaks.index.values)
snips = np.array([lfp.values[i - _n_side:i + _n_side] for i in _idxs
                  if _n_side <= i < lfp.values.size - _n_side])
_tax = np.arange(-_n_side, _n_side) / LFP_RATE * 1000

ax = fig.add_subplot(gs[1, 0])
ax.plot(_tax, snips.mean(0) * 1e3, color="k")
ax.set_xlabel("time from ripple peak (ms)")
ax.set_ylabel("mV")
ax.set_title(f"Ripple-triggered average LFP (n={len(snips)})")

ax = fig.add_subplot(gs[1, 1])
_fr, _spec = sig.welch(snips, fs=int(LFP_RATE), nperseg=256, axis=1)
_off = np.sort(RNG.integers(_n_side, lfp.values.size - _n_side, size=1500))
_off = _off[~mask_in(_lt[_off], ripples)]
_bsnips = np.array([lfp.values[i - _n_side:i + _n_side] for i in _off])
_bspec = sig.welch(_bsnips, fs=int(LFP_RATE), nperseg=256, axis=1)[1]
ax.semilogy(_fr, _spec.mean(0), color="#C44E52", label="ripple windows")
ax.semilogy(_fr, _bspec.mean(0), color="#4C72B0", label="random windows")
ax.axvspan(*RIPPLE_BAND, color="gray", alpha=0.2)
ax.set_xlim(0, 400)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (V²/Hz)")
ax.set_title("Power spectrum in ripple windows")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
_pe = nap.compute_perievent(mua_all, post_peaks, window=(-0.4, 0.4))
_allt = np.concatenate([_pe[k].index.values for k in _pe.keys()])
ax.hist(_allt, bins=80, color="#55A868",
        weights=np.full(_allt.size, 1.0 / (len(_pe) * 0.01)))
ax.set_xlabel("time from ripple peak (s)")
ax.set_ylabel("pyramidal population rate (Hz)")
ax.set_title("Population firing is elevated during ripples")

ax = fig.add_subplot(gs[2, 0])
ax.hist((ripples.end - ripples.start)[_is_post] * 1000, bins=30, color="#4C72B0",
        edgecolor="w")
ax.set_xlabel("ripple duration (ms)")
ax.set_ylabel("count")
ax.set_title("Duration")

ax = fig.add_subplot(gs[2, 1])
ax.hist(rip_peak_z[_is_post], bins=40, color="#8172B3", edgecolor="w")
ax.set_xlabel("peak envelope (z)")
ax.set_ylabel("count")
ax.set_title("Peak ripple-band amplitude")

ax = fig.add_subplot(gs[2, 2])
_pf = []
for s, e in zip(ripples.start[_is_post][:400], ripples.end[_is_post][:400]):
    _seg = filt.get(s, e).values
    if _seg.size > 32:
        _ff, _pp = sig.welch(_seg, fs=int(LFP_RATE), nperseg=min(128, _seg.size))
        _pf.append(_ff[np.argmax(_pp)])
ax.hist(_pf, bins=25, color="#DD8452", edgecolor="w")
ax.set_xlabel("peak frequency (Hz)")
ax.set_ylabel("count")
ax.set_title("Intra-ripple peak frequency")

fig.suptitle(f"Sharp-wave ripple detection on LFP channel {best_ch}", fontsize=14)
plt.savefig("fig04_ripple_detection.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote fig04_ripple_detection.png")

# %% [markdown]
# ## 5. Candidate replay events
#
# Ripple envelope crossings alone give events with a median duration of ~55 ms, which
# is only two or three decoding bins. The standard fix is to define the event from the
# population burst that accompanies the ripple: the place-cell population rate is
# smoothed at 15 ms, z-scored against non-REM, and each event runs from where the rate
# rises above its mean to where it falls back, with a peak of at least 3 SD. Bursts
# that do not contain a detected ripple are discarded, which keeps the events anchored
# to genuine SWRs.

# %%
_mt = np.sort(np.concatenate([place_cells[u].index.values for u in place_cells.keys()]))
_grid = np.arange(_mt[0], _mt[-1], 0.005)
_cnt, _ = np.histogram(_mt, bins=np.append(_grid, _grid[-1] + 0.005))
_rate = nap.Tsd(t=_grid, d=_cnt / 0.005).smooth(MUA_SIGMA, size_factor=6)
_rb = _rate.restrict(nrem).values
zrate = nap.Tsd(t=_grid, d=(_rate.values - _rb.mean()) / _rb.std())

_bursts = zrate.threshold(0.0, "above").time_support.drop_short_intervals(MIN_EVENT_DUR)
_bs = np.searchsorted(_grid, _bursts.start)
_be = np.searchsorted(_grid, _bursts.end)
_bpeak = np.array([zrate.values[s:e].max() if e > s else -np.inf
                   for s, e in zip(_bs, _be)])
_has_rip = np.array([np.any((rip_peak_t >= s) & (rip_peak_t < e))
                     for s, e in zip(_bursts.start, _bursts.end)])
_ok = (_bpeak >= MUA_Z_PEAK) & _has_rip & \
      ((_bursts.end - _bursts.start) <= MAX_EVENT_DUR)
events = nap.IntervalSet(start=_bursts.start[_ok], end=_bursts.end[_ok])
print(f"{len(events)} candidate SWR events, median duration "
      f"{np.median(events.end - events.start)*1000:.0f} ms")

label_eps = {"PRE_wake": epochs["PRE"].intersect(awake),
             "PRE_NREM": epochs["PRE"].intersect(nrem),
             "MAZE": epochs["MAZE"],
             "POST_wake": epochs["POST"].intersect(awake),
             "POST_NREM": epochs["POST"].intersect(nrem)}
_mid = (events.start + events.end) / 2
ev_epoch = np.full(len(events), "OTHER", dtype=object)
for k, ep in label_eps.items():
    ev_epoch[mask_in(_mid, ep)] = k
print({k: int((ev_epoch == k).sum()) for k in EPOCH_ORDER})

# %% [markdown]
# ## 6. Decoding and replay scoring
#
# Inside each event a posterior over the 100 states (50 positions x 2 directions) is
# computed for every 20 ms bin under the usual Poisson/independence assumptions.
# The replay statistic is the posterior-weighted correlation between decoded position
# and time, computed separately within each direction block and reported for whichever
# block gives the larger absolute value.
#
# Significance uses two within-event shuffles, each with the *same* best-of-two-blocks
# maximum applied to the null so that the null matches the statistic:
#
# * **column cycle**: each time bin's posterior is circularly shifted by an independent
#   random amount, which destroys the sequence but preserves the per-bin posterior shape;
# * **time-bin permutation**: the order of the time bins is permuted.
#
# An event counts as replay only if it beats both nulls at p < 0.05.

# %%
def wcorr_batch(P, x, t):
    """Posterior-weighted position-time correlation for a stack (S, nbins, nstates)."""
    w = P / P.sum(axis=(1, 2), keepdims=True)
    mx = (w * x[None, None, :]).sum(axis=(1, 2))
    mt = (w * t[None, :, None]).sum(axis=(1, 2))
    dx = x[None, None, :] - mx[:, None, None]
    dt = t[None, :, None] - mt[:, None, None]
    cov = (w * dx * dt).sum(axis=(1, 2))
    vx = (w * dx ** 2).sum(axis=(1, 2))
    vt = (w * dt ** 2).sum(axis=(1, 2))
    out = np.zeros_like(cov)
    ok = (vx > 0) & (vt > 0)
    out[ok] = cov[ok] / np.sqrt(vx[ok] * vt[ok])
    return out, cov, vt


def score_event(trel, P, rng, n_shuffle=N_SHUFFLE):
    nb = P.shape[0]
    obs, slopes, mass = [], [], []
    null_cycle = np.zeros((2, n_shuffle))
    null_perm = np.zeros((2, n_shuffle))
    for d, Pb in enumerate([P[:, :NB_BINS], P[:, NB_BINS:]]):
        mass.append(Pb.sum() / P.sum())
        Pn = Pb / Pb.sum(axis=1, keepdims=True)
        wc, cov, vt = wcorr_batch(Pn[None], bin_centers, trel)
        obs.append(wc[0])
        slopes.append(cov[0] / vt[0] if vt[0] > 0 else 0.0)

        shifts = rng.integers(0, NB_BINS, size=(n_shuffle, nb))
        idx = (np.arange(NB_BINS)[None, None, :] - shifts[:, :, None]) % NB_BINS
        Psh = np.take_along_axis(np.broadcast_to(Pn, (n_shuffle, nb, NB_BINS)),
                                 idx, axis=2)
        null_cycle[d] = wcorr_batch(Psh, bin_centers, trel)[0]
        perm = np.argsort(rng.random((n_shuffle, nb)), axis=1)
        null_perm[d] = wcorr_batch(Pn[perm], bin_centers, trel)[0]

    d_best = int(np.argmax(np.abs(obs)))
    stat = abs(obs[d_best])
    p_cycle = (np.sum(np.abs(null_cycle).max(0) >= stat) + 1) / (n_shuffle + 1)
    p_perm = (np.sum(np.abs(null_perm).max(0) >= stat) + 1) / (n_shuffle + 1)
    return dict(direction=["rightward", "leftward"][d_best], dir_mass=mass[d_best],
                wcorr=obs[d_best], slope=slopes[d_best], p_cycle=p_cycle,
                p_perm=p_perm, p_max=max(p_cycle, p_perm))


def split_posterior(proba, evs):
    """One (t_rel, posterior) pair per interval of `evs`."""
    pt, pv = proba.index.values, np.asarray(proba.values)
    ei = np.searchsorted(evs.start, pt, side="right") - 1
    inside = (ei >= 0) & (pt < evs.end[np.maximum(ei, 0)])
    idx = np.flatnonzero(inside)[np.argsort(ei[inside], kind="stable")]
    bounds = np.searchsorted(ei[idx], np.arange(len(evs) + 1))
    out = [None] * len(evs)
    for i in range(len(evs)):
        sl = idx[bounds[i]:bounds[i + 1]]
        if sl.size:
            out[i] = (pt[sl] - pt[sl][0], pv[sl])
    return out


_, proba = nap.decode_bayes(make_tc(combined), place_cells, events, BIN)
per_event = split_posterior(proba, events)
n_active = np.array([sum(len(place_cells[u].get(s, e)) > 0 for u in place_cells.keys())
                     for s, e in zip(events.start, events.end)])
n_spikes = np.array([sum(len(place_cells[u].get(s, e)) for u in place_cells.keys())
                     for s, e in zip(events.start, events.end)])

rows, posteriors = [], {}
for i in tqdm(range(len(events)), desc="scoring events"):
    if per_event[i] is None:
        continue
    trel, P = per_event[i]
    if P.shape[0] < MIN_BINS or n_active[i] < MIN_ACTIVE_CELLS:
        continue
    rows.append(dict(event=i, epoch=ev_epoch[i], start=events.start[i],
                     end=events.end[i], dur=events.end[i] - events.start[i],
                     n_bins=P.shape[0], n_active=n_active[i], n_spikes=n_spikes[i],
                     **score_event(trel, P, RNG_SCORE)))
    posteriors[i] = (trel, P)

df = pd.DataFrame(rows)
df["significant"] = df.p_max < ALPHA
df["forward"] = df.slope * np.where(df.direction == "rightward", 1.0, -1.0) > 0
df["speed"] = df.slope.abs()
print(f"{len(df)} events scored")

# %% [markdown]
# ### Pipeline-level null: shuffling which place field belongs to which cell
#
# The within-event shuffles test one event at a time. A stronger check is to break the
# relationship between cells and positions entirely, keeping the spike trains and the
# shape of the field library intact, and run the whole pipeline again. The fraction of
# events this calls significant is the false-positive rate of the analysis as a whole.
# It is run separately on a matched subset of PRE and POST non-REM events, so each
# epoch has its own paired control rather than sharing one number.

# %%
ctrl = {}
for _ep_name in ["PRE_NREM", "POST_NREM"]:
    _ids = df[df.epoch == _ep_name].event.values
    _sub_idx = np.array(sorted(RNG_CTRL.choice(
        _ids, size=min(N_CONTROL_EVENTS, _ids.size), replace=False)))
    _sub = nap.IntervalSet(start=events.start[_sub_idx], end=events.end[_sub_idx])
    _fr = []
    for _ in tqdm(range(N_IDENTITY_SHUFFLE), desc=f"identity shuffle {_ep_name}"):
        _perm = RNG_CTRL.permutation(combined.shape[1])
        _, _ps = nap.decode_bayes(make_tc(combined[:, _perm]), place_cells, _sub, BIN)
        _pe = split_posterior(_ps, _sub)
        _fr.append(np.mean([
            score_event(p[0], p[1], RNG_CTRL, n_shuffle=500)["p_max"] < ALPHA
            for p in _pe if p is not None and p[1].shape[0] >= MIN_BINS]))
    ctrl[_ep_name] = np.array(_fr)
    print(f"{_ep_name}: identity shuffle {100*ctrl[_ep_name].mean():.1f}% "
          f"(SD {100*ctrl[_ep_name].std():.1f}%) vs "
          f"{100*df[df.epoch == _ep_name].significant.mean():.1f}% with real fields")
ctrl_frac = np.concatenate([ctrl["PRE_NREM"], ctrl["POST_NREM"]])

# %% [markdown]
# ## 7. Results

# %%
summary = pd.DataFrame([
    dict(epoch=k, minutes=float(label_eps[k].tot_length()) / 60,
         n_events=int((df.epoch == k).sum()),
         n_sig=int(df[df.epoch == k].significant.sum()),
         frac_sig=float(df[df.epoch == k].significant.mean()),
         median_abs_wcorr=float(df[df.epoch == k].wcorr.abs().median()),
         events_per_min=(df.epoch == k).sum() / (float(label_eps[k].tot_length()) / 60),
         sig_per_min=int(df[df.epoch == k].significant.sum()) /
                     (float(label_eps[k].tot_length()) / 60))
    for k in EPOCH_ORDER])
print(summary.to_string(index=False))

pre, post, maze = (df[df.epoch == k] for k in ["PRE_NREM", "POST_NREM", "MAZE"])
tests = {
    "fisher_POST_vs_PRE_frac_sig": stats.fisher_exact(
        [[int(post.significant.sum()), int((~post.significant).sum())],
         [int(pre.significant.sum()), int((~pre.significant).sum())]],
        alternative="greater"),
    "mwu_POST_vs_PRE_abs_wcorr": stats.mannwhitneyu(
        post.wcorr.abs(), pre.wcorr.abs(), alternative="greater"),
    "binom_POST_vs_own_identity_control": stats.binomtest(
        int(post.significant.sum()), len(post), ctrl["POST_NREM"].mean(),
        alternative="greater"),
    "binom_PRE_vs_own_identity_control": stats.binomtest(
        int(pre.significant.sum()), len(pre), ctrl["PRE_NREM"].mean(),
        alternative="greater"),
    "binom_MAZE_vs_pooled_identity_control": stats.binomtest(
        int(maze.significant.sum()), len(maze), ctrl_frac.mean(), alternative="greater"),
}
print()
for k, v in tests.items():
    print(f"  {k}: p = {v.pvalue:.3g}")

_sig = df[df.significant]
fwd_rev = {k: (int(_sig[_sig.epoch == k].forward.sum()),
               int((~_sig[_sig.epoch == k].forward).sum())) for k in EPOCH_ORDER}
print("\nforward / reverse counts:", fwd_rev)
replay_speed = _sig[_sig.epoch.isin(["MAZE", "POST_NREM", "POST_wake"])].speed.median()
print(f"replay speed (significant MAZE + POST events): median {replay_speed:.1f} m/s "
      f"vs {run_speed:.2f} m/s while running "
      f"({replay_speed/run_speed:.0f}x compression)")

df.to_csv("replay_events.csv", index=False)
summary.to_csv("replay_summary.csv", index=False)
with open("statistics.txt", "w") as fh:
    fh.write(summary.to_string(index=False) + "\n\n")
    fh.write(f"cross-validated decoding error during running: "
             f"{np.median(err)*100:.1f} cm (chance {chance_err*100:.1f} cm); "
             f"direction correct {100*dir_acc:.1f}%\n")
    for k, v in ctrl.items():
        fh.write(f"cell-identity shuffle false-positive rate on {k}: "
                 f"{100*v.mean():.2f}% (SD {100*v.std():.2f}%, "
                 f"{N_IDENTITY_SHUFFLE} shuffles x {N_CONTROL_EVENTS} events)\n")
    fh.write("\n")
    for k, v in tests.items():
        fh.write(f"{k}: p = {v.pvalue:.4g}\n")
    fh.write(f"\nforward/reverse counts: {fwd_rev}\n")

# %% [markdown]
# ### Example replayed trajectories

# %%
peak_pos = {"rightward": bin_centers[np.argmax(tc_right, axis=0)],
            "leftward": bin_centers[np.argmax(tc_left, axis=0)]}
_pool = df[df.significant & (df.n_bins >= 8)]
picks = []
for ep in ["MAZE", "POST_NREM"]:
    for fwd in [True, False]:
        s = _pool[(_pool.epoch == ep) & (_pool.forward == fwd)]
        picks.extend(s.sort_values("wcorr", key=np.abs, ascending=False)
                     .head(2).to_dict("records"))

cmap = LinearSegmentedColormap.from_list("post", ["white", "#3B4CC0", "#B40426"])
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.3)
for k, ev in enumerate(picks[:8]):
    r, c = divmod(k, 4)
    trel, P = posteriors[int(ev["event"])]
    blk = slice(0, NB_BINS) if ev["direction"] == "rightward" else slice(NB_BINS, None)
    Pb = P[:, blk]
    Pb = Pb / Pb.sum(axis=1, keepdims=True)

    _sub_gs = gs[r, c].subgridspec(2, 1, height_ratios=[0.55, 1], hspace=0.08)
    ax_r = fig.add_subplot(_sub_gs[0])
    ax_p = fig.add_subplot(_sub_gs[1], sharex=ax_r)
    pk = peak_pos[ev["direction"]]
    for j, u in enumerate(place_cells.keys()):
        st = place_cells[u].get(ev["start"], ev["end"]).index.values
        if st.size:
            ax_r.plot((st - ev["start"]) * 1000, np.full(st.size, pk[j]), "|",
                      ms=5, color="k")
    ax_r.set_ylim(0, TRACK_LEN)
    ax_r.set_ylabel("field peak (m)", fontsize=8)
    ax_r.tick_params(labelsize=8, labelbottom=False)
    ax_r.set_title(f"{ev['epoch']}  t={ev['start']:.0f} s\n{ev['direction']} "
                   f"{'forward' if ev['forward'] else 'reverse'}, "
                   f"r={ev['wcorr']:.2f}, p={ev['p_max']:.3f}", fontsize=9)

    dt = np.median(np.diff(trel)) if trel.size > 1 else BIN
    ax_p.imshow(Pb.T, aspect="auto", origin="lower", cmap=cmap,
                extent=[0, (trel[-1] + dt) * 1000, 0, TRACK_LEN],
                vmin=0, vmax=np.percentile(Pb, 99.5))
    w = Pb / Pb.sum()
    mx = (w * bin_centers[None, :]).sum()
    mt = (w * trel[:, None]).sum()
    tl = np.array([0, trel[-1] + dt])
    ax_p.plot(tl * 1000, mx + ev["slope"] * (tl - mt), "--", lw=1.6, color="0.25")
    ax_p.set_ylim(0, TRACK_LEN)
    ax_p.set_xlabel("time in event (ms)", fontsize=8)
    ax_p.set_ylabel("decoded position (m)", fontsize=8)
    ax_p.tick_params(labelsize=8)
fig.suptitle("Decoded trajectories during sharp-wave ripples "
             "(top: place-cell spikes ordered by field peak; bottom: posterior)",
             fontsize=13)
plt.savefig("fig05_example_replay_events.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote fig05_example_replay_events.png")

# %% [markdown]
# ### Population summary

# %%
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)
sm = summary.set_index("epoch").loc[EPOCH_ORDER]
x = np.arange(len(EPOCH_ORDER))
_cols = [EPOCH_COLOR[e] for e in EPOCH_ORDER]

ax = fig.add_subplot(gs[0, 0])
ax.bar(x, sm.frac_sig * 100, color=_cols)
for k, v in ctrl.items():
    xi = EPOCH_ORDER.index(k)
    ax.errorbar(xi + 0.32, v.mean() * 100, yerr=v.std() * 100, fmt="_", ms=16,
                color="k", lw=1.5, capsize=4,
                label="cell-identity shuffle" if k == "PRE_NREM" else None)
for xi in range(len(x)):
    ax.text(xi - 0.1, sm.frac_sig.iloc[xi] * 100 + 0.6,
            f"{sm.n_sig.iloc[xi]}/{sm.n_events.iloc[xi]}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(EPOCH_ORDER, rotation=20, ha="right")
ax.set_ylim(0, 24)
ax.set_ylabel("% of SWR events with\nsignificant replay")
ax.set_title("Replay is above chance only\nafter the animal has run the maze", fontsize=11)
ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[0, 1])
for e in EPOCH_ORDER:
    v = np.sort(df[df.epoch == e].wcorr.abs().values)
    ax.plot(v, np.linspace(0, 1, v.size), color=EPOCH_COLOR[e], lw=2, label=e)
ax.set_xlabel("|weighted correlation| of the decoded trajectory")
ax.set_ylabel("cumulative fraction of events")
ax.set_title("Sequence quality of every event", fontsize=11)
ax.legend(fontsize=8, loc="lower right")

ax = fig.add_subplot(gs[0, 2])
ax.bar(x, sm.sig_per_min, color=_cols)
ax.set_xticks(x)
ax.set_xticklabels(EPOCH_ORDER, rotation=20, ha="right")
ax.set_ylabel("significant replay events per minute")
ax.set_title("Rate of replay", fontsize=11)

ax = fig.add_subplot(gs[1, 0])
_s = df[df.significant & df.epoch.isin(["MAZE", "POST_NREM", "POST_wake"])]
ax.hist(_s.speed, bins=np.linspace(0, 20, 30), color="#8172B3", edgecolor="w")
ax.axvline(_s.speed.median(), color="#C44E52", ls="--",
           label=f"median {_s.speed.median():.1f} m/s")
ax.axvline(run_speed, color="k", ls=":", label=f"running {run_speed:.2f} m/s")
ax.set_xlabel("replay speed |dx/dt| (m/s)")
ax.set_ylabel("# events")
ax.set_title("Replayed trajectories are time-compressed", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
_c = (df[df.significant].groupby(["epoch", "forward"]).size()
      .unstack(fill_value=0).reindex(EPOCH_ORDER).fillna(0))
w = 0.38
ax.bar(x - w / 2, _c.get(True, 0), w, color="#4C72B0", label="forward")
ax.bar(x + w / 2, _c.get(False, 0), w, color="#C44E52", label="reverse")
ax.set_xticks(x)
ax.set_xticklabels(EPOCH_ORDER, rotation=20, ha="right")
ax.set_ylabel("# significant replay events")
ax.set_title("Forward vs reverse\n(the same bias appears in chance-level PRE)",
             fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
p = df[df.epoch.isin(["POST_NREM", "POST_wake"])].copy()
p["hrs"] = (p.start - epochs["POST"].start[0]) / 3600
_bins = np.arange(0, p.hrs.max() + 0.5, 0.5)
_tot, _ = np.histogram(p.hrs, bins=_bins)
_ns, _ = np.histogram(p[p.significant].hrs, bins=_bins)
_ctr = (_bins[:-1] + _bins[1:]) / 2
_ok = _tot > 30
ax.plot(_ctr[_ok], 100 * _ns[_ok] / _tot[_ok], "o-", color="#55A868", label="POST")
ax.axhline(ctrl["POST_NREM"].mean() * 100, color="k", ls="--", lw=1,
           label="identity shuffle")
ax.axhline(sm.loc["PRE_NREM", "frac_sig"] * 100, color="#4C72B0", ls=":", lw=1.5,
           label="PRE non-REM")
ax.set_ylim(0, None)
ax.set_xlabel("hours into POST sleep")
ax.set_ylabel("% significant replay events")
ax.set_title("Replay across POST sleep", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("Hippocampal replay of a novel 1.6 m linear track "
             f"(DANDI:000044, rat Achilles, {len(place_ids)} CA1 place cells)",
             fontsize=14)
plt.savefig("fig06_replay_summary.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote fig06_replay_summary.png")

# %% [markdown]
# ## Interpretation
#
# **The decoder is sound.** Cross-validated decoding of real running position from
# these place fields has a median error of about 5 cm on a 1.6 m track (chance ~50 cm)
# and recovers the running direction in ~96% of 250 ms bins. Whatever the decoder
# reports inside a ripple, it is not reporting noise about running behaviour.
#
# **Replay is present, and it is specific to experience.** The pipeline calls 6.2%
# of PRE-sleep SWR events "replay of the maze". The cell-identity shuffle run on the
# same PRE events also returns 6.2%, so PRE sits exactly on the noise floor
# (binomial p = 0.50). That is the correct answer for PRE: the animal had never been
# in the maze room. The identical analysis on POST sleep gives 8.4% against a 5.4%
# identity-shuffle floor on the same events (p = 2e-6), and POST exceeds PRE directly
# (Fisher p = 0.016) with higher per-event sequence quality as well (Mann-Whitney on
# |weighted correlation|, p = 1e-4). Awake SWRs recorded on the track itself are by
# far the strongest: 20.7% of them contain a significant trajectory (p = 5e-13).
# Because the SWR rate is also more than twice as high in POST as in PRE, the rate of
# significant replay events per minute of sleep is roughly 2.5x higher after the maze.
#
# **The replayed trajectories look like compressed running.** Significant events
# sweep across the track at a median of about 4.6 m/s, roughly eight times the
# animal's ~0.56 m/s running speed, and the individual posteriors show the
# characteristic clean diagonal band rather than a scatter of positions.
#
# **One caveat is worth stating.** Reverse-going trajectories outnumber forward-going
# ones in every epoch, including PRE, where the content is at chance. A genuine
# forward/reverse asymmetry cannot be read off these counts, because whatever produces
# the asymmetry is already present when there is nothing to replay. It most likely
# reflects the non-uniform distribution of place-field peaks combined with the
# stereotyped time course of a population burst. Resolving it would need a control
# that matches the field distribution, which is beyond what is done here.
#
# **Scope.** This is a single session from one animal. The PRE/POST contrast is
# within-session and therefore well controlled for cell yield and field quality, but
# the size of the POST-over-PRE effect should not be generalised from n = 1 session.
# The dataset contains eight sessions from four rats, and the code above runs on any
# of them by changing `S3_URL`.
