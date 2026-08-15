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
# # Hippocampal place cells in rat CA1
#
# ## DANDI:000044, Grosmark & Buzsáki (2016)
#
# A hippocampal place cell fires when the animal occupies a particular part of its
# environment and is nearly silent elsewhere. This notebook demonstrates the
# phenomenon from raw archived data: it streams eight bilateral CA1 silicon-probe
# recordings from the DANDI Archive, builds occupancy-normalized firing rate maps
# for every well-isolated pyramidal cell, tests each cell against a shuffled null,
# and then shows that the resulting population code is rich enough to reconstruct
# the animal's position from spikes alone.
#
# **Dataset.** [DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity
# in neural firing dynamics supports both rigid and learned hippocampal sequences"
# (Grosmark & Buzsáki, *Science* 2016). Four Long-Evans rats (Achilles, Buddy,
# Cicero, Gatsby), eight sessions. Each session contains a pre-task sleep epoch, a
# maze epoch in which the rat shuttles for water reward on a 1.6 m or 2 m linear
# track or runs laps on a circular track, and a post-task sleep epoch. The NWB
# files provide spike-sorted units labelled as excitatory or inhibitory, the
# animal's 2D position and a linearized track coordinate at ~39 Hz, and a 128- or
# 96-channel 1250 Hz LFP.
#
# **What is analysed.**
#
# 1. Behaviour and raw spiking during single track traversals.
# 2. Occupancy-normalized rate maps, computed separately per running direction.
# 3. Place-cell classification against a circular-shift null, with spatial
#    information, field width, sparsity, and split-half reliability.
# 4. Cross-validated Bayesian decoding of position and running direction.
# 5. Theta phase precession within place fields, from the CA1 LFP.
# 6. Replication across all eight sessions and four animals.
# 7. A Poisson GLM (NeMoS) comparing position, direction, speed, and spike history.
#
# Data are streamed with `remfile` and a local disk cache, so nothing is downloaded
# in full: the NWB files are 5-9 GB each but only the units table, the position
# series, and a few LFP channels are ever read.
#
# Expect roughly 25-35 minutes of wall-clock time on a first run, dominated by the
# spike-shuffling null (200 surrogates per unit per direction per session) and by
# the eight-session loop.

# %% [markdown]
# ## Setup

# %%
import json
import os
import urllib.error
import urllib.request
import warnings

warnings.simplefilter("ignore")

import h5py
import matplotlib
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import stats
from scipy.signal import hilbert
from tqdm.auto import tqdm

matplotlib.rcParams["figure.dpi"] = 110
matplotlib.rcParams["savefig.bbox"] = "tight"

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
DANDISET = "000044"
COL = {"right": "#1f77b4", "left": "#d62728"}
DIRS = ("right", "left")

# Analysis parameters, fixed once so every figure uses the same numbers.
BIN_WIDTH = 0.02       # m, spatial bin for the rate maps
SPEED_THRESH = 0.05    # m/s, minimum running speed
SPEED_SMOOTH_S = 0.25  # s, boxcar applied to the velocity trace
MIN_RUN_DUR = 0.4      # s, discard very short run fragments
MIN_TRAVERSAL_SPAN = 0.5   # m, a traversal must cover at least this much track
MIN_TRAVERSALS = 10    # per direction, else that direction is not analysed
TC_SMOOTH_BINS = 1.5   # gaussian sigma, in bins, for the rate maps
MIN_MEAN_RATE = 0.1    # Hz over the maze epoch, to keep a unit at all
MAX_MEAN_RATE = 5.0    # Hz, excludes fast-firing units from the pyramidal pool
MIN_PEAK_RATE = 1.0    # Hz, in-field peak of the smoothed rate map
MIN_RELIABILITY = 0.4  # odd/even traversal map correlation
MIN_SPIKES = 30        # spikes emitted while running in that direction
N_SHUFFLE = 200        # surrogates for the spatial-information null

# %% [markdown]
# ### Locating the assets on DANDI
#
# The DANDI REST API is asked for every NWB asset in the dandiset; the asset
# download endpoint redirects to a stable S3 blob URL, which is what `remfile`
# streams from.

# %%
def resolve_session_urls(dandiset=DANDISET):
    """Map session name -> stable S3 blob URL for every NWB asset."""
    api = f"https://api.dandiarchive.org/api/dandisets/{dandiset}/versions/draft/assets/"
    assets = []
    url = api + "?page_size=100"
    while url:
        with urllib.request.urlopen(url) as r:
            page = json.load(r)
        assets += page["results"]
        url = page.get("next")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **kw):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    out = {}
    for a in sorted(assets, key=lambda a: a["path"]):
        if not a["path"].endswith(".nwb"):
            continue
        name = a["path"].split("/")[-1].split("_ses-")[1].split("_")[0]
        req = urllib.request.Request(api + a["asset_id"] + "/download/", method="HEAD")
        try:
            opener.open(req)
            raise RuntimeError("expected a redirect to S3")
        except urllib.error.HTTPError as e:
            out[name] = e.headers["Location"].split("?")[0]
    return out


SESSION_URLS = resolve_session_urls()
for k, v in SESSION_URLS.items():
    print(f"{k:20s} {v}")

# %% [markdown]
# ## Loading and inspecting one session
#
# `Achilles_10252013` is used to develop the analysis: a 1.6 m linear track with
# 137 sorted units in left and right CA1.

# %%
def load_session(url):
    """Stream one NWB file from the DANDI S3 bucket and wrap it for pynapple."""
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


SESSION = "Achilles_10252013"
nwb, nwbfile, io = load_session(SESSION_URLS[SESSION])
print(nwb)
print()
print(nwbfile.epochs.to_dataframe())
print()
print(f"subject {nwbfile.subject.subject_id}, {nwbfile.subject.species}, "
      f"{nwbfile.subject.sex}")
print(f"units table columns: {nwbfile.units.colnames}")
print(nwbfile.units.to_dataframe()[["location", "shank_id", "cell_type"]].groupby(
    ["location", "cell_type"]).size())

# %% [markdown]
# ### Behavioural variables
#
# The linearized position is defined only while the animal is on the track proper.
# Samples recorded in the reward areas at either end are NaN and are dropped, which
# means the position trace consists of one contiguous stretch per traversal
# separated by long gaps. That structure has to be respected when differentiating
# to get velocity, otherwise the gaps produce spurious velocities.

# %%
def maze_name(nwbfile):
    """e.g. '1.6mLinearMaze', 'CircularMaze', '2mLinearMaze'."""
    b = nwbfile.processing["behavior"]
    key = [k for k in b.data_interfaces if k.endswith("LinearizedPosition")][0]
    return key.replace("LinearizedPosition", "")


def maze_epoch(nwbfile):
    df = nwbfile.epochs.to_dataframe()
    row = df[df.label == "MazeEpoch"]
    return nap.IntervalSet(start=row.start_time.values, end=row.stop_time.values)


def get_position(nwb, nwbfile):
    """Linearized position as a Tsd, plus the track length in metres."""
    tsdframe = nwb[maze_name(nwbfile) + "LinearizedTimeSeries"]
    v = np.asarray(tsdframe.values).ravel()
    t = np.asarray(tsdframe.t)
    ok = np.isfinite(v)
    return nap.Tsd(t=t[ok], d=v[ok]), float(np.ceil(np.nanmax(v) / BIN_WIDTH) * BIN_WIDTH)


def get_position_2d(nwb, nwbfile):
    tsdframe = nwb[maze_name(nwbfile) + "SpatialSeries"]
    v, t = np.asarray(tsdframe.values), np.asarray(tsdframe.t)
    ok = np.isfinite(v).all(axis=1)
    return nap.TsdFrame(t=t[ok], d=v[ok], columns=["x", "y"])


def n_bins(track_length):
    return int(round(track_length / BIN_WIDTH))


def _boxcar(x, n):
    return x if n < 2 else np.convolve(x, np.ones(n) / n, mode="same")


def _mask_to_intervalset(t, mask, med_dt):
    """Contiguous True runs of `mask` -> IntervalSet, breaking on sampling gaps."""
    m = mask.astype(int) * np.concatenate([[True], np.diff(t) < 5 * med_dt])
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    if len(starts) == 0:
        return nap.IntervalSet(start=np.array([]), end=np.array([]))
    return nap.IntervalSet(start=t[starts], end=t[ends])


def running_epochs(pos, speed_thresh=SPEED_THRESH, min_dur=MIN_RUN_DUR):
    """Split track traversals into rightward and leftward running epochs."""
    t = pos.t
    dt = np.diff(t)
    med_dt = np.median(dt)
    seg = np.concatenate([[0], np.where(dt > 5 * med_dt)[0] + 1, [len(t)]])

    vel = np.full(len(t), np.nan)
    for a, b in zip(seg[:-1], seg[1:]):
        if b - a >= 3:
            vel[a:b] = np.gradient(pos.values[a:b], t[a:b])
    nsm = max(1, int(round(SPEED_SMOOTH_S / med_dt)))
    for a, b in zip(seg[:-1], seg[1:]):
        if b - a > nsm:
            vel[a:b] = _boxcar(vel[a:b], nsm)

    ok = np.isfinite(vel)
    vel_tsd = nap.Tsd(t=t[ok], d=vel[ok])

    eps = {}
    for name, mask in [("right", vel_tsd.values > speed_thresh),
                       ("left", vel_tsd.values < -speed_thresh)]:
        ep = _mask_to_intervalset(vel_tsd.t, mask, med_dt).drop_short_intervals(min_dur)
        keep = []
        for k in range(len(ep)):
            p = pos.restrict(ep[k:k + 1])
            if len(p) > 2 and (p.values.max() - p.values.min()) >= MIN_TRAVERSAL_SPAN:
                keep.append(k)
        eps[name] = nap.IntervalSet(start=ep.start[keep], end=ep.end[keep])
    eps["run"] = eps["right"].union(eps["left"])
    return vel_tsd, eps


maze = maze_epoch(nwbfile)
pos, L = get_position(nwb, nwbfile)
pos2d = get_position_2d(nwb, nwbfile)
vel, eps = running_epochs(pos)
print(f"{maze_name(nwbfile)}, track length {L:.2f} m")
print(f"maze epoch {maze.tot_length():.0f} s; position sampled at "
      f"{1 / np.median(np.diff(pos.t)):.1f} Hz on the track "
      f"({len(pos)} valid samples)")
print(f"traversals: {len(eps['right'])} rightward, {len(eps['left'])} leftward, "
      f"{eps['run'].tot_length():.0f} s of running in total")

# %% [markdown]
# ### Unit selection
#
# The units table already carries a `cell_type` label. Putative pyramidal cells are
# the excitatory units whose mean rate over the maze epoch falls between 0.1 and
# 5 Hz; the upper bound removes a handful of units with interneuron-like rates that
# are nonetheless labelled excitatory.

# %%
def select_pyramidal(units, maze_ep, min_rate=MIN_MEAN_RATE, max_rate=MAX_MEAN_RATE):
    rates = units.restrict(maze_ep).rate
    ctype = units.get_info("cell_type")
    keep = [i for i in units.index
            if ctype[i] == "excitatory" and min_rate <= rates[i] <= max_rate]
    return units[keep]


units = nwb["units"]
pyr = select_pyramidal(units, maze)
print(f"{len(units)} sorted units -> {len(pyr)} putative CA1 pyramidal cells")
print(pyr.restrict(maze).rate.describe())

# %% [markdown]
# ## Figure 1: behaviour and raw population activity
#
# The rat shuttles end to end, pausing at the reward areas. The bottom panels take
# a single traversal in each direction and plot every pyramidal cell's spikes with
# the cells ordered by where their rightward place field peaks. The activity sweeps
# diagonally through the ordered population as the animal advances, which is the
# raw signature of a place code, and it does so only for the direction the ordering
# was derived from.

# %%
def tuning_curves(units, pos, ep, track_length):
    """Occupancy-normalized 1D rate maps via pynapple.

    Returns (DataFrame [bin x unit] in Hz, occupancy in seconds, bin edges).
    """
    fs = 1.0 / np.median(np.diff(pos.t))
    xr = nap.compute_tuning_curves(
        units, pos, bins=n_bins(track_length), range=[(0.0, track_length)],
        epochs=ep, fs=fs,
    )
    bins = np.asarray(xr.attrs["bin_edges"]).ravel()
    occ = np.asarray(xr.attrs["occupancy"]).ravel().astype(float) / fs
    tc = pd.DataFrame(np.asarray(xr.values).T,
                      index=0.5 * (bins[:-1] + bins[1:]), columns=list(units.index))
    return tc, occ, bins


def smooth_tc(tc, sigma=TC_SMOOTH_BINS):
    """Gaussian-smooth each column of a tuning-curve DataFrame (no wrap-around)."""
    if sigma <= 0:
        return tc
    half = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    out = tc.copy()
    for c in tc.columns:
        y = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        pad = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
        out[c] = np.convolve(pad, k, mode="same")[half:-half]
    return out


def field_stats(tc, bins):
    """Peak rate, peak location, and field width (contiguous >=50% of peak)."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    bw = centers[1] - centers[0]
    rows = []
    for c in tc.columns:
        lam = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        i = int(np.argmax(lam))
        peak = lam[i]
        if peak <= 0:
            rows.append((c, 0.0, np.nan, np.nan))
            continue
        thr = 0.5 * peak
        lo, hi = i, i
        while lo > 0 and lam[lo - 1] >= thr:
            lo -= 1
        while hi < len(lam) - 1 and lam[hi + 1] >= thr:
            hi += 1
        rows.append((c, peak, centers[i], (hi - lo + 1) * bw))
    return pd.DataFrame(rows, columns=["unit", "peak_rate", "peak_pos", "width"]
                        ).set_index("unit")


tc_raw, occ_raw, bins_raw = {}, {}, None
tcs, occs, fst = {}, {}, {}
for d in DIRS:
    tc, occ, bins = tuning_curves(pyr, pos, eps[d], L)
    tcs[d], occs[d] = smooth_tc(tc), occ
    fst[d] = field_stats(tcs[d], bins)
    bins_raw = bins
centers = 0.5 * (bins_raw[:-1] + bins_raw[1:])
uidx = np.asarray(pyr.index)

# %%
order = np.argsort(fst["right"].peak_pos.values)
sorted_units = uidx[order]
t0 = float(eps["right"].start[6])
win = nap.IntervalSet(start=t0 - 10, end=t0 + 290)

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 2.6],
                      width_ratios=[1.15, 1.15, 1.0], hspace=0.4, wspace=0.55)

ax = fig.add_subplot(gs[0, 0:2])
p = pos.restrict(win)
ax.plot(p.t, p.values, "k.", ms=2)
for d in DIRS:
    ei = eps[d].intersect(win)
    for k in range(len(ei)):
        ax.axvspan(float(ei.start[k]), float(ei.end[k]), color=COL[d], alpha=0.25, lw=0)
ax.set_ylabel("track position (m)")
ax.set_title(f"{SESSION}: shuttling on the {L:.1f} m linear track "
             "(blue = rightward, red = leftward)", fontsize=11)
ax.set_xlim(float(win.start[0]), float(win.end[0]))
ax.set_xticklabels([])

ax = fig.add_subplot(gs[1, 0:2])
v = vel.restrict(win)
ax.plot(v.t, v.values, "k-", lw=0.7)
ax.axhline(0, color="0.6", lw=0.5)
for sv in (SPEED_THRESH, -SPEED_THRESH):
    ax.axhline(sv, color="g", ls="--", lw=0.7)
ax.set_ylabel("velocity (m/s)")
ax.set_xlabel("time (s)")
ax.set_xlim(float(win.start[0]), float(win.end[0]))

for col, d in enumerate(DIRS):
    e = eps[d][6:7]
    pad = 0.5
    zw = nap.IntervalSet(start=float(e.start[0]) - pad, end=float(e.end[0]) + pad)
    ax = fig.add_subplot(gs[2, col])
    for row, uid in enumerate(sorted_units):
        st = np.asarray(pyr[uid].restrict(zw).t)
        ax.plot(st - float(e.start[0]), np.full(len(st), row), "|",
                color="k", ms=4, mew=0.9)
    ax.axvspan(0, float(e.end[0]) - float(e.start[0]), color=COL[d], alpha=0.15, lw=0)
    ax2 = ax.twinx()
    pz = pos.restrict(zw)
    ax2.plot(np.asarray(pz.t) - float(e.start[0]), pz.values, color=COL[d], lw=2)
    ax2.set_ylim(0, L)
    ax2.set_ylabel("position (m)", color=COL[d])
    ax2.tick_params(axis="y", colors=COL[d])
    ax.set_xlim(-pad, float(e.end[0]) - float(e.start[0]) + pad)
    ax.set_ylim(-1, len(sorted_units))
    ax.set_xlabel("time from run onset (s)")
    if col == 0:
        ax.set_ylabel("CA1 pyramidal cell\n(sorted by rightward field position)")
    ax.set_title(f"single {d}ward traversal", fontsize=11)

ax = fig.add_subplot(gs[0:2, 2])
pp = pos2d.restrict(maze)
lin_at = np.interp(np.asarray(pp.t), pos.t, pos.values, left=np.nan, right=np.nan)
sc = ax.scatter(pp.values[:, 0], pp.values[:, 1], c=lin_at, s=1, cmap="viridis")
plt.colorbar(sc, ax=ax, label="linearized (m)", fraction=0.08)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("2D tracking, whole maze epoch", fontsize=11)

ax = fig.add_subplot(gs[2, 2])
for d in DIRS:
    ax.plot(centers, occs[d], color=COL[d], label=f"{d} ({len(eps[d])} runs)")
ax.set_xlabel("track position (m)")
ax.set_ylabel("occupancy (s / 2 cm bin)")
ax.set_title("Occupancy", fontsize=11)
ax.legend(fontsize=8)
ax.set_ylim(0, None)
fig.savefig("fig01_behavior_and_raster.png", dpi=150)
plt.show()

# %% [markdown]
# ## Place-cell criteria
#
# Rate maps alone do not establish that a cell is spatially tuned: a cell that
# fires only a handful of times will have an apparently sharp map by chance, and
# spatial information per spike is strongly inflated at low spike counts. Four
# criteria are therefore applied together.
#
# 1. **Spatial information above a circular-shift null.** Skaggs information,
#    $\mathrm{SI} = \sum_i p_i \frac{\lambda_i}{\bar\lambda}
#    \log_2 \frac{\lambda_i}{\bar\lambda}$ bits per spike, must exceed the 95th
#    percentile of the same statistic computed from 200 surrogates in which the
#    cell's spike train is circularly shifted along a timeline formed by
#    concatenating the running epochs. The shift preserves spike count and
#    fine-scale ISI structure but destroys alignment to position. The surrogate
#    maps are smoothed exactly like the observed maps, which matters: smoothing
#    lowers spatial information, so comparing a smoothed observation against an
#    unsmoothed null would throw away most of the test's power.
# 2. **In-field peak rate** of at least 1 Hz.
# 3. **Split-half reliability**: the correlation between rate maps built from
#    odd-numbered and even-numbered traversals must exceed 0.4.
# 4. **At least 30 spikes** emitted while running in that direction.
#
# Because place fields on a linear track are direction-selective, every quantity is
# computed separately for rightward and leftward runs, and a unit is counted as a
# place cell if it qualifies in at least one direction.

# %%
def spatial_info_bits_per_spike(tc, occ):
    p = occ / occ.sum()
    out = {}
    for c in tc.columns:
        lam = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        lam_bar = np.sum(p * lam)
        if lam_bar <= 0:
            out[c] = 0.0
            continue
        nz = lam > 0
        ratio = lam[nz] / lam_bar
        out[c] = float(np.sum(p[nz] * ratio * np.log2(ratio)))
    return pd.Series(out)


def sparsity(tc, occ):
    """Spatial sparsity <lam>^2 / <lam^2>; low values mean a compact field."""
    p = occ / occ.sum()
    out = {}
    for c in tc.columns:
        lam = np.nan_to_num(tc[c].values.astype(float), nan=0.0)
        den = np.sum(p * lam ** 2)
        out[c] = float(np.sum(p * lam) ** 2 / den) if den > 0 else np.nan
    return pd.Series(out)


def map_corr(a, b):
    a, b = np.nan_to_num(np.asarray(a, float)), np.nan_to_num(np.asarray(b, float))
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def shuffled_si(units, pos, ep, occ, track_length, n_shuffle=N_SHUFFLE, rng=None):
    """Null distribution of spatial information from circularly shifted spikes."""
    rng = rng if rng is not None else np.random.default_rng(0)
    ep_dur = ep.tot_length()
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    cum = np.concatenate([[0], np.cumsum(ends - starts)])

    def to_flat(ts):
        i = np.clip(np.searchsorted(starts, ts, side="right") - 1, 0, len(starts) - 1)
        return cum[i] + (ts - starts[i])

    def from_flat(x):
        i = np.clip(np.searchsorted(cum[1:], x, side="right"), 0, len(starts) - 1)
        return starts[i] + (x - cum[i])

    flat = {i: to_flat(np.asarray(units[i].restrict(ep).t)) for i in units.index}
    null = {i: np.zeros(n_shuffle) for i in units.index}
    for s in range(n_shuffle):
        shift = rng.uniform(0.1 * ep_dur, 0.9 * ep_dur)
        shifted = {}
        for i, fl in flat.items():
            if len(fl) == 0:
                shifted[i] = nap.Ts(t=np.array([]), time_support=ep)
            else:
                shifted[i] = nap.Ts(t=np.sort(from_flat(np.mod(fl + shift, ep_dur))),
                                    time_support=ep)
        grp = nap.TsGroup(shifted, time_support=ep)
        tc, _, _ = tuning_curves(grp, pos, ep, track_length)
        si = spatial_info_bits_per_spike(smooth_tc(tc), occ)
        for i in units.index:
            null[i][s] = si[i]
    thr = pd.Series({i: np.percentile(null[i], 95) for i in units.index})
    return thr, null


def direction_metrics(pyr, pos, ep, track_length, n_shuffle=N_SHUFFLE, rng=None):
    """All place-field metrics and the place-cell call for one running direction."""
    rng = rng if rng is not None else np.random.default_rng(0)
    tc, occ, bins = tuning_curves(pyr, pos, ep, track_length)
    tcs = smooth_tc(tc)
    thr, null = shuffled_si(pyr, pos, ep, occ, track_length, n_shuffle, rng)
    fstats = field_stats(tcs, bins)

    odd = nap.IntervalSet(start=ep.start[::2], end=ep.end[::2])
    even = nap.IntervalSet(start=ep.start[1::2], end=ep.end[1::2])
    tc_o = smooth_tc(tuning_curves(pyr, pos, odd, track_length)[0])
    tc_e = smooth_tc(tuning_curves(pyr, pos, even, track_length)[0])

    df = pd.DataFrame({
        "si": spatial_info_bits_per_spike(tcs, occ),
        "si_thresh": thr,
        "sparsity": sparsity(tcs, occ),
        "reliability": pd.Series({i: map_corr(tc_o[i].values, tc_e[i].values)
                                  for i in tc.columns}),
        "n_spikes": pd.Series({i: len(pyr[i].restrict(ep)) for i in pyr.index}),
        "peak_rate": fstats.peak_rate,
        "peak_pos": fstats.peak_pos,
        "width": fstats.width,
    })
    # `is_place_cell_norel` omits the reliability criterion. Comparing a
    # within-direction split-half correlation against a between-direction
    # correlation is only fair if selection did not itself put a floor on the
    # former, so the directionality analyses use this mask instead.
    df["is_place_cell_norel"] = ((df.si > df.si_thresh)
                                & (df.peak_rate >= MIN_PEAK_RATE)
                                & (df.n_spikes >= MIN_SPIKES))
    df["is_place_cell"] = df.is_place_cell_norel & (df.reliability >= MIN_RELIABILITY)
    return {"tc": tc, "tc_smooth": tcs, "occ": occ, "bins": bins,
            "metrics": df, "null": null}


rng = np.random.default_rng(0)
res = {d: direction_metrics(pyr, pos, eps[d], L, rng=rng) for d in DIRS}
mets = {d: res[d]["metrics"] for d in DIRS}
tcs = {d: res[d]["tc_smooth"] for d in DIRS}
occs = {d: res[d]["occ"] for d in DIRS}
nulls = {d: res[d]["null"] for d in DIRS}

for d in DIRS:
    m = mets[d]
    print(f"{d:>5}ward: {int(m.is_place_cell.sum()):3d}/{len(pyr)} place cells, "
          f"median SI {np.median(m.si[m.is_place_cell]):.2f} bits/spike, "
          f"median width {100 * np.median(m.width[m.is_place_cell]):.0f} cm")
pc_either = mets["right"].is_place_cell.values | mets["left"].is_place_cell.values
print(f"place cell in at least one direction: {pc_either.sum()}/{len(pyr)} "
      f"({100 * pc_either.mean():.0f}%)")

# %% [markdown]
# ## Figure 2: example place cells
#
# For each of six cells the top two rows plot one dot per spike, at the position
# where it occurred (x) and on which traversal (y). Spikes pile into a vertical
# band that recurs on essentially every traversal, which is the phenomenon itself.
# The bottom row is the corresponding occupancy-normalized rate map.

# %%
def spike_positions(ts, ep):
    """Interpolated track position of each spike, plus its traversal number."""
    xs, trav = [], []
    for k in range(len(ep)):
        st = np.asarray(ts.restrict(ep[k:k + 1]).t)
        if len(st) == 0:
            continue
        xs.append(np.interp(st, pos.t, pos.values))
        trav.append(np.full(len(st), k))
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(trav)


score = np.where(mets["right"].is_place_cell.values & mets["left"].is_place_cell.values,
                 mets["right"].reliability.values + mets["left"].reliability.values, -1.0)
chosen, taken = [], []
for i in np.argsort(-score):
    if score[i] < 0:
        break
    pk = mets["right"].peak_pos.values[i]
    if any(abs(pk - q) < 0.15 for q in taken):
        continue
    chosen.append(i)
    taken.append(pk)
    if len(chosen) == 6:
        break
chosen = [chosen[k] for k in np.argsort([mets["right"].peak_pos.values[i] for i in chosen])]

fig, axs = plt.subplots(3, 6, figsize=(17, 8.5),
                        gridspec_kw={"height_ratios": [2, 2, 1.5],
                                     "hspace": 0.32, "wspace": 0.34})
for j, i in enumerate(chosen):
    uid = uidx[i]
    for r, d in enumerate(DIRS):
        ax = axs[r, j]
        x, tr = spike_positions(pyr[uid], eps[d])
        ax.plot(x, tr, ".", color=COL[d], ms=3)
        ax.set_xlim(0, L)
        ax.set_ylim(-1, len(eps[d]))
        if j == 0:
            ax.set_ylabel(f"{d}ward\ntraversal #")
        if r == 0:
            ax.set_title(f"unit {uid} ({pyr.get_info('location')[uid]})\n"
                         f"SI {mets['right'].si[uid]:.1f} / {mets['left'].si[uid]:.1f}"
                         " bits per spike", fontsize=10)
        ax.set_xticklabels([])
    ax = axs[2, j]
    for d in DIRS:
        ax.plot(centers, tcs[d][uid].values, color=COL[d], lw=1.8, label=d)
    ax.set_xlim(0, L)
    ax.set_ylim(0, None)
    ax.set_xlabel("position (m)")
    if j == 0:
        ax.set_ylabel("firing rate (Hz)")
    if j == 5:
        ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Example CA1 place cells: spikes recur at the same track location on "
             "traversal after traversal", fontsize=13, y=0.97)
fig.savefig("fig02_example_place_cells.png", dpi=150)
plt.show()

# %% [markdown]
# ## Figure 3: the ensemble tiles the track
#
# Stacking the peak-normalized rate maps of all significant place cells, sorted by
# where each field peaks, produces a clean diagonal: the population covers every
# part of the track roughly uniformly. The third panel keeps the ordering derived
# from rightward runs but plots the leftward maps of the same cells. The diagonal
# breaks up, which is the standard demonstration that place fields on a linear
# track are direction-specific rather than purely spatial.

# %%
def norm_maps(mat):
    mx = mat.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    return mat / mx


sel = {d: np.where(mets[d].is_place_cell.values)[0] for d in DIRS}
ordr = {d: sel[d][np.argsort(mets[d].peak_pos.values[sel[d]])] for d in DIRS}

fig, axs = plt.subplots(1, 3, figsize=(16, 6))
for ax, d in zip(axs[:2], DIRS):
    o = ordr[d]
    im = ax.imshow(norm_maps(tcs[d].values[:, o].T), aspect="auto", origin="lower",
                   extent=[0, L, 0, len(o)], cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("track position (m)")
    ax.set_ylabel("place cell (sorted by field peak)")
    ax.set_title(f"{d}ward runs: {len(o)} place cells", fontsize=11)
    plt.colorbar(im, ax=ax, label="rate / peak rate", fraction=0.046)

ax = axs[2]
o = ordr["right"]
im = ax.imshow(norm_maps(tcs["left"].values[:, o].T), aspect="auto", origin="lower",
               extent=[0, L, 0, len(o)], cmap="magma", vmin=0, vmax=1)
ax.set_xlabel("track position (m)")
ax.set_ylabel("same cells, same order as left panel")
ax.set_title("Rightward-sorted cells, leftward maps:\nthe diagonal breaks up "
             "(directionality)", fontsize=11)
plt.colorbar(im, ax=ax, label="rate / peak rate", fraction=0.046)
fig.suptitle(f"{SESSION}: the CA1 place-cell ensemble tiles the whole track", fontsize=13)
fig.tight_layout()
fig.savefig("fig03_population_maps.png", dpi=150)
plt.show()

# %% [markdown]
# ## Figure 4: population statistics
#
# The observed spatial information is far above the circular-shift null; field
# widths, peak rates, sparsity, and reliability all fall in the ranges reported for
# rodent CA1 on linear tracks. Panel (h) makes the directionality claim
# quantitative by contrasting two rate-map correlations for the same cells: the
# split-half correlation within a direction (how repeatable the map is at all)
# against the correlation between the two directions.

# %%
fig, axs = plt.subplots(2, 4, figsize=(19, 9))
axs = axs.ravel()

ax = axs[0]
obs = np.concatenate([mets[d].si.values for d in DIRS])
nul = np.concatenate([np.concatenate([nulls[d][u] for u in uidx]) for d in DIRS])
pcsi = np.concatenate([mets[d].si.values[mets[d].is_place_cell.values] for d in DIRS])
edges = np.linspace(0, 5, 46)
ax.hist(nul, edges, density=True, color="0.7", label=f"circular-shift null (n={len(nul)})")
ax.hist(obs, edges, density=True, histtype="step", color="k", lw=1.8,
        label=f"observed (n={len(obs)})")
ax.hist(pcsi, edges, density=True, histtype="step", color="crimson", lw=1.8,
        label=f"place cells (n={len(pcsi)})")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("probability density")
ax.set_title("(a) Spatial information vs shuffled null", fontsize=11)
ax.legend(fontsize=8)
u_si = stats.mannwhitneyu(obs, nul, alternative="greater")
ax.text(0.55, 0.55, f"Mann-Whitney U\np = {u_si.pvalue:.1e}",
        transform=ax.transAxes, fontsize=8)

ax = axs[1]
handles = []
for d in DIRS:
    m = mets[d]
    ax.scatter(m.si_thresh[~m.is_place_cell], m.si[~m.is_place_cell], s=16, c="0.75",
               edgecolors="none")
    ax.scatter(m.si_thresh[m.is_place_cell], m.si[m.is_place_cell], s=18, c=COL[d],
               edgecolors="none")
    handles.append(plt.Line2D([], [], ls="", marker="o", color=COL[d],
                              label=f"{d}: {int(m.is_place_cell.sum())} place cells"))
handles.append(plt.Line2D([], [], ls="", marker="o", color="0.75", label="not classified"))
ax.plot([0, obs.max()], [0, obs.max()], "k--", lw=1)
ax.set_xlim(0, 3)
ax.set_ylim(0, obs.max() * 1.05)
ax.set_xlabel("95th percentile of own shuffle null (bits/spike)")
ax.set_ylabel("observed SI (bits/spike)")
ax.set_title("(b) Per-unit significance test", fontsize=11)
ax.legend(handles=handles, fontsize=8, loc="upper left")

ax = axs[2]
w = 100 * np.concatenate([mets[d].width.values[mets[d].is_place_cell.values] for d in DIRS])
ax.hist(w, np.arange(0, 105, 5), color="#4c72b0", edgecolor="w")
ax.axvline(np.median(w), color="k", ls="--")
ax.set_xlabel("place-field width (cm, >=50% of peak)")
ax.set_ylabel("count")
ax.set_title(f"(c) Field width, median {np.median(w):.0f} cm", fontsize=11)

ax = axs[3]
pr = np.concatenate([mets[d].peak_rate.values[mets[d].is_place_cell.values] for d in DIRS])
ax.hist(pr, np.arange(0, 55, 2.5), color="#dd8452", edgecolor="w")
ax.axvline(np.median(pr), color="k", ls="--")
ax.set_xlabel("in-field peak rate (Hz)")
ax.set_ylabel("count")
ax.set_title(f"(d) Peak rate, median {np.median(pr):.1f} Hz", fontsize=11)

ax = axs[4]
sp_pc = np.concatenate([mets[d].sparsity.values[mets[d].is_place_cell.values] for d in DIRS])
sp_no = np.concatenate([mets[d].sparsity.values[~mets[d].is_place_cell.values] for d in DIRS])
e = np.linspace(0, 1, 26)
ax.hist(sp_no, e, color="0.75", density=True, label="not classified")
ax.hist(sp_pc, e, histtype="step", color="crimson", lw=1.8, density=True, label="place cells")
ax.set_xlabel(r"sparsity  $\langle\lambda\rangle^2/\langle\lambda^2\rangle$")
ax.set_ylabel("probability density")
ax.set_title(f"(e) Sparsity, place-cell median {np.median(sp_pc):.2f}", fontsize=11)
ax.legend(fontsize=8)

ax = axs[5]
rl_pc = np.concatenate([mets[d].reliability.values[mets[d].is_place_cell.values] for d in DIRS])
rl_no = np.concatenate([mets[d].reliability.values[~mets[d].is_place_cell.values] for d in DIRS])
e = np.linspace(-0.6, 1, 33)
ax.hist(rl_no, e, color="0.75", density=True, label="not classified")
ax.hist(rl_pc, e, histtype="step", color="crimson", lw=1.8, density=True, label="place cells")
ax.axvline(MIN_RELIABILITY, color="g", ls="--", lw=1)
ax.set_xlabel("odd vs even traversal map correlation")
ax.set_ylabel("probability density")
ax.set_title(f"(f) Within-session reliability\nplace-cell median r = "
             f"{np.median(rl_pc):.2f}", fontsize=11)
ax.legend(fontsize=8, loc="upper left")

ax = axs[6]
for d in DIRS:
    ax.hist(mets[d].peak_pos.values[mets[d].is_place_cell.values],
            np.linspace(0, L, 17), histtype="step", lw=1.8, color=COL[d], label=d)
ax.set_xlabel("place-field peak position (m)")
ax.set_ylabel("count")
ax.set_title("(g) Fields cover the whole track", fontsize=11)
ax.legend(fontsize=8)

ax = axs[7]
dir_corr = np.array([map_corr(tcs["right"].values[:, j], tcs["left"].values[:, j])
                     for j in range(len(uidx))])
norel = (mets["right"].is_place_cell_norel.values
         | mets["left"].is_place_cell_norel.values)
dc = dir_corr[norel]
dc = dc[np.isfinite(dc)]
rel_within = np.concatenate(
    [mets[d].reliability.values[mets[d].is_place_cell_norel.values] for d in DIRS])
rel_within = rel_within[np.isfinite(rel_within)]
e = np.linspace(-1, 1, 33)
ax.hist(dc, e, color="0.75", density=True,
        label=f"opposite directions (median {np.median(dc):.2f})")
ax.hist(rel_within, e, histtype="step", color="k", lw=1.8, density=True,
        label=f"same direction, odd vs even (median {np.median(rel_within):.2f})")
ax.axvline(np.median(dc), color="0.4", ls="--")
ax.axvline(np.median(rel_within), color="k", ls="--")
ax.set_xlabel("rate-map correlation")
ax.set_ylabel("probability density")
u_dir = stats.mannwhitneyu(rel_within, dc, alternative="greater")
ax.set_title("(h) Fields are direction-specific:\nmaps repeat within, not across, "
             f"direction (p = {u_dir.pvalue:.1e})", fontsize=11)
ax.legend(fontsize=7.5, loc="upper left")

fig.suptitle(f"{SESSION}: place-cell statistics ({len(pyr)} CA1 pyramidal cells, "
             f"{int(pc_either.sum())} place cells)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig04_statistics.png", dpi=150)
plt.show()

# %% [markdown]
# ## Figure 5: reading the code back out
#
# A place code should let an observer with access only to the spikes recover where
# the animal is. Rate maps are estimated on one half of the traversals and used to
# decode the other half with a Poisson naive-Bayes decoder
# (`nap.decode_bayes`, Zhang et al. 1998), so no data is both fit and tested.
#
# The odd/even split is taken **within each direction**. Splitting the pooled run
# epochs would be wrong here: the rat alternates direction on successive runs, so
# every other traversal is every rightward traversal, and a decoder would be
# trained on one direction and tested on the other. The trailing partial time bin
# of each traversal is also dropped, since it covers less than a full bin's worth
# of spikes.
#
# Position and running direction are decoded jointly, by binning the feature space
# as position x direction and marginalizing the posterior as needed.

# %%
BIN_SIZE = 0.25
BIN_SIZES = [0.05, 0.1, 0.25, 0.5, 1.0]
fs_pos = 1.0 / np.median(np.diff(pos.t))
pc_ids = uidx[pc_either]
pc = pyr[list(pc_ids)]
print(f"decoding from {len(pc)} place cells of {len(pyr)} pyramidal cells")


def joint_feature(ep):
    """(position, direction code) as a 2-column TsdFrame over `ep`."""
    p = pos.restrict(ep)
    v = np.interp(np.asarray(p.t), np.asarray(vel.t), vel.values)
    return nap.TsdFrame(t=p.t, d=np.c_[p.values, (v > 0).astype(float)],
                        columns=["pos", "dir"], time_support=ep)


def half(sl):
    a = nap.IntervalSet(start=eps["right"].start[sl], end=eps["right"].end[sl])
    b = nap.IntervalSet(start=eps["left"].start[sl], end=eps["left"].end[sl])
    return a.union(b)


fold_a, fold_b = half(slice(None, None, 2)), half(slice(1, None, 2))
folds = [(fold_a, fold_b), (fold_b, fold_a)]


def full_bin_mask(t, ep, bin_size):
    """Drop the trailing partial bin of every interval, which is spike-starved."""
    ends = np.asarray(ep.end)
    i = np.clip(np.searchsorted(ends, t, side="left"), 0, len(ends) - 1)
    return t + bin_size / 2 <= ends[i] + 1e-9


def run_cv(bin_size, directional=True, group=None):
    """Cross-validated decoding; returns pooled true and decoded values."""
    group = group if group is not None else pc
    acc = {k: [] for k in ("t", "true_pos", "dec_pos", "true_dir", "dec_dir")}
    for train_ep, test_ep in folds:
        feat_tr = joint_feature(train_ep)
        if directional:
            tc = nap.compute_tuning_curves(group, feat_tr, bins=[n_bins(L), 2],
                                           range=[(0, L), (-0.5, 1.5)],
                                           epochs=train_ep, fs=fs_pos)
        else:
            pos_only = nap.Tsd(t=feat_tr.t, d=feat_tr.values[:, 0], time_support=train_ep)
            tc = nap.compute_tuning_curves(group, pos_only, bins=n_bins(L),
                                           range=[(0, L)], epochs=train_ep, fs=fs_pos)
        dec, P = nap.decode_bayes(tc, group, test_ep, bin_size=bin_size)
        tt = np.asarray(dec.t)
        kf = full_bin_mask(tt, test_ep, bin_size)
        tt, decv = tt[kf], np.asarray(dec.values)[kf]
        feat_te = joint_feature(test_ep)
        acc["t"].append(tt)
        acc["true_pos"].append(np.interp(tt, np.asarray(feat_te.t), feat_te.values[:, 0]))
        acc["true_dir"].append(np.interp(tt, np.asarray(feat_te.t), feat_te.values[:, 1]))
        if directional:
            acc["dec_pos"].append(decv[:, 0])
            acc["dec_dir"].append(decv[:, 1])
        else:
            acc["dec_pos"].append(decv.ravel())
            acc["dec_dir"].append(np.full(len(tt), np.nan))
    o = {k: np.concatenate(v) for k, v in acc.items()}
    order = np.argsort(o["t"])
    o = {k: v[order] for k, v in o.items()}
    o["err"] = np.abs(o["dec_pos"] - o["true_pos"])
    return o


res_dir = run_cv(BIN_SIZE, directional=True)
res_nodir = run_cv(BIN_SIZE, directional=False)
chance = np.abs(rng.permutation(res_dir["dec_pos"]) - res_dir["true_pos"])
dir_ok = np.round(res_dir["dec_dir"]) == np.round(res_dir["true_dir"])
print(f"median error, position x direction maps: {100 * np.nanmedian(res_dir['err']):.1f} cm")
print(f"median error, position-only maps      : {100 * np.nanmedian(res_nodir['err']):.1f} cm")
print(f"median error, chance                  : {100 * np.nanmedian(chance):.1f} cm")
print(f"running direction correct             : {100 * np.nanmean(dir_ok):.1f}% of bins")

by_bin = {}
for b in BIN_SIZES:
    by_bin[b] = np.nanmedian(run_cv(b, True)["err"])
    print(f"  bin {1000 * b:4.0f} ms -> median error {100 * by_bin[b]:5.1f} cm")

by_n = {}
allpc = list(pc_ids)
for n in [2, 5, 10, 20, 40, len(allpc)]:
    errs = [np.nanmedian(run_cv(BIN_SIZE, True,
                                group=pc[list(rng.choice(allpc, min(n, len(allpc)),
                                                         replace=False))])["err"])
            for _ in range(3)]
    by_n[n] = (float(np.mean(errs)), float(np.std(errs)))
    print(f"  {n:3d} cells -> median error {100 * by_n[n][0]:5.1f} cm")

# %%
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1.3, 1.1], hspace=0.5, wspace=0.3)

train_ep, test_ep = folds[0]
feat_tr = joint_feature(train_ep)
tc_dec = nap.compute_tuning_curves(pc, feat_tr, bins=[n_bins(L), 2],
                                   range=[(0, L), (-0.5, 1.5)], epochs=train_ep, fs=fs_pos)
dec, P = nap.decode_bayes(tc_dec, pc, test_ep, bin_size=BIN_SIZE)
t = np.asarray(dec.t)
kf = full_bin_mask(t, test_ep, BIN_SIZE)
Pm = np.asarray(P).sum(axis=2)[kf]   # marginalize over direction
dpos = np.asarray(dec.values)[kf, 0]
t = t[kf]

sgs = gs[0, :].subgridspec(1, 8, wspace=0.12)
for k in range(8):
    e0, e1 = float(test_ep.start[4 + k]), float(test_ep.end[4 + k])
    ax = fig.add_subplot(sgs[0, k])
    m = (t >= e0) & (t <= e1)
    et = np.concatenate([t[m] - BIN_SIZE / 2, [t[m][-1] + BIN_SIZE / 2]]) - e0
    ax.pcolormesh(et, np.linspace(0, L, n_bins(L) + 1), Pm[m].T, cmap="Greys",
                  vmin=0, vmax=np.percentile(Pm[m], 99))
    pt = pos.restrict(nap.IntervalSet(start=e0, end=e1))
    ax.plot(np.asarray(pt.t) - e0, pt.values, color="#1f77b4", lw=2.2,
            label="true" if k == 0 else None)
    ax.plot(t[m] - e0, dpos[m], ".", color="crimson", ms=7,
            label="decoded" if k == 0 else None)
    ax.set_ylim(0, L)
    ax.set_xlim(0, e1 - e0)
    ax.set_xlabel("t (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    if k:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("track position (m)")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_title(f"run {k + 1}", fontsize=9)
fig.text(0.5, 0.99, f"Bayesian decoding from {len(pc)} CA1 place cells, "
         f"{1000 * BIN_SIZE:.0f} ms bins, eight consecutive held-out traversals "
         "(grey = posterior)", ha="center", fontsize=12)

ax = fig.add_subplot(gs[1, 0])
edges = np.linspace(0, L, 33)
H, _, _ = np.histogram2d(res_dir["true_pos"], res_dir["dec_pos"], [edges, edges])
H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
im = ax.imshow(H.T, origin="lower", extent=[0, L, 0, L], cmap="viridis", aspect="equal")
ax.plot([0, L], [0, L], "w--", lw=1)
ax.set_xlabel("true position (m)")
ax.set_ylabel("decoded position (m)")
ax.set_title("Confusion matrix\n(rows normalized)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label="P(decoded | true)")

ax = fig.add_subplot(gs[1, 1])
e = np.linspace(0, L, 41) * 100
ax.hist(100 * chance, e, color="0.8", density=True,
        label=f"chance ({100 * np.nanmedian(chance):.0f} cm)")
ax.hist(100 * res_nodir["err"], e, histtype="step", color="k", lw=1.8, density=True,
        label=f"position-only maps ({100 * np.nanmedian(res_nodir['err']):.1f} cm)")
ax.hist(100 * res_dir["err"], e, histtype="step", color="crimson", lw=1.8, density=True,
        label=f"position x direction maps ({100 * np.nanmedian(res_dir['err']):.1f} cm)")
ax.set_xlabel("absolute decoding error (cm)")
ax.set_ylabel("probability density")
ax.set_title("Decoding error, cross-validated", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
acc = 100 * np.nanmean(dir_ok)
ax.bar([0, 1], [acc, 50], color=["crimson", "0.8"], width=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["decoded", "chance"])
ax.set_ylabel("running direction correct (%)")
ax.set_ylim(0, 105)
ax.text(0, acc + 2, f"{acc:.1f}%", ha="center")
ax.set_title("Running direction is decodable\nfrom the same population", fontsize=11)

ax = fig.add_subplot(gs[2, 0])
ax.plot([1000 * b for b in BIN_SIZES], [100 * by_bin[b] for b in BIN_SIZES], "o-",
        color="crimson")
ax.set_xscale("log")
ax.set_xlabel("decoding time bin (ms)")
ax.set_ylabel("median error (cm)")
ax.set_title("Error vs time bin", fontsize=11)

ax = fig.add_subplot(gs[2, 1])
ns = sorted(by_n)
ax.errorbar(ns, [100 * by_n[n][0] for n in ns], yerr=[100 * by_n[n][1] for n in ns],
            fmt="o-", color="#1f77b4", capsize=3)
ax.axhline(100 * np.nanmedian(chance), color="0.6", ls="--", label="chance")
ax.set_xscale("log")
ax.set_xlabel("number of place cells in the decoder")
ax.set_ylabel("median error (cm)")
ax.set_title("Error vs population size\n(mean +/- SD over 3 random subsets)", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, 2])
bc = 0.5 * (edges[:-1] + edges[1:])
idx = np.digitize(res_dir["true_pos"], edges) - 1
med = [100 * np.nanmedian(res_dir["err"][idx == k]) if np.sum(idx == k) > 5 else np.nan
       for k in range(len(bc))]
ax.plot(bc, med, "o-", color="crimson", ms=4)
ax.set_xlabel("true position (m)")
ax.set_ylabel("median error (cm)")
ax.set_ylim(0, None)
ax.set_title("Error vs position on the track", fontsize=11)

fig.text(0.5, 1.03, f"{SESSION}: position is recoverable from the CA1 place-cell "
         "population", ha="center", fontsize=14)
fig.savefig("fig05_decoding.png", dpi=150)
plt.show()

# %% [markdown]
# ## Figure 6: theta phase precession
#
# The place code also has a temporal component. While the rat runs, the CA1 LFP is
# dominated by a 6-10 Hz theta rhythm, and a place cell fires at a progressively
# earlier theta phase as the animal moves through its field. Extracting this needs
# the LFP: one channel per shank is streamed for the maze epoch, the channel with
# the strongest theta relative to delta is kept, and theta phase is taken from the
# Hilbert transform of the band-passed signal.
#
# For each place field, spikes within one field width either side of the peak are
# collected, their position is expressed as a fraction of that window (0 at entry,
# 1 at exit, in the direction of travel), and phase is regressed on position with
# the circular-linear method of Kempter et al. (2012). A negative slope means
# precession.

# %%
THETA_BAND = (6.0, 10.0)
es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs_lfp = float(es.rate)
i0 = int((float(maze.start[0]) - es.starting_time) * fs_lfp)
i1 = int((float(maze.end[0]) - es.starting_time) * fs_lfp)
elec = nwbfile.electrodes.to_dataframe()
cands = [int(g.index[len(g) // 2]) for _, g in elec.groupby("group_name")]
t_lfp = es.starting_time + np.arange(i0, i1) / fs_lfp
print(f"LFP {es.data.shape} at {fs_lfp:.0f} Hz; screening {len(cands)} channels")

best, best_ratio, best_x = None, -np.inf, None
for ch in tqdm(cands, desc="LFP channels"):
    x = np.asarray(es.data[i0:i1, ch], dtype=float) * es.conversion
    psd = nap.compute_mean_power_spectral_density(nap.Tsd(t=t_lfp, d=x), 1.0,
                                                 fs=fs_lfp, ep=eps["run"])
    f = np.asarray(psd.index)
    p = np.abs(np.asarray(psd.values).ravel()) ** 2
    ratio = (p[(f >= THETA_BAND[0]) & (f <= THETA_BAND[1])].mean()
             / p[(f >= 1) & (f <= 4)].mean())
    if ratio > best_ratio:
        best, best_ratio, best_x = ch, ratio, x
print(f"using channel {best} ({elec.group_name[best]}), theta/delta = {best_ratio:.2f}")

lfp = nap.Tsd(t=t_lfp, d=best_x)
theta_tsd = nap.apply_bandpass_filter(lfp, THETA_BAND, fs=fs_lfp)
analytic = hilbert(np.asarray(theta_tsd.values))
phase = nap.Tsd(t=t_lfp, d=np.mod(np.angle(analytic), 2 * np.pi))


def circ_lin_corr(x, ph, n_slopes=1601):
    """Circular-linear regression and correlation (Kempter et al. 2012).

    Returns (rho, slope in rad per unit x, phase offset).
    """
    if len(x) < 20:
        return np.nan, np.nan, np.nan
    slopes = np.linspace(-8 * np.pi, 8 * np.pi, n_slopes)
    # |mean(exp(i(ph - a x)))| for every candidate slope, evaluated as a matrix
    # product of the unit phasors against exp(-i a x).
    z = np.exp(1j * ph)
    R = np.abs(np.exp(-1j * np.outer(slopes, x)) @ z) / len(x)
    slope = float(slopes[int(np.argmax(R))])
    phi0 = float(np.angle(np.exp(1j * (ph - slope * x)).mean()))
    theta = np.mod(slope * x, 2 * np.pi)
    sp = np.sin(ph - np.angle(np.exp(1j * ph).mean()))
    st = np.sin(theta - np.angle(np.exp(1j * theta).mean()))
    den = np.sqrt(np.sum(sp ** 2) * np.sum(st ** 2))
    return (float(np.sum(sp * st) / den) if den > 0 else np.nan), slope, phi0


records, pooled_x, pooled_ph = [], [], []
for d in DIRS:
    m = mets[d]
    for uid in pyr.index:
        if not m.is_place_cell[uid]:
            continue
        pk, wd = m.peak_pos[uid], m.width[uid]
        lo, hi = pk - wd, pk + wd
        st = np.asarray(pyr[uid].restrict(eps[d]).t)
        sp = np.interp(st, pos.t, pos.values)
        ins = (sp >= lo) & (sp <= hi)
        st, sp = st[ins], sp[ins]
        if len(st) < 40:
            continue
        ph = np.mod(np.interp(st, phase.t, np.unwrap(phase.values)), 2 * np.pi)
        xn = (sp - lo) / (hi - lo)
        if d == "left":
            xn = 1 - xn      # express position in the direction of travel
        rho, slope, phi0 = circ_lin_corr(xn, ph)
        records.append({"unit": uid, "direction": d, "n_spikes": len(st), "rho": rho,
                        "slope_cycles": slope / (2 * np.pi), "peak_pos": pk, "width": wd})
        pooled_x.append(xn)
        pooled_ph.append(ph)

prec = pd.DataFrame(records)
pooled_x, pooled_ph = np.concatenate(pooled_x), np.concatenate(pooled_ph)
rho_all, slope_all, phi0_all = circ_lin_corr(pooled_x, pooled_ph)
nperm = 200
null_rho = np.array([circ_lin_corr(rng.permutation(pooled_x), pooled_ph, n_slopes=401)[0]
                     for _ in range(nperm)])
p_perm = (np.sum(null_rho <= rho_all) + 1) / (nperm + 1)
print(f"{len(prec)} fields, {len(pooled_x)} in-field spikes")
print(f"pooled rho = {rho_all:.3f}, slope = {slope_all / (2 * np.pi):.2f} theta cycles "
      f"per field, permutation p = {p_perm:.4f}")
print(f"per-field slope: median {prec.slope_cycles.median():.2f} cycles, "
      f"{100 * (prec.slope_cycles < 0).mean():.0f}% negative")

# %%
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.3, 1.1], hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, :])
wz = nap.IntervalSet(start=float(eps["right"].start[6]), end=float(eps["right"].end[6]))
lr, tr = lfp.restrict(wz), theta_tsd.restrict(wz)
ax.plot(lr.t, 1e3 * lr.values, color="0.6", lw=0.8, label="raw LFP")
ax.plot(tr.t, 1e3 * tr.values, color="k", lw=1.5,
        label=f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz")
ax2 = ax.twinx()
pr = phase.restrict(wz)
ax2.plot(pr.t, pr.values, color="#e377c2", lw=0.7, alpha=0.8)
ax2.set_ylabel("theta phase (rad)", color="#e377c2")
ax2.tick_params(axis="y", colors="#e377c2")
ax.set_xlabel("time (s)")
ax.set_ylabel("LFP (mV)")
ax.set_title(f"{SESSION}: CA1 LFP on channel {best} during one traversal", fontsize=11)
ax.legend(fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[1, 0])
still = maze.set_diff(eps["run"])
for ep, lab, c in [(eps["run"], "running", "crimson"), (still, "immobile", "0.5")]:
    psd = nap.compute_mean_power_spectral_density(lfp, 1.0, fs=fs_lfp, ep=ep)
    f = np.asarray(psd.index)
    p = np.abs(np.asarray(psd.values).ravel()) ** 2
    m = (f > 0.5) & (f < 60)
    ax.semilogy(f[m], p[m], color=c, lw=1.2, label=lab)
ax.axvspan(*THETA_BAND, color="gold", alpha=0.25, lw=0)
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("power (a.u.)")
ax.set_title("LFP power spectrum", fontsize=11)
ax.legend(fontsize=8)

best3 = prec.reindex(prec.rho.abs().sort_values(ascending=False).index).head(3)
for k, (_, row) in enumerate(best3.iterrows()):
    ax = fig.add_subplot(gs[1, 1 + k] if k < 2 else gs[2, 0])
    uid, d = int(row.unit), row.direction
    lo, hi = row.peak_pos - row.width, row.peak_pos + row.width
    st = np.asarray(pyr[uid].restrict(eps[d]).t)
    sp = np.interp(st, pos.t, pos.values)
    ins = (sp >= lo) & (sp <= hi)
    st, sp = st[ins], sp[ins]
    ph = np.mod(np.interp(st, phase.t, np.unwrap(phase.values)), 2 * np.pi)
    xn = (sp - lo) / (hi - lo)
    if d == "left":
        xn = 1 - xn
    for rep in (0, 1):
        ax.plot(xn, ph + rep * 2 * np.pi, ".", color=COL[d], ms=3, alpha=0.7)
    _, sl, p0 = circ_lin_corr(xn, ph)
    xx = np.linspace(0, 1, 200)
    for rep in (0, 1, 2):
        yy = np.mod(sl * xx + p0, 2 * np.pi) + rep * 2 * np.pi
        yy[np.abs(np.diff(yy, prepend=yy[0])) > np.pi] = np.nan
        ax.plot(xx, yy, "k-", lw=1.5)
    ax.set_xlabel("position within field (entry -> exit)")
    ax.set_ylabel("theta phase (rad)")
    ax.set_ylim(0, 4 * np.pi)
    ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$", "$3\\pi$", "$4\\pi$"])
    ax.set_title(f"unit {uid}, {d}ward\nrho = {row.rho:.2f}, "
                 f"{row.slope_cycles:.2f} cycles/field", fontsize=10)

ax = fig.add_subplot(gs[2, 1])
H, xe, ye = np.histogram2d(pooled_x, pooled_ph,
                           [np.linspace(0, 1, 21), np.linspace(0, 2 * np.pi, 25)])
H = H / H.sum(axis=1, keepdims=True)
im = ax.imshow(np.vstack([H.T, H.T]), aspect="auto", origin="lower",
               extent=[0, 1, 0, 4 * np.pi], cmap="magma")
xx = np.linspace(0, 1, 100)
for rep in (0, 1, 2):
    yy = np.mod(slope_all * xx + phi0_all, 2 * np.pi) + rep * 2 * np.pi
    yy[np.abs(np.diff(yy, prepend=yy[0])) > np.pi] = np.nan
    ax.plot(xx, yy, "c-", lw=2)
ax.set_ylim(0, 4 * np.pi)
ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_yticklabels(["0", "$\\pi$", "$2\\pi$", "$3\\pi$", "$4\\pi$"])
ax.set_xlabel("position within field (entry -> exit)")
ax.set_ylabel("theta phase (rad)")
ax.set_title(f"All {len(prec)} fields pooled ({len(pooled_x)} spikes)\n"
             f"rho = {rho_all:.2f}, slope = {slope_all / (2 * np.pi):.2f} cycles, "
             f"p = {p_perm:.3f}", fontsize=10)
plt.colorbar(im, ax=ax, label="P(phase | position)", fraction=0.046)

ax = fig.add_subplot(gs[2, 2])
ax.hist(prec.slope_cycles, np.linspace(-3, 3, 37), color="#4c72b0", edgecolor="w")
ax.axvline(0, color="k", lw=1)
ax.axvline(prec.slope_cycles.median(), color="crimson", ls="--", lw=1.5)
ax.set_xlabel("phase-position slope (theta cycles per field)")
ax.set_ylabel("number of fields")
ax.set_title(f"Slopes are negative in {100 * (prec.slope_cycles < 0).mean():.0f}% "
             f"of fields\n(median {prec.slope_cycles.median():.2f} cycles)", fontsize=10)

fig.suptitle(f"{SESSION}: theta phase precession within CA1 place fields", fontsize=13)
fig.savefig("fig08_theta_precession.png", dpi=150)
plt.show()

# %% [markdown]
# ## Figures 7 and 8: all eight sessions
#
# The pipeline is now applied unchanged to every session in the dandiset. Three of
# the eight use a circular maze on which the rat runs laps in one direction; the
# linearized coordinate is treated as a straight line there, which is adequate for
# rate maps but means a field straddling the start/end of the lap would be split.
# Directions with fewer than ten traversals are not analysed, which is what leaves
# the circular sessions with a single direction each.

# %%
def analyze_session(name, url, n_shuffle=N_SHUFFLE, seed=0, verbose=True):
    nwb_, nwbfile_, io_ = load_session(url)
    maze_ = maze_epoch(nwbfile_)
    pos_, L_ = get_position(nwb_, nwbfile_)
    vel_, eps_ = running_epochs(pos_)
    units_ = nwb_["units"]
    pyr_ = select_pyramidal(units_, maze_)
    out = {
        "session": name, "subject": nwbfile_.subject.subject_id,
        "maze_type": maze_name(nwbfile_), "track_length": L_,
        "maze_duration": float(maze_.tot_length()),
        "n_units": len(units_), "n_pyramidal": len(pyr_),
        "n_traversals": {d: len(eps_[d]) for d in DIRS},
        "run_duration": {d: float(eps_[d].tot_length()) for d in DIRS + ("run",)},
        "unit_index": np.asarray(pyr_.index), "per_direction": {},
    }
    if verbose:
        print(f"{name} ({out['subject']}, {out['maze_type']}, {L_:.2f} m): "
              f"{len(units_)} units -> {len(pyr_)} pyramidal; "
              f"{len(eps_['right'])} rightward / {len(eps_['left'])} leftward traversals")
    r = np.random.default_rng(seed)
    for d in DIRS:
        if len(eps_[d]) < MIN_TRAVERSALS:
            if verbose:
                print(f"  {d}: only {len(eps_[d])} traversals, skipped")
            continue
        o = direction_metrics(pyr_, pos_, eps_[d], L_, n_shuffle=n_shuffle, rng=r)
        out["per_direction"][d] = {"tc_smooth": o["tc_smooth"].values,
                                   "bins": o["bins"], "occupancy": o["occ"],
                                   "metrics": o["metrics"]}
        if verbose:
            m = o["metrics"]
            print(f"  {d}: {int(m.is_place_cell.sum())}/{len(pyr_)} place cells, "
                  f"median SI {np.median(m.si[m.is_place_cell]):.2f} bits/spike, "
                  f"median width {100 * np.median(m.width[m.is_place_cell]):.0f} cm")
    analysed = list(out["per_direction"])
    mask = np.zeros(len(pyr_), bool)
    for d in analysed:
        mask |= out["per_direction"][d]["metrics"].is_place_cell.values
    out["is_place_cell_either"] = mask
    out["directions_analysed"] = analysed
    if len(analysed) == 2:
        a = out["per_direction"]["right"]["tc_smooth"]
        b = out["per_direction"]["left"]["tc_smooth"]
        out["dir_corr"] = np.array([map_corr(a[:, j], b[:, j]) for j in range(a.shape[1])])
    else:
        out["dir_corr"] = np.full(len(pyr_), np.nan)
    io_.close()
    return out


all_results = {}
for name, url in tqdm(list(SESSION_URLS.items()), desc="sessions"):
    all_results[name] = analyze_session(name, url)

# %%
rows, pooled = [], []
for name, r in all_results.items():
    n_pc = int(r["is_place_cell_either"].sum())
    rows.append({"session": name, "subject": r["subject"], "maze": r["maze_type"],
                 "track_m": r["track_length"], "n_units": r["n_units"],
                 "n_pyr": r["n_pyramidal"], "n_pc": n_pc,
                 "frac_pc": n_pc / max(r["n_pyramidal"], 1),
                 "n_trav": sum(r["n_traversals"][d] for d in DIRS),
                 "run_s": r["run_duration"]["run"],
                 "dirs": ",".join(r["directions_analysed"])})
    for d in r["directions_analysed"]:
        m = r["per_direction"][d]["metrics"].copy()
        m["session"], m["subject"] = name, r["subject"]
        m["maze"], m["direction"] = r["maze_type"], d
        m["track_m"] = r["track_length"]
        m["dir_corr"] = r["dir_corr"]
        pooled.append(m)

summary = pd.DataFrame(rows)
pooled = pd.concat(pooled, ignore_index=True)
summary.to_csv("session_summary.csv", index=False)
pooled.to_csv("pooled_units.csv", index=False)
print(summary.to_string(index=False))
pcp = pooled[pooled.is_place_cell]
print(f"\npooled: {len(pooled)} unit-direction pairs, {len(pcp)} significant fields")
print(f"median SI          {pcp.si.median():.2f} bits/spike")
print(f"median field width {100 * pcp.width.median():.0f} cm")
print(f"median peak rate   {pcp.peak_rate.median():.1f} Hz")
print(f"median sparsity    {pcp.sparsity.median():.2f}")
print(f"median reliability {pcp.reliability.median():.2f}")
print(f"place-cell fraction per session: {summary.frac_pc.min():.2f}-"
      f"{summary.frac_pc.max():.2f} (mean {summary.frac_pc.mean():.2f})")

# %%
SUBJ_COL = dict(zip(sorted(summary.subject.unique()),
                    ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]))
fig, axs = plt.subplots(2, 4, figsize=(19, 9))
axs = axs.ravel()

ax = axs[0]
o = np.argsort(summary.frac_pc.values)
y = np.arange(len(summary))
ax.barh(y, 100 * summary.frac_pc.values[o],
        color=[SUBJ_COL[s] for s in summary.subject.values[o]])
ax.set_yticks(y)
ax.set_yticklabels([f"{s}\n({m}, {n} cells)" for s, m, n in
                    zip(summary.session.values[o], summary.maze.values[o],
                        summary.n_pyr.values[o])], fontsize=7)
ax.axvline(100 * summary.frac_pc.mean(), color="k", ls="--", lw=1)
ax.set_xlabel("pyramidal cells with a place field (%)")
ax.set_title("(a) Place cells are found in every session", fontsize=11)

ax = axs[1]
for sname, g in pcp.groupby("session"):
    ax.hist(g.si, np.linspace(0, 4, 33), histtype="step", lw=1.2, density=True,
            color=SUBJ_COL[g.subject.iloc[0]], alpha=0.8)
ax.hist(pcp.si, np.linspace(0, 4, 33), histtype="step", lw=2.5, color="k",
        density=True, label=f"pooled (n={len(pcp)})")
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("probability density")
ax.set_title(f"(b) Spatial information, median {pcp.si.median():.2f} bits/spike",
             fontsize=11)
ax.legend(fontsize=8)

ax = axs[2]
groups = list(pcp.groupby("session"))
bp = ax.boxplot([100 * g.width.dropna().values for _, g in groups],
                labels=[s[:3] + s.split("_")[1][:4] for s, _ in groups],
                showfliers=False, patch_artist=True)
for patch, (_, g) in zip(bp["boxes"], groups):
    patch.set_facecolor(SUBJ_COL[g.subject.iloc[0]])
ax.tick_params(axis="x", rotation=60, labelsize=7)
ax.set_ylabel("place-field width (cm)")
ax.set_title(f"(c) Field width, pooled median {100 * pcp.width.median():.0f} cm",
             fontsize=11)

ax = axs[3]
ax.scatter(pcp.peak_rate, 100 * pcp.width, s=10,
           c=[SUBJ_COL[s] for s in pcp.subject], alpha=0.6, edgecolors="none")
ax.set_xscale("log")
ax.set_xlabel("in-field peak rate (Hz)")
ax.set_ylabel("field width (cm)")
ax.set_title("(d) Peak rate vs field width", fontsize=11)
ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=c, label=s)
                   for s, c in SUBJ_COL.items()],
          fontsize=8, title="rat", title_fontsize=8)

ax = axs[4]
ax.hist(pcp.sparsity.dropna(), np.linspace(0, 1, 33), color="#4c72b0", edgecolor="w")
ax.axvline(pcp.sparsity.median(), color="k", ls="--")
ax.set_xlabel(r"sparsity $\langle\lambda\rangle^2/\langle\lambda^2\rangle$")
ax.set_ylabel("count")
ax.set_title(f"(e) Sparsity, median {pcp.sparsity.median():.2f}", fontsize=11)

ax = axs[5]
ax.hist(pcp.reliability.dropna(), np.linspace(-0.2, 1, 33), color="#55a868", edgecolor="w")
ax.axvline(pcp.reliability.median(), color="k", ls="--")
ax.set_xlabel("odd vs even traversal map correlation\n(>=0.4 by selection)")
ax.set_ylabel("count")
ax.set_title(f"(f) Reliability, median r = {pcp.reliability.median():.2f}", fontsize=11)

ax = axs[6]
for sname, g in pcp.groupby("session"):
    ax.hist(g.peak_pos / g.track_m.iloc[0], np.linspace(0, 1, 17), histtype="step",
            lw=1.2, density=True, color=SUBJ_COL[g.subject.iloc[0]], alpha=0.8)
ax.hist(pcp.peak_pos / pcp.track_m, np.linspace(0, 1, 17), histtype="step", lw=2.5,
        color="k", density=True)
ax.set_xlabel("field peak, fraction of track length")
ax.set_ylabel("probability density")
ax.set_title("(g) Fields cover the track in every session", fontsize=11)

ax = axs[7]
bi = [s for s in pooled.session.unique()
      if len(all_results[s]["directions_analysed"]) == 2]
sub = pooled[pooled.session.isin(bi) & pooled.is_place_cell_norel]
d1 = sub.dir_corr.dropna()
d2 = sub.reliability.dropna()
e = np.linspace(-1, 1, 33)
ax.hist(d1, e, color="0.75", density=True, label=f"opposite directions ({d1.median():.2f})")
ax.hist(d2, e, histtype="step", color="k", lw=1.8, density=True,
        label=f"same direction, odd vs even ({d2.median():.2f})")
u_md = stats.mannwhitneyu(d2, d1, alternative="greater")
ax.set_xlabel("rate-map correlation")
ax.set_ylabel("probability density")
ax.set_title(f"(h) Direction specificity, {len(bi)} bidirectional sessions\n"
             f"(n = {len(d1)} fields, p = {u_md.pvalue:.1e})", fontsize=11)
ax.legend(fontsize=7.5, loc="upper left")
print(f"directionality (no reliability selection, n={len(d1)}): "
      f"cross-direction r = {d1.median():.2f}, within-direction r = {d2.median():.2f}, "
      f"Mann-Whitney p = {u_md.pvalue:.2e}")

fig.suptitle("DANDI:000044, all 8 sessions / 4 rats: place fields replicate across "
             "animals and mazes", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig06_multisession.png", dpi=150)
plt.show()

# %%
fig, axs = plt.subplots(2, 4, figsize=(19, 8.5))
for ax, (name, r) in zip(axs.ravel(), all_results.items()):
    # show whichever direction the animal ran most
    d = max(r["directions_analysed"], key=lambda k: r["n_traversals"][k])
    pdd = r["per_direction"][d]
    m = pdd["metrics"]
    s_ = np.where(m.is_place_cell.values)[0]
    o = s_[np.argsort(m.peak_pos.values[s_])]
    M = pdd["tc_smooth"][:, o].T
    mx = M.max(axis=1, keepdims=True)
    mx[mx == 0] = 1
    ax.imshow(M / mx, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
              extent=[0, r["track_length"], 0, len(o)])
    ax.set_title(f"{name}\n{r['maze_type']}, {d}ward, {len(o)} place cells", fontsize=9)
    ax.set_xlabel("position (m)", fontsize=8)
    ax.set_ylabel("place cell", fontsize=8)
    ax.tick_params(labelsize=8)
fig.suptitle("Sorted place-field maps, one direction per session", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig07_multisession_maps.png", dpi=150)
plt.show()

# %% [markdown]
# ## Figure 9: a Poisson GLM of the place code (NeMoS)
#
# The occupancy-normalized rate map is a nonparametric estimate. A Poisson GLM with
# a B-spline basis over position gives a smooth, regularized version of the same
# tuning, and, more usefully, provides a framework for asking how much of a place
# cell's spiking position accounts for relative to running direction, running
# speed, and the cell's own recent spike history. Models are compared by
# cross-validated McFadden pseudo-$R^2$ on held-out traversals, using the same
# within-direction odd/even split as the decoder.
#
# Direction-specific fields are represented by gating the same position basis with
# the direction indicator, so "position x direction" has one spline expansion per
# direction. Note that the history term convolves across traversal boundaries in
# the concatenated running epochs, which slightly overstates its contribution.

# %%
GLM_BIN = 0.025
N_POS_BASIS, N_SPEED_BASIS = 12, 5
HIST_WINDOW_S = 0.2
run = eps["run"]

counts = pc.count(GLM_BIN, ep=run)
pos_b = pos.interpolate(counts, ep=run)
vel_b = vel.interpolate(counts, ep=run)
speed_b = np.abs(np.asarray(vel_b.values))
dir_b = (np.asarray(vel_b.values) > 0).astype(float)
print(f"{counts.shape[0]} time bins of {1000 * GLM_BIN:.0f} ms over "
      f"{run.tot_length():.0f} s of running, {len(pc)} place cells")

pos_basis = nmo.basis.BSplineEval(n_basis_funcs=N_POS_BASIS, label="position")
pos_basis.set_input_shape(1)
speed_basis = nmo.basis.BSplineEval(n_basis_funcs=N_SPEED_BASIS, label="speed")
speed_basis.set_input_shape(1)
hist_basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=5, window_size=int(HIST_WINDOW_S / GLM_BIN), label="history")

Xpos = pos_basis.compute_features(np.clip(np.asarray(pos_b.values) / L, 0, 1))
Xspeed = speed_basis.compute_features(np.clip(speed_b / 1.5, 0, 1))
Xposdir = np.hstack([Xpos * dir_b[:, None], Xpos * (1 - dir_b)[:, None]])

MODELS = {
    "position": lambda uid: Xpos,
    "position x direction": lambda uid: Xposdir,
    "position + speed": lambda uid: np.hstack([Xpos, Xspeed]),
    "position x direction\n+ speed": lambda uid: np.hstack([Xposdir, Xspeed]),
    "position x direction\n+ speed + history": lambda uid: np.hstack(
        [Xposdir, Xspeed, np.asarray(hist_basis.compute_features(counts.loc[uid]))]),
}

tvec = np.asarray(counts.t)


def in_ep(ep):
    m = np.zeros(len(tvec), bool)
    for a, b in zip(np.asarray(ep.start), np.asarray(ep.end)):
        m |= (tvec >= a) & (tvec <= b)
    return m


masks = [(in_ep(fold_a), in_ep(fold_b)), (in_ep(fold_b), in_ep(fold_a))]

scores = {k: [] for k in MODELS}
pred_curves = {}
for uid in tqdm(pc_ids, desc="GLM units"):
    y = np.asarray(counts.loc[uid].values, dtype=float)
    for mname, build in MODELS.items():
        X = build(uid)
        ok = np.isfinite(X).all(axis=1)
        vals = []
        for tr_m, te_m in masks:
            model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                                regularizer_strength=1e-4)
            model.fit(X[tr_m & ok], y[tr_m & ok])
            vals.append(float(model.score(X[te_m & ok], y[te_m & ok],
                                          score_type="pseudo-r2-McFadden")))
        scores[mname].append(float(np.mean(vals)))
    ok = np.isfinite(Xposdir).all(axis=1)
    model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge",
                        regularizer_strength=1e-4).fit(Xposdir[ok], y[ok])
    grid = np.linspace(0, 1, 200)
    Bg = pos_basis.compute_features(grid)
    curves = {}
    for k, d in enumerate(DIRS):
        Xg = np.zeros((len(grid), Xposdir.shape[1]))
        Xg[:, k * N_POS_BASIS:(k + 1) * N_POS_BASIS] = Bg
        curves[d] = np.asarray(model.predict(Xg)) / GLM_BIN
    pred_curves[uid] = curves

sc = pd.DataFrame(scores, index=pc_ids)
sc.to_csv("glm_scores.csv")
print("\ncross-validated McFadden pseudo-R^2:")
for k in MODELS:
    print(f"  {k.replace(chr(10), ' '):45s} mean {sc[k].mean():.4f}  "
          f"median {sc[k].median():.4f}")

# %%
fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 1.2], hspace=0.55, wspace=0.35)

best8 = sc["position x direction"].sort_values(ascending=False).index[:8]
for j, uid in enumerate(best8):
    ax = fig.add_subplot(gs[j // 4, j % 4])
    for d in DIRS:
        ax.plot(centers, tcs[d][uid].values, color=COL[d], lw=1.0, alpha=0.5)
        ax.plot(np.linspace(0, L, 200), pred_curves[uid][d], color=COL[d], lw=2.0)
    ax.set_xlim(0, L)
    ax.set_ylim(0, None)
    ax.set_title(f"unit {uid}  pseudo-$R^2$ = {sc['position x direction'][uid]:.3f}",
                 fontsize=9)
    if j % 4 == 0:
        ax.set_ylabel("firing rate (Hz)")
    if j >= 4:
        ax.set_xlabel("position (m)")
fig.text(0.5, 0.935, "Poisson GLM with a 12-element B-spline basis over position "
         "(thick) against the occupancy-normalized rate map (thin)",
         ha="center", fontsize=11)

ax = fig.add_subplot(gs[2, 0:2])
names = list(MODELS)
bp = ax.boxplot([sc[k].values for k in names],
                labels=[n.replace("\n", " ") for n in names], showfliers=False,
                patch_artist=True, widths=0.6)
for patch in bp["boxes"]:
    patch.set_facecolor("#4c72b0")
for k, kk in enumerate(names):
    v = sc[kk].values
    ax.plot(np.full(len(v), k + 1) + rng.normal(0, 0.06, len(v)), v, ".", color="k",
            ms=3, alpha=0.4)
ax.set_ylabel("cross-validated pseudo-$R^2$")
ax.tick_params(axis="x", rotation=25, labelsize=8)
for lab in ax.get_xticklabels():
    lab.set_ha("right")
ax.set_title(f"Model comparison across {len(pc)} place cells", fontsize=11)

ax = fig.add_subplot(gs[2, 2])
ax.scatter(sc["position"], sc["position x direction"], s=14, c="k", alpha=0.6,
           edgecolors="none")
lim = [0, max(sc["position x direction"].max(), sc["position"].max()) * 1.05]
ax.plot(lim, lim, "r--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("pseudo-$R^2$, position only")
ax.set_ylabel("pseudo-$R^2$, position x direction")
ax.set_title(f"Direction helps in "
             f"{100 * np.mean(sc['position x direction'] > sc['position']):.0f}% of cells",
             fontsize=11)

ax = fig.add_subplot(gs[2, 3])
gridx, bvals = pos_basis.evaluate_on_grid(200)
ax.plot(gridx * L, bvals, lw=1.2)
ax.set_xlabel("position (m)")
ax.set_ylabel("basis value")
ax.set_title(f"{N_POS_BASIS} B-spline basis functions", fontsize=11)

fig.savefig("fig09_glm.png", dpi=150)
plt.show()

# %% [markdown]
# ## Summary
#
# Every element of the classical place-cell phenomenology is recovered from the
# archived data without any tuning of the analysis to the dataset:
#
# - Roughly two thirds to three quarters of the well-isolated CA1 pyramidal cells
#   in each session have a spatially localized, statistically significant firing
#   field, with a median width near a fifth of the track and spatial information
#   near 1 bit per spike, well above the circular-shift null.
# - Fields are highly repeatable within a session (odd/even traversal map
#   correlation around 0.85) and tile the track roughly uniformly.
# - Fields on the linear tracks are strongly direction-specific: the same cell's
#   maps for the two running directions correlate far less than its own split-half
#   maps within a direction.
# - The population code is dense enough that position is recoverable to under
#   10 cm from 250 ms of spiking, against a chance level above 40 cm, and running
#   direction is recoverable in about 95% of time bins. Decoding error falls
#   monotonically with the number of cells included.
# - Spikes within a field arrive at a systematically earlier theta phase as the
#   animal advances through it, the classic phase-precession signature.
# - A Poisson GLM over a spline basis in position reproduces the empirical rate
#   maps, and adding running direction improves held-out fit for the large majority
#   of cells.
#
# The main caveats are that the circular-maze sessions are analysed with a
# linearized coordinate treated as a straight line (a field spanning the lap
# boundary would be split), that the spike-history term in the GLM convolves across
# traversal boundaries, and that pyramidal-cell identity is taken from the
# `cell_type` labels already in the NWB files rather than re-derived from waveform
# features.
