# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hippocampal Place Cells in DANDI:000044
#
# Pyramidal cells in hippocampal area CA1 fire selectively when an animal occupies a
# particular part of its environment. On a linear track, individual cells fire in
# restricted stretches of the track, and the population as a whole tiles the track, so
# that the animal's position can be read back out of the spiking activity alone.
#
# This notebook demonstrates that phenomenon end to end using real recordings from the
# DANDI Archive: [DANDI:000044](https://dandiarchive.org/dandiset/000044), "Diversity in
# neural firing dynamics supports both rigid and learned hippocampal sequences"
# (Grosmark & Buzsáki, *Science* 2016; the CRCNS `hc-11` dataset). Rats ran back and
# forth on a linear track for water reward at both ends while bilateral silicon probes
# recorded from dorsal CA1.
#
# The analysis:
#
# 1. Stream the NWB files from DANDI with LINDI (no full downloads).
# 2. Rebuild a single linear spatial frame with a running-direction label.
# 3. Compute occupancy-normalised firing rate maps per unit and per running direction.
# 4. Score each unit with Skaggs spatial information against a within-epoch circular
#    shuffle null, plus an odd/even-lap stability criterion.
# 5. Show that the population tiles the track, that fields are direction-selective, and
#    that a naive Bayes decoder recovers the animal's position from held-out laps.
# 6. Repeat the whole pipeline over all five linear-track sessions in the dandiset.

# %% [markdown]
# ## Setup

# %%
import os
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from tqdm.auto import tqdm

import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

warnings.filterwarnings("ignore")
nap.nap_config.suppress_conversion_warnings = True

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Analysis constants
# One session used a 2 m rather than a 1.6 m track, so spatial bins are given a fixed
# physical width instead of a fixed count, which keeps field widths comparable.
BIN_WIDTH_M = 0.04     # spatial bin width, 4 cm (40 bins on the 1.6 m track)
SMOOTH_SD_BINS = 1.5   # gaussian smoothing of rate maps, in bins
SPEED_THRESH = 0.10    # m/s; samples below this are excluded from rate maps
N_SHUFFLES = 200       # circular shuffles for the spatial-information null
MIN_SPIKES = 50        # minimum spikes during running for a unit to be considered
MIN_PEAK_RATE = 1.0    # Hz
MIN_STABILITY = 0.5    # odd/even lap rate-map correlation
DECODE_BIN = 0.25      # s; time bin for Bayesian position decoding


def n_bins(beh):
    """Number of spatial bins for this track, at a fixed physical bin width."""
    return int(round(beh["track_len"] / BIN_WIDTH_M))

# %% [markdown]
# ## Streaming the data
#
# Each session in DANDI:000044 is an 8–9 GB NWB file dominated by 128-channel LFP. We
# never need most of it, so the files are opened through LINDI, which turns the remote
# HDF5 into a JSON index and fetches only the chunks that are actually touched. A local
# cache makes repeated runs fast.

# %%
LINDI_BASE = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/{}/nwb.lindi.json"
CACHE_DIR = "/tmp/lindi_cache_000044"

SESSIONS = {
    "Achilles_10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Achilles_11012013": "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a",
    "Cicero_09012014":   "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero_09102014":   "f61dfe09-3db2-464a-b386-2e828b2e7276",
    "Cicero_09172014":   "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby_08022013":   "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Gatsby_08282013":   "f7687af7-3bc9-4d20-8d88-ef293d2a3381",
    "Buddy_06272013":    "82714afb-724f-4e2b-b102-c9c47b5cba73",
}

# Three of the eight sessions used a circular maze. The direction-flip below assumes an
# open-ended linear track, so the analysis is limited to the five linear-track sessions
# (Cicero_09172014 used a 2 m track, the rest 1.6 m).
LINEAR_SESSIONS = ["Achilles_10252013", "Cicero_09012014", "Cicero_09172014",
                   "Gatsby_08022013", "Buddy_06272013"]

PRIMARY = "Achilles_10252013"


def open_nwb(session):
    """Stream one session's NWB file through LINDI with a local cache."""
    url = LINDI_BASE.format(SESSIONS[session])
    f = lindi.LindiH5pyFile.from_lindi_file(
        url, local_cache=lindi.LocalCache(cache_dir=CACHE_DIR))
    return NWBHDF5IO(file=f, mode="r").read()


nwbfile = open_nwb(PRIMARY)
print(nwbfile.session_description[:200])
print("subject:", nwbfile.subject.subject_id, "|", nwbfile.subject.species)
print("behavior:", list(nwbfile.processing["behavior"].data_interfaces))
print("epochs:\n", nwbfile.epochs.to_dataframe())
print("units:", len(nwbfile.units.id), "columns", nwbfile.units.colnames)

# %% [markdown]
# ## Behaviour: rebuilding a single spatial frame
#
# Two quirks of the archived behaviour data have to be handled explicitly.
#
# **Timestamps.** The position `SpatialSeries` objects carry a `rate` field of
# 0.0256, which is the sampling *period* in seconds (39.06 Hz), not a rate. Taking it at
# face value would stretch a 34-minute session across 36 days. Timestamps are therefore
# rebuilt as `starting_time + i * period`, which lands exactly on the end of the maze
# epoch listed in the file.
#
# **Traversal segmentation.** The archived "linearised" position is only defined while
# the animal is actually moving along the track; 87% of the maze-epoch samples are NaN,
# corresponding to time spent in the end wells. Contiguous stretches of defined position
# are therefore the individual traversals, and their direction is the sign of the change
# in position from start to end. Two details matter: brief tracking dropouts (< 0.5 s)
# split what is really one traversal into several blocks and are bridged by
# interpolation, and blocks covering less than 60% of the track are partial excursions
# rather than end-to-end laps and are discarded.

# %%
def _spatial_series(nwbfile, kind):
    beh = nwbfile.processing["behavior"]
    name = [k for k in beh.data_interfaces
            if "Maze" in k and ("Linearized" in k) == (kind == "Linearized")][0]
    return list(beh[name].spatial_series.values())[0]


def load_behavior(nwbfile, max_gap_s=0.5, min_coverage=0.6):
    """Linear position, running direction, speed, and one IntervalSet row per traversal."""
    lin_ss = _spatial_series(nwbfile, "Linearized")
    xy_ss = _spatial_series(nwbfile, "Spatial")
    period = float(lin_ss.rate)                 # seconds per sample (mislabelled as rate)
    lin = np.asarray(lin_ss.data[:]).squeeze().astype(float)
    xy = np.asarray(xy_ss.data[:])
    t = float(lin_ss.starting_time) + np.arange(lin.size) * period
    track_len = float(np.nanmax(lin))

    # bridge short tracking dropouts so that one traversal stays one contiguous block
    defined = np.isfinite(lin)
    idx = np.flatnonzero(defined)
    max_gap = int(round(max_gap_s / period))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            lin[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [lin[a], lin[b]])
            defined[a + 1:b] = True

    edges = np.flatnonzero(np.diff(defined.astype(int)))
    starts = np.r_[0 if defined[0] else [], edges[defined[edges + 1]] + 1].astype(int)
    stops = np.r_[edges[~defined[edges + 1]] + 1, lin.size if defined[-1] else []].astype(int)

    keep = np.zeros(lin.shape, dtype=bool)
    heading = np.full(lin.shape, np.nan)
    runs = []
    for a, b in zip(np.atleast_1d(starts), np.atleast_1d(stops)):
        travel = lin[b - 1] - lin[a]
        if abs(travel) < min_coverage * track_len:      # partial excursion, not a lap
            continue
        keep[a:b] = True
        heading[a:b] = np.sign(travel)
        runs.append((t[a], t[b - 1], np.sign(travel)))
    runs = np.array(runs)

    position = nap.Tsd(t=t[keep], d=lin[keep])
    speed = nap.Tsd(t=position.t, d=np.abs(np.gradient(position.d, position.t))).smooth(0.25)
    mk = lambda sel: nap.IntervalSet(start=runs[sel, 0], end=runs[sel, 1])
    return dict(
        position=position,
        heading=nap.Tsd(t=t[keep], d=heading[keep]),
        speed=speed,
        right_ep=mk(runs[:, 2] > 0),
        left_ep=mk(runs[:, 2] < 0),
        track_len=track_len,
        raw_xy=nap.TsdFrame(t=t, d=xy, columns=["x", "y"]),
        period=period,
    )


def load_units(nwbfile):
    """TsGroup of sorted units with cell_type / location / shank metadata."""
    ut = nwbfile.units
    st = ut["spike_times"]
    tsg = nap.TsGroup({i: np.asarray(st[i]) for i in range(len(ut.id))})
    for c in ("cell_type", "location", "shank_id"):
        tsg.set_info(**{c: np.asarray(ut[c][:])})
    return tsg


def select(units, mask):
    """Subset a TsGroup by a boolean mask over its units, keeping the original keys."""
    keys = np.asarray(list(units.keys()))
    return units[keys[np.asarray(mask, dtype=bool)]]


beh = load_behavior(nwbfile)
units = load_units(nwbfile)
pyr = select(units, units.cell_type == "excitatory")

print(f"track length      : {beh['track_len']:.2f} m")
print(f"traversals R / L  : {len(beh['right_ep'])} / {len(beh['left_ep'])}")
print(f"units             : {len(units)} total, {len(pyr)} putative pyramidal")
print(f"recording sites   : {sorted(set(units.location))}")

# %% [markdown]
# ### Figure 1 — behaviour and the linearisation check
#
# The left-hand panels show the raw camera trajectory and the shuttling pattern over a
# five-minute stretch. The lower-left panel is the validation that matters for everything
# downstream: the linearised position is plotted against the raw camera *x* coordinate
# separately for each heading. The two clouds fall on the same straight line, so a given
# linearised value means the same physical location whichever way the animal is running.

# %%
def running_epochs(beh, direction, laps=None):
    """Traversals for one heading, restricted to samples above the running-speed floor."""
    ep = beh["right_ep"] if direction == "right" else beh["left_ep"]
    if laps is not None:
        ep = ep[np.asarray(laps)]
    fast = beh["speed"].threshold(SPEED_THRESH, "above").time_support
    return ep.intersect(fast).drop_short_intervals(0.2)


fig = plt.figure(figsize=(13, 7))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
xy = beh["raw_xy"]
ax.plot(xy["x"].d, xy["y"].d, lw=0.3, color="0.4")
ax.set(xlabel="camera x (m)", ylabel="camera y (m)", title="Raw trajectory, maze epoch")
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1:])
w = (float(beh["right_ep"].start[0]), float(beh["right_ep"].start[0]) + 300)
p = beh["position"].restrict(nap.IntervalSet(start=w[0], end=w[1]))
ax.plot(p.t, p.d, ".", ms=1.5, color="k")
for ep, c in [(beh["right_ep"], "tab:blue"), (beh["left_ep"], "tab:red")]:
    for s, e in zip(ep.start, ep.end):
        if w[0] <= s <= w[1]:
            ax.axvspan(s, e, color=c, alpha=0.18, lw=0)
ax.set(xlabel="time (s)", ylabel="linear position (m)", xlim=w,
       title="Track traversals (blue = rightward, red = leftward)")

ax = fig.add_subplot(gs[1, 0])
x_at_pos = beh["raw_xy"]["x"].interpolate(beh["position"])
h = beh["heading"].d
for v, c, lab in [(1, "tab:blue", "rightward"), (-1, "tab:red", "leftward")]:
    m = h == v
    ax.plot(x_at_pos.d[m], beh["position"].d[m], ".", ms=1, color=c, alpha=0.3, label=lab)
ax.legend(markerscale=6, frameon=False, fontsize=8)
ax.set(xlabel="raw camera x (m)", ylabel="linearised position (m)",
       title="Same frame for both headings")

ax = fig.add_subplot(gs[1, 1])
for d, c in [("right", "tab:blue"), ("left", "tab:red")]:
    ep = running_epochs(beh, d)
    pos = beh["position"].restrict(ep).d
    occ, edges = np.histogram(pos, bins=n_bins(beh), range=(0, beh["track_len"]))
    ax.plot(edges[:-1] + np.diff(edges) / 2, occ * beh["period"], color=c, label=d)
ax.legend(frameon=False, fontsize=8)
ax.set(xlabel="position (m)", ylabel="occupancy (s)", title="Occupancy while running")

ax = fig.add_subplot(gs[1, 2])
for d, c in [("right", "tab:blue"), ("left", "tab:red")]:
    ep = running_epochs(beh, d)
    pos, sp = beh["position"].restrict(ep).d, beh["speed"].restrict(ep).d
    b = np.linspace(0, beh["track_len"], 21)
    idx = np.clip(np.digitize(pos, b) - 1, 0, len(b) - 2)
    m = np.array([sp[idx == i].mean() if (idx == i).any() else np.nan
                  for i in range(len(b) - 1)])
    ax.plot(b[:-1] + np.diff(b) / 2, m, color=c, label=d)
ax.legend(frameon=False, fontsize=8)
ax.set(xlabel="position (m)", ylabel="mean speed (m/s)", title="Running speed profile")

fig.suptitle(f"DANDI:000044 — {PRIMARY}: behaviour on the linear track", y=0.98)
fig.savefig(f"{FIGDIR}/fig01_behavior.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Rate maps and place-field statistics
#
# Rate maps are occupancy-normalised firing rates in 40 spatial bins, computed separately
# for each running direction and smoothed with a 1.5-bin gaussian. Spatial information is
# the Skaggs measure in bits per spike,
#
# $$\mathrm{SI} = \sum_i p_i \frac{\lambda_i}{\bar\lambda}\log_2\frac{\lambda_i}{\bar\lambda},$$
#
# where $p_i$ is the fraction of running time spent in bin $i$ and $\lambda_i$ the firing
# rate there.

# %%
def rate_maps(units, beh, epochs, bins=None, smooth=SMOOTH_SD_BINS):
    bins = n_bins(beh) if bins is None else bins
    tc = nap.compute_tuning_curves(units, beh["position"], bins=bins,
                                   range=[(0.0, beh["track_len"])], epochs=epochs,
                                   feature_names=["position"])
    occ = tc.attrs["occupancy"]
    data = np.nan_to_num(tc.values, nan=0.0)
    if smooth:
        data = gaussian_filter1d(data, smooth, axis=-1, mode="nearest")
    tc = tc.copy(data=data)
    tc.attrs["occupancy"] = occ
    return tc


def spatial_information(tc):
    """Skaggs spatial information, bits per spike, one value per unit."""
    p = np.asarray(tc.attrs["occupancy"], dtype=float)
    p = p / p.sum()
    lam = np.asarray(tc.values)
    mean_rate = (lam * p).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / mean_rate[:, None]
        term = np.where(lam > 0, p * ratio * np.log2(ratio), 0.0)
    return np.where(mean_rate > 0, term.sum(axis=1), 0.0)


def sparsity(tc):
    """Skaggs sparsity <lambda>^2 / <lambda^2>; low values mean compact fields."""
    p = np.asarray(tc.attrs["occupancy"], dtype=float)
    p = p / p.sum()
    lam = np.asarray(tc.values)
    num, den = (lam * p).sum(axis=1) ** 2, (lam ** 2 * p).sum(axis=1)
    return np.where(den > 0, num / den, np.nan)


def field_stats(tc):
    """Peak rate, peak location, and width of the contiguous region above half-maximum."""
    lam = np.asarray(tc.values)
    centers = np.asarray(tc.coords["position"].values)
    bin_w = centers[1] - centers[0]
    peak_rate, peak_idx = lam.max(axis=1), lam.argmax(axis=1)
    widths = np.zeros(lam.shape[0])
    for i in range(lam.shape[0]):
        above = lam[i] >= 0.5 * peak_rate[i]
        lo = hi = peak_idx[i]
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < lam.shape[1] - 1 and above[hi + 1]:
            hi += 1
        widths[i] = (hi - lo + 1) * bin_w
    return dict(peak_rate=peak_rate, peak_pos=centers[peak_idx], width=widths)


# %% [markdown]
# ### The shuffle control
#
# The null distribution has to destroy the spike–position relationship without changing
# how much the cell fires. Circularly shifting spike times across the whole maze epoch
# does not do that here: only about 12% of the maze epoch is spent traversing the track,
# so a whole-epoch shift drags spikes emitted during immobility into the running windows
# and inflates the shuffled maps by roughly sixfold. Instead, the running epochs are
# concatenated into a gap-free timeline, the shift is applied there, and the result is
# mapped back. Spike counts inside the analysis window are then preserved exactly.

# %%
def circular_shift(units, epochs, rng, min_shift=5.0):
    starts = np.asarray(epochs.start, dtype=float)
    dur = np.asarray(epochs.end, dtype=float) - starts
    offsets = np.r_[0.0, np.cumsum(dur)[:-1]]
    total = dur.sum()
    out = {}
    for i, k in enumerate(units.keys()):
        t = units[k].restrict(epochs).t
        if t.size == 0:
            out[i] = np.array([])
            continue
        idx = np.searchsorted(starts, t, side="right") - 1
        u = np.mod(offsets[idx] + (t - starts[idx])
                   + rng.uniform(min_shift, total - min_shift), total)
        j = np.clip(np.searchsorted(offsets, u, side="right") - 1, 0, len(starts) - 1)
        out[i] = np.sort(starts[j] + (u - offsets[j]))
    return nap.TsGroup(out, time_support=epochs)


def si_null(units, beh, epochs, n_shuffles=N_SHUFFLES, seed=0, progress=True):
    rng = np.random.default_rng(seed)
    null = np.empty((n_shuffles, len(units)))
    it = tqdm(range(n_shuffles), desc="SI shuffles", leave=False) if progress else range(n_shuffles)
    for j in it:
        null[j] = spatial_information(rate_maps(circular_shift(units, epochs, rng), beh, epochs))
    return null


def corr_rows(a, b):
    r = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        if a[i].std() > 0 and b[i].std() > 0:
            r[i] = np.corrcoef(a[i], b[i])[0, 1]
    return r


def odd_even_laps(beh, direction):
    n = len(beh["right_ep"] if direction == "right" else beh["left_ep"])
    return (running_epochs(beh, direction, np.arange(0, n, 2)),
            running_epochs(beh, direction, np.arange(1, n, 2)))


def classify(units, beh, direction, n_shuffles=N_SHUFFLES, seed=0, progress=True):
    """Place-field measures and a place-cell label for one running direction."""
    ep = running_epochs(beh, direction)
    tc = rate_maps(units, beh, ep)
    si = spatial_information(tc)
    null = si_null(units, beh, ep, n_shuffles=n_shuffles, seed=seed, progress=progress)
    p95 = np.percentile(null, 95, axis=0)
    odd, even = odd_even_laps(beh, direction)
    stab = corr_rows(np.asarray(rate_maps(units, beh, odd).values),
                     np.asarray(rate_maps(units, beh, even).values))
    fs = field_stats(tc)
    n_spk = np.array([len(units[k].restrict(ep)) for k in units.keys()])
    active = (n_spk >= MIN_SPIKES) & (fs["peak_rate"] >= MIN_PEAK_RATE)
    return dict(tc=tc, si=si, si_null=null, si_p95=p95, stability=stab,
                sparsity=sparsity(tc), n_spikes=n_spk, active=active, epochs=ep,
                is_place_cell=active & (si > p95) & (stab > MIN_STABILITY), **fs)


res = {d: classify(pyr, beh, d) for d in ("right", "left")}
for d, r in res.items():
    pc = r["is_place_cell"]
    print(f"{d:5s}: {r['active'].sum():3d} active units, {pc.sum():3d} place cells "
          f"({100 * pc.sum() / len(pyr):.0f}% of pyramidal cells) | "
          f"median SI {np.median(r['si'][pc]):.2f} bits/spike, "
          f"peak {np.median(r['peak_rate'][pc]):.1f} Hz, "
          f"width {100 * np.median(r['width'][pc]):.0f} cm")

# %% [markdown]
# ### Figure 2 — raw activity during single traversals
#
# Before any averaging, here is what the data look like: CA1 LFP (dominated by theta
# while the animal runs), the spike raster of every putative pyramidal cell ordered by
# the position of its rightward place field, and the animal's position. Each traversal
# sweeps a diagonal band through the raster, which is the population code for position
# visible directly in the raw spikes.

# %%
def load_lfp(nwbfile, window, channels):
    es = list(nwbfile.processing["ecephys"]["LFP"].electrical_series.values())[0]
    fs = float(es.rate)
    i0 = max(int(round((window[0] - es.starting_time) * fs)), 0)
    i1 = min(int(round((window[1] - es.starting_time) * fs)), es.data.shape[0])
    channels = np.atleast_1d(channels)
    d = np.stack([es.data[i0:i1, c] for c in channels], axis=1).astype(float) * es.conversion
    return nap.TsdFrame(t=es.starting_time + np.arange(i0, i1) / fs, d=d,
                        columns=[f"ch{c}" for c in channels])


def pick_theta_channel(nwbfile, window, candidates):
    """Channel with the largest 6-10 Hz to 1-4 Hz power ratio inside `window`."""
    lfp = load_lfp(nwbfile, window, candidates)
    fs = 1.0 / np.median(np.diff(lfp.t))
    f, p = welch(np.asarray(lfp.values).T, fs=fs, nperseg=int(4 * fs))
    ratio = p[:, (f >= 6) & (f <= 10)].mean(1) / p[:, (f >= 1) & (f <= 4)].mean(1)
    return int(np.atleast_1d(candidates)[np.argmax(ratio)]), ratio


# Order units by the peak of their rightward rate map, using rightward place cells.
pc_right = res["right"]["is_place_cell"]
peak_order = np.argsort(res["right"]["peak_pos"][pc_right])
pc_keys = np.asarray(list(pyr.keys()))[pc_right][peak_order]

t_start = float(res["right"]["epochs"].start[0])
win = (t_start - 3, t_start + 57)
n_trav = int(((beh["right_ep"].start >= win[0]) & (beh["right_ep"].start <= win[1])).sum()
             + ((beh["left_ep"].start >= win[0]) & (beh["left_ep"].start <= win[1])).sum())
theta_ch, _ = pick_theta_channel(nwbfile, (t_start, t_start + 120), np.arange(0, 128, 16))
lfp = load_lfp(nwbfile, win, [theta_ch])[:, 0]
theta = nap.apply_bandpass_filter(lfp, (6, 10))

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={"height_ratios": [1, 3, 1], "hspace": 0.12})
axes[0].plot(lfp.t, lfp.d * 1e3, lw=0.3, color="0.6", label=f"LFP ch{theta_ch}")
axes[0].plot(theta.t, theta.d * 1e3, lw=0.8, color="tab:purple", label="6-10 Hz")
axes[0].legend(ncol=2, frameon=False, fontsize=8, loc="upper right")
axes[0].set_ylabel("mV")
axes[0].set_title(f"{PRIMARY}: raw CA1 activity across {n_trav} track traversals")

win_ep = nap.IntervalSet(start=win[0], end=win[1])
for row, k in enumerate(pc_keys):
    t = pyr[k].restrict(win_ep).t
    axes[1].plot(t, np.full_like(t, row), "|", ms=4, color="k", mew=0.7)
axes[1].set_ylabel("place cell\n(ordered by field position)")
axes[1].set_ylim(-1, len(pc_keys))

p = beh["position"].restrict(win_ep)
axes[2].plot(p.t, p.d, ".", ms=2, color="k")
for ep, c, lab in [(beh["right_ep"], "tab:blue", "rightward"),
                   (beh["left_ep"], "tab:red", "leftward")]:
    for s, e in zip(ep.start, ep.end):
        if win[0] <= s <= win[1]:
            for a in axes:
                a.axvspan(s, e, color=c, alpha=0.12, lw=0)
axes[2].set(xlabel="time (s)", ylabel="position (m)", xlim=win)
fig.savefig(f"{FIGDIR}/fig02_raw_activity.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 3 — example place cells
#
# Six place cells, chosen so that their fields are spread across the track rather than
# simply the six with the highest spatial information (which would all sit at the reward
# ends). On the left, every spike is plotted at the position and traversal number at
# which it occurred, rightward laps below the dashed line and leftward laps above. On
# the right, the resulting rate maps. Spikes cluster into a narrow vertical band that
# repeats lap after lap: that band is the place field.

# %%
def lap_raster(ax, unit, beh):
    """Spike position vs. traversal number, rightward laps below, leftward laps above."""
    n_right = len(beh["right_ep"])
    for ep, base, color in [(beh["right_ep"], 0, "tab:blue"),
                            (beh["left_ep"], n_right + 3, "tab:red")]:
        for lap in range(len(ep)):
            lap_ep = ep[lap]
            t = unit.restrict(lap_ep).t
            if t.size == 0:
                continue
            x = beh["position"].interpolate(nap.Ts(t=t), ep=lap_ep).d
            ax.plot(x, np.full_like(x, base + lap), ".", ms=2, color=color, alpha=0.85)
    ax.axhline(n_right + 1.5, color="0.5", lw=0.8, ls="--")
    return n_right + 3 + len(beh["left_ep"])


keys = np.asarray(list(pyr.keys()))
pcR = res["right"]["is_place_cell"]
targets = np.linspace(0.08, 0.92, 6) * beh["track_len"]
examples, used = [], set()
for target in targets:                       # best-informed place cell nearest each target
    cost = np.where(pcR, np.abs(res["right"]["peak_pos"] - target) - 0.05 * res["right"]["si"],
                    np.inf)
    for i in np.argsort(cost):
        if i not in used:
            used.add(int(i))
            examples.append(int(i))
            break
x_bins = np.asarray(res["right"]["tc"].coords["position"].values)

fig, axes = plt.subplots(3, 4, figsize=(13.5, 9),
                         gridspec_kw={"width_ratios": [2, 1, 2, 1], "hspace": 0.5, "wspace": 0.4})
for n, i in enumerate(examples):
    r, c = divmod(n, 2)
    ax_r, ax_t = axes[r, 2 * c], axes[r, 2 * c + 1]
    ymax = lap_raster(ax_r, pyr[keys[i]], beh)
    ax_r.set(xlim=(0, beh["track_len"]), ylim=(-1, ymax),
             xlabel="position (m)", ylabel="traversal #")
    ax_r.set_title(f"unit {keys[i]}   SI = {res['right']['si'][i]:.2f} bits/spk", fontsize=9)
    ax_t.plot(x_bins, res["right"]["tc"].values[i], color="tab:blue", label="rightward")
    ax_t.plot(x_bins, res["left"]["tc"].values[i], color="tab:red", label="leftward")
    ax_t.set(xlim=(0, beh["track_len"]), xlabel="position (m)", ylabel="rate (Hz)")
    if n == 0:
        ax_t.legend(frameon=False, fontsize=7)
fig.suptitle(f"{PRIMARY}: spikes by traversal (left) and rate maps (right)", y=0.99)
fig.savefig(f"{FIGDIR}/fig03_example_place_cells.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 4 — the population tiles the track, and the map is directional
#
# Sorting normalised rate maps by the location of each cell's peak produces the
# characteristic diagonal: place fields cover the whole track roughly uniformly. The
# right-hand panel applies the *rightward* sort order to the *leftward* maps. The
# diagonal disappears, which is the standard demonstration that CA1 fields on a linear
# track are direction-selective rather than purely positional.

# %%
def sorted_maps(tc_values, order=None):
    norm = tc_values / np.maximum(tc_values.max(axis=1, keepdims=True), 1e-9)
    if order is None:
        order = np.argsort(tc_values.argmax(axis=1))
    return norm[order], order


either = res["right"]["is_place_cell"] | res["left"]["is_place_cell"]
tcR = np.asarray(res["right"]["tc"].values)[either]
tcL = np.asarray(res["left"]["tc"].values)[either]

fig, axes = plt.subplots(1, 3, figsize=(13, 6))
mR, orderR = sorted_maps(tcR)
im = axes[0].imshow(mR, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, beh["track_len"], 0, mR.shape[0]])
axes[0].set(xlabel="position (m)", ylabel="unit (sorted by rightward peak)",
            title="Rightward runs\n(sorted by own peak)")

mL, _ = sorted_maps(tcL)
axes[1].imshow(mL, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, beh["track_len"], 0, mL.shape[0]])
axes[1].set(xlabel="position (m)", ylabel="unit (sorted by leftward peak)",
            title="Leftward runs\n(sorted by own peak)")

mLx, _ = sorted_maps(tcL, orderR)
axes[2].imshow(mLx, aspect="auto", origin="lower", cmap="viridis",
               extent=[0, beh["track_len"], 0, mLx.shape[0]])
axes[2].set(xlabel="position (m)", ylabel="unit (sorted by rightward peak)",
            title="Leftward runs\n(sorted by RIGHTWARD peak)")
fig.colorbar(im, ax=axes, shrink=0.75, label="rate / peak rate", pad=0.02)
fig.suptitle(f"{PRIMARY}: {either.sum()} place cells tile the track", y=1.0)
fig.savefig(f"{FIGDIR}/fig04_population_maps.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 5 — spatial information against the shuffle null

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
r = res["right"]
axes[0].hist(r["si_null"].ravel(), bins=60, range=(0, 3), color="0.7",
             density=True, label="circular shuffle")
axes[0].hist(r["si"][r["active"]], bins=30, range=(0, 3), color="tab:blue",
             density=True, alpha=0.75, label="observed (active units)")
axes[0].set(xlabel="spatial information (bits/spike)", ylabel="density",
            title="Observed vs. shuffled")
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(r["si_p95"][r["active"]], r["si"][r["active"]], "o", ms=4,
             color="tab:blue", alpha=0.7)
lim = [0, max(r["si"][r["active"]].max(), r["si_p95"][r["active"]].max()) * 1.05]
axes[1].plot(lim, lim, "k--", lw=1)
axes[1].set(xlabel="95th percentile of unit's null", ylabel="observed SI",
            xlim=lim, ylim=lim, title="Per-unit significance")

counts = [len(pyr), r["active"].sum(),
          (r["active"] & (r["si"] > r["si_p95"])).sum(), r["is_place_cell"].sum()]
labels = ["pyramidal", "active\nduring runs", "+ SI > null", "+ stable\n(place cells)"]
axes[2].bar(labels, counts, color=["0.75", "0.6", "tab:cyan", "tab:blue"])
for i, c in enumerate(counts):
    axes[2].text(i, c + 2, str(c), ha="center", fontsize=9)
axes[2].set(ylabel="units", title="Selection cascade (rightward runs)")
axes[2].tick_params(axis="x", labelsize=8)
fig.suptitle(f"{PRIMARY}: place-cell classification", y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig05_spatial_information.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 6 — place-field properties and directionality

# %%
pcR = res["right"]["is_place_cell"]
tc_r = np.asarray(res["right"]["tc"].values)
tc_l = np.asarray(res["left"]["tc"].values)
peakR, peakL = tc_r.max(1), tc_l.max(1)
dsi = np.abs(peakR - peakL) / np.maximum(peakR + peakL, 1e-9)
dir_corr = corr_rows(tc_r, tc_l)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
panels = [
    ("peak firing rate (Hz)", res["right"]["peak_rate"][pcR], np.linspace(0, 30, 25)),
    ("field width at half max (cm)", 100 * res["right"]["width"][pcR], np.linspace(0, 100, 25)),
    ("spatial information (bits/spike)", res["right"]["si"][pcR], np.linspace(0, 3, 25)),
    ("sparsity", res["right"]["sparsity"][pcR], np.linspace(0, 1, 25)),
    ("odd/even lap stability (r)", res["right"]["stability"][pcR], np.linspace(-0.2, 1, 25)),
]
for ax, (name, vals, bins) in zip(axes.ravel(), panels):
    ax.hist(vals, bins=bins, color="tab:blue", alpha=0.85)
    ax.axvline(np.median(vals), color="k", ls="--", lw=1)
    ax.set(xlabel=name, ylabel="place cells")
    ax.set_title(f"median {np.median(vals):.2f}", fontsize=9)

ax = axes[1, 2]
ax.hist(dir_corr[pcR], bins=np.linspace(-1, 1, 25), color="tab:orange", alpha=0.85)
ax.axvline(np.nanmedian(dir_corr[pcR]), color="k", ls="--", lw=1)
ax.set(xlabel="right vs. left rate-map correlation", ylabel="place cells")
ax.set_title(f"median {np.nanmedian(dir_corr[pcR]):.2f}  |  DSI {np.nanmedian(dsi[pcR]):.2f}",
             fontsize=9)
fig.suptitle(f"{PRIMARY}: place-field properties (rightward runs, n={pcR.sum()})", y=1.0)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig06_field_properties.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Reading position back out of the population
#
# If the population really carries a spatial code, the animal's position should be
# recoverable from the spikes alone. A naive Bayes decoder does this: given the rate maps
# $\lambda_i(x)$ and the spike counts $n_i$ in a 250 ms window, and assuming the cells are
# conditionally independent Poisson emitters,
#
# $$P(x \mid \mathbf{n}) \propto P(x) \prod_i \frac{\lambda_i(x)^{n_i} e^{-\lambda_i(x)}}{n_i!}.$$
#
# Two choices keep this honest. First, the rate maps are built from *held-out* laps: odd
# laps supply the maps that decode even laps and vice versa, so no time bin is ever
# decoded with a map it helped build. Second, decoding uses **all** putative pyramidal
# cells, not the subset labelled as place cells, because that label was derived using every
# lap and would leak the test data into the cell selection.
#
# The control is a decoder with the same rate maps randomly reassigned among units. It
# preserves every marginal property of the maps and destroys only the correspondence
# between a unit's spikes and its own field.

# %%
def decode_cv(units, beh, direction, bin_size=DECODE_BIN, permute_maps=False, seed=0):
    """Two-fold cross-validated Bayesian decoding of position within one running direction."""
    rng = np.random.default_rng(seed)
    odd, even = odd_even_laps(beh, direction)
    t, dec, true, post, eps = [], [], [], [], []
    for train_ep, test_ep in [(odd, even), (even, odd)]:
        tc = rate_maps(units, beh, train_ep)
        if permute_maps:
            tc = tc.copy(data=np.asarray(tc.values)[rng.permutation(tc.shape[0])])
        decoded, proba = nap.decode_bayes(tc, units, test_ep, bin_size)
        t.append(decoded.t)
        dec.append(decoded.d)
        true.append(beh["position"].interpolate(decoded, ep=test_ep).d)
        post.append(proba)
        eps.append(test_ep)
    o = np.argsort(np.concatenate(t))
    return dict(t=np.concatenate(t)[o], decoded=np.concatenate(dec)[o],
                true=np.concatenate(true)[o], posteriors=post, test_eps=eps,
                direction=direction)


dec = {d: decode_cv(pyr, beh, d) for d in ("right", "left")}
ctl = {d: decode_cv(pyr, beh, d, permute_maps=True) for d in ("right", "left")}

for d in ("right", "left"):
    e = np.abs(dec[d]["decoded"] - dec[d]["true"])
    c = np.abs(ctl[d]["decoded"] - ctl[d]["true"])
    print(f"{d:5s}: {np.isfinite(e).sum():4d} time bins | median error "
          f"{100 * np.nanmedian(e):4.1f} cm (shuffled maps {100 * np.nanmedian(c):4.1f} cm) | "
          f"r = {np.corrcoef(dec[d]['decoded'], dec[d]['true'])[0, 1]:.3f}")

# %% [markdown]
# ### Figure 7 — decoded position on held-out laps
#
# The top panel is the full posterior over position, one column per 250 ms bin, with the
# animal's true position drawn on top. The posterior forms a narrow ridge that follows the
# animal up and down the track. Note that this is a *reconstruction*: the decoder sees only
# spike counts and rate maps estimated from other laps.

# %%
def merged_posterior(dec_results, gap_factor=1.5):
    """All folds' and directions' posteriors on one time axis, split into contiguous runs.

    Laps are separated by seconds spent in the reward wells, during which nothing is
    decoded. Drawing the whole session as one pcolormesh would stretch the 250 ms bins
    that flank each gap across it, so the columns are returned as separate segments and
    each is drawn on its own.
    """
    P = np.vstack([np.asarray(p.values) for r in dec_results for p in r["posteriors"]])
    t = np.concatenate([p.t for r in dec_results for p in r["posteriors"]])
    o = np.argsort(t)
    t, P = t[o], P[o]
    breaks = np.flatnonzero(np.diff(t) > gap_factor * DECODE_BIN) + 1
    bins = np.asarray(dec_results[0]["posteriors"][0].columns, dtype=float)
    return [(t[a:b], P[a:b]) for a, b in
            zip(np.r_[0, breaks], np.r_[breaks, len(t)])], bins


fig = plt.figure(figsize=(13.5, 8))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1], hspace=0.42, wspace=0.32)

# --- posterior over a stretch of consecutive held-out laps, both directions merged
ax = fig.add_subplot(gs[0, :])
segments, xb = merged_posterior([dec["right"], dec["left"]])
w0 = float(dec["right"]["test_eps"][0].start[0]) - 4
w1 = w0 + 100
vmax = np.percentile(np.vstack([P for _, P in segments]), 99.0)
pm = None
for t_seg, P_seg in segments:
    if t_seg[-1] < w0 or t_seg[0] > w1 or len(t_seg) < 2:
        continue
    pm = ax.pcolormesh(t_seg, xb, P_seg.T, cmap="magma", shading="nearest",
                       vmin=0, vmax=vmax)
pos_w = beh["position"].restrict(nap.IntervalSet(start=w0, end=w1))
ax.plot(pos_w.t, pos_w.d, "-", lw=2.5, color="tab:cyan", alpha=0.55)
ax.set(xlabel="time (s)", ylabel="position (m)", xlim=(w0, w1), ylim=(0, beh["track_len"]),
       title="Posterior P(position | spikes) on held-out laps, all pyramidal cells "
             "(cyan = true position)")
fig.colorbar(pm, ax=ax, pad=0.01, label="posterior")

# --- confusion matrix, pooled over directions
ax = fig.add_subplot(gs[1, 0])
true_all = np.concatenate([dec[d]["true"] for d in ("right", "left")])
pred_all = np.concatenate([dec[d]["decoded"] for d in ("right", "left")])
ok = np.isfinite(true_all) & np.isfinite(pred_all)
edges = np.linspace(0, beh["track_len"], n_bins(beh) + 1)
H, _, _ = np.histogram2d(true_all[ok], pred_all[ok], bins=[edges, edges])
H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
ax.imshow(H.T, origin="lower", aspect="equal", cmap="viridis",
          extent=[0, beh["track_len"], 0, beh["track_len"]])
ax.plot([0, beh["track_len"]], [0, beh["track_len"]], "w--", lw=1)
ax.set(xlabel="true position (m)", ylabel="decoded position (m)",
       title=f"Confusion matrix (r = {np.corrcoef(true_all[ok], pred_all[ok])[0, 1]:.3f})")

# --- error distribution against the shuffled-map control
ax = fig.add_subplot(gs[1, 1])
err = 100 * np.abs(pred_all - true_all)[ok]
cerr_all = 100 * np.abs(np.concatenate([ctl[d]["decoded"] - ctl[d]["true"]
                                        for d in ("right", "left")]))
cerr_all = cerr_all[np.isfinite(cerr_all)]
bins_e = np.linspace(0, 100 * beh["track_len"], 40)
ax.hist(cerr_all, bins=bins_e, color="0.7", density=True, label="shuffled maps")
ax.hist(err, bins=bins_e, color="tab:green", density=True, alpha=0.8, label="decoder")
ax.axvline(np.median(err), color="tab:green", ls="--", lw=1.2)
ax.axvline(np.median(cerr_all), color="0.4", ls="--", lw=1.2)
ax.legend(frameon=False, fontsize=8)
ax.set(xlabel="absolute decoding error (cm)", ylabel="density",
       title=f"median {np.median(err):.1f} cm vs {np.median(cerr_all):.1f} cm")

# --- error as a function of the number of cells used
ax = fig.add_subplot(gs[1, 2])
rng = np.random.default_rng(1)
keys_all = np.asarray(list(pyr.keys()))
sizes = [2, 5, 10, 20, 40, 80, len(keys_all)]
curve = []
for n in tqdm(sizes, desc="cells vs error", leave=False):
    reps = []
    for _ in range(5 if n < len(keys_all) else 1):
        sub = pyr[rng.choice(keys_all, size=n, replace=False)]
        r = decode_cv(sub, beh, "right")
        reps.append(100 * np.nanmedian(np.abs(r["decoded"] - r["true"])))
    curve.append((np.mean(reps), np.std(reps)))
curve = np.array(curve)
ax.errorbar(sizes, curve[:, 0], yerr=curve[:, 1], marker="o", ms=4, color="tab:green")
ax.axhline(np.median(cerr_all), color="0.4", ls="--", lw=1.2, label="shuffled maps")
ax.set(xscale="log", xlabel="number of pyramidal cells", ylabel="median error (cm)",
       title="Decoding improves with population size")
ax.legend(frameon=False, fontsize=8)

fig.suptitle(f"{PRIMARY}: position decoded from CA1 spikes on held-out laps", y=0.97)
fig.savefig(f"{FIGDIR}/fig07_decoding.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Every linear-track session in the dandiset
#
# The pipeline so far has run on one session. DANDI:000044 contains eight sessions from
# four rats; three used a circular maze and are skipped, since the traversal segmentation
# above assumes an open-ended track. The remaining five (four rats, one on a 2 m rather
# than a 1.6 m track) are now processed identically, start to finish, with no per-session
# tuning.

# %%
def analyze_session(name, n_shuffles=N_SHUFFLES, progress=False):
    """Run the whole pipeline on one session and return its summary."""
    nwbf = open_nwb(name)
    b = load_behavior(nwbf)
    u = load_units(nwbf)
    p = select(u, u.cell_type == "excitatory")
    out = dict(session=name, subject=nwbf.subject.subject_id, track_len=b["track_len"],
               n_units=len(u), n_pyr=len(p),
               n_laps=len(b["right_ep"]) + len(b["left_ep"]), beh=b)
    for d in ("right", "left"):
        r = classify(p, b, d, n_shuffles=n_shuffles, progress=progress)
        dd = decode_cv(p, b, d)
        cc = decode_cv(p, b, d, permute_maps=True)
        r["decode_err"] = 100 * np.nanmedian(np.abs(dd["decoded"] - dd["true"]))
        r["decode_ctl"] = 100 * np.nanmedian(np.abs(cc["decoded"] - cc["true"]))
        r["decode_r"] = np.corrcoef(dd["decoded"], dd["true"])[0, 1]
        out[d] = r
    out["either"] = out["right"]["is_place_cell"] | out["left"]["is_place_cell"]
    return out


sessions = {}
for name in tqdm(LINEAR_SESSIONS, desc="sessions"):
    sessions[name] = analyze_session(name)
    s = sessions[name]
    print(f"{name:18s} {s['subject']:9s} {s['track_len']:.1f} m  {s['n_laps']:3d} laps  "
          f"{s['n_pyr']:3d} pyr  {s['either'].sum():3d} place cells "
          f"({100 * s['either'].mean():4.1f}%)  decode "
          f"{np.mean([s[d]['decode_err'] for d in ('right', 'left')]):4.1f} cm "
          f"(ctl {np.mean([s[d]['decode_ctl'] for d in ('right', 'left')]):4.1f} cm)")

# %% [markdown]
# ### Figure 8 — the result holds across sessions and animals
#
# Each of the five sessions yields a large population of significantly and stably
# spatially tuned CA1 pyramidal cells, with consistent field widths, peak rates and
# spatial information, and in every session the decoder recovers position far better than
# the shuffled-map control. The bottom-right panel pools the normalised rate maps of every
# place cell from every session on a common relative-position axis.

# %%
COLORS = plt.cm.tab10(np.arange(len(LINEAR_SESSIONS)))
short = {n: n.split("_")[0] + "\n" + n.split("_")[1] for n in LINEAR_SESSIONS}


def pooled(key, per_direction=True):
    """Concatenate a per-unit measure over place cells, per session."""
    out = {}
    for n, s in sessions.items():
        vals = [s[d][key][s[d]["is_place_cell"]] for d in
                (("right", "left") if per_direction else ("right",))]
        out[n] = np.concatenate(vals)
    return out


fig, axes = plt.subplots(2, 3, figsize=(14, 8))

ax = axes[0, 0]
frac = [100 * sessions[n]["either"].mean() for n in LINEAR_SESSIONS]
ax.bar(range(len(LINEAR_SESSIONS)), frac, color=COLORS)
for i, (f, n) in enumerate(zip(frac, LINEAR_SESSIONS)):
    ax.text(i, f + 1, f"{sessions[n]['either'].sum()}/{sessions[n]['n_pyr']}",
            ha="center", fontsize=7)
ax.set_xticks(range(len(LINEAR_SESSIONS)))
ax.set_xticklabels([short[n] for n in LINEAR_SESSIONS], fontsize=7)
ax.set(ylabel="% of pyramidal cells", title="Place cells per session",
       ylim=(0, max(frac) * 1.2))

for ax, (key, label, scale, bins) in zip(
        [axes[0, 1], axes[0, 2], axes[1, 0]],
        [("si", "spatial information (bits/spike)", 1, np.linspace(0, 3, 22)),
         ("width", "field width at half max (cm)", 100, np.linspace(0, 100, 22)),
         ("peak_rate", "peak firing rate (Hz)", 1, np.linspace(0, 30, 22))]):
    vals = pooled(key)
    for n, c in zip(LINEAR_SESSIONS, COLORS):
        ax.hist(scale * vals[n], bins=bins, histtype="step", lw=1.4, density=True,
                color=c, label=f"{short[n].replace(chr(10), ' ')}")
    allv = scale * np.concatenate([vals[n] for n in LINEAR_SESSIONS])
    ax.axvline(np.median(allv), color="k", ls="--", lw=1.2)
    ax.set(xlabel=label, ylabel="density", title=f"pooled median {np.median(allv):.2f}")

axes[0, 1].legend(frameon=False, fontsize=6.5, loc="upper right")

ax = axes[1, 1]
w = 0.38
x = np.arange(len(LINEAR_SESSIONS))
errs = [np.mean([sessions[n][d]["decode_err"] for d in ("right", "left")])
        for n in LINEAR_SESSIONS]
ctls = [np.mean([sessions[n][d]["decode_ctl"] for d in ("right", "left")])
        for n in LINEAR_SESSIONS]
ax.bar(x - w / 2, errs, w, color="tab:green", label="decoder")
ax.bar(x + w / 2, ctls, w, color="0.7", label="shuffled maps")
ax.set_xticks(x)
ax.set_xticklabels([short[n] for n in LINEAR_SESSIONS], fontsize=7)
ax.legend(frameon=False, fontsize=8)
ax.set(ylabel="median decoding error (cm)", title="Held-out position decoding")

ax = axes[1, 2]
maps = []
for n, s in sessions.items():
    for d in ("right", "left"):
        pc = s[d]["is_place_cell"]
        if not pc.any():
            continue
        m = np.asarray(s[d]["tc"].values)[pc]
        maps.append(m / np.maximum(m.max(1, keepdims=True), 1e-9))
common = min(m.shape[1] for m in maps)
maps = np.vstack([np.stack([np.interp(np.linspace(0, 1, common),
                                      np.linspace(0, 1, m.shape[1]), row) for row in m])
                  for m in maps])
maps = maps[np.argsort(maps.argmax(axis=1))]
im = ax.imshow(maps, aspect="auto", origin="lower", cmap="viridis", extent=[0, 1, 0, len(maps)])
fig.colorbar(im, ax=ax, pad=0.02, label="rate / peak rate")
ax.set(xlabel="relative position along track", ylabel="place cell (all sessions)",
       title=f"{len(maps)} place fields tile the track")

fig.suptitle("DANDI:000044 — all five linear-track sessions, identical pipeline", y=1.0)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig08_across_sessions.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# Streaming five sessions of CA1 tetrode-equivalent silicon-probe data from
# [DANDI:000044](https://dandiarchive.org/dandiset/000044) reproduces the defining
# properties of hippocampal place cells with no per-session tuning:
#
# * A large fraction of putative pyramidal cells fire in a restricted stretch of the
#   linear track, with spatial information above a within-running-epoch circular shuffle
#   null and rate maps that reproduce across odd and even laps.
# * Fields are compact (tens of centimetres at half maximum), sparse, and low-rate outside
#   the field.
# * The population tiles the track, and fields are strongly direction-selective: the
#   sort order that produces a clean diagonal for rightward runs does not for leftward runs.
# * The code is legible enough that a naive Bayes decoder trained on held-out laps
#   recovers the animal's position to within a few centimetres, many times better than a
#   control in which the rate maps are randomly reassigned among units, and the error falls
#   steadily as more cells are added.

# %%
print(f"{sum(int(s['either'].sum()) for s in sessions.values())} place cells across "
      f"{len(sessions)} sessions and "
      f"{len(set(s['subject'] for s in sessions.values()))} rats")
