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
# **Data: [DANDI:000044](https://dandiarchive.org/dandiset/000044)** — Grosmark & Buzsáki
# (2016), *Diversity in neural firing dynamics supports both rigid and learned
# hippocampal sequences*, Science 351:1440. Eight bilateral silicon-probe sessions
# from dorsal CA1 in four rats (Achilles, Buddy, Cicero, Gatsby), each consisting
# of a pre-behaviour sleep period, a period of running on a maze for water reward,
# and a post-behaviour sleep period.
#
# A **place cell** is a hippocampal pyramidal neuron that fires when, and only
# when, the animal occupies a particular part of its environment (O'Keefe &
# Dostrovsky, 1971). This notebook demonstrates the phenomenon end to end on real
# data:
#
# 1. Stream the NWB files from the DANDI Archive and reconstruct position, running
#    laps and spike trains as pynapple objects.
# 2. Build occupancy-normalised firing rate maps for each unit and each running
#    direction.
# 3. Test every unit against a null distribution built by circularly shifting its
#    own spike train, and against the independent criterion of split-half
#    stability across laps.
# 4. Characterise the resulting fields: peak rate, width, coverage of the track,
#    reliability lap by lap, and the direction-specific remapping that is typical
#    of linear tracks.
# 5. Decode the animal's position from held-out laps by Bayesian inference, which
#    turns the single-cell result into a statement about the population code.
# 6. Fit Poisson GLMs with `nemos` to check that the fields reflect position
#    rather than the running-speed profile along the track.
# 7. Repeat the whole analysis over all eight sessions.
#
# Everything runs from streamed data with on-disk caching; nothing is downloaded
# in full and no data are simulated.

# %%
import re
import time

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import remfile
import requests
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm

import pynapple as nap
import nemos as nmo

nap.nap_config.suppress_conversion_warnings = True

# %% [markdown]
# ## Analysis parameters
#
# Rate maps use 4 cm bins smoothed with a 6 cm Gaussian. A lap counts only if the
# animal covered at least 60% of the track in a consistent direction, and only the
# parts of a lap where it was running faster than 5 cm/s enter the maps. A unit is
# called a place cell if, in at least one running direction, its spatial
# information exceeds the 99th percentile of its own circular-shift null and it
# fires enough spikes to make the map meaningful.

# %%
DANDISET_ID = "000044"


DANDI_API = "https://api.dandiarchive.org/api"


REMFILE_CACHE = "/tmp/remfile_cache"


TRACK_LENGTH_CM = 160.0        # default; overridden per session


BIN_WIDTH_CM = 4.0


N_POS_BINS = 40                # default; recomputed per session as L / BIN_WIDTH_CM


SMOOTH_SIGMA_BINS = 1.5        # ~6 cm Gaussian smoothing of rate maps


MIN_LAP_DURATION_S = 0.5


MIN_LAP_COVERAGE = 0.6         # a lap must traverse >=60% of the track


MIN_LAP_MONOTONICITY = 0.8     # ... and do so without doubling back much


MIN_SPEED_CM_S = 5.0


MIN_SPIKES_ON_TRACK = 50


MIN_PEAK_RATE_HZ = 1.0


N_SHUFFLES = 1000


SHUFFLE_ALPHA = 0.01


DIRECTIONS = ("rightward", "leftward")


DIR_SIGN = {"rightward": 1, "leftward": -1}

# %% [markdown]
# ## Streaming the data from DANDI
#
# The eight assets are 5–9 GB each, almost all of it raw electrophysiology, so
# they are streamed with `remfile` and an on-disk chunk cache rather than
# downloaded. Only the spike times, the behavioural position and short snippets of
# LFP are ever pulled across the network.
#
# Three properties of these particular NWB files shape the loader:
#
# 1. The behavioural `SpatialSeries` store the sampling **period** (0.0256 s) in
#    the `rate` field rather than the rate, so timestamps have to be rebuilt as
#    `starting_time + arange(n) * rate`. The resulting 39.06 Hz stream spans
#    exactly the maze epoch.
# 2. The linearised position is `NaN` whenever the animal is not on the track, so
#    the contiguous non-`NaN` segments bracket its time on the track. Those
#    segments include brief shuffles at the reward ports as well as full runs, so
#    laps are selected by requiring a segment to cross most of the track in one
#    direction.
# 3. The track differs between sessions (1.6 m linear, 2 m linear, or a ~2.9 m
#    circular maze) and is named in the `Position` container, so bin edges are
#    derived per session rather than hard-coded.

# %%
def list_session_urls(dandiset_id=DANDISET_ID):
    """Return {asset_path: public S3 URL} for every NWB asset in the dandiset."""
    r = requests.get(
        f"{DANDI_API}/dandisets/{dandiset_id}/versions/draft/assets/",
        params={"page_size": 100},
    )
    r.raise_for_status()
    assets = r.json()["results"]
    urls = {}
    for a in sorted(assets, key=lambda x: x["path"]):
        rr = requests.get(f"{DANDI_API}/assets/{a['asset_id']}/download/",
                          allow_redirects=False)
        # Strip the presigned query string: DANDI blobs are publicly readable and
        # the bare URL is stable, which keeps the remfile disk cache valid.
        urls[a["path"]] = rr.headers["Location"].split("?")[0]
    return urls


def open_nwb(s3_url, cache_dir=REMFILE_CACHE):
    """Stream an NWB file from S3 with on-disk chunk caching."""
    rf = remfile.File(s3_url, disk_cache=remfile.DiskCache(cache_dir))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    return io.read(), io

# %%
def load_session(s3_url, cache_dir=REMFILE_CACHE):
    """Build the pynapple objects needed for the place-field analysis.

    Returns a dict with:
        position  : Tsd, linearised position in cm (NaN samples dropped)
        run_ep    : IntervalSet, one interval per track traversal
        lap_dir   : (n_laps,) array of +1 (rightward) / -1 (leftward)
        speed     : Tsd, |dx/dt| in cm/s on the run epochs
        units     : TsGroup of all sorted units with location/cell_type metadata
        maze_ep   : IntervalSet covering the whole maze epoch
        epochs    : IntervalSet of PRE / MAZE / POST with an `label` column
        nwbfile, io, session_id, subject_id
    """
    nwbfile, io = open_nwb(s3_url, cache_dir)

    # --- session epochs -----------------------------------------------------
    ep_df = nwbfile.epochs.to_dataframe()
    epochs = nap.IntervalSet(
        start=ep_df["start_time"].values,
        end=ep_df["stop_time"].values,
        metadata={"label": ep_df["label"].values},
    )
    maze_row = ep_df[ep_df["label"].str.contains("Maze", case=False)].iloc[0]
    maze_ep = nap.IntervalSet(start=maze_row["start_time"], end=maze_row["stop_time"])

    # --- linearised position ------------------------------------------------
    beh = nwbfile.processing["behavior"]
    lin_key = [k for k in beh.data_interfaces if "Linearized" in k][0]
    lin_container = beh[lin_key]
    ss = lin_container[list(lin_container.spatial_series)[0]]
    dt = ss.rate                      # NB: this field actually holds the period
    raw = np.asarray(ss.data[:]).squeeze()
    t = ss.starting_time + np.arange(raw.size) * dt
    pos_cm = raw * 100.0              # metres -> cm

    valid = ~np.isnan(pos_cm)
    position = nap.Tsd(t=t[valid], d=pos_cm[valid])

    # --- 2D position (for the trajectory overview) --------------------------
    pos2_key = [k for k in beh.data_interfaces
                if "Position" in k and "Linearized" not in k][0]
    ss2 = beh[pos2_key][list(beh[pos2_key].spatial_series)[0]]
    raw2 = np.asarray(ss2.data[:]) * 100.0
    t2 = ss2.starting_time + np.arange(raw2.shape[0]) * ss2.rate
    ok2 = np.all(np.isfinite(raw2), axis=1)
    position_2d = nap.TsdFrame(t=t2[ok2], d=raw2[ok2], columns=["x", "y"])

    # --- maze geometry ------------------------------------------------------
    maze_type = "circular" if "Circular" in lin_key else "linear"
    m = re.match(r"([\d.]+)m", lin_key)
    track_cm = float(m.group(1)) * 100.0 if m else float(np.ceil(np.nanmax(pos_cm)))
    n_bins = int(round(track_cm / BIN_WIDTH_CM))

    # --- laps: contiguous valid segments that traverse most of the track ----
    segs = _contiguous_epochs(t, valid, dt).drop_short_intervals(MIN_LAP_DURATION_S)
    run_ep, lap_dir = _select_traversals(position, segs, track_cm)

    # --- speed --------------------------------------------------------------
    speed = _speed_cm_s(position, run_ep)

    # --- units --------------------------------------------------------------
    udf = nwbfile.units.to_dataframe()
    spike_dict = {int(i): np.asarray(v) for i, v in udf["spike_times"].items()}
    meta = pd.DataFrame(
        {
            "location": udf["location"].values,
            "cell_type": udf["cell_type"].values,
            "shank_id": udf["shank_id"].values,
        },
        index=[int(i) for i in udf.index],
    )
    units = nap.TsGroup(spike_dict, metadata=meta)

    return dict(
        position=position,
        position_2d=position_2d,
        run_ep=run_ep,
        lap_dir=lap_dir,
        speed=speed,
        units=units,
        maze_ep=maze_ep,
        epochs=epochs,
        dt=dt,
        track_cm=track_cm,
        n_bins=n_bins,
        maze_type=maze_type,
        maze_name=lin_key.replace("LinearizedPosition", ""),
        nwbfile=nwbfile,
        io=io,
        session_id=nwbfile.session_id,
        subject_id=nwbfile.subject.subject_id,
    )


def _contiguous_epochs(t, valid, dt, max_gap_factor=1.5):
    """IntervalSet spanning each run of consecutive `valid` samples."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return nap.IntervalSet(start=[], end=[])
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return nap.IntervalSet(start=t[starts], end=t[ends] + dt * (max_gap_factor - 0.5))


def _select_traversals(position, segs, track_cm,
                       min_coverage=MIN_LAP_COVERAGE,
                       min_monotonicity=MIN_LAP_MONOTONICITY):
    """Keep only segments that are genuine end-to-end runs across the track.

    The NaN gaps in the linearised position bracket every period the animal was
    on the track, which includes brief shuffles at the reward ports as well as
    full traversals. A lap must cover at least `min_coverage` of the track and
    move in a consistent direction for at least `min_monotonicity` of its
    samples.
    """
    keep, dirs = [], []
    for i in range(len(segs)):
        p = position.restrict(segs[i]).values
        if p.size < 5:
            continue
        span = p[-1] - p[0]
        if abs(span) < min_coverage * track_cm:
            continue
        sign = 1 if span > 0 else -1
        dv = np.diff(p)
        if np.mean(np.sign(dv) == sign) < min_monotonicity:
            continue
        keep.append(i)
        dirs.append(sign)
    if not keep:
        return nap.IntervalSet(start=[], end=[]), np.zeros(0, dtype=int)
    return segs[np.array(keep)], np.array(dirs, dtype=int)


def _speed_cm_s(position, run_ep):
    """Absolute running speed, computed lap-by-lap so gaps are not differenced."""
    ts, vs = [], []
    for i in range(len(run_ep)):
        p = position.restrict(run_ep[i])
        if len(p) < 3:
            continue
        v = np.gradient(p.values, p.t)
        ts.append(p.t)
        vs.append(np.abs(v))
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vs), time_support=run_ep)

# %%
def lap_epochs(run_ep, lap_dir, direction):
    """Lap-level IntervalSet for one travel direction (no speed masking)."""
    return run_ep[np.flatnonzero(lap_dir == DIR_SIGN[direction])]


def apply_speed_mask(ep, speed, min_speed=MIN_SPEED_CM_S):
    """Keep only the portions of `ep` where the animal is running."""
    fast = speed.restrict(ep).threshold(min_speed, "above").time_support
    return ep.intersect(fast).drop_short_intervals(0.2)


def direction_epochs(run_ep, lap_dir, speed=None, min_speed=MIN_SPEED_CM_S):
    """Split run epochs by travel direction, optionally masked by running speed."""
    out = {}
    for name in DIRECTIONS:
        ep = lap_epochs(run_ep, lap_dir, name)
        if speed is not None:
            ep = apply_speed_mask(ep, speed, min_speed)
        out[name] = ep
    return out

# %% [markdown]
# Load the session used throughout the first half of the notebook: rat Achilles on
# a 1.6 m linear track.

# %%
PRIMARY = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
urls = list_session_urls()
print(f"{len(urls)} NWB assets in DANDI:{DANDISET_ID}")

S = load_session(urls[PRIMARY])
print(S["nwbfile"])

# %%
print(f"session      : {S['session_id']}  (subject {S['subject_id']})")
print(f"maze         : {S['maze_name']}, {S['track_cm']:.0f} cm {S['maze_type']}, "
      f"{S['n_bins']} position bins")
print(f"epochs       :\n{S['epochs']}")
print(f"units        : {len(S['units'])} "
      f"({(S['units'].metadata['cell_type'] == 'excitatory').sum()} excitatory, "
      f"{(S['units'].metadata['cell_type'] == 'inhibitory').sum()} inhibitory)")
print(f"position     : {len(S['position'])} samples at "
      f"{1 / S['dt']:.2f} Hz, {S['position'].values.min():.0f}-"
      f"{S['position'].values.max():.0f} cm")
print(f"laps         : {(S['lap_dir'] == 1).sum()} rightward, "
      f"{(S['lap_dir'] == -1).sum()} leftward, "
      f"{(S['run_ep'].end - S['run_ep'].start).sum():.0f} s of running")
print(f"running speed: median {np.median(S['speed'].values):.0f} cm/s")

# %% [markdown]
# ### Checking the behavioural stream before analysing it
#
# The panels below confirm that the reconstructed timestamps, the lap segmentation
# and the direction assignment are all correct: the animal shuttles end to end,
# each detected lap is a clean monotonic traversal, and the two directions
# alternate.

# %%
fig, ax = plt.subplots(3, 1, figsize=(13, 8))
fig.subplots_adjust(hspace=0.55)

pos = S["position"]
ax[0].plot(pos.t, pos.values, ".", color="0.75", ms=1)
for name in DIRECTIONS:
    laps = lap_epochs(S["run_ep"], S["lap_dir"], name)
    for i in range(len(laps)):
        seg = pos.restrict(laps[i])
        ax[0].plot(seg.t, seg.values, lw=1,
                   color="#1b6ca8" if name == "rightward" else "#d1495b")
ax[0].set_xlim(float(S["maze_ep"].start[0]), float(S["maze_ep"].end[0]))
ax[0].set_ylabel("position (cm)")
ax[0].set_xlabel("time (s)")
ax[0].set_title("Linearised position over the maze epoch "
                "(grey = on the track, coloured = accepted laps)")

t0 = float(S["run_ep"].start[3]) - 5
zoom = nap.IntervalSet(start=t0, end=t0 + 90)
seg = pos.restrict(zoom)
ax[1].plot(seg.t - t0, seg.values, "k.-", ms=2, lw=0.5)
for i in range(len(S["run_ep"])):
    a, b = float(S["run_ep"].start[i]), float(S["run_ep"].end[i])
    if b > t0 and a < t0 + 90:
        ax[1].axvspan(a - t0, b - t0, color="gold", alpha=0.35)
ax[1].set_xlim(0, 90)
ax[1].set_ylabel("position (cm)")
ax[1].set_xlabel("time from start of window (s)")
ax[1].set_title("90 s zoom: shaded bands are the detected laps")

ax[2].hist(S["speed"].values, bins=60, color="0.4")
ax[2].axvline(MIN_SPEED_CM_S, color="crimson", lw=2)
ax[2].set_xlabel("running speed (cm/s)")
ax[2].set_ylabel("samples")
ax[2].set_title(f"Speed during laps; only samples above the red line "
                f"({MIN_SPEED_CM_S:.0f} cm/s) enter the rate maps")
fig.savefig("fig00_behaviour_check.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Rate maps and place-field statistics
#
# A rate map is the number of spikes fired in each position bin divided by the
# time spent there. The implementation below is byte-for-byte equivalent to
# pynapple's `compute_1d_tuning_curves` (verified further down); it is written out
# explicitly because the same binning has to be reused thousands of times inside
# the shuffle control, where going through pynapple would be far too slow.
#
# Spatial information follows Skaggs et al. (1993),
#
# $$\mathrm{SI} = \sum_i p_i \frac{\lambda_i}{\bar\lambda}
#    \log_2 \frac{\lambda_i}{\bar\lambda} \quad \text{bits/spike},$$
#
# where $p_i$ is the fraction of time spent in bin $i$ and $\lambda_i$ the firing
# rate there.

# %%
def _bin_edges(n_bins=N_POS_BINS, track_cm=TRACK_LENGTH_CM):
    return np.linspace(0.0, track_cm, n_bins + 1)


def rate_maps(units, position, epochs, dt, n_bins=N_POS_BINS,
              track_cm=TRACK_LENGTH_CM, sigma=SMOOTH_SIGMA_BINS,
              circular=False):
    """Occupancy-normalised firing-rate maps.

    Returns (rates [n_units, n_bins] Hz, occupancy [n_bins] s, bin_centres,
    spike_counts [n_units, n_bins]).

    Each position sample is treated as occupying `dt` seconds, which matches the
    way pynapple's tuning-curve routine normalises but lets us reuse the same
    binning for the shuffle control.
    """
    edges = _bin_edges(n_bins, track_cm)
    centres = 0.5 * (edges[:-1] + edges[1:])

    p = position.restrict(epochs)
    pos_bin = np.clip(np.digitize(p.values, edges) - 1, 0, n_bins - 1)
    occ = np.bincount(pos_bin, minlength=n_bins).astype(float) * dt

    counts = np.zeros((len(units), n_bins))
    for k, uid in enumerate(units.index):
        st = units[uid].restrict(epochs).t
        if st.size == 0:
            continue
        b = _spike_position_bins(st, p.t, pos_bin, dt)
        counts[k] = np.bincount(b, minlength=n_bins)

    rates = np.divide(counts, occ, out=np.zeros_like(counts), where=occ > 0)
    if sigma:
        rates = gaussian_filter1d(rates, sigma, axis=1,
                                  mode="wrap" if circular else "nearest")
    return rates, occ, centres, counts


def _spike_position_bins(spike_t, sample_t, pos_bin, dt):
    """Position bin of the behavioural sample nearest each spike."""
    return pos_bin[_spike_sample_index(spike_t, sample_t)]


def _spike_sample_index(spike_t, sample_t):
    """Index of the behavioural sample nearest each spike time."""
    i = np.searchsorted(sample_t, spike_t)
    i = np.clip(i, 1, sample_t.size - 1)
    left = spike_t - sample_t[i - 1]
    right = sample_t[i] - spike_t
    return np.where(right < left, i, i - 1)

# %%
def spatial_information(rates, occ):
    """Skaggs et al. (1993) spatial information in bits per spike.

    SI = sum_i p_i * (r_i / r_bar) * log2(r_i / r_bar)
    """
    p = occ / occ.sum()
    rbar = (rates * p).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rates / rbar[:, None]
        term = np.where(rates > 0, p * ratio * np.log2(ratio), 0.0)
    si = np.nansum(term, axis=1)
    return np.where(rbar > 0, si, 0.0)


def sparsity(rates, occ):
    """Spatial sparsity: (sum p r)^2 / sum p r^2. Low values = compact fields."""
    p = occ / occ.sum()
    num = ((rates * p).sum(axis=1)) ** 2
    den = ((rates ** 2) * p).sum(axis=1)
    return np.divide(num, den, out=np.ones_like(num), where=den > 0)


def _field_extent(r, frac=0.5):
    """(lo, hi) bin indices of the contiguous region around the peak >= frac*peak."""
    pk = int(np.argmax(r))
    thr = frac * r[pk]
    lo = pk
    while lo > 0 and r[lo - 1] >= thr:
        lo -= 1
    hi = pk
    while hi < r.size - 1 and r[hi + 1] >= thr:
        hi += 1
    return lo, hi


def field_width_cm(rates, centres, frac=0.5):
    """Width of the contiguous region around the peak exceeding `frac` of peak."""
    bin_w = centres[1] - centres[0]
    out = np.zeros(rates.shape[0])
    for k, r in enumerate(rates):
        if r.max() <= 0:
            continue
        lo, hi = _field_extent(r, frac)
        out[k] = (hi - lo + 1) * bin_w
    return out


def field_contrast(rates, frac=0.5):
    """Mean in-field rate divided by mean out-of-field rate.

    A direct, spike-count-robust measure of how selective the firing is: a cell
    with a genuine place field fires many times faster inside its field than
    anywhere else on the track.
    """
    out = np.full(rates.shape[0], np.nan)
    for k, r in enumerate(rates):
        if r.max() <= 0:
            continue
        lo, hi = _field_extent(r, frac)
        mask = np.zeros(r.size, dtype=bool)
        mask[lo:hi + 1] = True
        if mask.all():
            continue
        outside = r[~mask].mean()
        out[k] = r[mask].mean() / outside if outside > 0 else np.inf
    return out


def lap_participation(units, position, laps, speed, dt, rates,
                      n_bins=N_POS_BINS, track_cm=TRACK_LENGTH_CM,
                      min_speed=MIN_SPEED_CM_S, frac=0.5):
    """Fraction of laps on which each unit fires at least once inside its field.

    Unlike spatial information, sparsity or in/out rate contrast, this is not
    inflated by a low spike count: a cell whose map looks peaked only because it
    fired a handful of spikes in one place will have been active on very few
    laps. It is the reliability criterion that distinguishes a real place field
    from a noisy rate map.
    """
    ep = apply_speed_mask(laps, speed, min_speed)
    edges = _bin_edges(n_bins, track_cm)
    p = position.restrict(ep)
    pos_bin = np.clip(np.digitize(p.values, edges) - 1, 0, n_bins - 1)

    # Which lap each behavioural sample belongs to.
    lap_of_sample = np.searchsorted(laps.start, p.t, side="right") - 1
    n_laps = len(laps)

    out = np.full(len(units), np.nan)
    for k, uid in enumerate(units.index):
        if rates[k].max() <= 0:
            continue
        lo, hi = _field_extent(rates[k], frac)
        st = units[uid].restrict(ep).t
        if st.size == 0:
            out[k] = 0.0
            continue
        i = _spike_sample_index(st, p.t)
        inside = (pos_bin[i] >= lo) & (pos_bin[i] <= hi)
        # A lap only counts if the animal actually ran through the field on it.
        visited = np.unique(lap_of_sample[(pos_bin >= lo) & (pos_bin <= hi)])
        if visited.size == 0:
            continue
        active = np.unique(lap_of_sample[i[inside]])
        out[k] = np.isin(visited, active).mean()
    return out

# %% [markdown]
# Check the rate maps against pynapple's own routine before going further.

# %%
_laps = lap_epochs(S["run_ep"], S["lap_dir"], "rightward")
_ep = apply_speed_mask(_laps, S["speed"])
_mine, _, _, _ = rate_maps(S["units"], S["position"], _ep, S["dt"],
                          n_bins=S["n_bins"], track_cm=S["track_cm"], sigma=0)
_theirs = nap.compute_1d_tuning_curves(S["units"], S["position"],
                                       nb_bins=S["n_bins"],
                                       minmax=(0, S["track_cm"]), ep=_ep)
print("largest discrepancy with nap.compute_1d_tuning_curves: "
      f"{np.nanmax(np.abs(_theirs.values.T - _mine)):.2e} Hz")

# %% [markdown]
# ## The shuffle control
#
# The central worry with any rate map is that a peak could arise by chance,
# because the animal samples the track unevenly and neurons fire in bursts. The
# control used here circularly shifts each unit's spike train by a random lag
# *within the concatenated running epochs*. This preserves the exact number of
# spikes, the inter-spike-interval structure including bursting, and the occupancy
# map, while destroying the alignment between spiking and position. Repeating it a
# thousand times gives every cell its own null distribution of spatial
# information.
#
# This matters more than it might seem. Raw bits/spike is strongly inflated by a
# low spike count: a cell that fires twenty spikes on the track will have a spiky
# rate map and a high spatial information even with no spatial tuning at all.
# Comparing each cell only against its own null is what makes the test fair.

# %%
def shuffle_spatial_information(units, position, epochs, dt,
                                n_shuffles=N_SHUFFLES, n_bins=N_POS_BINS,
                                track_cm=TRACK_LENGTH_CM, sigma=SMOOTH_SIGMA_BINS,
                                circular=False, rng=None, progress=None):
    """Null distribution of spatial information from circularly shifted spikes.

    Each unit's spike train is circularly shifted by a random lag within the
    concatenated running epochs. This preserves the number of spikes, the
    inter-spike-interval structure (including bursting) and the occupancy map
    exactly, while destroying the alignment between spiking and position, so it
    is a conservative null for spatially modulated firing.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    edges = _bin_edges(n_bins, track_cm)
    p = position.restrict(epochs)
    pos_bin = np.clip(np.digitize(p.values, edges) - 1, 0, n_bins - 1)
    occ = np.bincount(pos_bin, minlength=n_bins).astype(float) * dt

    # Map every behavioural sample onto a continuous "elapsed running time" axis
    # so a circular shift never lands a spike in a reward pause.
    n_samp = p.t.size
    elapsed = np.arange(n_samp) * dt
    total = n_samp * dt

    # Position of each real spike on that axis.
    spike_idx = []
    for uid in units.index:
        st = units[uid].restrict(epochs).t
        spike_idx.append(_spike_sample_index(st, p.t) if st.size else
                         np.zeros(0, dtype=int))

    out = np.zeros((n_shuffles, len(units)))
    it = range(n_shuffles)
    if progress is not None:
        it = progress(it, desc="shuffles")
    for s in it:
        lag = rng.uniform(0.05 * total, 0.95 * total, size=len(spike_idx))
        for k, si_idx in enumerate(spike_idx):
            if si_idx.size == 0:
                continue
            shifted = np.mod(elapsed[si_idx] + lag[k], total)
            b = pos_bin[np.minimum((shifted / dt).astype(int), n_samp - 1)]
            c = np.bincount(b, minlength=n_bins)
            r = np.divide(c, occ, out=np.zeros(n_bins), where=occ > 0)
            if sigma:
                r = gaussian_filter1d(r, sigma,
                                      mode="wrap" if circular else "nearest")
            out[s, k] = spatial_information(r[None, :], occ)[0]
    return out

# %%
def split_half_stability(units, position, laps, speed, dt, n_bins=N_POS_BINS,
                         track_cm=TRACK_LENGTH_CM, sigma=SMOOTH_SIGMA_BINS,
                         circular=False, min_speed=MIN_SPEED_CM_S):
    """Correlation between rate maps built from odd and from even laps.

    An independent, shuffle-free check that a field is a stable property of the
    cell rather than a transient burst.
    """
    odd = apply_speed_mask(laps[np.arange(0, len(laps), 2)], speed, min_speed)
    even = apply_speed_mask(laps[np.arange(1, len(laps), 2)], speed, min_speed)
    r1, _, _, _ = rate_maps(units, position, odd, dt, n_bins=n_bins,
                            track_cm=track_cm, sigma=sigma, circular=circular)
    r2, _, _, _ = rate_maps(units, position, even, dt, n_bins=n_bins,
                            track_cm=track_cm, sigma=sigma, circular=circular)
    out = np.full(r1.shape[0], np.nan)
    for k in range(r1.shape[0]):
        if r1[k].std() > 0 and r2[k].std() > 0:
            out[k] = np.corrcoef(r1[k], r2[k])[0, 1]
    return out, r1, r2


def classify_place_cells(units, position, run_ep, lap_dir, speed, dt,
                         n_bins=N_POS_BINS, track_cm=TRACK_LENGTH_CM,
                         circular=False, min_laps=6,
                         n_shuffles=N_SHUFFLES, rng=None, progress=None):
    """Per-unit, per-direction place-field statistics plus a shuffle test.

    A unit is called a place cell if, in at least one running direction, it
    fires at least MIN_SPIKES_ON_TRACK spikes, reaches a peak rate of at least
    MIN_PEAK_RATE_HZ, and its spatial information exceeds the
    (1 - SHUFFLE_ALPHA) quantile of its own circular-shift null distribution.
    Interneurons (`cell_type == "inhibitory"`) are excluded by definition.
    """
    per_dir = {}
    for name in DIRECTIONS:
        laps = lap_epochs(run_ep, lap_dir, name)
        # On a circular maze the animal runs one way only, so the other
        # "direction" has no laps and must not produce an empty rate map.
        if len(laps) < min_laps:
            continue
        ep = apply_speed_mask(laps, speed)
        rates, occ, centres, counts = rate_maps(units, position, ep, dt,
                                                n_bins=n_bins, track_cm=track_cm,
                                                circular=circular)
        si = spatial_information(rates, occ)
        null = shuffle_spatial_information(units, position, ep, dt,
                                           n_bins=n_bins, track_cm=track_cm,
                                           circular=circular,
                                           n_shuffles=n_shuffles, rng=rng,
                                           progress=progress)
        pval = (null >= si[None, :]).sum(axis=0) / (n_shuffles + 1.0)
        # Raw spatial information is strongly biased by spike count: a cell with
        # few spikes has a noisy rate map and therefore high SI even with no
        # spatial tuning. Normalising by the cell's own shift-control null
        # removes that bias and is what should be compared across cells.
        si_z = (si - null.mean(axis=0)) / np.maximum(null.std(axis=0), 1e-9)
        stab, r_odd, r_even = split_half_stability(units, position, laps, speed,
                                                   dt, n_bins=n_bins,
                                                   track_cm=track_cm,
                                                   circular=circular)
        per_dir[name] = dict(
            rates=rates, occupancy=occ, centres=centres, counts=counts,
            si=si, si_z=si_z, null=null, pval=pval, stability=stab,
            rates_odd=r_odd, rates_even=r_even,
            peak_rate=rates.max(axis=1),
            peak_pos=centres[np.argmax(rates, axis=1)],
            mean_rate=(rates * (occ / occ.sum())).sum(axis=1),
            sparsity=sparsity(rates, occ),
            width=field_width_cm(rates, centres),
            contrast=field_contrast(rates),
            participation=lap_participation(units, position, laps, speed, dt,
                                            rates, n_bins=n_bins,
                                            track_cm=track_cm),
            n_spikes=counts.sum(axis=1),
            epochs=ep, laps=laps,
        )

    rows = []
    for k, uid in enumerate(units.index):
        rec = {"unit": int(uid),
               "cell_type": units.metadata["cell_type"].iloc[k],
               "location": units.metadata["location"].iloc[k]}
        passed = []
        for name, d in per_dir.items():
            rec[f"si_{name}"] = d["si"][k]
            rec[f"si_z_{name}"] = d["si_z"][k]
            rec[f"p_{name}"] = d["pval"][k]
            rec[f"peak_rate_{name}"] = d["peak_rate"][k]
            rec[f"peak_pos_{name}"] = d["peak_pos"][k]
            rec[f"mean_rate_{name}"] = d["mean_rate"][k]
            rec[f"sparsity_{name}"] = d["sparsity"][k]
            rec[f"width_{name}"] = d["width"][k]
            rec[f"contrast_{name}"] = d["contrast"][k]
            rec[f"participation_{name}"] = d["participation"][k]
            rec[f"stability_{name}"] = d["stability"][k]
            rec[f"n_spikes_{name}"] = d["n_spikes"][k]
            passed.append(
                (d["pval"][k] < SHUFFLE_ALPHA)
                and (d["n_spikes"][k] >= MIN_SPIKES_ON_TRACK)
                and (d["peak_rate"][k] >= MIN_PEAK_RATE_HZ)
            )
        rec["is_place_cell"] = bool(np.any(passed)) and rec["cell_type"] == "excitatory"
        rows.append(rec)
    stats = pd.DataFrame(rows).set_index("unit")
    return stats, per_dir


def field_table(units, stats, per_dir):
    """One row per (cell, direction) that qualified as a place field.

    Field properties (peak rate, width, sparsity, stability) are only meaningful
    for the direction in which the cell actually has a field, so they must not
    be pooled over the direction in which the same cell is silent.
    """
    rows = []
    for name, d in per_dir.items():
        ok = ((d["pval"] < SHUFFLE_ALPHA)
              & (d["n_spikes"] >= MIN_SPIKES_ON_TRACK)
              & (d["peak_rate"] >= MIN_PEAK_RATE_HZ))
        for k in np.flatnonzero(ok):
            uid = int(units.index[k])
            if not stats.loc[uid, "is_place_cell"]:
                continue          # excludes interneurons
            rows.append(dict(unit=uid, direction=name, k=k,
                             si=d["si"][k], si_z=d["si_z"][k], p=d["pval"][k],
                             peak_rate=d["peak_rate"][k],
                             peak_pos=d["peak_pos"][k],
                             mean_rate=d["mean_rate"][k],
                             width=d["width"][k],
                             sparsity=d["sparsity"][k],
                             contrast=d["contrast"][k],
                             participation=d["participation"][k],
                             stability=d["stability"][k],
                             n_spikes=d["n_spikes"][k]))
    return pd.DataFrame(rows)


def best_direction(per_dir):
    """Per unit: index of the direction with the strongest spatial tuning."""
    names = list(per_dir)
    z = np.stack([per_dir[n]["si_z"] for n in names])
    return np.array(names)[np.argmax(z, axis=0)], z.max(axis=0)

# %%
stats, per_dir = classify_place_cells(
    S["units"], S["position"], S["run_ep"], S["lap_dir"], S["speed"], S["dt"],
    n_bins=S["n_bins"], track_cm=S["track_cm"],
    circular=S["maze_type"] == "circular", n_shuffles=N_SHUFFLES, progress=tqdm)
fields = field_table(S["units"], stats, per_dir)

n_exc = int((stats["cell_type"] == "excitatory").sum())
print(f"{stats['is_place_cell'].sum()} place cells out of {n_exc} excitatory "
      f"units ({stats['is_place_cell'].sum() / n_exc:.0%}), "
      f"{len(fields)} fields in total")
print(fields[["si", "peak_rate", "width", "participation", "stability"]]
      .describe().loc[["50%", "mean"]].round(2))

# %% [markdown]
# ## Bayesian decoding and the LFP
#
# Decoding turns the single-cell result into a claim about the population: if
# these cells really carry position, an observer with access only to their spikes
# should be able to recover where the animal is. Tuning curves are fitted to the
# odd laps and used to decode the even laps, and vice versa, so no lap ever
# contributes to both the encoding model and the test set.

# %%
def _tc_xarray(rates, centres, units):
    """Package rate maps as the xarray that nap.decode_bayes expects."""
    return xr.DataArray(rates, dims=("unit", "position"),
                        coords={"unit": list(units.index), "position": centres})


def crossvalidated_decoding(units, position, laps, speed, dt, bin_size=0.25,
                            n_bins=N_POS_BINS, track_cm=TRACK_LENGTH_CM,
                            sigma=SMOOTH_SIGMA_BINS, circular=False,
                            min_speed=MIN_SPEED_CM_S):
    """Odd/even-lap cross-validated Bayesian decoding of position.

    `laps` must be the *lap-level* IntervalSet for one travel direction. The
    odd/even split is applied to whole laps before the running-speed mask so
    that no lap contributes to both the encoding model and the test set.
    """
    edges = _bin_edges(n_bins, track_cm)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dec_t, dec_v, true_v = [], [], []
    for parity in (0, 1):
        test_laps = laps[np.arange(parity, len(laps), 2)]
        train_laps = laps[np.arange(1 - parity, len(laps), 2)]
        if len(test_laps) == 0 or len(train_laps) == 0:
            continue
        train = apply_speed_mask(train_laps, speed, min_speed)
        test = apply_speed_mask(test_laps, speed, min_speed)
        rates, occ, _, _ = rate_maps(units, position, train, dt,
                                     n_bins=n_bins, track_cm=track_cm,
                                     sigma=sigma, circular=circular)
        decoded, _ = nap.decode_bayes(_tc_xarray(rates, centres, units),
                                      units, test, bin_size)
        truth = position.restrict(test).interpolate(decoded)
        ok = np.isfinite(truth.values)
        dec_t.append(decoded.t[ok])
        dec_v.append(decoded.values[ok])
        true_v.append(truth.values[ok])
    if not dec_t:
        empty = np.zeros(0)
        return empty, empty, empty
    order = np.argsort(np.concatenate(dec_t))
    return (np.concatenate(dec_t)[order],
            np.concatenate(dec_v)[order],
            np.concatenate(true_v)[order])


def decode_lap_block(units, position, laps, speed, dt, block, bin_size=0.25,
                     n_bins=N_POS_BINS, track_cm=TRACK_LENGTH_CM,
                     sigma=SMOOTH_SIGMA_BINS, circular=False,
                     min_speed=MIN_SPEED_CM_S):
    """Decode a contiguous block of held-out laps and return the posterior.

    `block` is an array of lap indices. The encoding model is built from every
    other lap in `laps`, so the decoded block is genuinely held out.
    """
    edges = _bin_edges(n_bins, track_cm)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rest = np.setdiff1d(np.arange(len(laps)), block)
    train = apply_speed_mask(laps[rest], speed, min_speed)
    test_laps = laps[block]
    test = apply_speed_mask(test_laps, speed, min_speed)
    rates, _, _, _ = rate_maps(units, position, train, dt, n_bins=n_bins,
                               track_cm=track_cm, sigma=sigma, circular=circular)
    decoded, proba = nap.decode_bayes(_tc_xarray(rates, centres, units),
                                      units, test, bin_size)
    truth = position.restrict(test).interpolate(decoded)
    return dict(decoded=decoded, proba=proba, truth=truth, centres=centres,
                test_laps=test_laps)

# %%
def lfp_series(nwbfile):
    return nwbfile.processing["ecephys"]["LFP"]["LFP"]


def pick_lfp_channel(nwbfile, t_start, duration=20.0, stride=8):
    """Channel with the most theta-band power, a proxy for the CA1 cell layer.

    Only every `stride`-th channel is examined so that the search reads a
    handful of HDF5 chunks rather than the whole 128-channel array.
    """
    es = lfp_series(nwbfile)
    fs = es.rate
    i0 = int(t_start * fs)
    i1 = i0 + int(duration * fs)
    best, best_pow = None, -np.inf
    for ch in range(0, es.data.shape[1], stride):
        x = np.asarray(es.data[i0:i1, ch], dtype=float) * es.conversion
        f, p = _welch(x, fs)
        band = p[(f >= 6) & (f <= 12)].sum() / p[(f >= 1) & (f <= 100)].sum()
        if band > best_pow:
            best, best_pow = ch, band
    return best, best_pow


def _welch(x, fs):
    from scipy.signal import welch
    return welch(x - x.mean(), fs=fs, nperseg=min(len(x), int(2 * fs)))


def lfp_snippet(nwbfile, t_start, t_stop, channel):
    """Tsd of one LFP channel in volts over [t_start, t_stop]."""
    es = lfp_series(nwbfile)
    fs = es.rate
    i0, i1 = int(t_start * fs), int(t_stop * fs)
    x = np.asarray(es.data[i0:i1, channel], dtype=float) * es.conversion
    t = es.starting_time + np.arange(i0, i1) / fs
    return nap.Tsd(t=t, d=x)

# %% [markdown]
# ## Figures

# %%
DIR_COLOR = {"rightward": "#1b6ca8", "leftward": "#d1495b"}


def _peak_order(per_dir, direction, idx):
    """Indices `idx` reordered by place-field peak position in `direction`."""
    peaks = per_dir[direction]["peak_pos"][idx]
    return idx[np.argsort(peaks)]


def _norm_map(rates):
    return rates / np.maximum(rates.max(axis=1, keepdims=True), 1e-9)


MAZE_COLOR = {"linear": "#1b6ca8", "circular": "#8ac926"}


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# %% [markdown]
# ### Figure 1 — the raw data
#
# The recording, the behaviour and the spikes, with nothing computed except the
# ordering of the cells. Panel (d) is the phenomenon in its rawest form: with the
# place cells sorted by the position of their fields, a single traversal of the
# track drives a diagonal sweep of activity across the ensemble.

# %%
def fig_session_overview(S, stats, per_dir, lfp_channel, fname, pad=1.5):
    units = S["units"]
    pc_mask = stats["is_place_cell"].values
    pc_idx = np.flatnonzero(pc_mask)
    ref_dir = "rightward" if "rightward" in per_dir else list(per_dir)[0]
    order = _peak_order(per_dir, ref_dir, pc_idx)

    # Zoom on a single brisk traversal: use the speed-masked running epochs so
    # the window is not dominated by the pause at the reward port.
    runs = per_dir[ref_dir]["epochs"]
    dur = runs.end - runs.start
    long_enough = np.flatnonzero(dur > np.percentile(dur, 75))
    j = int(long_enough[np.argmin(np.abs(dur[long_enough]
                                         - np.median(dur[long_enough])))])
    t0, t1 = float(runs.start[j]) - pad, float(runs.end[j]) + pad
    zoom = nap.IntervalSet(start=t0, end=t1)

    fig = plt.figure(figsize=(14, 13.5))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.7, 1.7, 0.5],
                  hspace=0.62, wspace=0.24)

    # (a) 2D trajectory ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    p2 = S["position_2d"].restrict(S["maze_ep"])
    ax.plot(p2["x"].values, p2["y"].values, color="0.8", lw=0.4, alpha=0.8)
    for name in per_dir:
        ep_d = per_dir[name]["epochs"]
        for i in range(len(ep_d)):
            seg = S["position_2d"].restrict(ep_d[i])
            ax.plot(seg["x"].values, seg["y"].values, color=DIR_COLOR[name],
                    lw=0.5, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"(a) Trajectory, {S['maze_name']}\n"
                 f"{S['subject_id']}, session {S['session_id']}")

    # (b) session timeline ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ep = S["epochs"]
    colors = {"PREEpoch": "#8ecae6", "MazeEpoch": "#ffb703", "POSTEpoch": "#adb5bd"}
    for i in range(len(ep)):
        lab = ep.metadata["label"].iloc[i]
        ax.barh(0, (ep.end[i] - ep.start[i]) / 60, left=ep.start[i] / 60,
                height=0.35, color=colors.get(lab, "0.6"),
                label=lab.replace("Epoch", ""))
    ax.set_yticks([])
    ax.set_ylim(-1.7, 1.1)
    ax.set_xlabel("time from session start (min)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.set_title("(b) Session structure: sleep - track - sleep")
    ax.text(0.0, 0.30,
            f"{len(units)} sorted units "
            f"({(stats.cell_type == 'excitatory').sum()} excitatory, "
            f"{(stats.cell_type == 'inhibitory').sum()} inhibitory)\n"
            f"{len(S['run_ep'])} track traversals, {pc_mask.sum()} place cells",
            transform=ax.transAxes, va="top", fontsize=8.5)

    # (c) linearised position over the whole maze epoch ----------------------
    ax = fig.add_subplot(gs[1, :])
    pos = S["position"]
    ax.plot(pos.t, pos.values, color="0.8", lw=0.5)
    for name in per_dir:
        ep_d = per_dir[name]["epochs"]
        for i in range(len(ep_d)):
            seg = pos.restrict(ep_d[i])
            ax.plot(seg.t, seg.values, color=DIR_COLOR[name], lw=1.0)
    ax.axvspan(t0, t1, color="gold", alpha=0.6, zorder=0)
    ax.set_xlim(float(S["maze_ep"].start[0]), float(S["maze_ep"].end[0]))
    ax.set_ylim(-10, S["track_cm"] + 60)
    ax.set_ylabel("position (cm)")
    ax.set_xlabel("time (s)")
    ax.set_title("(c) Linearised position over the maze epoch; coloured segments "
                 "are the running laps kept for the rate maps")
    ax.legend(handles=[Line2D([], [], color=DIR_COLOR[n], lw=2, label=n)
                       for n in per_dir]
              + [Line2D([], [], color="gold", lw=6, label="window in (d)")],
              loc="upper right", ncol=3, fontsize=8)

    # (d) raster of place cells ordered by field position --------------------
    ax = fig.add_subplot(gs[2, :])
    for row, k in enumerate(order):
        uid = units.index[k]
        st = units[uid].restrict(zoom).t - t0
        ax.plot(st, np.full(st.size, row), "|", color="k", ms=5, mew=0.9)
    ax.set_ylim(-2, len(order) + 1)
    ax.set_ylabel(f"place cell, sorted by field position (n={len(order)})")
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("time from start of window (s)")
    ax.set_title("(d) Raw spiking during one traversal: activity sweeps through "
                 "the ensemble in field order as the animal advances")
    axp = ax.twinx()
    seg = pos.restrict(zoom)
    axp.plot(seg.t - t0, seg.values, color="#ffb703", lw=2.5)
    axp.set_ylim(-5, S["track_cm"] + 5)
    axp.set_ylabel("position (cm)", color="#c98900")
    axp.tick_params(axis="y", colors="#c98900")
    axp.spines["right"].set_visible(True)
    axp.spines["right"].set_color("#c98900")

    # (e) LFP ----------------------------------------------------------------
    ax = fig.add_subplot(gs[3, :])
    lfp = lfp_snippet(S["nwbfile"], t0, t1, lfp_channel)
    ax.plot(lfp.t - t0, lfp.values * 1e3, color="#2a9d8f", lw=0.7)
    ax.set_xlim(0, t1 - t0)
    ax.set_xlabel("time from start of window (s)")
    ax.set_ylabel("LFP (mV)")
    ax.set_title(f"(e) Simultaneous CA1 local field potential (channel "
                 f"{lfp_channel}); the running-speed theta rhythm is visible")

    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
lfp_channel, theta_ratio = pick_lfp_channel(S["nwbfile"],
                                            float(S["run_ep"].start[0]))
print(f"LFP channel {lfp_channel}: {theta_ratio:.0%} of 1-100 Hz power in the "
      f"6-12 Hz theta band")
fig_session_overview(S, stats, per_dir, lfp_channel, "fig01_session_overview.png")

# %% [markdown]
# ### Figure 2 — individual place cells
#
# Each pair of panels shows one cell: above, the position of every spike it fired,
# lap by lap; below, the rate map for each direction. The spikes cluster into a
# horizontal band, present on nearly every lap, that occupies a small part of the
# track. Examples were chosen for compactness, stability and firing rate rather
# than for spatial information, which as noted above is not a fair ranking across
# cells with different spike counts.

# %%
def select_examples(S, fields, n_examples=8, max_sparsity=0.4,
                    min_stability=0.8, min_peak_rate=3.0):
    """Compact, stable, well-driven fields spread along the length of the track.

    Selection deliberately does not rank on spatial information: raw bits/spike
    is inflated by low spike counts and the z-scored version rewards high firing
    rates, so neither picks out the textbook single-peaked field. Sparsity,
    split-half stability and peak rate do.
    """
    good = fields[(fields["sparsity"] < max_sparsity)
                  & (fields["stability"] > min_stability)
                  & (fields["peak_rate"] > min_peak_rate)]
    if len(good) < n_examples:            # fall back to a looser cut
        good = fields.nsmallest(max(n_examples * 3, 12), "sparsity")
    quality = (1 - good["sparsity"].values) * good["stability"].values
    peaks = good["peak_pos"].values
    chosen = []
    for target in np.linspace(0.08, 0.92, n_examples) * S["track_cm"]:
        score = np.abs(peaks - target) - 25.0 * quality
        for c in np.argsort(score):
            if int(good.iloc[c]["k"]) not in chosen:
                chosen.append(int(good.iloc[c]["k"]))
                break
    chosen = np.array(chosen)
    peak_of_chosen = np.array([peaks[list(good["k"].values).index(k)]
                               for k in chosen])
    return chosen[np.argsort(peak_of_chosen)]


def fig_example_cells(S, stats, per_dir, fields, fname, n_examples=8):
    units = S["units"]
    chosen = select_examples(S, fields, n_examples)

    ncol = 4
    nrow = int(np.ceil(len(chosen) / ncol))
    fig = plt.figure(figsize=(4.1 * ncol, 5.0 * nrow))
    gs = GridSpec(2 * nrow, ncol, figure=fig, height_ratios=[2.0, 1.0] * nrow,
                  hspace=0.75, wspace=0.38)

    for m, k in enumerate(chosen):
        r, c = divmod(m, ncol)
        uid = units.index[k]
        ax_r = fig.add_subplot(gs[2 * r, c])
        ax_t = fig.add_subplot(gs[2 * r + 1, c], sharex=ax_r)

        lap_offset = 0
        for j, name in enumerate(per_dir):
            laps = per_dir[name]["laps"]
            block_start = lap_offset
            for i in range(len(laps)):
                st = units[uid].restrict(laps[i])
                if len(st):
                    xp = st.value_from(S["position"]).values
                    ax_r.plot(xp, np.full(xp.size, lap_offset), "|",
                              color=DIR_COLOR[name], ms=3.5, mew=0.8)
                lap_offset += 1
            ax_r.text(0.99, (block_start + lap_offset) / 2.0, name[0].upper() + " ",
                      color=DIR_COLOR[name], fontsize=8, ha="right", va="center",
                      transform=ax_r.get_yaxis_transform(), clip_on=False)
            if j < len(per_dir) - 1:
                ax_r.axhline(lap_offset - 0.5, color="0.6", lw=0.8)
        ax_r.set_ylim(-1, lap_offset)
        ax_r.set_ylabel("lap")
        ax_r.set_title(f"unit {uid}\n"
                       + "   ".join(f"{n[0].upper()} {per_dir[n]['si'][k]:.2f}"
                                    for n in per_dir)
                       + " bits/spike", fontsize=9)
        plt.setp(ax_r.get_xticklabels(), visible=False)

        for name in per_dir:
            d = per_dir[name]
            ax_t.plot(d["centres"], d["rates"][k], color=DIR_COLOR[name], lw=1.6,
                      label=name)
        ax_t.set_xlabel("position (cm)")
        ax_t.set_ylabel("rate (Hz)")
        ax_t.set_xlim(0, S["track_cm"])
        if m == 0 and len(per_dir) > 1:
            ax_t.legend(fontsize=7, loc="upper center", ncol=2)

    fig.suptitle("Example CA1 place cells, ordered by field position along the "
                 "track\nTop of each pair: position of every spike, lap by lap "
                 "(blue = rightward laps, red = leftward laps).  "
                 "Bottom: occupancy-normalised rate map.", y=0.995, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
fig_example_cells(S, stats, per_dir, fields, "fig02_example_place_cells.png")

# %% [markdown]
# ### Figure 3 — the population
#
# Stacking the normalised rate maps and sorting by field position shows that the
# fields tile the track continuously. Sorting the leftward maps by the *rightward*
# peak scrambles the diagonal, which is the directional remapping characteristic
# of one-dimensional tracks: the same cell generally has unrelated fields in the
# two directions of travel.

# %%
def fig_population_maps(S, stats, per_dir, fname):
    pc_idx = np.flatnonzero(stats["is_place_cell"].values)
    dirs = list(per_dir)
    ref = dirs[0]
    order_ref = _peak_order(per_dir, ref, pc_idx)

    panels = [(d, _peak_order(per_dir, d, pc_idx), f"sorted by its own peak")
              for d in dirs]
    if len(dirs) > 1:
        other = dirs[1]
        panels.append((other, order_ref, f"sorted by the {ref} peak"))

    ncol = len(panels) + (1 if len(dirs) > 1 else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 5.4))
    axes = np.atleast_1d(axes)
    letters = "abcdefg"
    for i, (d, order, sub) in enumerate(panels):
        ax = axes[i]
        im = ax.imshow(_norm_map(per_dir[d]["rates"][order]), aspect="auto",
                       origin="lower", cmap="viridis", vmin=0, vmax=1,
                       extent=[0, S["track_cm"], 0, len(order)])
        ax.set_xlabel("position (cm)")
        ax.set_ylabel("place cell #")
        ax.set_title(f"({letters[i]}) {d} runs,\n{sub}", fontsize=9.5)
    fig.colorbar(im, ax=axes[len(panels) - 1], label="rate / peak rate",
                 fraction=0.06, pad=0.04)

    if len(dirs) > 1:
        ax = axes[-1]
        a, b = dirs[0], dirs[1]
        pa, pb = per_dir[a]["peak_pos"][pc_idx], per_dir[b]["peak_pos"][pc_idx]
        ax.scatter(pa, pb, s=20, color="0.25", alpha=0.8)
        lim = [0, S["track_cm"]]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"peak position, {a} (cm)")
        ax.set_ylabel(f"peak position, {b} (cm)")
        ax.set_title(f"({letters[len(panels)]}) the two directional maps\n"
                     f"are largely independent (r = {np.corrcoef(pa, pb)[0,1]:.2f})",
                     fontsize=9.5)

    fig.suptitle("Population rate maps: place fields tile the whole track, and "
                 "the ordering is specific to the direction of travel", y=1.02,
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
fig_population_maps(S, stats, per_dir, "fig03_population_maps.png")

# %% [markdown]
# ### Figure 4 — is the tuning real?

# %%
def fig_spatial_information(S, stats, per_dir, fname):
    exc = (stats["cell_type"] == "excitatory").values
    inh = (stats["cell_type"] == "inhibitory").values
    pc = stats["is_place_cell"].values
    ref = list(per_dir)[0]
    d = per_dir[ref]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    ax = axes[0, 0]
    bins = np.linspace(0, max(3.0, np.nanpercentile(d["si"][exc], 99.5)), 40)
    ax.hist(d["null"][:, exc].ravel(), bins=bins, density=True, color="0.7",
            label="circularly shifted spikes (null)")
    ax.hist(d["si"][exc], bins=bins, density=True, histtype="step", lw=2,
            color="#1b6ca8", label="observed")
    ax.set_xlabel("spatial information (bits/spike)")
    ax.set_ylabel("density")
    ax.set_title(f"(a) Observed spatial information far exceeds the\n"
                 f"shift-control null ({ref} runs, excitatory cells)")
    ax.legend()

    ax = axes[0, 1]
    thr = np.percentile(d["null"], 100 * (1 - SHUFFLE_ALPHA), axis=0)
    active = (d["n_spikes"] >= MIN_SPIKES_ON_TRACK) & \
             (d["peak_rate"] >= MIN_PEAK_RATE_HZ)
    sig = d["pval"] < SHUFFLE_ALPHA
    ax.scatter(thr[active & exc & ~sig], d["si"][active & exc & ~sig], s=16,
               color="0.6", label="not significant")
    ax.scatter(thr[active & exc & sig], d["si"][active & exc & sig], s=18,
               color="#1b6ca8", label="significant (p < 0.01)")
    ax.scatter(thr[active & inh], d["si"][active & inh], s=26, color="#d1495b",
               marker="^", label="interneuron")
    lim = [0, max(thr[active].max(), d["si"][active].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"per-cell {100*(1-SHUFFLE_ALPHA):.0f}th percentile of null "
                  "(bits/spike)")
    ax.set_ylabel("observed spatial information (bits/spike)")
    ax.set_title(f"(b) Every cell is tested against its own null;\n"
                 f"points above the diagonal are significant\n"
                 f"({(~active).sum()} sparsely firing cells not shown)",
                 fontsize=9.5)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 0]
    eligible = exc & sig & active
    best = np.argmax(np.where(eligible, d["si"], -np.inf))
    ax.hist(d["null"][:, best], bins=40, color="0.7", label="null")
    ax.axvline(d["si"][best], color="#1b6ca8", lw=2)
    ax.text(d["si"][best], ax.get_ylim()[1] * 0.92,
            f"  observed\n  p < {max(d['pval'][best], 1/d['null'].shape[0]):.3f}",
            color="#1b6ca8", va="top")
    ax.set_xlim(0, max(d["si"][best] * 1.15, d["null"][:, best].max()))
    ax.set_xlabel("spatial information (bits/spike)")
    ax.set_ylabel(f"count (of {d['null'].shape[0]} shuffles)")
    ax.set_title(f"(c) Null distribution for the most informative\n"
                 f"place cell (unit {S['units'].index[best]})")

    ax = axes[1, 1]
    # Observed spatial information is only interpretable next to each cell's own
    # null: a sparsely firing cell has a noisy rate map and therefore a high
    # value under both. Pairing the two makes the comparison honest.
    null_med = np.median(d["null"], axis=0)
    pos, ticks, labels = [], [], []
    for i, (m, lab, col) in enumerate(((pc, "place cells", "#1b6ca8"),
                                       (exc & ~pc, "other\nexcitatory", "0.6"),
                                       (inh, "interneurons", "#d1495b"))):
        if not m.sum():
            continue
        x = 2 * i
        for dx, vals, fc, ec in ((-0.32, d["si"][m], col, col),
                                 (0.32, null_med[m], "white", col)):
            bp = ax.boxplot([vals], positions=[x + dx], widths=0.55,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(fc)
            bp["boxes"][0].set_edgecolor(ec)
            bp["boxes"][0].set_alpha(0.8)
            ax.scatter(np.random.default_rng(1).normal(x + dx, 0.06, vals.size),
                       vals, s=8, color=col, alpha=0.6, zorder=3)
        ticks.append(x)
        labels.append(f"{lab}\n(n={m.sum()})")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-1.2, 2 * 2 + 1.2)
    ax.set_ylabel("spatial information (bits/spike)")
    ax.legend(handles=[Patch(facecolor="0.35", edgecolor="0.35", label="observed"),
                       Patch(facecolor="white", edgecolor="0.35",
                             label="median of that cell's own null")],
              fontsize=8, loc="upper right")
    ax.set_title("(d) Observed information beside each cell's own null.\n"
                 "Place cells exceed theirs several-fold; the high raw values\n"
                 "among other excitatory cells are matched by a high null.",
                 fontsize=9.5)

    fig.suptitle("Place fields are statistically reliable, not a by-product of "
                 "uneven sampling of the track", y=0.98, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
fig_spatial_information(S, stats, per_dir, "fig04_spatial_information.png")

# %% [markdown]
# ### Figure 5 — what the fields look like
#
# Panel (c) is the one that actually separates place cells from cells whose rate
# map merely happens to be peaked: a real field produces at least one spike on the
# large majority of the passes through it, whereas an incidental peak does not.

# %%
def fig_field_properties(S, stats, per_dir, fields, fname):
    """`fields` is the one-row-per-(cell, direction) table from field_table."""
    pc = stats["is_place_cell"].values
    exc = (stats["cell_type"] == "excitatory").values
    dirs = list(per_dir)
    best_name, best_z = best_direction(per_dir)

    def per_cell(key, mask):
        """Value of `key` in each cell's best-tuned direction."""
        v = np.array([per_dir[n][key][k] for k, n in enumerate(best_name)])
        return v[mask]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    fig.subplots_adjust(hspace=0.5, wspace=0.33)

    ax = axes[0, 0]
    v = fields["peak_rate"].values
    ax.hist(v, bins=np.linspace(0, np.percentile(v, 99), 30), color="#1b6ca8",
            alpha=0.85)
    ax.axvline(np.median(v), color="k", ls="--", lw=1)
    ax.set_xlabel("in-field peak rate (Hz)")
    ax.set_ylabel("number of fields")
    ax.set_title(f"(a) Peak firing rate\nmedian {np.median(v):.1f} Hz")

    ax = axes[0, 1]
    v = fields["width"].values
    v = v[v > 0]
    ax.hist(v, bins=25, color="#1b6ca8", alpha=0.85)
    ax.axvline(np.median(v), color="k", ls="--", lw=1)
    ax.set_xlabel("field width at half maximum (cm)")
    ax.set_ylabel("number of fields")
    ax.set_title(f"(b) Field width\nmedian {np.median(v):.0f} cm "
                 f"({100*np.median(v)/S['track_cm']:.0f}% of the track)")

    ax = axes[0, 2]
    other = per_cell("participation", exc & ~pc)
    other = other[np.isfinite(other)]
    mine = fields["participation"].values
    mine = mine[np.isfinite(mine)]
    bins = np.linspace(0, 1, 21)
    ax.hist(other, bins=bins, color="0.65", alpha=0.9,
            label=f"other excitatory (median {np.median(other):.0%})")
    ax.hist(mine, bins=bins, color="#1b6ca8", alpha=0.8,
            label=f"place fields (median {np.median(mine):.0%})")
    ax.set_xlabel("fraction of laps with a spike inside the field")
    ax.set_ylabel("count")
    ax.set_title("(c) A place field fires on nearly every pass;\n"
                 "an incidentally peaked map does not")
    ax.legend(fontsize=7.5, loc="upper center")

    ax = axes[1, 0]
    for m, lab, col in ((pc, "place cells", "#1b6ca8"),
                        (exc & ~pc, "other excitatory", "0.65")):
        if m.sum():
            v = per_cell("stability", m)
            v = v[np.isfinite(v)]
            ax.hist(v, bins=np.linspace(-1, 1, 30), alpha=0.75, color=col,
                    label=f"{lab} (median {np.median(v):.2f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("odd-lap vs even-lap map correlation")
    ax.set_ylabel("number of cells")
    ax.set_title("(d) Fields are stable across independent laps")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    for d in dirs:
        sub = fields[fields["direction"] == d]
        ax.hist(sub["peak_pos"], bins=np.linspace(0, S["track_cm"], 21),
                histtype="step", lw=2, color=DIR_COLOR[d],
                label=f"{d} (n={len(sub)})")
    ax.set_xlabel("field peak position (cm)")
    ax.set_ylabel("number of fields")
    ax.set_title("(e) Fields cover the whole track, with the\n"
                 "usual over-representation of the reward ends")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    mid = fields[(fields["peak_pos"] > 0.15 * S["track_cm"])
                 & (fields["peak_pos"] < 0.85 * S["track_cm"])]
    row = mid.loc[mid["si_z"].idxmax()]
    d0, k = per_dir[row["direction"]], int(row["k"])
    ax.plot(d0["centres"], d0["rates_odd"][k], color="#1b6ca8", lw=1.8,
            label="odd laps")
    ax.plot(d0["centres"], d0["rates_even"][k], color="#f4a261", lw=1.8,
            label="even laps")
    ax.set_xlabel("position (cm)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"(f) Independent halves of the data give the\n"
                 f"same field (unit {int(row['unit'])}, {row['direction']}, "
                 f"r = {row['stability']:.2f})")
    ax.legend()

    fig.suptitle("Properties of the identified place fields "
                 f"({len(fields)} fields from {fields['unit'].nunique()} cells)",
                 y=0.98, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
fig_field_properties(S, stats, per_dir, fields, "fig05_field_properties.png")

# %% [markdown]
# ### Figure 6 — decoding position from the population

# %%
def decoding_results(S, stats, per_dir, bin_size=0.25, rng=None):
    """Cross-validated decoding for each direction, plus an ensemble-size sweep."""
    rng = np.random.default_rng(0) if rng is None else rng
    pc_ids = stats.index[stats["is_place_cell"]].values
    ensemble = S["units"][list(pc_ids)]
    out = {}
    for name in per_dir:
        laps = per_dir[name]["laps"]
        t, dec, true = crossvalidated_decoding(
            ensemble, S["position"], laps, S["speed"], S["dt"],
            bin_size=bin_size, n_bins=S["n_bins"], track_cm=S["track_cm"],
            circular=S["maze_type"] == "circular")
        out[name] = dict(t=t, decoded=dec, true=true, error=np.abs(dec - true))
    # chance: pair each true position with a decoded value from another time bin
    allerr = np.concatenate([v["error"] for v in out.values()])
    alltrue = np.concatenate([v["true"] for v in out.values()])
    alldec = np.concatenate([v["decoded"] for v in out.values()])
    chance = np.concatenate([np.abs(rng.permutation(alldec) - alltrue)
                             for _ in range(20)])

    sizes = [n for n in (2, 5, 10, 20, 40, 80) if n <= 0.7 * len(pc_ids)]
    sizes.append(len(pc_ids))
    sweep = {}
    ref = list(per_dir)[0]
    laps = per_dir[ref]["laps"]
    for n in sizes:
        errs = []
        for _ in range(8 if n < len(pc_ids) else 1):
            sub = rng.choice(pc_ids, size=n, replace=False)
            _, dcd, tru = crossvalidated_decoding(
                S["units"][list(sub)], S["position"], laps, S["speed"], S["dt"],
                bin_size=bin_size, n_bins=S["n_bins"], track_cm=S["track_cm"],
                circular=S["maze_type"] == "circular")
            errs.append(np.median(np.abs(dcd - tru)))
        sweep[n] = np.array(errs)

    # A held-out block of consecutive laps, kept with its posterior for display.
    n_block = min(6, max(2, len(laps) // 4))
    start = max(0, len(laps) // 2 - n_block // 2)
    block = decode_lap_block(
        ensemble, S["position"], laps, S["speed"], S["dt"],
        np.arange(start, start + n_block), bin_size=bin_size,
        n_bins=S["n_bins"], track_cm=S["track_cm"],
        circular=S["maze_type"] == "circular")

    return dict(per_dir=out, error=allerr, chance=chance, sweep=sweep,
                block=block, ref=ref, n_cells=len(pc_ids), bin_size=bin_size)


def fig_decoding(S, stats, per_dir, dec, fname):
    fig = plt.figure(figsize=(15, 9.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.42,
                  height_ratios=[1.0, 0.95])
    ref = dec["ref"]
    blk = dec["block"]

    # (a) posterior over a held-out block of laps, laps drawn side by side so
    # that the pauses between traversals are not spanned by a line.
    ax = fig.add_subplot(gs[0, :])
    laps = blk["test_laps"]
    edges = np.linspace(0, S["track_cm"], S["n_bins"] + 1)
    x_off = 0.0
    ticks, labels = [], []
    for i in range(len(laps)):
        dsub = blk["decoded"].restrict(laps[i])
        if len(dsub) == 0:
            continue
        psub = blk["proba"].restrict(laps[i])
        tsub = blk["truth"].restrict(laps[i])
        rel = dsub.t - float(laps[i].start[0])
        dt_bin = dec["bin_size"]
        tedges = np.concatenate([rel - dt_bin / 2, [rel[-1] + dt_bin / 2]])
        pm = ax.pcolormesh(x_off + tedges, edges, np.asarray(psub).T,
                           cmap="magma", vmin=0, vmax=0.35, shading="flat")
        ax.plot(x_off + rel, tsub.values, color="#8ecae6", lw=2.2)
        ticks.append(x_off + rel.mean())
        labels.append(f"lap {i+1}")
        x_off += rel[-1] + 2 * dt_bin
        ax.axvline(x_off - dt_bin, color="0.5", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(0, x_off - dt_bin)
    ax.set_ylim(0, S["track_cm"])
    ax.set_ylabel("position (cm)")
    fig.colorbar(pm, ax=ax, fraction=0.02, pad=0.01,
                 label="posterior probability")
    ax.set_title(f"(a) Posterior probability over position, decoded from "
                 f"{dec['n_cells']} place cells in {int(dec['bin_size']*1000)} ms "
                 f"bins, on {len(laps)} consecutive {ref} laps held out of the "
                 f"encoding model\n"
                 f"(blue line: the animal's true position)", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    tr = np.concatenate([v["true"] for v in dec["per_dir"].values()])
    dc = np.concatenate([v["decoded"] for v in dec["per_dir"].values()])
    edges = np.linspace(0, S["track_cm"], S["n_bins"] + 1)
    H, _, _ = np.histogram2d(tr, dc, bins=[edges, edges])
    H = H / np.maximum(H.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(H.T, origin="lower", aspect="equal", cmap="magma",
                   extent=[0, S["track_cm"], 0, S["track_cm"]])
    ax.plot([0, S["track_cm"]], [0, S["track_cm"]], "w--", lw=0.8)
    ax.set_xlabel("true position (cm)")
    ax.set_ylabel("decoded position (cm)")
    ax.set_title("(b) Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(decoded | true)")

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, S["track_cm"], 40)
    ax.hist(dec["chance"], bins=bins, density=True, color="0.75",
            label=f"chance (median {np.median(dec['chance']):.0f} cm)")
    ax.hist(dec["error"], bins=bins, density=True, histtype="step", lw=2,
            color="#e76f51",
            label=f"observed (median {np.median(dec['error']):.1f} cm)")
    ax.set_xlabel("absolute decoding error (cm)")
    ax.set_ylabel("density")
    ax.set_title("(c) Decoding error vs chance")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ns = sorted(dec["sweep"])
    med = [np.median(dec["sweep"][n]) for n in ns]
    lo = [np.min(dec["sweep"][n]) for n in ns]
    hi = [np.max(dec["sweep"][n]) for n in ns]
    ax.fill_between(ns, lo, hi, color="#e76f51", alpha=0.25)
    ax.plot(ns, med, "-o", color="#e76f51")
    ax.axhline(np.median(dec["chance"]), color="0.5", ls="--", label="chance")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("number of place cells in the ensemble")
    ax.set_ylabel("median decoding error (cm)")
    ax.set_title("(d) Accuracy improves with ensemble size\n"
                 "(shaded: range over 8 random subsets)")
    ax.legend()

    fig.suptitle("The population code is read-out-able: position is recoverable "
                 "from held-out laps to within a few centimetres", y=0.97,
                 fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
dec = decoding_results(S, stats, per_dir)
print(f"median decoding error {np.median(dec['error']):.1f} cm "
      f"(chance {np.median(dec['chance']):.0f} cm) from "
      f"{dec['n_cells']} place cells in {int(dec['bin_size'] * 1000)} ms bins")
fig_decoding(S, stats, per_dir, dec, "fig06_decoding.png")

# %% [markdown]
# ### Figure 7 — position or running speed?
#
# Running speed is not uniform along a linear track: the animal accelerates away
# from one reward port and decelerates into the other. A purely speed-tuned cell
# would therefore still produce a peaked rate map. Fitting Poisson GLMs with
# `nemos` separates the two: a spline basis over position, a spline basis over
# speed, and both together, each scored by held-out log-likelihood relative to a
# constant-rate model and expressed in bits per spike, the same units as the
# Skaggs measure.

# %%
def glm_position_vs_speed(units, position, speed, laps, dt, bin_size=0.04,
                          n_folds=4, n_pos_basis=12, n_speed_basis=5,
                          n_bins=N_POS_BINS, track_cm=TRACK_LENGTH_CM,
                          min_speed=MIN_SPEED_CM_S, reg_strength=1e-4,
                          min_spikes=20):
    """Cross-validated Poisson GLMs of spiking on position, speed, and both.

    On a linear track running speed is not uniform along the track, so a cell
    that was purely speed tuned would still produce a peaked rate map. Fitting
    speed-only, position-only and joint models and comparing their held-out
    likelihoods separates the two.

    Returns per-unit cross-validated log-likelihood gains over a constant-rate
    model, in bits per spike (the same units as the Skaggs measure), plus the
    position tuning implied by the fitted joint model.
    """
    import nemos as nmo

    ep = apply_speed_mask(laps, speed, min_speed)
    counts = units.count(bin_size, ep=ep)
    pos_b = position.interpolate(counts, ep=counts.time_support)
    spd_b = speed.interpolate(counts, ep=counts.time_support)

    y = np.asarray(counts)
    ok = np.isfinite(pos_b.values) & np.isfinite(spd_b.values)
    y, pv, sv, tv = y[ok], pos_b.values[ok], spd_b.values[ok], counts.t[ok]

    fold = (np.searchsorted(laps.start, tv, side="right") - 1) % n_folds

    # Units with no spikes in some training fold cannot have their intercept
    # initialised from the mean rate, so they are dropped and reported as NaN.
    per_fold = np.stack([y[fold != f].sum(axis=0) for f in range(n_folds)])
    keep = np.flatnonzero((per_fold >= 1).all(axis=0) & (y.sum(axis=0) >= min_spikes))
    y = y[:, keep]

    pos_basis = nmo.basis.BSplineEval(n_basis_funcs=n_pos_basis,
                                      bounds=(0.0, track_cm), label="position")
    spd_basis = nmo.basis.BSplineEval(n_basis_funcs=n_speed_basis,
                                      bounds=(float(sv.min()), float(sv.max())),
                                      label="speed")
    X_pos = np.asarray(pos_basis.compute_features(pv))
    X_spd = np.asarray(spd_basis.compute_features(sv))
    designs = {"position": X_pos, "speed": X_spd,
               "position+speed": np.hstack([X_pos, X_spd])}

    n_units = len(units)
    gains = {k: np.zeros(y.shape[1]) for k in designs}
    n_test_spikes = np.zeros(y.shape[1])
    tuning = None
    grid = np.linspace(0, track_cm, n_bins)
    X_grid = np.hstack([np.asarray(pos_basis.compute_features(grid)),
                        np.zeros((n_bins, n_speed_basis))])

    for f in range(n_folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        lam_null = np.maximum(y[tr].mean(axis=0), 1e-8)
        n_test_spikes += y[te].sum(axis=0)
        for name, X in designs.items():
            model = nmo.glm.PopulationGLM(
                solver_name="LBFGS", regularizer="Ridge",
                regularizer_strength=reg_strength,
                solver_kwargs={"maxiter": 3000, "tol": 1e-8})
            model.fit(X[tr], y[tr])
            lam = np.maximum(np.asarray(model.predict(X[te])), 1e-8)
            ll = (y[te] * np.log(lam) - lam).sum(axis=0)
            ll0 = (y[te] * np.log(lam_null) - lam_null).sum(axis=0)
            gains[name] += ll - ll0
            if name == "position+speed" and f == 0:
                # Position tuning at the mean speed, for display.
                sm = np.asarray(spd_basis.compute_features(
                    np.full(n_bins, sv.mean())))
                Xg = np.hstack([X_grid[:, :n_pos_basis], sm])
                tuning = np.asarray(model.predict(Xg)) / bin_size

    full = {}
    for k in gains:
        v = np.full(n_units, np.nan)
        v[keep] = gains[k] / np.maximum(n_test_spikes, 1) / np.log(2)
        full[k] = v
    tuning_full = np.full((n_bins, n_units), np.nan)
    if tuning is not None:
        tuning_full[:, keep] = tuning
    nts = np.full(n_units, np.nan)
    nts[keep] = n_test_spikes
    return dict(gains=full, grid=grid, tuning=tuning_full, kept=keep,
                n_test_spikes=nts, bin_size=bin_size)

# %%
def fig_glm(S, stats, per_dir, fields, glm, direction, fname, n_examples=3):
    pc = stats["is_place_cell"].values
    d = per_dir[direction]
    ok = np.isfinite(glm["gains"]["position"])

    fig = plt.figure(figsize=(15, 8.2))
    gs = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=1.1)

    # top row: model-based tuning next to the empirical rate map
    sub = fields[(fields["direction"] == direction)
                 & fields["k"].isin(np.flatnonzero(ok))]
    sub = sub.nlargest(n_examples * 4, "participation").nlargest(n_examples, "peak_rate")
    for m, (_, row) in enumerate(sub.iterrows()):
        k = int(row["k"])
        ax = fig.add_subplot(gs[0, 2 * m:2 * m + 2])
        ax.plot(d["centres"], d["rates"][k], color="0.4", lw=1.6,
                label="binned rate map")
        ax.plot(glm["grid"], glm["tuning"][:, k], color="#6a4c93", lw=2.0,
                label="Poisson GLM")
        ax.set_xlabel("position (cm)")
        ax.set_ylabel("rate (Hz)")
        ax.set_title(f"({'abc'[m]}) unit {int(row['unit'])}", fontsize=9.5)
        if m == 0:
            ax.legend(fontsize=8)

    gp = glm["gains"]["position"]
    gsp = glm["gains"]["speed"]
    gb = glm["gains"]["position+speed"]

    ax = fig.add_subplot(gs[1, 0:2])
    ax.scatter(gsp[ok & ~pc], gp[ok & ~pc], s=16, color="0.6", label="other units")
    ax.scatter(gsp[ok & pc], gp[ok & pc], s=20, color="#1b6ca8", label="place cells")
    lim = [min(0, np.nanmin(gsp[ok])) - 0.05, np.nanmax(gp[ok]) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("speed-only model (bits/spike)")
    ax.set_ylabel("position-only model (bits/spike)")
    frac = float(np.mean(gp[ok & pc] > gsp[ok & pc]))
    ax.set_title(f"(d) Held-out likelihood gain: position beats\n"
                 f"speed for {frac:.0%} of place cells", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[1, 2:4])
    names = ["speed", "position", "position+speed"]
    for i, name in enumerate(names):
        for j, (m, col, off) in enumerate(((pc & ok, "#1b6ca8", -0.18),
                                           (~pc & ok, "0.6", 0.18))):
            v = glm["gains"][name][m]
            bp = ax.boxplot([v], positions=[i + off], widths=0.3,
                            patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.75)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("cross-validated gain (bits/spike)")
    ax.legend(handles=[Patch(facecolor="#1b6ca8", alpha=0.75, label="place cells"),
                       Patch(facecolor="0.6", alpha=0.75, label="other units")],
              fontsize=8, loc="upper left")
    ax.set_title("(e) Speed alone explains little of the\nspiking; position explains most of it",
                 fontsize=9.5)

    ax = fig.add_subplot(gs[1, 4:6])
    extra = gb - gp
    ax.hist(extra[ok & pc], bins=25, color="#1b6ca8", alpha=0.85)
    ax.axvline(np.nanmedian(extra[ok & pc]), color="k", ls="--", lw=1)
    ax.set_xlabel("gain from adding speed to the position model\n(bits/spike)")
    ax.set_ylabel("number of place cells")
    ax.set_title(f"(f) Once position is in the model, speed adds\n"
                 f"almost nothing (median "
                 f"{np.nanmedian(extra[ok & pc]):.3f} bits/spike)", fontsize=9.5)

    fig.suptitle("A Poisson GLM confirms the fields are spatial, not a by-product "
                 f"of the speed profile along the track ({direction} runs)",
                 y=0.99, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
ref = max(per_dir, key=lambda d: len(per_dir[d]["laps"]))
glm = glm_position_vs_speed(S["units"], S["position"], S["speed"],
                            per_dir[ref]["laps"], S["dt"],
                            n_bins=S["n_bins"], track_cm=S["track_cm"])
pc_mask = stats["is_place_cell"].values
for name, g in glm["gains"].items():
    print(f"{name:>16s}: {np.nanmedian(g[pc_mask]):.3f} bits/spike "
          f"(median over place cells)")
fig_glm(S, stats, per_dir, fields, glm, ref, "fig07_glm_position_vs_speed.png")

# %% [markdown]
# ## All eight sessions
#
# The whole pipeline is now run over every session in the dandiset, including the
# three in which the animal ran on a circular maze. There the linearised
# coordinate wraps, so rate maps are smoothed circularly and the session is
# treated as having a single running direction.

# %%
COMMON_BINS = 40


def analyse(S, n_shuffles=N_SHUFFLES, progress=None):
    stats, per_dir = classify_place_cells(
        S["units"], S["position"], S["run_ep"], S["lap_dir"], S["speed"], S["dt"],
        n_bins=S["n_bins"], track_cm=S["track_cm"],
        circular=S["maze_type"] == "circular",
        n_shuffles=n_shuffles, progress=progress)
    fields = field_table(S["units"], stats, per_dir)
    return stats, per_dir, fields


def decode_session(S, stats, per_dir, rng):
    pc_ids = stats.index[stats["is_place_cell"]].values
    ens = S["units"][list(pc_ids)]
    errs, trues, decs = [], [], []
    for name in per_dir:
        _, dcd, tru = crossvalidated_decoding(
            ens, S["position"], per_dir[name]["laps"], S["speed"], S["dt"],
            n_bins=S["n_bins"], track_cm=S["track_cm"],
            circular=S["maze_type"] == "circular")
        errs.append(np.abs(dcd - tru))
        trues.append(tru)
        decs.append(dcd)
    err = np.concatenate(errs)
    tru, dcd = np.concatenate(trues), np.concatenate(decs)
    chance = np.concatenate([np.abs(rng.permutation(dcd) - tru) for _ in range(20)])
    return float(np.median(err)), float(np.median(chance))


def resample_map(rates, n_out=COMMON_BINS):
    """Interpolate rate maps onto a common fraction-of-track axis."""
    src = np.linspace(0, 1, rates.shape[1])
    dst = np.linspace(0, 1, n_out)
    return np.stack([np.interp(dst, src, r) for r in rates])

# %%
def fig_multisession(summary, pooled_fields, pooled_maps, pooled_glm, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    fig.subplots_adjust(hspace=0.75, wspace=0.32)
    order = summary.sort_values(["maze_type", "session"]).index
    sm = summary.loc[order]
    x = np.arange(len(sm))
    labels = [f"{s}\n{int(t)} cm {mt}" for s, t, mt in
              zip(sm["session"], sm["track_cm"], sm["maze_type"])]
    cols = [MAZE_COLOR[m] for m in sm["maze_type"]]

    ax = axes[0, 0]
    ax.bar(x, 100 * sm["place_cell_fraction"], color=cols)
    for xi, (frac, n, ne) in enumerate(zip(sm["place_cell_fraction"],
                                           sm["n_place_cells"],
                                           sm["n_excitatory"])):
        ax.text(xi, 100 * frac + 1.5, f"{int(n)}/{int(ne)}", ha="center",
                fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("place cells (% of excitatory units)")
    ax.set_ylim(0, 108)
    lo, hi = 100 * sm["place_cell_fraction"].min(), 100 * sm["place_cell_fraction"].max()
    ax.set_title(f"(a) A large fraction of CA1 pyramidal cells has\n"
                 f"a place field in every session ({lo:.0f}-{hi:.0f}%)")
    ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in MAZE_COLOR.items()],
              fontsize=8, loc="upper center", ncol=2)

    ax = axes[0, 1]
    groups = [pooled_fields.loc[pooled_fields["session"] == s, "si"].values
              for s in sm["session"]]
    bp = ax.boxplot(groups, positions=x, widths=0.6, patch_artist=True,
                    showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for g, xi, c in zip(groups, x, cols):
        ax.scatter(np.random.default_rng(2).normal(xi, 0.08, g.size), g, s=6,
                   color=c, alpha=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("spatial information (bits/spike)")
    ax.set_title("(b) Spatial information per field")

    ax = axes[0, 2]
    ax.bar(x, sm["decode_error_cm"], color=cols)
    ax.plot(x, sm["chance_error_cm"], "k_", ms=18, mew=2, label="chance")
    for xi, (e, c) in enumerate(zip(sm["decode_error_cm"], sm["chance_error_cm"])):
        ax.text(xi, e + 1, f"{e:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("median decoding error (cm)")
    ax.set_title("(c) Held-out position is decodable\nin every session")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    maps = pooled_maps / np.maximum(pooled_maps.max(axis=1, keepdims=True), 1e-9)
    maps = maps[np.argsort(np.argmax(maps, axis=1))]
    im = ax.imshow(maps, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=1, extent=[0, 100, 0, maps.shape[0]])
    ax.set_xlabel("position (% of track)")
    ax.set_ylabel("place field #")
    ax.set_title(f"(d) All {maps.shape[0]} fields from all "
                 f"{len(sm)} sessions, sorted by peak")
    fig.colorbar(im, ax=ax, fraction=0.05, label="rate / peak rate")

    ax = axes[1, 1]
    for key, col, lab in (("width", "#1b6ca8", "field width (cm)"),):
        ax.hist(pooled_fields[key], bins=30, color=col, alpha=0.85)
        ax.axvline(pooled_fields[key].median(), color="k", ls="--", lw=1)
        ax.set_xlabel(lab)
    ax.set_ylabel("number of fields")
    rel = (pooled_fields["width"] / pooled_fields["track_cm"]).median()
    ax.set_title(f"(e) Pooled field width\n"
                 f"median {pooled_fields['width'].median():.0f} cm "
                 f"({rel:.0%} of the animal's own track)")

    ax = axes[1, 2]
    g = pooled_glm
    ax.scatter(g["speed"], g["position"], s=14, c=[MAZE_COLOR[m] for m in g["maze_type"]],
               alpha=0.75)
    lim = [min(0, g["speed"].min()) - 0.05, g["position"].max() * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("speed-only GLM (bits/spike)")
    ax.set_ylabel("position-only GLM (bits/spike)")
    ax.legend(handles=[Patch(facecolor=c, label=m) for m, c in MAZE_COLOR.items()],
              fontsize=8, loc="lower right")
    ax.set_title(f"(f) Position beats speed for {(g['position'] > g['speed']).mean():.0%}\n"
                 f"of place cells, pooled over sessions")

    fig.suptitle("The same result holds across all 8 sessions of DANDI:000044 "
                 "(4 rats, three maze geometries)", y=0.98, fontsize=11)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    return fname

# %%
rng = np.random.default_rng(0)
rows, all_fields, all_maps, glm_rows = [], [], [], []

for path, url in tqdm(list(urls.items()), desc="sessions"):
    Si = S if path == PRIMARY else load_session(url)
    st, pd_, fl = (stats, per_dir, fields) if path == PRIMARY else analyse(Si)
    med_err, chance_err = decode_session(Si, st, pd_, rng)

    refi = max(pd_, key=lambda d: len(pd_[d]["laps"]))
    gi = glm_position_vs_speed(Si["units"], Si["position"], Si["speed"],
                               pd_[refi]["laps"], Si["dt"],
                               n_bins=Si["n_bins"], track_cm=Si["track_cm"])
    m = st["is_place_cell"].values & np.isfinite(gi["gains"]["position"])
    glm_rows.append(pd.DataFrame({
        "session": Si["session_id"], "maze_type": Si["maze_type"],
        "position": gi["gains"]["position"][m],
        "speed": gi["gains"]["speed"][m],
        "position+speed": gi["gains"]["position+speed"][m]}))

    n_exc = int((st["cell_type"] == "excitatory").sum())
    rows.append(dict(
        session=Si["session_id"], subject=Si["subject_id"],
        maze_type=Si["maze_type"], track_cm=Si["track_cm"],
        n_units=len(Si["units"]), n_excitatory=n_exc,
        n_laps=len(Si["run_ep"]),
        run_time_s=float((Si["run_ep"].end - Si["run_ep"].start).sum()),
        n_place_cells=int(st["is_place_cell"].sum()),
        place_cell_fraction=float(st["is_place_cell"].sum()) / n_exc,
        n_fields=len(fl),
        median_si=float(fl["si"].median()),
        median_width_cm=float(fl["width"].median()),
        median_peak_rate=float(fl["peak_rate"].median()),
        median_participation=float(fl["participation"].median()),
        decode_error_cm=med_err, chance_error_cm=chance_err))

    fl = fl.assign(session=Si["session_id"], track_cm=Si["track_cm"],
                   maze_type=Si["maze_type"])
    all_fields.append(fl)
    for name, d in pd_.items():
        sel = fl.loc[fl["direction"] == name, "k"].astype(int).values
        if sel.size:
            all_maps.append(resample_map(d["rates"][sel]))
    if path != PRIMARY:
        Si["io"].close()

summary = pd.DataFrame(rows)
pooled_fields = pd.concat(all_fields, ignore_index=True)
pooled_maps = np.concatenate(all_maps)
pooled_glm = pd.concat(glm_rows, ignore_index=True)

summary.to_csv("session_summary.csv", index=False)
pooled_fields.to_csv("place_fields_all_sessions.csv", index=False)
stats.to_csv("place_cell_stats_primary.csv")
fields.to_csv("place_fields_primary.csv", index=False)

summary[["session", "subject", "maze_type", "track_cm", "n_excitatory",
         "n_place_cells", "place_cell_fraction", "n_fields", "median_si",
         "median_width_cm", "decode_error_cm", "chance_error_cm"]].round(3)

# %%
fig_multisession(summary, pooled_fields, pooled_maps, pooled_glm,
                 "fig08_across_sessions.png")

# %% [markdown]
# ## Summary of what the data show
#
# Across all eight sessions of DANDI:000044 — four rats, three maze geometries —
# the analysis recovers the defining properties of hippocampal place cells:
#
# - **Spatial firing.** Between a third and nine tenths of the CA1 excitatory
#   units in each session have a firing field that survives a per-cell
#   circular-shift test at p < 0.01. Pooled over sessions this is 322 place cells
#   carrying 412 fields.
# - **Compact and reliable fields.** The median field is about 32 cm wide, roughly
#   a fifth of the track, with a median in-field peak of a few to a few tens of Hz.
#   A field produces at least one spike on about 85% of the passes through it,
#   against roughly 20% for excitatory cells whose rate maps merely happen to be
#   peaked, and the maps built from odd and even laps correlate at about 0.95.
# - **Complete coverage and directional remapping.** Field peaks tile the entire
#   track, with the usual over-representation of the reward ends. On the linear
#   tracks the two directions of travel carry essentially independent maps: peak
#   positions in the two directions correlate at only about 0.2.
# - **A read-out-able population code.** Bayesian decoding from held-out laps
#   recovers position to a median error of 4–13 cm depending on the session,
#   against a chance level of 47–84 cm, and the error falls steadily as more cells
#   are added to the ensemble.
# - **Position, not speed.** Poisson GLMs give position roughly an order of
#   magnitude more held-out likelihood than running speed, position wins for about
#   90% of place cells, and adding speed to a position model buys essentially
#   nothing.
#
# Two methodological points are worth carrying away. First, raw spatial
# information in bits per spike is not comparable across cells with different
# spike counts: in this data set the excitatory units that *fail* the place-cell
# test have a higher median raw spatial information than the ones that pass,
# because sparse firing produces noisy, spiky rate maps. Only the comparison
# against each cell's own null, or a reliability measure such as lap-by-lap
# participation, separates them. Second, the fast-spiking interneurons in this
# data set are weakly but detectably spatially modulated; excluding them by cell
# type, rather than by any map statistic, is what keeps the place-cell population
# clean.
