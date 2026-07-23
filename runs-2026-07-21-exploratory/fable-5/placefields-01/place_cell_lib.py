"""Core routines for the hippocampal place-cell demonstration on DANDI:000044.

The dandiset is the Grosmark & Buzsaki (2016) hc-11 data set: bilateral silicon
probe recordings from dorsal CA1 while rats shuttled back and forth on a linear
track, flanked by pre- and post-behaviour sleep sessions. Five of the eight
sessions use a linear track (1.6 m or 2 m); the other three use a circular maze,
on which the animal runs in one direction only and the linearised coordinate
wraps, so those sessions are smoothed circularly and analysed as a single
running direction.

Three quirks of these NWB files drive the loading code below:

1. The behavioural SpatialSeries store the sampling *period* (0.0256 s) in the
   `rate` field rather than the rate, so timestamps have to be rebuilt as
   ``starting_time + arange(n) * rate``.  The resulting 39.06 Hz stream spans
   exactly the MazeEpoch.
2. The linearised position is only defined while the animal is on the track;
   samples during the reward pauses at either end are NaN.  The contiguous
   non-NaN segments therefore bracket the periods on the track, but they include
   brief shuffles at the reward ports as well as full end-to-end runs, so laps
   are selected by requiring a segment to cover most of the track in a
   consistent direction.
3. The track length differs between sessions and is encoded in the name of the
   Position container (e.g. "1.6mLinearMazeLinearizedPosition"), so bin edges
   are derived per session rather than hard-coded.
"""

import re

import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d

nap.nap_config.suppress_conversion_warnings = True

DANDISET_ID = "000044"
DANDI_API = "https://api.dandiarchive.org/api"
REMFILE_CACHE = "/tmp/remfile_cache"

# Analysis constants
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


# ----------------------------------------------------------------------------
# DANDI access
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Session assembly
# ----------------------------------------------------------------------------
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


DIRECTIONS = ("rightward", "leftward")
DIR_SIGN = {"rightward": 1, "leftward": -1}


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


# ----------------------------------------------------------------------------
# Rate maps and place-field statistics
# ----------------------------------------------------------------------------
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


def _spike_sample_index(spike_t, sample_t):
    """Index of the behavioural sample nearest each spike time."""
    i = np.searchsorted(sample_t, spike_t)
    i = np.clip(i, 1, sample_t.size - 1)
    left = spike_t - sample_t[i - 1]
    right = sample_t[i] - spike_t
    return np.where(right < left, i, i - 1)


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


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# LFP
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# GLM: does position explain spiking over and above running speed?
# ----------------------------------------------------------------------------
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
