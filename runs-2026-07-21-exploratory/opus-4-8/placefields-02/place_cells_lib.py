"""Shared helpers for the hippocampal place-cell analysis of DANDI:000044.

The dandiset is the Grosmark & Buzsaki (2016) hc-11 recordings: bilateral
silicon-probe recordings from dorsal CA1 in freely moving Long-Evans rats.  Each
session is a concatenation of a PRE sleep epoch, a novel MAZE running epoch, and
a POST sleep epoch.  We only use the MAZE epoch here.

Everything is streamed from the archive through LINDI, so no whole-file download
is required even though the source NWB files are 5-9 GB each.
"""

import os

import lindi
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d

nap.nap_config.suppress_conversion_warnings = True

DANDISET = "000044"
LINDI_CACHE = os.environ.get("LINDI_CACHE_DIR", "/tmp/lindi_cache")

# The five linear-track sessions of DANDI:000044 (the other three used a
# circular platform, where a single linear coordinate is not comparable).
SESSIONS = [
    dict(subject="Achilles", session="10252013", maze="1.6mLinearMaze",
         asset_id="5349c68b-c0a7-46c0-9900-cda050722fa4"),
    dict(subject="Buddy", session="06272013", maze="1.6mLinearMaze",
         asset_id="82714afb-724f-4e2b-b102-c9c47b5cba73"),
    dict(subject="Cicero", session="09012014", maze="1.6mLinearMaze",
         asset_id="3cc5b7b3-02e2-490a-9f19-d20670355084"),
    dict(subject="Gatsby", session="08022013", maze="1.6mLinearMaze",
         asset_id="31ea0aab-4777-424e-9a93-9605b2bdcc29"),
    dict(subject="Cicero", session="09172014", maze="2mLinearMaze",
         asset_id="e381ebb3-128e-4f3f-9517-11277d7aed9b"),
]

# Analysis parameters
BIN_SIZE_M = 0.02          # 2 cm spatial bins
SPEED_THRESHOLD = 0.05     # m/s, minimum running speed for a "run" sample
SMOOTH_BINS = 1.5          # gaussian sigma (in bins) applied to rate maps
MIN_PEAK_RATE = 1.0        # Hz, minimum peak rate for a place-cell candidate
MIN_MEAN_RATE = 0.05       # Hz, minimum overall rate on the maze
N_SHUFFLES = 500


def open_nwb(asset_id, dandiset=DANDISET):
    """Open a DANDI asset over the network via LINDI + a local chunk cache."""
    url = f"https://lindi.neurosift.org/dandi/dandisets/{dandiset}/assets/{asset_id}/nwb.lindi.json"
    cache = lindi.LocalCache(cache_dir=LINDI_CACHE)
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=cache)
    io = NWBHDF5IO(file=f, mode="r")
    return io.read()


def _behavior_series(nwbfile, maze):
    beh = nwbfile.processing["behavior"]
    lin = beh[f"{maze}LinearizedPosition"][f"{maze}LinearizedTimeSeries"]
    xy = beh[f"{maze}Position"][f"{maze}SpatialSeries"]
    return lin, xy


def load_session(session, verbose=True):
    """Load one session and return the objects needed for place-field analysis.

    Returns a dict with:
      ``nwbfile``   the pynwb file object
      ``epochs``    IntervalSet of PRE / MAZE / POST with a ``label`` column
      ``maze_ep``   IntervalSet covering the MAZE epoch only
      ``lin``       Tsd of linearized position (m) over the MAZE epoch
      ``xy``        TsdFrame of raw 2-D position (m) over the MAZE epoch
      ``speed``     Tsd of running speed (m/s), same sampling as ``lin``
      ``units``     TsGroup of all sorted units with cell_type / location metadata
    """
    nwbfile = open_nwb(session["asset_id"])
    lin_ts, xy_ts = _behavior_series(nwbfile, session["maze"])

    ep_df = nwbfile.epochs.to_dataframe()
    maze_row = ep_df[ep_df.label.str.contains("Maze")].iloc[0]
    maze_ep = nap.IntervalSet(start=float(maze_row.start_time), end=float(maze_row.stop_time))

    lin_raw = np.asarray(lin_ts.data[:], dtype=float).ravel()
    xy_raw = np.asarray(xy_ts.data[:], dtype=float)

    # The `rate` attribute of these TimeSeries actually holds the sampling
    # *period* (1/39.06 s), not the rate: n_samples * rate reproduces the MAZE
    # epoch duration to within a millisecond.  Build the time vector from that.
    dt = float(lin_ts.rate)
    t = float(lin_ts.starting_time) + np.arange(lin_raw.size) * dt
    assert abs(lin_raw.size * dt - (maze_ep.end[0] - maze_ep.start[0])) < 0.01

    # The file's linearized coordinate is only defined while the animal is on
    # the track proper (the reward areas beyond each end are left as NaN), which
    # is exactly the restriction we want for a 1-D place-field analysis.
    lin_full = lin_raw
    lin = nap.Tsd(t=t, d=lin_full, time_support=maze_ep)
    xy = nap.TsdFrame(t=t, d=xy_raw, columns=["x", "y"], time_support=maze_ep)

    # Speed from the linear coordinate; smooth lightly before differentiating
    # so that tracking jitter does not dominate.
    valid = np.isfinite(lin_full)
    lin_s = np.full_like(lin_full, np.nan)
    lin_s[valid] = gaussian_filter1d(lin_full[valid], sigma=4.0, mode="nearest")
    speed_d = np.full_like(lin_full, np.nan)
    speed_d[valid] = np.gradient(lin_s[valid], t[valid])
    speed = nap.Tsd(t=t, d=speed_d, time_support=maze_ep)

    units = _load_units(nwbfile)

    if verbose:
        frac = np.mean(valid)
        print(f"{session['subject']} {session['session']}: maze "
              f"{maze_ep.end[0] - maze_ep.start[0]:.0f} s, "
              f"{units.metadata.shape[0]} units, "
              f"tracking valid on {100 * frac:.0f}% of samples, dt={dt * 1e3:.2f} ms")
    return dict(nwbfile=nwbfile, session=session, epochs=ep_df, maze_ep=maze_ep,
                lin=lin, xy=xy, speed=speed, units=units, dt=dt)


def _load_units(nwbfile):
    """Build a TsGroup of all sorted units with their metadata."""
    ut = nwbfile.units
    spike_times = {int(uid): nap.Ts(np.asarray(ut["spike_times"][i], dtype=float))
                   for i, uid in enumerate(ut.id[:])}
    meta = dict(
        cell_type=[str(v) for v in ut["cell_type"].data[:]],
        location=[str(v) for v in ut["location"].data[:]],
        shank_id=[int(v) for v in ut["shank_id"].data[:]],
    )
    return nap.TsGroup(spike_times, metadata=meta)


# ---------------------------------------------------------------------------
# Run epochs
# ---------------------------------------------------------------------------

def run_epochs(lin, speed, dt, speed_threshold=SPEED_THRESHOLD, min_duration=0.5):
    """Split the maze epoch into rightward and leftward running epochs.

    A sample counts as running when the position is tracked and |speed| exceeds
    ``speed_threshold``.  Directions are separated by the sign of the velocity so
    that the two travel directions can be analysed independently, which is the
    standard treatment for a linear track: CA1 place fields there are strongly
    direction selective.
    """
    ok = np.isfinite(lin.values) & np.isfinite(speed.values)
    out = {}
    for name, mask in (("right", ok & (speed.values > speed_threshold)),
                       ("left", ok & (speed.values < -speed_threshold))):
        ep = _mask_to_intervalset(lin.t, mask, dt)
        out[name] = ep.merge_close_intervals(2 * dt).drop_short_intervals(min_duration)
    out["run"] = out["right"].union(out["left"])
    return out


def _mask_to_intervalset(t, mask, dt):
    if not mask.any():
        return nap.IntervalSet(start=[], end=[])
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size - 1]
    return nap.IntervalSet(start=t[starts] - dt / 2, end=t[ends] + dt / 2)


def select_pyramidal(units, min_rate=MIN_MEAN_RATE, ep=None):
    """Excitatory (pyramidal) units firing at least ``min_rate`` Hz inside ``ep``.

    The rate criterion is evaluated on ``ep`` but the returned spike trains are
    not restricted, so the same object can be used for rasters over arbitrary
    windows as well as for the epoch-restricted analyses.
    """
    exc = units[np.array(units.metadata["cell_type"]) == "excitatory"]
    rates = np.asarray(exc.restrict(ep).rates) if ep is not None else np.asarray(exc.rates)
    return exc[rates >= min_rate]


# ---------------------------------------------------------------------------
# Rate maps and place-field statistics
# ---------------------------------------------------------------------------

def rate_maps(units, lin, ep, bin_size=BIN_SIZE_M, smooth=SMOOTH_BINS, minmax=None,
              return_xr=False):
    """Occupancy-normalised firing-rate maps via pynapple, lightly smoothed.

    Returns ``(rates, centers, occupancy)`` with ``rates`` of shape
    (n_units, n_bins) in Hz and ``occupancy`` in seconds.  With ``return_xr`` the
    smoothed maps are also returned as the xarray object pynapple's decoders
    expect (unvisited bins filled with zero).
    """
    if minmax is None:
        minmax = (float(np.nanmin(lin.values)), float(np.nanmax(lin.values)))
    nbins = int(round((minmax[1] - minmax[0]) / bin_size))
    tc = nap.compute_tuning_curves(units, lin, bins=nbins, range=[minmax],
                                   epochs=ep, feature_names=["position"])
    centers = np.asarray(tc.coords["position"].values)
    # pynapple reports occupancy as a sample count; convert to seconds.
    sample_dt = float(np.median(np.diff(lin.t)))
    occ = np.asarray(tc.attrs["occupancy"], dtype=float).ravel() * sample_dt
    rates = np.asarray(tc.values, dtype=float)
    rates[:, occ <= 0] = np.nan
    rates = _smooth_nan(rates, smooth)
    if return_xr:
        tc = tc.copy(data=np.where(np.isfinite(rates), rates, 0.0))
        return rates, centers, occ, tc
    return rates, centers, occ


def _smooth_nan(rates, sigma):
    """Gaussian smoothing along the position axis that ignores unvisited bins."""
    if sigma <= 0:
        return rates
    valid = np.isfinite(rates).astype(float)
    filled = np.where(np.isfinite(rates), rates, 0.0)
    num = gaussian_filter1d(filled, sigma, axis=-1, mode="nearest")
    den = gaussian_filter1d(valid, sigma, axis=-1, mode="nearest")
    out = np.where(den > 1e-6, num / den, np.nan)
    out[~np.isfinite(rates)] = np.nan
    return out


def spatial_information(rates, occupancy):
    """Skaggs spatial information in bits per spike, one value per unit."""
    occ = np.asarray(occupancy, dtype=float)
    ok = np.isfinite(rates).all(axis=0) & (occ > 0)
    p = occ[ok] / occ[ok].sum()
    r = rates[:, ok]
    rbar = (p[None, :] * r).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p[None, :] * (r / rbar[:, None]) * np.log2(r / rbar[:, None])
    term[~np.isfinite(term)] = 0.0
    return term.sum(axis=1), rbar


def sparsity(rates, occupancy):
    """Skaggs sparsity: <r>^2 / <r^2>; low values mean a compact field."""
    occ = np.asarray(occupancy, dtype=float)
    ok = np.isfinite(rates).all(axis=0) & (occ > 0)
    p = occ[ok] / occ[ok].sum()
    r = rates[:, ok]
    num = (p[None, :] * r).sum(axis=1) ** 2
    den = (p[None, :] * r ** 2).sum(axis=1)
    return np.where(den > 0, num / den, np.nan)


def field_width(rates, centers, frac=0.5):
    """Width (m) of the contiguous region around the peak above ``frac`` * peak."""
    widths = np.full(rates.shape[0], np.nan)
    step = float(np.mean(np.diff(centers)))
    for i, r in enumerate(rates):
        if not np.isfinite(r).any():
            continue
        rr = np.where(np.isfinite(r), r, 0.0)
        pk = int(np.argmax(rr))
        thr = frac * rr[pk]
        if thr <= 0:
            continue
        lo = pk
        while lo > 0 and rr[lo - 1] >= thr:
            lo -= 1
        hi = pk
        while hi < rr.size - 1 and rr[hi + 1] >= thr:
            hi += 1
        widths[i] = (hi - lo + 1) * step
    return widths


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------

class CircularShuffler:
    """Circularly shift spike trains inside a set of (possibly gappy) epochs.

    Time inside ``ep`` is concatenated into a single "virtual" axis, spikes are
    rolled by a random offset on that axis, and the shifted times are mapped
    back.  This destroys the spike-position relationship while preserving each
    unit's spike count and its fine-timescale autocorrelation structure, which
    is the standard null for spatial-information significance tests.
    """

    def __init__(self, ep, rng):
        self.start = np.asarray(ep.start, dtype=float)
        self.end = np.asarray(ep.end, dtype=float)
        self.dur = self.end - self.start
        self.cum = np.r_[0.0, np.cumsum(self.dur)]
        self.total = float(self.cum[-1])
        self.rng = rng

    def to_virtual(self, t):
        i = np.clip(np.searchsorted(self.start, t, side="right") - 1, 0, self.dur.size - 1)
        return self.cum[i] + (t - self.start[i])

    def from_virtual(self, v):
        i = np.clip(np.searchsorted(self.cum[1:], v, side="right"), 0, self.dur.size - 1)
        return self.start[i] + (v - self.cum[i])

    def shift(self, t, offset=None):
        if offset is None:
            offset = self.rng.uniform(self.total * 0.05, self.total * 0.95)
        return np.sort(self.from_virtual((self.to_virtual(t) + offset) % self.total))


def shuffled_si(units, lin, ep, n_shuffles=N_SHUFFLES, seed=0, **kw):
    """Null distribution of spatial information under circular shifts.

    Returns an array of shape (n_shuffles, n_units).
    """
    from tqdm.auto import tqdm

    rng = np.random.default_rng(seed)
    shuffler = CircularShuffler(ep, rng)
    spk = {k: np.asarray(units[k].restrict(ep).t) for k in units.keys()}
    null = np.full((n_shuffles, len(units)), np.nan)
    for s in tqdm(range(n_shuffles), desc="shuffles", leave=False):
        offsets = rng.uniform(shuffler.total * 0.05, shuffler.total * 0.95, len(units))
        shifted = {k: nap.Ts(shuffler.shift(spk[k], offsets[i]), time_support=ep)
                   for i, k in enumerate(units.keys())}
        grp = nap.TsGroup(shifted, time_support=ep)
        r, _, occ = rate_maps(grp, lin, ep, **kw)
        null[s], _ = spatial_information(r, occ)
    return null


# ---------------------------------------------------------------------------
# Per-session analysis
# ---------------------------------------------------------------------------

def analyze_direction(units, lin, ep, n_shuffles=N_SHUFFLES, seed=0, alpha=0.05,
                      minmax=None, **kw):
    """Rate maps plus place-field statistics and a shuffle test for one direction."""
    rates, centers, occ = rate_maps(units, lin, ep, minmax=minmax, **kw)
    si, mean_rate = spatial_information(rates, occ)
    null = shuffled_si(units, lin, ep, n_shuffles=n_shuffles, seed=seed,
                       minmax=minmax, **kw)
    pval = (np.sum(null >= si[None, :], axis=0) + 1) / (n_shuffles + 1)
    peak = np.nanmax(rates, axis=1)
    is_place = (pval < alpha) & (peak >= MIN_PEAK_RATE)
    return dict(rates=rates, centers=centers, occupancy=occ, si=si, null=null,
                pval=pval, peak_rate=peak, mean_rate=mean_rate,
                sparsity=sparsity(rates, occ), width=field_width(rates, centers),
                peak_pos=centers[np.nanargmax(np.where(np.isfinite(rates), rates, -np.inf), axis=1)],
                is_place=is_place, ep=ep, unit_ids=np.array(list(units.keys())))


def analyze_session(session, n_shuffles=N_SHUFFLES, seed=0, verbose=True):
    """Full single-session pipeline: load, define run epochs, analyse both directions."""
    s = load_session(session, verbose=verbose)
    eps = run_epochs(s["lin"], s["speed"], s["dt"])
    pyr = select_pyramidal(s["units"], ep=eps["run"])
    minmax = (float(np.nanmin(s["lin"].values)), float(np.nanmax(s["lin"].values)))
    res = {d: analyze_direction(pyr, s["lin"], eps[d], n_shuffles=n_shuffles,
                                seed=seed + i, minmax=minmax)
           for i, d in enumerate(("right", "left"))}
    s.update(eps=eps, pyr=pyr, res=res, minmax=minmax)
    if verbose:
        n = len(pyr)
        both = res["right"]["is_place"] | res["left"]["is_place"]
        print(f"  {n} pyramidal units, {both.sum()} ({100 * both.mean():.0f}%) "
              f"classified as place cells in at least one direction")
    return s


def directionality_index(res):
    """(peak_right - peak_left) / (peak_right + peak_left) per unit."""
    a, b = res["right"]["peak_rate"], res["left"]["peak_rate"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)
