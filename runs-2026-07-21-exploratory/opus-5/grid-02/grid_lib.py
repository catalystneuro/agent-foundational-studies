"""Core routines for the DANDI:000582 grid-cell analysis.

Data: Sargolini et al. (2006), tetrode recordings from dorsocaudal medial
entorhinal cortex of freely foraging Long-Evans rats in a 1 x 1 m box.

Everything here works on real streamed NWB data; no simulation anywhere.
"""

import json
import os

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import ndimage
from scipy.signal import fftconvolve

DANDISET = "000582"
ASSET_URL = (
    "https://api.dandiarchive.org/api/dandisets/000582/versions/draft/assets/%s/download/"
)
CACHE_DIR = os.environ.get("GRID_CACHE", "/tmp/remfile_cache_grid")

# Analysis parameters. Coordinates are in cm despite the NWB unit string saying
# "meters": the recorded enclosures span ~96 cm (the standard 1 x 1 m box) and,
# in a minority of sessions, ~190 cm. The map extent is therefore derived from
# the tracking data of each session rather than hard-coded.
BIN_SIZE = 2.5  # cm per spatial bin (-> 40 x 40 map in the 1 m box)
SMOOTH_SIGMA_BINS = 2.0  # Gaussian sd for rate-map smoothing (= 5 cm)
MIN_OCCUPANCY = 0.15  # s; bins visited for less than this are treated as unvisited
SPEED_THRESH = 2.5  # cm/s; samples slower than this are excluded
MIN_OVERLAP_BINS = 20  # minimum overlapping bins for a valid autocorrelation lag


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def list_assets(path="session_scan.json"):
    return json.load(open(path))


def open_nwb(asset_id):
    """Stream an NWB file from the DANDI S3 mirror with an on-disk cache."""
    f = remfile.File(ASSET_URL % asset_id, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True)
    return io.read()


def load_session(asset_id):
    """Return (units TsGroup, position TsdFrame, head-direction Tsd or None, speed Tsd).

    Position is resampled onto the LED-1 timestamps, short tracking dropouts are
    linearly interpolated, and samples with speed below SPEED_THRESH are removed
    from the epoch support so that rate maps reflect active foraging.
    """
    nwb = open_nwb(asset_id)
    pos_mod = nwb.processing["behavior"]["Position"]

    led1 = pos_mod.spatial_series["SpatialSeriesLED1"]
    t = np.asarray(led1.timestamps[:], dtype=float)
    xy1 = np.asarray(led1.data[:], dtype=float)

    xy2 = None
    if "SpatialSeriesLED2" in pos_mod.spatial_series:
        xy2 = np.asarray(pos_mod.spatial_series["SpatialSeriesLED2"].data[:], dtype=float)

    good = np.isfinite(xy1).all(axis=1)
    if xy2 is not None:
        good_hd = good & np.isfinite(xy2).all(axis=1)
    xy1 = _interp_gaps(t, xy1)
    if xy2 is not None:
        xy2 = _interp_gaps(t, xy2)

    # midpoint of the two LEDs is the standard proxy for the animal's position
    xy = xy1 if xy2 is None else 0.5 * (xy1 + xy2)

    dt = float(np.median(np.diff(t)))
    speed = np.hypot(*np.gradient(xy, dt, axis=0).T)
    speed = ndimage.uniform_filter1d(speed, size=int(round(0.4 / dt)))

    keep = np.isfinite(xy).all(axis=1) & (speed >= SPEED_THRESH)
    if xy2 is not None:
        keep_hd = keep & good_hd

    position = nap.TsdFrame(t=t, d=xy, columns=["x", "y"])
    speed_tsd = nap.Tsd(t=t, d=speed)

    hd = None
    if xy2 is not None:
        ang = np.arctan2(xy1[:, 1] - xy2[:, 1], xy1[:, 0] - xy2[:, 0])
        hd = nap.Tsd(t=t, d=np.mod(ang, 2 * np.pi))
        hd = hd.restrict(_bool_to_epochs(t, keep_hd, dt))

    run_ep = _bool_to_epochs(t, keep, dt)

    units = {}
    meta = {"unit_name": [], "histology": [], "depth": []}
    u = nwb.units
    for i in range(len(u)):
        units[i] = np.asarray(u["spike_times"][i], dtype=float)
        meta["unit_name"].append(str(u["unit_name"].data[i]))
        meta["histology"].append(str(u["histology"].data[i]))
        meta["depth"].append(float(u["depth"].data[i]))
    tsgroup = nap.TsGroup(
        {k: nap.Ts(t=v) for k, v in units.items()},
        time_support=run_ep,
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    return dict(
        units=tsgroup,
        position=position.restrict(run_ep),
        position_all=position,
        hd=hd,
        speed=speed_tsd,
        run_ep=run_ep,
        subject=str(nwb.subject.subject_id),
        session=str(nwb.session_id) if nwb.session_id else "",
        dt=dt,
    )


def _interp_gaps(t, xy, max_gap=1.0):
    """Linearly interpolate tracking dropouts shorter than max_gap seconds."""
    out = xy.copy()
    for j in range(xy.shape[1]):
        v = out[:, j]
        bad = ~np.isfinite(v)
        if not bad.any():
            continue
        idx = np.flatnonzero(bad)
        # group consecutive missing samples
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        gi = np.flatnonzero(~bad)
        for grp in splits:
            if len(grp) == 0:
                continue
            if (t[grp[-1]] - t[grp[0]]) <= max_gap and gi.size:
                v[grp] = np.interp(t[grp], t[gi], v[gi])
    return out


def _bool_to_epochs(t, keep, dt):
    """Convert a boolean sample mask into a pynapple IntervalSet."""
    k = keep.astype(int)
    edges = np.diff(np.concatenate([[0], k, [0]]))
    starts = t[np.flatnonzero(edges == 1)]
    ends = t[np.flatnonzero(edges == -1) - 1] + dt
    return nap.IntervalSet(start=starts, end=ends)


# --------------------------------------------------------------------------
# Rate maps
# --------------------------------------------------------------------------
def map_edges(half_width):
    n = int(np.ceil(2 * half_width / BIN_SIZE))
    half = n * BIN_SIZE / 2.0
    return np.linspace(-half, half, n + 1)


def session_extent(position):
    """Half-width (cm) of the square map covering this session's enclosure.

    The centre of the tracked area is taken as the origin so that maps from
    sessions with different camera offsets are directly comparable.
    """
    xy = np.asarray(position.values)
    xy = xy[np.isfinite(xy).all(axis=1)]
    lo = np.percentile(xy, 0.2, axis=0)
    hi = np.percentile(xy, 99.8, axis=0)
    center = 0.5 * (lo + hi)
    half = float(np.max(0.5 * (hi - lo))) + BIN_SIZE
    return half, center


def occupancy_map(position, dt, edges, center):
    xy = np.asarray(position.values) - center
    occ, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges, edges])
    return occ * dt


def covered_region(occ):
    """Bins inside the part of the box the animal actually covered.

    At 2.5 cm bins and 50 Hz tracking a 10 min session leaves scattered single
    bins unvisited by chance, so the visited mask is morphologically closed and
    its interior holes filled; only bins outside that region stay undefined.
    """
    visited = occ > 0
    closed = ndimage.binary_closing(visited, structure=np.ones((3, 3)), iterations=2)
    return ndimage.binary_fill_holes(closed)


def rate_map(spike_xy, occ, edges, sigma=SMOOTH_SIGMA_BINS):
    """Occupancy-normalised, Gaussian-smoothed rate map.

    The spike-count and occupancy maps are smoothed separately and then divided
    (Leutgeb et al. 2007), which weights each bin by how well it was sampled and
    keeps the estimate defined in briefly visited bins.
    """
    spk, _, _ = np.histogram2d(spike_xy[:, 0], spike_xy[:, 1], bins=[edges, edges])
    region = covered_region(occ)
    if sigma > 0:
        occ_s = ndimage.gaussian_filter(occ, sigma, mode="constant")
        spk_s = ndimage.gaussian_filter(spk, sigma, mode="constant")
    else:
        occ_s, spk_s = occ, spk
    valid = region & (occ_s >= MIN_OCCUPANCY)
    with np.errstate(invalid="ignore", divide="ignore"):
        rm = spk_s / occ_s
    rm[~valid] = np.nan
    return rm.T, spk.T, valid.T  # transpose so rows = y, cols = x


def spike_positions(spike_times, position, center=0.0):
    """Position of the animal at each spike (nearest tracking sample)."""
    t = position.t
    if len(spike_times) == 0:
        return np.empty((0, 2))
    center = np.asarray(center)
    idx = np.searchsorted(t, spike_times)
    idx = np.clip(idx, 1, len(t) - 1)
    left = np.abs(spike_times - t[idx - 1]) < np.abs(t[idx] - spike_times)
    idx = np.where(left, idx - 1, idx)
    return np.asarray(position.values)[idx] - center


# --------------------------------------------------------------------------
# Spatial autocorrelogram and gridness
# --------------------------------------------------------------------------
def _corr_full(a, b):
    return fftconvolve(a, b[::-1, ::-1], mode="full")


def autocorrelogram(rm):
    """Unbiased 2-D spatial autocorrelogram (Sargolini et al. 2006, eq. 1).

    Pearson correlation computed at each spatial lag over the overlapping,
    non-NaN bins only; lags with fewer than MIN_OVERLAP_BINS overlapping bins
    are returned as NaN.
    """
    m = np.isfinite(rm).astype(float)
    a = np.where(np.isfinite(rm), rm, 0.0)
    n = _corr_full(m, m)
    s12 = _corr_full(a, a)
    s1 = _corr_full(a, m)
    s2 = _corr_full(m, a)
    q1 = _corr_full(a * a, m)
    q2 = _corr_full(m, a * a)
    num = n * s12 - s1 * s2
    d1 = n * q1 - s1 ** 2
    d2 = n * q2 - s2 ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(np.clip(d1, 0, None) * np.clip(d2, 0, None))
    r[n < MIN_OVERLAP_BINS] = np.nan
    return np.clip(r, -1, 1)


def _radius_grid(shape):
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(yy - cy, xx - cx)


def autocorr_peaks(ac, min_distance=2):
    """Local maxima of the autocorrelogram, sorted by distance from the centre."""
    a = np.where(np.isfinite(ac), ac, -np.inf)
    mx = ndimage.maximum_filter(a, size=2 * min_distance + 1)
    peaks = (a == mx) & np.isfinite(ac) & (ac > 0.1)
    ys, xs = np.nonzero(peaks)
    cy, cx = (ac.shape[0] - 1) / 2.0, (ac.shape[1] - 1) / 2.0
    d = np.hypot(ys - cy, xs - cx)
    order = np.argsort(d)
    return ys[order], xs[order], d[order]


def central_peak_radius(ac):
    """Radius of the central autocorrelation peak, used as the annulus inner edge."""
    rad = _radius_grid(ac.shape)
    # walk outward until the radially averaged correlation stops decreasing
    rs = np.arange(0, min(ac.shape) // 2)
    prof = np.array(
        [np.nanmean(ac[(rad >= r) & (rad < r + 1)]) for r in rs]
    )
    prof = np.where(np.isfinite(prof), prof, 0.0)
    d = np.diff(prof)
    turn = np.flatnonzero(d > 0)
    r0 = rs[turn[0]] if turn.size else 3
    return max(float(r0), 2.0)


def gridness_score(ac, return_detail=False):
    """Expanding-annulus gridness score.

    For a range of outer radii the central peak is masked out, the remaining
    annulus is rotated by 30..150 deg and correlated with the unrotated annulus.
    Gridness = min(r60, r120) - max(r30, r90, r150), maximised over radii.
    """
    rad = _radius_grid(ac.shape)
    r_in = central_peak_radius(ac)
    r_max = min(ac.shape) / 2.0 - 1
    if r_max <= r_in + 4:
        return (np.nan, np.nan, np.nan, {}) if return_detail else np.nan

    filled = np.where(np.isfinite(ac), ac, 0.0)
    valid = np.isfinite(ac)
    rots = {}
    for ang in (30, 60, 90, 120, 150):
        rots[ang] = (
            ndimage.rotate(filled, ang, reshape=False, order=1, mode="constant"),
            ndimage.rotate(valid.astype(float), ang, reshape=False, order=1, mode="constant")
            > 0.9,
        )

    best, best_r, best_detail = -np.inf, np.nan, {}
    for r_out in np.arange(r_in + 4, r_max + 0.01, 1.0):
        base_mask = (rad >= r_in) & (rad <= r_out) & valid
        if base_mask.sum() < 50:
            continue
        cors = {}
        for ang, (rot, rvalid) in rots.items():
            m = base_mask & rvalid
            if m.sum() < 50:
                cors = {}
                break
            cors[ang] = float(np.corrcoef(ac[m], rot[m])[0, 1])
        if not cors:
            continue
        g = min(cors[60], cors[120]) - max(cors[30], cors[90], cors[150])
        if g > best:
            best, best_r, best_detail = g, r_out, cors
    if not np.isfinite(best):
        return (np.nan, np.nan, np.nan, {}) if return_detail else np.nan
    return (best, r_in, best_r, best_detail) if return_detail else best


def grid_geometry(ac, bin_size=BIN_SIZE, r_in=None):
    """Spacing (cm) and orientation (deg) from the six peaks nearest the centre.

    Spacing is the median centre-to-peak distance of the six closest non-central
    peaks; orientation is the angle of the peak axis closest to horizontal,
    folded into 0-60 deg because of the hexagon's symmetry.
    """
    if r_in is None:
        r_in = central_peak_radius(ac)
    ys, xs, d = autocorr_peaks(ac)
    cy, cx = (ac.shape[0] - 1) / 2.0, (ac.shape[1] - 1) / 2.0
    keep = d >= max(r_in, 3.0)
    ys, xs, d = ys[keep], xs[keep], d[keep]
    if len(d) < 3:
        return np.nan, np.nan, 0
    n = min(6, len(d))
    spacing = float(np.median(d[:n]) * bin_size)
    ang = np.degrees(np.arctan2(ys[:n] - cy, xs[:n] - cx)) % 180
    orient = float(np.min(ang % 60))
    return spacing, orient, n


def spatial_information(rm, occ_map):
    """Skaggs spatial information in bits per spike."""
    valid = np.isfinite(rm) & (occ_map.T > 0)
    lam = rm[valid]
    p = occ_map.T[valid]
    p = p / p.sum()
    mean_rate = np.sum(p * lam)
    if mean_rate <= 0:
        return np.nan
    nz = lam > 0
    return float(np.sum(p[nz] * (lam[nz] / mean_rate) * np.log2(lam[nz] / mean_rate)))


# --------------------------------------------------------------------------
# Head direction
# --------------------------------------------------------------------------
def hd_tuning(spike_times, hd, nbins=36):
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    t = hd.t
    occ, _ = np.histogram(np.asarray(hd.values), bins=edges)
    dt = float(np.median(np.diff(t)))
    occ = occ * dt
    idx = np.searchsorted(t, spike_times)
    idx = np.clip(idx, 0, len(t) - 1)
    ok = np.abs(t[idx] - spike_times) < 0.05
    sph = np.asarray(hd.values)[idx[ok]]
    cnt, _ = np.histogram(sph, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc = cnt / occ
    tc[occ < 0.5] = np.nan
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, tc


def mean_vector_length(centers, tc):
    ok = np.isfinite(tc) & (tc > 0)
    if ok.sum() < 5 or tc[ok].sum() == 0:
        return np.nan, np.nan
    w = tc[ok]
    v = np.sum(w * np.exp(1j * centers[ok])) / np.sum(w)
    return float(np.abs(v)), float(np.angle(v) % (2 * np.pi))


# --------------------------------------------------------------------------
# Shuffling
# --------------------------------------------------------------------------
def analyze_unit(spike_times, position, occ, edges, center):
    """Rate map -> autocorrelogram -> gridness / spacing / orientation / SI."""
    sxy = spike_positions(spike_times, position, center)
    rm, spk, valid = rate_map(sxy, occ, edges)
    ac = autocorrelogram(rm)
    g, r_in, r_out, cors = gridness_score(ac, return_detail=True)
    spacing, orient, npk = grid_geometry(ac, r_in=r_in)
    return dict(
        spike_xy=sxy, rate_map=rm, autocorr=ac, gridness=g, r_in=r_in, r_out=r_out,
        rot_corr=cors, spacing=spacing, orientation=orient, n_peaks=npk,
        peak_rate=float(np.nanmax(rm)) if np.isfinite(rm).any() else np.nan,
        mean_rate=float(np.nansum(spk) / occ.sum()) if occ.sum() > 0 else np.nan,
        spatial_info=spatial_information(rm, occ),
    )


def shift_spikes(spike_times, t0, t1, shift):
    """Circularly shift spike times within the session (minimum shift enforced)."""
    dur = t1 - t0
    return t0 + np.mod(spike_times - t0 + shift, dur)
