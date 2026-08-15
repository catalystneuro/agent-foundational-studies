"""Helper functions for demonstrating grid cells in medial entorhinal cortex.

Data source: DANDI dandiset 000582 (Sargolini et al., Science 2006),
MEC recordings from freely foraging rats in a 100 x 100 cm open field.

All spatial analyses follow the standard Moser-lab pipeline: occupancy-normalized
2D rate maps, spatial autocorrelograms, and the Sargolini "gridness" score.
"""

import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter, maximum_filter
from dandi.dandiapi import DandiAPIClient

DANDISET = "000582"
CACHE_DIR = "/tmp/remfile_cache"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def get_asset_url(path, dandiset=DANDISET):
    client = DandiAPIClient()
    ds = client.get_dandiset(dandiset, "draft")
    a = ds.get_asset_by_path(path)
    return a.get_content_url(follow_redirects=1, strip_query=True)


def load_session(path):
    """Open a streamed NWB file and return the pynwb NWBFile object."""
    url = get_asset_url(path)
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rf, "r"))
    return io.read()


def extract_position(nwbfile):
    """Return (t, xy) with xy in centimeters, averaged over available LEDs.

    Position is stored as one SpatialSeries per tracking LED. When two LEDs are
    present we average them to get the animal's head centroid.
    """
    pos = nwbfile.processing["behavior"]["Position"]
    series = list(pos.spatial_series.values())
    t = series[0].timestamps[:]
    xs, ys = [], []
    for ss in series:
        d = ss.data[:]
        xs.append(d[:, 0])
        ys.append(d[:, 1])
    x = np.nanmean(np.vstack(xs), axis=0)
    y = np.nanmean(np.vstack(ys), axis=0)
    xy = np.column_stack([x, y])
    return t, xy


def extract_units(nwbfile):
    """Return list of dicts with spike times and metadata for each unit."""
    u = nwbfile.units
    out = []
    for i in range(len(u)):
        out.append(dict(
            name=str(u["unit_name"][i]),
            spike_times=np.asarray(u["spike_times"][i]),
            histology=str(u["histology"][i]),
            depth=float(u["depth"][i]),
        ))
    return out


# ----------------------------------------------------------------------------
# Rate maps
# ----------------------------------------------------------------------------
def occupancy_map(t, xy, bins, box, sigma=1.0, min_count=1):
    """Time-in-bin occupancy map (seconds). Returns (occ, edges_x, edges_y).

    Bins visited fewer than ``min_count`` times are marked NaN (unvisited).
    """
    dt = np.median(np.diff(t))
    (lo, hi) = box
    edges = np.linspace(lo, hi, bins + 1)
    counts, ex, ey = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges, edges])
    occ = counts * dt  # counts -> seconds
    occ_s = gaussian_filter(occ, sigma)
    occ_s[counts < min_count] = np.nan  # unvisited bins
    return occ_s, ex, ey


def rate_map(spike_times, t, xy, occ, edges, sigma=1.0):
    """Occupancy-normalized, Gaussian-smoothed 2D firing-rate map (Hz)."""
    # position of the animal at each spike (nearest-neighbor interpolation)
    sx = np.interp(spike_times, t, xy[:, 0])
    sy = np.interp(spike_times, t, xy[:, 1])
    spk, _, _ = np.histogram2d(sx, sy, bins=[edges, edges])
    spk_s = gaussian_filter(spk, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        rm = spk_s / occ
    return rm


def spatial_information(rm, occ):
    """Skaggs spatial information (bits/spike)."""
    valid = np.isfinite(rm) & np.isfinite(occ) & (occ > 0)
    r = rm[valid]
    p = occ[valid] / occ[valid].sum()
    mean_r = np.sum(p * r)
    if mean_r <= 0:
        return 0.0
    nz = r > 0
    return float(np.sum(p[nz] * r[nz] / mean_r * np.log2(r[nz] / mean_r)))


# ----------------------------------------------------------------------------
# Spatial autocorrelogram + gridness
# ----------------------------------------------------------------------------
def autocorrelogram(rm, min_overlap=20):
    """2D spatial autocorrelogram using Pearson r at each spatial lag.

    Implements the standard grid-cell autocorrelation (Sargolini 2006):
    r(tx,ty) = [n*S(l1 l2) - S(l1)S(l2)] /
               sqrt([n S(l1^2)-S(l1)^2][n S(l2^2)-S(l2)^2])
    computed over the n bins where the map and its shifted copy overlap and are
    both finite. Returns an array of shape (2*ny-1, 2*nx-1) in [-1, 1].
    """
    m = rm.copy()
    mask = np.isfinite(m)
    m0 = np.where(mask, m, 0.0)
    ny, nx = m.shape
    ac = np.full((2 * ny - 1, 2 * nx - 1), np.nan)
    for iy, ly in enumerate(range(-(ny - 1), ny)):
        for ix, lx in enumerate(range(-(nx - 1), nx)):
            # overlap region between map and map shifted by (lx, ly)
            ax0, ax1 = max(0, lx), min(nx, nx + lx)
            ay0, ay1 = max(0, ly), min(ny, ny + ly)
            bx0, bx1 = max(0, -lx), min(nx, nx - lx)
            by0, by1 = max(0, -ly), min(ny, ny - ly)
            A = m0[ay0:ay1, ax0:ax1]
            Ma = mask[ay0:ay1, ax0:ax1]
            B = m0[by0:by1, bx0:bx1]
            Mb = mask[by0:by1, bx0:bx1]
            both = Ma & Mb
            n = both.sum()
            if n < min_overlap:
                continue
            a = A[both]
            b = B[both]
            sa, sb = a.sum(), b.sum()
            saa, sbb, sab = (a * a).sum(), (b * b).sum(), (a * b).sum()
            denom = np.sqrt((n * saa - sa * sa) * (n * sbb - sb * sb))
            if denom <= 0:
                continue
            ac[iy, ix] = (n * sab - sa * sb) / denom
    return ac


def _ring_peaks(ac, bin_size):
    """Find the 6 autocorrelogram peaks nearest the center (excluding center)."""
    cy, cx = np.array(ac.shape) // 2
    filt = maximum_filter(np.nan_to_num(ac, nan=-2), size=3)
    ismax = (ac == filt) & np.isfinite(ac) & (ac > 0.1)
    ys, xs = np.where(ismax)
    d = np.hypot(xs - cx, ys - cy)
    keep = d > 2  # exclude the central peak
    ys, xs, d = ys[keep], xs[keep], d[keep]
    order = np.argsort(d)[:6]
    px, py, pd = xs[order], ys[order], d[order]
    ang = np.degrees(np.arctan2(-(py - cy), px - cx)) % 360
    return dict(px=px, py=py, dist=pd, angle=ang,
                spacing_cm=float(np.median(pd)) * bin_size, cx=cx, cy=cy)


def gridness_score(ac, bin_size):
    """Sargolini gridness score from an annular ring of the autocorrelogram.

    The central-peak-excluded annulus is rotated by 30/60/90/120/150 deg and
    Pearson-correlated with the unrotated annulus. Gridness is
    min(corr60, corr120) - max(corr30, corr90, corr150), maximized over a set of
    outer radii. A true hexagonal grid scores > 0 (peaks recur at 60/120 deg).
    """
    from scipy.ndimage import rotate

    peaks = _ring_peaks(ac, bin_size)
    if len(peaks["dist"]) < 3:
        return dict(gridness=np.nan, **peaks)
    cy, cx = peaks["cy"], peaks["cx"]
    r_peak = float(np.median(peaks["dist"]))
    inner = 0.5 * r_peak

    yy, xx = np.mgrid[0:ac.shape[0], 0:ac.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)

    best = -np.inf
    best_outer = np.nan
    for outer in np.linspace(1.1 * r_peak, 2.0 * r_peak, 12):
        ring_mask = (rr >= inner) & (rr <= outer)
        base = np.where(ring_mask & np.isfinite(ac), ac, np.nan)
        corrs = {}
        for a in (30, 60, 90, 120, 150):
            rot = rotate(np.nan_to_num(base, nan=0.0), a, reshape=False, order=1)
            rot_mask = rotate(ring_mask.astype(float), a, reshape=False, order=1) > 0.5
            valid = ring_mask & rot_mask & np.isfinite(base)
            if valid.sum() < 20:
                corrs[a] = np.nan
                continue
            u = base[valid]
            v = rot[valid]
            if u.std() == 0 or v.std() == 0:
                corrs[a] = np.nan
                continue
            corrs[a] = float(np.corrcoef(u, v)[0, 1])
        vals = [corrs[a] for a in (30, 60, 90, 120, 150)]
        if any(np.isnan(vals)):
            continue
        g = min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])
        if g > best:
            best = g
            best_outer = outer
    out = dict(peaks)
    out.update(gridness=(best if np.isfinite(best) else np.nan),
               orientation_deg=float(np.min(peaks["angle"])) if len(peaks["angle"]) else np.nan,
               outer_radius=best_outer)
    return out


def analyze_unit(spike_times, t, xy, occ, edges, bin_size, sigma=1.0):
    """Convenience wrapper: rate map, autocorrelogram, gridness, spatial info."""
    rm = rate_map(spike_times, t, xy, occ, edges, sigma=sigma)
    ac = autocorrelogram(rm)
    g = gridness_score(ac, bin_size)
    si = spatial_information(rm, occ)
    return rm, ac, g, si


def shuffle_gridness(spike_times, t, xy, occ, edges, bin_size,
                     n_shuffles=30, min_shift=20.0, seed=0):
    """Null gridness distribution from circularly time-shifted spike trains.

    Each shuffle rigidly shifts all spikes by a random offset (>= ``min_shift``
    seconds) and wraps within the session, destroying the spike-position
    relationship while preserving the spike-train autostructure. This is the
    standard control for classifying grid cells (Sargolini 2006; Langston 2010).
    """
    rng = np.random.default_rng(seed)
    t0, t1 = t[0], t[-1]
    dur = t1 - t0
    null = np.full(n_shuffles, np.nan)
    for k in range(n_shuffles):
        shift = rng.uniform(min_shift, dur - min_shift)
        st = t0 + np.mod((spike_times - t0) + shift, dur)
        rm = rate_map(st, t, xy, occ, edges, sigma=1.0)
        null[k] = gridness_score(rm_autocorr(rm), bin_size)["gridness"]
    return null


def rm_autocorr(rm):
    return autocorrelogram(rm)
