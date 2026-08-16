"""Helpers for demonstrating grid cells in medial entorhinal cortex.

Data source: DANDI:000582, "Conjunctive Representation of Position, Direction,
and Velocity in Entorhinal Cortex" (Sargolini et al., Science 2006; Moser lab).

Everything is streamed from the DANDI S3 bucket with remfile + a local disk
cache; nothing is downloaded in full.  Analysis is built on Pynapple objects
(TsGroup, Tsd, TsdFrame, IntervalSet).
"""

import json
import os
import urllib.request

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import ndimage, signal

DANDISET = "000582"
API = "https://api.dandiarchive.org/api/dandisets"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000582")
ASSET_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets.json")

# --- analysis parameters (fixed once, used everywhere) ------------------------
BIN_CM = 3.0           # spatial bin size for rate maps
SMOOTH_BINS = 2.0      # Gaussian sigma for rate-map smoothing, in bins (6 cm)
SPEED_MIN = 2.5        # cm/s, exclude immobility from spatial analyses
SPEED_MAX = 100.0      # cm/s, reject tracking jumps
HD_BINS = 60           # head-direction tuning-curve bins (6 deg)


# =============================================================================
# Asset listing / loading
# =============================================================================
def list_assets(refresh=False):
    """Return [{'path', 'id', 'size'}] for every NWB file in the dandiset."""
    if os.path.exists(ASSET_JSON) and not refresh:
        return json.load(open(ASSET_JSON))
    url = f"{API}/{DANDISET}/versions/draft/assets/?page_size=200"
    out = []
    while url:
        d = json.load(urllib.request.urlopen(url))
        out += d["results"]
        url = d["next"]
    out = sorted(
        ({"path": a["path"], "id": a["asset_id"], "size": a["size"]} for a in out),
        key=lambda a: a["path"],
    )
    json.dump(out, open(ASSET_JSON, "w"), indent=0)
    return out


def open_nwb(asset):
    """Stream one NWB file (dict from list_assets, or a path string)."""
    if isinstance(asset, str):
        asset = [a for a in list_assets() if a["path"] == asset][0]
    url = f"{API}/{DANDISET}/versions/draft/assets/{asset['id']}/download/"
    fh = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR)), "r")
    return NWBHDF5IO(file=fh, load_namespaces=True).read()


def _interp_nans(x, t, max_gap_s=0.5):
    """Linearly interpolate short NaN gaps; leave long gaps as NaN."""
    x = x.copy()
    bad = np.isnan(x)
    if not bad.any():
        return x
    good = ~bad
    x[bad] = np.interp(t[bad], t[good], x[good])
    # re-insert NaN for gaps longer than max_gap_s
    idx = np.flatnonzero(bad)
    for start, stop in _runs(idx):
        if t[stop] - t[start] > max_gap_s:
            x[start : stop + 1] = np.nan
    return x


def _runs(idx):
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, stops))


def load_session(asset):
    """Load one session into Pynapple objects.

    Returns a dict with:
        units    : nap.TsGroup of sorted units (metadata: unit_name, histology, ...)
        position : nap.TsdFrame ['x', 'y'] in cm, head/body midpoint of the two LEDs
        hd       : nap.Tsd head direction in radians (from the LED1->LED2 vector)
        speed    : nap.Tsd running speed in cm/s
        run_ep   : nap.IntervalSet where speed is within [SPEED_MIN, SPEED_MAX]
        meta     : session/subject descriptors
    """
    nwb = open_nwb(asset)
    pos = nwb.processing["behavior"]["Position"]
    s1 = pos["SpatialSeriesLED1"]
    t = np.asarray(s1.timestamps[:], dtype=float)
    p1 = np.asarray(s1.data[:], dtype=float)
    p1 = np.column_stack([_interp_nans(p1[:, i], t) for i in range(2)])

    # A few sessions were tracked with a single LED: then there is no head
    # direction and the animal's position is taken from that LED alone.
    has_led2 = "SpatialSeriesLED2" in pos.spatial_series
    if has_led2:
        s2 = pos["SpatialSeriesLED2"]
        p2 = np.asarray(s2.data[:], dtype=float)
        p2 = np.column_stack([_interp_nans(p2[:, i], t) for i in range(2)])
        mid = 0.5 * (p1 + p2)
        hd = np.arctan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    else:
        mid = p1
        hd = np.full(len(t), np.nan)

    ok = np.isfinite(mid).all(axis=1)
    t_ok, mid_ok, hd_ok = t[ok], mid[ok], hd[ok]

    # speed from a lightly smoothed trajectory (median filter kills tracking jitter)
    xs = ndimage.median_filter(mid_ok[:, 0], size=5)
    ys = ndimage.median_filter(mid_ok[:, 1], size=5)
    dt = np.gradient(t_ok)
    speed = np.hypot(np.gradient(xs) / dt, np.gradient(ys) / dt)
    speed = ndimage.uniform_filter1d(speed, size=13)

    position = nap.TsdFrame(t=t_ok, d=mid_ok, columns=["x", "y"])
    hd_tsd = nap.Tsd(t=t_ok, d=np.mod(hd_ok, 2 * np.pi))
    speed_tsd = nap.Tsd(t=t_ok, d=speed)
    run_ep = speed_tsd.threshold(SPEED_MIN, "above").threshold(SPEED_MAX, "below").time_support

    udf = nwb.units.to_dataframe()
    spikes = {}
    for i, row in udf.iterrows():
        st = np.asarray(row["spike_times"], dtype=float)
        spikes[int(i)] = nap.Ts(t=np.sort(st))
    meta = {
        "unit_name": list(udf["unit_name"]),
        "histology": list(udf["histology"]),
        "hemisphere": list(udf["hemisphere"]),
        "depth": list(udf["depth"]),
    }
    units = nap.TsGroup(spikes, time_support=position.time_support, **meta)

    return {
        "units": units,
        "position": position,
        "hd": hd_tsd,
        "speed": speed_tsd,
        "run_ep": run_ep,
        "meta": {
            "session_id": nwb.session_id,
            "subject_id": nwb.subject.subject_id,
            "path": asset["path"] if isinstance(asset, dict) else asset,
            "duration_s": float(t_ok[-1] - t_ok[0]),
            "has_hd": bool(has_led2),
            "session_description": nwb.session_description,
        },
        "nwb": nwb,
    }


# =============================================================================
# Rate maps
# =============================================================================
def map_edges(position, bin_cm=BIN_CM, pad=1.0):
    """Common bin edges for x and y covering the visited arena (square bins)."""
    x, y = position["x"].values, position["y"].values
    lo = min(np.percentile(x, 0.1), np.percentile(y, 0.1)) - pad
    hi = max(np.percentile(x, 99.9), np.percentile(y, 99.9)) + pad
    n = int(np.ceil((hi - lo) / bin_cm))
    return lo + np.arange(n + 1) * bin_cm


def rate_maps(units, position, ep, edges, sigma=SMOOTH_BINS, min_occ=0.04):
    """Occupancy-normalised, Gaussian-smoothed 2-D rate maps.

    Spike counts and occupancy are smoothed separately and then divided
    (Moser-lab convention); bins with less than `min_occ` seconds of raw
    occupancy (2 tracking samples) are treated as unvisited.

    Returns (maps [n_units, ny, nx], occupancy_seconds [ny, nx], centers).
    """
    tc = nap.compute_tuning_curves(
        units, position, bins=[edges, edges], epochs=ep, return_counts=True
    )
    counts = np.asarray(tc.values)                        # (n_units, nx, ny)
    # Pynapple returns occupancy as a sample count; convert to seconds.
    occ = np.asarray(tc.attrs["occupancy"], dtype=float) / float(tc.attrs["fs"])

    unvisited = occ < min_occ
    occ_s = _nan_gaussian(np.where(unvisited, np.nan, occ), sigma)
    out = np.empty((counts.shape[0], occ.shape[1], occ.shape[0]))
    for i in range(counts.shape[0]):
        c = _nan_gaussian(np.where(unvisited, np.nan, counts[i]), sigma)
        m = c / occ_s
        m[unvisited] = np.nan
        out[i] = m.T                                    # -> (ny, nx) for imshow
    centers = 0.5 * (edges[:-1] + edges[1:])
    return out, occ.T, centers


def _nan_gaussian(a, sigma):
    """Gaussian smoothing that ignores NaNs (normalised by the smoothed mask)."""
    if sigma <= 0:
        return a
    v = np.nan_to_num(a, nan=0.0)
    w = np.isfinite(a).astype(float)
    vs = ndimage.gaussian_filter(v, sigma, mode="constant")
    ws = ndimage.gaussian_filter(w, sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = vs / ws
    out[ws < 1e-6] = np.nan
    return out


# =============================================================================
# Spatial autocorrelogram and gridness
# =============================================================================
def spatial_autocorr(rmap, min_overlap=20):
    """Unbiased Pearson spatial autocorrelogram (Sargolini et al. 2006, eq. 1).

    r(tx, ty) is computed over the bins where both the map and the shifted map
    are defined; lags with fewer than `min_overlap` overlapping bins are NaN.
    """
    f = np.nan_to_num(rmap, nan=0.0)
    m = np.isfinite(rmap).astype(float)
    c = lambda a, b: signal.fftconvolve(a, b[::-1, ::-1], mode="full")
    n = c(m, m)
    sfg = c(f, f)
    sf = c(f, m)
    sg = c(m, f)
    sff = c(f * f, m)
    sgg = c(m, f * f)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * sfg - sf * sg
        den = np.sqrt(np.clip(n * sff - sf**2, 0, None) * np.clip(n * sgg - sg**2, 0, None))
        r = num / den
    r[n < min_overlap] = np.nan
    return np.clip(r, -1, 1)


def _radial_profile(ac):
    ny, nx = ac.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.hypot(yy - cy, xx - cx)
    rint = rr.astype(int)
    nmax = int(rr.max())
    prof = np.full(nmax + 1, np.nan)
    for k in range(nmax + 1):
        sel = (rint == k) & np.isfinite(ac)
        if sel.sum() >= 3:
            prof[k] = ac[sel].mean()
    return prof


def central_peak_radius(ac):
    """Radius (in bins) of the first minimum of the radial autocorr profile."""
    prof = _radial_profile(ac)
    prof = np.where(np.isfinite(prof), prof, 0.0)
    for k in range(1, len(prof) - 1):
        if prof[k] <= prof[k + 1] and prof[k] < prof[0]:
            return max(k, 2)
    return max(len(prof) // 6, 2)


def gridness(ac, r_in=None, n_radii=8):
    """Gridness score: max over expanding annuli of

        min(r60, r120) - max(r30, r90, r150)

    where rA is the Pearson correlation between the annulus of the
    autocorrelogram and the same annulus rotated by A degrees.
    """
    ny, nx = ac.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.hypot(yy - cy, xx - cx)
    r_max = min(cy, cx)
    if r_in is None:
        r_in = central_peak_radius(ac)
    # keep at least a usable annulus even when the central peak is broad
    r_in = min(r_in, max(r_max - 6, 2))
    if r_max <= r_in + 3:
        return np.nan, np.nan
    rot = {a: _rotate_about_center(ac, a) for a in (30, 60, 90, 120, 150)}
    best, best_r = -np.inf, np.nan
    for r_out in np.linspace(r_in + 3, r_max, n_radii):
        ring = (rr >= r_in) & (rr <= r_out)
        cs = {}
        for a, ra in rot.items():
            sel = ring & np.isfinite(ac) & np.isfinite(ra)
            if sel.sum() < 20:
                cs = None
                break
            cs[a] = np.corrcoef(ac[sel], ra[sel])[0, 1]
        if cs is None:
            continue
        g = min(cs[60], cs[120]) - max(cs[30], cs[90], cs[150])
        if g > best:
            best, best_r = g, r_out
    return (best if np.isfinite(best) else np.nan), best_r


def _rotate_about_center(a, deg):
    filled = np.nan_to_num(a, nan=0.0)
    mask = np.isfinite(a).astype(float)
    rf = ndimage.rotate(filled, deg, reshape=False, order=1, cval=0.0)
    rm = ndimage.rotate(mask, deg, reshape=False, order=1, cval=0.0)
    out = np.where(rm > 0.5, rf, np.nan)
    return out


def autocorr_peaks(ac, bin_cm=BIN_CM, n_peaks=6):
    """Locate the six autocorrelogram peaks closest to the centre.

    Returns (spacing_cm, orientation_deg, ellipticity, peak_xy_bins).
    """
    ny, nx = ac.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    a = np.nan_to_num(ac, nan=-1.0)
    loc = (ndimage.maximum_filter(a, size=5) == a) & (a > 0.1)
    ys, xs = np.nonzero(loc)
    d = np.hypot(ys - cy, xs - cx)
    keep = d > max(central_peak_radius(ac) * 0.8, 2)
    ys, xs, d = ys[keep], xs[keep], d[keep]
    if len(d) == 0:
        return np.nan, np.nan, np.nan, np.zeros((0, 2))
    order = np.argsort(d)[:n_peaks]
    ys, xs, d = ys[order], xs[order], d[order]
    spacing = float(np.median(d) * bin_cm)
    ang = np.degrees(np.arctan2(ys - cy, xs - cx))
    orientation = float(np.min(np.mod(ang, 60.0)))
    ellipticity = float(d.max() / d.min()) if len(d) >= 3 else np.nan
    return spacing, orientation, ellipticity, np.column_stack([xs, ys])


# =============================================================================
# Other single-cell measures
# =============================================================================
def spatial_information(rmap, occ):
    """Skaggs spatial information in bits/spike."""
    valid = np.isfinite(rmap) & (occ > 0)
    r, o = rmap[valid], occ[valid]
    p = o / o.sum()
    mean_r = np.sum(p * r)
    if mean_r <= 0:
        return np.nan
    nz = r > 0
    return float(np.sum(p[nz] * (r[nz] / mean_r) * np.log2(r[nz] / mean_r)))


def map_correlation(a, b):
    """Pearson correlation between two rate maps over commonly visited bins."""
    v = np.isfinite(a) & np.isfinite(b)
    if v.sum() < 20:
        return np.nan
    return float(np.corrcoef(a[v], b[v])[0, 1])


def hd_tuning(units, hd, ep, nbins=HD_BINS):
    """Head-direction tuning curves and mean vector length per unit."""
    tc = nap.compute_tuning_curves(units, hd, bins=nbins, range=[(0, 2 * np.pi)], epochs=ep)
    curves = np.asarray(tc.values).T                  # -> (nbins, n_units)
    centers = np.asarray(tc.coords[tc.dims[1]].values)  # dims are ('unit', feature)
    mvl = []
    for i in range(curves.shape[1]):
        c = np.nan_to_num(curves[:, i])
        if c.sum() <= 0:
            mvl.append(np.nan)
            continue
        mvl.append(float(np.abs(np.sum(c * np.exp(1j * centers))) / c.sum()))
    return curves, centers, np.array(mvl)


def shift_spikes(ts, shift, t0, t1):
    """Circularly shift spike times inside [t0, t1] (shuffle control)."""
    T = t1 - t0
    return nap.Ts(t=np.sort(t0 + np.mod(ts.times() - t0 + shift, T)))


def rotational_correlation(ac, r_in=None, r_out=None, angles=None):
    """Correlation of the autocorrelogram annulus with itself, vs rotation angle."""
    ny, nx = ac.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.hypot(yy - cy, xx - cx)
    if r_in is None:
        r_in = min(central_peak_radius(ac), max(min(cy, cx) - 6, 2))
    if r_out is None:
        r_out = min(cy, cx)
    if angles is None:
        angles = np.arange(0, 181, 3)
    ring = (rr >= r_in) & (rr <= r_out)
    out = []
    for a in angles:
        ra = _rotate_about_center(ac, a)
        sel = ring & np.isfinite(ac) & np.isfinite(ra)
        out.append(np.corrcoef(ac[sel], ra[sel])[0, 1] if sel.sum() > 20 else np.nan)
    return np.asarray(angles, dtype=float), np.asarray(out), (r_in, r_out)
