"""Core routines for grid-cell analysis of DANDI:000582 (Sargolini/Moser MEC open-field data).

All spatial units are centimeters (the NWB files label position "meters" but the
recorded values span a 1x1 m box in cm, e.g. -50..50).
"""
import numpy as np
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

DANDISET = "000582"
BOX = 100.0          # enclosure side length (cm)
BIN = 3.0            # rate-map bin size (cm)
SMOOTH_SIGMA_BINS = 2.0   # Gaussian smoothing of rate map (in bins)
SPEED_THRESH = 2.5   # cm/s; samples slower than this are dropped (rest periods)


def list_assets():
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/"
    out = {}
    page = requests.get(url, params={"page_size": 200}).json()
    for a in page["results"]:
        out[a["path"]] = a["asset_id"]
    return out


def s3_url(asset_id):
    info = requests.get(
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/{asset_id}/"
    ).json()
    for u in info["contentUrl"]:
        if "s3.amazonaws.com" in u:
            return u
    raise RuntimeError("no s3 url")


def load_session(s3):
    """Return (pos_t, pos_xy[N,2], units_dict) for one session, cleaned of NaN samples."""
    rf = remfile.File(s3, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    nwbf = NWBHDF5IO(file=h5py.File(rf, "r")).read()
    ss = nwbf.processing["behavior"]["Position"].spatial_series["SpatialSeriesLED1"]
    t = np.asarray(ss.timestamps[:], dtype=float)
    xy = np.asarray(ss.data[:], dtype=float)
    good = np.isfinite(xy).all(axis=1) & np.isfinite(t)
    t, xy = t[good], xy[good]
    u = nwbf.units
    units = {u["unit_name"][i]: np.asarray(u["spike_times"][i], dtype=float)
             for i in range(len(u))}
    meta = dict(subject=nwbf.subject.subject_id,
                session=nwbf.session_id or "",
                histology=[u["histology"][i] for i in range(len(u))])
    return t, xy, units, meta


def speed_filter(t, xy):
    """Boolean mask of samples where running speed exceeds SPEED_THRESH."""
    dt = np.gradient(t)
    v = np.sqrt(np.gradient(xy[:, 0])**2 + np.gradient(xy[:, 1])**2) / dt
    return v > SPEED_THRESH, v


def rate_map(spike_t, t, xy, mask=None, box=BOX, bin_sz=BIN, sigma=SMOOTH_SIGMA_BINS):
    """2D smoothed firing-rate map. Returns (rate[H,W], occ_seconds, edges)."""
    if mask is not None:
        t, xy = t[mask], xy[mask]
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    # symmetric box centred on the data extent
    edges_x = np.arange(x0, x1 + bin_sz, bin_sz)
    edges_y = np.arange(y0, y1 + bin_sz, bin_sz)
    dt = np.median(np.diff(t))
    occ, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges_x, edges_y])
    occ = occ * dt  # seconds per bin

    # assign each spike to nearest tracked position by interpolation
    sx = np.interp(spike_t, t, xy[:, 0])
    sy = np.interp(spike_t, t, xy[:, 1])
    inb = (spike_t >= t[0]) & (spike_t <= t[-1])
    spk, _, _ = np.histogram2d(sx[inb], sy[inb], bins=[edges_x, edges_y])

    occ_s = gaussian_filter(occ, sigma)
    spk_s = gaussian_filter(spk, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = spk_s / occ_s
    rate[occ < dt * 0.05] = np.nan  # unvisited bins
    return rate.T, occ.T, (edges_x, edges_y)


def spatial_autocorr(rate):
    """Normalised (Pearson) spatial autocorrelogram of a rate map with NaNs."""
    r = rate.copy()
    valid = np.isfinite(r).astype(float)
    r[~np.isfinite(r)] = 0.0
    n = correlate2d(r, r, mode="full")
    sum1 = correlate2d(r, valid, mode="full")
    sum2 = correlate2d(valid, r, mode="full")
    ss1 = correlate2d(r * r, valid, mode="full")
    ss2 = correlate2d(valid, r * r, mode="full")
    cnt = correlate2d(valid, valid, mode="full")
    with np.errstate(divide="ignore", invalid="ignore"):
        num = n - sum1 * sum2 / cnt
        den = np.sqrt((ss1 - sum1**2 / cnt) * (ss2 - sum2**2 / cnt))
        ac = num / den
    ac[cnt < 20] = np.nan
    return ac


def grid_score(ac):
    """Gridness score from a spatial autocorrelogram (Sargolini/Hafting method).

    Correlate the annulus (excluding the central peak) with rotated copies at
    30/60/90/120/150 deg; gridness = min(corr at 60,120) - max(corr at 30,90,150).
    Returns (score, grid_spacing_cm_or_nan).
    """
    from scipy.ndimage import rotate
    ny, nx = ac.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    # estimate ring radii from the six inner peaks: find central peak radius
    a = ac.copy()
    a[~np.isfinite(a)] = 0.0
    # central peak extent: first radius where azimuthal mean drops below 0.2
    rmax = int(dist.max())
    prof = np.array([a[(dist >= rr) & (dist < rr + 1)].mean() if np.any((dist >= rr) & (dist < rr + 1)) else 0
                     for rr in range(rmax)])
    # inner radius = first local min after the center peak
    inner = 1
    while inner < rmax - 1 and prof[inner] > prof[inner + 1]:
        inner += 1
    # outer radius: capture the 6 surrounding peaks (~2.5x inner as default), bounded by map
    outer = min(int(inner * 2.6), cx - 1, cy - 1)
    if outer <= inner + 2:
        return np.nan, np.nan

    ring = (dist >= inner) & (dist <= outer)
    base = np.where(ring, a, 0.0)
    ringmask = ring.astype(float)

    def rot_corr(deg):
        ar = rotate(base, deg, reshape=False, order=1)
        mr = rotate(ringmask, deg, reshape=False, order=1) > 0.5
        m = (ringmask > 0.5) & mr
        if m.sum() < 30:
            return np.nan
        x, y = base[m], ar[m]
        if x.std() == 0 or y.std() == 0:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    c = {d: rot_corr(d) for d in (30, 60, 90, 120, 150)}
    score = min(c[60], c[120]) - max(c[30], c[90], c[150])
    # grid spacing = radius of the nearest of the six peaks ~ inner..outer peak
    spacing = (inner + outer) / 2.0 * BIN  # rough
    return score, spacing
