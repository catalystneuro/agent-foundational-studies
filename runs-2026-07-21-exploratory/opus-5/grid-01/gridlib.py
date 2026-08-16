"""Core routines for demonstrating grid cells in medial entorhinal cortex.

Data source: DANDI:000582 (Sargolini et al., Science 2006), tetrode recordings
from dorsocaudal MEC of freely foraging Long-Evans rats.

Everything here works on real DANDI data streamed with remfile; there is no
simulation anywhere in this pipeline.
"""

import json
import warnings

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO
from scipy import ndimage, signal
from skimage.feature import peak_local_max

warnings.filterwarnings("ignore", category=UserWarning)

DANDISET = "000582"
CACHE_DIR = "/tmp/remfile_cache"

# ---------------------------------------------------------------------------
# Analysis parameters (fixed across all sessions)
# ---------------------------------------------------------------------------
BIN_CM = 3.0           # spatial bin size for rate maps
SMOOTH_SIGMA_CM = 4.5  # Gaussian smoothing sigma for rate maps
MIN_OCC_S = 0.06       # minimum occupancy per bin (s) to keep a bin
SPEED_MIN_CMS = 2.5    # running-speed threshold (classic grid-cell criterion)
SPEED_MAX_CMS = 100.0  # reject tracking jumps
MIN_MEAN_RATE_HZ = 0.1 # exclude near-silent units
MIN_SPIKES = 100


def asset_url(asset_id):
    return (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
            f"/versions/draft/assets/{asset_id}/download/")


def load_session(asset, keep_lfp=False):
    """Stream one NWB session and wrap it with pynapple.

    Returns a dict with a pynapple TsGroup of units, a TsdFrame of position
    (cm), a Tsd of speed (cm/s), and a Tsd of head direction (rad) when the
    session tracked two LEDs.
    """
    h5 = h5py.File(remfile.File(asset_url(asset["id"]),
                                disk_cache=remfile.DiskCache(CACHE_DIR)), "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)

    pos_iface = nwbfile.processing["behavior"]["Position"]
    led_keys = sorted(pos_iface.spatial_series.keys())
    led1 = pos_iface.spatial_series[led_keys[0]]
    t = np.asarray(led1.timestamps[:], dtype=float)
    xy1 = np.asarray(led1.data[:], dtype=float)

    xy2 = None
    if len(led_keys) > 1:
        led2 = pos_iface.spatial_series[led_keys[1]]
        xy2 = np.asarray(led2.data[:], dtype=float)

    # Head direction from the two LEDs (LED1 is the larger/front light).
    if xy2 is not None:
        d = xy2 - xy1
        hd = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    else:
        hd = None

    # Interpolate short tracking gaps in the midpoint trajectory.
    xy = xy1 if xy2 is None else 0.5 * (xy1 + xy2)
    good = np.isfinite(xy).all(axis=1)
    for j in range(2):
        xy[:, j] = np.interp(t, t[good], xy[good, j])
    if hd is not None:
        hgood = np.isfinite(hd)
        cs = np.interp(t, t[hgood], np.cos(hd[hgood]))
        sn = np.interp(t, t[hgood], np.sin(hd[hgood]))
        hd = np.arctan2(sn, cs) % (2 * np.pi)

    position = nap.TsdFrame(t=t, d=xy, columns=["x", "y"])

    # Speed from smoothed position (median filter removes single-frame jitter).
    dt = float(np.median(np.diff(t)))
    xs = ndimage.uniform_filter1d(xy[:, 0], 5)
    ys = ndimage.uniform_filter1d(xy[:, 1], 5)
    spd = np.hypot(np.gradient(xs, t), np.gradient(ys, t))
    speed = nap.Tsd(t=t, d=spd)

    units = nwb["units"]
    udf = nwbfile.units.to_dataframe()

    out = dict(
        path=asset["path"],
        subject=asset["path"].split("/")[0].replace("sub-", ""),
        session_id=nwbfile.session_id,
        units=units,
        unit_names=list(udf["unit_name"]),
        depth=list(udf["depth"]) if "depth" in udf else [np.nan] * len(udf),
        histology=list(udf["histology"]) if "histology" in udf else [""] * len(udf),
        position=position,
        speed=speed,
        hd=nap.Tsd(t=t, d=hd) if hd is not None else None,
        dt=dt,
        duration=float(t[-1] - t[0]),
        io=io,
        nwbfile=nwbfile,
    )
    if keep_lfp:
        out["lfp"] = nwb["ElectricalSeriesLFP"]
    return out


def run_epochs(session):
    """IntervalSet of epochs where the animal is running (speed-filtered)."""
    spd = session["speed"]
    ok = (spd.values >= SPEED_MIN_CMS) & (spd.values <= SPEED_MAX_CMS)
    return nap.Tsd(t=spd.index.values, d=ok.astype(float)).threshold(0.5).time_support


# ---------------------------------------------------------------------------
# Rate maps
# ---------------------------------------------------------------------------
def map_edges(position, ep):
    """Common bin edges for a session, from the sampled arena extent."""
    p = position.restrict(ep)
    edges = []
    for c in ["x", "y"]:
        lo, hi = np.nanmin(p[c].values), np.nanmax(p[c].values)
        n = max(int(np.ceil((hi - lo) / BIN_CM)), 8)
        edges.append(np.linspace(lo, lo + n * BIN_CM, n + 1))
    return edges


def occupancy_map(position, ep, edges, dt):
    p = position.restrict(ep)
    occ, _, _ = np.histogram2d(p["x"].values, p["y"].values, bins=edges)
    return occ * dt  # seconds per bin


def rate_map(spike_times, position, ep, edges, occ_s, smooth=True):
    """Smoothed firing-rate map (Hz), NaN in bins that were barely visited.

    Spike positions come from pynapple's `value_from`, which interpolates the
    tracked position at each spike time.
    """
    spk = spike_times.restrict(ep)
    if len(spk) == 0:
        return np.full(occ_s.shape, np.nan), np.zeros(occ_s.shape)
    sp = spk.value_from(position)
    cnt, _, _ = np.histogram2d(sp["x"].values, sp["y"].values, bins=edges)

    valid = occ_s >= MIN_OCC_S
    if smooth:
        sigma = SMOOTH_SIGMA_CM / BIN_CM
        c = ndimage.gaussian_filter(np.where(valid, cnt, 0.0), sigma, mode="constant")
        o = ndimage.gaussian_filter(np.where(valid, occ_s, 0.0), sigma, mode="constant")
        rm = np.where(o > 1e-9, c / np.maximum(o, 1e-9), np.nan)
    else:
        rm = np.where(valid, cnt / np.maximum(occ_s, 1e-9), np.nan)
    rm[~valid] = np.nan
    return rm, cnt


def spatial_information(rate, occ_s):
    """Skaggs spatial information in bits/spike."""
    m = np.isfinite(rate) & (occ_s > 0)
    if m.sum() == 0:
        return np.nan
    p = occ_s[m] / occ_s[m].sum()
    r = rate[m]
    mr = np.sum(p * r)
    if mr <= 0:
        return np.nan
    nz = r > 0
    return float(np.sum(p[nz] * (r[nz] / mr) * np.log2(r[nz] / mr)))


# ---------------------------------------------------------------------------
# Spatial autocorrelogram and grid score
# ---------------------------------------------------------------------------
def spatial_autocorr(rm, min_bins=20):
    """Pearson spatial autocorrelogram (Hafting et al. 2005, eq. 1)."""
    m = np.isfinite(rm)
    x = np.where(m, rm, 0.0).astype(float)
    v = m.astype(float)

    def cc(a, b):
        return signal.fftconvolve(a, b[::-1, ::-1], mode="full")

    n = cc(v, v)
    s1, s2 = cc(x, v), cc(v, x)
    q1, q2 = cc(x * x, v), cc(v, x * x)
    p = cc(x, x)

    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * p - s1 * s2
        den = np.sqrt(np.maximum(n * q1 - s1 ** 2, 0)) * \
              np.sqrt(np.maximum(n * q2 - s2 ** 2, 0))
        r = num / den
    r[n < min_bins] = np.nan
    return np.clip(r, -1, 1)


def _radial_profile(ac):
    cy, cx = (np.array(ac.shape) - 1) / 2.0
    yy, xx = np.mgrid[: ac.shape[0], : ac.shape[1]]
    rr = np.hypot(yy - cy, xx - cx)
    return rr


def central_peak_radius(ac, rr):
    """Radius (in bins) of the central peak: first minimum of the radial mean."""
    rmax = int(rr.max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = np.array([np.nanmean(ac[(rr >= k) & (rr < k + 1)])
                         for k in range(rmax)])
    prof = np.where(np.isfinite(prof), prof, 0.0)
    k = 1
    while k < len(prof) - 1 and prof[k + 1] < prof[k]:
        k += 1
    return max(k, 2)


def grid_score(ac, return_curve=False):
    """Gridness score with an expanding annulus (max over outer radii).

    gridness = min(r60, r120) - max(r30, r90, r150), where rθ is the Pearson
    correlation between the annulus of the autocorrelogram and a copy rotated
    by θ degrees.
    """
    rr = _radial_profile(ac)
    r_in = central_peak_radius(ac, rr)
    r_max = int(min(ac.shape) // 2)
    if r_max <= r_in + 4:
        return (np.nan, {}) if return_curve else np.nan

    a = np.where(np.isfinite(ac), ac, 0.0)
    valid = np.isfinite(ac)
    rot = {}
    for ang in (30, 60, 90, 120, 150):
        rot[ang] = (ndimage.rotate(a, ang, reshape=False, order=1, mode="constant"),
                    ndimage.rotate(valid.astype(float), ang, reshape=False,
                                   order=1, mode="constant") > 0.9)

    best, best_r, curve = -np.inf, np.nan, []
    for r_out in range(r_in + 5, r_max + 1):
        ann = (rr >= r_in) & (rr <= r_out) & valid
        cors = {}
        for ang, (ra, rv) in rot.items():
            m = ann & rv
            if m.sum() < 50:
                cors = None
                break
            u, w = ac[m], ra[m]
            if np.std(u) < 1e-9 or np.std(w) < 1e-9:
                cors = None
                break
            cors[ang] = float(np.corrcoef(u, w)[0, 1])
        if cors is None:
            continue
        g = min(cors[60], cors[120]) - max(cors[30], cors[90], cors[150])
        curve.append((r_out, g, cors))
        if g > best:
            best, best_r = g, r_out
    if not np.isfinite(best):
        return (np.nan, {}) if return_curve else np.nan
    if return_curve:
        info = dict(r_in=r_in, r_out=best_r, curve=curve,
                    cors=dict(next(c[2] for c in curve if c[0] == best_r)))
        return float(best), info
    return float(best)


def grid_geometry(ac, bin_cm=BIN_CM):
    """Spacing (cm) and orientation (deg) from the six peaks nearest the centre."""
    a = np.where(np.isfinite(ac), ac, -1.0)
    pk = peak_local_max(a, min_distance=3, threshold_abs=0.05, exclude_border=False)
    cy, cx = (np.array(ac.shape) - 1) / 2.0
    d = np.hypot(pk[:, 0] - cy, pk[:, 1] - cx)
    keep = d > 3
    pk, d = pk[keep], d[keep]
    if len(pk) < 3:
        return np.nan, np.nan, pk
    order = np.argsort(d)[:6]
    pk, d = pk[order], d[order]
    spacing = float(np.median(d) * bin_cm)
    ang = np.degrees(np.arctan2(-(pk[:, 0] - cy), pk[:, 1] - cx)) % 180.0
    ang60 = ang % 60.0
    orientation = float(np.min(ang60))
    return spacing, orientation, pk


# ---------------------------------------------------------------------------
# Head direction tuning
# ---------------------------------------------------------------------------
def hd_tuning(spike_times, hd, ep, nbins=36):
    tc = nap.compute_1d_tuning_curves(
        nap.TsGroup({0: spike_times}), hd, nb_bins=nbins,
        minmax=(0, 2 * np.pi), ep=ep)
    curve = tc[0].values
    centers = tc.index.values
    if np.nansum(curve) <= 0:
        return curve, centers, 0.0, np.nan
    w = np.nan_to_num(curve)
    z = np.sum(w * np.exp(1j * centers)) / np.sum(w)
    return curve, centers, float(np.abs(z)), float(np.angle(z) % (2 * np.pi))


# ---------------------------------------------------------------------------
# Shuffling
# ---------------------------------------------------------------------------
def shifted_spikes(spike_times, t0, t1, offset):
    """Circularly shift spike times within the session (standard grid shuffle)."""
    ts = spike_times.index.values
    T = t1 - t0
    return nap.Ts(t=np.sort(t0 + np.mod(ts - t0 + offset, T)))


def load_assets():
    return json.load(open("assets.json"))


def rotational_profile(ac, r_in, r_out, angles=np.arange(0, 181, 3)):
    """Correlation of the autocorrelogram annulus with rotated copies."""
    rr = _radial_profile(ac)
    valid = np.isfinite(ac)
    ann0 = (rr >= r_in) & (rr <= r_out) & valid
    a = np.where(valid, ac, 0.0)
    out = []
    for ang in angles:
        ra = ndimage.rotate(a, ang, reshape=False, order=1, mode="constant")
        rv = ndimage.rotate(valid.astype(float), ang, reshape=False, order=1,
                            mode="constant") > 0.9
        m = ann0 & rv
        out.append(np.corrcoef(ac[m], ra[m])[0, 1] if m.sum() > 50 else np.nan)
    return np.asarray(angles), np.asarray(out)
