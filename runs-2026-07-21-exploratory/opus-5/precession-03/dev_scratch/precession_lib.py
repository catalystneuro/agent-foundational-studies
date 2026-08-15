"""Core routines for theta phase precession analysis on DANDI:000044.

Grosmark & Buzsaki (2016), dual-hemisphere CA1 silicon probe recordings while
rats ran back and forth on a 1.6 m linear track.
"""
import numpy as np
import h5py
import remfile
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

DANDISET = "000044"
VERSION = "0.250624.0426"
LFP_FS = 1250.0
THETA_BAND = (6.0, 10.0)
SPEED_THRESH = 0.10          # m/s, running threshold
POS_BIN = 0.04               # m, spatial bin for rate maps
TRACK_LEN = 1.6              # m

SESSIONS = {
    "Achilles-10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Cicero-09012014": "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Gatsby-08022013": None,   # filled in by resolve_assets()
    "Buddy-06272013": None,
}


def resolve_assets():
    """Map session name -> DANDI asset id for every asset in the dandiset."""
    import requests
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/"
    out = {}
    for a in requests.get(url, params={"page_size": 100}).json()["results"]:
        ses = a["path"].split("_ses-")[1].split("_")[0]
        out[ses] = a["asset_id"]
    return out


def open_session(asset_id, cache_dir="/tmp/remfile_cache"):
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/"
           f"{VERSION}/assets/{asset_id}/download/")
    return h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(cache_dir)), "r")


# ---------------------------------------------------------------- loading ---

def load_position(h5):
    """Linearized position on the maze as a pynapple Tsd (NaN samples dropped).

    NOTE on a quirk of this NWB conversion: the `rate` attribute of
    `starting_time` holds the sampling *period* (0.0256 s), not the rate.
    Using it as a rate would stretch the behaviour 1500-fold past the end of
    the session; treating it as dt reproduces the 2067.5 s maze epoch exactly.

    Returns (Tsd, dt, maze) where `maze` describes the geometry: the dandiset
    contains 1.6 m and 2 m linear tracks and a ~2.9 m circular maze, and the
    circular sessions need a wrapping coordinate.
    """
    key = [k for k in h5["processing/behavior"] if "LinearizedPosition" in k][0]
    g = h5[f"processing/behavior/{key}"]
    g = g[list(g)[0]]
    t0 = float(g["starting_time"][()])
    dt = float(g["starting_time"].attrs["rate"])
    d = g["data"][:, 0]
    t = t0 + np.arange(len(d)) * dt
    ok = np.isfinite(d)
    pos = nap.Tsd(t=t[ok], d=d[ok])
    circular = key.startswith("Circular")
    if circular:
        length = float(np.ceil(pos.values.max() * 100) / 100)
    else:
        length = float(key.split("mLinear")[0])
    maze = dict(name=key.replace("LinearizedPosition", ""), circular=circular,
                length=length)
    return pos, dt, maze


def load_units(h5):
    """CA1 units as a pynapple TsGroup with cell_type / location metadata."""
    st = h5["units/spike_times"][:]
    idx = h5["units/spike_times_index"][:]
    bounds = np.r_[0, idx]
    dec = lambda a: np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    spikes = {i: st[bounds[i]:bounds[i + 1]] for i in range(len(idx))}
    tsg = nap.TsGroup(spikes)
    tsg.set_info(cell_type=dec(h5["units/cell_type"][:]),
                 location=dec(h5["units/location"][:]),
                 shank_id=h5["units/shank_id"][:])
    return tsg


def load_lfp_channel(h5, ch, t_start, t_stop):
    d = h5["processing/ecephys/LFP/LFP/data"]
    conv = float(d.attrs["conversion"])
    i0, i1 = int(round(t_start * LFP_FS)), int(round(t_stop * LFP_FS))
    x = d[i0:i1, ch].astype(np.float64) * conv
    t = (np.arange(i0, i1)) / LFP_FS
    return nap.Tsd(t=t, d=x)


def pick_theta_channel(h5, t_start, dur=300.0, stride=4):
    """Channel with the largest theta / (delta+beta) power ratio."""
    from scipy.signal import welch
    d = h5["processing/ecephys/LFP/LFP/data"]
    conv = float(d.attrs["conversion"])
    i0 = int(round(t_start * LFP_FS))
    i1 = i0 + int(dur * LFP_FS)
    chans = np.arange(0, d.shape[1], stride)
    ratios = []
    for ch in chans:
        x = d[i0:i1, ch].astype(float) * conv
        fr, P = welch(x, fs=LFP_FS, nperseg=2048)
        th = P[(fr >= THETA_BAND[0]) & (fr <= THETA_BAND[1])].mean()
        ref = P[((fr >= 2) & (fr <= 4)) | ((fr >= 12) & (fr <= 20))].mean()
        ratios.append(th / ref)
    return int(chans[int(np.argmax(ratios))]), np.array(chans), np.array(ratios)


# ------------------------------------------------------------ preprocessing --

def theta_phase(lfp):
    """Zero-phase 6-10 Hz filter + Hilbert.  Returns (filtered, phase in rad).

    Phase is wrapped to [-pi, pi) with 0 at the *peak* of the filtered LFP,
    which is the numpy/Hilbert convention.
    """
    b, a = butter(3, np.array(THETA_BAND) / (LFP_FS / 2), btype="bandpass")
    x = filtfilt(b, a, lfp.values)
    z = hilbert(x)
    return (nap.Tsd(t=lfp.index.values, d=x),
            nap.Tsd(t=lfp.index.values, d=np.angle(z)))


def wrap(dx, length, circular):
    """Signed displacement, wrapped to [-L/2, L/2) on a circular track."""
    if not circular:
        return dx
    return (dx + length / 2) % length - length / 2


def velocity(pos, maze, smooth_bins=3.0):
    """Running velocity (m/s) from linearised position.

    On the circular maze the coordinate jumps by a full lap at the wrap point,
    so differences are taken modulo the circumference before smoothing.
    """
    t, x = pos.index.values, pos.values
    dx = wrap(np.diff(x, prepend=x[0]), maze["length"], maze["circular"])
    dt_ = np.diff(t, prepend=t[0] - 1e-6)
    v = gaussian_filter1d(dx / dt_, smooth_bins)
    return nap.Tsd(t=t, d=v)


def run_epochs(pos, dt, maze, min_dur=0.5):
    """Split the tracked position into per-direction running epochs.

    Contiguous stretches of valid linearised position (the animal's track
    traversals) are cut at gaps in tracking, speed-thresholded, and labelled by
    the sign of the (wrapped) velocity.
    """
    t = pos.index.values
    vel = velocity(pos, maze)
    v = vel.values
    gaps = np.where(np.diff(t) > 3 * dt)[0]
    seg_starts = np.r_[0, gaps + 1]
    seg_stops = np.r_[gaps, len(t) - 1]

    ep = {1: [[], []], -1: [[], []]}
    for s0, s1 in zip(seg_starts, seg_stops):
        if t[s1] - t[s0] < min_dur:
            continue
        for direction in (1, -1):
            fast = np.zeros(s1 - s0 + 1, bool)
            seg_v = v[s0:s1 + 1]
            fast = (np.abs(seg_v) > SPEED_THRESH) & (np.sign(seg_v) == direction)
            if fast.sum() < 3:
                continue
            idx = np.where(fast)[0]
            splits = np.where(np.diff(idx) > 1)[0]
            for a, b in zip(np.r_[0, splits + 1], np.r_[splits, len(idx) - 1]):
                i0, i1 = idx[a] + s0, idx[b] + s0
                if t[i1] - t[i0] >= min_dur:
                    ep[direction][0].append(t[i0])
                    ep[direction][1].append(t[i1])
    return {k: nap.IntervalSet(start=np.array(v_[0]), end=np.array(v_[1]))
            for k, v_ in ep.items()}, vel


# ----------------------------------------------------------- place fields ---

def rate_maps(units, pos, ep, maze, sigma=1.0):
    mode = "wrap" if maze["circular"] else "nearest"
    tc = nap.compute_1d_tuning_curves(
        units, pos, nb_bins=int(round(maze["length"] / POS_BIN)), ep=ep,
        minmax=(0.0, maze["length"]))
    sm = tc.apply(lambda c: gaussian_filter1d(c.values, sigma, mode=mode), axis=0)
    return sm


def spatial_info(tc, occ):
    """Skaggs spatial information, bits/spike, for each column of tc."""
    p = occ / occ.sum()
    out = {}
    for u in tc.columns:
        r = tc[u].values
        rbar = np.nansum(p * r)
        if rbar <= 0:
            out[u] = 0.0
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            term = p * (r / rbar) * np.log2(r / rbar)
        out[u] = float(np.nansum(term[np.isfinite(term)]))
    return out


def occupancy(pos, ep, dt, maze):
    p = pos.restrict(ep)
    n = int(round(maze["length"] / POS_BIN))
    edges = np.linspace(0, maze["length"], n + 1)
    occ, _ = np.histogram(p.values, bins=edges)
    return occ * dt, edges


def detect_field(rate, edges, maze, frac=0.25, min_bins=3):
    """Largest contiguous run of bins above frac * peak around the peak bin.

    On the circular maze the search wraps around the ends of the rate map.
    Returns (lo, width, peak_bin) with lo the field entry in track coordinates.
    """
    n = len(rate)
    pk = int(np.argmax(rate))
    thr = frac * rate[pk]
    circ = maze["circular"]
    lo, hi = pk, pk               # inclusive bin indices, hi may exceed n-1
    while hi - lo + 1 < n:
        nxt_lo = lo - 1 if (circ or lo > 0) else None
        nxt_hi = hi + 1 if (circ or hi < n - 1) else None
        grew = False
        if nxt_lo is not None and rate[nxt_lo % n] >= thr:
            lo, grew = nxt_lo, True
        if nxt_hi is not None and rate[nxt_hi % n] >= thr:
            hi, grew = nxt_hi, True
        if not grew:
            break
    nb = hi - lo + 1
    if nb < min_bins or nb >= n:
        return None
    return edges[lo % n], nb * POS_BIN, pk % n


def in_field(sp_pos, lo, width, maze):
    """Boolean mask and traversed fraction, measured from the field entry."""
    u = np.mod(sp_pos - lo, maze["length"])
    return u <= width, u / width


def field_fraction(sp_pos, lo, width, direction, maze):
    """Fraction of the field traversed, measured along the direction of travel.

    On leftward runs the linearised coordinate *decreases*, so the raw
    coordinate has to be flipped or the precession slope comes out inverted.
    """
    _, frac = in_field(sp_pos, lo, width, maze)
    return frac if direction > 0 else 1.0 - frac


def circlin_fit(x, phi, slope_range=(-3.0, 3.0), n_grid=601):
    """Kempter et al. (2012) circular-linear regression.

    x: linear variable (normalised position, dimensionless)
    phi: circular variable (rad)
    Returns dict with slope (cycles per unit x), phase offset (rad),
    circular-linear correlation rho and the resultant length R.
    """
    a = np.linspace(slope_range[0], slope_range[1], n_grid)
    R = np.abs(np.exp(1j * (phi[None, :] - 2 * np.pi * a[:, None] * x[None, :])).mean(axis=1))
    k = int(np.argmax(R))
    slope = a[k]
    phi0 = np.angle(np.exp(1j * (phi - 2 * np.pi * slope * x)).mean())
    # circular-linear correlation (Kempter eq. 3)
    theta = np.mod(2 * np.pi * np.abs(slope) * x, 2 * np.pi)
    tbar = np.angle(np.exp(1j * theta).mean())
    pbar = np.angle(np.exp(1j * phi).mean())
    num = np.sum(np.sin(phi - pbar) * np.sin(theta - tbar))
    den = np.sqrt(np.sum(np.sin(phi - pbar) ** 2) * np.sum(np.sin(theta - tbar) ** 2))
    rho = num / den if den > 0 else 0.0
    return dict(slope=slope, phi0=phi0, rho=float(rho), R=float(R[k]), n=len(x))


def circlin_pvalue(x, phi, n_shuf=500, rng=None, **kw):
    rng = rng or np.random.default_rng(0)
    obs = circlin_fit(x, phi, **kw)
    null = np.empty(n_shuf)
    for i in range(n_shuf):
        null[i] = abs(circlin_fit(x, rng.permutation(phi), **kw)["rho"])
    obs["p"] = float((np.sum(null >= abs(obs["rho"])) + 1) / (n_shuf + 1))
    return obs
