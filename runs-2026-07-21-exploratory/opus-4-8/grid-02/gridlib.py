"""Core functions for grid-cell analysis of DANDI:000582 (Sargolini/Moser MEC data)."""
import numpy as np
import remfile, h5py
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate as ndrotate

DANDI = '000582'
CACHE = '/tmp/remfile_cache'

def load_session(asset_id):
    url = f'https://api.dandiarchive.org/api/dandisets/{DANDI}/versions/draft/assets/{asset_id}/download/'
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    nwbfile = NWBHDF5IO(file=h5py.File(rf, 'r')).read()
    nwb = nap.NWBFile(nwbfile)
    return nwbfile, nwb

def rate_map(unit_ts, pos, nbins=40, extent=50.0, sigma=1.5, min_occ=0.1):
    """2D smoothed firing-rate map (Hz). pos: TsdFrame x,y in cm."""
    x = pos[:, 0].values; y = pos[:, 1].values; t = pos.index.values
    dt = np.median(np.diff(t))
    rng = [[-extent, extent], [-extent, extent]]
    occ, xe, ye = np.histogram2d(x, y, bins=nbins, range=rng)
    occ_time = occ * dt
    st = unit_ts.index.values
    sx = np.interp(st, t, x); sy = np.interp(st, t, y)
    spk, _, _ = np.histogram2d(sx, sy, bins=nbins, range=rng)
    mask = occ_time >= min_occ
    # smooth counts and occupancy separately, then divide (handles unvisited bins)
    sc_s = gaussian_filter(spk * mask, sigma)
    oc_s = gaussian_filter(occ_time * mask, sigma)
    with np.errstate(invalid='ignore', divide='ignore'):
        rm = sc_s / oc_s
    rm[gaussian_filter(mask.astype(float), sigma) < 0.05] = np.nan
    return rm.T, xe, ye  # transpose so rows=y, cols=x for imshow origin lower

def spatial_autocorr(rm):
    """Unbiased Pearson spatial autocorrelogram (Sargolini 2006). Returns (2N-1)x(2N-1)."""
    f = rm.copy()
    valid = ~np.isnan(f)
    f0 = np.where(valid, f, 0.0)
    n = f.shape[0]
    from scipy.signal import fftconvolve
    rev = lambda a: a[::-1, ::-1]
    sum_fg = fftconvolve(f0, rev(f0), mode='full')
    sum_f = fftconvolve(f0, rev(valid.astype(float)), mode='full')
    sum_g = fftconvolve(valid.astype(float), rev(f0), mode='full')
    sum_ff = fftconvolve(f0**2, rev(valid.astype(float)), mode='full')
    sum_gg = fftconvolve(valid.astype(float), rev(f0**2), mode='full')
    nn = fftconvolve(valid.astype(float), rev(valid.astype(float)), mode='full')
    with np.errstate(invalid='ignore', divide='ignore'):
        num = nn * sum_fg - sum_f * sum_g
        den = np.sqrt((nn * sum_ff - sum_f**2) * (nn * sum_gg - sum_g**2))
        r = num / den
    r[nn < 20] = np.nan
    return r

GRID_ANGLES = [30, 60, 90, 120, 150]

def grid_score(ac):
    """Standard gridness score (Sargolini 2006). Rotations precomputed once per angle,
    then annulus radii are swept cheaply. Returns (best_score, (inner, outer))."""
    center = ac.shape[0] // 2
    maxr = center
    yy, xx = np.mgrid[0:ac.shape[0], 0:ac.shape[1]]
    rr = np.sqrt((xx - center)**2 + (yy - center)**2)
    base = ac
    rots = {a: ndrotate(np.nan_to_num(ac), a, reshape=False, order=1, cval=np.nan)
            for a in GRID_ANGLES}
    best = -np.inf; best_params = None
    for outer in np.arange(0.30*maxr, 0.95*maxr, 2):
        for inner in np.arange(0.10*maxr, outer-3, 2):
            mask = (rr >= inner) & (rr <= outer)
            a0 = base[mask]
            cc = {}
            ok = True
            for ang in GRID_ANGLES:
                b0 = rots[ang][mask]
                good = ~np.isnan(a0) & ~np.isnan(b0)
                if good.sum() < 20 or a0[good].std() == 0 or b0[good].std() == 0:
                    ok = False; break
                cc[ang] = np.corrcoef(a0[good], b0[good])[0, 1]
            if not ok:
                continue
            gs = min(cc[60], cc[120]) - max(cc[30], cc[90], cc[150])
            if gs > best:
                best = gs; best_params = (inner, outer)
    return best, best_params

def field_com_stats(ac, nbins=40, extent=50.0):
    """Estimate grid spacing/orientation from 6 nearest autocorr peaks around center."""
    from scipy.ndimage import maximum_filter, label
    center = ac.shape[0] // 2
    bin_cm = 2 * extent / nbins
    a = np.nan_to_num(ac)
    mx = maximum_filter(a, size=3)
    peaks = (a == mx) & (a > 0.1)
    ys, xs = np.where(peaks)
    d = np.sqrt((xs - center)**2 + (ys - center)**2)
    order = np.argsort(d)
    # skip central peak (d~0), take next 6
    sel = [i for i in order if d[i] > 3][:6]
    if len(sel) == 0:
        return np.nan, np.nan
    spacing = np.median(d[sel]) * bin_cm
    ang = np.degrees(np.arctan2(ys[sel]-center, xs[sel]-center)) % 60
    orient = np.median(ang)
    return spacing, orient

def shuffle_grid_scores(unit_ts, pos, n_shuffles=100, min_shift=20.0, seed=0,
                        nbins=40, extent=50.0):
    """Null distribution of grid scores via circular shift of spike times by >= min_shift s."""
    rng = np.random.default_rng(seed)
    t = pos.index.values; T0, T1 = t[0], t[-1]; dur = T1 - T0
    st = unit_ts.index.values
    null = np.empty(n_shuffles)
    for k in range(n_shuffles):
        shift = rng.uniform(min_shift, dur - min_shift)
        s_shift = T0 + np.mod(st - T0 + shift, dur)
        sh = nap.Ts(np.sort(s_shift))
        rm, _, _ = rate_map(sh, pos, nbins=nbins, extent=extent)
        ac = spatial_autocorr(rm)
        gs, _ = grid_score(ac)
        null[k] = gs
    return null
