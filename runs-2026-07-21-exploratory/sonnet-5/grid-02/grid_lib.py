"""Shared functions for grid cell analysis: spatial autocorrelation and gridness score."""
import numpy as np
from scipy.ndimage import rotate as ndi_rotate


def spatial_autocorrelogram(rate_map, min_overlap=20):
    """Pearson-correlation-based 2D spatial autocorrelogram (Sargolini/Moser-lab style).

    rate_map: 2D array, np.nan where unvisited.
    Returns autocorr of shape (2*ny-1, 2*nx-1), np.nan where overlap < min_overlap.
    """
    ny, nx = rate_map.shape
    valid = ~np.isnan(rate_map)
    r = np.where(valid, rate_map, 0.0)

    out_h, out_w = 2 * ny - 1, 2 * nx - 1
    autocorr = np.full((out_h, out_w), np.nan)

    for dy in range(-(ny - 1), ny):
        for dx in range(-(nx - 1), nx):
            # overlap region between rate_map and itself shifted by (dy, dx)
            y0a, y1a = max(0, dy), min(ny, ny + dy)
            x0a, x1a = max(0, dx), min(nx, nx + dx)
            y0b, y1b = max(0, -dy), min(ny, ny - dy)
            x0b, x1b = max(0, -dx), min(nx, nx - dx)

            va = valid[y0a:y1a, x0a:x1a]
            vb = valid[y0b:y1b, x0b:x1b]
            mask = va & vb
            n = mask.sum()
            if n < min_overlap:
                continue

            a = r[y0a:y1a, x0a:x1a][mask]
            b = r[y0b:y1b, x0b:x1b][mask]

            if a.std() == 0 or b.std() == 0:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            autocorr[dy + ny - 1, dx + nx - 1] = corr

    return autocorr


def radial_profile(autocorr):
    """Mean autocorrelation value as a function of integer-binned radius from center."""
    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.astype(int)
    max_r = min(cy, cx)

    valid = ~np.isnan(autocorr)
    profile = np.full(max_r + 1, np.nan)
    for rad in range(max_r + 1):
        mask = (r_int == rad) & valid
        if mask.sum() > 0:
            profile[rad] = autocorr[mask].mean()
    return profile


def find_annulus_radii(profile):
    """Find inner/outer radius of the ring containing the 6 grid peaks.

    inner radius = first local minimum after the central peak (r=0).
    outer radius = next local maximum's following minimum, else fall back to 1.4x the
    radius of the first peak after the inner minimum.
    """
    n = len(profile)
    valid_idx = np.where(~np.isnan(profile))[0]
    if len(valid_idx) < 5:
        return 1, n - 1

    # find first local minimum for r >= 1
    r_min1 = None
    for i in valid_idx:
        if i < 1:
            continue
        if i - 1 >= 0 and i + 1 < n and not np.isnan(profile[i - 1]) and not np.isnan(profile[i + 1]):
            if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1]:
                r_min1 = i
                break
    if r_min1 is None:
        r_min1 = max(1, valid_idx[0])

    # first local maximum after r_min1 (ring of 6 peaks)
    r_peak = None
    for i in range(r_min1 + 1, n - 1):
        if np.isnan(profile[i]) or np.isnan(profile[i - 1]) or np.isnan(profile[i + 1]):
            continue
        if profile[i] > profile[i - 1] and profile[i] >= profile[i + 1]:
            r_peak = i
            break
    if r_peak is None:
        r_peak = min(n - 1, r_min1 + max(2, n // 4))

    # next local minimum after r_peak
    r_min2 = None
    for i in range(r_peak + 1, n - 1):
        if np.isnan(profile[i]) or np.isnan(profile[i - 1]) or np.isnan(profile[i + 1]):
            continue
        if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1]:
            r_min2 = i
            break
    if r_min2 is None:
        r_min2 = min(n - 1, int(r_peak * 1.4))

    return r_min1, max(r_min2, r_min1 + 1)


def gridness_score(autocorr, return_details=False):
    """Standard gridness score: min(r60,r120) - max(r30,r90,r150) over an annulus mask."""
    profile = radial_profile(autocorr)
    r_in, r_out = find_annulus_radii(profile)

    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annulus = (r >= r_in) & (r <= r_out)

    base = np.where(annulus & ~np.isnan(autocorr), autocorr, np.nan)
    base_filled = np.nan_to_num(base, nan=0.0)
    base_mask = ~np.isnan(base)

    if base_mask.sum() < 20:
        return (np.nan, {}) if return_details else np.nan

    corrs = {}
    for angle in (30, 60, 90, 120, 150):
        rotated = ndi_rotate(base_filled, angle, reshape=False, order=1, mode='constant', cval=0.0)
        rot_mask = ndi_rotate(base_mask.astype(float), angle, reshape=False, order=1, mode='constant', cval=0.0) > 0.5
        joint_mask = base_mask & rot_mask
        if joint_mask.sum() < 20:
            corrs[angle] = np.nan
            continue
        a = base_filled[joint_mask]
        b = rotated[joint_mask]
        if a.std() == 0 or b.std() == 0:
            corrs[angle] = np.nan
            continue
        corrs[angle] = np.corrcoef(a, b)[0, 1]

    rot_vals = [corrs[a] for a in (30, 60, 90, 120, 150)]
    if any(np.isnan(v) for v in rot_vals):
        score = np.nan
    else:
        score = min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])

    if return_details:
        return score, {'r_in': r_in, 'r_out': r_out, 'corrs': corrs, 'profile': profile}
    return score
