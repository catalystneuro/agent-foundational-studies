"""Utility functions for grid-cell rate map, spatial autocorrelogram, and gridness score."""
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from skimage.feature import peak_local_max


def smooth_rate_map(rate_map, sigma=1.0):
    """Smooth a rate map (with NaNs for unvisited bins) using a Gaussian kernel
    that ignores NaNs (via a weighted normalization)."""
    valid = ~np.isnan(rate_map)
    filled = np.where(valid, rate_map, 0.0)
    smoothed_num = gaussian_filter(filled, sigma=sigma)
    smoothed_den = gaussian_filter(valid.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed_num / smoothed_den
    out[smoothed_den < 1e-6] = np.nan
    return out


def spatial_autocorrelogram(rate_map):
    """Compute the Pearson spatial autocorrelogram of a rate map with NaNs,
    following the standard grid-cell method (Sargolini et al. 2006):
    for every possible (dx, dy) offset, correlate the overlapping valid pixels.
    Returns a (2H-1, 2W-1) array.
    """
    H, W = rate_map.shape
    valid = ~np.isnan(rate_map)
    data = np.where(valid, rate_map, 0.0)

    out_h, out_w = 2 * H - 1, 2 * W - 1
    corr = np.full((out_h, out_w), np.nan)

    min_bins_overlap = 20  # require a minimum overlap to compute a correlation

    for dy in range(-(H - 1), H):
        for dx in range(-(W - 1), W):
            # Overlap region between rate_map and itself shifted by (dy, dx)
            y0a, y1a = max(0, dy), min(H, H + dy)
            x0a, x1a = max(0, dx), min(W, W + dx)
            y0b, y1b = max(0, -dy), min(H, H - dy)
            x0b, x1b = max(0, -dx), min(W, W - dx)

            a = data[y0a:y1a, x0a:x1a]
            b = data[y0b:y1b, x0b:x1b]
            va = valid[y0a:y1a, x0a:x1a]
            vb = valid[y0b:y1b, x0b:x1b]
            m = va & vb

            n = m.sum()
            if n < min_bins_overlap:
                continue

            av = a[m]
            bv = b[m]
            sa = av.sum()
            sb = bv.sum()
            saa = (av ** 2).sum()
            sbb = (bv ** 2).sum()
            sab = (av * bv).sum()

            num = n * sab - sa * sb
            den = np.sqrt((n * saa - sa ** 2) * (n * sbb - sb ** 2))
            if den <= 0:
                continue
            corr[H - 1 + dy, W - 1 + dx] = num / den

    return corr


def gridness_score(autocorr, pixel_size=1.0):
    """Compute the gridness score of a spatial autocorrelogram following the
    Sargolini/Langston rotational-correlation method.

    Returns (gridness, r_inner, r_outer, peak_coords) where peak_coords are the
    (row, col) locations of the 6 nearest peaks around the center used to set
    the annulus radius.
    """
    H, W = autocorr.shape
    cy, cx = H // 2, W // 2

    filled = np.nan_to_num(autocorr, nan=-1.0)

    # Find local maxima excluding the very center peak
    coords = peak_local_max(filled, min_distance=2, threshold_abs=0.0)
    dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)

    # Exclude the central peak (distance ~ 0)
    mask_not_center = dists > 3
    coords = coords[mask_not_center]
    dists = dists[mask_not_center]

    if len(dists) < 6:
        return np.nan, np.nan, np.nan, coords

    order = np.argsort(dists)
    nearest6 = coords[order[:6]]
    nearest6_dists = dists[order[:6]]

    r_outer = np.mean(nearest6_dists) * 1.25
    r_inner = np.min(nearest6_dists) * 0.5

    yy, xx = np.indices(autocorr.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annulus_mask = (rr >= r_inner) & (rr <= r_outer)

    ref = filled.copy()
    ref[~annulus_mask] = np.nan

    def masked_corr(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 20:
            return np.nan
        av, bv = a[m], b[m]
        return np.corrcoef(av, bv)[0, 1]

    rotation_corrs = {}
    for angle in [30, 60, 90, 120, 150]:
        rotated = rotate(filled, angle, reshape=False, order=1, cval=-1.0)
        rotated_masked = np.where(annulus_mask, rotated, np.nan)
        rotation_corrs[angle] = masked_corr(ref, rotated_masked)

    gridness = min(rotation_corrs[60], rotation_corrs[120]) - max(
        rotation_corrs[30], rotation_corrs[90], rotation_corrs[150]
    )
    return gridness, r_inner, r_outer, nearest6
