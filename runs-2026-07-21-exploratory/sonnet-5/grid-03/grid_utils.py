"""Helper functions for grid-cell analysis: rate maps, spatial autocorrelograms,
and gridness scores (Sargolini et al. 2006 / Krupic et al. style)."""

import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.stats import pearsonr


def smooth_ratemap(ratemap, sigma=1.0):
    """Gaussian-smooth a 2D rate map that may contain NaNs (unvisited bins)."""
    valid = ~np.isnan(ratemap)
    filled = np.where(valid, ratemap, 0.0)
    smoothed_filled = gaussian_filter(filled, sigma=sigma)
    smoothed_valid = gaussian_filter(valid.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed_filled / smoothed_valid
    out[smoothed_valid < 1e-6] = np.nan
    return out


def autocorrelogram(ratemap):
    """Spatial autocorrelogram of a 2D rate map, computed as Pearson correlation
    at every integer pixel lag (NaN bins excluded pairwise), following the
    standard grid-cell analysis approach (Sargolini et al. 2006)."""
    ny, nx = ratemap.shape
    valid = ~np.isnan(ratemap)
    map0 = np.where(valid, ratemap, 0.0)

    out_shape = (2 * ny - 1, 2 * nx - 1)
    corr = np.full(out_shape, np.nan)

    for dy in range(-(ny - 1), ny):
        for dx in range(-(nx - 1), nx):
            y0a, y1a = max(0, dy), min(ny, ny + dy)
            x0a, x1a = max(0, dx), min(nx, nx + dx)
            y0b, y1b = max(0, -dy), min(ny, ny - dy)
            x0b, x1b = max(0, -dx), min(nx, nx - dx)

            a = map0[y0a:y1a, x0a:x1a]
            b = map0[y0b:y1b, x0b:x1b]
            va = valid[y0a:y1a, x0a:x1a]
            vb = valid[y0b:y1b, x0b:x1b]
            mask = va & vb
            n = mask.sum()
            if n < 20:
                continue
            av = a[mask]
            bv = b[mask]
            if av.std() < 1e-9 or bv.std() < 1e-9:
                continue
            r = np.corrcoef(av, bv)[0, 1]
            corr[dy + ny - 1, dx + nx - 1] = r

    return corr


def _radial_profile(corr):
    """Mean autocorrelation value as a function of integer-pixel radius from center."""
    ny, nx = corr.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.astype(int)
    valid = ~np.isnan(corr)
    max_r = r_int.max()
    profile = np.full(max_r + 1, np.nan)
    for i in range(max_r + 1):
        mask = (r_int == i) & valid
        if mask.sum() > 0:
            profile[i] = corr[mask].mean()
    return profile


def find_gridness_annulus(corr, min_radius_px=2):
    """Estimate inner/outer radii (in pixels) of the annulus containing the
    six closest peaks surrounding the central peak, from the radial profile
    of the autocorrelogram (local peaks after the central peak decays)."""
    profile = _radial_profile(corr)
    # walk outward from the center until the profile stops decreasing (end of
    # central peak), that marks the inner radius
    r = min_radius_px
    while r < len(profile) - 1 and (np.isnan(profile[r + 1]) or profile[r + 1] < profile[r]):
        r += 1
        if r >= len(profile) - 1:
            break
    inner = max(r, min_radius_px)

    # find the next local maximum after the inner radius -> ring of six peaks
    r2 = inner
    best_r, best_val = inner, -np.inf
    search_end = min(len(profile), inner + 30)
    for i in range(inner, search_end):
        if not np.isnan(profile[i]) and profile[i] > best_val:
            best_val = profile[i]
            best_r = i
    outer = best_r + max(2, (best_r - inner) // 2)
    outer = min(outer, len(profile) - 1)
    return inner, outer


def gridness_score(corr, inner_radius=None, outer_radius=None):
    """Compute the gridness score of a spatial autocorrelogram by rotating an
    annular region and comparing to the original via Pearson correlation.

    gridness = min(r60, r120) - max(r30, r90, r150)
    """
    ny, nx = corr.shape
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if inner_radius is None or outer_radius is None:
        inner_radius, outer_radius = find_gridness_annulus(corr)

    annulus_mask = (r >= inner_radius) & (r <= outer_radius)
    ref = np.where(annulus_mask & ~np.isnan(corr), corr, np.nan)

    rotation_corrs = {}
    for angle in [30, 60, 90, 120, 150]:
        rotated = rotate(np.nan_to_num(corr, nan=0.0), angle, reshape=False, order=1)
        rotated_valid_mask = rotate(
            (~np.isnan(corr)).astype(float), angle, reshape=False, order=1
        ) > 0.9
        mask = annulus_mask & ~np.isnan(corr) & rotated_valid_mask
        if mask.sum() < 20:
            rotation_corrs[angle] = np.nan
            continue
        a = np.asarray(corr[mask], dtype=np.float64)
        b = np.asarray(rotated[mask], dtype=np.float64)
        if a.std() < 1e-9 or b.std() < 1e-9:
            rotation_corrs[angle] = np.nan
            continue
        rotation_corrs[angle] = pearsonr(a, b)[0]

    r60 = np.nanmin([rotation_corrs[60], rotation_corrs[120]])
    r30 = np.nanmax([rotation_corrs[30], rotation_corrs[90], rotation_corrs[150]])
    return r60 - r30, rotation_corrs, (inner_radius, outer_radius)


def shuffled_gridness_null(spike_times, position, ep, bins, rng, n_shuffles=100,
                            min_shift=20.0):
    """Build a null distribution of gridness scores by circularly shifting the
    spike train relative to position by a random amount (>= min_shift seconds,
    wrapped within the epoch), recomputing the rate map + gridness each time."""
    import pynapple as nap

    t0 = ep.start[0]
    dur = ep.end[0] - ep.start[0]
    null_scores = np.full(n_shuffles, np.nan)

    for i in range(n_shuffles):
        shift = rng.uniform(min_shift, dur - min_shift)
        shifted = ((spike_times.index.values - t0 + shift) % dur) + t0
        shifted_ts = nap.Ts(t=np.sort(shifted))
        tc = nap.compute_tuning_curves(shifted_ts, position, bins=bins, epochs=ep)
        ratemap = tc.values[0] if tc.values.ndim == 3 else tc.values
        smooth = smooth_ratemap(ratemap, sigma=1.0)
        corr = autocorrelogram(smooth)
        if np.all(np.isnan(corr)):
            continue
        score, _, _ = gridness_score(corr)
        null_scores[i] = score

    return null_scores
