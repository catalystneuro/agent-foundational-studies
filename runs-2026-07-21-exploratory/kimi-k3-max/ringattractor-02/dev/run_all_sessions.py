"""Compute per-session ring-attractor quantities for the 5-session panel.

Caches one npz per session in dev/cache/:
  - tc_hz, centers: wake tuning curves (Hz) and angle bin centers
  - mvl, pval, is_hd: HD-cell statistics
  - pref: preferred angle per unit (NaN for non-HD)
  - corr_wake/rem/nrem: pairwise Pearson corr matrices of HD cells (100 ms bins)
  - rates per state for HD cells
"""
import os
import numpy as np
import pynapple as nap
from tqdm import tqdm
from prototype_load import load_session, ASSETS

CACHE = "cache"
os.makedirs(CACHE, exist_ok=True)
rng = np.random.default_rng(42)

BIN_S = 0.1  # 100 ms bins for correlations


def resultant_length(angles):
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan
    return np.abs(np.exp(1j * angles).mean())


def binned_counts(group, epochs, bin_s):
    """Binned spike counts within epochs -> (n_bins, n_units) float array."""
    counts = group.count(bin_s, epochs)  # TsdFrame, bins restricted to epochs
    return np.asarray(counts.values, dtype=float)


def corr_matrix(counts):
    """Pearson correlation across units; NaN-safe (zero-variance columns -> NaN)."""
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(counts.T)
    return c


for session in ASSETS:
    out = os.path.join(CACHE, f"{session}.npz")
    if os.path.exists(out):
        print("cached:", session)
        continue
    print("=" * 60, "\n", session)
    nwb = load_session(session)
    states = nwb["states"]
    wake = states[states["label"] == "Awake"]
    rem = states[states["label"] == "REM"]
    nrem = states[states["label"] == "Non-REM"]

    # HD from LEDs
    red = np.asarray(nwb["SubjectPosition/RedLED"].values)
    blue = np.asarray(nwb["SubjectPosition/BlueLED"].values)
    t = nwb["SubjectPosition/RedLED"].t
    valid = (red[:, 0] > 0) & (red[:, 1] > 0) & (blue[:, 0] > 0) & (blue[:, 1] > 0)
    hd = np.arctan2(red[:, 1] - blue[:, 1], red[:, 0] - blue[:, 0]) % (2 * np.pi)
    hd[~valid] = np.nan
    hd_tsd = nap.Tsd(t=t, d=hd)

    # units with sorted spike times
    units = nwb["units"]
    group = nap.TsGroup({k: nap.Ts(np.sort(units[k].t)) for k in units.keys()})
    keys = list(group.keys())

    # wake tuning curves
    nbins = 60
    tc_da = nap.compute_tuning_curves(group, hd_tsd, bins=nbins, range=(0, 2 * np.pi),
                                      epochs=wake, return_counts=True)
    occupancy = tc_da.attrs["occupancy"] / tc_da.attrs["fs"]
    counts = np.asarray(tc_da)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc_hz = counts / occupancy[None, :]
    tc_hz = np.where(occupancy[None, :] > 0, tc_hz, np.nan)
    bin_dim = [d for d in tc_da.dims if d != "unit"][0]
    centers = np.asarray(tc_da.coords[bin_dim])

    # HD-cell selection
    wake_mask = wake.in_interval(hd_tsd) >= 0
    valid_wake = valid & wake_mask & ~np.isnan(hd)
    hd_valid_a = hd[valid_wake]
    mvl = np.full(len(group), np.nan)
    pval = np.full(len(group), np.nan)
    nshuf = 500
    for i, k in enumerate(tqdm(keys, desc=f"HD stats {session}")):
        ang = group[k].restrict(wake).value_from(hd_tsd)
        ang = ang[~np.isnan(ang)]
        n = len(ang)
        if n < 100:
            continue
        mvl[i] = resultant_length(ang)
        n_cap = min(n, 100_000)
        null = np.empty(nshuf)
        for s in range(nshuf):
            idx = rng.integers(0, len(hd_valid_a), size=n_cap)
            null[s] = np.abs(np.exp(1j * hd_valid_a[idx]).mean())
        pval[i] = (np.sum(null >= mvl[i]) + 1) / (nshuf + 1)
    is_hd = (mvl > 0.3) & (pval < 0.05)
    hd_idx = np.where(is_hd)[0]
    print("HD cells:", len(hd_idx), "of", len(group))

    pref = np.full(len(group), np.nan)
    pref[hd_idx] = centers[np.nanargmax(tc_hz[hd_idx], axis=1)]

    # per-state binned counts for HD cells only
    hd_group = nap.TsGroup({keys[i]: group[keys[i]] for i in hd_idx})
    cw = binned_counts(hd_group, wake, BIN_S)
    cr = binned_counts(hd_group, rem, BIN_S)
    cn = binned_counts(hd_group, nrem, BIN_S)
    print("bins wake/rem/nrem:", cw.shape[0], cr.shape[0], cn.shape[0])

    np.savez(out, tc_hz=tc_hz, centers=centers, mvl=mvl, pval=pval, is_hd=is_hd,
             pref=pref, hd_idx=hd_idx,
             corr_wake=corr_matrix(cw), corr_rem=corr_matrix(cr), corr_nrem=corr_matrix(cn),
             n_bins=np.array([cw.shape[0], cr.shape[0], cn.shape[0]]),
             mean_rates=np.array([cw.mean(0), cr.mean(0), cn.mean(0)]) / BIN_S)
    print("saved", out)
