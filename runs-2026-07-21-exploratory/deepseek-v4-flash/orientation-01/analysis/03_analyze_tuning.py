"""Orientation tuning + selectivity for the Allen Visual Coding static gratings.

Per unit (good quality only):
- mean firing-rate response matrix R[orientation, spatial_frequency]
- tuning curve r(theta) at the unit's preferred spatial frequency
- gOSI (vector-based orientation selectivity), OSI (orthogonal pair), preferred
  orientation (doubled-angle circular mean), one-way ANOVA p across orientations
- permutation-test p-value on gOSI: trial labels are shuffled and the FULL
  statistic (including per-unit preferred-SF selection) is recomputed on every
  permutation; units with a NaN gOSI (no response) are excluded.

Run:    python analysis/03_analyze_tuning.py
Output: analysis/tuning_results_s715093703.npz
"""
import numpy as np
from scipy import stats
from tqdm import tqdm

RUN = 'analysis'
S = '715093703'
ORIS = np.array([0., 30., 60., 90., 120., 150.])
N_PERM = 250
SEED = 21


def load():
    d = np.load(f'{RUN}/static_s{S}.npz')
    m = np.load(f'{RUN}/meta_s{S}.npz')
    dr = np.load(f'{RUN}/drifting_s{S}.npz')
    return dict(
        counts=d['counts'].astype(np.float64),
        onset=d['onset'], offset=d['offset'],
        orient=d['orientation'], sf=d['spatial_freq'],
        region=m['region'], quality=m['quality'],
        drift_counts=dr['counts'].astype(np.float64),
        drift_onset=dr['onset'], drift_offset=dr['offset'],
        drift_orient=dr['orientation'],
    )


def cond_index(orient, sf):
    """Map (orientation, sf) pairs to a single integer condition index.

    Rows with NaN orientation or spatial frequency are blank sweeps (gray
    screen); they get condition code -1 and are excluded from the mapping.
    """
    ok = np.isfinite(orient) & np.isfinite(sf)
    oris = np.unique(orient[ok])
    sfs = np.unique(sf[ok])
    mapping = {(o, s): k for k, (o, s) in enumerate(
        (o, s) for o in oris for s in sfs)}
    cond = np.full(len(orient), -1, dtype=int)
    cond[ok] = [mapping[(o, s)] for o, s in zip(orient[ok], sf[ok])]
    return cond, len(mapping), oris, sfs


def full_stat(means, n_ori):
    """Vectorized orientation statistic from a (unit, cond) mean matrix.

    For each unit the preferred spatial frequency (max mean response across
    orientations) is selected, then a tuning curve is formed and three
    selectivity measures are computed:
      gOSI   = |sum r(θ) e^{2iθ}| / sum r(θ)        (vector-based, 0..1)
      OSI    = (r_pref - r_orth) / (r_pref + r_orth)  using nearest orthogonal test
      pref   = 0.5 * arg(sum r e^{2iθ}) mod 180
    """
    n_u = means.shape[0]
    n_s = means.shape[1] // n_ori
    M = means.reshape(n_u, n_ori, n_s)

    mean_over_ori = M.mean(axis=1)
    pref_sf = mean_over_ori.argmax(axis=1)
    uv = np.arange(n_u)
    r = np.clip(M[uv, :, pref_sf], 0, None)          # (n_u, n_ori)
    theta = np.deg2rad(ORIS[:n_ori])[None, :]

    tot = r.sum(axis=1)
    sx = (r * np.cos(2 * theta)).sum(axis=1)
    sy = (r * np.sin(2 * theta)).sum(axis=1)
    gosi = np.hypot(sx, sy) / np.maximum(tot, 1e-12)
    gosi = np.where(tot <= 0, np.nan, gosi)

    pref_ori = np.mod(np.rad2deg(np.arctan2(sy, sx)) / 2.0, 180.0)
    pref_ori = np.where(tot <= 0, np.nan, pref_ori)

    # paired OSI: response at nearest tested orientation to pref+90 deg
    wanted = np.mod(pref_ori + 90, 180)
    op_idx = np.argmin(np.abs(ORIS[:, None] - wanted[None, :]), axis=0)
    r_pref = r.max(axis=1)
    r_orth = r[uv, op_idx]
    osi = np.where(r_pref + r_orth > 0,
                   (r_pref - r_orth) / (r_pref + r_orth), np.nan)
    return gosi, pref_ori, osi, r, pref_sf, tot


def anova_p_trial(counts, orient):
    """One-way ANOVA of per-trial spike counts across the 6 orientations."""
    n_u = counts.shape[0]
    F = np.full(n_u, np.nan)
    P = np.full(n_u, np.nan)
    ok = np.isfinite(orient)
    for u in tqdm(range(n_u), desc='ANOVA'):
        groups = [counts[u, (orient == o) & ok] for o in ORIS]
        groups = [g for g in groups if g.size > 1]
        if len(groups) >= 2:
            F[u], P[u] = stats.f_oneway(*groups)
    return F, P


def permutation_gosi(counts, orient, sf, dur, good, n_perm=N_PERM, seed=SEED):
    """Vectorized permutation test of the FULL gOSI statistic.

    The trial-condition schedule is shared by all units, so one permutation of
    the labels applies to all samples simultaneously. Each permutation
    recomputes the full statistic (including per-unit preferred-SF selection).
    Only good units with an observed finite gOSI are tested.
    """
    rng = np.random.default_rng(seed)
    cond, n_cond, _, _ = cond_index(orient, sf)
    valid = cond >= 0                 # drop blank (NaN) sweeps
    counts = counts[:, valid]
    dur = dur[valid]
    cond = cond[valid]
    rates = counts / dur[None, :]     # Hz per trial

    I0 = np.zeros((n_t := rates.shape[1], n_cond))
    I0[np.arange(n_t), cond] = 1.0
    n_per_cond = I0.sum(axis=0)
    obs_g = full_stat((rates @ I0) / n_per_cond[None, :], len(ORIS))[0]

    usable = good & np.isfinite(obs_g)
    Gidx = np.where(usable)[0]
    sub = rates[Gidx]                  # (G, n_t)
    obs_sub = obs_g[Gidx]
    G = len(Gidx)

    n_ori = len(ORIS)
    null = np.zeros((n_perm, G))
    Itmp = np.zeros((n_t, n_cond))
    for k_ in range(n_perm):
        pcond = rng.permutation(cond)
        Itmp.fill(0.0)
        Itmp[np.arange(n_t), pcond] = 1.0
        m = sub @ Itmp / n_per_cond[None, :]
        g_ = full_stat(m, n_ori)[0]
        null[k_, :] = np.where(np.isfinite(g_), g_.clip(0, None), 0.0)

    p = (null >= obs_sub[None, :] - 1e-12).mean(axis=0)   # (G,)
    pvals = np.full(counts.shape[0], np.nan)
    pvals[Gidx] = p
    return pvals


def main():
    D = load()
    counts, orient = D['counts'], D['orient']
    sf, region, quality = D['sf'], D['region'], D['quality']
    dur = D['offset'] - D['onset']
    good = quality == 'good'

    # Baseline (blank) rate from gray-screen sweeps: drifting rows with NaN
    # orientation plus static rows with NaN orientation.
    blank_acc = []
    dr_blank = np.isnan(D['drift_orient'])
    if dr_blank.sum():
        blank_acc.append((D['drift_counts'][:, dr_blank],
                          (D['drift_offset'] - D['drift_onset'])[dr_blank]))
    st_blank = np.isnan(orient)
    if st_blank.sum():
        blank_acc.append((counts[:, st_blank], dur[st_blank]))
    blank_rate = np.zeros(counts.shape[0])
    if blank_acc:
        tot_dur = sum(bd.sum() for _, bd in blank_acc)
        blank_rate = sum(bl.sum(axis=1) for bl, _ in blank_acc) / tot_dur

    # Condition matrix over finite (orientation, spatial-frequency) rows only.
    cond, n_cond, oris, sfs = cond_index(orient, sf)
    vrow = cond >= 0
    rates = counts[:, vrow] / dur[vrow][None, :]
    condv = cond[vrow]
    I0 = np.zeros((rates.shape[1], n_cond))
    I0[np.arange(rates.shape[1]), condv] = 1.0
    n_per = I0.sum(axis=0)
    means = (rates @ I0) / n_per[None, :]

    responsive = (means.max(axis=1) > blank_rate + 1.0)

    gosi, pref_ori, osi_pair, rcurve, pref_sf, tot = full_stat(means, len(ORIS))
    F, P = anova_p_trial(counts, orient)
    p_perm = permutation_gosi(counts, orient, sf, dur, good)

    res = dict(
        unit=np.arange(len(counts)),
        region=region, quality=quality, good=good,
        gosi=gosi, osi=osi_pair, pref_ori=pref_ori, pref_sf=pref_sf,
        total_rate=tot, blank_rate=blank_rate,
        anova_F=F, anova_p=P, gosi_perm_p=p_perm,
        responsive=responsive,
    )
    out = f'{RUN}/tuning_results_s{S}.npz'
    np.savez_compressed(out, **res)
    print('saved', out)
    print('summary (good units only):')

    def summ(reg):
        sel = good & (region == reg) & np.isfinite(gosi)
        frac = np.nanmean(p_perm[sel] < 0.05)
        return (f'{reg:5s} n={sel.sum():4d}  mean gOSI={np.nanmean(gosi[sel]):.3f}  '
                f'selective(p<0.05)={frac:.3f}')

    print(summ('VISp'))
    print(summ('VISl'))
    print(summ('VISam'))
    print(summ('VISpm'))
    print(summ('LGd'))


if __name__ == '__main__':
    main()