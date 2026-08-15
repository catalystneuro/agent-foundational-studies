"""Bayesian decoding of position during ripples and replay scoring.

Place-cell rate templates come from track running; within each POST-sleep ripple
we memoryless-Bayesian decode position in fine time bins and test whether the
decoded trajectory forms a spatially coherent line (replay) versus shuffles.
"""
import numpy as np


def build_template(tsg, pos, run_ep, n_pos=50, occ_min=0.15, peak_min=2.0, smooth=1.5):
    """Rate map (n_pos, n_cells) for excitatory cells over running, plus the
    positions kept (occupied bins) and the selected unit ids."""
    import pynapple as nap
    from scipy.ndimage import gaussian_filter1d
    grp = tsg[tsg.cell_type == "excitatory"]
    lo, hi = np.nanmin(pos.values), np.nanmax(pos.values)
    tc = nap.compute_1d_tuning_curves(grp, pos, nb_bins=n_pos, ep=run_ep, minmax=(lo, hi))
    bins = np.linspace(lo, hi, n_pos + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    # occupancy to drop unvisited bins
    posr = pos.restrict(run_ep).values
    posr = posr[~np.isnan(posr)]
    occ = np.histogram(posr, bins)[0].astype(float)
    keep_pos = occ > (occ.max() * occ_min)
    rate = tc.values  # (n_pos, n_cells)
    rate = np.apply_along_axis(lambda c: gaussian_filter1d(c, smooth), 0, rate)
    rate = np.nan_to_num(rate)
    # select place cells
    unit_ids = np.array(tc.columns)
    is_place = np.nanmax(rate, 0) > peak_min
    rate = rate[:, is_place][keep_pos, :]
    return rate, centers[keep_pos], unit_ids[is_place]


def bayes_decode(counts, rate, dt):
    """counts (n_bins, n_cells) spike counts; rate (n_pos, n_cells) Hz.
    Returns posterior (n_bins, n_pos)."""
    eps = 1e-9
    logr = np.log(rate + eps)                      # (n_pos, n_cells)
    loglik = counts @ logr.T                        # (n_bins, n_pos)
    loglik -= dt * rate.sum(1)[None, :]
    loglik -= loglik.max(1, keepdims=True)
    post = np.exp(loglik)
    post /= post.sum(1, keepdims=True) + eps
    return post


def bin_spikes(spike_list, unit_ids, t0, t1, dt):
    edges = np.arange(t0, t1 + dt, dt)
    counts = np.zeros((len(edges) - 1, len(unit_ids)))
    for j, uid in enumerate(unit_ids):
        counts[:, j] = np.histogram(spike_list[uid], edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2
    return counts, centers


def weighted_corr(post, centers_pos, t):
    """Posterior-weighted correlation between time and decoded position."""
    P = post / (post.sum() + 1e-12)
    X = np.repeat(t[:, None], len(centers_pos), axis=1)
    Y = np.repeat(centers_pos[None, :], len(t), axis=0)
    mx = (P * X).sum(); my = (P * Y).sum()
    cov = (P * (X - mx) * (Y - my)).sum()
    vx = (P * (X - mx) ** 2).sum(); vy = (P * (Y - my) ** 2).sum()
    return cov / np.sqrt(vx * vy + 1e-12)


def score_ripple(spike_list, unit_ids, rate, centers_pos, t0, t1, dt=0.02,
                 min_active=5, min_bins=4):
    counts, tc = bin_spikes(spike_list, unit_ids, t0, t1, dt)
    if counts.shape[0] < min_bins:
        return None
    active = (counts.sum(0) > 0).sum()
    if active < min_active:
        return None
    # keep only bins with at least one spike
    good = counts.sum(1) > 0
    if good.sum() < min_bins:
        return None
    post = bayes_decode(counts[good], rate, dt)
    r = weighted_corr(post, centers_pos, tc[good])
    return dict(post=post, tc=tc[good], r=r, n_active=int(active), counts=counts[good])


def shuffle_scores(spike_list, unit_ids, rate, centers_pos, t0, t1, dt, n_shuf, rng):
    """Column (cell-identity) shuffle of the rate template -> null replay scores."""
    counts, tc = bin_spikes(spike_list, unit_ids, t0, t1, dt)
    good = counts.sum(1) > 0
    counts = counts[good]; tc = tc[good]
    out = np.empty(n_shuf)
    for s in range(n_shuf):
        perm = rng.permutation(rate.shape[1])
        post = bayes_decode(counts, rate[:, perm], dt)
        out[s] = weighted_corr(post, centers_pos, tc)
    return out
