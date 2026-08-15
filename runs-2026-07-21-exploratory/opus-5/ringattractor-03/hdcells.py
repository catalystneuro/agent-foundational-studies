"""Head-direction cell identification from wake tuning curves."""
import numpy as np
import pandas as pd
import pynapple as nap

NBINS = 60
TWO_PI = 2 * np.pi
ANG = (np.arange(NBINS) + 0.5) * TWO_PI / NBINS
EDGES = np.linspace(0, TWO_PI, NBINS + 1)


def tuning(spikes, hd, ep, nb_bins=NBINS):
    """Head-direction tuning curves, (nb_bins, n_units) DataFrame in Hz."""
    return nap.compute_tuning_curves(spikes, hd, bins=nb_bins, range=(0, TWO_PI),
                                     epochs=ep, return_pandas=True)


def mvl_pref(tc_values):
    """Mean vector length and preferred direction from a (nbins, nunits) array."""
    v = np.nan_to_num(np.asarray(tc_values, dtype=float))
    tot = v.sum(0)
    z = (v * np.exp(1j * ANG)[:, None]).sum(0) / np.where(tot > 0, tot, np.nan)
    return np.abs(z), np.angle(z) % TWO_PI


def _spike_hd_index(spikes, hd, ep):
    """For each unit, the index of the nearest HD sample at each of its spikes."""
    hd_ep = hd.restrict(ep)
    ht = hd_ep.index.values
    out = {}
    for u in spikes.keys():
        st = spikes[u].restrict(ep).index.values
        idx = np.searchsorted(ht, st)
        idx = np.clip(idx, 0, len(ht) - 1)
        out[u] = idx
    return hd_ep, out


def screen_hd_cells(spikes, hd, wake, min_rate=0.5, max_rate=50.0,
                    mvl_thresh=0.25, n_shuffle=500, stab_thresh=0.5, seed=0):
    """Per-unit head-direction statistics with an `is_hd` flag.

    A unit counts as an HD cell when (a) its wake firing rate is in a plausible
    single-unit range, (b) its mean vector length exceeds `mvl_thresh` and the
    99th percentile of a circular-shift null, and (c) tuning curves built from
    the two halves of the wake epoch correlate above `stab_thresh`.
    """
    rng = np.random.default_rng(seed)
    tc = tuning(spikes, hd, wake)
    mvl, pref = mvl_pref(tc.values)
    rate = np.array([len(spikes[u].restrict(wake)) / wake.tot_length() for u in spikes.keys()])

    # split-half stability across the two halves of the wake epoch
    t0, t1 = wake.start, wake.end
    k = np.searchsorted(np.cumsum(t1 - t0), wake.tot_length() / 2)
    h1 = nap.IntervalSet(start=t0[:k + 1], end=t1[:k + 1])
    h2 = nap.IntervalSet(start=t0[k + 1:], end=t1[k + 1:])
    tc1 = np.nan_to_num(tuning(spikes, hd, h1).values)
    tc2 = np.nan_to_num(tuning(spikes, hd, h2).values)
    stab = np.array([np.corrcoef(tc1[:, i], tc2[:, i])[0, 1] for i in range(tc1.shape[1])])

    # circular-shift null: shift each unit's spike-to-HD assignment as a block,
    # which preserves both the occupancy map and the unit's spike-count
    hd_ep, sidx = _spike_hd_index(spikes, hd, wake)
    hd_bin = np.clip(np.digitize(hd_ep.values, EDGES) - 1, 0, NBINS - 1)
    dt = np.median(np.diff(hd_ep.index.values))
    occ = np.bincount(hd_bin, minlength=NBINS) * dt
    null = np.zeros((n_shuffle, len(sidx)))
    n_hd = len(hd_bin)
    for j in range(n_shuffle):
        shift = rng.integers(n_hd)
        rolled = np.roll(hd_bin, shift)
        for i, u in enumerate(spikes.keys()):
            cnt = np.bincount(rolled[sidx[u]], minlength=NBINS)
            null[j, i] = mvl_pref((cnt / occ)[:, None])[0][0]
    p99 = np.nanpercentile(null, 99, axis=0)

    df = pd.DataFrame(dict(unit=list(spikes.keys()), rate=rate, mvl=mvl, pref=pref,
                           stability=stab, mvl_null99=p99))
    df["is_hd"] = ((df.rate > min_rate) & (df.rate < max_rate) & (df.mvl > mvl_thresh)
                   & (df.mvl > df.mvl_null99) & (df.stability > stab_thresh))
    return df, tc
