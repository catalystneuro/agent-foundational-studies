"""Place-field analysis routines built on pynapple."""
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

nap.nap_config.suppress_conversion_warnings = True

N_BINS = 40          # 4 cm bins over a 1.6 m track
SMOOTH_SD_BINS = 1.5
SPEED_THRESH = 0.10  # m/s


def traversals(beh, direction):
    """One IntervalSet row per end-to-end traversal of the track in `direction`."""
    return beh["right_ep"] if direction == "right" else beh["left_ep"]


def running_epochs(beh, direction, laps=None):
    """Traversals for one heading, restricted to samples above a running-speed floor.

    `laps` optionally selects a subset of traversals (used for odd/even cross-validation).
    """
    ep = traversals(beh, direction)
    if laps is not None:
        ep = ep[np.asarray(laps)]
    fast = beh["speed"].threshold(SPEED_THRESH, "above").time_support
    return ep.intersect(fast).drop_short_intervals(0.2)


def rate_maps(units, beh, epochs, bins=N_BINS, smooth=SMOOTH_SD_BINS):
    """Occupancy-normalised firing rate vs. position, gaussian-smoothed."""
    tc = nap.compute_tuning_curves(
        units, beh["position"], bins=bins, range=[(0.0, beh["track_len"])],
        epochs=epochs, feature_names=["position"],
    )
    occ = tc.attrs["occupancy"]
    data = np.nan_to_num(tc.values, nan=0.0)
    if smooth:
        data = gaussian_filter1d(data, smooth, axis=-1, mode="nearest")
    tc = tc.copy(data=data)
    tc.attrs["occupancy"] = occ
    return tc


def spatial_information(tc):
    """Skaggs spatial information in bits per spike, one value per unit."""
    occ = np.asarray(tc.attrs["occupancy"], dtype=float)
    p = occ / occ.sum()
    lam = np.asarray(tc.values)                      # (n_units, n_bins)
    mean_rate = (lam * p).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / mean_rate[:, None]
        term = np.where(lam > 0, p * ratio * np.log2(ratio), 0.0)
    return np.where(mean_rate > 0, term.sum(axis=1), 0.0)


def sparsity(tc):
    """Skaggs sparsity: <lambda>^2 / <lambda^2>. Low values = compact fields."""
    occ = np.asarray(tc.attrs["occupancy"], dtype=float)
    p = occ / occ.sum()
    lam = np.asarray(tc.values)
    num = (lam * p).sum(axis=1) ** 2
    den = (lam ** 2 * p).sum(axis=1)
    return np.where(den > 0, num / den, np.nan)


def _compress(epochs):
    """Map an IntervalSet onto a gap-free timeline; returns (starts, durations, offsets)."""
    starts = np.asarray(epochs.start, dtype=float)
    ends = np.asarray(epochs.end, dtype=float)
    dur = ends - starts
    offsets = np.r_[0.0, np.cumsum(dur)[:-1]]
    return starts, dur, offsets


def circular_shift(units, epochs, rng, min_shift=5.0):
    """Circularly shift each spike train *within the concatenated analysis epochs*.

    Shifting over the whole session would move spikes from immobility into the running
    windows and inflate the shuffled rate maps, so the shift is applied on a gap-free
    timeline built from the running epochs only. Spike counts are exactly preserved.
    """
    starts, dur, offsets = _compress(epochs)
    total = dur.sum()
    out = {}
    for i, k in enumerate(units.keys()):
        t = units[k].restrict(epochs).t
        if t.size == 0:
            out[i] = np.array([])
            continue
        idx = np.searchsorted(starts, t, side="right") - 1
        u = offsets[idx] + (t - starts[idx])
        u = np.mod(u + rng.uniform(min_shift, total - min_shift), total)
        j = np.clip(np.searchsorted(offsets, u, side="right") - 1, 0, len(starts) - 1)
        out[i] = np.sort(starts[j] + (u - offsets[j]))
    return nap.TsGroup(out, time_support=epochs)


def si_null(units, beh, epochs, n_shuffles=200, seed=0, progress=True):
    """Null distribution of spatial information from circularly shifted spike trains."""
    rng = np.random.default_rng(seed)
    null = np.empty((n_shuffles, len(units)))
    it = range(n_shuffles)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc="SI shuffles", leave=False)
    for j in it:
        sh = circular_shift(units, epochs, rng)
        null[j] = spatial_information(rate_maps(sh, beh, epochs))
    return null


def field_stats(tc, track_len):
    """Peak rate, peak location, and field width (contiguous run above 50% of peak)."""
    lam = np.asarray(tc.values)
    centers = np.asarray(tc.coords["position"].values)
    bin_w = centers[1] - centers[0]
    peak_rate = lam.max(axis=1)
    peak_idx = lam.argmax(axis=1)
    widths = np.zeros(lam.shape[0])
    for i in range(lam.shape[0]):
        above = lam[i] >= 0.5 * peak_rate[i]
        j = peak_idx[i]
        lo = j
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = j
        while hi < lam.shape[1] - 1 and above[hi + 1]:
            hi += 1
        widths[i] = (hi - lo + 1) * bin_w
    return dict(peak_rate=peak_rate, peak_pos=centers[peak_idx], width=widths)


def odd_even_laps(beh, direction):
    n = len(traversals(beh, direction))
    return (running_epochs(beh, direction, np.arange(0, n, 2)),
            running_epochs(beh, direction, np.arange(1, n, 2)))


def corr_rows(a, b):
    r = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        if a[i].std() > 0 and b[i].std() > 0:
            r[i] = np.corrcoef(a[i], b[i])[0, 1]
    return r


def split_half_stability(units, beh, direction, bins=N_BINS):
    """Correlation between rate maps built from odd- and even-numbered traversals."""
    odd, even = odd_even_laps(beh, direction)
    a = np.asarray(rate_maps(units, beh, odd, bins=bins).values)
    b = np.asarray(rate_maps(units, beh, even, bins=bins).values)
    return corr_rows(a, b)


def classify(units, beh, direction, n_shuffles=200, seed=0, bins=N_BINS,
             min_spikes=50, min_peak=1.0, min_stability=0.5, progress=True):
    """Per-unit place-field measures and a place-cell label for one running direction.

    A unit is called a place cell if it fires enough during running (>= `min_spikes`
    spikes, peak rate >= `min_peak` Hz), its spatial information exceeds the 95th
    percentile of a within-epoch circular-shift null, and its odd/even-lap rate maps
    correlate above `min_stability`.
    """
    ep = running_epochs(beh, direction)
    tc = rate_maps(units, beh, ep, bins=bins)
    si = spatial_information(tc)
    null = si_null(units, beh, ep, n_shuffles=n_shuffles, seed=seed, progress=progress)
    p95 = np.percentile(null, 95, axis=0)
    pval = (null >= si[None, :]).mean(axis=0)
    stab = split_half_stability(units, beh, direction, bins=bins)
    fs = field_stats(tc, beh["track_len"])
    n_spk = np.array([len(units[k].restrict(ep)) for k in units.keys()])
    active = (n_spk >= min_spikes) & (fs["peak_rate"] >= min_peak)
    is_place = active & (si > p95) & (stab > min_stability)
    return dict(tc=tc, si=si, si_null=null, si_p95=p95, pval=pval, stability=stab,
                sparsity=sparsity(tc), n_spikes=n_spk, active=active,
                is_place_cell=is_place, epochs=ep, **fs)


def decode_crossval(units, beh, direction, bin_size=0.2, bins=N_BINS, sliding=None):
    """Train place maps on odd traversals, Bayesian-decode position on even ones."""
    train, test = odd_even_laps(beh, direction)
    tc = rate_maps(units, beh, train, bins=bins)
    decoded, prob = nap.decode_bayes(tc, units, test, bin_size=bin_size,
                                     sliding_window_size=sliding)
    truth = beh["position"].interpolate(decoded, ep=test)
    error = np.abs(np.asarray(decoded.d) - np.asarray(truth.d))
    return dict(decoded=decoded, prob=prob, truth=truth, error=error,
                tc_train=tc, test_ep=test)


def directionality(units, beh, bins=N_BINS):
    """Correlation between the two heading-specific rate maps of each unit."""
    a = np.asarray(rate_maps(units, beh, running_epochs(beh, "right"), bins=bins).values)
    b = np.asarray(rate_maps(units, beh, running_epochs(beh, "left"), bins=bins).values)
    peak_a, peak_b = a.max(1), b.max(1)
    dsi = np.abs(peak_a - peak_b) / np.maximum(peak_a + peak_b, 1e-9)
    return dict(r=corr_rows(a, b), dsi=dsi, tc_right=a, tc_left=b)


def select(units, mask):
    """Subset a TsGroup by a boolean mask over its units, keeping original keys."""
    keys = np.asarray(list(units.keys()))
    return units[keys[np.asarray(mask, dtype=bool)]]
