"""Frequency tuning of single auditory-nerve fibres (DANDI:001262, Mongolian gerbil).

Each fibre was probed with a frequency x level grid of 50 ms tone bursts (3 repeats per
cell). From the resulting frequency response area we derive the classical frequency
threshold curve: the lowest sound level at which each frequency drives the fibre above its
spontaneous rate. The tip of that curve defines the characteristic frequency (CF), and its
width 10 dB above the tip defines the sharpness of tuning, Q10 = CF / bandwidth.
"""

import numpy as np
import pynapple as nap
from scipy.ndimage import median_filter

import dandi_io as dio

nap.nap_config.suppress_conversion_warnings = True

SLOT = 0.25          # spacing used to lay 200 ms sweeps on a common timeline
TONE_WIN = (0.010, 0.060)   # tone burst, measured from the sweep PSTH
SPONT_WIN = (0.130, 0.200)  # late part of the sweep, after post-stimulus suppression
CRITERION = 20.0     # spikes/s above spontaneous rate, the standard threshold criterion


def fibre_to_pynapple(fib):
    """Put a fibre's sweeps on one timeline and return (spikes, tone_ep, spont_ep)."""
    starts = np.arange(len(fib["sweeps"])) * SLOT
    times = np.sort(np.concatenate([s + t0 for s, t0 in zip(fib["sweeps"], starts)]))
    spikes = nap.Ts(times)
    tone_ep = nap.IntervalSet(start=starts + TONE_WIN[0], end=starts + TONE_WIN[1])
    spont_ep = nap.IntervalSet(start=starts + SPONT_WIN[0], end=starts + SPONT_WIN[1])
    return spikes, tone_ep, spont_ep


def sweep_rates(spikes, ep, win):
    """Spike rate in every interval of `ep` (one value per sweep)."""
    width = win[1] - win[0]
    counts = spikes.count(bin_size=width + 1e-9, ep=ep)
    return np.asarray(counts.values).ravel() / width


def response_area(fib):
    """Mean driven rate for every (frequency, level) cell of the grid."""
    spikes, tone_ep, spont_ep = fibre_to_pynapple(fib)
    r_tone = sweep_rates(spikes, tone_ep, TONE_WIN)
    r_spont_per_sweep = sweep_rates(spikes, spont_ep, SPONT_WIN)

    freqs = np.unique(fib["frequency"])
    levels = np.unique(fib["level"])
    fra = np.full((len(levels), len(freqs)), np.nan)
    for i, lv in enumerate(levels):
        for j, fq in enumerate(freqs):
            m = (fib["level"] == lv) & (fib["frequency"] == fq)
            if m.any():
                fra[i, j] = r_tone[m].mean()
    return dict(
        fra=fra,
        freqs=freqs,
        levels=levels,
        spont=float(np.mean(r_spont_per_sweep)),
        spont_sd=float(np.std(r_spont_per_sweep)),
        rate_per_sweep=r_tone,
    )


def threshold_curve(ra):
    """Lowest level at which each frequency exceeds spontaneous rate + criterion.

    Returns thresholds (NaN where the fibre never responded), CF, CF threshold and Q10.
    """
    fra, levels, freqs = ra["fra"], ra["levels"], ra["freqs"]
    crit = ra["spont"] + CRITERION
    thresholds = np.full(len(freqs), np.nan)
    for j in range(len(freqs)):
        above = np.where(fra[:, j] >= crit)[0]
        if len(above):
            # require the response to be sustained at higher levels too, which rejects
            # single noisy cells at low level
            for i in above:
                if np.all(fra[i:, j] >= crit):
                    thresholds[j] = levels[i]
                    break
    out = dict(thresholds=thresholds, cf=np.nan, cf_threshold=np.nan, q10=np.nan)
    if np.all(np.isnan(thresholds)):
        return out

    # frequencies that never drove the fibre have a threshold above the tested range;
    # a 3-point median filter then removes single-cell noise in the curve, which
    # otherwise truncates the bandwidth at an arbitrary point
    filled = np.where(np.isnan(thresholds), levels.max() + np.diff(levels).mean(), thresholds)
    smooth = median_filter(filled, size=3, mode="nearest")

    thr = float(np.min(smooth))
    tied = np.where(smooth == thr)[0]
    cf = float(np.exp(np.mean(np.log(freqs[tied]))))
    j_cf = int(tied[len(tied) // 2])
    out["cf"], out["cf_threshold"] = cf, thr
    out["thresholds_smooth"] = smooth

    # bandwidth 10 dB above threshold, by linear interpolation on the flanks
    lo = _cross(freqs, smooth, j_cf, thr + 10, -1)
    hi = _cross(freqs, smooth, j_cf, thr + 10, +1)
    if lo is not None and hi is not None and hi > lo:
        out["q10"] = float(cf / (hi - lo))
        out["bw10"] = (lo, hi)
        out["bw10_octaves"] = float(np.log2(hi / lo))
    return out


def _cross(freqs, thresholds, j0, target, step):
    """Walk out from the CF until the threshold curve crosses `target` dB; interpolate."""
    j = j0
    while 0 <= j + step < len(freqs):
        j1 = j + step
        t0, t1 = thresholds[j], thresholds[j1]
        if np.isnan(t1):
            return None
        if t1 >= target:
            if t1 == t0:
                return freqs[j1]
            return freqs[j] + (freqs[j1] - freqs[j]) * (target - t0) / (t1 - t0)
        j = j1
    return None


def half_max_bandwidth(freqs, driven):
    """Width of a tuning curve at half its peak, in octaves, around the peak.

    The curve is median-filtered first: the frequency grid is much finer than the
    fibre's bandwidth, so an unsmoothed curve can dip below half maximum one point away
    from the peak by chance and yield an arbitrarily small width.

    Returns (bandwidth_octaves, best_frequency, peak_at_edge). The bandwidth is NaN when
    the curve does not fall to half maximum on both sides inside the tested range.
    """
    d = np.clip(driven, 0, None)
    if d.max() <= 0:
        return np.nan, np.nan, True
    ds = median_filter(d, size=3, mode="nearest") if len(d) >= 5 else d
    lf = np.log2(freqs)
    pk = int(np.argmax(ds))
    half = ds[pk] / 2
    at_edge = pk < 2 or pk > len(ds) - 3

    def edge(step):
        j = pk
        while 0 <= j + step < len(ds):
            if ds[j + step] < half:
                f0, f1 = lf[j], lf[j + step]
                return f0 + (f1 - f0) * (ds[j] - half) / (ds[j] - ds[j + step])
            j += step
        return np.nan

    lo, hi = edge(-1), edge(+1)
    bw = hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan
    return bw, float(freqs[pk]), bool(at_edge)


def iso_level_tuning(fib, n_perm=2000, rng=None):
    """Frequency tuning from the fixed-level BF sweeps, present in most fibre files."""
    rng = rng or np.random.default_rng(0)
    freq = fib["iso_frequency"]
    if len(freq) < 20 or len(np.unique(freq)) < 5:
        return None

    starts = np.arange(len(fib["iso_sweeps"])) * SLOT
    times = np.sort(np.concatenate([s + t0 for s, t0 in zip(fib["iso_sweeps"], starts)]))
    spikes = nap.Ts(times)
    tone_ep = nap.IntervalSet(start=starts + TONE_WIN[0], end=starts + TONE_WIN[1])
    spont_ep = nap.IntervalSet(start=starts + SPONT_WIN[0], end=starts + SPONT_WIN[1])
    r_tone = sweep_rates(spikes, tone_ep, TONE_WIN)
    spont = float(sweep_rates(spikes, spont_ep, SPONT_WIN).mean())

    freqs = np.unique(freq)
    tuning = np.array([r_tone[freq == f].mean() for f in freqs])
    driven = tuning - spont

    # permutation test on the between-frequency variance of the mean driven rate
    G = np.array([(freq == f) / np.sum(freq == f) for f in freqs])
    obs = np.var(tuning)
    with np.errstate(all="ignore"):   # spurious FP flags from Accelerate's matmul
        null = np.array([np.var(G @ r_tone[rng.permutation(len(freq))])
                         for _ in range(n_perm)])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)

    bw, bf, at_edge = half_max_bandwidth(freqs, driven)
    return dict(iso_freqs=freqs, iso_tuning=tuning, iso_driven=driven, iso_spont=spont,
                iso_bf=bf, iso_bw_octaves=bw, iso_p=float(p), iso_bf_at_edge=at_edge,
                n_iso_sweeps=len(fib["iso_sweeps"]))


def analyze_fibre(url, rng=None):
    """Per-fibre pipeline. Returns None if the file has neither stimulus set."""
    fib = dio.load_fibre_001262(url)
    out = dict(fibre=fib["fibre"], subject=fib["subject"], age_days=fib["age_days"],
               n_sweeps=len(fib["sweeps"]))

    has_grid = len(fib["sweeps"]) >= 50 and len(np.unique(fib["frequency"])) >= 5
    if has_grid:
        ra = response_area(fib)
        out.update(**ra, **threshold_curve(ra))

    iso = iso_level_tuning(fib, rng=rng)
    if iso is not None:
        out.update(**iso)

    return out if (has_grid or iso is not None) else None


if __name__ == "__main__":
    files = dio.list_assets("001262", "0.241205.0959")
    res = analyze_fibre(files[0][1])
    print(res["fibre"], "spont %.1f Hz" % res["spont"])
    print("freqs", res["freqs"])
    print("thresholds", res["thresholds"])
    print("CF %.0f Hz  threshold %.0f dB  Q10 %.2f" % (res["cf"], res["cf_threshold"], res["q10"]))
