"""Sharp-wave ripple detection from a CA1 LFP channel using Pynapple + SciPy.

Standard pipeline (Buzsaki lab convention):
  1. Band-pass the LFP to the ripple band (150-250 Hz).
  2. Compute the analytic-signal amplitude envelope (Hilbert).
  3. Smooth and z-score the envelope over the target epoch.
  4. Threshold: candidate events cross a high threshold (>5 SD peak) and are
     bounded where the envelope falls below a low threshold (>2 SD).
  5. Keep events within a plausible duration window (15-250 ms) and merge
     events separated by < 15 ms.
"""
import numpy as np
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import next_fast_len


def bandpass(sig, fs, lo=150.0, hi=250.0, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def _fast_hilbert_env(filt):
    """Amplitude envelope via Hilbert, padded to a fast FFT length.

    Prime/awkward signal lengths make scipy.hilbert extremely slow, so we pad
    to next_fast_len and crop back.
    """
    n = filt.size
    nfast = next_fast_len(n)
    padded = np.zeros(nfast, dtype=np.float64)
    padded[:n] = filt
    env = np.abs(hilbert(padded))[:n]
    return env


def ripple_envelope(lfp_tsd, fs, lo=150.0, hi=250.0, smooth_ms=8.0):
    """Return (filtered Tsd, smoothed envelope Tsd) for a pynapple Tsd LFP."""
    sig = np.asarray(lfp_tsd.values, dtype=np.float64)
    filt = bandpass(sig, fs, lo, hi)
    env = _fast_hilbert_env(filt)
    # Gaussian smoothing of the envelope
    win = int(round(smooth_ms * 1e-3 * fs))
    if win > 1:
        t = np.arange(-3 * win, 3 * win + 1)
        g = np.exp(-0.5 * (t / win) ** 2)
        g /= g.sum()
        env = np.convolve(env, g, mode="same")
    filt_tsd = nap.Tsd(t=lfp_tsd.index.values, d=filt)
    env_tsd = nap.Tsd(t=lfp_tsd.index.values, d=env)
    return filt_tsd, env_tsd


def detect_ripples(env_tsd, restrict_ep, fs,
                   low_sd=2.0, high_sd=5.0,
                   min_dur=0.015, max_dur=0.250, merge_gap=0.015):
    """Threshold the ripple envelope within restrict_ep and return an IntervalSet.

    Returns (ripple_epochs IntervalSet, peak_times array, peak_z array, mean, sd).
    """
    env_r = env_tsd.restrict(restrict_ep)
    z = (env_r.values - env_r.values.mean()) / env_r.values.std()
    t = env_r.index.values
    mean, sd = env_r.values.mean(), env_r.values.std()

    above_low = z > low_sd
    # find contiguous runs above the low threshold
    idx = np.flatnonzero(np.diff(np.concatenate([[0], above_low.view(np.int8), [0]])))
    starts, ends = idx[0::2], idx[1::2] - 1
    starts_t, ends_t = t[starts], t[ends]

    # merge events closer than merge_gap
    if len(starts_t) > 1:
        keep_s, keep_e = [starts_t[0]], [ends_t[0]]
        for s, e in zip(starts_t[1:], ends_t[1:]):
            if s - keep_e[-1] < merge_gap:
                keep_e[-1] = e
            else:
                keep_s.append(s); keep_e.append(e)
        starts_t, ends_t = np.array(keep_s), np.array(keep_e)

    peak_t, peak_z = [], []
    good_s, good_e = [], []
    for s, e in zip(starts_t, ends_t):
        m = (t >= s) & (t <= e)
        if not m.any():
            continue
        zz = z[m]
        if zz.max() < high_sd:              # require a strong peak
            continue
        dur = e - s
        if dur < min_dur or dur > max_dur:  # duration gate
            continue
        good_s.append(s); good_e.append(e)
        pk = np.argmax(zz)
        peak_t.append(t[m][pk]); peak_z.append(zz[pk])

    rip = nap.IntervalSet(start=np.array(good_s), end=np.array(good_e))
    return rip, np.array(peak_t), np.array(peak_z), mean, sd
