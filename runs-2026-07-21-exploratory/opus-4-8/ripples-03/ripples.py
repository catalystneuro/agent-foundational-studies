"""Sharp-wave ripple detection from a CA1 LFP channel."""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d

RIPPLE_BAND = (150.0, 250.0)


def bandpass(x, fs, band=RIPPLE_BAND, order=4):
    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def ripple_envelope(lfp, fs, band=RIPPLE_BAND, smooth_ms=8.0):
    filt = bandpass(lfp, fs, band)
    env = np.abs(hilbert(filt))
    env = gaussian_filter1d(env, sigma=smooth_ms / 1000.0 * fs)
    return filt, env


def detect_ripples(lfp, lfp_t, fs, mask, low_thr=2.0, high_thr=5.0,
                   min_dur=0.015, max_dur=0.250, merge_gap=0.015, band=RIPPLE_BAND):
    """Detect ripples on `lfp`. `mask` selects samples used for z-scoring and
    where events are allowed (e.g. POST-sleep / Non-REM). Returns dict of
    per-event arrays (start, peak, stop times in seconds, peak z-score)."""
    filt, env = ripple_envelope(lfp, fs, band)
    mu, sd = env[mask].mean(), env[mask].std()
    z = (env - mu) / sd
    zt = z.copy()
    zt[~mask] = 0.0  # forbid events outside mask

    above = zt > low_thr
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        stops = np.r_[stops, len(above) - 1]

    # merge nearby candidate periods
    merged = []
    for s, e in zip(starts, stops):
        if merged and (s - merged[-1][1]) / fs < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    ev_s, ev_p, ev_e, ev_z = [], [], [], []
    for s, e in merged:
        seg = zt[s:e]
        if seg.size == 0:
            continue
        pk = seg.max()
        dur = (e - s) / fs
        if pk >= high_thr and min_dur <= dur <= max_dur:
            pki = s + int(np.argmax(seg))
            ev_s.append(lfp_t[s]); ev_p.append(lfp_t[pki])
            ev_e.append(lfp_t[e]); ev_z.append(float(pk))
    return dict(start=np.array(ev_s), peak=np.array(ev_p),
                stop=np.array(ev_e), zpeak=np.array(ev_z),
                filt=filt, env=env, z=z)


def build_mask(lfp_t, intervals):
    """Boolean mask over lfp_t for a list of (start, stop) intervals."""
    mask = np.zeros(len(lfp_t), dtype=bool)
    for s, e in intervals:
        i0 = np.searchsorted(lfp_t, s)
        i1 = np.searchsorted(lfp_t, e)
        mask[i0:i1] = True
    return mask
