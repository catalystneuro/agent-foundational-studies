import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d


def bandpass(x, fs, low=100, high=250, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


def detect_ripples(raw, fs, t_offset=0.0, low=100, high=250,
                    thresh_high=4.0, thresh_low=1.0,
                    min_dur=0.02, max_dur=0.35, merge_gap=0.03,
                    smooth_sigma_s=0.004):
    filt = bandpass(raw, fs, low, high)
    env = np.abs(hilbert(filt))
    env = gaussian_filter1d(env, sigma=smooth_sigma_s * fs)
    z = (env - env.mean()) / env.std()

    above_high = z > thresh_high
    above_low = z > thresh_low

    # find contiguous runs of above_low
    edges = np.diff(above_low.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if above_low[0]:
        starts = np.r_[0, starts]
    if above_low[-1]:
        ends = np.r_[ends, len(above_low)]

    events = []
    for s, e in zip(starts, ends):
        if not above_high[s:e].any():
            continue
        dur = (e - s) / fs
        if dur < min_dur or dur > max_dur:
            continue
        peak_idx = s + np.argmax(z[s:e])
        events.append((s, e, peak_idx))

    if not events:
        return np.zeros((0, 3))

    # merge events separated by less than merge_gap
    merged = [events[0]]
    for s, e, p in events[1:]:
        ps, pe, pp = merged[-1]
        if (s - pe) / fs < merge_gap:
            new_peak = pp if z[pp] >= z[p] else p
            merged[-1] = (ps, e, new_peak)
        else:
            merged.append((s, e, p))

    out = np.array(merged, dtype=float)
    out[:, 0] = out[:, 0] / fs + t_offset
    out[:, 1] = out[:, 1] / fs + t_offset
    out[:, 2] = out[:, 2] / fs + t_offset
    return out  # columns: start_time, end_time, peak_time


if __name__ == '__main__':
    import time
    from scratch_load import load_nwb

    nwbfile, io = load_nwb()
    lfp = nwbfile.processing['ecephys']['LFP']['LFP']
    fs = lfp.rate
    ch = 55

    epochs = nwbfile.epochs.to_dataframe()
    pre_end = epochs.loc[0, 'stop_time']
    post_start = epochs.loc[2, 'start_time']

    pre_win = (pre_end - 1800, pre_end)
    post_win = (post_start, post_start + 2700)

    for name, (t0, t1) in [('PRE', pre_win), ('POST', post_win)]:
        i0, i1 = int(t0 * fs), int(t1 * fs)
        tt = time.time()
        raw = lfp.data[i0:i1, ch].astype(np.float32)
        print(name, 'fetch', time.time() - tt, raw.shape)
        ripples = detect_ripples(raw, fs, t_offset=t0)
        print(name, 'n_ripples', len(ripples), 'rate/min', len(ripples) / ((t1 - t0) / 60))
        np.save(f'scratch_ripples_{name}.npy', ripples)
        np.save(f'scratch_raw_{name}.npy', raw)
