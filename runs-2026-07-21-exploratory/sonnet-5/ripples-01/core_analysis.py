import numpy as np
import pandas as pd
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr


def load_nwb(url):
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_hdf5_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f, mode='r')
    nwbfile = io.read()
    return nwbfile, io


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


def select_ripple_channel(lfp, fs, t_start, t_dur, low=100, high=250):
    """Pick the LFP channel with highest ripple-band power over a probe window."""
    i0 = int(t_start * fs)
    i1 = int((t_start + t_dur) * fs)
    chunk = lfp.data[i0:i1, :].astype(np.float32)
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    power = np.zeros(chunk.shape[1])
    for ch in range(chunk.shape[1]):
        filt = filtfilt(b, a, chunk[:, ch])
        env = np.abs(hilbert(filt))
        power[ch] = np.mean(env ** 2)
    return int(np.argmax(power)), power


def get_behavior_keys(nwbfile):
    beh = nwbfile.processing['behavior']
    keys = list(beh.data_interfaces.keys())
    lin_key = [k for k in keys if k.endswith('LinearizedPosition')][0]
    return lin_key


def get_linearized_position(nwbfile):
    beh = nwbfile.processing['behavior']
    lin_key = get_behavior_keys(nwbfile)
    lpos = beh[lin_key]
    ss = list(lpos.spatial_series.values())[0]
    dt = ss.rate  # this field stores the sample period (s), not a rate in Hz
    t = ss.starting_time + np.arange(ss.data.shape[0]) * dt
    d = np.asarray(ss.data[:]).reshape(ss.data.shape[0], -1)[:, 0]
    valid = ~np.isnan(d)
    t_valid = t[valid]
    d_valid = d[valid]

    idx = np.where(valid)[0]
    gap = np.where(np.diff(idx) > 1)[0]
    starts_i = np.r_[0, gap + 1]
    ends_i = np.r_[gap, len(idx) - 1]
    seg_start_t = t[idx[starts_i]]
    seg_end_t = t[idx[ends_i]]
    # guard against zero-length / inverted segments
    keep = seg_end_t > seg_start_t
    run_ep = nap.IntervalSet(start=seg_start_t[keep], end=seg_end_t[keep])

    pos = nap.Tsd(t=t_valid, d=d_valid, time_support=run_ep)
    return pos, run_ep


def compute_place_fields(nwbfile, nwb, min_peak_rate=1.0, min_selectivity=3.0, n_bins=40):
    pos, run_ep = get_linearized_position(nwbfile)
    units = nwb['units']
    meta = units.metadata
    exc = meta[meta['cell_type'] == 'excitatory'].index.values
    pyr = units[exc]
    tc = nap.compute_tuning_curves(pyr, pos, bins=n_bins, epochs=run_ep, return_pandas=True)

    peak_rate = tc.max(axis=0)
    mean_rate = tc.mean(axis=0)
    selectivity = peak_rate / (mean_rate + 1e-9)
    place_cells = tc.columns[(peak_rate > min_peak_rate) & (selectivity > min_selectivity)]
    peak_pos = tc[place_cells].idxmax(axis=0)
    order_sorted = peak_pos.sort_values()
    template_rank = {cid: rank for rank, cid in enumerate(order_sorted.index)}
    return tc, place_cells, template_rank, pos, run_ep


def batched_perm_corr(event_rank, base_ranks, n_shuffle, rng):
    n = len(event_rank)
    rvals = rng.random((n_shuffle, n))
    perm_idx = np.argsort(rvals, axis=1)
    shuf = base_ranks[perm_idx]
    x = event_rank - event_rank.mean()
    y = shuf - shuf.mean(axis=1, keepdims=True)
    num = (x * y).sum(axis=1)
    den = np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum(axis=1))
    den[den == 0] = np.nan
    return num / den


def analyze_replay(units, ripples, template_rank, rng, min_active_cells=4, n_shuffle=200):
    rows = []
    all_shuffle_rhos = []
    cell_ids = np.array(list(template_rank.keys()))
    spike_arrays = {cid: units[cid].t for cid in cell_ids}
    for s, e, p in ripples:
        spike_dict = {}
        for cid in cell_ids:
            ts = spike_arrays[cid]
            mask = (ts >= s) & (ts <= e)
            if mask.any():
                spike_dict[cid] = ts[mask].min()
        if len(spike_dict) < min_active_cells:
            continue
        active_cells = list(spike_dict.keys())
        event_times = np.array([spike_dict[c] for c in active_cells])
        order = np.argsort(event_times)
        event_rank = np.empty(len(active_cells))
        event_rank[order] = np.arange(len(active_cells))
        base_ranks = np.array([template_rank[c] for c in active_cells], dtype=float)
        rho, _ = spearmanr(event_rank, base_ranks)

        shuf_rhos = batched_perm_corr(event_rank, base_ranks, n_shuffle, rng)
        shuf_rhos = shuf_rhos[~np.isnan(shuf_rhos)]
        pval = (np.sum(np.abs(shuf_rhos) >= np.abs(rho)) + 1) / (len(shuf_rhos) + 1)

        rows.append({'start': s, 'end': e, 'rho': rho, 'n_active': len(active_cells), 'pval': pval})
        all_shuffle_rhos.append(shuf_rhos)
    df = pd.DataFrame(rows)
    shuffle_rhos = np.concatenate(all_shuffle_rhos) if all_shuffle_rhos else np.array([])
    return df, shuffle_rhos
