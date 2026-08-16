"""Core STRF pipeline for one 000986 session. Imported by later scripts."""
import numpy as np
import lindi
from pynwb import NWBHDF5IO
from scipy import stats

FREQ_VALS = np.array([2000., 4000., 8000., 16000., 32000.])
BIN_EDGES = np.arange(-0.05, 0.155, 0.005)   # -50..150 ms @ 5 ms
BIN_C = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BASE_MASK = BIN_C < 0
EVOKED_MASK = (BIN_C >= 0.005) & (BIN_C < 0.060)   # 5-60 ms response window (memory-proven)

ASSETS = {
    'LA3_ses3':  '5e111970-9331-41d0-81b2-829e1c0f8040',
    'LA8_ses1':  '60303460-38be-44a0-951e-82c7957d1217',
    'LA8_ses2':  'ce06d820-e471-4413-a3a8-9c0b21da8680',
    'LA9_ses1':  'd19d0ca3-7c9a-41fe-bd97-b4ee8899a612',
    'LA11_ses2': 'b8d3abca-0e78-4df1-9a51-d122a383be63',
    'LA12_ses1': 'eb82c81a-87a0-40a4-b70e-535ac0909c86',
}

def load_session(asset_id):
    url = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{asset_id}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache())
    nwbfile = NWBHDF5IO(file=f).read()
    sp = f['units/spike_times'][:]
    idx = f['units/spike_times_index'][:]
    starts = np.concatenate([[0], idx[:-1]])
    unit_spikes = [np.asarray(sp[s:e]) for s, e in zip(starts, idx)]
    tr = nwbfile.trials.to_dataframe()
    onsets = tr['start_time'].values
    freqs = tr['stim_frequency'].values.astype(float)
    order = np.argsort(onsets)
    return unit_spikes, onsets[order], freqs[order], nwbfile

def trial_bin_counts(spikes, onsets):
    """counts[trial, timebin] of spikes relative to onset."""
    counts = np.zeros((len(onsets), len(BIN_C)), dtype=np.int16)
    for ti, t0 in enumerate(onsets):
        lo, hi = np.searchsorted(spikes, [t0 + BIN_EDGES[0], t0 + BIN_EDGES[-1]])
        if hi > lo:
            c, _ = np.histogram(spikes[lo:hi] - t0, bins=BIN_EDGES)
            counts[ti] = c
    return counts

def rate_map_strf(counts, freqs):
    """Baseline-subtracted mean rate per (freq, timebin). Returns (5, n_bins)."""
    out = np.zeros((len(FREQ_VALS), len(BIN_C)))
    for fi, fv in enumerate(FREQ_VALS):
        m = freqs == fv
        rate = counts[m].mean(axis=0) / 0.005
        out[fi] = rate - counts[m][:, BASE_MASK].mean() / 0.005
    return out

def responsiveness_stats(counts, freqs):
    """Wilcoxon evoked-vs-baseline (best freq) + Kruskal-Wallis across freqs."""
    ev = counts[:, EVOKED_MASK].sum(axis=1)
    ba = counts[:, BASE_MASK].sum(axis=1) * (EVOKED_MASK.sum() / BASE_MASK.sum())
    # best frequency by evoked count
    best_p, best_kw = 1.0, 1.0
    per_freq_ev = []
    for fv in FREQ_VALS:
        m = freqs == fv
        per_freq_ev.append(ev[m])
        if m.sum() > 10:
            try:
                p = stats.wilcoxon(ev[m], ba[m]).pvalue
            except ValueError:
                p = 1.0
            best_p = min(best_p, p)
    try:
        best_kw = stats.kruskal(*per_freq_ev).pvalue
    except ValueError:
        pass
    return best_p, best_kw

def characterize_strf(strf):
    """BF, peak latency, peak/baseline stats, suppression, separability (SVD on 0-100ms)."""
    resp = strf[:, ~BASE_MASK & (BIN_C < 0.100)]
    t = BIN_C[~BASE_MASK & (BIN_C < 0.100)]
    pk = np.unravel_index(np.argmax(resp), resp.shape)
    bf = FREQ_VALS[pk[0]]
    peak_lat = t[pk[1]]
    peak_hz = resp[pk]
    min_hz = resp.min()
    # separability: variance explained by rank-1
    M = resp - resp.mean()
    sv = np.linalg.svd(M, compute_uv=False)
    sep = sv[0]**2 / np.sum(sv**2) if sv.sum() > 0 else np.nan
    return dict(bf=bf, peak_lat=peak_lat, peak_hz=peak_hz, min_hz=min_hz, separability=sep)
