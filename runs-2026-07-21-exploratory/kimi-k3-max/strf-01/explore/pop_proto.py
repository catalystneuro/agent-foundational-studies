"""Per-fiber STRF analysis function + test on a few fibers."""
import json
import re

import h5py
import numpy as np
import remfile
from scipy.ndimage import gaussian_filter1d

CACHE = remfile.DiskCache("/tmp/remfile_cache_strf")

BIN_S = 0.0005
EDGES = np.arange(-0.005, 0.0705, BIN_S)
CENTERS = EDGES[:-1] + BIN_S / 2
RESP_WIN = (0.005, 0.055)     # for tuning curve
LAT_WIN = (0.0, 0.030)        # for latency


def load_fiber(url):
    """Return dict with sweep spike trains and metadata for one fiber file."""
    f = h5py.File(remfile.File(url, disk_cache=CACHE), "r")
    spike_times = np.asarray(f["units/spike_times"][:], dtype=np.float64)
    spike_index = np.asarray(f["units/spike_times_index"][:], dtype=np.int64)
    tags = f["units/tag"][:].astype(str)
    starts = np.concatenate([[0], spike_index[:-1]])
    sweeps = [spike_times[s:e] for s, e in zip(starts, spike_index)]
    st = "general/intracellular_ephys/intracellular_recordings/stimuli"
    meta = {k: f[f"{st}/{k}"][:] for k in ("stimtype", "frequency", "level_dBSPL", "delay", "duration")}
    at = "analysis/analysis_table"
    analysis = {k: f[f"{at}/{k}"][:] for k in ("experiment", "results_bf", "results_sr", "results_threshold")}
    age = f["general/subject/age"][()]
    age = age.decode() if isinstance(age, bytes) else str(age)
    # sweep duration + sampling rate from the first acquisition series
    k0 = tags[0]
    n_samp = f["acquisition"][k0]["data"].shape[0]
    rate = float(f["acquisition"][k0]["starting_time"].attrs["rate"])
    out = dict(sweeps=sweeps, tags=tags, meta=meta, analysis=analysis, age=age,
               sweep_dur=n_samp / rate, fs=rate)
    return out, f


def bf_sweeps_by_level(data):
    """Parse BF tone sweeps -> {level: {freq: [sweep_idx]}}."""
    tags = data["tags"]
    lev = data["meta"]["level_dBSPL"]
    by_level = {}
    for i, t in enumerate(tags):
        m = re.match(r"BF_FREQ(\d+)_rep(\d+)", t)
        if m:
            fr = int(m.group(1))
            lv = float(lev[i])
            by_level.setdefault(lv, {}).setdefault(fr, []).append(i)
    return by_level


def pick_level(by_level):
    """Pick the level with the most (freq x rep) coverage; tie -> higher level."""
    best, best_key = None, None
    for lv, d in by_level.items():
        n = sum(len(v) for v in d.values())
        key = (n, lv)
        if best_key is None or key > best_key:
            best, best_key = lv, key
    return best


def compute_map(data, level):
    """Per-frequency PSTH map at one level. Returns freqs, net smoothed map, raw psth, sr."""
    by_level = bf_sweeps_by_level(data)
    d = by_level[level]
    freqs = np.array(sorted(d))
    sweeps = data["sweeps"]
    # modal delay across BF sweeps at this level
    idx_all = [i for v in d.values() for i in v]
    delay = float(np.median(data["meta"]["delay"][idx_all]))
    # spontaneous rate from silent sweeps
    sil = [i for i, t in enumerate(data["tags"]) if t.startswith("BF_silent")]
    if sil:
        sr = np.mean([np.sum(np.isfinite(sweeps[i])) for i in sil]) / data["sweep_dur"]
    else:
        sr = np.nan
    psth = np.zeros((len(freqs), len(CENTERS)))
    for fi, fr in enumerate(freqs):
        sp = np.concatenate([sweeps[i] - delay for i in d[fr]])
        sp = sp[np.isfinite(sp)]
        sp = sp[(sp >= EDGES[0]) & (sp < EDGES[-1])]
        c, _ = np.histogram(sp, bins=EDGES)
        psth[fi] = c / (len(d[fr]) * BIN_S)
    if not np.isfinite(sr):  # fallback: pre-onset baseline
        pre = CENTERS < 0
        sr = np.nanmean(psth[:, pre])
    net = psth - sr
    net_s = gaussian_filter1d(net, sigma=2, axis=1)
    return freqs, psth, net, net_s, sr, delay


def fiber_metrics(freqs, net, net_s, sr):
    """CF, latency, bandwidth, peak rate from a net-rate map."""
    rw = (CENTERS >= RESP_WIN[0]) & (CENTERS < RESP_WIN[1])
    tc = net[:, rw].mean(axis=1)
    ci = int(np.argmax(tc))
    cf = float(freqs[ci])
    peak = float(tc[ci])
    # latency at CF: half-peak crossing on smoothed CF row
    lw = (CENTERS >= LAT_WIN[0]) & (CENTERS < LAT_WIN[1])
    row = net_s[ci, lw]
    tt = CENTERS[lw]
    lat = np.nan
    if row.max() > 0:
        thr = 0.5 * row.max()
        above = np.where(row >= thr)[0]
        if len(above):
            lat = float(tt[above[0]])
    # bandwidth: half-max width in octaves (contiguous around CF)
    bw = np.nan
    if peak > 0:
        half = peak / 2
        lo, hi = ci, ci
        while lo > 0 and tc[lo - 1] >= half:
            lo -= 1
        while hi < len(tc) - 1 and tc[hi + 1] >= half:
            hi += 1
        if hi > lo:
            bw = float(np.log2(freqs[hi] / freqs[lo]))
    return dict(cf=cf, peak_net_rate=peak, latency_s=lat, bandwidth_oct=bw, sr=float(sr))


def analyze_fiber(url):
    data, f = load_fiber(url)
    by_level = bf_sweeps_by_level(data)
    level = pick_level(by_level)
    freqs, psth, net, net_s, sr, delay = compute_map(data, level)
    m = fiber_metrics(freqs, net, net_s, sr)
    m.update(dict(level=level, n_freqs=len(freqs),
                  n_reps=int(np.mean([len(v) for v in by_level[level].values()])),
                  age=data["age"],
                  bf_reported=float(data["analysis"]["results_bf"][0]),
                  sr_reported=float(data["analysis"]["results_sr"][0]),
                  delay=delay))
    f.close()
    return m, freqs, net_s


if __name__ == "__main__":
    import pandas as pd
    sel = pd.read_csv("explore/selected_fibers.csv")
    for _, r in sel.iloc[[0, 15, 30, 45, 58]].iterrows():
        m, freqs, net_s = analyze_fiber(r["url"])
        print(f"{r['session']:22s} level={m['level']:.0f} nF={m['n_freqs']} reps={m['n_reps']} "
              f"CF={m['cf']:.0f}Hz (rep {m['bf_reported']:.0f}) peak={m['peak_net_rate']:.0f} sp/s "
              f"lat={m['latency_s']*1e3:.1f}ms bw={m['bandwidth_oct']:.2f}oct SR={m['sr']:.0f} age={m['age']}")
