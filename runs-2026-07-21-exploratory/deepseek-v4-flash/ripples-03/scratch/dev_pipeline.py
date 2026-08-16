"""Development version of the ripple/replay pipeline (debugging, not final)."""
import os, time
import h5py
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.signal as dsp
import scipy.ndimage as ndi
from scipy import stats
import remfile
from tqdm import tqdm
from pynwb import NWBHDF5IO
import pynapple as nap

SEED = 7
rng = np.random.default_rng(SEED)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white"})

# ---------- load ----------
DANDISET_ID = "000044"
ASSET_ID = "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d"
resp = requests.get(
    f"https://api.dandiarchive.org/api/dandisets/{DANDISET_ID}/versions/draft/assets/{ASSET_ID}/download/",
    allow_redirects=False, timeout=120)
s3_url = resp.headers.get("Location")
assert s3_url, "no S3 url"
print("S3 URL resolved")

disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(s3_url, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

units = nwb["units"]
cell_types = np.asarray(units.get_info("cell_type").values, dtype="<U32")
unit_keys = np.array(list(units.keys()), dtype=np.int64)
print("n units:", len(units), "exc:", (cell_types == "excitatory").sum())

epochs_df = nwb["epochs"].as_dataframe()
states_df = nwb["states"].as_dataframe().astype({"start": "float64", "end": "float64"})
print(epochs_df)

pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos_t = np.asarray(pos2d.t)
lin_v = np.asarray(lin.values)
if lin_v.ndim > 1:
    lin_v = lin_v[:, 0]
lin_v = np.asarray(lin_v, dtype=np.float64)
print("pos:", len(pos_t), "rate:", 1/np.median(np.diff(pos_t)))
print("lin valid frac:", np.mean(np.isfinite(lin_v)))

lfp_ds = h5py_file["processing/ecephys/LFP/LFP/data"]
n_lfp, n_chan = lfp_ds.shape
LFP_FS = 1250.0
CONV = float(lfp_ds.attrs.get("conversion", 3.815e-7))

spike_times = np.asarray(h5py_file["units/spike_times"], dtype=np.float64)
spike_index = np.asarray(h5py_file["units/spike_times_index"], dtype=np.int64)
spike_cum = np.concatenate([[0], np.cumsum(spike_index)])
def spikes_for_unit(k):
    """Spike times for unit k, read from the pynapple TsGroup (reliable
    source; the raw h5py spike arrays in this file are partitioned
    differently than the index table implies)."""
    return np.asarray(units[k].t, dtype=np.float64)

exc_keys = unit_keys[cell_types == "excitatory"]

maze_end = epochs_df.loc[epochs_df["label"] == "MazeEpoch", "end"].values[0]
post_start = epochs_df.loc[epochs_df["label"] == "POSTEpoch", "start"].values[0]
pre_end = epochs_df.loc[epochs_df["label"] == "PREEpoch", "end"].values[0]

# ---------- channel selection ----------
post_nrem = states_df[(states_df["label"] == "Non-REM") & (states_df["start"] >= post_start)]
post_nrem = post_nrem.sort_values("end", ascending=False).reset_index(drop=True)
sel_start, sel_end = float(post_nrem["start"].iloc[0]), float(post_nrem["end"].iloc[0])
sel_end = min(sel_end, sel_start + 60.0)
i0, i1 = int(sel_start*LFP_FS), int(sel_end*LFP_FS)
block = np.asarray(lfp_ds[i0:i1, :], dtype=np.float64) * CONV * 1e6
sos = dsp.butter(4, [100.0, 250.0], btype="bandpass", fs=LFP_FS, output="sos")
peak_scores = np.zeros(n_chan)
for ch in tqdm(range(n_chan), desc="channel metric"):
    x = block[:, ch]
    env = np.abs(dsp.sosfiltfilt(sos, x))
    env = ndi.gaussian_filter1d(env, sigma=4)
    pk = np.percentile(env, 99)/np.median(env)
    f, Ps = dsp.welch(x, fs=LFP_FS, nperseg=4096)
    rd = np.mean(Ps[(f >= 100) & (f <= 250)]) / np.mean(Ps[(f >= 1) & (f <= 10)])
    peak_scores[ch] = pk * rd
best_ch = int(np.argmax(peak_scores))
print("best channel:", best_ch)
del block

# ---------- ripple detection ----------
t0 = time.time()
lfp_ch = (np.asarray(lfp_ds[:, best_ch], dtype=np.float32) * CONV * 1e6).astype(np.float32)
print("LFP loaded in", time.time()-t0)
env_full = np.abs(dsp.sosfiltfilt(sos, lfp_ch)).astype(np.float32)
env_full = ndi.gaussian_filter1d(env_full, sigma=4)

nrem_mask = np.zeros(len(env_full), dtype=bool)
for _, r in states_df[states_df["label"] == "Non-REM"].iterrows():
    nrem_mask[int(r["start"]*LFP_FS):int(r["end"]*LFP_FS)] = True
env_mean, env_sd = env_full[nrem_mask].mean(), env_full[nrem_mask].std()
peak_thr, edge_thr = env_mean + 4*env_sd, env_mean + env_sd
print(f"env mean {env_mean:.2f} sd {env_sd:.2f}, peak {peak_thr:.2f} edge {edge_thr:.2f}")

over_edge = env_full > edge_thr
over_peak = env_full > peak_thr
min_gap = int(0.030*LFP_FS)
n = len(env_full)
runs = []
i = 0
while i < n:
    while i < n and not over_edge[i]:
        i += 1
    if i >= n:
        break
    j = i
    while j < n and over_edge[j]:
        j += 1
    k = j
    while k < n:
        g0 = k
        while k < n and not over_edge[k]:
            k += 1
        if k >= n or k - g0 >= min_gap:
            break
        while k < n and over_edge[k]:
            k += 1
        j = k
    if np.any(over_peak[i:j]):
        pk = i + int(np.argmax(env_full[i:j]))
        runs.append((i, pk, j))
    i = j

ripples = np.array(runs, dtype=np.int64)
print("raw candidates:", len(ripples))
dur_s = (ripples[:, 2] - ripples[:, 0]) / LFP_FS
nrem_intervals = states_df.loc[states_df["label"] == "Non-REM", ["start", "end"]].values
good = np.zeros(len(ripples), dtype=bool)
for k in range(len(ripples)):
    s0, e0 = ripples[k, 0]/LFP_FS, ripples[k, 2]/LFP_FS
    good[k] = (0.03 <= dur_s[k] <= 0.5) and np.any((s0 >= nrem_intervals[:, 0] - 1e-3) & (e0 <= nrem_intervals[:, 1] + 1e-3))
ripples = ripples[good]
centers = ripples[:, 1]/LFP_FS
dur_s = (ripples[:, 2] - ripples[:, 0]) / LFP_FS
pre_mask = centers < pre_end
post_mask = centers >= post_start
print("ripples:", len(ripples), "PRE:", pre_mask.sum(), "POST:", post_mask.sum())

# =====================================================================
# FIGURE 1: detection example
# =====================================================================
win_start, win_len = 20300.0, 12.0
i0w, i1w = int(win_start*LFP_FS), int((win_start+win_len)*LFP_FS)
tw = np.arange(i0w, i1w)/LFP_FS
xw = lfp_ch[i0w:i1w]
ew = env_full[i0w:i1w]
in_win = (ripples[:,0] >= i0w) & (ripples[:,2] <= i1w)

fig, ax1 = plt.subplots(figsize=(11, 4.2))
ax1.plot(tw, xw, lw=0.5, color="0.25")
ax2 = ax1.twinx()
ax2.plot(tw, ew, lw=0.8, color="C0")
ax2.axhline(peak_thr, color="r", ls="--", lw=1, label="peak thr")
ax2.axhline(edge_thr, color="k", lw=0.7, ls=":", label="edge thr")
for s0, _, e0 in ripples[in_win]:
    ax1.axvspan(s0/LFP_FS, e0/LFP_FS, color="C3", alpha=0.4)
ax1.set_xlabel("time (s)"); ax1.set_ylabel("LFP (uV)")
ax2.set_ylabel("envelope (uV)")
ax2.legend(loc="upper right")
ax1.set_title("Ripple detection on channel {}: {} ripples in this window".format(best_ch, int(in_win.sum())))
fig.tight_layout(); fig.savefig("figures/fig1_ripple_detection.png"); plt.close(fig)
print("fig1 saved")

# =====================================================================
# FIGURE 2: ripple statistics
# =====================================================================
n_pre, n_post = int(pre_mask.sum()), int(post_mask.sum())
pre_nrem = states_df[(states_df["label"]=="Non-REM") & (states_df["end"] <= pre_end)]
post_nrem = states_df[(states_df["label"]=="Non-REM") & (states_df["start"] >= post_start)]
nonrem_pre_min = float((pre_nrem["end"]-pre_nrem["start"]).sum())/60.0
nonrem_post_min = float((post_nrem["end"]-post_nrem["start"]).sum())/60.0
rate_pre = n_pre/nonrem_pre_min; rate_post = n_post/nonrem_post_min
print("Non-REMPRE {:.1f} min, POST {:.1f} min; rate/min {:.2f} vs {:.2f}".format(
    nonrem_pre_min, nonrem_post_min, rate_pre, rate_post))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].bar(["PRE","POST"], [n_pre, n_post], color=["#7f7f7f", "#1f77b4"])
ax[0].set_ylabel("ripple count"); ax[0].set_title("SWR count")
bins_hist = np.linspace(0, 0.16, 40)
ax[1].hist(dur_s[pre_mask], bins=bins_hist, alpha=0.6, color="#7f7f7f", label="PRE")
ax[1].hist(dur_s[post_mask], bins=bins_hist, alpha=0.6, color="#1f77b4", label="POST")
ax[1].set_xlabel("ripple duration (s)"); ax[1].set_ylabel("count"); ax[1].legend()
ax[1].set_title("duration distribution")
# ripple rate vs time in 20-min bins
binw = 1200.0
n_time = int(np.ceil(len(env_full)/LFP_FS/binw))
ct_rate = np.zeros(n_time); nrem_min = np.zeros(n_time)
for s0, e0 in states_df.loc[states_df["label"]=="Non-REM", ["start","end"]].values:
    b0 = int(s0//binw); b1 = int(e0//binw)
    for k in range(b0, min(b1, n_time-1)+1):
        nrem_min[k] += (min(e0, (k+1)*binw) - max(s0, k*binw))/60.0
for c in centers:
    k = int(c//binw)
    if k < n_time:
        ct_rate[k] += 1
with np.errstate(divide="ignore", invalid="ignore"):
    rate_t = np.where(nrem_min > 0, ct_rate/nrem_min, np.nan)
tt = (np.arange(n_time)+0.5)*binw/3600.0
ax[2].plot(tt, rate_t, color="k", lw=1.2)
ax[2].axvline(pre_end/3600, color="gray", ls=":", lw=1)
ax[2].axvline(post_start/3600, color="gray", ls=":", lw=1)
ax[2].set_xlabel("time (h)"); ax[2].set_ylabel("ripples/min Non-REM")
ax[2].set_title("rate over session")
fig.tight_layout(); fig.savefig("figures/fig2_ripple_stats.png"); plt.close(fig)
print("fig2 saved")

# =====================================================================
# FIGURE 3: ripple-triggered average + spectrogram
# =====================================================================
post_rips = ripples[post_mask]
n_trig = min(300, len(post_rips))
ax_win = int(0.25*LFP_FS)
b_mat = np.zeros((n_trig, 2*ax_win+1), dtype=np.float32)
tr_sel = rng.choice(len(post_rips), n_trig, replace=False)
spg, cnt_sp = None, 0
f_s = t_s_f = None
for k, ri in enumerate(tr_sel):
    pk = post_rips[ri, 1]
    a0, a1 = pk-ax_win, pk+ax_win+1
    if a0 < 0 or a1 > len(lfp_ch):
        continue
    bw = dsp.sosfiltfilt(sos, lfp_ch[a0:a1])
    b_mat[k] = bw
    ft, ts, Sxx = dsp.spectrogram(lfp_ch[a0:a1], fs=LFP_FS, nperseg=64, noverlap=60)
    if spg is None:
        spg = np.zeros_like(Sxx); f_s = ft; t_s_f = ts
    spg += 10*np.log10(Sxx + 1e-12)
    cnt_sp += 1
b_mat = b_mat[b_mat.sum(1) != 0]
bt = np.arange(-ax_win, ax_win+1)/LFP_FS
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(bt, b_mat.T, color="C0", lw=0.4, alpha=0.25)
ax[0].plot(bt, b_mat.mean(0), color="C3", lw=1.8)
ax[0].axvline(0, color="k", ls="--", lw=0.7)
ax[0].set_xlabel("time from ripple peak (s)"); ax[0].set_ylabel("BP LFP (uV)")
ax[0].set_title("POST ripple-triggered average")
im = ax[1].pcolormesh(t_s_f - ax_win/LFP_FS, f_s, spg/cnt_sp, shading="auto", cmap="inferno")
ax[1].set_xlim(-0.15, 0.15); ax[1].set_ylim(0, 300)
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("freq (Hz)")
ax[1].set_title("ripple-locked spectrogram")
plt.colorbar(im, ax=ax[1], label="dB")
ax[2].plot(bt, b_mat.mean(0), color="C3", lw=1.8)
ax[2].set_xlim(-0.1, 0.1); ax[2].axvline(0, color="k", ls="--", lw=0.7)
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("BP LFP (uV)")
ax[2].set_title("zoom")
fig.tight_layout(); fig.savefig("figures/fig3_ripple_waveform.png"); plt.close(fig)
print("fig3 saved")

# =====================================================================
# PLACE CELLS  (direction-pooled, from the linear maze runs)
# =====================================================================
dt_pos = np.median(np.diff(pos_t))

valid = np.isfinite(lin_v)
lin_s = np.where(valid, lin_v, np.nan)
d = np.diff(lin_s) / dt_pos  # forward finite difference; NaN propagates
run_ok_all = np.isfinite(d) & (np.abs(d) > 0.10)
run_ok_all = np.concatenate([run_ok_all, [False]])

# merge runs separated by <= 0.3 s of gap
gap_samples = int(0.31 / dt_pos)
ridx = np.where(run_ok_all)[0]
brm = np.where(np.diff(ridx) > 1 + gap_samples)[0]
bouts_idx = np.split(ridx, brm + 1)
bouts_idx = [b for b in bouts_idx if pos_t[b[-1]] - pos_t[b[0]] > 0.5]
print("run bouts:", len(bouts_idx), "run time: {:.1f} s".format(
    sum(pos_t[b[-1]] - pos_t[b[0]] for b in bouts_idx)))

# concatenated run tape: wall clock times and positions
run_t_all = np.concatenate([pos_t[b] for b in bouts_idx])
run_x_all = np.concatenate([lin_v[b] for b in bouts_idx])

# tau axis: cumulative time along the concatenated run tape
tau_arr = np.empty(len(run_t_all))
offset = 0.0
start = 0
bout_offsets = []
for b in bouts_idx:
    tb = pos_t[b]
    bout_offsets.append(offset)
    tau_arr[start:start + len(b)] = offset + (tb - tb[0])
    offset += tb[-1] - tb[0] + dt_pos
    start += len(b)
bout_offsets = np.array(bout_offsets)
bout_t0 = np.array([pos_t[b[0]] for b in bouts_idx])
bout_t1 = np.array([pos_t[b[-1]] for b in bouts_idx])
tau_total = offset
print("tau total {:.1f} s; tau monotonic: {}".format(
    tau_total, bool(np.all(np.diff(tau_arr) > 0))))

def wall_to_tau(ts):
    """Map wall-clock spike times onto the concatenated run tau axis."""
    b = np.searchsorted(bout_t0, ts, side="right") - 1
    ok = (b >= 0) & (ts >= bout_t0[b]) & (ts <= bout_t1[b] + 1e-6)
    tau = np.full(len(ts), np.nan)
    tau[ok] = bout_offsets[b[ok]] + (ts[ok] - bout_t0[b[ok]])
    return tau

def tau_to_pos(tau):
    """Position at a tau value along the concatenated run tape (nearest sample)."""
    idx = np.clip(np.searchsorted(tau_arr, tau, side="right") - 1, 0, len(tau_arr) - 1)
    return run_x_all[idx]

# occupancy from the run tape
n_bins = 50
edges = np.linspace(0.0, 1.6, n_bins + 1)
bin_c = (edges[:-1] + edges[1:]) / 2
occ = np.histogram(run_x_all, bins=edges)[0].astype(np.float64)
occ_time = occ * dt_pos

def skaggs_si(counts):
    tot = counts.sum()
    if tot == 0 or occ_time.sum() == 0:
        return 0.0
    rate = counts / np.maximum(occ_time, 1e-9)
    lam = tot / occ_time.sum()
    keep = occ_time > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.sum((occ_time[keep] / occ_time.sum()) * (rate[keep] / lam) *
                    np.log2(np.maximum(rate[keep] / lam, 1e-12)))
    return float(si)

n_shuf = 500
n_exc = len(exc_keys)
si_obs = np.zeros(n_exc)
peak_rate = np.zeros(n_exc)
mean_rate = np.zeros(n_exc)
counts_exc = np.zeros((n_exc, n_bins))
p_val = np.ones(n_exc)

for iu in tqdm(range(n_exc), desc="place cells"):
    sp = spikes_for_unit(exc_keys[iu])
    if len(sp) == 0:
        continue
    tau_sp = wall_to_tau(sp)
    keep = np.isfinite(tau_sp)
    if keep.sum() == 0:
        continue
    counts_exc[iu] = np.histogram(tau_to_pos(tau_sp[keep]), bins=edges)[0]
    si_obs[iu] = skaggs_si(counts_exc[iu])
    rate_sm = ndi.gaussian_filter1d(counts_exc[iu] / np.maximum(occ_time, 1e-8), sigma=1.0)
    mean_rate[iu] = counts_exc[iu].sum() / occ_time.sum()
    peak_rate[iu] = rate_sm.max()
    t_sp = tau_sp[keep]
    shuf = np.zeros(n_shuf)
    for s_ in range(n_shuf):
        delta = rng.uniform(0, tau_total)
        tau_sh = (t_sp + delta) % tau_total
        x_sh = tau_to_pos(tau_sh)
        shuf[s_] = skaggs_si(np.histogram(x_sh, bins=edges)[0])
    p_val[iu] = (np.sum(shuf >= si_obs[iu]) + 1) / (n_shuf + 1)

# ---- select place cells ----
place_mask = (p_val < 0.05) & (peak_rate >= 1.0) & (mean_rate >= 0.1)
place_idx = np.where(place_mask)[0]
n_place = len(place_idx)
print("place cells: {} / {} excitatory".format(n_place, n_exc))
print("SI obs: median {:.3f}; shuffle-max q50 {:.3f}".format(
    np.median(si_obs[place_mask]),
    np.median(si_obs[~place_mask]) if (~place_mask).sum() else 0))

# FIGURE 4: place fields
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
ordr = place_idx[np.argsort(si_obs[place_idx])[::-1]]
for j, iu in enumerate(ordr[:3]):
    rate = counts_exc[iu] / np.maximum(occ_time, 1e-8)
    rate_sm = ndi.gaussian_filter1d(rate, sigma=1.0)
    axes[0].plot(bin_c, rate_sm, lw=1.3, label="unit {}".format(exc_keys[iu]))
axes[0].set_xlabel("position (m)"); axes[0].set_ylabel("rate (Hz)")
axes[0].legend(title="top place cells"); axes[0].set_title("example tuning")
# heatmap of place cell rate maps (smoothed, normalized per cell)
im = axes[1].imshow(np.vstack([
    ndi.gaussian_filter1d(counts_exc[iu] / np.maximum(occ_time, 1e-8), 1.0)
    for iu in ordr]), aspect="auto", cmap="magma", origin="lower")
axes[1].set_yticks([]); axes[1].set_xticks(np.linspace(0, n_bins - 1, 5),
                                            np.round(np.linspace(0, 1.6, 5), 1))
axes[1].set_xlabel("position (m)")
axes[1].set_title("{} place cells (by SI)".format(n_place))
plt.colorbar(im, ax=axes[1], label="Hz")
axes[2].hist(si_obs[place_mask], bins=20, color="C0", alpha=0.8, label="place")
axes[2].hist(si_obs[~place_mask], bins=20, color="gray", alpha=0.7, label="non-place")
axes[2].set_xlabel("SI (bits/spike)"); axes[2].set_ylabel("count"); axes[2].legend()
axes[2].set_title("spatial info distribution")
axes[3].bar(["place", "other"], [n_place, n_exc - n_place], color=["C0", "gray"])
axes[3].set_ylabel("cells"); axes[3].set_title("place cell count")
fig.tight_layout(); fig.savefig("figures/fig4_place_fields.png"); plt.close(fig)
print("fig4 saved")

# =====================================================================

# =====================================================================
# REPLAY DECODING: 20 ms bins, Bayesian posterior over position
# =====================================================================
BIN_S = 0.02
N_SHUFF = 500

# Bayes template from place cells (rate maps normalized to unit sum)
lam_map = counts_exc[place_idx] / np.maximum(occ_time, 1e-9)   # (C, n_bin) Hz
lam_map = lam_map / lam_map.sum(axis=1, keepdims=True)          # normalized per cell
loglam = np.log(lam_map + 1e-9)                                 # (C, n_bin)
lam_sum = lam_map.sum(axis=0)                                   # (n_bin,) Poisson penalty
prior = occ_time / occ_time.sum()                               # occupancy prior

n_place = len(place_idx)
place_cell_sp = [spikes_for_unit(exc_keys[i]) for i in place_idx]

def weighted_corr(pos_vec, w):
    """Weighted Pearson correlation of time-bin index with decoded position,
    weights = posterior probability mass per bin."""
    k = w > 1e-9
    if k.sum() < 3:
        return 0.0
    t = np.arange(len(pos_vec))[k]
    p = pos_vec[k]
    ww = w[k]
    tw = np.average(t, weights=ww)
    pw = np.average(p, weights=ww)
    cov = np.sum(ww * (t - tw) * (p - pw))
    vart = np.sum(ww * (t - tw) ** 2)
    varp = np.sum(ww * (p - pw) ** 2)
    if vart <= 0 or varp <= 0:
        return 0.0
    return cov / np.sqrt(vart * varp)

def decode_events(rip_ids):
    out = []
    for k in tqdm(rip_ids, desc="replay decode"):
        s0 = ripples[k, 0] / LFP_FS
        e0 = ripples[k, 2] / LFP_FS
        # 20 ms bins inside the ripple
        tbs = np.arange(s0, e0 + 0.5 * BIN_S, BIN_S)
        T = len(tbs) - 1
        if T < 5:
            continue
        counts = np.zeros((n_place, T))
        for c in range(n_place):
            sp = place_cell_sp[c]
            if len(sp) == 0:
                continue
            i = np.clip(np.searchsorted(tbs[1:], sp), 0, T - 1)
            counts[c, i] += 1
        nz_cells = (counts.sum(1) > 0).sum()
        nz_bins = (counts.sum(0) > 0).sum()
        if nz_cells < 5 or nz_bins < 5:
            continue
        # observed posterior
        logp = counts.T @ loglam - BIN_S * lam_sum + np.log(prior)
        logp -= logp.max(axis=1, keepdims=True)
        post = np.exp(logp)
        post /= post.sum(axis=1, keepdims=True)
        pos_obs = post @ bin_c
        w_obs = post.max(axis=1)
        rho_obs = weighted_corr(pos_obs, w_obs)
        # cell-identity shuffle: permute rows of the count matrix (same spike trains,
        # random place-field assignment)
        perms = rng.integers(0, n_place, size=(N_SHUFF, n_place))
        cnts_sh = counts[perms]                       # (S, C, T)
        logp_sh = np.einsum("sct,cb->stb", cnts_sh, loglam)
        logp_sh += (-BIN_S * lam_sum + np.log(prior))[None, None, :]
        logp_sh -= logp_sh.max(axis=2, keepdims=True)
        post_sh = np.exp(logp_sh)
        post_sh /= post_sh.sum(axis=2, keepdims=True)
        pos_sh = post_sh @ bin_c                        # (S, T)
        w_sh = post_sh.max(axis=2)
        rho_sh = np.array([weighted_corr(pos_sh[s], w_sh[s]) for s in range(N_SHUFF)])
        pval = (np.abs(rho_sh) >= np.abs(rho_obs)).mean()
        out.append((k, s0, e0, rho_obs, pval, nz_cells))
    return out

print("decoding PRE...")
res_pre = decode_events(np.where(pre_mask)[0])
print("decoding POST...")
res_post = decode_events(np.where(post_mask)[0])

def summarize(res, label):
    if len(res) == 0:
        print(label, ": no decodable events"); return None
    arr = np.array(res)   # (k, s0, e0, rho, pval, nz)
    sig = arr[arr[:, 4] < 0.05]
    fwd = sig[sig[:, 3] > 0]
    rev = sig[sig[:, 3] < 0]
    print(f"{label}: decodable {len(arr)}, significant {len(sig)} ({100*len(sig)/len(arr):.1f}%), "
          f"forward {len(fwd)}, reverse {len(rev)}")
    return arr

arr_pre = summarize(res_pre, "PRE ")
arr_post = summarize(res_post, "POST")

# statistical comparison: fraction significant, Fisher exact
n1, s1 = len(arr_pre), int((arr_pre[:, 4] < 0.05).sum())
n2, s2 = len(arr_post), int((arr_post[:, 4] < 0.05).sum())
odds, p_fisher = stats.fisher_exact([[s1, n1 - s1], [s2, n2 - s2]])
print(f"Fisher fraction-sig PRE {s1}/{n1} vs POST {s2}/{n2}: p={p_fisher:.3e}")
u, p_mw = stats.mannwhitneyu(np.abs(arr_pre[:, 3]), np.abs(arr_post[:, 3]), alternative="greater")
print(f"|rho| POST > PRE? Mann-Whitney U p={p_mw:.3e}  (medians {np.median(np.abs(arr_pre[:,3])):.3f} vs {np.median(np.abs(arr_post[:,3])):.3f})")

np.savez("scratch/replay_results.npz",
         arr_pre=arr_pre, arr_post=arr_post,
         res_pre=np.array(res_pre), res_post=np.array(res_post))

# =====================================================================
# FIGURE 4b/5: replay examples: pick single ripples with significant
# replay (forward/reverse) and plot decoded posterior vs time
# =====================================================================
def plot_event(ax, k, title):
    s0 = ripples[k, 0] / LFP_FS
    e0 = ripples[k, 2] / LFP_FS
    tbs = np.arange(s0, e0 + 0.5 * BIN_S, BIN_S)
    T = len(tbs) - 1
    counts = np.zeros((n_place, T))
    for c in range(n_place):
        sp = place_cell_sp[c]
        if len(sp) == 0:
            continue
        i = np.clip(np.searchsorted(tbs[1:], sp), 0, T - 1)
        counts[c, i] += 1
    logp = counts.T @ loglam - BIN_S * lam_sum + np.log(prior)
    logp -= logp.max(axis=1, keepdims=True)
    post = np.exp(logp); post /= post.sum(axis=1, keepdims=True)
    im = ax.pcolormesh((tbs[:-1] - s0) * 1000, bin_c, post.T, cmap="inferno",
                       shading="auto", vmin=0, vmax=post.max() * 0.6)
    pos_line = post @ bin_c
    ax.plot((tbs[:-1] - s0) * 1000, pos_line, color="cyan", lw=1.6)
    ax.set_xlabel("time from ripple start (ms)")
    ax.set_ylabel("position (m)")
    ax.set_title(title)
    return im

sig_post = arr_post[arr_post[:, 4] < 0.05]
fwd_post = sig_post[sig_post[:, 3] > 0]
rev_post = sig_post[sig_post[:, 3] < 0]
fwd_sorted = fwd_post[np.argsort(fwd_post[:, 3])[::-1]][:2]
rev_sorted = rev_post[np.argsort(rev_post[:, 3])][:2]

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
im = None
for i, (k, title) in enumerate([(int(fwd_sorted[i,0]), f"POST forward |rho|={fwd_sorted[i,3]:.2f} p={fwd_sorted[i,4]:.3f}") for i in range(2)]):
    im = plot_event(axs[0, i], k, title)
for i, (k, title) in enumerate([(int(rev_sorted[i,0]), f"POST reverse |rho|={abs(rev_sorted[i,3]):.2f} p={rev_sorted[i,4]:.3f}") for i in range(2)]):
    plot_event(axs[1, i], k, title)
if im is not None:
    fig.colorbar(im, ax=axs, label="posterior", shrink=0.8)
fig.suptitle("Decoded replay trajectories during POST ripples")
fig.tight_layout()
fig.savefig("figures/fig5_replay_examples.png"); plt.close(fig)
print("fig5 saved")

# =====================================================================
# FIGURE 6: replay statistics
# =====================================================================
fig, ax = plt.subplots(2, 2, figsize=(9, 8))
# fraction significant
fracs = [s1/n1, s2/n2]
ses = [np.sqrt(f*(1-f)/n) for f, n in zip(fracs, [n1, n2])]
ax[0,0].bar(["PRE", "POST"], fracs, yerr=ses, color=["#7f7f7f", "#1f77b4"], capsize=4)
ax[0,0].set_ylabel("fraction replay events")
ax[0,0].set_title(f"sig fraction: Fisher p={p_fisher:.3f}")
ax[0,0].axhline(0.05, color="r", ls="--", lw=1, label="chance (alpha)")
ax[0,0].legend()
# |rho| distributions
ax[0,1].hist(np.abs(arr_pre[:, 3]), bins=np.linspace(0, 1, 21), alpha=0.6,
             color="#7f7f7f", label="PRE", density=True)
ax[0,1].hist(np.abs(arr_post[:, 3]), bins=np.linspace(0, 1, 21), alpha=0.6,
             color="#1f77b4", label="POST", density=True)
ax[0,1].set_xlabel("|weighted corr| rho"); ax[0,1].set_ylabel("density")
ax[0,1].legend(); ax[0,1].set_title(f"Mann-Whitney p={p_mw:.2g}")
# forward/reverse counts
ax[1,0].bar(["fwd", "rev"], [ (sig_post[:,3]>0).sum(), (sig_post[:,3]<0).sum() ],
            color=["#2ca02c", "#d62728"])
ax[1,0].set_ylabel("POST sig events"); ax[1,0].set_title("direction of POST replay")
# rho vs event center time
tc0 = centers[arr_post[:,0].astype(int)]
ax[1,1].scatter((tc0 - post_start)/3600, arr_post[:, 3], s=4, color="0.5", alpha=0.5)
ax[1,1].axhline(0, color="k", lw=0.7)
ax[1,1].axvline((post_start - post_start)/3600, color="k", ls="--", lw=0.7)
ax[1,1].set_xlabel("POST time (h)"); ax[1,1].set_ylabel("rho")
ax[1,1].set_title("POST replay score over time")
fig.tight_layout(); fig.savefig("figures/fig6_replay_stats.png"); plt.close(fig)
print("fig6 saved")
