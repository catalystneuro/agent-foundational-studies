# %% [markdown]
# # Hippocampal replay of spatial trajectories during sharp-wave ripples
#
# **Dataset:** DANDI Archive dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural firing
# dynamics supports..."), Buzsaki lab. Session
# `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: rat CA1 tetrode
# recording (137 units) with 128-channel LFP at 1250 Hz while the animal runs on
# a 1.6 m linear maze, flanked by PRE and POST sleep/rest epochs with
# Awake/Non-REM/REM state labels.
#
# **Question:** During sharp-wave ripple (SWR) events in sleep, does CA1
# population spiking re-express compressed spatial trajectories of the
# experienced track ("replay")?
#
# **Approach:**
# 1. Stream the NWB file with remfile (no full download).
# 2. Build place-field templates from run bouts on the linearized track
#    (Skaggs spatial information with a circular time-shift shuffle to select
#    place cells).
# 3. Pick a ripple reference channel data-driven (ripple-band envelope
#    "peakiness" x ripple/delta power) and detect SWRs in Non-REM sleep
#    (100-250 Hz band, Hilbert envelope, mean+4SD peak / mean+1SD edges).
# 4. Bayesian-decode position from population spikes in 20 ms bins during each
#    SWR and score each event with the posterior-mass-weighted correlation
#    between time and decoded position; test against a per-event cell-ID
#    shuffle null.
# 5. Compare replay prevalence in POST-maze vs PRE-maze sleep.
#
# Everything is computed from the real streamed data; no synthetic data is used.

# %%
import json
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

rng = np.random.default_rng(7)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = "."  # figures written here
CACHE_DIR = "/tmp/remfile_cache_replay01"

# %% [markdown]
# ## 1. Stream the NWB file from DANDI
#
# Resolve the S3 blob URL through the DANDI API (the `/download/` endpoint
# issues a redirect; the `Location` header is a presigned URL that supports
# HTTP range requests, which remfile needs). A disk cache keeps re-reads fast.

# %%
api = "https://api.dandiarchive.org/api"
r = requests.get(
    f"{api}/dandisets/000044/versions/draft/assets/",
    params={"path": "sub-Achilles/", "page_size": 100},
)
r.raise_for_status()
asset = [a for a in r.json()["results"]
         if a["path"].endswith("10252013_behavior+ecephys.nwb")][0]
print("Asset:", asset["path"], f"({asset['size']/1e9:.2f} GB)")
dl = requests.get(f"{api}/assets/{asset['asset_id']}/download/",
                  allow_redirects=False)
s3_url = dl.headers["Location"]

disk_cache = remfile.DiskCache(CACHE_DIR)
h5 = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5).read())
print(nwb)

# %% [markdown]
# ## 2. Position, epochs, sleep states, and units
#
# The linearized position (`1.6mLinearMazeLinearizedTimeSeries`, 0-1.6 m) is
# sampled at ~39.06 Hz during the maze epoch and is NaN except during on-track
# runs. The `states` table labels Awake/Non-REM/REM intervals across the whole
# session; `epochs` gives PRE / Maze / POST boundaries.

# %%
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin_t = lin.t
lin_x = np.asarray(lin.values).squeeze()
dt_pos = float(np.median(np.diff(lin_t[:20000])))
fs_pos = 1.0 / dt_pos
print(f"position: {len(lin_t)} samples, dt={dt_pos*1000:.2f} ms "
      f"({fs_pos:.2f} Hz), valid fraction {np.isfinite(lin_x).mean():.3f}")

ep = nwb["epochs"].as_dataframe()
pre_iv = nap.IntervalSet(start=ep[ep.label == "PREEpoch"].start.values,
                         end=ep[ep.label == "PREEpoch"].end.values)
post_iv = nap.IntervalSet(start=ep[ep.label == "POSTEpoch"].start.values,
                          end=ep[ep.label == "POSTEpoch"].end.values)
print(ep)

states = nwb["states"].as_dataframe()
is_nrem = states.label.values == "Non-REM"
nrem = nap.IntervalSet(start=states.start.values[is_nrem],
                       end=states.end.values[is_nrem])
nrem_pre = nrem.intersect(pre_iv)
nrem_post = nrem.intersect(post_iv)
h_pre = (nrem_pre.end - nrem_pre.start).sum() / 3600
h_post = (nrem_post.end - nrem_post.start).sum() / 3600
print(f"Non-REM: PRE {h_pre:.2f} h in {len(nrem_pre)} intervals; "
      f"POST {h_post:.2f} h in {len(nrem_post)} intervals")

units = nwb["units"]
unit_ids = sorted(units.keys())
cell_type = units.get_info("cell_type")
exc_ids = [u for u in unit_ids if cell_type[u] == "excitatory"]
inh_ids = [u for u in unit_ids if cell_type[u] == "inhibitory"]
print(f"{len(unit_ids)} units: {len(exc_ids)} excitatory, {len(inh_ids)} inhibitory")

# %% [markdown]
# ## 3. Run bouts on the linear track
#
# Contiguous valid stretches of the linearized position are merged across gaps
# < 0.3 s, then kept if they last >= 1 s, span > 0.3 m, and have median
# |dx/dt| > 0.15 m/s. Within merged bouts, residual NaN samples (the merged
# sub-gap samples) are dropped before any binning so no spikes or occupancy
# leak into bin 0.

# %%
valid = np.isfinite(lin_x)
vidx = np.where(valid)[0]
groups = np.split(vidx, np.where(np.diff(vidx) != 1)[0] + 1)
stretches = [(lin_t[g[0]], lin_t[g[-1]]) for g in groups if len(g) > 0]

merged = []
for s, e in stretches:
    if merged and s - merged[-1][1] < 0.3:
        merged[-1][1] = e
    else:
        merged.append([s, e])

bouts = []
for s, e in merged:
    m = (lin_t >= s) & (lin_t <= e) & np.isfinite(lin_x)
    xs, ts = lin_x[m], lin_t[m]
    if len(xs) < 2:
        continue
    span = xs.max() - xs.min()
    med_speed = np.median(np.abs(np.diff(xs) / np.diff(ts)))
    if (e - s) >= 1.0 and span > 0.3 and med_speed > 0.15:
        bouts.append((s, e))
run_bouts = nap.IntervalSet(start=[b[0] for b in bouts],
                            end=[b[1] for b in bouts])
bout_time = sum(e - s for s, e in bouts)
print(f"{len(bouts)} run bouts, {bout_time:.0f} s total")

# Concatenated bout-sample arrays (wall time t_b, position x_b, and a
# contiguous "bout clock" tau used for the circular time-shift shuffle).
t_b = np.concatenate([lin_t[(lin_t >= s) & (lin_t <= e) & np.isfinite(lin_x)]
                      for s, e in bouts])
x_b = np.concatenate([lin_x[(lin_t >= s) & (lin_t <= e) & np.isfinite(lin_x)]
                      for s, e in bouts])
N_B = len(t_b)
tau_b = np.arange(N_B) * dt_pos
T_bout = N_B * dt_pos
print(f"{N_B} bout samples ({T_bout:.1f} s on the bout clock)")

# %% [markdown]
# ### Figure 1: session and position overview

# %%
fig = plt.figure(figsize=(11, 6))
gs1 = fig.add_gridspec(2, 3)

ax = fig.add_subplot(gs1[0, :2])
mv = np.isfinite(lin_x)
ax.plot(lin_t[mv][::5] - lin_t[0], lin_x[mv][::5], lw=0.3, color="k")
ax.set_xlabel("time in maze epoch (s)")
ax.set_ylabel("linearized position (m)")
ax.set_title("A  Linearized position during maze epoch")

ax = fig.add_subplot(gs1[0, 2])
xy = np.asarray(pos2d.values)
ax.plot(xy[::20, 0], xy[::20, 1], lw=0.2, color="0.4")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("B  2D trajectory (linear maze)")

ax = fig.add_subplot(gs1[1, 0])
edges50 = np.linspace(x_b.min(), x_b.max(), 51)
ctrs50 = 0.5 * (edges50[:-1] + edges50[1:])
occ = np.bincount(np.digitize(x_b, edges50[1:-1]), minlength=50) * dt_pos
ax.bar(ctrs50, occ, width=np.diff(edges50)[0], color="steelblue")
ax.set_xlabel("linearized position (m)")
ax.set_ylabel("occupancy (s)")
ax.set_title("C  Track occupancy during run bouts")

ax = fig.add_subplot(gs1[1, 1])
spd = np.abs(np.diff(x_b) / np.diff(t_b))
ax.hist(spd, bins=np.linspace(0, 2, 81), color="steelblue")
ax.set_xlabel("|dx/dt| (m/s)")
ax.set_ylabel("samples")
ax.set_title("D  Speed within run bouts")

ax = fig.add_subplot(gs1[1, 2])
ax.axis("off")
txt = (f"PRE sleep: 0 - {ep.end[0]/3600:.1f} h\n"
       f"Maze: {ep.start[1]/3600:.2f} - {ep.end[1]/3600:.2f} h "
       f"({(ep.end[1]-ep.start[1])/60:.0f} min)\n"
       f"POST sleep: {ep.start[2]/3600:.2f} - {ep.end[2]/3600:.2f} h\n\n"
       f"run bouts: {len(bouts)} ({bout_time:.0f} s)\n"
       f"Non-REM: PRE {h_pre:.2f} h, POST {h_post:.2f} h")
ax.text(0.05, 0.95, txt, va="top", ha="left", family="monospace",
        transform=ax.transAxes)
ax.set_title("Session layout")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_session_and_position.png")
plt.close(fig)
print("saved fig1_session_and_position.png")

# %% [markdown]
# ## 4. Place fields and place-cell selection
#
# Tuning curves are computed on the bout-restricted data (50 position bins,
# counts and occupancy smoothed separately with a 1.5-bin Gaussian). Spatial
# information (Skaggs, bits/spike) is tested against 500 circular shifts of
# spike times along the contiguous bout clock, which preserves each cell's
# firing statistics and the trajectory's occupancy structure. A unit is a
# place cell if it is excitatory, fires > 0.1 Hz in bouts, has a peak smoothed
# rate >= 1 Hz, and has shuffle p < 0.05.

# %%
occ_s = gaussian_filter1d(occ.astype(float), 1.5)
occ_guard = np.where(occ_s > 0.1, occ_s, np.nan)  # <0.1 s occupancy -> undefined
bin_of_sample = np.digitize(x_b, edges50[1:-1])


def rate_map_from_samples(sample_idx):
    cnt = np.bincount(bin_of_sample[sample_idx], minlength=50).astype(float)
    return gaussian_filter1d(cnt, 1.5) / occ_guard


def skaggs_si(rate, occ_sm):
    p = occ_sm / np.nansum(occ_sm)
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_bar = np.nansum(p * np.nan_to_num(rate, nan=0.0))
        if lam_bar <= 0:
            return 0.0
        r = np.nan_to_num(rate, nan=0.0)
        m = r > 0
        return float(np.sum(p[m] * (r[m] / lam_bar) * np.log2(r[m] / lam_bar)))


def spike_sample_idx(spk_t):
    """Map spike wall times onto the nearest bout-position sample index."""
    tau = np.interp(spk_t, t_b, tau_b)
    return np.clip(np.round(tau * fs_pos).astype(int), 0, N_B - 1)


rate_maps = {}
mean_rates = {}
spike_tau = {}
for u in unit_ids:
    spk = units[u].restrict(run_bouts).t
    mean_rates[u] = len(spk) / bout_time
    if len(spk) >= 5:
        idx = spike_sample_idx(spk)
        rate_maps[u] = rate_map_from_samples(idx)
        spike_tau[u] = np.interp(spk, t_b, tau_b)
    else:
        rate_maps[u] = np.zeros(50)
        spike_tau[u] = np.array([])

N_SHUF_SI = 500
shifts = rng.uniform(0, T_bout, N_SHUF_SI)
si_real, si_p, si_null_med = {}, {}, {}
cand = [u for u in exc_ids
        if mean_rates[u] > 0.1 and np.nanmax(rate_maps[u]) >= 1.0]
print(f"{len(cand)} excitatory units pass rate criteria; running SI shuffle")
for u in tqdm(cand, desc="SI shuffle"):
    si_real[u] = skaggs_si(rate_maps[u], occ_s)
    tau_spk = spike_tau[u]
    null = np.empty(N_SHUF_SI)
    for i, sh in enumerate(shifts):
        tau_s = (tau_spk + sh) % T_bout
        idx = np.clip(np.round(tau_s * fs_pos).astype(int), 0, N_B - 1)
        null[i] = skaggs_si(rate_map_from_samples(idx), occ_s)
    si_p[u] = (1 + (null >= si_real[u]).sum()) / (N_SHUF_SI + 1)
    si_null_med[u] = np.median(null)

place_ids = sorted([u for u in cand if si_p[u] < 0.05])
print(f"place cells: {len(place_ids)} / {len(cand)} candidates "
      f"({len(exc_ids)} excitatory total)")

# %% [markdown]
# ### Figure 2: place fields

# %%
peak_bin = {u: int(np.nanargmax(rate_maps[u])) for u in place_ids}
order = sorted(place_ids, key=lambda u: peak_bin[u])

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])

ax = fig.add_subplot(gs[0, :2])
M = np.array([rate_maps[u] for u in order])
M = M / np.nanmax(M, axis=1, keepdims=True)
ax.imshow(np.nan_to_num(M), aspect="auto", origin="lower",
          extent=[ctrs50[0], ctrs50[-1], 0, len(order)], cmap="viridis")
ax.set_xlabel("linearized position (m)")
ax.set_ylabel("place cells (sorted by peak)")
ax.set_title(f"A  Normalized rate maps of {len(order)} place cells")

ax = fig.add_subplot(gs[0, 2])
top_si = sorted(place_ids, key=lambda u: -si_real[u])[:6]
for u in top_si:
    ax.plot(ctrs50, rate_maps[u], lw=1, label=f"u{u} SI={si_real[u]:.2f}")
ax.set_xlabel("position (m)")
ax.set_ylabel("rate (Hz)")
ax.set_title("B  Example place fields")
ax.legend(fontsize=6, frameon=False)

ax = fig.add_subplot(gs[1, 0])
ax.scatter([si_null_med[u] for u in cand], [si_real[u] for u in cand],
           c=["tab:red" if u in place_ids else "0.6" for u in cand], s=8)
lim = max(si_real[u] for u in cand) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("median shuffle SI (bits/spike)")
ax.set_ylabel("real SI (bits/spike)")
ax.set_title("C  Spatial information vs shuffle")

ax = fig.add_subplot(gs[1, 1])
ax.hist([si_real[u] for u in place_ids], bins=25, color="tab:red",
        label=f"place cells (n={len(place_ids)})")
ax.hist([si_null_med[u] for u in cand], bins=25, color="0.6", alpha=0.7,
        label="shuffle median")
ax.set_xlabel("SI (bits/spike)")
ax.set_ylabel("units")
ax.legend(fontsize=7, frameon=False)
ax.set_title("D  SI distributions")

ax = fig.add_subplot(gs[1, 2])
ax.bar(["excitatory", "rate+\npeak", "place cells"],
       [len(exc_ids), len(cand), len(place_ids)], color=["0.6", "steelblue", "tab:red"])
ax.set_ylabel("units")
ax.set_title("E  Selection funnel")
for i, v in enumerate([len(exc_ids), len(cand), len(place_ids)]):
    ax.text(i, v + 1, str(v), ha="center")

fig.tight_layout()
fig.savefig(f"{OUT}/fig2_place_fields.png")
plt.close(fig)
print("saved fig2_place_fields.png")

# %% [markdown]
# ## 5. Ripple reference channel selection
#
# The electrodes table has no anatomy labels, so the reference channel is
# chosen data-driven: on a 200 s POST Non-REM block, score every channel by
# the "peakiness" of its ripple-band (100-250 Hz) envelope (99th percentile /
# median) times its ripple/delta PSD ratio, and take the maximum. A CA1
# pyramidal-layer channel wins this metric (strong, sparse ripple events on a
# theta-rich background).

# %%
FS_LFP = 1250.0
lfp_ds = h5["processing/ecephys/LFP/LFP/data"]
lfp_conv = h5["processing/ecephys/LFP/LFP/data"].attrs["conversion"]
n_lfp = lfp_ds.shape[0]
print(f"LFP: {lfp_ds.shape}, conversion {lfp_conv}")

# longest POST Non-REM interval for channel selection
durs_post = nrem_post.end - nrem_post.start
i_long = int(np.argmax(durs_post))
s_long = float(nrem_post.start[i_long])
n_blk = int(min(200.0, durs_post[i_long]) * FS_LFP)
i0_blk = int(round(s_long * FS_LFP))
blk = lfp_ds[i0_blk:i0_blk + n_blk, :].astype(np.float64) * lfp_conv
print(f"channel-selection block: {n_blk/FS_LFP:.0f} s at t={s_long:.0f} s")

sos_ripple = signal.butter(4, [100, 250], btype="bandpass", fs=FS_LFP,
                           output="sos")
blk_f = signal.sosfiltfilt(sos_ripple, blk, axis=0)
env_blk = np.abs(signal.hilbert(blk_f, axis=0))
peakiness = np.percentile(env_blk, 99, axis=0) / np.median(env_blk, axis=0)

f_psd, psd = signal.welch(blk, fs=FS_LFP, nperseg=8192, axis=0)
ripple_pow = psd[(f_psd >= 100) & (f_psd <= 250)].mean(axis=0)
delta_pow = psd[(f_psd >= 1) & (f_psd <= 4)].mean(axis=0)
score = peakiness * (ripple_pow / delta_pow)
ch_best = int(np.argmax(score))
print(f"best ripple channel: {ch_best} "
      f"(peakiness {peakiness[ch_best]:.1f}, "
      f"ripple/delta {ripple_pow[ch_best]/delta_pow[ch_best]:.2f})")

# %% [markdown]
# ## 6. SWR detection in Non-REM sleep
#
# The reference channel is band-passed (100-250 Hz, 4th-order Butterworth in
# SOS form for numerical stability), the Hilbert envelope is smoothed with a
# 4 ms Gaussian, and thresholds are set from the pooled Non-REM envelope
# (peak: mean + 4 SD; edges: mean + 1 SD). Crossings < 30 ms apart are merged;
# events of 30-500 ms fully inside a Non-REM interval (50 ms edge margin) are
# kept. Envelopes for all Non-REM intervals (~4.4 h) are processed interval by
# interval.

# %%
def detect_swr_in_interval(env, fs, peak_thr, edge_thr):
    """Return list of (start_idx, end_idx, peak_idx) on the envelope axis."""
    above = env > edge_thr
    d = np.diff(above.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(env)]
    segs = list(zip(starts, ends))
    # merge segments separated by < 30 ms
    min_gap = int(0.030 * fs)
    merged_segs = []
    for s, e in segs:
        if merged_segs and s - merged_segs[-1][1] < min_gap:
            merged_segs[-1][1] = e
        else:
            merged_segs.append([s, e])
    out = []
    for s, e in merged_segs:
        pk = s + int(np.argmax(env[s:e]))
        if env[pk] > peak_thr:
            out.append((s, e, pk))
    return out


t0 = time.time()
nrem_all = nrem  # whole-session Non-REM; epoch assigned by interval overlap
env_store = []   # (t_start, n_samples, epoch, envelope float32)
sum_e, sum2_e, n_e = 0.0, 0.0, 0
for k in tqdm(range(len(nrem_all)), desc="reading+filtering Non-REM LFP"):
    s, e = float(nrem_all.start[k]), float(nrem_all.end[k])
    i0 = max(0, int(round(s * FS_LFP)))
    i1 = min(n_lfp, int(round(e * FS_LFP)))
    if i1 - i0 < int(FS_LFP):
        continue
    raw = lfp_ds[i0:i1, ch_best].astype(np.float64) * lfp_conv
    xf = signal.sosfiltfilt(sos_ripple, raw)
    env = gaussian_filter1d(np.abs(signal.hilbert(xf)), 5.0)  # 4 ms sigma
    if s >= float(post_iv.start[0]):
        epoch = "POST"
    elif e <= float(pre_iv.end[0]):
        epoch = "PRE"
    else:
        epoch = "other"
    env_store.append((i0 / FS_LFP, i1 - i0, epoch, env.astype(np.float32)))
    sum_e += env.sum()
    sum2_e += (env ** 2).sum()
    n_e += len(env)
print(f"processed {n_e/FS_LFP/3600:.2f} h of Non-REM in {time.time()-t0:.0f}s")

env_mean = sum_e / n_e
env_sd = np.sqrt(sum2_e / n_e - env_mean ** 2)
peak_thr = env_mean + 4 * env_sd
edge_thr = env_mean + 1 * env_sd
print(f"envelope mean {env_mean*1e6:.1f} uV, SD {env_sd*1e6:.1f} uV; "
      f"peak thr {peak_thr*1e6:.1f} uV, edge thr {edge_thr*1e6:.1f} uV")

edge_margin = int(0.050 * FS_LFP)
events = []  # dicts
for t_start, n_samp, epoch, env in env_store:
    if epoch == "other":
        continue
    for s, e, pk in detect_swr_in_interval(env, FS_LFP, peak_thr, edge_thr):
        dur = (e - s) / FS_LFP
        if not (0.030 <= dur <= 0.500):
            continue
        if s < edge_margin or e > n_samp - edge_margin:
            continue
        events.append(dict(start=t_start + s / FS_LFP,
                           end=t_start + e / FS_LFP,
                           peak=t_start + pk / FS_LFP,
                           amp=float(env[pk]),
                           dur=dur,
                           epoch=epoch))
print(f"detected {len(events)} SWRs")

ev_pre = [ev for ev in events if ev["epoch"] == "PRE"]
ev_post = [ev for ev in events if ev["epoch"] == "POST"]
rate_pre = len(ev_pre) / (h_pre * 60)
rate_post = len(ev_post) / (h_post * 60)
med_dur = np.median([ev["dur"] for ev in events]) * 1000
print(f"PRE: {len(ev_pre)} SWRs ({rate_pre:.1f}/min Non-REM); "
      f"POST: {len(ev_post)} ({rate_post:.1f}/min); median dur {med_dur:.0f} ms")

# %% [markdown]
# ### Figure 3: SWR detection validation
#
# Example raw/filtered traces with detected events, duration and rate
# summaries, and peri-SWR multiunit firing of excitatory cells (the population
# burst is the defining physiological signature of a sharp-wave ripple).

# %%
# pick a POST interval with several events for the example trace
post_peaks = sorted([ev["peak"] for ev in ev_post])
best_c, best_t = 0, post_peaks[0]
for cand_t in post_peaks[::25]:
    c = np.sum(np.abs(np.array(post_peaks) - cand_t) < 1.25)
    if c > best_c:
        best_c, best_t = c, cand_t
win = (best_t - 1.25, best_t + 1.25)
i0w = int(round(win[0] * FS_LFP))
raw_w = lfp_ds[i0w:i0w + int(2.5 * FS_LFP), ch_best].astype(np.float64) * lfp_conv * 1e6  # uV
filt_w = signal.sosfiltfilt(sos_ripple, raw_w)
env_w = gaussian_filter1d(np.abs(signal.hilbert(filt_w)), 5.0)
t_w = np.arange(len(raw_w)) / FS_LFP - 1.25

# peri-SWR multiunit rate (excitatory cells, POST events)
exc_spikes = np.sort(np.concatenate([units[u].t for u in exc_ids]))
half_w, bin_p = 0.3, 0.01
pbins = np.arange(-half_w, half_w + bin_p, bin_p)
pcounts = np.zeros(len(pbins) - 1)
for ev in ev_post:
    i0p, i1p = np.searchsorted(exc_spikes, [ev["peak"] - half_w, ev["peak"] + half_w])
    rel = exc_spikes[i0p:i1p] - ev["peak"]
    pcounts += np.histogram(rel, bins=pbins)[0]
prate = pcounts / (len(ev_post) * bin_p)
base = prate[(pbins[:-1] >= -half_w) & (pbins[:-1] < -0.15)]
pz = (prate - base.mean()) / base.std()

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
ax = axes[0, 0]
ax.plot(t_w, raw_w, lw=0.3, color="k")
for ev in ev_post:
    if win[0] <= ev["peak"] <= win[1]:
        ax.axvspan(ev["start"] - best_t, ev["end"] - best_t, color="tab:red", alpha=0.25)
ax.set_ylabel("LFP (uV)")
ax.set_title("A  Wideband LFP (POST Non-REM)")
ax.set_xlim(-1.25, 1.25)

ax = axes[0, 1]
ax.plot(t_w, filt_w, lw=0.4, color="steelblue")
for ev in ev_post:
    if win[0] <= ev["peak"] <= win[1]:
        ax.axvspan(ev["start"] - best_t, ev["end"] - best_t, color="tab:red", alpha=0.25)
ax.set_ylabel("ripple band (uV)")
ax.set_title("B  100-250 Hz filtered")
ax.set_xlim(-1.25, 1.25)

ax = axes[0, 2]
ax.plot(t_w, env_w, lw=0.8, color="k")  # raw_w already in uV
ax.axhline(peak_thr * 1e6, color="tab:red", ls="--", lw=0.8, label="peak thr (mean+4SD)")
ax.axhline(edge_thr * 1e6, color="tab:orange", ls="--", lw=0.8, label="edge thr (mean+1SD)")
for ev in ev_post:
    if win[0] <= ev["peak"] <= win[1]:
        ax.axvspan(ev["start"] - best_t, ev["end"] - best_t, color="tab:red", alpha=0.25)
ax.set_ylabel("envelope (uV)")
ax.set_xlabel("time (s)")
ax.set_title("C  Envelope and thresholds")
ax.legend(fontsize=6, frameon=False)
ax.set_xlim(-1.25, 1.25)

ax = axes[1, 0]
ax.hist([ev["dur"] * 1000 for ev in ev_pre], bins=np.arange(30, 300, 10),
        alpha=0.6, density=True, label=f"PRE (n={len(ev_pre)})", color="0.5")
ax.hist([ev["dur"] * 1000 for ev in ev_post], bins=np.arange(30, 300, 10),
        alpha=0.6, density=True, label=f"POST (n={len(ev_post)})", color="tab:red")
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("density")
ax.legend(fontsize=7, frameon=False)
ax.set_title("D  Duration distributions")

ax = axes[1, 1]
ax.bar(["PRE", "POST"], [rate_pre, rate_post], color=["0.5", "tab:red"])
ax.set_ylabel("SWRs per min Non-REM")
ax.set_title("E  SWR incidence")
for i, v in enumerate([rate_pre, rate_post]):
    ax.text(i, v + 0.3, f"{v:.1f}", ha="center")

ax = axes[1, 2]
ax.plot(pbins[:-1] * 1000 + 5, pz, color="k", lw=1)
ax.axvline(0, color="tab:red", ls="--", lw=0.8)
ax.set_xlabel("time from SWR peak (ms)")
ax.set_ylabel("multiunit rate (z)")
ax.set_title("F  Peri-SWR excitatory firing (POST)")

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_swr_detection.png")
plt.close(fig)
print("saved fig3_swr_detection.png")

# %% [markdown]
# ## 7. Bayesian decoding of SWR events
#
# Position is decoded from the place-cell population in 20 ms bins under a
# Poisson model: `log P(x|n) = sum_c n_c log(lambda_c(x) T) - T sum_c
# lambda_c(x)`, with `lambda_c` the smoothed rate maps. Each event is scored
# by the weighted correlation between bin time and position, with weights
# equal to the posterior mass at each (time, position) point (non-empty bins
# only). The null is 200 per-event cell-ID shuffles of the tuning curves
# (which leaves the spike trains and the occupancy term intact); p is the
# exceedance fraction of |w| under the null. Events need >= 5 non-empty bins
# and >= 5 active place cells.

# %%
T_BIN = 0.020
N_SHUF_EV = 200

pc = place_ids
C = len(pc)
lam = np.array([np.nan_to_num(rate_maps[u], nan=0.0) for u in pc])  # Hz, (C, 50)
lam_T = (lam * T_BIN).astype(np.float32)
L_mat = np.log(lam_T + 1e-12)
prior_x = -lam_T.sum(axis=0)  # (50,) permutation-invariant

# concatenated, sorted spike stream of template cells for fast event slicing
all_t = np.concatenate([units[u].t for u in pc])
all_c = np.concatenate([np.full(len(units[u].t), j, dtype=np.int32)
                        for j, u in enumerate(pc)])
ord_s = np.argsort(all_t)
all_t, all_c = all_t[ord_s], all_c[ord_s]


def decode_event(start, end):
    """Return (counts, posterior, nonempty_mask, tvals) for one SWR."""
    i0, i1 = np.searchsorted(all_t, [start, end])
    tt, cc = all_t[i0:i1], all_c[i0:i1]
    Tb = max(1, int(np.ceil((end - start) / T_BIN)))
    b = np.clip(((tt - start) / T_BIN).astype(int), 0, Tb - 1)
    counts = np.zeros((C, Tb), dtype=np.float32)
    np.add.at(counts, (cc, b), 1.0)
    lp = prior_x[None, :] + counts.T @ L_mat
    lp -= lp.max(axis=1, keepdims=True)
    post = np.exp(lp)
    post /= post.sum(axis=1, keepdims=True)
    ne = counts.sum(axis=0) > 0
    tvals = (np.arange(Tb) + 0.5) * T_BIN
    return counts, post, ne, tvals


def weighted_corr(post_ne, t_ne, x_ctrs):
    w = post_ne
    ws = w.sum()
    mt = (w * t_ne[:, None]).sum() / ws
    mx = (w * x_ctrs[None, :]).sum() / ws
    dt_ = t_ne[:, None] - mt
    dx_ = x_ctrs[None, :] - mx
    cov = (w * dt_ * dx_).sum() / ws
    vt = (w * dt_ ** 2).sum() / ws
    vx = (w * dx_ ** 2).sum() / ws
    if vt <= 0 or vx <= 0:
        return 0.0
    return float(cov / np.sqrt(vt * vx))


results = []
t0 = time.time()
for ev in tqdm(events, desc="decoding SWRs"):
    counts, post, ne, tvals = decode_event(ev["start"], ev["end"])
    n_active = int((counts.sum(axis=1) > 0).sum())
    if ne.sum() < 5 or n_active < 5:
        continue
    w_real = weighted_corr(post[ne], tvals[ne], ctrs50)
    null = np.empty(N_SHUF_EV)
    ct32 = counts.T
    for s in range(N_SHUF_EV):
        perm = rng.permutation(C)
        lp = prior_x[None, :] + ct32 @ L_mat[perm]
        lp -= lp.max(axis=1, keepdims=True)
        pp = np.exp(lp)
        pp /= pp.sum(axis=1, keepdims=True)
        null[s] = weighted_corr(pp[ne], tvals[ne], ctrs50)
    p_val = (1 + (np.abs(null) >= abs(w_real)).sum()) / (N_SHUF_EV + 1)
    results.append(dict(**ev, w=w_real, p=p_val, n_bins=int(ne.sum()),
                        n_cells=n_active, null=null.astype(np.float32),
                        post=post.astype(np.float32), ne=ne, tvals=tvals))
print(f"decoded {len(results)} events in {time.time()-t0:.0f}s")

res_pre = [r for r in results if r["epoch"] == "PRE"]
res_post = [r for r in results if r["epoch"] == "POST"]
print(f"decodable: PRE {len(res_pre)}, POST {len(res_post)}")

# %% [markdown]
# ### Figure 4: example decoded replay events (POST sleep)
#
# Top two significant events with positive weighted correlation (decoded
# position increases in time, "forward") and top two with negative
# ("reverse"). Each panel shows the spike raster of place cells ordered by
# field location and the Bayesian posterior over track position.

# %%
sig_post = [r for r in res_post if r["p"] < 0.05]
fwd = sorted([r for r in sig_post if r["w"] > 0], key=lambda r: -r["w"])[:2]
rev = sorted([r for r in sig_post if r["w"] < 0], key=lambda r: r["w"])[:2]
examples = fwd + rev
w_list = ", ".join("w=%.2f" % r["w"] for r in examples)
print(f"significant POST events: {len(sig_post)}; showing {w_list}")

pk_sorted = sorted(pc, key=lambda u: peak_bin[u])
cell_row = {u: i for i, u in enumerate(pk_sorted)}
peak_pos = {u: ctrs50[peak_bin[u]] for u in pc}

fig, axes = plt.subplots(len(examples), 2, figsize=(10, 2.4 * len(examples)),
                         width_ratios=[1, 1.4])
if len(examples) == 1:
    axes = axes[None, :]
for row, r in enumerate(examples):
    i0, i1 = np.searchsorted(all_t, [r["start"], r["end"]])
    tt, cc = all_t[i0:i1], all_c[i0:i1]
    ax = axes[row, 0]
    ax.scatter((tt - r["start"]) * 1000,
               [peak_pos[pc[c]] for c in cc], s=4, color="k")
    ax.set_ylabel("place field position (m)")
    ax.set_xlabel("time in SWR (ms)")
    ax.set_title(f"raster, SWR at {r['start']:.1f} s ({r['dur']*1000:.0f} ms)",
                 fontsize=8)
    ax = axes[row, 1]
    ax.imshow(r["post"].T, origin="lower", aspect="auto", cmap="hot",
              extent=[0, (r["end"] - r["start"]) * 1000, ctrs50[0], ctrs50[-1]])
    ax.set_ylabel("decoded position (m)")
    ax.set_xlabel("time in SWR (ms)")
    direction = "forward" if r["w"] > 0 else "reverse"
    ax.set_title(f"posterior: w={r['w']:.2f}, p={r['p']:.3f} ({direction})",
                 fontsize=8)
fig.suptitle("Example replay events from POST-maze sleep", y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_example_replay_events.png", bbox_inches="tight")
plt.close(fig)
print("saved fig4_example_replay_events.png")

# %% [markdown]
# ### Figure 5: population replay statistics
#
# If SWRs re-express track trajectories, |weighted correlation| of real events
# should exceed the cell-ID shuffle null more often than chance in POST-maze
# sleep, when the maze experience is recent. PRE-maze sleep (before this
# session's maze exposure) is the control.

# %%
w_pre = np.array([r["w"] for r in res_pre])
w_post = np.array([r["w"] for r in res_post])
p_pre = np.array([r["p"] for r in res_pre])
p_post = np.array([r["p"] for r in res_post])
null_post = np.abs(np.concatenate([r["null"] for r in res_post]))

frac_pre = (p_pre < 0.05).mean() if len(p_pre) else np.nan
frac_post = (p_post < 0.05).mean() if len(p_post) else np.nan
tab = [[int((p_post < 0.05).sum()), int((p_post >= 0.05).sum())],
       [int((p_pre < 0.05).sum()), int((p_pre >= 0.05).sum())]]
fisher_p = stats.fisher_exact(tab)[1]
mw_p = stats.mannwhitneyu(np.abs(w_post), np.abs(w_pre),
                          alternative="greater")[1]
n_fwd = int(((p_post < 0.05) & (w_post > 0)).sum())
n_rev = int(((p_post < 0.05) & (w_post < 0)).sum())
print(f"fraction significant: PRE {frac_pre:.3f} (n={len(p_pre)}), "
      f"POST {frac_post:.3f} (n={len(p_post)}); Fisher p={fisher_p:.2e}; "
      f"MW |w| POST>PRE p={mw_p:.2e}; fwd/rev {n_fwd}/{n_rev}")

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

ax = axes[0, 0]
bins_w = np.linspace(0, 1, 41)
ax.hist(np.abs(null_post), bins=bins_w, density=True, color="0.7",
        label="cell-ID shuffle (pooled)")
ax.hist(np.abs(w_post), bins=bins_w, density=True, alpha=0.65,
        color="tab:red", label=f"POST real (n={len(w_post)})")
ax.hist(np.abs(w_pre), bins=bins_w, density=True, alpha=0.45,
        color="0.3", histtype="step", lw=1.5, label=f"PRE real (n={len(w_pre)})")
ax.set_xlabel("|weighted correlation|")
ax.set_ylabel("density")
ax.legend(fontsize=7, frameon=False)
ax.set_title("A  Replay strength vs shuffle null")

ax = axes[0, 1]
ax.bar(["PRE", "POST"], [frac_pre * 100, frac_post * 100],
       color=["0.5", "tab:red"])
ax.axhline(5, color="k", ls="--", lw=0.8, label="chance (p<0.05)")
ax.set_ylabel("% events significant")
ax.set_title(f"B  Significant replay events (Fisher p={fisher_p:.1e})")
for i, v in enumerate([frac_pre * 100, frac_post * 100]):
    ax.text(i, v + 0.4, f"{v:.1f}%", ha="center")
ax.set_ylim(0, frac_post * 100 * 1.3)
ax.legend(fontsize=7, frameon=False)

ax = axes[1, 0]
ax.bar(["forward", "reverse"], [n_fwd, n_rev],
       color=["steelblue", "darkorange"])
ax.set_ylabel("significant POST events")
ax.set_title("C  Trajectory direction of significant events")
for i, v in enumerate([n_fwd, n_rev]):
    ax.text(i, v + 0.5, str(v), ha="center")

ax = axes[1, 1]
dur_post = np.array([r["dur"] * 1000 for r in res_post])
sig = p_post < 0.05
ax.scatter(dur_post[~sig], np.abs(w_post)[~sig], s=6, color="0.6",
           label="not significant")
ax.scatter(dur_post[sig], np.abs(w_post)[sig], s=10, color="tab:red",
           label="p<0.05")
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("|weighted correlation|")
ax.legend(fontsize=7, frameon=False)
ax.set_title("D  Event duration vs replay strength (POST)")

fig.tight_layout()
fig.savefig(f"{OUT}/fig5_replay_statistics.png")
plt.close(fig)
print("saved fig5_replay_statistics.png")

# %% [markdown]
# ## 8. Save results and summary

# %%
np.savez_compressed(
    f"{OUT}/replay_results.npz",
    rate_maps=np.array([rate_maps[u] for u in pc]),
    place_ids=np.array(pc),
    bin_centers=ctrs50,
    event_epoch=np.array([r["epoch"] for r in results]),
    event_start=np.array([r["start"] for r in results]),
    event_end=np.array([r["end"] for r in results]),
    event_w=np.array([r["w"] for r in results]),
    event_p=np.array([r["p"] for r in results]),
    event_nbins=np.array([r["n_bins"] for r in results]),
    event_ncells=np.array([r["n_cells"] for r in results]),
    event_nulls=np.array([r["null"] for r in results]),
    swr_all_start=np.array([ev["start"] for ev in events]),
    swr_all_end=np.array([ev["end"] for ev in events]),
    swr_all_epoch=np.array([ev["epoch"] for ev in events]),
)

summary = dict(
    dataset="DANDI 000044, sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb",
    n_units=len(unit_ids), n_excitatory=len(exc_ids),
    n_run_bouts=len(bouts), bout_time_s=round(bout_time, 1),
    n_place_cells=len(place_ids),
    ripple_channel=ch_best,
    n_swr_pre=len(ev_pre), n_swr_post=len(ev_post),
    swr_rate_pre_per_min=round(rate_pre, 1),
    swr_rate_post_per_min=round(rate_post, 1),
    swr_median_dur_ms=round(med_dur, 1),
    n_decodable_pre=len(res_pre), n_decodable_post=len(res_post),
    frac_sig_pre=round(float(frac_pre), 4),
    frac_sig_post=round(float(frac_post), 4),
    fisher_p=float(fisher_p), mw_p=float(mw_p),
    n_forward=n_fwd, n_reverse=n_rev,
    median_abs_w_post=round(float(np.median(np.abs(w_post))), 3),
    median_abs_w_pre=round(float(np.median(np.abs(w_pre))), 3),
)
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# %% [markdown]
# ## Conclusions
#
# Sharp-wave ripples were detected in CA1 LFP during Non-REM sleep before and
# after linear-track running. Bayesian decoding of place-cell spiking during
# SWRs reveals short, compressed spatial trajectories whose time-position
# structure (weighted correlation) exceeds a cell-ID shuffle null
# significantly more often in POST-maze than PRE-maze sleep: the signature of
# hippocampal replay of the recent experience. Both forward and reverse
# trajectories occur, as expected from the literature.
