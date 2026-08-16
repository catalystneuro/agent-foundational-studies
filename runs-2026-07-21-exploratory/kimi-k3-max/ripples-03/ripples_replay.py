# %% [markdown]
# # Sharp-Wave Ripples and Replay in the Hippocampus
#
# This notebook demonstrates two classic hippocampal phenomena using real data
# from the DANDI Archive:
#
# 1. **Sharp-wave ripples (SWRs)**: brief (~30-100 ms), high-frequency
#    (100-250 Hz) oscillations of the CA1 local field potential that occur
#    during non-REM sleep and quiet rest.
# 2. **Replay**: during SWRs after a spatial experience, place cells reactivate
#    in sequences that recapitulate trajectories through the environment on a
#    compressed timescale.
#
# **Dataset**: DANDI dandiset
# [000044](https://dandiarchive.org/dandiset/000044) ("Diversity in neural
# firing dynamics supports both rigid and learned hippocampal sequences",
# Buzsáki lab). Session `sub-Achilles_ses-Achilles-10252013_behavior+ecephys`:
# rat CA1 tetrode recording (137 sorted units, 128-channel LFP at 1250 Hz)
# with a PRE sleep epoch, a ~35 min run on a 1.6 m linear track, and a POST
# sleep epoch. Sleep stages (Awake / Non-REM / REM) are provided.
#
# **Approach**:
# - Compute place fields on the linear track (run bouts only) and identify
#   place cells with a spatial-information shuffle test.
# - Detect SWRs in the POST/PRE Non-REM LFP (bandpass 100-250 Hz, Hilbert
#   envelope, threshold at mean + 4 SD).
# - Bayesian-decode position from place-cell spikes in each SWR (20 ms bins)
#   and score each event with the posterior-mass-weighted correlation between
#   time and decoded position. Significance is assessed against a cell-ID
#   shuffle null (500 shuffles per event), and the fraction of significant
#   events is compared between PRE (baseline) and POST (after track exposure)
#   sleep.

# %% [markdown]
# ## Setup and data loading
#
# The NWB file (8.7 GB) is streamed from the DANDI S3 bucket with `remfile`
# plus a local disk cache, so only the chunks we read are downloaded.

# %%
import numpy as np
import h5py
import requests
import remfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from scipy import stats as sstats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

rng = np.random.default_rng(42)

# Resolve the S3 URL for the Achilles session via the DANDI API
api = ('https://api.dandiarchive.org/api/dandisets/000044/versions/draft/'
       'assets/')
r = requests.get(api, params={
    'path': 'sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb'})
asset = r.json()['results'][0]
r2 = requests.get(
    f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/",
    allow_redirects=False)
s3_url = r2.headers['Location'].split('?')[0]  # bare blob URL: supports range requests
print('streaming:', s3_url)

disk_cache = remfile.DiskCache('/tmp/remfile_cache_ripples03')
h5py_file = h5py.File(remfile.File(s3_url, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
print(nwb)

units = nwb['units']
epochs = nwb['epochs']
states = nwb['states']
pos = nwb['1.6mLinearMazeSpatialSeries']
lin = nwb['1.6mLinearMazeLinearizedTimeSeries']
lfp_data = h5py_file['processing/ecephys/LFP/LFP/data']
conv = lfp_data.attrs['conversion']
FS = 1250.0
n_samples = lfp_data.shape[0]

cell_type = units.get_info('cell_type')
unit_keys = list(units.keys())
exc_keys = [k for k in unit_keys if cell_type[k] == 'excitatory']
print(f'{len(unit_keys)} units ({len(exc_keys)} excitatory, '
      f'{len(unit_keys) - len(exc_keys)} inhibitory)')
print(epochs)

MAZE_START, MAZE_END = 18079.5, 20147.0  # from the epochs table

# %% [markdown]
# ## Session overview
#
# The maze epoch contains ~42 laps on the linear track (the linearized
# position is only valid during fast runs; the rat sits at the reward
# platforms between runs). The LFP shows strong theta (~8 Hz) during running
# and large slow-wave activity with sharp-wave events during Non-REM sleep.

# %%
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
lt = lin.t
lv = np.asarray(lin.values).ravel()
ax.plot(lt[::10], lv[::10], lw=0.4, color='k')
ax.set_title('Linearized position (maze epoch)')
ax.set_xlabel('time (s)')
ax.set_ylabel('track position (m)')
ax.set_xlim(MAZE_START, MAZE_END)

ax = fig.add_subplot(gs[0, 1])
pv = np.asarray(pos.values)
ax.plot(pv[::5, 0], pv[::5, 1], lw=0.2, color='darkblue')
ax.set_title('2D position (linear track)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

ax = fig.add_subplot(gs[1, 0])
ch_demo = 117
i0 = int(19000 * FS)
seg = np.asarray(lfp_data[i0:i0 + int(2 * FS), ch_demo]) * conv * 1e6
ax.plot(np.arange(seg.size) / FS, seg, lw=0.4, color='k')
ax.set_title('LFP ch117 during run (19000-19002 s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('uV')

ax = fig.add_subplot(gs[1, 1])
nrem = states[states.label == 'Non-REM']
post_nrem = nrem[np.asarray(nrem.start) > MAZE_END]
pre_nrem = nrem[np.asarray(nrem.end) < MAZE_START]
t0n = float(np.asarray(post_nrem.start)[0]) + 5
i0 = int(t0n * FS)
seg = np.asarray(lfp_data[i0:i0 + int(2 * FS), ch_demo]) * conv * 1e6
ax.plot(np.arange(seg.size) / FS, seg, lw=0.4, color='k')
ax.set_title(f'LFP ch117 during POST Non-REM ({t0n:.0f}-{t0n+2:.0f} s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('uV')

ax = fig.add_subplot(gs[2, 0])
for label, tstart in [('run (maze)', 19000), ('POST Non-REM', t0n)]:
    i0 = int(tstart * FS)
    seg = np.asarray(lfp_data[i0:i0 + int(60 * FS), ch_demo]) * conv
    f, pxx = signal.welch(seg, fs=FS, nperseg=int(4 * FS))
    ax.semilogy(f, pxx * 1e12, label=label, lw=1)
ax.set_xlim(0, 300)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('PSD (uV^2/Hz)')
ax.set_title('LFP power spectrum (ch117)')
ax.legend()
ax.axvspan(100, 250, color='red', alpha=0.08)

ax = fig.add_subplot(gs[2, 1])
t0r, t1r = 19000, 19010
for i, u in enumerate(unit_keys[:60]):
    sp = units[u].t
    sp = sp[(sp >= t0r) & (sp < t1r)]
    ax.plot(sp - t0r, np.full_like(sp, i), '|', color='k', ms=2)
ax.set_title('Raster, first 60 units (19000-19010 s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('unit index')

fig.savefig('fig01_session_overview.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ## Place fields on the linear track
#
# Run bouts are contiguous stretches of valid linearized position (merged
# across gaps < 0.3 s, >= 1 s long, spanning > 0.3 m, median speed > 0.15
# m/s). Tuning curves are computed in 50 position bins with 1.5-bin Gaussian
# smoothing. A unit is called a place cell if it is excitatory, has peak in-
# field rate >= 1 Hz and mean rate > 0.1 Hz during runs, and its Skaggs
# spatial information exceeds a circular time-shift shuffle null (500 shifts;
# the shift is applied to spike times on the concatenated run-bout time axis
# so shuffled spikes stay on the real trajectory).

# %%
pos_fs = 1.0 / np.median(np.diff(lt))

# --- run bouts ---
valid = np.isfinite(lv)
d = np.diff(valid.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if valid[0]:
    starts = [0] + starts
if valid[-1]:
    ends = ends + [len(valid)]
bouts = [[lt[s], lt[e - 1]] for s, e in zip(starts, ends)]
merged = [bouts[0]]
for b in bouts[1:]:
    if b[0] - merged[-1][1] < 0.3:
        merged[-1][1] = b[1]
    else:
        merged.append(b)
keep = []
for b in merged:
    i0, i1 = np.searchsorted(lt, b[0]), np.searchsorted(lt, b[1])
    seg = lv[i0:i1]
    seg = seg[np.isfinite(seg)]
    dur = b[1] - b[0]
    if dur < 1.0 or len(seg) < 5:
        continue
    if (seg.max() - seg.min()) > 0.3 and np.median(np.abs(np.diff(seg)) * pos_fs) > 0.15:
        keep.append(b)
print(f'run bouts: {len(keep)}, total {sum(b[1]-b[0] for b in keep):.1f} s')

# --- concatenated run-bout ("tau") time axis for the shuffle ---
bout_dur = np.array([b[1] - b[0] for b in keep])
tau_edges = np.concatenate([[0], np.cumsum(bout_dur)])
total_tau = tau_edges[-1]
bout_start_wall = np.array([b[0] for b in keep])

def wall_to_tau(t):
    tau = np.full_like(t, np.nan, dtype=float)
    bi = np.searchsorted(bout_start_wall, t, side='right') - 1
    for j in np.where(bi >= 0)[0]:
        b = bi[j]
        off = t[j] - bout_start_wall[b]
        if 0 <= off <= bout_dur[b]:
            tau[j] = tau_edges[b] + off
    return tau

def tau_to_wall(tau):
    tau = np.mod(tau, total_tau)
    bi = np.clip(np.searchsorted(tau_edges, tau, side='right') - 1, 0, len(bout_dur) - 1)
    return bout_start_wall[bi] + (tau - tau_edges[bi])

fin = np.isfinite(lv)
lt_f, lv_f = lt[fin], lv[fin]
def pos_at(t):
    return np.interp(t, lt_f, lv_f)

# --- tuning curves ---
nbins = 50
edges = np.linspace(0, 1.6, nbins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
pos_tau = wall_to_tau(lt)
in_bout = np.isfinite(pos_tau)
occ, _ = np.histogram(lv[in_bout & fin], bins=edges)
occ_t = occ / pos_fs

spike_pos, spike_tau = {}, {}
for k in unit_keys:
    sp = units[k].t
    tau = wall_to_tau(sp)
    m = np.isfinite(tau)
    spike_tau[k] = tau[m]
    spike_pos[k] = pos_at(sp[m])

tuning = {}
for k in unit_keys:
    cnt, _ = np.histogram(spike_pos[k], bins=edges)
    tuning[k] = gaussian_filter1d(cnt / np.maximum(occ_t, 1e-12), 1.5, mode='nearest')

def skaggs_si(rate, occ_t):
    p = occ_t / occ_t.sum()
    mean_r = (rate * p).sum()
    m = (rate > 0) & (p > 0) & (mean_r > 0)
    if mean_r <= 0:
        return 0.0
    return np.sum(p[m] * (rate[m] / mean_r) * np.log2(rate[m] / mean_r))

# --- SI shuffle ---
nshuf = 500
tau_grid = np.linspace(0, total_tau, 20001)
pos_grid = pos_at(tau_to_wall(tau_grid))
si_real, si_p = {}, {}
for k in tqdm(exc_keys, desc='SI shuffle'):
    si_r = skaggs_si(tuning[k], occ_t)
    si_real[k] = si_r
    tau_sp = spike_tau[k]
    if len(tau_sp) < 10:
        si_p[k] = 1.0
        continue
    cnt_ge = 0
    for s in rng.uniform(0, total_tau, nshuf):
        pos_sh = np.interp(np.mod(tau_sp + s, total_tau), tau_grid, pos_grid)
        cnt, _ = np.histogram(pos_sh, bins=edges)
        rate_sh = gaussian_filter1d(cnt / np.maximum(occ_t, 1e-12), 1.5, mode='nearest')
        if skaggs_si(rate_sh, occ_t) >= si_r:
            cnt_ge += 1
    si_p[k] = (cnt_ge + 1) / (nshuf + 1)

place = {}
for k in exc_keys:
    mean_rate = len(spike_pos[k]) / max(occ_t.sum(), 1e-9)
    place[k] = (tuning[k].max() >= 1.0) and (mean_rate > 0.1) and (si_p[k] < 0.05)
place_keys = [k for k in exc_keys if place[k]]
print(f'place cells: {len(place_keys)} / {len(exc_keys)} excitatory')

# --- figure ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax = axes[0]
order = np.argsort(np.argmax(np.array([tuning[k] for k in place_keys]), axis=1))
T = np.array([tuning[k] for k in place_keys])[order]
ax.imshow(T / T.max(axis=1, keepdims=True), aspect='auto', cmap='viridis',
          extent=[centers[0], centers[-1], len(place_keys), 0])
ax.set_title(f'Place-field template ({len(place_keys)} place cells)')
ax.set_xlabel('track position (m)')
ax.set_ylabel('place cell (sorted by peak)')

ax = axes[1]
peaks = {k: centers[np.argmax(tuning[k])] for k in place_keys}
ex = []
for tpos in np.linspace(0.1, 1.5, 6):
    k = min(place_keys, key=lambda j: abs(peaks[j] - tpos))
    if k not in ex:
        ex.append(k)
off = np.percentile([tuning[k].max() for k in place_keys], 75) * 1.3
for i, k in enumerate(ex):
    ax.plot(centers, tuning[k] + i * off, color=plt.cm.tab10(i % 10), lw=1.4)
ax.set_title('Example place fields (stacked)')
ax.set_xlabel('track position (m)')
ax.set_ylabel('firing rate (Hz, offset)')

ax = axes[2]
ax.scatter([si_real[k] for k in exc_keys], [si_p[k] for k in exc_keys],
           c=['tab:red' if place[k] else 'gray' for k in exc_keys], s=12)
ax.axhline(0.05, color='k', ls='--', lw=0.8)
ax.set_xlabel('spatial information (bits/spike)')
ax.set_ylabel('shuffle p-value')
ax.set_title('Place-cell selection')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('fig02_place_fields.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ## Sharp-wave ripple detection
#
# The detection channel is chosen data-driven: among the 20 channels with the
# highest ripple/delta PSD ratio in a POST Non-REM block, we pick the one with
# the "peaki-est" band-passed envelope (p99/median x ripple/delta ratio). The
# LFP is then band-passed at 100-250 Hz (4th-order Butterworth, SOS form),
# the Hilbert envelope is smoothed with a 4 ms Gaussian, and events are
# detected with a mean+4 SD peak threshold and mean+1 SD boundaries (stats
# computed on Non-REM samples). Events < 30 ms apart are merged; kept events
# are 30-500 ms long and lie fully inside Non-REM intervals.

# %%
# --- channel selection on the longest POST Non-REM block ---
durs = np.asarray(post_nrem.end) - np.asarray(post_nrem.start)
ibest = np.argmax(durs)
t0 = float(np.asarray(post_nrem.start)[ibest])
t1 = float(min(np.asarray(post_nrem.end)[ibest], t0 + 600))
print(f'channel-selection block: {t0:.0f}-{t1:.0f} s')
block = np.asarray(lfp_data[int(t0 * FS):int(t1 * FS), :], dtype=np.float32) * conv * 1e6

sos_ripple = signal.butter(4, [100, 250], btype='bandpass', fs=FS, output='sos')
f, pxx = signal.welch(block[::4, :], fs=FS / 4, nperseg=1024, axis=0)
ratio = pxx[(f >= 100) & (f <= 250)].mean(axis=0) / pxx[(f >= 1) & (f <= 4)].mean(axis=0)
scores = {}
for ch in tqdm(np.argsort(ratio)[-20:], desc='channel peakiness'):
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos_ripple, block[:, ch])))
    scores[ch] = (np.percentile(env, 99) / np.median(env)) * ratio[ch]
best_ch = max(scores, key=scores.get)
print('best ripple channel:', best_ch)

# --- full-session envelope on the best channel ---
print('loading + filtering full-session channel (this takes a few minutes)...')
lfp_ch = np.asarray(lfp_data[:, best_ch], dtype=np.float32) * conv * 1e6
filt = signal.sosfiltfilt(sos_ripple, lfp_ch)
env_sm = gaussian_filter1d(np.abs(signal.hilbert(filt)), sigma=int(0.004 * FS))

mask = np.zeros(n_samples, dtype=bool)
for s, e in zip(np.asarray(nrem.start), np.asarray(nrem.end)):
    mask[int(s * FS):int(e * FS)] = True
mu, sd = env_sm[mask].mean(), env_sm[mask].std()
hi, lo = mu + 4 * sd, mu + 1 * sd
print(f'envelope Non-REM mean {mu:.1f} uV, sd {sd:.1f}; thresholds lo={lo:.1f}, hi={hi:.1f}')

# --- event detection ---
above = env_sm > lo
d = np.diff(above.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if above[0]:
    starts = [0] + starts
if above[-1]:
    ends = ends + [n_samples]
events = [[s, e] for s, e in zip(starts, ends) if env_sm[s:e].max() > hi]
merged = [events[0]]
for ev in events[1:]:
    if (ev[0] - merged[-1][1]) / FS < 0.030:
        merged[-1][1] = ev[1]
    else:
        merged.append(ev)
ripples = []
for s, e in merged:
    dur = (e - s) / FS
    if 0.030 <= dur <= 0.500 and mask[s:e].all():
        pk = s + np.argmax(env_sm[s:e])
        ripples.append((s / FS, e / FS, pk / FS, dur))
rip_start = np.array([r[0] for r in ripples])
rip_end = np.array([r[1] for r in ripples])
rip_peak = np.array([r[2] for r in ripples])
rip_dur = np.array([r[3] for r in ripples])
is_post = rip_start > MAZE_END
is_pre = rip_end < MAZE_START
nrem_pre_s = np.sum(np.asarray(pre_nrem.end) - np.asarray(pre_nrem.start))
nrem_post_s = np.sum(np.asarray(post_nrem.end) - np.asarray(post_nrem.start))
print(f'PRE: {is_pre.sum()} ripples ({is_pre.sum()/(nrem_pre_s/60):.1f}/min Non-REM); '
      f'POST: {is_post.sum()} ({is_post.sum()/(nrem_post_s/60):.1f}/min); '
      f'median duration {np.median(rip_dur)*1000:.0f} ms')

# %% [markdown]
# ### Ripple detection examples
#
# Top: 10 s of POST Non-REM LFP with detected events shaded. Middle: a single
# ripple, showing the fast 100-250 Hz oscillation riding on the slower sharp
# wave. Bottom: the ripple-triggered average shows the sharp-wave deflection
# in the raw LFP together with the ripple-band envelope peak.

# %%
post_peaks = rip_peak[is_post]
fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw=dict(hspace=0.5))

ax = axes[0]
tp = post_peaks[len(post_peaks) // 3]
w0, w1 = tp - 5, tp + 5
i0, i1 = int(w0 * FS), int(w1 * FS)
tt = np.arange(i0, i1) / FS
ax.plot(tt, lfp_ch[i0:i1], lw=0.4, color='k', label='raw LFP')
ax.plot(tt, env_sm[i0:i1] * 10, lw=1.0, color='tab:red', label='ripple envelope x10')
ax.axhline(lo * 10, color='tab:red', ls=':', lw=0.8)
ax.axhline(hi * 10, color='tab:red', ls='--', lw=0.8)
in_win = (rip_start >= w0) & (rip_end <= w1)
for s, e in zip(rip_start[in_win], rip_end[in_win]):
    ax.axvspan(s, e, color='tab:red', alpha=0.15)
ax.set_title(f'Ripple detection on ch{best_ch} (POST Non-REM, {w0:.0f}-{w1:.0f} s)')
ax.set_xlabel('time (s)')
ax.set_ylabel('uV')
ax.legend(loc='upper right')

ax = axes[1]
tp = rip_peak[in_win][0] if in_win.any() else post_peaks[0]
i0, i1 = int((tp - 0.15) * FS), int((tp + 0.15) * FS)
tt = (np.arange(i0, i1) / FS - tp) * 1000
ax.plot(tt, lfp_ch[i0:i1], lw=0.7, color='k', label='raw LFP')
ax.plot(tt, filt[i0:i1], lw=1.0, color='tab:blue', label='100-250 Hz')
ax.plot(tt, env_sm[i0:i1], lw=1.2, color='tab:red', label='envelope')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_title(f'Single ripple zoom (peak at {tp:.2f} s)')
ax.set_xlabel('time from ripple peak (ms)')
ax.set_ylabel('uV')
ax.legend(loc='upper right')

ax = axes[2]
half = int(0.1 * FS)
avg, avg_env, count = np.zeros(2 * half), np.zeros(2 * half), 0
for p in post_peaks[::max(1, len(post_peaks) // 800)]:
    ip = int(p * FS)
    if ip - half < 0 or ip + half > n_samples:
        continue
    avg += lfp_ch[ip - half:ip + half]
    avg_env += env_sm[ip - half:ip + half]
    count += 1
avg /= count
avg_env /= count
tt = (np.arange(2 * half) - half) / FS * 1000
ax.plot(tt, avg, lw=1.2, color='k')
ax.set_ylabel('sharp wave (uV)', color='k')
ax.set_xlabel('time from ripple peak (ms)')
ax.set_xlim(-80, 80)
ax2 = ax.twinx()
ax2.plot(tt, avg_env, lw=1.2, color='tab:red')
ax2.set_ylabel('envelope (uV)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_title(f'Ripple-triggered averages (n={count} POST ripples)')
lines = [plt.Line2D([], [], color='k'), plt.Line2D([], [], color='tab:red')]
ax.legend(lines, ['raw LFP (sharp wave)', 'ripple-band envelope'], loc='upper right')

fig.savefig('fig03_ripple_detection.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ### Ripple properties
#
# Durations peak around 50 ms, events cluster within Non-REM bouts, the peri-
# ripple spectrogram shows the characteristic 100-250 Hz power bump, and both
# pyramidal cells and interneurons increase firing around the ripple peak
# (interneurons with a longer ramp), as expected for SWRs.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw=dict(hspace=0.4, wspace=0.3))

ax = axes[0, 0]
ax.hist(rip_dur[is_post] * 1000, bins=np.arange(30, 300, 5), color='tab:blue',
        alpha=0.7, label='POST')
ax.hist(rip_dur[is_pre] * 1000, bins=np.arange(30, 300, 5), color='tab:gray',
        alpha=0.5, label='PRE')
ax.set_xlabel('ripple duration (ms)')
ax.set_ylabel('count')
ax.set_title('Ripple duration distribution')
ax.legend()

ax = axes[0, 1]
ax.hist(rip_peak[is_post], bins=np.arange(MAZE_END, 34861, 300), color='tab:blue')
ax.set_xlabel('session time (s)')
ax.set_ylabel('ripples per 5 min')
ax.set_title('POST ripple rate over time')

ax = axes[1, 0]
half = int(0.25 * FS)
specs = []
for p in tqdm(post_peaks[::max(1, len(post_peaks) // 400)], desc='spectrogram'):
    ip = int(p * FS)
    if ip - half < 0 or ip + half > n_samples:
        continue
    f, tsp, Sxx = signal.spectrogram(lfp_ch[ip - half:ip + half], fs=FS,
                                     nperseg=128, noverlap=112)
    specs.append(Sxx)
Savg = np.mean(specs, axis=0)
m = (f >= 50) & (f <= 300)
ax.pcolormesh((tsp - 0.25) * 1000, f[m], 10 * np.log10(Savg[m] + 1e-12),
              cmap='inferno', shading='auto')
ax.axvline(0, color='cyan', lw=0.8, ls='--')
ax.set_xlabel('time from ripple peak (ms)')
ax.set_ylabel('frequency (Hz)')
ax.set_title('Peri-ripple spectrogram (POST avg)')

ax = axes[1, 1]
binw = 0.01
pedges = np.arange(-0.25, 0.25 + binw, binw)
for label, keys, color in [('excitatory (pyramidal)', exc_keys, 'tab:red'),
                           ('inhibitory (interneuron)',
                            [k for k in unit_keys if cell_type[k] == 'inhibitory'],
                            'tab:blue')]:
    counts = np.zeros(len(pedges) - 1)
    for k in keys:
        sp = units[k].t
        sp = sp[sp > MAZE_END]
        idx = np.searchsorted(post_peaks, sp)
        for j in (idx - 1, idx):
            jj = np.clip(j, 0, len(post_peaks) - 1)
            dt = sp - post_peaks[jj]
            counts += np.histogram(dt[np.abs(dt) <= 0.25], bins=pedges)[0]
    rate = counts / (len(post_peaks) * binw) / len(keys)
    ax.plot(0.5 * (pedges[:-1] + pedges[1:]) * 1000, rate, color=color, lw=1.2,
            label=label)
ax.axvline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('time from ripple peak (ms)')
ax.set_ylabel('rate per cell (Hz)')
ax.set_title('Peri-ripple firing (POST)')
ax.legend()

fig.savefig('fig04_ripple_properties.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ## Replay during POST-sleep ripples
#
# Each SWR is decoded with a Bayesian decoder using the direction-pooled
# place-field template (88 place cells, 50 position bins) in 20 ms time bins:
#
#     log P(x | spikes in bin t)  ∝  sum_c n_c(t) log f_c(x)  -  dt * sum_c f_c(x)
#
# Events with >= 5 spiked bins and >= 5 active place cells are scored with
# the **weighted correlation**: the correlation between time and position,
# weighted by the posterior mass at each (time, position) pixel. Each event's
# score is compared against 500 cell-ID shuffles of the template (which
# destroy the place-cell/position relationship while preserving spike
# statistics). PRE-sleep SWRs, decoded with the same maze template, serve as
# the baseline.

# %%
key_to_idx = {k: i for i, k in enumerate(unit_keys)}
place_idx = np.array([key_to_idx[k] for k in place_keys])
F = np.array([tuning[k] for k in place_keys])  # (C, P) Hz
C, P = F.shape
logF = np.log(np.maximum(F, 1e-6))
sumF = F.sum(axis=0)
spikes = [units[k].t for k in place_keys]

BIN = 0.020
NSHUF = 500

def decode_event(t0, t1):
    nbins = int(np.ceil((t1 - t0) / BIN))
    if nbins < 5:
        return None, None
    N = np.zeros((nbins, C), dtype=np.float32)
    for c, sp in enumerate(spikes):
        sp_ev = sp[(sp >= t0) & (sp < t1)]
        if sp_ev.size:
            np.add.at(N[:, c], ((sp_ev - t0) / BIN).astype(int), 1)
    return N, t0 + (np.arange(nbins) + 0.5) * BIN

def posterior_from_counts(N, logF_, sumF_):
    logL = N @ logF_ - BIN * sumF_[None, :]
    logL -= logL.max(axis=1, keepdims=True)
    Pp = np.exp(logL)
    return Pp / Pp.sum(axis=1, keepdims=True)

def weighted_corr(post, tbins, xbins):
    Tn = post.shape[0]
    w = post.ravel()
    tt = np.repeat(tbins, P)
    xx = np.tile(xbins, Tn)
    mt, mx = (w * tt).sum(), (w * xx).sum()
    cov = (w * (tt - mt) * (xx - mx)).sum()
    vt = (w * (tt - mt) ** 2).sum()
    vx = (w * (xx - mx) ** 2).sum()
    return 0.0 if vt <= 0 or vx <= 0 else cov / np.sqrt(vt * vx)

def analyze_events(mask, label):
    results = []
    for t0, t1 in tqdm(zip(rip_start[mask], rip_end[mask]),
                       total=mask.sum(), desc=f'{label} decode'):
        N, tbins = decode_event(t0, t1)
        if N is None:
            continue
        if (N.sum(axis=1) > 0).sum() < 5 or (N.sum(axis=0) > 0).sum() < 5:
            continue
        post = posterior_from_counts(N, logF, sumF)
        score = weighted_corr(post, tbins - tbins.mean(), centers)
        sh = np.empty(NSHUF)
        for s in range(NSHUF):
            post_sh = posterior_from_counts(N, logF[rng.permutation(C)], sumF)
            sh[s] = weighted_corr(post_sh, tbins - tbins.mean(), centers)
        p = (np.sum(np.abs(sh) >= np.abs(score)) + 1) / (NSHUF + 1)
        results.append(dict(t0=t0, t1=t1, score=score, p=p))
    return results

res_post = analyze_events(is_post, 'POST')
res_pre = analyze_events(is_pre, 'PRE')
scores_post = np.array([r['score'] for r in res_post])
scores_pre = np.array([r['score'] for r in res_pre])
sig_post = np.array([r['p'] < 0.05 for r in res_post])
sig_pre = np.array([r['p'] < 0.05 for r in res_pre])
print(f'qualified events: POST {len(res_post)}, PRE {len(res_pre)}')
print(f'POST significant: {sig_post.sum()}/{len(res_post)} = {sig_post.mean()*100:.1f}% '
      f'({np.sum(sig_post & (scores_post > 0))} forward, '
      f'{np.sum(sig_post & (scores_post < 0))} reverse)')
print(f'PRE  significant: {sig_pre.sum()}/{len(res_pre)} = {sig_pre.mean()*100:.1f}%')

# %% [markdown]
# ### Example replay events
#
# Rasters (cells sorted by place-field peak) and decoded position posteriors
# for the four most significant POST events. The cyan line is the posterior-
# mass-weighted least-squares trajectory. Note the sequential sweeps across
# the track within ~100 ms.

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw=dict(hspace=0.6, wspace=0.35))
peak_pos = centers[np.argmax(F, axis=1)]
cell_order = np.argsort(peak_pos)
shown = 0
for idx in np.argsort(-np.abs(scores_post)):
    r = res_post[idx]
    if r['p'] >= 0.01:
        continue
    N, tbins = decode_event(r['t0'], r['t1'])
    post = posterior_from_counts(N, logF, sumF)
    axr, axp = axes[0, shown], axes[1, shown]
    for rank, c in enumerate(cell_order):
        sp = spikes[c]
        sp = sp[(sp >= r['t0']) & (sp < r['t1'])]
        axr.plot((sp - r['t0']) * 1000, np.full_like(sp, rank), '|', color='k', ms=3)
    axr.set_title(f"event @{r['t0']:.1f}s  score={r['score']:.2f} p={r['p']:.3f}",
                  fontsize=9)
    axr.set_xlabel('time (ms)')
    axr.set_ylabel('cell (sorted by field)')
    axr.set_xlim(0, (r['t1'] - r['t0']) * 1000)
    axp.imshow(post.T, aspect='auto', origin='lower', cmap='hot_r', vmin=0, vmax=0.3,
               extent=[0, (r['t1'] - r['t0']) * 1000, centers[0], centers[-1]])
    tt = (tbins - r['t0']) * 1000
    w = post.ravel()
    ttall = np.repeat(tt, P)
    xxall = np.tile(centers, post.shape[0])
    mt = (w * ttall).sum() / w.sum()
    mx = (w * xxall).sum() / w.sum()
    b = (w * (ttall - mt) * (xxall - mx)).sum() / (w * (ttall - mt) ** 2).sum()
    a = mx - b * mt
    axp.plot([tt[0], tt[-1]], [a + b * tt[0], a + b * tt[-1]], color='cyan', lw=1.5)
    axp.set_xlabel('time (ms)')
    axp.set_ylabel('position (m)')
    shown += 1
    if shown == 4:
        break
fig.suptitle('Example POST-sleep replay events (raster + decoded posterior)', y=0.98)
fig.savefig('fig05_replay_examples.png', dpi=150)
plt.close(fig)

# %% [markdown]
# ### Replay statistics
#
# POST-sleep events show larger |weighted correlations| than PRE-sleep events
# (Mann-Whitney test), and the fraction of individually significant events
# exceeds the 5% chance level in POST but not PRE sleep (binomial test;
# Fisher exact test for the POST vs PRE comparison).

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw=dict(wspace=0.3))

ax = axes[0]
bins = np.linspace(-0.35, 0.35, 41)
ax.hist(scores_pre, bins=bins, density=True, color='tab:gray', alpha=0.6, label='PRE')
ax.hist(scores_post, bins=bins, density=True, color='tab:blue', alpha=0.7, label='POST')
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('weighted correlation (time vs position)')
ax.set_ylabel('density')
ax.set_title('Replay score distribution')
ax.legend()

ax = axes[1]
binom_p = sstats.binomtest(int(sig_post.sum()), len(res_post), 0.05,
                           alternative='greater').pvalue
fisher_p = sstats.fisher_exact([[int(sig_post.sum()), int((~sig_post).sum())],
                                [int(sig_pre.sum()), int((~sig_pre).sum())]],
                               alternative='greater').pvalue
frac = [sig_pre.mean() * 100, sig_post.mean() * 100]
bars = ax.bar(['PRE sleep', 'POST sleep'], frac, color=['tab:gray', 'tab:blue'])
ax.axhline(5, color='k', ls='--', lw=0.8)
for b_, f_, n in zip(bars, frac, [len(res_pre), len(res_post)]):
    ax.text(b_.get_x() + b_.get_width() / 2, f_ + 0.3, f'{f_:.1f}%\n(n={n})',
            ha='center', fontsize=9)
ax.text(1, frac[1] * 0.45, f'vs chance\np={binom_p:.1e}\nvs PRE\np={fisher_p:.1e}',
        ha='center', fontsize=8, color='white')
nfwd = int(np.sum(sig_post & (scores_post > 0)))
nrev = int(np.sum(sig_post & (scores_post < 0)))
ax.text(0.03, 0.97, f'POST: {nfwd} forward,\n{nrev} reverse', transform=ax.transAxes,
        fontsize=9, va='top')
ax.text(0.97, 0.97, 'dashed line:\nchance (5%)', transform=ax.transAxes, fontsize=8,
        va='top', ha='right', color='k')
ax.set_ylabel('% events significant (p<0.05)')
ax.set_title('Significant replay fraction')
ax.set_ylim(0, max(frac) * 1.35)

ax = axes[2]
abs_post = np.sort(np.abs(scores_post))
ax.plot(abs_post, np.arange(1, len(abs_post) + 1) / len(abs_post),
        color='tab:blue', label='POST |score|')
abs_pre = np.sort(np.abs(scores_pre))
ax.plot(abs_pre, np.arange(1, len(abs_pre) + 1) / len(abs_pre),
        color='tab:gray', label='PRE |score|')
ax.set_xlabel('|weighted correlation|')
ax.set_ylabel('CDF')
ax.set_title('Score magnitude, PRE vs POST')
ax.legend()

fig.savefig('fig06_replay_stats.png', dpi=150)
plt.close(fig)

mw = sstats.mannwhitneyu(np.abs(scores_post), np.abs(scores_pre), alternative='greater')
print(f'Mann-Whitney |score| POST > PRE: p = {mw.pvalue:.2e}')
print(f'binomial POST vs 5% chance: p = {binom_p:.2e}')
print(f'Fisher exact POST vs PRE: p = {fisher_p:.2e}')

# %% [markdown]
# ## Summary
#
# Using the Achilles session from DANDI 000044 (rat CA1, 137 units, 128-ch
# LFP):
#
# - **Place fields**: 88 of 120 excitatory units are significant place cells
#   (spatial-information shuffle, p < 0.05), tiling the 1.6 m track.
# - **Sharp-wave ripples**: detected ~2900 SWRs in POST Non-REM sleep
#   (~33/min, median duration ~50 ms) on the channel with the strongest
#   ripple-band peakiness (ch 117). Events show the classic 100-250 Hz
#   oscillation riding on a sharp wave, with pyramidal-cell and interneuron
#   firing concentrated around the ripple peak.
# - **Replay**: Bayesian decoding of SWR spike content with the place-field
#   template reveals significantly more trajectory-like (high |weighted
#   correlation|) events in POST than PRE sleep. The fraction of
#   shuffle-significant events exceeds chance only in POST sleep, with a
#   forward bias, replicating the classic experience-dependent replay
#   finding.
