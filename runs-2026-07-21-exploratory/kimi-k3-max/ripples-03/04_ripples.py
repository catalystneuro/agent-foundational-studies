"""Sharp-wave ripple detection: channel selection, bandpass + envelope, thresholding."""
import numpy as np
import h5py
import remfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

FS = 1250.0
with open('/tmp/achilles_url.txt') as f:
    bare = f.read().strip().split('?')[0]
disk_cache = remfile.DiskCache('/tmp/remfile_cache_ripples03')
h5py_file = h5py.File(remfile.File(bare, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())

states = nwb['states']
lfp_data = h5py_file['processing/ecephys/LFP/LFP/data']
conv = lfp_data.attrs['conversion']
n_samples = lfp_data.shape[0]
print('LFP samples:', n_samples, 'duration h:', n_samples / FS / 3600)

nrem = states[states.label == 'Non-REM']
post_nrem = nrem[nrem.start > 20147]
pre_nrem = nrem[nrem.end < 18079.5]
print(f'Non-REM intervals: {len(nrem)} total, {len(pre_nrem)} PRE, {len(post_nrem)} POST')
print(f'Non-REM time: PRE {np.sum(pre_nrem.end - pre_nrem.start):.0f} s, '
      f'POST {np.sum(post_nrem.end - post_nrem.start):.0f} s')

# ---------------- channel selection on a POST Non-REM block ----------------
t0, t1 = 21350, 21950  # first POST Non-REM interval is 21350-21399; extend into following block
# use the longest POST Non-REM interval instead
durs = np.asarray(post_nrem.end) - np.asarray(post_nrem.start)
ibest = np.argmax(durs)
t0 = float(np.asarray(post_nrem.start)[ibest])
t1 = float(min(np.asarray(post_nrem.end)[ibest], t0 + 600))
print(f'channel-selection block: {t0:.0f}-{t1:.0f} s ({t1-t0:.0f} s)')
i0, i1 = int(t0 * FS), int(t1 * FS)
block = np.asarray(lfp_data[i0:i1, :], dtype=np.float32) * conv * 1e6  # uV, (T, 128)

sos_ripple = signal.butter(4, [100, 250], btype='bandpass', fs=FS, output='sos')
# stage 1: ripple/delta PSD ratio per channel (cheap, on strided data)
stride = 4
f, pxx = signal.welch(block[::stride, :], fs=FS / stride, nperseg=1024, axis=0)
ripple_pow = pxx[(f >= 100) & (f <= 250)].mean(axis=0)
delta_pow = pxx[(f >= 1) & (f <= 4)].mean(axis=0)
ratio = ripple_pow / delta_pow
top = np.argsort(ratio)[-20:]
# stage 2: envelope peakiness on top-20 channels
scores = {}
for ch in tqdm(top, desc='channel peakiness'):
    filt = signal.sosfiltfilt(sos_ripple, block[:, ch])
    env = np.abs(signal.hilbert(filt))
    scores[ch] = (np.percentile(env, 99) / np.median(env)) * ratio[ch]
best_ch = max(scores, key=scores.get)
print('best ripple channel:', best_ch, 'score:', scores[best_ch])

# ---------------- full-session envelope on best channel ----------------
print('loading full-session channel...')
lfp_ch = np.asarray(lfp_data[:, best_ch], dtype=np.float32) * conv * 1e6  # uV
print('filtering...')
filt = signal.sosfiltfilt(sos_ripple, lfp_ch)
env = np.abs(signal.hilbert(filt))
env_sm = gaussian_filter1d(env, sigma=int(0.004 * FS))  # 4 ms Gaussian

# Non-REM sample mask
mask = np.zeros(n_samples, dtype=bool)
for s, e in zip(np.asarray(nrem.start), np.asarray(nrem.end)):
    mask[int(s * FS):int(e * FS)] = True
env_nrem = env_sm[mask]
mu, sd = env_nrem.mean(), env_nrem.std()
hi = mu + 4 * sd
lo = mu + 1 * sd
print(f'envelope Non-REM mean {mu:.2f} uV, sd {sd:.2f}, hi {hi:.2f}, lo {lo}')

# ---------------- detection ----------------
above = env_sm > lo
d = np.diff(above.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if above[0]:
    starts = [0] + starts
if above[-1]:
    ends = ends + [n_samples]
events = [[s, e] for s, e in zip(starts, ends)]
# keep events with peak > hi
events = [ev for ev in events if env_sm[ev[0]:ev[1]].max() > hi]
# merge events separated by < 30 ms
merged = [events[0]]
for ev in events[1:]:
    if (ev[0] - merged[-1][1]) / FS < 0.030:
        merged[-1][1] = ev[1]
    else:
        merged.append(ev)
# duration 30-500 ms and fully inside Non-REM
ripples = []
for s, e in merged:
    dur = (e - s) / FS
    if not (0.030 <= dur <= 0.500):
        continue
    if not (mask[s] and mask[e - 1]):
        continue
    # fully inside a single Non-REM interval: check no False gap between
    if not mask[s:e].all():
        continue
    peak_idx = s + np.argmax(env_sm[s:e])
    ripples.append(dict(start=s / FS, end=e / FS, peak=peak_idx / FS,
                        dur=dur, peak_amp=env_sm[peak_idx]))
print(f'detected ripples (all Non-REM): {len(ripples)}')

rip_start = np.array([r['start'] for r in ripples])
rip_end = np.array([r['end'] for r in ripples])
rip_peak = np.array([r['peak'] for r in ripples])
rip_dur = np.array([r['dur'] for r in ripples])
is_post = rip_start > 20147
is_pre = rip_end < 18079.5
print(f'PRE ripples: {is_pre.sum()} ({is_pre.sum() / (np.sum(pre_nrem.end-pre_nrem.start)/60):.2f}/min), '
      f'POST ripples: {is_post.sum()} ({is_post.sum() / (np.sum(post_nrem.end-post_nrem.start)/60):.2f}/min), '
      f'median dur {np.median(rip_dur)*1000:.0f} ms')

np.savez('ripples.npz',
         best_ch=best_ch, hi=hi, lo=lo, mu=mu, sd=sd,
         rip_start=rip_start, rip_end=rip_end, rip_peak=rip_peak,
         rip_dur=rip_dur, is_pre=is_pre, is_post=is_post)

# ---------------- fig03: detection examples ----------------
fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw=dict(hspace=0.5))
# (a) 10 s window with several events
post_peaks = rip_peak[is_post]
tp = post_peaks[len(post_peaks) // 3]
w0, w1 = tp - 5, tp + 5
i0, i1 = int(w0 * FS), int(w1 * FS)
tt = np.arange(i0, i1) / FS
ax = axes[0]
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

# (b) zoom on one ripple: raw + filtered
ax = axes[1]
tp = rip_peak[in_win][0] if in_win.any() else post_peaks[0]
w0, w1 = tp - 0.15, tp + 0.15
i0, i1 = int(w0 * FS), int(w1 * FS)
tt = (np.arange(i0, i1) / FS - tp) * 1000  # ms relative to peak
ax.plot(tt, lfp_ch[i0:i1], lw=0.7, color='k', label='raw LFP')
ax.plot(tt, filt[i0:i1], lw=1.0, color='tab:blue', label='100-250 Hz')
ax.plot(tt, env_sm[i0:i1], lw=1.2, color='tab:red', label='envelope')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_title(f'Single ripple zoom (peak at {tp:.2f} s)')
ax.set_xlabel('time from ripple peak (ms)')
ax.set_ylabel('uV')
ax.legend(loc='upper right')

# (c) ripple-triggered averages: raw LFP (sharp wave) + envelope (ripple power)
ax = axes[2]
half = int(0.1 * FS)
avg = np.zeros(2 * half)
avg_env = np.zeros(2 * half)
count = 0
for p in post_peaks[::max(1, len(post_peaks)//800)]:
    ip = int(p * FS)
    if ip - half < 0 or ip + half > n_samples:
        continue
    avg += lfp_ch[ip - half:ip + half]
    avg_env += env_sm[ip - half:ip + half]
    count += 1
avg /= count
avg_env /= count
tt = (np.arange(2 * half) - half) / FS * 1000
ax.plot(tt, avg, lw=1.2, color='k', label='raw LFP (sharp wave)')
ax.set_ylabel('sharp wave (uV)', color='k')
ax.set_xlabel('time from ripple peak (ms)')
ax.set_xlim(-80, 80)
ax2 = ax.twinx()
ax2.plot(tt, avg_env, lw=1.2, color='tab:red', label='ripple-band envelope')
ax2.set_ylabel('envelope (uV)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax.axvline(0, color='gray', ls=':', lw=0.8)
ax.set_title(f'Ripple-triggered averages (n={count} POST ripples)')
lines = [plt.Line2D([], [], color='k'), plt.Line2D([], [], color='tab:red')]
ax.legend(lines, ['raw LFP (sharp wave)', 'ripple-band envelope'], loc='upper right')

fig.savefig('fig03_ripple_detection.png', dpi=150)
print('saved fig03_ripple_detection.png')

# ---------------- fig04: ripple properties ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw=dict(hspace=0.4, wspace=0.3))

ax = axes[0, 0]
ax.hist(rip_dur[is_post] * 1000, bins=np.arange(30, 300, 5), color='tab:blue', alpha=0.7, label='POST')
ax.hist(rip_dur[is_pre] * 1000, bins=np.arange(30, 300, 5), color='tab:gray', alpha=0.5, label='PRE')
ax.set_xlabel('ripple duration (ms)')
ax.set_ylabel('count')
ax.set_title('Ripple duration distribution')
ax.legend()

# rate over time in POST
ax = axes[0, 1]
bins = np.arange(20147, 34861, 300)
ax.hist(rip_peak[is_post], bins=bins, color='tab:blue')
ax.set_xlabel('session time (s)')
ax.set_ylabel('ripples per 5 min')
ax.set_title('POST ripple rate over time')

# ripple-band spectrogram around peak (average)
ax = axes[1, 0]
half = int(0.25 * FS)
nfft = 256
specs = []
sel = post_peaks[::max(1, len(post_peaks)//400)]
for p in tqdm(sel, desc='spectrogram'):
    ip = int(p * FS)
    if ip - half < 0 or ip + half > n_samples:
        continue
    seg = lfp_ch[ip - half:ip + half]
    f, tsp, Sxx = signal.spectrogram(seg, fs=FS, nperseg=128, noverlap=112)
    specs.append(Sxx)
Savg = np.mean(specs, axis=0)
tms = (tsp - 0.25) * 1000
m = (f >= 50) & (f <= 300)
ax.pcolormesh(tms, f[m], 10 * np.log10(Savg[m] + 1e-12), cmap='inferno', shading='auto')
ax.axvline(0, color='cyan', lw=0.8, ls='--')
ax.set_xlabel('time from ripple peak (ms)')
ax.set_ylabel('frequency (Hz)')
ax.set_title('Peri-ripple spectrogram (POST avg)')

# peri-ripple firing rate, exc vs inh
ax = axes[1, 1]
units = nwb['units']
cell_type = units.get_info('cell_type')
binw = 0.01
edges = np.arange(-0.25, 0.25 + binw, binw)
for label, keys, color in [('excitatory (pyramidal)', [k for k in units.keys() if cell_type[k] == 'excitatory'], 'tab:red'),
                           ('inhibitory (interneuron)', [k for k in units.keys() if cell_type[k] == 'inhibitory'], 'tab:blue')]:
    counts = np.zeros(len(edges) - 1)
    nspikes_total = 0
    for k in keys:
        sp = units[k].t
        sp = sp[(sp > 20147)]
        # distance from each spike to nearest ripple peak
        idx = np.searchsorted(post_peaks, sp)
        for j in (idx - 1, idx):
            jj = np.clip(j, 0, len(post_peaks) - 1)
            dt = sp - post_peaks[jj]
            m2 = np.abs(dt) <= 0.25
            counts += np.histogram(dt[m2], bins=edges)[0]
    rate = counts / (len(post_peaks) * binw) / len(keys)
    ax.plot(0.5 * (edges[:-1] + edges[1:]) * 1000, rate, color=color, lw=1.2, label=label)
ax.axvline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('time from ripple peak (ms)')
ax.set_ylabel('rate per cell (Hz)')
ax.set_title('Peri-ripple firing (POST)')
ax.legend()

fig.savefig('fig04_ripple_properties.png', dpi=150)
print('saved fig04_ripple_properties.png')
