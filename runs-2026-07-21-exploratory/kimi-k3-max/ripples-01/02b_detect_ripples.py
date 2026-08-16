# 02b: detect sharp-wave ripples on ch 117 across the full POST epoch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from tqdm import tqdm
from common import open_nwb, load_lfp_channel, get_states, get_epochs, LFP_FS

FS = LFP_FS
PEAK_Z = 4.0       # peak threshold in SD above mean
EDGE_Z = 1.0       # event boundary threshold
MERGE_GAP = 0.030  # s
MIN_DUR = 0.030    # s
MAX_DUR = 0.500    # s
SMOOTH_SIGMA_S = 0.004  # Gaussian smoothing of envelope

nwb, h5 = open_nwb()
states = get_states(nwb)
epochs = get_epochs(nwb)
post = epochs["POSTEpoch"]
post_start, post_end = float(post.start[0]), float(post.end[0])
nrem = states["Non-REM"]
nrem_post = nrem[(nrem.end > post_start) & (nrem.start < post_end)]

best_ch = int(np.load("ripple_channel.npz")["best_ch"])
print(f"ripple channel: {best_ch}")

# --- load the full POST epoch for this channel, in chunks ---
chunk_s = 200.0
edges = np.arange(post_start, post_end + chunk_s, chunk_s)
parts = []
for a, b in tqdm(list(zip(edges[:-1], edges[1:])), desc="loading LFP"):
    t, x = load_lfp_channel(h5, best_ch, a, b)
    parts.append(x)
lfp = np.concatenate(parts)
t_full = np.arange(len(lfp)) / FS + post_start
print("loaded POST LFP:", lfp.shape, f"({len(lfp)/FS:.0f} s)")

# --- ripple-band envelope ---
sos = signal.butter(4, [100, 250], btype="band", fs=FS, output="sos")
lfp_filt = signal.sosfiltfilt(sos, lfp)
env = np.abs(signal.hilbert(lfp_filt))
sigma_samp = SMOOTH_SIGMA_S * FS
env_s = ndimage.gaussian_filter1d(env, sigma_samp)

# --- Non-REM sample mask (in POST-epoch coordinates) ---
nrem_mask = np.zeros(len(lfp), dtype=bool)
for s, e in zip(nrem_post.start, nrem_post.end):
    i0 = max(0, int((s - post_start) * FS))
    i1 = min(len(lfp), int((e - post_start) * FS))
    nrem_mask[i0:i1] = True
print(f"Non-REM samples: {nrem_mask.sum()/FS:.0f} s")

mu = env_s[nrem_mask].mean()
sd = env_s[nrem_mask].std()
thr_peak = mu + PEAK_Z * sd
thr_edge = mu + EDGE_Z * sd
print(f"envelope mean {mu:.2f} uV, sd {sd:.2f}, peak thr {thr_peak:.2f}, edge thr {thr_edge:.2f}")

# --- detection: supra-threshold segments, expanded to edge threshold ---
above = env_s > thr_peak
d = np.diff(above.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if above[0]:
    starts = [0] + starts
if above[-1]:
    ends = ends + [len(above)]

events = []
for s, e in zip(starts, ends):
    # expand left/right to edge threshold
    while s > 0 and env_s[s] > thr_edge:
        s -= 1
    while e < len(env_s) and env_s[e] > thr_edge:
        e += 1
    events.append([s, e])

# merge events separated by < MERGE_GAP
merged = []
for ev in events:
    if merged and (ev[0] - merged[-1][1]) < MERGE_GAP * FS:
        merged[-1][1] = ev[1]
    else:
        merged.append(list(ev))

# duration + Non-REM containment criteria
keep = []
for s, e in merged:
    dur = (e - s) / FS
    if dur < MIN_DUR or dur > MAX_DUR:
        continue
    if not nrem_mask[s:e].all():
        continue
    keep.append((s, e))
print(f"detected {len(keep)} ripple events in POST Non-REM "
      f"({len(keep)/(nrem_mask.sum()/FS)*60:.2f} per min)")

ev_start = np.array([t_full[s] for s, e in keep])
ev_end = np.array([t_full[e] for s, e in keep])
ev_dur = ev_end - ev_start
ev_peak_t = np.array([t_full[s + np.argmax(env_s[s:e])] for s, e in keep])
ev_peak_amp = np.array([env_s[s:e].max() for s, e in keep])
np.savez("ripples_post.npz", start=ev_start, end=ev_end, peak_t=ev_peak_t,
         peak_amp=ev_peak_amp, ch=best_ch, mu=mu, sd=sd,
         thr_peak=thr_peak, thr_edge=thr_edge)

# ---------- figures ----------
# 1) example 4 s segment with detections
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
seg_t0 = 25010.0
seg = (t_full >= seg_t0) & (t_full < seg_t0 + 4)
tt = t_full[seg]
axes[0].plot(tt, lfp[seg], color="gray", lw=0.5)
axes[0].set_ylabel("raw LFP (uV)")
axes[0].set_title(f"Ripple detection on ch {best_ch}, POST Non-REM example")
axes[1].plot(tt, lfp_filt[seg], color="navy", lw=0.7)
axes[1].set_ylabel("100-250 Hz (uV)")
axes[2].plot(tt, env_s[seg], color="darkred", lw=0.8)
axes[2].axhline(thr_peak, color="red", ls="--", lw=1, label=f"peak thr (mean+{PEAK_Z}SD)")
axes[2].axhline(thr_edge, color="orange", ls="--", lw=1, label=f"edge thr (mean+{EDGE_Z}SD)")
axes[2].set_ylabel("envelope (uV)")
axes[2].set_xlabel("time (s)")
axes[2].legend(loc="upper right", fontsize=8)
for ax in axes:
    for s, e in zip(ev_start, ev_end):
        if e > seg_t0 and s < seg_t0 + 4:
            ax.axvspan(s, e, color="red", alpha=0.15)
plt.tight_layout()
plt.savefig("fig_ripple_detection_example.png", dpi=150)

# 2) envelope distribution + thresholds
fig, ax = plt.subplots(figsize=(7, 4))
vals = env_s[nrem_mask][::20]
ax.hist(vals, bins=200, range=(0, np.percentile(vals, 99.9)), color="steelblue", density=True)
ax.axvline(thr_peak, color="red", ls="--", label=f"peak thr = {thr_peak:.1f} uV")
ax.axvline(thr_edge, color="orange", ls="--", label=f"edge thr = {thr_edge:.1f} uV")
ax.set_xlabel("ripple-band envelope (uV)")
ax.set_ylabel("density")
ax.set_title("Envelope distribution over POST Non-REM")
ax.legend()
plt.tight_layout()
plt.savefig("fig_envelope_distribution.png", dpi=150)

# 3) event property distributions
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(ev_dur * 1000, bins=50, color="steelblue")
axes[0].set_xlabel("duration (ms)"); axes[0].set_ylabel("count")
axes[0].set_title(f"Ripple durations (median {np.median(ev_dur)*1000:.0f} ms)")
axes[1].hist(ev_peak_amp, bins=50, color="darkorange")
axes[1].set_xlabel("peak envelope (uV)")
axes[1].set_title(f"Peak amplitudes (median {np.median(ev_peak_amp):.0f} uV)")
# rate over time in 10-min bins
bins = np.arange(post_start, post_end, 600)
counts, _ = np.histogram(ev_start, bins=bins)
axes[2].plot((bins[:-1] - post_start) / 3600, counts / 10, color="seagreen")
axes[2].set_xlabel("hours into POST"); axes[2].set_ylabel("ripples per min")
axes[2].set_title("Ripple rate across POST epoch")
plt.tight_layout()
plt.savefig("fig_ripple_properties.png", dpi=150)

# 4) peri-event averages aligned to envelope peak
win = 0.4
nw = int(win * FS)
segs_filt, segs_raw, segs_env = [], [], []
for pt in ev_peak_t:
    c = int((pt - post_start) * FS)
    if c - nw >= 0 and c + nw < len(lfp):
        segs_filt.append(lfp_filt[c - nw:c + nw])
        segs_raw.append(lfp[c - nw:c + nw])
        segs_env.append(env_s[c - nw:c + nw])
segs_filt = np.array(segs_filt); segs_raw = np.array(segs_raw); segs_env = np.array(segs_env)
taxis = (np.arange(2 * nw) - nw) / FS * 1000
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
m = segs_raw.mean(axis=0); s_ = segs_raw.std(axis=0) / np.sqrt(len(segs_raw))
axes[0].plot(taxis, m, color="gray"); axes[0].fill_between(taxis, m - s_, m + s_, alpha=0.3, color="gray")
axes[0].set_title("Mean raw LFP (sharp wave)"); axes[0].set_ylabel("uV"); axes[0].set_xlabel("ms from peak")
m = segs_filt.mean(axis=0); s_ = segs_filt.std(axis=0) / np.sqrt(len(segs_filt))
axes[1].plot(taxis, m, color="navy"); axes[1].fill_between(taxis, m - s_, m + s_, alpha=0.3, color="navy")
axes[1].set_title("Mean ripple-band LFP"); axes[1].set_xlabel("ms from peak")
m = segs_env.mean(axis=0); s_ = segs_env.std(axis=0) / np.sqrt(len(segs_env))
axes[2].plot(taxis, m, color="darkred"); axes[2].fill_between(taxis, m - s_, m + s_, alpha=0.3, color="darkred")
axes[2].set_title("Mean ripple envelope"); axes[2].set_xlabel("ms from peak")
for ax in axes:
    ax.axvline(0, color="k", ls=":", lw=0.8)
plt.tight_layout()
plt.savefig("fig_ripple_perievent.png", dpi=150)
print("saved ripple figures")
print(f"n events for perievent average: {len(segs_raw)}")
