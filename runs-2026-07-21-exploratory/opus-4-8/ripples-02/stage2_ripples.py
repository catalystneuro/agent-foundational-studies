"""Stage 2: detect sharp-wave ripples in the CA1 LFP and characterize them."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
import swr_common as C

nwb = C.open_nwb()
epochs = C.get_epochs(nwb)
fs = C.LFP_FS

# --- Load channel-2 LFP across MAZE + POST (awake-immobility + sleep) ---
t_start = epochs["MAZE"][0]
t_stop = epochs["POST"][1]
print(f"Loading LFP {t_start:.0f}-{t_stop:.0f} s ...")
t, v = C.get_lfp_channel(nwb, t0=t_start, t1=t_stop)
print("LFP samples:", v.shape)

# --- Ripple band envelope ---
b, a = butter(4, [r / (fs / 2) for r in C.RIPPLE_BAND], btype="band")
v_rip = filtfilt(b, a, v)
env = np.abs(hilbert(v_rip))
env = gaussian_filter1d(env, sigma=int(0.008 * fs))   # ~8 ms smoothing
z = (env - env.mean()) / env.std()

# --- Threshold-based event detection ---
LOW, HIGH = 2.0, 4.0                 # SD: edges / peak
MIN_DUR, MAX_DUR, MERGE = 0.020, 0.250, 0.030   # seconds
above = z > LOW
edges = np.diff(above.astype(int))
starts = np.where(edges == 1)[0] + 1
stops = np.where(edges == -1)[0] + 1
if above[0]:
    starts = np.r_[0, starts]
if above[-1]:
    stops = np.r_[stops, len(z)]

events = []
for s, e in zip(starts, stops):
    peak = z[s:e].max()
    if peak < HIGH:
        continue
    events.append([s, e, s + int(np.argmax(z[s:e]))])
events = np.array(events, dtype=int)

# merge events separated by < MERGE
merged = []
for ev in events:
    if merged and (ev[0] - merged[-1][1]) / fs < MERGE:
        merged[-1][1] = ev[1]
        if z[ev[2]] > z[merged[-1][2]]:
            merged[-1][2] = ev[2]
    else:
        merged.append(list(ev))
merged = np.array(merged)

# duration filter
dur = (merged[:, 1] - merged[:, 0]) / fs
keep = (dur >= MIN_DUR) & (dur <= MAX_DUR)
merged = merged[keep]
dur = dur[keep]

rip_start = t_start + merged[:, 0] / fs
rip_stop = t_start + merged[:, 1] / fs
rip_peak_t = t_start + merged[:, 2] / fs
rip_peak_z = z[merged[:, 2]]
print(f"Detected {len(merged)} ripples "
      f"({np.mean(rip_peak_t < epochs['MAZE'][1])*len(merged):.0f} in MAZE, rest POST)")

# --- Ripple rate by brain state (POST) ---
states = nwb.processing["behavior"]["states"].to_dataframe()
state_rate = {}
for lab in ["Awake", "Non-REM", "REM"]:
    rows = states[states["label"] == lab]
    tot = 0.0
    cnt = 0
    for _, r in rows.iterrows():
        s = max(r["start_time"], t_start)      # clip to analysis window (MAZE+POST)
        e = min(r["stop_time"], t_stop)
        if e <= s:
            continue
        tot += e - s
        cnt += np.sum((rip_peak_t >= s) & (rip_peak_t < e))
    state_rate[lab] = (cnt / tot if tot > 0 else np.nan, tot, cnt)
    print(f"  {lab}: {cnt} ripples in {tot:.0f}s -> {state_rate[lab][0]*60:.2f}/min")

# --- Peak frequency: FFT of filtered snippets around peaks ---
win = int(0.05 * fs)
peakfreqs = []
freqs = np.fft.rfftfreq(2 * win, 1 / fs)
for pk in merged[:, 2]:
    if pk - win < 0 or pk + win > len(v_rip):
        continue
    seg = v_rip[pk - win: pk + win]
    ps = np.abs(np.fft.rfft(seg)) ** 2
    m = (freqs > 100) & (freqs < 300)
    peakfreqs.append(freqs[m][np.argmax(ps[m])])
peakfreqs = np.array(peakfreqs)

# --- Ripple-triggered LFP + envelope average (POST ripples) ---
w = int(0.1 * fs)
post_pk = merged[rip_peak_t >= epochs["MAZE"][1], 2]
sta_lfp, sta_rip = [], []
for pk in post_pk:
    if pk - w < 0 or pk + w > len(v):
        continue
    sta_lfp.append(v[pk - w: pk + w])
    sta_rip.append(v_rip[pk - w: pk + w])
sta_lfp = np.array(sta_lfp); sta_rip = np.array(sta_rip)
tt = (np.arange(-w, w) / fs) * 1000

# ---------------- Figure ----------------
fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

# Example ripples (broadband + filtered), pick 3 high-power POST ripples
ax = axs[0, 0]
order = np.argsort(rip_peak_z)[::-1]
examples = [i for i in order if rip_peak_t[i] >= epochs["MAZE"][1]][:3]
for j, i in enumerate(examples):
    pk = merged[i, 2]
    seg_t = np.arange(-w, w) / fs
    ax.plot(seg_t * 1000, v[pk - w:pk + w] * 1e3 + j * 1.2, color="k", lw=0.7)
    ax.plot(seg_t * 1000, v_rip[pk - w:pk + w] * 1e3 + j * 1.2 - 0.5,
            color="crimson", lw=0.7)
ax.set_xlabel("Time from peak (ms)"); ax.set_ylabel("LFP (mV, offset)")
ax.set_title("(a) Example POST-sleep ripples")

# Ripple-triggered average
ax = axs[0, 1]
ax.plot(tt, sta_lfp.mean(0) * 1e3, color="k", label="broadband")
ax.plot(tt, sta_rip.mean(0) * 1e3, color="crimson", label="150-250 Hz")
ax.axvline(0, color="gray", ls=":")
ax.set_xlabel("Time from peak (ms)"); ax.set_ylabel("LFP (mV)")
ax.legend(fontsize=8); ax.set_title(f"(b) Ripple-triggered average (n={len(sta_lfp)})")

# Peak-frequency distribution
ax = axs[0, 2]
ax.hist(peakfreqs, bins=np.arange(100, 260, 10), color="#756bb1", edgecolor="w")
ax.axvline(np.median(peakfreqs), color="k", ls="--",
           label=f"median {np.median(peakfreqs):.0f} Hz")
ax.set_xlabel("Peak frequency (Hz)"); ax.set_ylabel("Ripple count")
ax.legend(fontsize=8); ax.set_title("(c) Intra-ripple frequency")

# Duration distribution
ax = axs[1, 0]
ax.hist(dur * 1000, bins=np.arange(20, 210, 15), color="#3182bd", edgecolor="w")
ax.axvline(np.median(dur) * 1000, color="k", ls="--",
           label=f"median {np.median(dur)*1000:.0f} ms")
ax.set_xlabel("Duration (ms)"); ax.set_ylabel("Ripple count")
ax.legend(fontsize=8); ax.set_title("(d) Ripple duration")

# Peak amplitude (z)
ax = axs[1, 1]
ax.hist(rip_peak_z, bins=np.arange(4, 20, 1), color="#e6550d", edgecolor="w")
ax.set_xlabel("Peak envelope (SD)"); ax.set_ylabel("Ripple count")
ax.set_title("(e) Ripple peak amplitude")

# Ripple rate by state
ax = axs[1, 2]
labs = ["Awake", "Non-REM", "REM"]
rates = [state_rate[l][0] * 60 for l in labs]
ax.bar(labs, rates, color=["#fdae6b", "#6baed6", "#74c476"], edgecolor="k")
ax.set_ylabel("Ripple rate (per min)")
ax.set_title("(f) Ripple rate by brain state")
for i, r in enumerate(rates):
    ax.text(i, r, f"{r:.1f}", ha="center", va="bottom")

fig.savefig("fig2_ripples.png", dpi=130)
print("saved fig2_ripples.png")

np.savez("cache_ripples.npz",
         rip_start=rip_start, rip_stop=rip_stop, rip_peak_t=rip_peak_t,
         rip_peak_z=rip_peak_z, dur=dur, peakfreqs=peakfreqs,
         maze=epochs["MAZE"], post=epochs["POST"])
print("saved cache_ripples.npz")
