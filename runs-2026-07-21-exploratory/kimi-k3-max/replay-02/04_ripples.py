"""Step 4: detect sharp-wave ripples on the best LFP channel during Non-REM.

Recipe: bandpass 100-250 Hz (SOS butter 4), Hilbert envelope, Gaussian smooth
sigma=4 ms. Thresholds from Non-REM envelope samples only: event edges at
mean+1SD, peak must exceed mean+4SD; merge events <30 ms apart; keep 30-500 ms
events fully inside Non-REM intervals.

Saves cache/ripples.npz and figures/fig03_ripple_detection.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

FS = 1250.0
lfp = np.load("cache/lfp_ripple_ch.npy")
b = np.load("cache/behavior.npz")
state_start, state_end, state_label = b["state_start"], b["state_end"], b["state_label"]
epoch_start, epoch_end, epoch_label = b["epoch_start"], b["epoch_end"], b["epoch_label"]
t_lfp = np.arange(len(lfp)) / FS

# Non-REM mask over LFP samples (used for thresholds and event restriction)
nonrem = np.zeros(len(lfp), bool)
for s0, e0 in zip(state_start[state_label == "Non-REM"], state_end[state_label == "Non-REM"]):
    nonrem[int(s0 * FS):int(e0 * FS)] = True
print(f"Non-REM total: {nonrem.sum() / FS:.0f} s")

# --- ripple-band envelope ------------------------------------------------------
sos = signal.butter(4, [100, 250], btype="bandpass", fs=FS, output="sos")
filt = signal.sosfiltfilt(sos, lfp)
env = np.abs(signal.hilbert(filt))
sigma_samp = 0.004 * FS
env = signal.convolve(env, signal.windows.gaussian(int(8 * sigma_samp) + 1, sigma_samp)
                      / (np.sqrt(2 * np.pi) * sigma_samp), mode="same")

mu, sd = env[nonrem].mean(), env[nonrem].std()
lo, hi = mu + 1 * sd, mu + 4 * sd
print(f"envelope thresholds: edge {lo * 1e6:.1f} uV, peak {hi * 1e6:.1f} uV")

# --- event detection -------------------------------------------------------------
above = env > lo
d = np.diff(above.astype(int))
starts = np.where(d == 1)[0] + 1
ends = np.where(d == -1)[0]
if above[0]:
    starts = np.r_[0, starts]
if above[-1]:
    ends = np.r_[ends, len(above) - 1]
# merge events separated by < 30 ms
merged = []
i = 0
while i < len(starts):
    j = i
    while j < len(starts) - 1 and (starts[j + 1] - ends[j]) / FS < 0.03:
        j += 1
    merged.append((starts[i], ends[j]))
    i = j + 1
# keep: peak > hi, duration 30-500 ms, fully inside Non-REM
events = []
for s_i, e_i in merged:
    dur = (e_i - s_i) / FS
    if dur < 0.03 or dur > 0.5:
        continue
    if env[s_i:e_i].max() < hi:
        continue
    if not nonrem[s_i:e_i].all():
        continue
    events.append((s_i / FS, e_i / FS))
events = np.array(events)
print(f"detected {len(events)} SWRs during Non-REM")

post_start = epoch_start[epoch_label == "POSTEpoch"][0]
pre = events[events[:, 1] < epoch_end[epoch_label == "PREEpoch"][0]]
post = events[events[:, 0] >= post_start]
durs = (events[:, 1] - events[:, 0]) * 1000
print(f"PRE: {len(pre)}, POST: {len(post)}, median duration {np.median(durs):.0f} ms")
nonrem_pre = sum(e - s for s, e in zip(state_start[state_label == "Non-REM"],
                                       state_end[state_label == "Non-REM"]) if e < 18079.5)
nonrem_post = sum(e - s for s, e in zip(state_start[state_label == "Non-REM"],
                                        state_end[state_label == "Non-REM"]) if s >= post_start)
print(f"rates: PRE {len(pre) / nonrem_pre * 60:.1f}/min, POST {len(post) / nonrem_post * 60:.1f}/min of Non-REM")

np.savez("cache/ripples.npz", events=events, pre=pre, post=post,
         env_lo=lo, env_hi=hi, env_mean=mu, env_sd=sd)

# --- figure ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
# example Non-REM block in POST with detections
ex_s, ex_e = post[len(post) // 2, 0] - 2, post[len(post) // 2, 0] + 3
m = (t_lfp >= ex_s) & (t_lfp <= ex_e)
tt = t_lfp[m]
ax = axes[0]
ax.plot(tt, lfp[m] * 1e6, lw=0.4, color="gray", label="wideband")
ax.plot(tt, filt[m] * 1e6, lw=0.6, color="tab:blue", label="100-250 Hz")
for s0, e0 in post:
    if e0 < ex_s or s0 > ex_e:
        continue
    ax.axvspan(s0, e0, color="red", alpha=0.25, lw=0)
ax.set_ylabel("uV")
ax.set_title("Example POST Non-REM segment with detected SWRs (red)")
ax.legend(loc="upper right", fontsize=8)
ax = axes[1]
ax.plot(tt, env[m] * 1e6, lw=0.8, color="k")
ax.axhline(lo * 1e6, color="tab:blue", ls="--", lw=1, label="edge (mean+1SD)")
ax.axhline(hi * 1e6, color="tab:red", ls="--", lw=1, label="peak (mean+4SD)")
for s0, e0 in post:
    if e0 < ex_s or s0 > ex_e:
        continue
    ax.axvspan(s0, e0, color="red", alpha=0.25, lw=0)
ax.set_ylabel("envelope (uV)")
ax.set_title("Ripple-band envelope and thresholds")
ax.legend(loc="upper right", fontsize=8)
ax = axes[2]
ax.hist(durs, bins=np.arange(30, 300, 5), color="tab:blue", alpha=0.8)
ax.set_xlabel("SWR duration (ms)")
ax.set_ylabel("count")
ax.set_title(f"SWR duration distribution (n={len(events)}, median {np.median(durs):.0f} ms)")
fig.tight_layout()
fig.savefig("figures/fig03_ripple_detection.png", dpi=150)
plt.close(fig)
print("DONE")
