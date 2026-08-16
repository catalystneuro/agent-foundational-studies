# 02a: pick the LFP channel with the strongest ripple content during POST Non-REM
# Metric: ripple-band RMS + envelope "peakiness" (p99/median) to favor transient
# ripple events over steadily noisy channels. SOS filtering for stability.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from common import open_nwb, load_lfp_block, get_states, get_epochs, LFP_FS

nwb, h5 = open_nwb()
states = get_states(nwb)
epochs = get_epochs(nwb)
post = epochs["POSTEpoch"]
nrem = states["Non-REM"]
post_start, post_end = post.start[0], post.end[0]
nrem_post = nrem[(nrem.end > post_start) & (nrem.start < post_end)]
total_nrem_post = np.sum(nrem_post.end - nrem_post.start)
print(f"total POST Non-REM: {total_nrem_post:.1f} s in {len(nrem_post)} bouts")

durs = nrem_post.end - nrem_post.start
i_best = int(np.argmax(durs))
t0 = nrem_post.start[i_best] + 5
t1 = t0 + 120
print(f"channel-selection window: {t0}-{t1} s (Non-REM bout of {durs[i_best]:.0f} s)")

t, X = load_lfp_block(h5, range(128), t0, t1)

sos_rip = signal.butter(4, [100, 250], btype="band", fs=LFP_FS, output="sos")
sos_del = signal.butter(4, [1, 4], btype="band", fs=LFP_FS, output="sos")
Xr = signal.sosfiltfilt(sos_rip, X, axis=0)
Xd = signal.sosfiltfilt(sos_del, X, axis=0)
rip_rms = np.sqrt(np.mean(Xr**2, axis=0))
del_rms = np.sqrt(np.mean(Xd**2, axis=0))
ratio = rip_rms / del_rms

# envelope peakiness: p99 / median of |hilbert| of ripple band
env = np.abs(signal.hilbert(Xr, axis=0))
peakiness = np.percentile(env, 99, axis=0) / (np.median(env, axis=0) + 1e-9)

# combined score: peakiness weighted by ripple/delta (favor clean ripple channels)
score = peakiness * ratio
best_ch = int(np.argmax(score))
print(f"best channel by peakiness*ripple/delta: {best_ch} "
      f"(peakiness {peakiness[best_ch]:.2f}, ripple RMS {rip_rms[best_ch]:.2f} uV, "
      f"ripple/delta {ratio[best_ch]:.3f})")
top = np.argsort(score)[-8:][::-1]
for ch in top:
    print(f"  ch {ch:3d}: score {score[ch]:6.2f} peakiness {peakiness[ch]:5.2f} "
          f"rip_rms {rip_rms[ch]:6.2f} ratio {ratio[ch]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(128), rip_rms, color="steelblue")
axes[0].axvline(best_ch, color="red", ls="--")
axes[0].set_xlabel("LFP channel"); axes[0].set_ylabel("ripple-band RMS (uV)")
axes[0].set_title("Ripple-band (100-250 Hz) power")
axes[1].bar(range(128), peakiness, color="seagreen")
axes[1].axvline(best_ch, color="red", ls="--")
axes[1].set_xlabel("LFP channel"); axes[1].set_ylabel("envelope p99 / median")
axes[1].set_title("Ripple envelope peakiness")
axes[2].bar(range(128), score, color="darkorange")
axes[2].axvline(best_ch, color="red", ls="--", label=f"ch {best_ch}")
axes[2].set_xlabel("LFP channel"); axes[2].set_ylabel("peakiness x ripple/delta")
axes[2].set_title("Combined channel score")
axes[2].legend()
plt.tight_layout()
plt.savefig("fig_channel_selection.png", dpi=150)

# visual check: 1.5 s of raw + ripple-filtered trace for the top 4 channels
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
seg = (t >= t0 + 30) & (t < t0 + 31.5)
for ax, ch in zip(axes, top[:4]):
    ax.plot(t[seg], X[seg, ch], color="gray", lw=0.5, label="raw")
    ax.plot(t[seg], Xr[seg, ch] * 3, color="navy", lw=0.7, label="ripple band x3")
    ax.set_ylabel(f"ch {ch}\n(uV)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"channel {ch}: peakiness {peakiness[ch]:.2f}, ripple/delta {ratio[ch]:.3f}",
                 fontsize=10)
axes[-1].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("fig_channel_traces.png", dpi=150)
print("saved fig_channel_selection.png and fig_channel_traces.png")

np.savez("ripple_channel.npz", best_ch=best_ch, rip_rms=rip_rms, del_rms=del_rms,
         ratio=ratio, peakiness=peakiness, score=score, sel_window=np.array([t0, t1]))
