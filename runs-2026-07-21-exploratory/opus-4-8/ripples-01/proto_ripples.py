import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import welch
from load_data import load_cache
from ripples import ripple_envelope, detect_ripples, bandpass

d = load_cache()
fs = d["lfp_fs"]
lfp_t = d["lfp_t0"] + np.arange(d["lfp_ch"].size) / fs
lfp = nap.Tsd(t=lfp_t, d=d["lfp_ch"].astype(np.float64))
pre = nap.IntervalSet(*d["epochs"]["PREEpoch"])
maze = nap.IntervalSet(*d["epochs"]["MazeEpoch"])
post = nap.IntervalSet(*d["epochs"]["POSTEpoch"])
print("epochs s:", {k: v for k, v in d["epochs"].items()})

filt, env = ripple_envelope(lfp, fs)
rip_post, pk_t, pk_z, mean, sd = detect_ripples(env, post, fs)
print("POST duration (s):", float(post.tot_length()))
print("N ripples POST:", len(rip_post))
print("ripple rate (Hz):", len(rip_post) / float(post.tot_length()))
durs = (rip_post.end - rip_post.start) * 1e3
print("duration ms: median %.1f  IQR[%.1f-%.1f]" % (
    np.median(durs), np.percentile(durs, 25), np.percentile(durs, 75)))

# peak frequency per ripple (on filtered LFP within +/-50 ms of peak)
pkfreq = []
for pt in pk_t[:400]:
    seg = filt.restrict(nap.IntervalSet(pt - 0.05, pt + 0.05)).values
    f, P = welch(seg, fs=fs, nperseg=min(128, len(seg)))
    band = (f >= 100) & (f <= 300)
    pkfreq.append(f[band][np.argmax(P[band])])
pkfreq = np.array(pkfreq)
print("peak freq Hz: median %.1f" % np.median(pkfreq))

# ---- Figure: example ripples + summary ----
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
# pick 3 strong ripples
order = np.argsort(pk_z)[::-1]
examples = pk_t[order[[0, 5, 12]]]
raw = lfp
for j, pt in enumerate(examples):
    ep = nap.IntervalSet(pt - 0.1, pt + 0.1)
    rr = raw.restrict(ep); ff = filt.restrict(ep)
    tt = (rr.index.values - pt) * 1e3
    axes[0, j].plot(tt, rr.values * 1e3, "k", lw=0.6)
    axes[0, j].set_title("Raw LFP  (ripple @ %.1f s)" % pt, fontsize=9)
    axes[1, j].plot((ff.index.values - pt) * 1e3, ff.values * 1e3, "C3", lw=0.6)
    axes[1, j].set_title("150-250 Hz filtered", fontsize=9)
    for ax in (axes[0, j], axes[1, j]):
        ax.axvspan(-durs[order[[0,5,12][j]]]/2, durs[order[[0,5,12][j]]]/2,
                   color="C1", alpha=0.12)
        ax.set_xlabel("time from peak (ms)")
    axes[0, j].set_ylabel("mV"); axes[1, j].set_ylabel("mV")
# summary row
axes[2, 0].hist(durs, bins=40, color="C0"); axes[2, 0].set_title("Ripple durations")
axes[2, 0].set_xlabel("duration (ms)"); axes[2, 0].set_ylabel("count")
axes[2, 1].hist(pkfreq, bins=30, color="C2"); axes[2, 1].set_title("Peak frequency")
axes[2, 1].set_xlabel("Hz"); axes[2, 1].set_ylabel("count")
axes[2, 2].hist(pk_z, bins=40, color="C3"); axes[2, 2].set_title("Peak envelope (z)")
axes[2, 2].set_xlabel("z-score"); axes[2, 2].set_ylabel("count")
plt.tight_layout()
plt.savefig("fig_ripple_detection.png", dpi=130)
print("saved fig_ripple_detection.png")

np.savez("ripples_post.npz", start=rip_post.start, end=rip_post.end,
         peak_t=pk_t, peak_z=pk_z)
