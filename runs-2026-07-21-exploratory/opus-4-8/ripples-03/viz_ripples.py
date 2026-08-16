"""Sharp-wave ripple validation figure."""
import numpy as np
import matplotlib.pyplot as plt
from load_data import load
from ripples import detect_ripples, build_mask, bandpass

d = load()
lfp, lfp_t, fs = d["lfp"], d["lfp_t"], float(d["lfp_fs"])
epl, eps, epe = d["ep_label"], d["ep_start"], d["ep_stop"]
post = (float(eps[epl == "POSTEpoch"][0]), float(epe[epl == "POSTEpoch"][0]))
stl, sts, ste = d["st_label"], d["st_start"], d["st_stop"]
nrem = [(max(s, post[0]), min(e, post[1])) for s, e, l in zip(sts, ste, stl)
        if l == "Non-REM" and e > post[0] and s < post[1]]
mask = build_mask(lfp_t, nrem)
rip = detect_ripples(lfp, lfp_t, fs, mask, high_thr=5.0, low_thr=2.0)
filt, z = rip["filt"], rip["z"]
n = len(rip["start"])
print("N ripples:", n)

# all spikes pooled = multiunit activity
spikes = d["spikes"]
allspk = np.sort(np.concatenate(list(spikes)))

# ---- peri-ripple windows ----
half = int(0.1 * fs)  # +-100 ms
peaks_i = np.searchsorted(lfp_t, rip["peak"])
peaks_i = peaks_i[(peaks_i > half) & (peaks_i < len(lfp) - half)]
raw_snips = np.stack([lfp[i - half:i + half] for i in peaks_i])
filt_snips = np.stack([filt[i - half:i + half] for i in peaks_i])
tt = (np.arange(-half, half) / fs) * 1000

# ---- Morlet ripple-triggered spectrogram ----
freqs = np.linspace(80, 280, 60)
def morlet_power(sig, f, fs, w=6.0):
    t = np.arange(-int(0.05 * fs), int(0.05 * fs) + 1) / fs
    out = np.empty((len(f), len(sig)))
    for k, fi in enumerate(f):
        sigma = w / (2 * np.pi * fi)
        wav = np.exp(2j * np.pi * fi * t) * np.exp(-t**2 / (2 * sigma**2))
        wav /= np.sqrt(np.sum(np.abs(wav)**2))
        out[k] = np.abs(np.convolve(sig, wav, mode="same"))**2
    return out
# average scalogram over a subset of ripples for speed
sub = peaks_i[np.linspace(0, len(peaks_i) - 1, 300).astype(int)]
scal = np.zeros((len(freqs), 2 * half))
for i in sub:
    scal += morlet_power(lfp[i - half:i + half], freqs, fs)
scal /= len(sub)

# ---- peri-ripple MUA ----
bin_w = 0.005
edges = np.arange(-0.1, 0.1 + bin_w, bin_w)
centers = (edges[:-1] + edges[1:]) / 2 * 1000
mua = np.zeros(len(edges) - 1)
for pt in rip["peak"]:
    rel = allspk[(allspk > pt - 0.1) & (allspk < pt + 0.1)] - pt
    mua += np.histogram(rel, edges)[0]
mua_rate = mua / (n * bin_w) / len(spikes)  # per neuron

# ---- example ripple with the strongest peak ----
best = peaks_i[np.argmax(rip["zpeak"][(np.searchsorted(lfp_t, rip["peak"]) > half) &
                                      (np.searchsorted(lfp_t, rip["peak"]) < len(lfp) - half)])]

# ================= figure =================
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.3)

# (A) example ripple raw + filtered
ax = fig.add_subplot(gs[0, :2])
seg = slice(best - half, best + half)
ax.plot(tt, lfp[seg] * 1e6, color="k", lw=0.8, label="wideband LFP")
ax.plot(tt, filt[seg] * 1e6 - 300, color="C3", lw=0.8, label="150-250 Hz")
ax.set(xlabel="time from ripple peak (ms)", ylabel="µV", title="A. Example sharp-wave ripple")
ax.legend(loc="upper right", fontsize=8)

# (B) envelope z-score with thresholds, longer window
ax = fig.add_subplot(gs[0, 2])
w2 = int(0.4 * fs)
tt2 = (np.arange(-w2, w2) / fs) * 1000
ax.plot(tt2, z[best - w2:best + w2], color="C0", lw=0.9)
ax.axhline(5, color="C3", ls="--", lw=0.8, label="5 SD (peak)")
ax.axhline(2, color="grey", ls=":", lw=0.8, label="2 SD (edge)")
ax.set(xlabel="time (ms)", ylabel="envelope (z)", title="B. Ripple-band envelope")
ax.legend(fontsize=7)

# (C) ripple-triggered average LFP
ax = fig.add_subplot(gs[1, 0])
m = raw_snips.mean(0) * 1e6
sem = raw_snips.std(0) / np.sqrt(len(raw_snips)) * 1e6
ax.plot(tt, m, color="k")
ax.fill_between(tt, m - sem, m + sem, color="k", alpha=0.25)
ax.set(xlabel="time from peak (ms)", ylabel="µV",
       title="C. Ripple-triggered avg LFP\n(sharp wave)")

# (D) average scalogram
ax = fig.add_subplot(gs[1, 1])
im = ax.pcolormesh(tt, freqs, scal / scal.mean(0, keepdims=True), cmap="magma", shading="auto")
ax.axhspan(150, 250, color="w", alpha=0.0)
ax.set(xlabel="time from peak (ms)", ylabel="frequency (Hz)",
       title="D. Ripple-triggered\nspectrogram")
fig.colorbar(im, ax=ax, label="norm. power", fraction=0.046)

# (E) peri-ripple MUA
ax = fig.add_subplot(gs[1, 2])
ax.bar(centers, mua_rate, width=5, color="C2", alpha=0.8)
ax.set(xlabel="time from peak (ms)", ylabel="rate (Hz / neuron)",
       title="E. Peri-ripple population firing")

# (F) duration distribution
ax = fig.add_subplot(gs[2, 0])
dur = (rip["stop"] - rip["start"]) * 1000
ax.hist(dur, bins=40, color="C0", alpha=0.8)
ax.axvline(np.median(dur), color="k", ls="--", label=f"median {np.median(dur):.0f} ms")
ax.set(xlabel="ripple duration (ms)", ylabel="count", title="F. Duration distribution")
ax.legend(fontsize=8)

# (G) peak amplitude distribution
ax = fig.add_subplot(gs[2, 1])
ax.hist(rip["zpeak"], bins=40, color="C3", alpha=0.8)
ax.set(xlabel="peak envelope (z)", ylabel="count", title="G. Peak amplitude distribution")

# (H) ripple rate across POST time
ax = fig.add_subplot(gs[2, 2])
tbins = np.arange(post[0], post[1], 60)
rr = np.histogram(rip["peak"], tbins)[0] / 60
ax.plot((tbins[:-1] - post[0]) / 60, rr, color="C4")
ax.set(xlabel="time in POST sleep (min)", ylabel="ripple rate (Hz)",
       title="H. Ripple rate over POST sleep")

fig.suptitle(f"Sharp-wave ripples in CA1 (DANDI 000044, Achilles) — {n} ripples in POST-sleep Non-REM",
             fontsize=13, y=0.98)
fig.savefig("fig_ripples.png", dpi=130, bbox_inches="tight")
print("saved fig_ripples.png")
