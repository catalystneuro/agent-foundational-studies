"""Stage 3: characterize the detected events as bona fide sharp-wave ripples.

Checks: LFP waveform (sharp wave + ripple), time-frequency signature, event
statistics, the spiking response of pyramidal cells and interneurons, the
state-dependence of ripple rate, and reproducibility on an independent shank.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.signal import welch

import swr_utils as su
from importlib import import_module

d2 = import_module("02_detect_ripples")

SESSION = "Achilles-10252013"
HALF = 0.25  # s around ripple peak


def event_matrix(x, fs, t0, peak_t, half=HALF):
    """Stack signal snippets centred on each event peak."""
    n = int(half * fs)
    idx = np.round((peak_t - t0) * fs).astype(int)
    idx = idx[(idx > n) & (idx < len(x) - n - 1)]
    offs = np.arange(-n, n + 1)
    return x[idx[:, None] + offs], offs / fs


h5 = su.open_session(SESSION)
epochs = su.load_epochs(h5)
states = su.load_states(h5)
units = su.load_units(h5)
_, fs, t0 = su.lfp_meta(h5)

rip = np.load("ripples.npz")
ripples = nap.IntervalSet(start=rip["start"], end=rip["end"])
peak_t, peak_z = rip["peak_t"], rip["peak_z"]
raw = np.load("lfp_raw.npy")
filt = np.load("lfp_filt.npy")
print("%d ripples" % len(peak_t))

# ------------------------------------------------------------------ waveforms
snips_raw, lag = event_matrix(raw, fs, t0, peak_t)
snips_filt, _ = event_matrix(filt, fs, t0, peak_t)
snips_raw = snips_raw - snips_raw[:, :int(0.05 * fs)].mean(axis=1, keepdims=True)

# ------------------------------------------- intra-ripple frequency and duration
dur = (rip["end"] - rip["start"]) * 1000
n_core = int(0.04 * fs)
mid = snips_filt.shape[1] // 2
core = snips_filt[:, mid - n_core: mid + n_core]
fx, px = welch(core, fs=fs, nperseg=core.shape[1], axis=1)
band = (fx >= 100) & (fx <= 300)
peak_freq = fx[band][np.argmax(px[:, band], axis=1)]
print("duration %.1f +- %.1f ms; peak frequency %.1f +- %.1f Hz"
      % (dur.mean(), dur.std(), peak_freq.mean(), peak_freq.std()))

# ------------------------------------------------ time-frequency around ripples
freqs = np.geomspace(30, 400, 60)
sub = np.random.default_rng(0).choice(len(snips_raw), size=min(800, len(snips_raw)), replace=False)
tf = np.zeros((len(freqs), snips_raw.shape[1]))
for i in sub:
    sig = nap.Tsd(t=lag - lag[0] + 1.0, d=snips_raw[i].astype(float))
    w = nap.compute_wavelet_transform(sig, freqs, fs=fs)
    tf += np.abs(np.asarray(w)).T ** 2
tf /= len(sub)

# ------------------------------------------------------- spiking around ripples
pyr = units.getby_category("cell_type")["excitatory"]
inh = units.getby_category("cell_type")["inhibitory"]
peaks = nap.Ts(t=peak_t)
bin_size = 0.005
psth = {}
for name, grp in [("pyramidal", pyr), ("interneuron", inh)]:
    pe = nap.compute_perievent(grp, peaks, window=(-HALF, HALF))
    rates = []
    for k in pe.keys():
        c = pe[k].count(bin_size, nap.IntervalSet(-HALF, HALF))
        rates.append(np.asarray(c).sum(axis=1) / (bin_size * len(peak_t)))
    psth[name] = (np.asarray(c.index), np.array(rates))
    print(name, "n=%d, peak PETH rate %.1f Hz" % (len(grp), np.mean(rates, 0).max()))

# ------------------------------------------------ reproducibility on 2nd shank
best_ch, alt_ch = np.load("channels.npy")
lfp2 = su.load_lfp_channel(h5, int(alt_ch))
filt2 = d2.bandpass(lfp2.values, d2.LOW, d2.HIGH, fs)
env2 = d2.envelope(filt2, fs)
env2_z = nap.Tsd(t=lfp2.index, d=(env2 - env2.mean()) / env2.std())
rip2, peak_t2, _ = d2.detect(env2_z)
match = np.array([np.min(np.abs(peak_t2 - t)) < 0.05 for t in peak_t])
print("channel %d: %d ripples; %.1f%% of channel-%d ripples matched within 50 ms"
      % (alt_ch, len(peak_t2), 100 * match.mean(), best_ch))
np.savez("ripples_ch2.npz", peak_t=peak_t2)

# ================================== figure: waveform / spectro / stats / PETH ==
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
m, s = snips_raw.mean(0), snips_raw.std(0) / np.sqrt(len(snips_raw))
ax.plot(lag * 1000, m, color="k", lw=1.2)
ax.fill_between(lag * 1000, m - s, m + s, color="k", alpha=0.3)
ax.axvline(0, color="C3", ls=":", lw=1)
ax.set(xlabel="time from ripple peak (ms)", ylabel="LFP (a.u.)",
       title="Ripple-triggered average LFP\n(sharp wave)")

ax = fig.add_subplot(gs[0, 1])
ax.plot(lag * 1000, snips_filt.mean(0), color="C3", lw=1)
ax.set(xlabel="time from ripple peak (ms)", ylabel="130-250 Hz (a.u.)",
       title="Ripple-triggered average\nof the filtered band", xlim=(-100, 100))

ax = fig.add_subplot(gs[0, 2])
ax.pcolormesh(lag * 1000, freqs, tf, shading="auto", cmap="magma")
ax.set(yscale="log", xlabel="time from ripple peak (ms)", ylabel="frequency (Hz)",
       title="Mean wavelet power\n(n=%d events)" % len(sub), xlim=(-150, 150))
ax.axhline(130, color="w", ls=":", lw=0.8)
ax.axhline(250, color="w", ls=":", lw=0.8)

ax = fig.add_subplot(gs[1, 0])
ax.hist(dur, bins=40, color="C0")
ax.set(xlabel="duration (ms)", ylabel="count", title="Event duration\nmedian %.0f ms" % np.median(dur))

ax = fig.add_subplot(gs[1, 1])
ax.hist(peak_freq, bins=40, color="C0")
ax.set(xlabel="intra-ripple peak frequency (Hz)", ylabel="count",
       title="Ripple frequency\nmedian %.0f Hz" % np.median(peak_freq))

ax = fig.add_subplot(gs[1, 2])
ax.hist(peak_z, bins=np.arange(4, 25, 0.5), color="C0")
ax.set(xlabel="peak envelope (z)", ylabel="count", title="Ripple amplitude", yscale="log")

ax = fig.add_subplot(gs[2, 0])
for name, color in [("pyramidal", "C0"), ("interneuron", "C1")]:
    tt, r = psth[name]
    mm = r.mean(0)
    se = r.std(0) / np.sqrt(len(r))
    ax.plot(tt * 1000, mm, color=color, label="%s (n=%d)" % (name, len(r)))
    ax.fill_between(tt * 1000, mm - se, mm + se, color=color, alpha=0.3)
ax.axvline(0, color="0.5", ls=":")
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="time from ripple peak (ms)", ylabel="firing rate (Hz)",
       title="Spiking is locked to ripples")

ax = fig.add_subplot(gs[2, 1])
names, rates = [], []
for name, ep in list(epochs.items()) + list(states.items()):
    n = int(((peak_t[:, None] >= ep.start) & (peak_t[:, None] <= ep.end)).any(1).sum())
    names.append(name)
    rates.append(n / ep.tot_length())
ax.bar(names, rates, color=["C0"] * 3 + ["C2"] * 3)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=35, ha="right")
ax.set(ylabel="ripple rate (Hz)", title="Ripples are a non-REM /\nimmobility phenomenon")

ax = fig.add_subplot(gs[2, 2])
ax.hist(peak_t2[np.argmin(np.abs(peak_t2[None, :] - peak_t[:, None]), axis=1)] - peak_t,
        bins=np.arange(-0.05, 0.0501, 0.004), color="C0")
ax.set(xlabel="peak-time difference (s)", ylabel="count",
       title="Same events on shank of ch %d\n(%.0f%% matched)" % (alt_ch, 100 * match.mean()))

fig.suptitle("Sharp-wave ripples in CA1, %s (DANDI:000044), n=%d events" % (SESSION, len(peak_t)))
fig.savefig("fig03_ripple_characterization.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================ figure: example events with spikes
order = np.argsort(peak_z)[::-1]
nrem_post = states["Non-REM"].intersect(epochs["POST"])
in_nrem = ((peak_t[:, None] >= nrem_post.start) & (peak_t[:, None] <= nrem_post.end)).any(1)
sel = [i for i in order if in_nrem[i]][:6]

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
for ax, i in zip(axes.ravel(), sel):
    tc = peak_t[i]
    w = nap.IntervalSet(tc - 0.2, tc + 0.2)
    sl = slice(int((tc - 0.2 - t0) * fs), int((tc + 0.2 - t0) * fs))
    tt = (np.arange(sl.start, sl.stop) / fs + t0 - tc) * 1000
    ax.plot(tt, raw[sl] - raw[sl].mean(), color="k", lw=0.7)
    ax.plot(tt, filt[sl] - 2200, color="C3", lw=0.7)
    sp = pyr.restrict(w)
    for j, k in enumerate(sp.keys()):
        s = (sp[k].index - tc) * 1000
        ax.plot(s, np.full_like(s, -3200 - 22 * j), "|", color="C0", ms=3, mew=0.7)
    ax.axvspan((rip["start"][i] - tc) * 1000, (rip["end"][i] - tc) * 1000, color="C3", alpha=0.12)
    ax.set(title="peak %.1f z, t=%.1f s" % (peak_z[i], tc), yticks=[])
    ax.set_xlabel("time from ripple peak (ms)")
axes[0, 0].set_ylabel("LFP / filtered / spikes")
axes[1, 0].set_ylabel("LFP / filtered / spikes")
fig.suptitle("Example sharp-wave ripples during post-task non-REM sleep (raster: pyramidal cells)")
fig.tight_layout()
fig.savefig("fig04_example_ripples.png", dpi=150)
plt.close(fig)
print("wrote fig03, fig04")

np.savez("ripple_stats.npz", dur=dur, peak_freq=peak_freq, peak_z=peak_z,
         lag=lag, eta_raw=snips_raw.mean(0), eta_filt=snips_filt.mean(0),
         tf=tf, freqs=freqs, psth_t=psth["pyramidal"][0],
         psth_pyr=psth["pyramidal"][1], psth_inh=psth["interneuron"][1],
         match_frac=match.mean())
