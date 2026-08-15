"""Load one auditory-nerve fibre from DANDI:001262 and validate every data stream."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import anf_lib as al

ASSET = "5817b204-bf2b-40fb-b852-21072af4b628"   # sub-G190617 fibre 1p-242

fb = al.read_fibre(ASSET)
print("fibre info:", fb["info"])
print("token duration", al.token_duration(fb), "s")
for k, v in al.noise_summary(fb).items():
    print(" ", k, v)

wave = fb["stim"]["NOISE_1"]
coch = al.cochleagram(wave)
cfs = al.band_cfs()
print("cochleagram", coch.shape, "dB range", coch.min(), coch.max())

d = al.build_pynapple(fb, "NOISE_NOISE_4", coch)
print("spikes", len(d["spikes"]), "reps", d["n_reps"], "counts", d["counts"].shape)
psth = d["counts"].mean(0) / al.BIN
print("mean rate", psth.mean(), "Hz; ceiling", al.split_half_ceiling(d["counts"]))

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[0.7, 1.3, 1.4, 0.9], hspace=0.3))
t0, t1 = 0.0, 2.40
tw = np.arange(len(wave)) / al.FS_STIM
m = (tw >= t0) & (tw <= t1)
axes[0].plot(tw[m][::4], wave[m][::4], lw=0.3, color="0.25")
axes[0].set_ylabel("pressure (a.u.)")
axes[0].set_title(f"Frozen broadband noise and the response of ANF {fb['info']['session']} "
                  f"(BF {fb['info']['bf_hz']:.0f} Hz)\n"
                  "the stored token is two 1 s noise bursts, each followed by silence")

mb = (d["tvec"] >= t0) & (d["tvec"] <= t1)
im = axes[1].pcolormesh(d["tvec"][mb], cfs, coch[mb].T, cmap="magma", shading="nearest")
axes[1].set_yscale("log")
axes[1].set_ylabel("band CF (Hz)")
axes[1].axhline(fb["info"]["bf_hz"], color="c", ls="--", lw=1)
cb = fig.colorbar(im, ax=axes[1], pad=0.01, fraction=0.03)
cb.set_label("level (dB re max)", fontsize=8)

for j in range(d["n_reps"]):
    s = fb["spikes"][[i for i, t in enumerate(fb["tags"])
                      if t.startswith("NOISE_NOISE_4_rep")][j]]
    s = s[(s >= t0) & (s <= t1)]
    axes[2].plot(s, np.full_like(s, j), "|", color="k", ms=2.5, mew=0.5)
axes[2].set_ylabel("noise repeat")
axes[2].set_ylim(-1, d["n_reps"])

axes[3].plot(d["tvec"][mb], psth[mb], color="C3", lw=0.7)
axes[3].set_ylabel("PSTH (spikes/s)")
axes[3].set_xlabel("time in token (s)")
axes[3].set_xlim(t0, t1)
fig.savefig("fig01_raw_streams.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ stimulus + response diagnostics
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
f, P = __import__("scipy.signal").signal.welch(wave, fs=al.FS_STIM, nperseg=8192)
axes[0].semilogx(f, 10 * np.log10(P / P.max()), color="0.2")
axes[0].set_xlim(200, 24000)
axes[0].set_ylim(-40, 3)
axes[0].axhline(-3, color="C1", ls=":")
axes[0].set_xlabel("frequency (Hz)")
axes[0].set_ylabel("power (dB re max)")
axes[0].set_title("noise token spectrum")

axes[1].plot(cfs, coch.mean(0), "o-", ms=3)
axes[1].set_xscale("log")
axes[1].set_xlabel("band CF (Hz)")
axes[1].set_ylabel("mean level (dB)")
axes[1].set_title("filterbank drive")

summ = al.noise_summary(fb)
axes[2].plot([1, 2, 3, 4], [summ[t]["rate"] for t in al.NOISE_TOKENS], "o-")
axes[2].set_xticks([1, 2, 3, 4])
axes[2].set_xlabel("noise presentation (increasing level)")
axes[2].set_ylabel("driven rate (spikes/s)")
axes[2].set_title("rate grows with level")

period = al.token_period(coch)
mask = al.analysis_mask(coch)
print("repeating unit", period, "bins; ongoing-noise bins", int(mask.sum()), "of", len(mask))
isi = np.concatenate([np.diff(fb["spikes"][i]) for i, t in enumerate(fb["tags"])
                      if t.startswith("NOISE_NOISE_4_rep")])
axes[3].hist(isi * 1e3, bins=np.arange(0, 20, 0.25), color="0.3")
axes[3].set_xlabel("inter-spike interval (ms)")
axes[3].set_ylabel("count")
axes[3].set_title("refractoriness intact (no ISIs < 0.5 ms)")
fig.tight_layout()
fig.savefig("fig02_stimulus_and_quality.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("min ISI (ms)", isi.min() * 1e3)
print("wrote fig01_raw_streams.png fig02_stimulus_and_quality.png")
