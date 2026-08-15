"""Classic reverse correlation (de Boer & de Jongh 1978) at low and high BF.

The STRF above is built from a band-envelope representation of the stimulus, which
throws away the carrier. Averaging the raw pressure waveform preceding each spike
keeps it, and shows directly that low-BF fibres lock to the fine structure of the
noise while high-BF fibres do not.
"""
import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import anf_lib as al

df = pd.read_csv("population_strf.csv")
df = df[df.rate > 30].sort_values("bf_tone")
picks = [df.iloc[0], df.iloc[len(df) // 2], df.iloc[-1]]
print("fibres chosen:", [(p.subject, round(p.bf_tone)) for p in picks])

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
for k, p in enumerate(picks):
    fb = al.read_fibre(p.asset_id)
    tok = p.token
    wave = fb["stim"][tok.split("_", 1)[1]]
    rows = [i for i, t in enumerate(fb["tags"]) if t.startswith(tok + "_rep")]
    sp = np.concatenate([fb["spikes"][i] for i in rows])
    lag_s, sta, n_used = al.revcor(sp, wave)
    fr, mag, fpk = al.revcor_spectrum(sta)

    ax = axes[0, k]
    ax.plot(lag_s * 1e3, sta, color="0.2", lw=0.9)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("lag before spike (ms)")
    if k == 0:
        ax.set_ylabel("mean pressure (a.u.)")
    ax.set_title(f"{p.subject}, BF {p.bf_tone/1000:.1f} kHz\n"
                 f"revcor from {n_used} spikes", fontsize=9)

    ax = axes[1, k]
    ax.plot(fr, mag / mag.max(), color="0.2")
    ax.axvline(p.bf_tone, color="C3", ls="--", label=f"tone BF {p.bf_tone:.0f} Hz")
    ax.axvline(fpk, color="C0", ls=":", label=f"revcor peak {fpk:.0f} Hz")
    ax.axvline(p.strf_bf_hz, color="C2", ls="-.", label=f"STRF BF {p.strf_bf_hz:.0f} Hz")
    ax.set_xscale("log")
    ax.set_xlim(300, 20000)
    ax.set_xlabel("frequency (Hz)")
    if k == 0:
        ax.set_ylabel("normalized magnitude")
    ax.legend(fontsize=7, frameon=False)
    print(f"{p.subject} BF {p.bf_tone:.0f}: revcor peak {fpk:.0f} Hz, "
          f"STRF BF {p.strf_bf_hz:.0f} Hz, revcor amplitude {np.abs(sta).max():.4f}")
fig.suptitle("Spike-triggered average of the pressure waveform: phase locking survives "
             "at low BF, not at high BF", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig10_revcor.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig10_revcor.png")
