"""Prototype the auditory-nerve-fibre frequency x time receptive field (001262)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import strf_lib as S

paths = [
    "sub-G201103/sub-G201103_ses-G201103-1p-6_icephys.nwb",
    "sub-G201210/sub-G201210_ses-G201210-1p-187_icephys.nwb",
    "sub-G151104/sub-G151104_ses-G151104-2P-530nm_icephys.nwb",
    "sub-G220301/sub-G220301_ses-G220301-1p-565_icephys.nwb",
]

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, p in zip(axes, paths):
    fb = S.load_anf_fibre(S.asset_url("001262", p))
    R, t, fq = S.anf_strf(fb, bin_ms=1.0)
    print(p.split("/")[-1], "freqs", len(fq), fq[0], fq[-1],
          "dur", fb["trial_dur"], "bf_pub", fb["bf_published"],
          "sr", fb["sr_published"], "reps", len(fb["spikes"][fq[0]]))
    Rs = gaussian_filter(R, (1.0, 0.6))
    im = ax.pcolormesh(t * 1000, fq / 1000, Rs.T, cmap="magma", shading="auto")
    ax.axhline(fb["bf_published"] / 1000, color="c", ls="--", lw=1)
    ax.set_xlabel("time from tone onset (ms)")
    ax.set_ylabel("tone frequency (kHz)")
    ax.set_title(f"{p.split('/')[0][4:]}  BF(pub)={fb['bf_published']:.0f} Hz", fontsize=9)
    plt.colorbar(im, ax=ax, label="rate (sp/s)")
fig.suptitle("Dandiset 001262 - gerbil auditory-nerve fibres, tone frequency sweep", y=1.02)
fig.tight_layout()
fig.savefig("figures/proto_anf.png", dpi=140, bbox_inches="tight")
