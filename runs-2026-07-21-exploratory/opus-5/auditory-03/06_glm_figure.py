"""Step 6: visualise the NeMoS GLM fits. Writes figures/06_glm.png"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from common import EXAMPLE_SESSION, FIGDIR, FREQS, FREQ_LABELS, RESDIR
from glm import BIN, WINDOW

os.makedirs(FIGDIR, exist_ok=True)
name = EXAMPLE_SESSION[0]
glm_df = pd.read_csv(f"{RESDIR}/glm_{name}.csv")
npz = np.load(f"{RESDIR}/glm_filters.npz")
fit_ids = list(npz["units"])
kf_all = {u: k for u, k in zip(fit_ids, npz["freq_filters"])}
kh_all = {u: k for u, k in zip(fit_ids, npz["hist_filters"])}

FCOLORS = colormaps["viridis"](np.linspace(0, 0.92, len(FREQS)))
lags_ms = (np.arange(WINDOW) + 1) * BIN * 1e3
match = (glm_df.glm_bf_hz == glm_df.emp_bf_hz).mean()

# One example per preferred frequency, low, middle and high.
examples = []
for f in [FREQS[0], FREQS[2], FREQS[4]]:
    cand = glm_df[glm_df.glm_bf_hz == f].sort_values("pseudo_r2", ascending=False)
    if len(cand):
        examples.append(int(cand.iloc[0].unit))
examples = examples or [int(u) for u in fit_ids[:3]]

fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)
panel = "abc"

for c, u in enumerate(examples):
    ax = fig.add_subplot(gs[0, c])
    for fi in range(len(FREQS)):
        ax.plot(lags_ms, kf_all[u][fi], color=FCOLORS[fi], lw=1.4,
                label=f"{FREQ_LABELS[fi]} kHz")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("time since tone onset (ms)")
    ax.set_ylabel("filter gain (log rate)")
    r = glm_df[glm_df.unit == u].iloc[0]
    ax.set_title(f"({panel[c]}) unit {u}, GLM BF = {r.glm_bf_hz / 1e3:.0f} kHz\n"
                 f"pseudo-$R^2$ = {r.pseudo_r2:.3f}", fontsize=11)
    if c == 0:
        ax.legend(fontsize=8, ncol=2)

ax = fig.add_subplot(gs[1, 0])
for u in examples:
    ax.plot(lags_ms, kh_all[u], lw=1.4, label=f"unit {u}")
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("time since own spike (ms)")
ax.set_ylabel("filter gain (log rate)")
ax.set_title("(d) Spike-history filters", fontsize=11)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
gl = glm_df[[f"glm_{int(f)}" for f in FREQS]].values.ravel()
em = glm_df[[f"emp_{int(f)}" for f in FREQS]].values.ravel()
ax.plot(em, gl, ".", ms=5, color="tab:blue")
lim = [min(em.min(), gl.min()), max(em.max(), gl.max())]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("window-average evoked rate (Hz)")
ax.set_ylabel("GLM-predicted evoked rate (Hz)")
ax.set_title(f"(e) GLM vs direct estimate\nr = {np.corrcoef(em, gl)[0, 1]:.2f}, "
             f"{len(glm_df)} units x 5 tones", fontsize=11)

ax = fig.add_subplot(gs[1, 2])
ax.hist(glm_df.pseudo_r2, bins=15, color="tab:blue")
ax.axvline(glm_df.pseudo_r2.median(), color="k", ls="--",
           label=f"median {glm_df.pseudo_r2.median():.3f}")
ax.set_xlabel("McFadden pseudo-$R^2$ (5 ms bins)")
ax.set_ylabel("units")
ax.set_title(f"(f) Model fit quality\nGLM and direct BF agree in "
             f"{100 * match:.0f}% of units", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle(f"NeMoS Poisson GLM: tone-evoked spiking with spike history "
             f"({name}, {len(glm_df)} tuned units)", fontsize=13, y=0.97)
fig.savefig(f"{FIGDIR}/06_glm.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGDIR}/06_glm.png")
print(f"median pseudo-R2 {glm_df.pseudo_r2.median():.3f}, "
      f"BF agreement {100 * match:.0f}%, "
      f"median r = {glm_df.r_glm_emp.median():.2f}")
