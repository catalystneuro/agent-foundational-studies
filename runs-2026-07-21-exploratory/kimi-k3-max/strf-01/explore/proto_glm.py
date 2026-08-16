"""Prototype: GLM-based STRF for one fiber (encoding-model formulation).

Stimulus = one-hot frequency channels (tone present/absent per 1 ms bin),
each convolved with a raised-cosine temporal basis -> Poisson GLM.
Recovered per-frequency temporal kernels = STRF estimate.
"""
import json
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import remfile
from scipy.ndimage import gaussian_filter1d

import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

OUT = "explore/figs"

sample = json.load(open("explore/sample_urls.json"))
a = [x for x in sample if "G160504-519" in x["path"]][0]
disk_cache = remfile.DiskCache("/tmp/remfile_cache_strf")
f = h5py.File(remfile.File(a["url"], disk_cache=disk_cache), "r")

spike_times = np.asarray(f["units/spike_times"][:], dtype=np.float64)
spike_index = np.asarray(f["units/spike_times_index"][:], dtype=np.int64)
tags = f["units/tag"][:].astype(str)
starts = np.concatenate([[0], spike_index[:-1]])
sweep_spikes = [spike_times[s:e] for s, e in zip(starts, spike_index)]

DELAY = 0.00775
DUR = 0.05
SWEEP_T = 0.08          # BF sweep length (s)
GAP = 0.02              # artificial gap between concatenated sweeps (s)

bf = {}
for i, t in enumerate(tags):
    m = re.match(r"BF_FREQ(\d+)_rep(\d+)", t)
    if m:
        bf.setdefault(int(m.group(1)), []).append(i)
freqs = np.array(sorted(bf))
F = len(freqs)
f2i = {fr: k for k, fr in enumerate(freqs)}

sil_idx = [i for i, t in enumerate(tags) if t.startswith("BF_silent")]
sil_rate = np.mean([len(sweep_spikes[i]) for i in sil_idx]) / SWEEP_T

# ---- build concatenated stimulus matrix + spike counts at 1 ms ----------
bin_s = 0.001
sweep_bins = int(SWEEP_T / bin_s)          # 80
gap_bins = int(GAP / bin_s)                # 20
step = sweep_bins + gap_bins
sweep_list = [(fr, i) for fr in freqs for i in bf[fr]]   # 205 sweeps
n_bins = len(sweep_list) * step
X_stim = np.zeros((n_bins, F))
y = np.zeros(n_bins)
for n, (fr, i) in enumerate(sweep_list):
    b0 = n * step
    on = int(DELAY / bin_s)
    off = int((DELAY + DUR) / bin_s)
    X_stim[b0 + on:b0 + off, f2i[fr]] = 1.0
    sp = sweep_spikes[i]
    sp = sp[(sp >= 0) & (sp < SWEEP_T)]
    cb, _ = np.histogram(sp, bins=np.arange(sweep_bins + 1) * bin_s)
    y[b0:b0 + sweep_bins] = cb
print("design:", X_stim.shape, "total spikes:", y.sum())

# ---- temporal basis + GLM ----------------------------------------------
win = 40   # 40 ms kernel window
basis = nmo.basis.RaisedCosineLinearConv(n_basis_funcs=6, window_size=win)
X = basis.compute_features(X_stim)     # (n_bins, F*6)
print("convolved design:", X.shape)

model = nmo.glm.GLM(solver_name="LBFGS", regularizer="Ridge", regularizer_strength=1e-4)
model.fit(X, y)
print("score:", model.score(X, y))

# reconstruct per-frequency kernels
_, kernels = basis.evaluate_on_grid(win)   # (win, n_basis)
coef = np.asarray(model.coef_).reshape(F, 6)
strf_glm = coef @ kernels.T                # (F, win) in log-rate units
# convert to rate modulation: exp(intercept + k) - exp(intercept) ~ net rate
intercept = float(np.asarray(model.intercept_).ravel()[0])
rate_map = np.exp(intercept + strf_glm) - np.exp(intercept)   # sp/s per bin? counts/bin
rate_map = rate_map / bin_s                                    # -> sp/s

# ---- PSTH-based map for comparison --------------------------------------
edges = np.arange(-0.005, 0.0705, 0.0005)
centers = edges[:-1] + 0.00025
psth = np.zeros((F, len(centers)))
for fi, fr in enumerate(freqs):
    sp = np.concatenate([sweep_spikes[i] - DELAY for i in bf[fr]])
    sp = sp[(sp >= edges[0]) & (sp < edges[-1])]
    c, _ = np.histogram(sp, bins=edges)
    psth[fi] = c / (len(bf[fr]) * 0.0005)
net = gaussian_filter1d(psth - sil_rate, 2, axis=1)

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
vmax = np.percentile(np.abs(net), 99)
axes[0].pcolormesh(centers * 1e3, freqs, net, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[0].set_title("PSTH-based STRF (net rate)")
axes[0].set_xlabel("time from onset (ms)"); axes[0].set_ylabel("frequency (Hz)")

t_k = np.arange(win) * 1.0   # ms
vmax2 = np.percentile(np.abs(rate_map), 99)
axes[1].pcolormesh(t_k, freqs, rate_map, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)
axes[1].set_title("GLM-based STRF (net rate)")
axes[1].set_xlabel("latency (ms)")

# compare at CF
cf_i = np.argmax(net[:, (centers > 0.005) & (centers < 0.055)].mean(axis=1))
axes[2].plot(centers * 1e3, net[cf_i], label="PSTH", color="k")
axes[2].plot(t_k, rate_map[cf_i], label="GLM", color="r")
axes[2].set_title(f"kernel at CF={freqs[cf_i]} Hz")
axes[2].set_xlabel("time (ms)"); axes[2].legend()
# correlation between maps over matching window
m = (centers >= 0) & (centers < 0.040)
psth_ds = net[:, m]
# average psth into 1 ms bins to match GLM grid
psth_1ms = psth_ds.reshape(F, 40, 2).mean(axis=2)
cc = np.corrcoef(psth_1ms.ravel(), rate_map[:, :40].ravel())[0, 1]
axes[2].set_title(f"kernel at CF={freqs[cf_i]} Hz; map corr={cc:.2f}")
fig.tight_layout()
fig.savefig(f"{OUT}/proto_glm.png", dpi=150)
print("saved proto_glm.png, map corr:", cc)
f.close()
