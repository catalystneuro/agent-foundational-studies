"""Prototype the cortical STRF on a single 000986 session."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import strf_lib as S

DT = 0.005
N_LAGS = 50  # 250 ms
FIGDIR = "figures"

url = S.asset_url("000986", "sub-LA11/sub-LA11_ses-1_behavior.nwb")
d = S.load_ac_session(url)
units, trials = d["units"], d["trials"]

freqs = S.TONE_FREQS_HZ
fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
onsets = trials["start_time"].values

t0, t1 = onsets[0] - 1.0, onsets[-1] + 2.0
t_bins = np.arange(t0, t1, DT)
Sm = S.build_tone_stimulus(onsets, fidx, len(freqs), t_bins, 0.025)
print("stimulus matrix", Sm.shape, "tone bins per channel", Sm.sum(0))

# ---- raw data sanity figure -------------------------------------------------
rate_all = np.array([len(units[i]) / (t1 - t0) for i in units.index])
top = units.index[np.argsort(rate_all)[::-1][:12]]

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                       gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.12))
w0 = onsets[100] - 0.3
win = (w0, w0 + 8)
for k in range(len(freqs)):
    on = onsets[(fidx == k) & (onsets > win[0]) & (onsets < win[1])]
    ax[0].vlines(on, k - 0.4, k + 0.4, color=plt.cm.viridis(k / 4), lw=3)
ax[0].set_yticks(range(5))
ax[0].set_yticklabels([f"{f/1000:g} kHz" for f in freqs])
ax[0].set_ylabel("tone pip")
ax[0].set_title("Dandiset 000986 - mouse auditory cortex, passive pure tones (session LA11-1)")
for r, i in enumerate(top):
    st = units[i].t
    st = st[(st > win[0]) & (st < win[1])]
    ax[1].vlines(st, r, r + 0.85, color="k", lw=0.7)
ax[1].set_ylabel("unit (12 highest-rate)")
ax[1].set_xlabel("time (s)")
ax[1].set_xlim(win)
fig.savefig(f"{FIGDIR}/proto_raw.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- STRFs for a few units --------------------------------------------------
sel = top[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
lags = np.arange(N_LAGS) * DT * 1000
for a, i in zip(axes.ravel(), sel):
    c = S.bin_spikes(units[i].t, t_bins)
    strf = S.sta_strf(c, Sm, N_LAGS) / DT
    v = np.abs(strf).max()
    im = a.imshow(strf.T, aspect="auto", origin="lower", cmap="RdBu_r",
                  vmin=-v, vmax=v,
                  extent=[lags[0] - DT * 500, lags[-1] + DT * 500, -0.5, 4.5])
    a.set_yticks(np.arange(5))
    a.set_yticklabels([f"{f/1000:g}" for f in freqs])
    a.set_title(f"unit {i}  ({len(units[i])/(t1-t0):.1f} Hz)", fontsize=9)
    plt.colorbar(im, ax=a, label="Δ rate (sp/s)")
for a in axes[-1]:
    a.set_xlabel("lag (ms)")
for a in axes[:, 0]:
    a.set_ylabel("tone freq (kHz)")
fig.suptitle("Tone-pip STRFs (spike-triggered average)", y=0.99)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/proto_strf.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("done")
