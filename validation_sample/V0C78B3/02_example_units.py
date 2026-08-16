"""Example single units: tone-aligned rasters, PSTHs, and frequency tuning curves."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import analysis
import common

SESSION = "sub-LA11/sub-LA11_ses-1_behavior.nwb"
s = common.load_session(SESSION)
onsets = s["trials"]["start_time"].values
freqs = np.unique(s["frequency"])
colors = dict(zip(freqs, plt.get_cmap("viridis")(np.linspace(0, 0.92, len(freqs)))))

tuning, sem, driven, st, _ = analysis.session_tuning(s["units"], onsets, s["frequency"])
st["sig_freq"] = analysis.fdr(st["p_freq"].values)

# one strongly driven, frequency-selective example per best frequency
examples = []
for f in freqs:
    cand = st[(st.sig_freq) & (st.best_freq == f)].sort_values(
        "peak_evoked", ascending=False)
    if len(cand):
        examples.append(cand.index[0])
print("example units:", examples)

n = len(examples)
fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 8.6),
                         gridspec_kw=dict(height_ratios=[2.2, 1.2, 1.2], hspace=0.42,
                                          wspace=0.32))
for c, u in enumerate(examples):
    ax_r, ax_p, ax_t = axes[0, c], axes[1, c], axes[2, c]

    # --- raster: trials grouped by frequency, 120 trials shown per frequency
    row = 0
    for f in freqs:
        ev = onsets[s["frequency"] == f][:120]
        peth = nap.compute_perievent(s["units"][u], nap.Ts(ev), (-0.1, 0.25))
        for i, k in enumerate(peth.keys()):
            t = peth[k].t
            ax_r.plot(t, np.full_like(t, row + i), "|", color=colors[f], ms=2.2, mew=0.6)
        row += len(ev)
        ax_r.axhline(row, color="0.75", lw=0.5)
    ax_r.axvspan(0, s["duration"], color="0.85", zorder=0)
    ax_r.set_xlim(-0.1, 0.25)
    ax_r.set_ylim(0, row)
    ax_r.set_title(f"unit {u}   BF = {int(st.best_freq[u]/1000)} kHz", fontsize=10)
    if c == 0:
        ax_r.set_ylabel("trials (grouped by frequency)")
    ax_r.set_yticks(np.arange(len(freqs)) * 120 + 60)
    ax_r.set_yticklabels([f"{int(f/1000)}k" for f in freqs], fontsize=8)

    # --- PSTH per frequency
    for f in freqs:
        t, r = analysis.psth(s["units"][u], onsets[s["frequency"] == f])
        ax_p.plot(t, r, color=colors[f], lw=1.3, label=f"{int(f/1000)} kHz")
    ax_p.axvspan(0, s["duration"], color="0.85", zorder=0)
    ax_p.set_xlim(-0.1, 0.25)
    ax_p.set_xlabel("time from tone onset (s)")
    if c == 0:
        ax_p.set_ylabel("firing rate (spikes/s)")
        ax_p.legend(fontsize=6.5, frameon=False, ncol=1, loc="upper right",
                    handlelength=1.2, labelspacing=0.3)

    # --- tuning curve
    ax_t.errorbar(freqs / 1000, tuning.loc[u].values, yerr=sem.loc[u].values,
                  marker="o", color="k", lw=1.5, capsize=3)
    ax_t.axhline(0, color="0.6", lw=0.8, ls="--")
    ax_t.set_xscale("log", base=2)
    ax_t.set_xticks(freqs / 1000)
    ax_t.set_xticklabels([f"{int(f/1000)}" for f in freqs])
    ax_t.set_xlabel("tone frequency (kHz)")
    if c == 0:
        ax_t.set_ylabel("evoked rate\n(spikes/s above baseline)")
    ax_t.set_title(f"p(Kruskal-Wallis) = {st.p_freq[u]:.1e}", fontsize=8)

fig.suptitle(f"{s['subject']} session {s['session_id']}: tone-evoked responses of "
             "five frequency-tuned auditory-cortex units", y=0.985)
fig.savefig("fig02_example_units.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig02_example_units.png")
