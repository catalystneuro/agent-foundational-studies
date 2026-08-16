"""Frequency tuning in one session: per-unit PSTHs, tuning curves, statistics."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import audlib as A

path, url = A.list_assets()[0]
nwbfile, nap_nwb = A.load_session(url)
trials = A.trial_table(nwbfile)
spikes = nap_nwb["units"]
onsets = trials["onset"]
flab = trials["frequency"]
freqs = np.unique(flab)
log2f = np.log2(freqs / 1e3)

ev, bl = A.evoked_rates(spikes, onsets)
print("evoked/baseline arrays:", ev.shape)

# --- responsiveness: evoked vs baseline, paired across trials ---------------
t_resp, p_resp = stats.ttest_rel(ev, bl, axis=1)
p_resp_adj = A.bh_fdr(p_resp)
responsive = p_resp_adj < 0.01
print(f"sound-responsive units: {responsive.sum()}/{len(spikes)} "
      f"({100*responsive.mean():.0f}%), of which {np.sum(responsive & (t_resp>0))} "
      "are excited")

# --- frequency tuning: ANOVA over the 5 frequencies -------------------------
dev = ev - bl  # baseline-subtracted, per trial
p_tune, f_obs, f_null = A.permutation_p(dev, flab, freqs, n_perm=500, seed=1)
p_tune_adj = A.bh_fdr(p_tune)
tuned = p_tune_adj < 0.05
print(f"frequency-tuned units: {tuned.sum()}/{len(spikes)} ({100*tuned.mean():.0f}%)")
print(f"tuned among responsive: {np.sum(tuned & responsive)}/{responsive.sum()}")

tc = A.group_means(dev, flab, freqs)       # (n_units, n_freqs) evoked tuning curve
tc_sem = A.group_sems(dev, flab, freqs)
bf_idx = np.argmax(tc, axis=1)
bf = freqs[bf_idx]
spars = A.sparseness(tc)

np.savez("results_session0.npz", tc=tc, tc_sem=tc_sem, bf=bf, tuned=tuned,
         responsive=responsive, p_tune=p_tune_adj, freqs=freqs, spars=spars,
         f_obs=f_obs)

# --- figure 3: example units ------------------------------------------------
# pick three well-tuned, strongly driven units with different best frequencies
score = np.where(tuned & responsive, f_obs, -np.inf)
examples = []
for j in range(len(freqs)):
    cand = np.where((bf_idx == j) & np.isfinite(score))[0]
    if len(cand):
        examples.append(cand[np.argmax(score[cand])])
examples = sorted(examples, key=lambda i: -score[i])[:3]
examples = sorted(examples, key=lambda i: bf_idx[i])
print("example units:", examples, "BF (kHz):", bf[examples] / 1e3)

bins = np.arange(-0.05, 0.151, 0.002)
centers = bins[:-1] + 0.001
cmap = plt.get_cmap("viridis")
colors = [cmap(i / (len(freqs) - 1)) for i in range(len(freqs))]

fig, axes = plt.subplots(3, len(examples), figsize=(4.2 * len(examples), 10),
                         gridspec_kw={"height_ratios": [2.2, 1.3, 1.3]})
keys = list(spikes.keys())
for c, ui in enumerate(examples):
    t = spikes[keys[ui]].t
    # raster, trials grouped by frequency
    axr = axes[0, c]
    row = 0
    yticks, ylabels = [], []
    for j, f in enumerate(freqs):
        sub = onsets[flab == f][:120]
        rel, trial = A.peri_event_times(t, sub, bins[0], bins[-1])
        axr.plot(rel, trial + row, "|", color=colors[j], ms=2, mew=0.5)
        yticks.append(row + len(sub) / 2)
        row += len(sub)
        ylabels.append(f"{f/1e3:g}")
    axr.set_yticks(yticks)
    axr.set_yticklabels(ylabels)
    axr.set_ylim(0, row)
    axr.set_xlim(bins[0], bins[-1])
    axr.axvspan(0, 0.025, color="0.85", zorder=0)
    axr.set_ylabel("tone frequency (kHz)")
    axr.set_title(f"unit {ui}  (BF = {bf[ui]/1e3:g} kHz)")

    # PSTH per frequency
    axp = axes[1, c]
    for j, f in enumerate(freqs):
        axp.plot(centers, A.psth(t, onsets[flab == f], bins), color=colors[j],
                 lw=1.2, label=f"{f/1e3:g} kHz")
    axp.axvspan(0, 0.025, color="0.85", zorder=0)
    axp.set_xlabel("time from tone onset (s)")
    axp.set_ylabel("rate (spikes/s)")
    if c == 0:
        axp.legend(frameon=False, fontsize=8)

    # tuning curve
    axt = axes[2, c]
    axt.errorbar(log2f, tc[ui], yerr=tc_sem[ui], marker="o", color="k", capsize=3)
    axt.axhline(0, color="0.6", lw=0.8, ls="--")
    axt.set_xticks(log2f)
    axt.set_xticklabels([f"{f/1e3:g}" for f in freqs])
    axt.set_xlabel("tone frequency (kHz)")
    axt.set_ylabel("evoked rate\n(spikes/s over baseline)")
    axt.set_title(f"ANOVA F = {f_obs[ui]:.1f}, p = {p_tune_adj[ui]:.3g}", fontsize=10)

for ax in axes.ravel():
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Frequency tuning of single auditory-cortex units — {path}", y=1.0)
fig.tight_layout()
fig.savefig("fig03_example_units.png", dpi=150)
plt.close(fig)
print("saved fig03_example_units.png")
