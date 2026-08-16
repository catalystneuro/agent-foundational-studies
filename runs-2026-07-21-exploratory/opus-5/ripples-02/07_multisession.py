"""Run the full pipeline on every linear-track session and pool the results."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, fisher_exact, wilcoxon

from pipeline import analyze_session

SESSIONS = ["Achilles_10252013", "Buddy_06272013", "Cicero_09012014",
            "Cicero_09172014", "Gatsby_08022013"]

summaries, events = [], []
for s in SESSIONS:
    cache = f"cache_events_{s}.csv"
    if os.path.exists(cache) and os.path.exists(f"cache_summary_{s}.csv"):
        res = pd.read_csv(cache)
        summ = pd.read_csv(f"cache_summary_{s}.csv").iloc[0].to_dict()
    else:
        print(f"\n=== {s} ===")
        out = analyze_session(s)
        res, summ = out["res"], out["summary"]
        res.to_csv(cache, index=False)
        pd.DataFrame([summ]).to_csv(f"cache_summary_{s}.csv", index=False)
    res["session"] = s
    events.append(res)
    summaries.append(summ)

summary = pd.DataFrame(summaries)
allev = pd.concat(events, ignore_index=True)
summary.to_csv("multisession_summary.csv", index=False)

cols = ["session", "track", "n_pyr", "n_place", "n_laps", "n_ripples", "ripple_rate",
        "ripple_dur_ms", "ripple_freq", "rate_nrem", "rate_rem", "rate_awake",
        "n_events", "frac_sig_PRE", "frac_sig_POST", "frac_forward", "ev", "rev"]
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

a = allev[allev.epoch == "POST"].significant
b = allev[allev.epoch == "PRE"].significant
odds, p = fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
print(f"\npooled POST vs PRE: {a.mean()*100:.1f}% vs {b.mean()*100:.1f}%, "
      f"odds ratio {odds:.2f}, Fisher p = {p:.2e}")
w = wilcoxon(summary.frac_sig_POST, summary.frac_sig_PRE)
n_up = int((summary.frac_sig_POST > summary.frac_sig_PRE).sum())
p_sign = binomtest(n_up, len(summary), 0.5, alternative="greater").pvalue
print(f"paired across {len(summary)} sessions: POST > PRE in {n_up}/{len(summary)} "
      f"(sign test p = {p_sign:.3f}; Wilcoxon p = {w.pvalue:.3f}, "
      f"whose floor at n={len(summary)} is 0.031)")
print(f"EV {summary.ev.mean()*100:.1f}% vs reverse EV {summary.rev.mean()*100:.1f}% "
      f"(mean over sessions)")

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
x = np.arange(len(summary))
short = [s.split("_")[0] + "\n" + s.split("_")[1][:4] for s in summary.session]

ax = fig.add_subplot(gs[0, 0])
w_ = 0.2
for i, (k, lab, c) in enumerate([("rate_nrem", "non-REM", "#023047"),
                                 ("rate_awake", "awake immobile", "#adb5bd"),
                                 ("rate_rem", "REM", "#e63946"),
                                 ("rate_run", "running", "#fb8500")]):
    ax.bar(x + (i - 1.5) * w_, summary[k], w_, label=lab, color=c)
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("ripples / min")
ax.set_title("ripple rate by state", fontsize=10)
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.bar(x - 0.2, summary.ripple_dur_ms, 0.4, color="#023047", label="duration (ms)")
ax.bar(x + 0.2, summary.ripple_freq, 0.4, color="#e63946", label="peak freq (Hz)")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_title("ripple properties", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.bar(x - 0.2, summary.n_pyr, 0.4, color="#adb5bd", label="pyramidal cells")
ax.bar(x + 0.2, summary.n_place, 0.4, color="#023047", label="place cells")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_title("recorded population", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 0])
for i in range(len(summary)):
    ax.plot([0, 1], [summary.frac_sig_PRE[i] * 100, summary.frac_sig_POST[i] * 100],
            "o-", color="#023047", alpha=0.8)
ax.axhline(5, color="k", ls="--", lw=1, label="chance (5%)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["PRE sleep", "POST sleep"])
ax.set_ylabel("significant replay events (%)")
ax.set_title(f"replay increases after the track\n"
             f"(POST > PRE in {n_up}/{len(summary)} sessions, "
             f"sign test p = {p_sign:.3f})", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 1])
ax.bar(x - 0.2, summary.ev * 100, 0.4, color="#023047", label="EV")
ax.bar(x + 0.2, summary.rev * 100, 0.4, color="#adb5bd", label="reverse EV")
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("explained variance (%)")
ax.set_title("reactivation of run-time correlations", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 2])
sig = allev[allev.significant]
fwd = sig.groupby("session").forward.mean() * 100
ax.bar(range(len(fwd)), fwd, color="#023047")
ax.axhline(50, color="k", ls="--", lw=1)
ax.set_xticks(range(len(fwd)))
ax.set_xticklabels([s.split("_")[0] + "\n" + s.split("_")[1][:4] for s in fwd.index],
                   fontsize=8)
ax.set_ylabel("forward replay (%)")
ax.set_title("forward vs reverse", fontsize=10)

fig.suptitle(f"DANDI:000044 — {len(summary)} linear-track sessions, "
             f"{int(summary.n_ripples.sum())} ripples, "
             f"{len(allev)} decoded events", y=0.96)
fig.savefig("fig07_multisession.png", dpi=150, bbox_inches="tight")
print("wrote fig07_multisession.png")
