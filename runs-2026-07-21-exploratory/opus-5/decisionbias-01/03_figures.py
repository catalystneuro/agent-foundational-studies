"""All figures for the pre-stimulus decision-bias analysis."""
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import ibl_common as ic

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "savefig.bbox": "tight",
})

C_LEFT, C_RIGHT = "#2f6fb5", "#c2453a"   # left block (p(left)=0.8) / right block
C_NEURAL, C_NULL, C_MOVE = "#33413f", "#b9bdbb", "#d99b2e"

results = pickle.load(open("all_results.pkl", "rb"))
ex = results[0]                              # example session (arrays kept)
summary = pd.read_csv("session_summary.csv")
print(f"{len(results)} sessions; example = {ex['subject']}")


def stouffer(pvals):
    p = np.asarray([q for q in pvals if np.isfinite(q)])
    z = stats.norm.isf(p)
    return float(stats.norm.sf(z.sum() / np.sqrt(len(z)))), len(z)


def annotate_p(ax, p, x, y, prefix="p = "):
    txt = f"{prefix}< {1/201:.3f}" if p <= 1 / 201 + 1e-9 else f"{prefix}{p:.3f}"
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="top", fontsize=8)


# =====================================================================
# Figure 1 - the task's hidden block structure and the behavioural bias
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4),
                         gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.45,
                                      "wspace": 0.28})

ax = axes[0, 0]
trl = ex["_trials"]
t = np.arange(len(trl))
ax.fill_between(t, 0, 1, where=ex["_y_block"] == 1, color=C_LEFT, alpha=0.18, step="mid",
                transform=ax.get_xaxis_transform(), label="left block  p(left)=0.8")
ax.fill_between(t, 0, 1, where=ex["_y_block"] == 0, color=C_RIGHT, alpha=0.18, step="mid",
                transform=ax.get_xaxis_transform(), label="right block  p(left)=0.2")
run = pd.Series(ex["_y_choice"]).rolling(15, center=True, min_periods=5).mean()
ax.plot(t, run, color="k", lw=1.2)
ax.axhline(0.5, color="0.5", lw=0.7, ls=":")
ax.set(xlabel="trial", ylabel="P(report left)\n(15-trial running mean)", ylim=(0, 1),
       title=f"{ex['subject']}: choices track the hidden block")
ax.legend(loc="upper right", ncol=1, framealpha=0.9)

ax = axes[0, 1]
for name, col, lab in [("left_block", C_LEFT, "left block"),
                       ("right_block", C_RIGHT, "right block")]:
    g = ex["psychometric"][name]
    n = g["count"].values
    se = np.sqrt(g["mean"].values * (1 - g["mean"].values) / n)
    ax.errorbar(g.index.values, g["mean"].values, yerr=se, marker="o", ms=4, lw=1.4,
                color=col, capsize=2, label=lab)
ax.axvline(0, color="0.6", lw=0.7, ls=":")
ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
ax.set(xlabel="signed contrast (%)   negative = stimulus on the left",
       ylabel="P(report left)", ylim=(-0.03, 1.03), title="Psychometric curves shift with the block")
ax.legend(loc="upper right")

ax = axes[1, 0]
order = summary.sort_values("bias_zero_contrast").reset_index(drop=True)
cols = np.where(order.bias_chi2_p < 0.05, "#2f6fb5", "0.7")
ax.bar(np.arange(len(order)), order.bias_zero_contrast, color=cols)
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="session (sorted)", ylabel="bias on 0% contrast\nP(left|L block) - P(left|R block)",
       title="Behavioural bias on uninformative (0%) trials")
ax.text(0.03, 0.95, f"{int((order.bias_chi2_p < 0.05).sum())}/{len(order)} sessions p<0.05\n"
                    f"mean = {order.bias_zero_contrast.mean():.2f}",
        transform=ax.transAxes, va="top", fontsize=8)

ax = axes[1, 1]
w = ex["_move"]
for lab, sel, col in [("left block", ex["_y_block"] == 1, C_LEFT),
                      ("right block", ex["_y_block"] == 0, C_RIGHT)]:
    ax.hist(w[sel, 2], bins=np.linspace(0, 0.2, 40), histtype="step", density=True,
            color=col, lw=1.4, label=lab)
ax.set(xlabel="max |wheel velocity| in the pre-stimulus window (rad/s)",
       ylabel="density", title="The pre-stimulus window is a quiescence period")
ax.text(0.35, 0.6, f"mean |v|: {ex['wheel_absvel_left']*1e3:.1f} vs "
                   f"{ex['wheel_absvel_right']*1e3:.1f} mrad/s\n"
                   f"Mann-Whitney p = {ex['wheel_absvel_p']:.2f}",
        transform=ax.transAxes, fontsize=8)
ax.legend(loc="upper right")

fig.suptitle("Figure 1  |  IBL decision task (DANDI:000409): a hidden prior biases upcoming choices",
             y=0.99, fontsize=11)
fig.savefig("fig01_task_and_behavior.png")
plt.close(fig)

# =====================================================================
# Figure 2 - raw pre-stimulus neural activity
# =====================================================================
spikes, stim_on = ex["_spikes"], ex["_stim_on"]
yb = ex["_y_block"]
edges = np.arange(-1.0, 0.6001, 0.02)
centers = edges[:-1] + 0.01

unit_order = np.argsort(-np.abs(ex["unit_auc"] - 0.5))
sel_units = unit_order[:2]

fig, axes = plt.subplots(2, 3, figsize=(12, 6.2),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.3})

for c, u in enumerate(sel_units):
    st = spikes[u]
    rel = [st[(st > s + edges[0]) & (st < s + edges[-1])] - s for s in stim_on]
    ordering = np.concatenate([np.where(yb == 1)[0], np.where(yb == 0)[0]])
    ax = axes[0, c]
    for row, tr_i in enumerate(ordering):
        col = C_LEFT if yb[tr_i] == 1 else C_RIGHT
        ax.plot(rel[tr_i], np.full(len(rel[tr_i]), row), "|", ms=1.4, color=col, alpha=0.6)
    ax.axvline(0, color="k", lw=0.9)
    ax.axvspan(*ic.PRE_WIN, color="0.85", zorder=0)
    ax.set(xlabel="time from stimulus onset (s)", ylabel="trial (sorted by block)",
           title=f"unit {ex['_meta'].unit_index.values[u]} "
                 f"({ex['unit_region'][u][:22]})\nblock AUC = {ex['unit_auc'][u]:.2f}, "
                 f"p = {ex['unit_p'][u]:.3f}", xlim=(edges[0], edges[-1]))

    ax = axes[1, c]
    for lab, sel, col in [("left block", yb == 1, C_LEFT), ("right block", yb == 0, C_RIGHT)]:
        m = np.array([np.histogram(rel[i], edges)[0] for i in np.where(sel)[0]]) / 0.02
        mu, se = m.mean(0), m.std(0) / np.sqrt(len(m))
        ax.plot(centers, mu, color=col, lw=1.3, label=lab)
        ax.fill_between(centers, mu - se, mu + se, color=col, alpha=0.25, lw=0)
    ax.axvline(0, color="k", lw=0.9)
    ax.axvspan(*ic.PRE_WIN, color="0.85", zorder=0)
    ax.set(xlabel="time from stimulus onset (s)", ylabel="firing rate (Hz)",
           title="PSTH by block", xlim=(edges[0], edges[-1]))
    ax.legend(loc="upper left")

ax = axes[0, 2]
pop = np.array([np.mean([np.histogram(
    st[(st > s + edges[0]) & (st < s + edges[-1])] - s, edges)[0] for s in stim_on[:150]], 0)
    for st in spikes[:200]]) / 0.02
im = ax.imshow(pop[np.argsort(-pop.mean(1))], aspect="auto", cmap="magma",
               extent=[edges[0], edges[-1], 0, pop.shape[0]],
               vmax=np.percentile(pop, 99))
ax.axvline(0, color="w", lw=0.9)
ax.set(xlabel="time from stimulus onset (s)", ylabel="unit (sorted by rate)",
       title="Trial-averaged population activity\n(first 200 units, first 150 trials)")
plt.colorbar(im, ax=ax, label="Hz", pad=0.02)

ax = axes[1, 2]
counts = ex["_X"]
rate = counts.sum(1) / (ic.PRE_WIN[1] - ic.PRE_WIN[0]) / counts.shape[1]
ax.plot(np.arange(len(rate)), pd.Series(rate).rolling(11, center=True, min_periods=3).mean(),
        color="k", lw=1)
ax.fill_between(np.arange(len(rate)), 0, 1, where=yb == 1, color=C_LEFT, alpha=0.15,
                step="mid", transform=ax.get_xaxis_transform())
ax.fill_between(np.arange(len(rate)), 0, 1, where=yb == 0, color=C_RIGHT, alpha=0.15,
                step="mid", transform=ax.get_xaxis_transform())
ax.set(xlabel="trial", ylabel="mean pre-stimulus rate (Hz/unit)",
       title="Population rate drifts slowly across the session\n(why the null must preserve block autocorrelation)")

fig.suptitle("Figure 2  |  Raw pre-stimulus activity in the example session", y=0.995, fontsize=11)
fig.savefig("fig02_raw_activity.png")
plt.close(fig)

# =====================================================================
# Figure 3 - decoding the block (the prior) before stimulus onset
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4),
                         gridspec_kw={"hspace": 0.45, "wspace": 0.28})

ax = axes[0, 0]
oof = ex["_oof_block"]
ax.fill_between(t, 0, 1, where=yb == 1, color=C_LEFT, alpha=0.18, step="mid",
                transform=ax.get_xaxis_transform())
ax.fill_between(t, 0, 1, where=yb == 0, color=C_RIGHT, alpha=0.18, step="mid",
                transform=ax.get_xaxis_transform())
ax.plot(t, pd.Series(oof).rolling(11, center=True, min_periods=3).mean(), color="k", lw=1.1)
ax.axhline(0, color="0.5", lw=0.7, ls=":")
ax.set(xlabel="trial", ylabel="held-out decoder score\n(> 0 = 'left block')",
       title="Cross-validated pre-stimulus block decoder")

ax = axes[0, 1]
ax.hist(ex["null_block"], bins=25, color=C_NULL, label="pseudo-sessions (null)")
ax.axvline(ex["auc_block"], color=C_NEURAL, lw=2, label="observed")
ax.axvline(ex["auc_block_move"], color=C_MOVE, lw=2, ls="--", label="wheel + licks only")
ax.set(xlabel="AUC (block identity)", ylabel="count",
       title="Observed vs. task-matched surrogate blocks")
annotate_p(ax, ex["p_block"], 0.03, 0.96)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))

ax = axes[1, 0]
ax.plot(ex["time_centers"], ex["time_auc_block"], "-o", ms=3.5, color=C_NEURAL,
        label="observed")
ax.plot(ex["time_centers"], ex["time_null95"], color=C_NULL, lw=1.4, ls="--",
        label="95th pct of null")
ax.axvline(0, color="k", lw=0.9)
ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
ax.axvspan(*ic.PRE_WIN, color="0.9", zorder=0)
ax.set(xlabel="centre of 200 ms window, relative to stimulus onset (s)",
       ylabel="block decoding AUC", title="Time-resolved block decoding (example session)")
ax.legend(loc="upper left")

ax = axes[1, 1]
tc = np.array([r["time_centers"] for r in results])[0]
M = np.array([r["time_auc_block"] for r in results])
N = np.array([r["time_null95"] for r in results])
ax.plot(tc, M.mean(0), "-o", ms=3.5, color=C_NEURAL, label="mean over sessions")
ax.fill_between(tc, M.mean(0) - M.std(0) / np.sqrt(len(M)),
                M.mean(0) + M.std(0) / np.sqrt(len(M)), color=C_NEURAL, alpha=0.25, lw=0)
ax.plot(tc, N.mean(0), color=C_NULL, lw=1.4, ls="--", label="mean 95th pct of null")
ax.axvline(0, color="k", lw=0.9)
ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
ax.axvspan(*ic.PRE_WIN, color="0.9", zorder=0)
ax.set(xlabel="centre of 200 ms window, relative to stimulus onset (s)",
       ylabel="block decoding AUC", title=f"Time-resolved, all {len(results)} sessions")
ax.legend(loc="upper left")

fig.suptitle("Figure 3  |  The block prior is present in population activity before the stimulus",
             y=0.995, fontsize=11)
fig.savefig("fig03_block_decoding.png")
plt.close(fig)

# =====================================================================
# Figure 4 - predicting the upcoming choice on uninformative trials
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.4),
                         gridspec_kw={"hspace": 0.45, "wspace": 0.28})

ax = axes[0, 0]
_ok = ex["_score_ok"]
sz = ex["_score_z"][_ok]
yz = ex["_y_choice"][ex["_zero_mask"]][_ok]
parts = ax.violinplot([sz[yz == 1], sz[yz == 0]], positions=[0, 1], showmeans=True,
                      widths=0.7)
for pc, col in zip(parts["bodies"], [C_LEFT, C_RIGHT]):
    pc.set_facecolor(col); pc.set_alpha(0.45)
for k in ("cbars", "cmins", "cmaxes", "cmeans"):
    parts[k].set_color("0.3")
rng = np.random.default_rng(0)
for j, (v, col) in enumerate([(sz[yz == 1], C_LEFT), (sz[yz == 0], C_RIGHT)]):
    ax.plot(j + rng.normal(0, 0.05, len(v)), v, ".", ms=3, color=col, alpha=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["reports left", "reports right"])
ax.set(ylabel="pre-stimulus decoder score\n(trained on block, never on choice)",
       title="0%-contrast trials, example session")
ax.text(0.03, 0.96, f"AUC = {ex['auc_crossdecode_choice']:.2f}", transform=ax.transAxes,
        va="top", fontsize=9)
annotate_p(ax, ex["p_crossdecode"], 0.03, 0.88)

ax = axes[0, 1]
ax.hist(ex["null_crossdecode"], bins=25, color=C_NULL, label="circular-shift null")
ax.axvline(ex["auc_crossdecode_choice"], color=C_NEURAL, lw=2, label="observed")
ax.set(xlabel="AUC (upcoming choice on 0% trials)", ylabel="count",
       title="Cross-condition read-out vs. null")
ax.legend(loc="upper left")

ax = axes[1, 0]
o = summary.sort_values("auc_crossdecode_choice").reset_index(drop=True)
cols = np.where(o.p_crossdecode < 0.05, C_NEURAL, "0.75")
ax.bar(np.arange(len(o)), o.auc_crossdecode_choice - 0.5, bottom=0.5, color=cols)
ax.axhline(0.5, color="k", lw=0.8)
ax.set(xlabel="session (sorted)", ylabel="AUC: pre-stimulus score -> choice",
       title="Upcoming choice on 0% trials, all sessions", ylim=(0.3, 0.95))
pc, npc = stouffer(summary.p_crossdecode)
ax.text(0.03, 0.95, f"{int((o.p_crossdecode < 0.05).sum())}/{len(o)} sessions p<0.05\n"
                    f"mean AUC = {o.auc_crossdecode_choice.mean():.3f}\n"
                    f"Stouffer p = {pc:.1e}", transform=ax.transAxes, va="top", fontsize=8)

ax = axes[1, 1]
ax.axhline(0, color="k", lw=0.8)
xs = np.arange(len(summary))
ax.bar(xs - 0.2, summary.block_coef, width=0.4, color="0.45", label="block label")
ax.bar(xs + 0.2, summary.resid_coef, width=0.4, color=C_NEURAL,
       label="neural score (within block)")
ax.set(xlabel="session", ylabel="logistic coefficient on P(report left)",
       title="Does the neural score add anything beyond the block label?")
zr = stats.wilcoxon(summary.resid_coef.dropna())
ax.text(0.03, 0.95, f"neural coef > 0 in {int((summary.resid_coef > 0).sum())}/"
                    f"{len(summary)} sessions\nWilcoxon p = {zr.pvalue:.3f}",
        transform=ax.transAxes, va="top", fontsize=8)
ax.legend(loc="lower right")

fig.suptitle("Figure 4  |  Pre-stimulus activity predicts the upcoming choice when the stimulus is uninformative",
             y=0.995, fontsize=11)
fig.savefig("fig04_choice_prediction.png")
plt.close(fig)

# =====================================================================
# Figure 5 - population summary and controls
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 6.6),
                         gridspec_kw={"hspace": 0.5, "wspace": 0.3})

ax = axes[0, 0]
o = summary.sort_values("auc_block").reset_index(drop=True)
idx_map = summary.sort_values("auc_block").index
nulls = [results[i]["null_block"] for i in idx_map]
for j, nl in enumerate(nulls):
    lo, hi = np.percentile(nl, [2.5, 97.5])
    ax.plot([j, j], [lo, hi], color=C_NULL, lw=3, solid_capstyle="butt")
ax.plot(np.arange(len(o)), o.auc_block, "o", ms=5, color=C_NEURAL, label="observed")
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.set(xlabel="session (sorted)", ylabel="block decoding AUC",
       title="Pre-stimulus block decoding vs. per-session null (grey: 95% of surrogates)")
pb, _ = stouffer(summary.p_block)
ax.text(0.03, 0.95, f"{int((summary.p_block < 0.05).sum())}/{len(summary)} sessions p<0.05\n"
                    f"mean AUC = {summary.auc_block.mean():.3f}\nStouffer p = {pb:.1e}",
        transform=ax.transAxes, va="top", fontsize=8)
ax.legend(loc="lower right")

ax = axes[0, 1]
labels = ["neural", "neural minus\nmovement", "neural minus\nmovement +\nhistory + drift",
          "movement\nonly", "movement +\nhistory only"]
vals = [summary.auc_block, summary.auc_block_resid, summary.auc_block_resid_full,
        summary.auc_block_move, summary.auc_block_hist]
ps = [summary.p_block, summary.p_block_resid, summary.p_block_resid_full,
      summary.p_block_move, summary.p_block_hist]
colors = [C_NEURAL, "#4c7a6d", "#7fa8a0", C_MOVE, "#e0c187"]
for j, (v, col) in enumerate(zip(vals, colors)):
    ax.plot(j + np.random.default_rng(j).normal(0, 0.06, len(v)), v, ".", color=col, ms=5)
    ax.plot([j - 0.28, j + 0.28], [v.mean()] * 2, color="k", lw=2)
ax.axhline(0.5, color="k", lw=0.8, ls=":")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7)
ax.set(ylabel="block decoding AUC", title="Nuisance-variable controls")
for j, (v, p) in enumerate(zip(vals, ps)):
    ax.text(j, 1.0, f"{int((p<0.05).sum())}/{len(v)}", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7)

ax = axes[1, 0]
au = np.concatenate([r["unit_auc"] for r in results])
up = np.concatenate([r["unit_p"] for r in results])
ax.hist(au, bins=60, color=C_NULL, label=f"all units (n={len(au)})")
ax.hist(au[up < 0.05], bins=60, color=C_NEURAL, label=f"p<0.05 vs. surrogate blocks")
ax.axvline(0.5, color="k", lw=0.8, ls=":")
ax.set(xlabel="single-unit AUC: pre-stimulus rate -> block identity", ylabel="units",
       title="Single units carry only a weak block signal")
ax.text(0.02, 0.95, f"{100*np.mean(up<0.05):.1f}% significant (5% expected by chance)",
        transform=ax.transAxes, va="top", fontsize=8)
ax.legend(loc="upper right")

ax = axes[1, 1]
reg = pd.concat([r["region_auc"].assign(subject=r["subject"]) for r in results])
agg = reg.groupby("region").agg(auc=("auc", "mean"), n=("auc", "size"),
                                units=("n_units", "sum")).query("n >= 3")
agg = agg.sort_values("auc").tail(14)
ax.barh(np.arange(len(agg)), agg.auc - 0.5, left=0.5, color=C_NEURAL, height=0.7)
ax.axvline(0.5, color="k", lw=0.8)
ax.set_yticks(np.arange(len(agg)))
ax.set_yticklabels([f"{r[:30]} ({int(n)})" for r, n in zip(agg.index, agg.n)], fontsize=7)
ax.set(xlabel="block decoding AUC (mean over sessions)",
       title="By anatomical location (n sessions in brackets)")

fig.suptitle("Figure 5  |  Population summary and controls", y=0.995, fontsize=11)
fig.savefig("fig05_summary_and_controls.png")
plt.close(fig)

# =====================================================================
# Text summary
# =====================================================================
lines = []
add = lines.append
add(f"sessions: {len(summary)}   units (total): {int(summary.n_units.sum())}   "
    f"trials (total): {int(summary.n_trials.sum())}   0%-contrast trials: {int(summary.n_zero.sum())}")
add("")
add("behaviour  bias on 0%% trials: mean %.3f, %d/%d sessions chi2 p<0.05"
    % (summary.bias_zero_contrast.mean(), int((summary.bias_chi2_p < .05).sum()), len(summary)))
for key, pk, name in [
    ("auc_block", "p_block", "block decoding, pre-stimulus population"),
    ("auc_block_resid", "p_block_resid", "  ... after removing wheel+lick covariates"),
    ("auc_block_move", "p_block_move", "  ... wheel+lick features alone"),
    ("auc_block_hist", "p_block_hist", "  ... trial history + drift alone"),
    ("auc_block_resid_full", "p_block_resid_full", "  ... after removing movement+history+drift"),
    ("auc_choice_zero", "p_choice_zero", "choice decoding, 0%-contrast trials"),
    ("auc_choice_zero_move", "p_choice_zero_move", "  ... wheel+lick features alone"),
    ("auc_crossdecode_choice", "p_crossdecode", "block decoder -> upcoming 0% choice"),
    ("auc_crossdecode_choice_resid", "p_crossdecode_resid", "  ... movement+history removed"),
]:
    p_comb, _ = stouffer(summary[pk])
    add(f"{name:52s} AUC {summary[key].mean():.3f} +- {summary[key].sem():.3f}   "
        f"{int((summary[pk] < .05).sum()):2d}/{len(summary)} sessions p<0.05   Stouffer p = {p_comb:.2e}")
w = stats.wilcoxon(summary.resid_coef.dropna())
add(f"{'within-block neural coefficient on choice':52s} "
    f"median {summary.resid_coef.median():.3f}   "
    f"{int((summary.resid_coef>0).sum())}/{len(summary)} positive   Wilcoxon p = {w.pvalue:.3f}")
add("")
add(f"single units: {100*np.mean(up<0.05):.1f}% significant vs. surrogate blocks (5% expected)")
txt = "\n".join(lines)
open("results_summary.txt", "w").write(txt + "\n")
print(txt)
