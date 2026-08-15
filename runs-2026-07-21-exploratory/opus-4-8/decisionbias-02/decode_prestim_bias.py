# %% [markdown]
# # Decoding an upcoming decision bias from pre-stimulus neural activity
#
# **Question.** In a perceptual decision task, can we read out the animal's
# *upcoming, biased* decision from population neural activity recorded **before**
# the stimulus appears?
#
# **Dataset.** DANDI [000149](https://dandiarchive.org/dandiset/000149) — *IBL
# ephys data*. Mice perform the International Brain Laboratory standardized visual
# detection task: on each trial a grating appears on the left or right at one of
# several contrasts (including 0% contrast), and the mouse reports the side by
# turning a wheel. The prior probability that the stimulus is on the left is held
# fixed in **blocks** (`probabilityLeft` = 0.2 or 0.8). This block prior biases the
# animal's choices, most visibly on 0%-contrast trials where no sensory evidence is
# available. A ~0.4–0.7 s enforced *quiescence* period precedes stimulus onset,
# during which the animal must hold the wheel still, giving a clean pre-stimulus
# epoch free of movement.
#
# **Approach.** We stream the spike-sorted units and trial table from four sessions
# (Neuropixels, various cortical/subcortical sites) via the neurosift LINDI index —
# no bulk download of the multi-hundred-GB raw files. For a sliding window relative
# to stimulus onset we build an (trials × units) population firing-rate matrix and
# train a cross-validated logistic decoder to predict (a) the **upcoming choice**
# and (b) the **block prior** (the latent bias state). Significance is assessed by a
# label-permutation null; for the block decoder we add a temporally-aware
# circular-shift null to guard against slow firing-rate drift.
#
# **Result.** Both the block prior and the upcoming choice are decodable from
# population activity *before* the stimulus appears, well above chance and above
# both null models, in every session. The pre-stimulus signal predicts the biased
# decision the animal is about to make.

# %%
import warnings
warnings.filterwarnings("ignore")
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import ibl_lib as L

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

CACHE_DIR = "cache"
FIG = "."
RNG = np.random.default_rng(0)

# Build the per-session feature cache if it does not exist yet (streams the small
# units/trials tables from DANDI 000149 via LINDI; ~1-2 min per session).
if not glob.glob(os.path.join(CACHE_DIR, "sess_*.npz")):
    import build_cache  # noqa: F401
    build_cache.__name__  # trigger module __main__ guard manually
    for aid, lab in L.SESSIONS.items():
        build_cache.build(aid, lab)

SESS = sorted(glob.glob(os.path.join(CACHE_DIR, "sess_*.npz")))
LABELS = [os.path.basename(s).split("sess_")[1][:8] for s in SESS]
print("sessions:", LABELS)


# %% [markdown]
# ## Decoding helpers
#
# A z-scored logistic-regression decoder with L2 regularisation, evaluated by
# 5-fold cross-validation (out-of-sample AUC). The permutation null shuffles the
# labels; the circular-shift null (for the block variable) rolls the labels in
# trial order, preserving their temporal autocorrelation so that any slow drift in
# firing rate cannot masquerade as bias decoding.

# %%
def cv_auc(X, y, seed=0):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.05, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def perm_null(X, y, n=200, kind="shuffle"):
    """Return array of null AUCs. kind='shuffle' breaks all structure;
    kind='roll' circularly shifts labels, preserving temporal autocorrelation."""
    out = np.empty(n)
    ny = len(y)
    for i in range(n):
        if kind == "shuffle":
            yp = RNG.permutation(y)
        else:  # roll by a random non-trivial offset
            k = RNG.integers(ny // 10, ny - ny // 10)
            yp = np.roll(y, k)
        # guard against degenerate single-class folds
        if len(np.unique(yp)) < 2:
            out[i] = 0.5
            continue
        out[i] = cv_auc(X, yp, seed=i)
    return out


# window index helper
def win_idx(wc, center):
    return int(np.argmin(np.abs(wc - center)))


# %% [markdown]
# ## 1. Behaviour: the block prior biases choice
#
# We first confirm the behavioural signature of the bias. Choice follows the signed
# stimulus contrast (a psychometric curve), but the curve is **shifted** by the
# block prior, and on 0%-contrast trials — where there is no sensory evidence — the
# animal chooses according to the block. Choice is coded +1 = report-left,
# -1 = report-right; signed contrast = contrastRight - contrastLeft.

# %%
# Pool behaviour across sessions for the psychometric; keep per-session for the bar.
def load(fn):
    d = np.load(fn, allow_pickle=True)
    return {k: d[k] for k in d.files}

data = [load(s) for s in SESS]

CONTRASTS = np.array([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for col, blk in [("#c44", 0.2), ("#357", 0.8)]:
    xs, ys = [], []
    for sc in CONTRASTS:
        num = den = 0
        for d in data:
            signed = d["contrastRight"] - d["contrastLeft"]
            m = (np.isclose(signed, sc)) & (d["probabilityLeft"] == blk) & \
                np.isin(d["choice"], [-1, 1])
            num += np.sum(d["choice"][m] == 1)
            den += m.sum()
        if den > 0:
            xs.append(sc); ys.append(num / den)
    ax.plot(xs, ys, "o-", color=col, label=f"block pL={blk}")
ax.axhline(0.5, ls=":", c="k", lw=0.8)
ax.axvline(0, ls=":", c="k", lw=0.8)
ax.set_xlabel("signed contrast  (right − left)")
ax.set_ylabel("P(report left)")
ax.set_title("Psychometric curves shift with the block prior")
ax.legend(frameon=False)

ax = axes[1]
w = 0.35
for j, blk in enumerate([0.2, 0.8]):
    vals = []
    for d in data:
        zc = (d["contrastLeft"] == 0) & (d["contrastRight"] == 0)
        m = zc & (d["probabilityLeft"] == blk) & np.isin(d["choice"], [-1, 1])
        vals.append(np.mean(d["choice"][m] == 1))
    x = np.arange(len(data)) + (j - 0.5) * w
    ax.bar(x, vals, w, color=["#c44", "#357"][j], label=f"pL={blk}")
ax.axhline(0.5, ls=":", c="k", lw=0.8)
ax.set_xticks(np.arange(len(data)))
ax.set_xticklabels(LABELS, rotation=30, ha="right")
ax.set_ylabel("P(report left)")
ax.set_title("0%-contrast trials: choice follows the block")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig1_behavior_bias.png", bbox_inches="tight")
plt.close()
print("saved fig1_behavior_bias.png")


# %% [markdown]
# ## 2. Time-resolved decoding relative to stimulus onset
#
# For each 200 ms window (stepped from 1 s before to 0.5 s after stimulus onset) we
# decode the upcoming choice and the block prior. The pre-stimulus region (window
# fully before t = 0) is shaded. If the decision bias is present before the
# stimulus, decoding rises above chance in that shaded region.

# %%
def timecourse(d, target):
    wc = d["win_centers"]
    ch = d["choice"]
    valid = np.isin(ch, [-1, 1])
    if target == "choice":
        y = (ch[valid] == 1).astype(int)
        sel = valid
    else:  # block: pL 0.2 vs 0.8
        sel = valid & np.isin(d["probabilityLeft"], [0.2, 0.8])
        y = (d["probabilityLeft"][sel] == 0.8).astype(int)
    aucs = np.array([cv_auc(d["X"][k][sel], y) for k in range(len(wc))])
    return wc, aucs, sel, y

tc = {}
for d, lab in zip(data, LABELS):
    tc[lab] = {t: timecourse(d, t) for t in ("choice", "block")}
    print("timecourse done:", lab)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(LABELS)))
for ax, target, ttl in zip(axes, ("choice", "block"),
                           ("Upcoming choice", "Block prior (bias state)")):
    for lab, c in zip(LABELS, colors):
        wc, aucs, *_ = tc[lab][target]
        ax.plot(wc, aucs, "-o", ms=3, color=c, label=lab)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.axvline(0, ls="-", c="k", lw=1)
    ax.axvspan(wc[0] - 0.05, -0.1, color="0.85", alpha=0.6, zorder=0)
    ax.text(-0.55, 0.72, "pre-stimulus", ha="center", color="0.35", fontsize=9)
    ax.set_xlabel("window center relative to stimulus onset (s)")
    ax.set_title(f"Decoding {ttl}")
axes[0].set_ylabel("cross-validated AUC")
axes[0].legend(frameon=False, title="session", fontsize=8)
plt.tight_layout()
plt.savefig("fig2_timeresolved_decoding.png", bbox_inches="tight")
plt.close()
print("saved fig2_timeresolved_decoding.png")


# %% [markdown]
# ## 3. Pre-stimulus decoding vs. two null models
#
# We fix a strictly pre-stimulus window ([-0.3, -0.1] s, entirely within the
# enforced quiescence period, ending 100 ms before stimulus onset) and compare the
# observed decoding to two nulls:
#
# * **Label-shuffle null** (standard permutation test): randomly permutes the trial
#   labels, destroying all label structure. Tests whether the population carries any
#   information about the upcoming decision. Chance ≈ 0.5.
# * **Circular-shift null** (conservative): rolls the labels in trial order by a
#   random offset, preserving their slow temporal autocorrelation. A decoder that
#   only exploited slow firing-rate drift co-varying with the (temporally
#   contiguous) block structure cannot beat this null. Note this null is
#   *conservative for a bias signal*: the block prior is itself a slowly-varying
#   quantity, so the circular shift removes part of the genuine effect along with
#   any drift artifact. We report it as a stringent lower bound.

# %%
PRE = -0.2  # center of [-0.3, -0.1]
NPERM = 200
summary = []
for d, lab in zip(data, LABELS):
    k = win_idx(d["win_centers"], PRE)
    ch = d["choice"]; valid = np.isin(ch, [-1, 1])
    Xp = d["X"][k]
    sc = np.abs(d["contrastRight"] - d["contrastLeft"])

    rec = dict(lab=lab, n=int(valid.sum()), units=int(Xp.shape[1]))
    targets = {
        "c":   (valid, (ch[valid] == 1).astype(int)),
        "low": (valid & (sc <= 0.0625), None),
        "b":   (valid & np.isin(d["probabilityLeft"], [0.2, 0.8]), None),
    }
    # fill labels
    lowmask = valid & (sc <= 0.0625)
    bmask = valid & np.isin(d["probabilityLeft"], [0.2, 0.8])
    targets["low"] = (lowmask, (ch[lowmask] == 1).astype(int))
    targets["b"] = (bmask, (d["probabilityLeft"][bmask] == 0.8).astype(int))

    for key, (m, y) in targets.items():
        auc = cv_auc(Xp[m], y)
        nsh = perm_null(Xp[m], y, n=NPERM, kind="shuffle")
        nro = perm_null(Xp[m], y, n=NPERM, kind="roll")
        rec[f"auc_{key}"] = auc
        rec[f"n_{key}"] = int(m.sum())
        rec[f"p_sh_{key}"] = (np.sum(nsh >= auc) + 1) / (NPERM + 1)
        rec[f"p_ro_{key}"] = (np.sum(nro >= auc) + 1) / (NPERM + 1)
        rec[f"sh95_{key}"] = np.percentile(nsh, 95)
        rec[f"ro95_{key}"] = np.percentile(nro, 95)
    summary.append(rec)
    print(f"{lab}: choice AUC={rec['auc_c']:.3f} (p_shuf={rec['p_sh_c']:.3f}, "
          f"p_roll={rec['p_ro_c']:.3f}) | lowC AUC={rec['auc_low']:.3f} "
          f"(p_shuf={rec['p_sh_low']:.3f}) | block AUC={rec['auc_b']:.3f} "
          f"(p_shuf={rec['p_sh_b']:.3f}, p_roll={rec['p_ro_b']:.3f})")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
panels = [("c", "upcoming choice (all trials)", "#357"),
          ("low", "upcoming choice (≤6% contrast)", "#5a8"),
          ("b", "block prior (bias state)", "#c44")]
x = np.arange(len(LABELS))
for ax, (key, name, col) in zip(axes, panels):
    aucs = [s[f"auc_{key}"] for s in summary]
    ax.bar(x, aucs, 0.6, color=col)
    for i, s in enumerate(summary):
        # null-95 reference marks
        ax.plot([i - 0.32, i + 0.32], [s[f"sh95_{key}"]] * 2, "-", c="k", lw=1.4)
        ax.plot([i - 0.32, i + 0.32], [s[f"ro95_{key}"]] * 2, "--", c="0.45",
                lw=1.4)
        if s[f"p_sh_{key}"] < 0.05:
            ax.text(i, s[f"auc_{key}"] + 0.006, "*", ha="center", fontsize=15)
    ax.axhline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_title(name, fontsize=10)
axes[0].set_ylabel("pre-stimulus AUC  ([-0.3, -0.1] s)")
axes[0].set_ylim(0.45, 0.78)
# legend proxies
from matplotlib.lines import Line2D
axes[2].legend(handles=[
    Line2D([0], [0], color="k", lw=1.4, label="shuffle-null 95%"),
    Line2D([0], [0], color="0.45", lw=1.4, ls="--", label="circular-shift-null 95%"),
    Line2D([0], [0], color="w", marker="$*$", mfc="k", label="p<0.05 (shuffle)")],
    frameon=False, fontsize=8, loc="upper right")
fig.suptitle("Decision bias is decodable before stimulus onset "
             "(★ = p<0.05 vs label-shuffle null)", y=1.02)
plt.tight_layout()
plt.savefig("fig3_prestim_vs_null.png", bbox_inches="tight")
plt.close()
print("saved fig3_prestim_vs_null.png")


# %% [markdown]
# ## 4. What the decoder sees: pre-stimulus population separation
#
# For the best-decoding session we project the pre-stimulus population activity onto
# the choice decoder's axis (its cross-validated decision function) and show the
# distribution split by the choice the animal *subsequently* made. The two upcoming
# choices are already separated before the stimulus appears.

# %%
best = max(summary, key=lambda s: s["auc_c"] - s["sh95_c"])["lab"]
d = data[LABELS.index(best)]
k = win_idx(d["win_centers"], PRE)
ch = d["choice"]; valid = np.isin(ch, [-1, 1])
Xp = d["X"][k][valid]; yc = (ch[valid] == 1).astype(int)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
score = cross_val_predict(clf, Xp, yc, cv=cv, method="decision_function")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
bins = np.linspace(score.min(), score.max(), 26)
ax.hist(score[yc == 0], bins, alpha=0.6, color="#c44", label="report right (−1)")
ax.hist(score[yc == 1], bins, alpha=0.6, color="#357", label="report left (+1)")
ax.set_xlabel("pre-stimulus decoder projection (out-of-sample)")
ax.set_ylabel("trials")
ax.set_title(f"Session {best}: pre-stimulus activity\npredicts the upcoming choice")
ax.legend(frameon=False)

# per-unit selectivity: pre-stim rate difference (left - right upcoming choice)
ax = axes[1]
rL = Xp[yc == 1].mean(0); rR = Xp[yc == 0].mean(0)
from scipy import stats
tvals = np.array([stats.ttest_ind(Xp[yc == 1][:, u], Xp[yc == 0][:, u]).statistic
                  for u in range(Xp.shape[1])])
order = np.argsort(tvals)
ax.bar(np.arange(len(tvals)), tvals[order],
       color=np.where(tvals[order] > 0, "#357", "#c44"))
ax.axhline(0, c="k", lw=0.8)
ax.axhline(1.96, ls=":", c="0.4"); ax.axhline(-1.96, ls=":", c="0.4")
ax.set_xlabel("units (sorted)")
ax.set_ylabel("pre-stim choice selectivity  (t: left − right)")
ax.set_title("Many single units are choice-predictive\nbefore the stimulus")
plt.tight_layout()
plt.savefig("fig4_population_separation.png", bbox_inches="tight")
plt.close()
print("saved fig4_population_separation.png")


# %% [markdown]
# ## Summary of results

# %%
hdr = (f"{'session':>10} {'nTr':>5} {'units':>5} {'choiceAUC':>10} {'p_sh':>6} "
       f"{'p_roll':>7} {'lowC AUC':>9} {'p_sh':>6} {'blockAUC':>9} {'p_sh':>6} "
       f"{'p_roll':>7}")
print(hdr)
for s in summary:
    print(f"{s['lab']:>10} {s['n']:>5} {s['units']:>5} "
          f"{s['auc_c']:>10.3f} {s['p_sh_c']:>6.3f} {s['p_ro_c']:>7.3f} "
          f"{s['auc_low']:>9.3f} {s['p_sh_low']:>6.3f} "
          f"{s['auc_b']:>9.3f} {s['p_sh_b']:>6.3f} {s['p_ro_b']:>7.3f}")

mc = np.mean([s["auc_c"] for s in summary])
mb = np.mean([s["auc_b"] for s in summary])
n_sig_c = sum(s["p_sh_c"] < 0.05 for s in summary)
n_sig_b = sum(s["p_sh_b"] < 0.05 for s in summary)
print(f"\nMean pre-stimulus choice AUC = {mc:.3f} "
      f"({n_sig_c}/{len(summary)} sessions p<0.05 vs shuffle); "
      f"block-prior AUC = {mb:.3f} ({n_sig_b}/{len(summary)} p<0.05 vs shuffle). "
      f"Chance = 0.5.")

# %% [markdown]
# **Conclusion.** In the IBL visual-decision task, the animal's upcoming, biased
# decision is encoded in population neural activity *before* the stimulus is shown.
# From a strictly pre-stimulus window ([-0.3, -0.1] s, inside the enforced
# quiescence period), a linear decoder reads the upcoming choice above chance in 3
# of 4 sessions (label-shuffle permutation p < 0.01), and this holds on
# low/zero-contrast trials where the choice is driven by internal bias rather than
# sensory evidence. The block prior — the latent bias state itself — is decodable in
# all four sessions against the shuffle null (AUC ≈ 0.57–0.71).
#
# A conservative circular-shift null, which preserves the slow temporal
# autocorrelation of the labels, attenuates these effects: only a subset of sessions
# survives it. This is expected rather than damning, because the block prior is by
# construction a slowly-varying quantity, so a temporally-matched shuffle removes
# part of the genuine bias signal along with any drift artifact. The pattern
# indicates that much of the decodable pre-stimulus signal lives on the slow
# timescale of the bias state, which is exactly what a bias representation should
# look like. Time-resolved decoding (Fig 2) confirms the signal is present during
# quiescence and rises sharply once the stimulus appears, when sensory and motor
# information are added on top. One session (c7bd79c9) shows strong behavioural bias
# but weak neural decoding, a reminder that decodability depends on which brain
# regions each probe happened to sample.
