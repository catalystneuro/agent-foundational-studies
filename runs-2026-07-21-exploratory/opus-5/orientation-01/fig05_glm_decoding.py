"""Figure 5: GLM encoding and population decoding of grating orientation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import dandi_io as dio
import plotting as pl
import tuning as tn
from run_glm_decode import session_trials

glm = pd.read_pickle("results_glm.pkl")
dec = pd.read_pickle("results_decoding.pkl")
p = pd.read_pickle("results_pooled_units.pkl")

fig = plt.figure(figsize=(13.5, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.34, left=0.07, right=0.96, top=0.89, bottom=0.09)

# --- A: pseudo-R2 with and without the orientation term -------------------
ax = fig.add_subplot(gs[0, 0])
d = glm[glm.region_group == "visual cortex"]
lim = (-0.01, np.nanpercentile(glm.pseudo_r2_full, 99.5))
for grp in ["hippocampus", "visual thalamus", "visual cortex"]:
    g = glm[glm.region_group == grp]
    ax.scatter(g.pseudo_r2_reduced, g.pseudo_r2_full, s=5, alpha=0.4, lw=0,
               color=pl.REGION_COLORS[grp], label=grp)
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("cross-validated pseudo-R²\n(spatial frequency + running only)")
ax.set_ylabel("cross-validated pseudo-R²\n(+ orientation)")
ax.legend(fontsize=7.5, markerscale=2.5, loc="lower right")
ax.set_title("A  Poisson GLM: adding orientation\nimproves held-out prediction", loc="left")

# --- B: orientation contribution by region --------------------------------
ax = fig.add_subplot(gs[0, 1])
for i, grp in enumerate(pl.REGION_ORDER):
    v = glm.loc[glm.region_group == grp, "d_ll_orientation"].dropna()
    if not len(v):
        continue
    parts = ax.violinplot(np.clip(v, -0.02, 0.06), positions=[i], widths=0.75,
                          showmedians=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(pl.REGION_COLORS[grp])
        b.set_alpha(0.6)
    parts["cmedians"].set_color("k")
    frac = (v > 0).mean()
    ax.text(i, 0.062, f"{frac:.0%}>0\nn={len(v)}", ha="center", fontsize=7.5)
ax.axhline(0, color="k", ls="--", lw=0.9)
ax.set_xticks(range(3))
ax.set_xticklabels([g.replace(" ", "\n") for g in pl.REGION_ORDER])
ax.set_ylabel("Δ held-out log-likelihood\nfrom orientation (nats / spike)")
ax.set_ylim(-0.025, 0.08)
u = stats.mannwhitneyu(glm.loc[glm.region_group == "visual cortex", "d_ll_orientation"].dropna(),
                       glm.loc[glm.region_group == "hippocampus", "d_ll_orientation"].dropna())
ax.set_title(f"B  Orientation term contribution\ncortex vs hippocampus p={u.pvalue:.1e}", loc="left")

# --- C: GLM contribution vs single-unit gOSI ------------------------------
ax = fig.add_subplot(gs[0, 2])
merged = glm.merge(p[["unit_id", "session_id", "sg_gOSI"]], on=["unit_id", "session_id"])
mc = merged[merged.region_group == "visual cortex"]
ax.scatter(mc.sg_gOSI, mc.d_ll_orientation, s=5, alpha=0.35, lw=0,
           color=pl.REGION_COLORS["visual cortex"])
ax.set_ylim(-0.02, np.nanpercentile(mc.d_ll_orientation, 99))
r = stats.spearmanr(mc.sg_gOSI, mc.d_ll_orientation, nan_policy="omit")
ax.axhline(0, color="k", ls="--", lw=0.9)
ax.set_xlabel("gOSI (static gratings)")
ax.set_ylabel("Δ held-out log-likelihood (nats / spike)")
ax.set_title(f"C  The two measures agree\ncortex Spearman ρ={r.statistic:.2f} (p={r.pvalue:.1e})",
             loc="left")

# --- D: decoding accuracy vs population size ------------------------------
ax = fig.add_subplot(gs[1, 0])
for grp in pl.REGION_ORDER:
    d = dec[dec.region_group == grp]
    if not len(d):
        continue
    m = d.groupby("size")["accuracy"].agg(["mean", "sem"])
    ax.errorbar(m.index, m["mean"], yerr=m["sem"], color=pl.REGION_COLORS[grp], lw=1.6,
                marker="o", ms=4, capsize=2, label=grp)
    ms = d.groupby("size")["accuracy_shuffled"].mean()
    ax.plot(ms.index, ms.values, color=pl.REGION_COLORS[grp], lw=1.0, ls=":")
ax.axhline(1 / 6, color="k", ls="--", lw=0.9)
ax.text(80, 1 / 6 + 0.012, "chance (1/6)", ha="right", fontsize=7.5)
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 40, 80])
ax.set_xticklabels([5, 10, 20, 40, 80])
ax.set_xlabel("number of simultaneously recorded units")
ax.set_ylabel("6-way orientation decoding accuracy")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_title("D  Population decoding\n(dotted: shuffled-label control)", loc="left")

# --- E: decoding accuracy per session at 40 units -------------------------
ax = fig.add_subplot(gs[1, 1])
d40 = dec[dec["size"] == 40].groupby(["session_id", "region_group"])["accuracy"].mean().reset_index()
for i, grp in enumerate(pl.REGION_ORDER):
    v = d40[d40.region_group == grp]["accuracy"].values
    if not len(v):
        continue
    ax.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, s=28,
               color=pl.REGION_COLORS[grp], zorder=3)
    ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k", lw=1.6, zorder=4)
ax.axhline(1 / 6, color="k", ls="--", lw=0.9)
ax.set_xticks(range(3))
ax.set_xticklabels([g.replace(" ", "\n") for g in pl.REGION_ORDER])
ax.set_ylabel("decoding accuracy, 40 units")
ax.set_title("E  Per-session decoding\n(one point per session)", loc="left")

# --- F: confusion matrix for a cortical population ------------------------
ax = fig.add_subplot(gs[1, 2])
s = dio.extract_session("sub-699733573/sub-699733573_ses-715093703.nwb")
tsg, meta = tn.make_tsgroup(s)
tab, counts, _ = session_trials(s, tsg)
labels = tab["orientation"].values.astype(int)
cols = np.where((meta.region_group == "visual cortex").values & (counts.std(axis=0) > 0))[0]
X = np.sqrt(counts[:, cols])
pred = np.empty_like(labels)
for train, test in StratifiedKFold(5, shuffle=True, random_state=0).split(X, labels):
    sc = StandardScaler().fit(X[train])
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(X[train]), labels[train])
    pred[test] = clf.predict(sc.transform(X[test]))
cm = confusion_matrix(labels, pred, normalize="true")
uo = np.unique(labels)
im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=cm.max())
ax.set_xticks(range(len(uo)))
ax.set_xticklabels(uo)
ax.set_yticks(range(len(uo)))
ax.set_yticklabels(uo)
ax.set_xlabel("decoded orientation (deg)")
ax.set_ylabel("presented orientation (deg)")
plt.colorbar(im, ax=ax, fraction=0.046, label="fraction of trials")
ax.set_title(f"F  Confusion matrix, {len(cols)} cortical units\nsession {s['session_id']}, "
             f"accuracy {np.mean(pred == labels):.2f}", loc="left")

fig.suptitle("Orientation encoding (NeMoS Poisson GLM) and population decoding", y=0.955, fontsize=12)
fig.savefig("fig05_glm_decoding.png", bbox_inches="tight")
print("wrote fig05_glm_decoding.png")
