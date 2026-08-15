"""Step 5: repeat the analysis across all 8 hc-11 sessions (4 rats) and pool."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

import precession_lib as pl
from precession_03 import analyse_session

SESSION_LIST = pl.LINEAR_SESSIONS

all_df, pooled_xy, nulls = [], [], []
for s in tqdm(SESSION_LIST, desc="sessions"):
    df, ctx = analyse_session(s, verbose=False)
    all_df.append(df)
    for (uid, d), fd in ctx["per_field"].items():
        nulls.append(fd["null"])
        if fd["pval"] < 0.05:
            pooled_xy.append((fd["x"], fd["phi"]))
    n_pc = int(df.is_place_cell.sum())
    n_sig = int((df.is_place_cell & (df.pval < 0.05)).sum())
    print(f"  {s}: {n_pc} place fields, {n_sig} significant "
          f"({100*n_sig/max(n_pc,1):.0f}%), median slope "
          f"{df[df.is_place_cell].slope_deg.median():.0f} deg/field")

df = pd.concat(all_df, ignore_index=True)
df.to_csv("precession_all_sessions.csv", index=False)
pc = df[df.is_place_cell].copy()
pc["subject"] = pc.session.str.split("-").str[0]
sig = pc.pval < 0.05

print("\n" + "=" * 70)
print(f"{len(pc)} place fields from {pc.session.nunique()} sessions / "
      f"{pc.subject.nunique()} rats")
print(f"significant precession: {sig.sum()} ({100*sig.mean():.1f}%)")
print(f"negative slope: {(pc.slope_cycles < 0).mean()*100:.1f}% of all fields, "
      f"{(pc.slope_cycles[sig] < 0).mean()*100:.1f}% of significant fields")
print(f"median slope {pc.slope_deg.median():.0f} deg/field "
      f"[IQR {pc.slope_deg.quantile(.25):.0f}, {pc.slope_deg.quantile(.75):.0f}]")
print(f"median rho {pc.rho.median():.3f}")

# sign test against the null that slopes are equally likely +/-
from scipy.stats import binomtest, wilcoxon

bt = binomtest((pc.slope_cycles < 0).sum(), len(pc), 0.5, alternative="greater")
wt = wilcoxon(pc.rho, alternative="less")
print(f"sign test (slopes negative): p = {bt.pvalue:.3e}")
print(f"Wilcoxon signed-rank on rho < 0: p = {wt.pvalue:.3e}")
print("=" * 70)

# ===========================================================================
# Figure 7: cross-session summary
# ===========================================================================
fig = plt.figure(figsize=(15, 9))
gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.42)

# per-session fraction significant
ax = fig.add_subplot(gs[0, 0])
g = pc.groupby("session").apply(
    lambda d: pd.Series({"n": len(d), "frac_sig": (d.pval < 0.05).mean(),
                         "median_slope": d.slope_deg.median()}), include_groups=False)
g = g.loc[SESSION_LIST]
colors = {"Achilles": "tab:blue", "Cicero": "tab:orange", "Gatsby": "tab:green",
          "Buddy": "tab:purple"}
ax.barh(range(len(g)), g.frac_sig * 100,
        color=[colors[s.split("-")[0]] for s in g.index])
ax.set_yticks(range(len(g)))
ax.set_yticklabels([f"{s}\n(n={int(n)})" for s, n in zip(g.index, g.n)], fontsize=8)
ax.set_xlabel("% of place fields with significant precession")
ax.set_title("Consistency across sessions and rats")
ax.invert_yaxis()

# per-session median slope
ax = fig.add_subplot(gs[0, 1])
for s, d in pc.groupby("session"):
    ax.scatter(np.full(len(d), list(g.index).index(s)) + np.random.uniform(-.18, .18, len(d)),
               d.slope_deg, s=12, alpha=0.45, color=colors[s.split("-")[0]])
ax.plot(range(len(g)), g.median_slope, "k_", ms=22, mew=2.5)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(g)))
ax.set_xticklabels([s.split("-")[0][:4] + "\n" + s.split("-")[1][:4] for s in g.index],
                   fontsize=7)
ax.set_ylabel("slope (deg per field traversal)")
ax.set_title("Slope distribution per session\n(black bars = medians)")

# pooled slope histogram
ax = fig.add_subplot(gs[0, 2])
ax.hist(pc.slope_deg, bins=np.arange(-540, 570, 45), color="0.35", edgecolor="w")
ax.axvline(0, color="k", lw=1.5)
ax.axvline(pc.slope_deg.median(), color="tab:red", lw=2)
ax.set_xlabel("slope (deg per field traversal)")
ax.set_ylabel("# place fields")
ax.set_title(f"All {len(pc)} fields\nmedian {pc.slope_deg.median():.0f}$\\degree$, "
             f"sign test p = {bt.pvalue:.1e}")

# pooled phase-position density
ax = fig.add_subplot(gs[1, :3])
X = np.concatenate([x for x, _ in pooled_xy])
P = np.degrees(np.concatenate([p for _, p in pooled_xy]))
H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360],
                           bins=[np.linspace(0, 1, 26), np.linspace(-180, 540, 49)])
H = H / H.sum(axis=1, keepdims=True)
im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
               extent=[0, 1, -180, 540], interpolation="bilinear")
# circular mean phase per position bin, plotted on top
xc = 0.5 * (xe[1:] + xe[:-1])
mp = []
for i in range(len(xc)):
    m = (X >= xe[i]) & (X < xe[i + 1])
    mp.append(np.degrees(np.angle(np.exp(1j * np.radians(P[m])).mean())) if m.sum() else np.nan)
mp = np.degrees(np.unwrap(np.radians(np.array(mp))))
for off in (-360, 0, 360):
    ax.plot(xc, mp + off, "o-", color="cyan", ms=4, lw=1.5)
ax.set_ylim(-180, 540)
ax.set_xlabel("normalised position in field (0 = entry, 1 = exit)")
ax.set_ylabel("theta phase (deg)")
ax.set_yticks([-180, 0, 180, 360, 540])
ax.set_title(f"Pooled spikes from {len(pooled_xy)} significant fields across all sessions "
             f"({len(X):,} spikes)\ncyan = circular mean phase per position bin; "
             "phase axis repeated over two cycles")
fig.colorbar(im, ax=ax, label="P(phase | position)")

# rho by subject
ax = fig.add_subplot(gs[1, 3])
subs = sorted(pc.subject.unique())
data = [pc.rho[pc.subject == s].values for s in subs]
bp = ax.boxplot(data, labels=[f"{s}\n(n={len(d)})" for s, d in zip(subs, data)],
                patch_artist=True, showfliers=False)
for patch, s in zip(bp["boxes"], subs):
    patch.set_facecolor(colors[s])
    patch.set_alpha(0.6)
ax.axhline(0, color="k", lw=1)
ax.set_ylabel("circular-linear correlation $\\rho$")
ax.set_title("Every rat shows negative $\\rho$")
ax.tick_params(axis="x", labelsize=8)

# observed rho against the pooled within-field shuffle null
ax = fig.add_subplot(gs[0, 3])
nullcat = np.concatenate(nulls)
bins_r = np.linspace(-0.8, 0.8, 41)
ax.hist(nullcat, bins=bins_r, density=True, color="0.75", label="shuffled\n(phase-position\npairing broken)")
ax.hist(pc.rho, bins=bins_r, density=True, histtype="step", color="tab:red", lw=2,
        label="observed")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("circular-linear correlation $\\rho$")
ax.set_ylabel("density")
ax.set_title("Observed vs shuffle null")
ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Theta phase precession in CA1 place cells across 5 linear-track sessions from 4 rats "
             "(DANDI:000044, Grosmark & Buzsáki hc-11)", fontsize=13)
fig.savefig("fig07_multisession_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)

np.savez("pooled_phase_position.npz", x=X, phi=P)
print("wrote fig07_multisession_summary.png, precession_all_sessions.csv")
