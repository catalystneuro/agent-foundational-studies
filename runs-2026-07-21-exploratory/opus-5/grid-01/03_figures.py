"""Figures demonstrating grid cells in MEC from the DANDI:000582 population."""

import itertools
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

import gridlib as G

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 130})

R = pickle.load(open("results.pkl", "rb"))
df = pd.DataFrame([{k: v for k, v in r.items() if k != "shuffle"} for r in R["rows"]])
inc = df[df.included].reset_index(drop=True)
shuffles = {(r["session"], r["unit"]): r["shuffle"]
            for r in R["rows"] if r["included"]}
pooled = np.concatenate(list(shuffles.values()))
THR = float(np.nanpercentile(pooled, 95))
inc["is_grid"] = inc.grid_score > THR
inc["big_arena"] = inc.arena_cm > 120
maps = R["maps"]

# head-direction shuffle control (02b)
hd_rows = pickle.load(open("hd_shuffles.pkl", "rb"))
hd_pooled = np.concatenate([r["mvl_shuffle"] for r in hd_rows])
MVL_THR = float(np.nanpercentile(hd_pooled, 95))
hd_map = {(r["session"], r["unit"]): r["mvl"] for r in hd_rows}
inc["mvl"] = [hd_map.get((s_, u_), np.nan) for s_, u_ in zip(inc.session, inc.unit)]
inc["is_hd"] = inc.mvl > MVL_THR

print(f"units total {len(df)}, analysed {len(inc)}, sessions {df.session.nunique()}, "
      f"rats {df.subject.nunique()}")
print(f"gridness threshold (95th pct of {len(pooled)} shuffles) = {THR:.3f}")
print(f"grid cells = {inc.is_grid.sum()} / {len(inc)} "
      f"({100 * inc.is_grid.mean():.1f}%)")
print(f"MVL threshold = {MVL_THR:.3f}")


def show_map(ax, m, edges=None, title=None, robust=True):
    """Rate map with a robust colour ceiling so weaker fields stay visible."""
    m = np.asarray(m, dtype=float)
    vmax = np.nanpercentile(m, 97) if robust else np.nanmax(m)
    kw = {}
    if edges is not None:
        kw["extent"] = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
    ax.imshow(np.ma.masked_invalid(m).T, origin="lower", cmap="jet", vmin=0,
              vmax=vmax, interpolation="nearest", **kw)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, pad=3)


def show_ac(ax, ac, title=None):
    ax.imshow(np.ma.masked_invalid(np.asarray(ac, dtype=float)).T, origin="lower",
              cmap="jet", vmin=-0.6, vmax=1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, pad=3)


def load_unit(row):
    asset = [a for a in G.load_assets() if a["path"] == row.session][0]
    s = G.load_session(asset)
    ep = G.run_epochs(s)
    st = s["units"][list(s["units"].keys())[s["unit_names"].index(row.unit)]]
    return s, ep, st


# ---------------------------------------------------------------------------
# Figure 2: example grid cells, one per rat
# ---------------------------------------------------------------------------
best = (inc[inc.is_grid].sort_values("grid_score", ascending=False)
        .groupby("subject").head(1).sort_values("grid_score", ascending=False)
        .head(6))

fig, axes = plt.subplots(3, len(best), figsize=(2.1 * len(best), 6.8))
for j, (_, row) in enumerate(best.iterrows()):
    d = maps[(row.session, row.unit)]
    s, ep, st = load_unit(row)
    sp = st.restrict(ep).value_from(s["position"])
    p = s["position"].restrict(ep)

    ax = axes[0, j]
    ax.plot(p["x"], p["y"], lw=0.25, color="0.75")
    ax.scatter(sp["x"], sp["y"], s=2.5, c="crimson", linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"rat {row.subject} {row.unit}\n{row.histology}, "
                 f"{int(row.arena_cm)} cm arena", pad=3)
    if j == 0:
        ax.set_ylabel("trajectory\n+ spikes")

    show_map(axes[1, j], d["rate_map"], d["edges"],
             f"peak {row.peak_rate:.1f} Hz | mean {row.mean_rate:.1f} Hz")
    if j == 0:
        axes[1, j].set_ylabel("rate map")

    show_ac(axes[2, j], d["autocorr"],
            f"gridness {row.grid_score:.2f}\nspacing {row.spacing_cm:.0f} cm")
    if j == 0:
        axes[2, j].set_ylabel("autocorrelogram")
    s["io"].close()

fig.suptitle("Grid cells in rat medial entorhinal cortex "
             "(DANDI:000582, Sargolini et al. 2006)", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig02_example_grid_cells.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig02_example_grid_cells.png")

# ---------------------------------------------------------------------------
# Figure 3: anatomy of the gridness measure (example from a 1 m box)
# ---------------------------------------------------------------------------
# a cell from a 1 m box whose spacing is small enough to show several fields
row = (inc[inc.is_grid & ~inc.big_arena & (inc.spacing_cm < 55)]
       .sort_values("grid_score", ascending=False).iloc[0])
d = maps[(row.session, row.unit)]
ac = d["autocorr"].astype(float)
r_in, r_out = int(row.r_in), int(row.r_out)
angles, prof = G.rotational_profile(ac, r_in, r_out)
spacing, orientation, pk = G.grid_geometry(ac)
s, ep, st = load_unit(row)
sp = st.restrict(ep).value_from(s["position"])
p = s["position"].restrict(ep)
s["io"].close()

fig, axes = plt.subplots(1, 5, figsize=(15, 3.4),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1.45, 1.45]})
axes[0].plot(p["x"], p["y"], lw=0.25, color="0.75")
axes[0].scatter(sp["x"], sp["y"], s=3, c="crimson", linewidths=0)
axes[0].set(aspect="equal", xticks=[], yticks=[],
            title=f"rat {row.subject} {row.unit}\n{len(sp)} spikes while running")

show_map(axes[1], d["rate_map"], d["edges"],
         f"rate map, peak {row.peak_rate:.1f} Hz")

ax = axes[2]
show_ac(ax, ac, f"autocorrelogram, 6 nearest peaks\nannulus {r_in}-{r_out} bins")
cy, cx = (np.array(ac.shape) - 1) / 2.0
for r_, c_ in [(r_in, "w"), (r_out, "k")]:
    ax.add_patch(plt.Circle((cx, cy), r_, fill=False, color=c_, lw=1.2, ls="--"))
ax.scatter(pk[:, 0], pk[:, 1], s=45, facecolors="none", edgecolors="k", lw=1.2)

ax = axes[3]
ax.plot(angles, prof, "-", color="k", lw=1.2)
for a_ in (60, 120):
    ax.axvline(a_, color="tab:green", lw=1, alpha=0.6)
for a_ in (30, 90, 150):
    ax.axvline(a_, color="tab:red", lw=1, alpha=0.6)
ax.set(xlabel="rotation of autocorrelogram (deg)", ylabel="Pearson r",
       xticks=np.arange(0, 181, 30),
       title=f"gridness = min(r60, r120) - max(r30, r90, r150)\n= "
             f"{row.grid_score:.2f}")

ax = axes[4]
sh = shuffles[(row.session, row.unit)]
ax.hist(sh, bins=25, color="0.7", label="shuffled (n=100)")
ax.axvline(row.grid_score, color="crimson", lw=2, label="observed")
ax.axvline(THR, color="k", ls="--", lw=1, label=f"pooled 95th pct = {THR:.2f}")
ax.set(xlabel="gridness score", ylabel="count",
       title=f"within-cell shuffle, p = {row.shuffle_p:.3f}")
ax.legend(fontsize=7)

fig.suptitle("How gridness is measured and tested", y=1.03)
fig.tight_layout()
fig.savefig("fig03_gridness_method.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig03_gridness_method.png")

# ---------------------------------------------------------------------------
# Figure 4: population statistics
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.6))

bins = np.linspace(-1.2, 1.8, 50)
axes[0].hist(pooled, bins=bins, density=True, color="0.75",
             label=f"shuffled ({len(pooled)})")
axes[0].hist(inc.grid_score, bins=bins, density=True, histtype="step", lw=1.8,
             color="crimson", label=f"observed ({len(inc)} cells)")
axes[0].axvline(THR, color="k", ls="--", lw=1.2, label=f"95th pct = {THR:.2f}")
axes[0].set(xlabel="gridness score", ylabel="probability density",
            title="Observed vs. shuffled gridness")
axes[0].legend(fontsize=7)

frac = inc.groupby("subject").is_grid.agg(["mean", "size"]).sort_values("mean")
axes[1].barh(np.arange(len(frac)), 100 * frac["mean"].values, color="steelblue")
axes[1].set_yticks(np.arange(len(frac)))
axes[1].set_yticklabels([f"{i} (n={int(n)})"
                         for i, n in zip(frac.index, frac["size"])], fontsize=7)
axes[1].axvline(100 * inc.is_grid.mean(), color="k", ls="--", lw=1,
                label=f"overall {100 * inc.is_grid.mean():.0f}%")
axes[1].set(xlabel="grid cells (% of units)", ylabel="rat",
            title="Grid cells in 14 of 15 rats")
axes[1].legend(fontsize=7, loc="lower right")

axes[2].scatter(inc.spatial_info, inc.stability, s=12,
                c=np.where(inc.is_grid, "crimson", "0.6"), linewidths=0)
u = stats.mannwhitneyu(inc[inc.is_grid].stability.dropna(),
                       inc[~inc.is_grid].stability.dropna())
axes[2].set(xlabel="spatial information (bits/spike)",
            ylabel="split-half map correlation",
            title=f"Grid cells are informative and stable\n"
                  f"stability {inc[inc.is_grid].stability.median():.2f} vs "
                  f"{inc[~inc.is_grid].stability.median():.2f}, "
                  f"p = {u.pvalue:.1e}")
for lab, sub, col in [("grid", inc[inc.is_grid], "crimson"),
                      ("non-grid", inc[~inc.is_grid], "0.6")]:
    axes[2].scatter([], [], c=col, label=f"{lab} (n={len(sub)})")
axes[2].legend(fontsize=7, loc="lower right")

lay = inc.assign(layer=inc.histology.str.replace("MEC ", "", regex=False))
lay = lay[lay.layer.str.startswith("L")]
tab = lay.groupby("layer").is_grid.agg(["size", "sum", "mean"]).sort_index()
axes[3].bar(np.arange(len(tab)), 100 * tab["mean"].values, color="steelblue")
axes[3].set_xticks(np.arange(len(tab)))
axes[3].set_xticklabels([f"{i}\n(n={int(n)})" for i, n in zip(tab.index, tab["size"])])
axes[3].set(xlabel="MEC layer (histology field)", ylabel="grid cells (%)",
            title="Grid cells are most common in the\nsuperficial layers")
chi = stats.chi2_contingency(np.stack([tab["sum"].values,
                                       (tab["size"] - tab["sum"]).values]))
axes[3].text(0.40, 0.97, f"chi2 p = {chi.pvalue:.1e}", transform=axes[3].transAxes,
             fontsize=7, va="top")

fig.tight_layout()
fig.savefig("fig04_population_statistics.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig04_population_statistics.png")

# ---------------------------------------------------------------------------
# Figure 5: geometry of the grids
# ---------------------------------------------------------------------------
gc = inc[inc.is_grid]
small = gc[~gc.big_arena]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

ax = axes[0]
bb = np.arange(15, 135, 5)
ax.hist([small.spacing_cm, gc[gc.big_arena].spacing_cm], bins=bb, stacked=True,
        color=["steelblue", "orange"], edgecolor="w",
        label=[f"~1 m box (n={len(small)})",
               f"large arena (n={int(gc.big_arena.sum())})"])
ax.axvline(small.spacing_cm.median(), color="k", ls="--",
           label=f"median, 1 m box = {small.spacing_cm.median():.0f} cm")
ax.set(xlabel="grid spacing (cm)", ylabel="grid cells",
       title="Spacing of the hexagonal lattice")
ax.legend(fontsize=7)

# module structure: simultaneously recorded grid cells share orientation
same_o, same_s = [], []
for _, sub in gc.groupby("session"):
    o = sub.orientation_deg.dropna().values
    sp_ = sub.spacing_cm.dropna().values
    for a_, b_ in itertools.combinations(o, 2):
        dd = abs(a_ - b_) % 60
        same_o.append(min(dd, 60 - dd))
    for a_, b_ in itertools.combinations(sp_, 2):
        same_s.append(max(a_, b_) / min(a_, b_))
rng = np.random.default_rng(0)
allo = gc.orientation_deg.dropna().values
alls = gc.spacing_cm.dropna().values
rand_o = [min(d_ % 60, 60 - d_ % 60) for d_ in
          np.abs(rng.choice(allo, 5000) - rng.choice(allo, 5000))]
pair = np.stack([rng.choice(alls, 5000), rng.choice(alls, 5000)])
rand_s = pair.max(0) / pair.min(0)

ax = axes[1]
ax.hist([same_o, rand_o], bins=np.arange(0, 31, 2), density=True,
        color=["crimson", "0.7"],
        label=[f"same session (n={len(same_o)})", "random pairs"])
u_o = stats.mannwhitneyu(same_o, rand_o)
ax.set(xlabel="|Δ orientation| between grid cells (deg, mod 60)",
       ylabel="probability density",
       title=f"Cells recorded together share orientation\n"
             f"median {np.median(same_o):.1f}° vs {np.median(rand_o):.1f}°, "
             f"p = {u_o.pvalue:.1e}")
ax.legend(fontsize=7)

ax = axes[2]
sub = small.dropna(subset=["depth_m"])
ax.scatter(1e3 * sub.depth_m, sub.spacing_cm, s=16, c="crimson", linewidths=0)
r_depth, p_depth = stats.pearsonr(sub.depth_m, sub.spacing_cm)
b_ = np.polyfit(1e3 * sub.depth_m, sub.spacing_cm, 1)
xs = np.linspace(1e3 * sub.depth_m.min(), 1e3 * sub.depth_m.max(), 10)
ax.plot(xs, np.polyval(b_, xs), "k--", lw=1,
        label=f"r = {r_depth:.2f}, p = {p_depth:.1e}")
ax.legend(fontsize=7)
ax.set(xlabel="electrode depth below dura (mm)", ylabel="grid spacing (cm)",
       title=f"Dorsoventral gradient (1 m box only, n={len(sub)})")

fig.tight_layout()
fig.savefig("fig05_grid_geometry.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig05_grid_geometry.png")
print(f"  spacing 1 m box: median {small.spacing_cm.median():.1f} cm, "
      f"IQR {small.spacing_cm.quantile(.25):.0f}-{small.spacing_cm.quantile(.75):.0f}")
print(f"  within-session spacing ratio {np.median(same_s):.2f} vs "
      f"{np.median(rand_s):.2f} for random pairs")

# ---------------------------------------------------------------------------
# Figure 6: conjunctive grid x head-direction cells
# ---------------------------------------------------------------------------
hd_df = inc.dropna(subset=["mvl"])
conj = hd_df[hd_df.is_grid & hd_df.is_hd].sort_values("mvl", ascending=False)
print(f"{len(hd_df)} units with HD tracking; {int(hd_df.is_grid.sum())} grid cells; "
      f"{len(conj)} conjunctive (MVL > {MVL_THR:.2f})")

fig = plt.figure(figsize=(13.5, 3.9))
gsp = fig.add_gridspec(1, 4, wspace=0.4)
ax = fig.add_subplot(gsp[0, 0])
ax.scatter(hd_df.mvl, hd_df.grid_score, s=12,
           c=np.where(hd_df.is_grid & hd_df.is_hd, "crimson",
                      np.where(hd_df.is_grid, "tab:orange", "0.6")), linewidths=0)
ax.axhline(THR, color="k", ls="--", lw=1)
ax.axvline(MVL_THR, color="k", ls=":", lw=1)
ax.set(xlabel="head-direction mean vector length", ylabel="gridness score",
       title=f"{len(conj)} of {int(hd_df.is_grid.sum())} grid cells are also\n"
             f"direction-tuned (thresholds from shuffles)")

examples = conj[conj.n_spikes > 500].head(3)
for j, (_, r_) in enumerate(examples.iterrows()):
    s, ep, st = load_unit(r_)
    curve, centers, mvl, pref = G.hd_tuning(st, s["hd"], ep)
    sm_ = ndimage.gaussian_filter1d(np.nan_to_num(curve), 1.0, mode="wrap")
    axp = fig.add_subplot(gsp[0, j + 1], projection="polar")
    cc_ = np.append(sm_, sm_[0])
    aa_ = np.append(centers, centers[0])
    axp.plot(aa_, cc_, color="crimson", lw=1.5)
    axp.fill(aa_, cc_, color="crimson", alpha=0.2)
    axp.set_title(f"rat {r_.subject} {r_.unit}\ngridness {r_.grid_score:.2f}, "
                  f"MVL {mvl:.2f}", pad=24, fontsize=8)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(285)
    s["io"].close()

fig.suptitle("Grid cells in MEC are often conjunctive with head direction "
             "(polar axes show firing rate in Hz)", y=1.05)
fig.savefig("fig06_conjunctive_cells.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig06_conjunctive_cells.png")

# ---------------------------------------------------------------------------
# Figure 7: gallery of the strongest grid cells
# ---------------------------------------------------------------------------
top = gc.sort_values("grid_score", ascending=False).head(24)
fig, axes = plt.subplots(6, 8, figsize=(14, 11.5))
for j, (_, r_) in enumerate(top.iterrows()):
    d = maps[(r_.session, r_.unit)]
    show_map(axes[j // 4, (j % 4) * 2], d["rate_map"], d["edges"],
             f"{r_.subject}/{r_.unit}\npeak {r_.peak_rate:.0f} Hz")
    show_ac(axes[j // 4, (j % 4) * 2 + 1], d["autocorr"],
            f"g = {r_.grid_score:.2f}\n{r_.spacing_cm:.0f} cm")
fig.suptitle("Rate map and spatial autocorrelogram of the 24 highest-gridness "
             "cells (24 of 193)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("fig07_grid_cell_gallery.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig07_grid_cell_gallery.png")

# ---------------------------------------------------------------------------
inc.drop(columns=["included"]).to_csv("unit_metrics.csv", index=False)
summary = dict(
    sessions=int(df.session.nunique()), rats=int(df.subject.nunique()),
    units_total=int(len(df)), units_analysed=int(len(inc)),
    gridness_threshold=round(THR, 3), grid_cells=int(inc.is_grid.sum()),
    pct_grid=round(100 * inc.is_grid.mean(), 1),
    rats_with_grid_cells=int((inc.groupby("subject").is_grid.sum() > 0).sum()),
    median_spacing_1m_box=round(float(small.spacing_cm.median()), 1),
    spacing_iqr_1m_box=[round(float(small.spacing_cm.quantile(.25)), 1),
                        round(float(small.spacing_cm.quantile(.75)), 1)],
    median_gridness_grid=round(float(gc.grid_score.median()), 2),
    median_stability_grid=round(float(gc.stability.median()), 2),
    median_stability_nongrid=round(float(inc[~inc.is_grid].stability.median()), 2),
    grid_cells_by_layer={k: f"{int(v)}/{int(n)}" for k, v, n
                         in zip(tab.index, tab["sum"], tab["size"])},
    halfsplit_gridness_r=round(float(stats.pearsonr(
        *gc[["grid_score_h1", "grid_score_h2"]].dropna().values.T)[0]), 2),
    within_session_dtheta=round(float(np.median(same_o)), 1),
    random_pair_dtheta=round(float(np.median(rand_o)), 1),
    depth_spacing_r_1m_box=round(float(r_depth), 2),
    mvl_threshold=round(MVL_THR, 3),
    units_with_hd=int(len(hd_df)), conjunctive_cells=int(len(conj)),
)
pd.Series(summary).to_json("summary.json", indent=1)
print(summary)
