"""Final figures for the grid-cell demonstration (DANDI:000582)."""
import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gridlib as gl

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 130})

units_df = pd.read_csv("unit_metrics.csv")
shuf_g = pd.read_csv("shuffle_gridness.csv")
shuf_m = pd.read_csv("shuffle_mvl.csv")
G_THR = float(np.nanpercentile(shuf_g.gridness, 95))
M_THR = float(np.nanpercentile(shuf_m.hd_mvl, 95))
units_df["is_grid"] = units_df.gridness > G_THR
units_df["is_hd"] = units_df.hd_mvl > M_THR
print("gridness threshold %.3f, HD mvl threshold %.3f" % (G_THR, M_THR))


@functools.lru_cache(maxsize=8)
def session(path):
    s = gl.load_session(path)
    s["edges"] = gl.map_edges(s["position"])
    s["maps"], s["occ"], _ = gl.rate_maps(s["units"], s["position"], s["run_ep"], s["edges"])
    return s


def unit_index(s, name):
    return list(s["units"].get_info("unit_name")).index(name)


def draw_cell(axes, path, name, label=""):
    """Three stacked panels: trajectory+spikes, rate map, autocorrelogram."""
    s = session(path)
    i = unit_index(s, name)
    pos, ep, edges = s["position"], s["run_ep"], s["edges"]
    ext = [edges[0], edges[-1], edges[0], edges[-1]]
    sp = s["units"][i].restrict(ep).value_from(pos)

    ax = axes[0]
    ax.plot(pos["x"].values, pos["y"].values, lw=0.25, color="0.8")
    ax.plot(sp["x"].values, sp["y"].values, ".", ms=1.6, color="crimson")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=8)

    ax = axes[1]
    ax.imshow(s["maps"][i], origin="lower", extent=ext, cmap="jet")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%.1f Hz" % np.nanmax(s["maps"][i]), fontsize=8)

    ac = gl.spatial_autocorr(s["maps"][i])
    g, _ = gl.gridness(ac)
    spacing, ori, ell, pk = gl.autocorr_peaks(ac)
    ax = axes[2]
    ax.imshow(ac, origin="lower", cmap="jet", vmin=-0.5, vmax=1.0)
    ax.plot(pk[:, 0], pk[:, 1], "o", mfc="none", mec="w", ms=4, mew=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("g = %.2f, %.0f cm" % (g, spacing), fontsize=8)
    return g, spacing


# =============================================================================
# Figure 1 - raw data streams
# =============================================================================
PROTO = "sub-11265/sub-11265_ses-06020601_behavior+ecephys.nwb"
s = session(PROTO)
pos, sp_tsd, units, ep = s["position"], s["speed"], s["units"], s["run_ep"]

fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
ax.plot(pos["x"].values, pos["y"].values, lw=0.3, color="0.4")
ax.set_aspect("equal"); ax.set_title("trajectory (10 min, 1 x 1 m box)")
ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")

ax = fig.add_subplot(gs[0, 1])
seg = pos.get(120, 180)
ax.plot(seg.index, seg["x"].values, lw=0.9, label="x")
ax.plot(seg.index, seg["y"].values, lw=0.9, label="y")
ax.legend(fontsize=7, loc="upper right"); ax.set_xlabel("time (s)"); ax.set_ylabel("cm")
ax.set_title("tracked position")

ax = fig.add_subplot(gs[0, 2])
seg = sp_tsd.get(120, 180)
ax.plot(seg.index, seg.values, lw=0.9, color="k")
ax.axhline(gl.SPEED_MIN, color="r", ls="--", lw=0.8)
ax.set_xlabel("time (s)"); ax.set_ylabel("cm/s")
ax.set_title("running speed (red = %.1f cm/s cut-off)" % gl.SPEED_MIN)

ax = fig.add_subplot(gs[1, :2])
names = list(units.get_info("unit_name"))
for i in range(len(units)):
    t = units[i].get(120, 180).times()
    ax.plot(t, np.full_like(t, i), "|", ms=4, color="k", lw=0.5)
ax.set_yticks(range(len(units))); ax.set_yticklabels(names, fontsize=6)
ax.set_xlabel("time (s)"); ax.set_title("simultaneously recorded MEC units (%s)" % PROTO.split("/")[0])

ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(s["occ"], origin="lower", cmap="viridis",
               extent=[s["edges"][0], s["edges"][-1], s["edges"][0], s["edges"][-1]])
ax.set_aspect("equal"); ax.set_title("occupancy (s per 3 cm bin)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[2, 0])
ax.hist(sp_tsd.values, bins=80, range=(0, 60), color="0.4")
ax.axvline(gl.SPEED_MIN, color="r", ls="--"); ax.set_xlabel("speed (cm/s)"); ax.set_ylabel("samples")
ax.set_title("speed distribution")

ax = fig.add_subplot(gs[2, 1])
ax.hist(np.degrees(s["hd"].values), bins=60, color="0.4")
ax.set_xlabel("head direction (deg)"); ax.set_title("head-direction sampling")

ax = fig.add_subplot(gs[2, 2])
ax.hist(units_df.mean_rate, bins=40, color="0.4")
ax.set_xlabel("mean firing rate (Hz)"); ax.set_ylabel("units")
ax.set_title("all %d units, %d sessions" % (len(units_df), units_df.path.nunique()))

fig.suptitle("DANDI:000582 - raw data streams (Sargolini et al. 2006, MEC + LED tracking)", y=0.98)
fig.savefig("fig01_raw_data_streams.png", bbox_inches="tight")
plt.close(fig)
print("saved fig01_raw_data_streams.png")

# =============================================================================
# Figure 2 - example grid cells, one per animal
# =============================================================================
best = (units_df[units_df.is_grid].sort_values("gridness", ascending=False)
        .drop_duplicates("subject").head(8))
fig, axes = plt.subplots(3, len(best), figsize=(1.55 * len(best), 5.4))
for j, (_, r) in enumerate(best.iterrows()):
    draw_cell(axes[:, j], r.path, r.unit,
              label="rat %s\n%s %s" % (r.subject, r.unit, r.layer.replace("MEC ", "")))
for row, lab in enumerate(["trajectory\n+ spikes", "rate map", "spatial\nautocorrelation"]):
    axes[row, 0].set_ylabel(lab, fontsize=8)
fig.suptitle("Grid cells in medial entorhinal cortex: eight example cells, eight rats", y=1.0)
fig.tight_layout()
fig.savefig("fig02_example_grid_cells.png", bbox_inches="tight")
plt.close(fig)
print("saved fig02_example_grid_cells.png")

# =============================================================================
# Figure 3 - how gridness is measured, plus a non-grid comparison
# =============================================================================
# a high-gridness cell with a short spacing shows several fields inside the box
ex = (units_df[(units_df.gridness > 1.2) & (units_df.spacing_cm < 50)]
      .sort_values("gridness", ascending=False).iloc[0])
non = (units_df[(~units_df.is_grid) & (units_df.n_spikes > 800)]
       .sort_values("gridness").iloc[0])

fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
for row, r in enumerate([ex, non]):
    s_ = session(r.path)
    i = unit_index(s_, r.unit)
    ac = gl.spatial_autocorr(s_["maps"][i])
    ext = [s_["edges"][0], s_["edges"][-1], s_["edges"][0], s_["edges"][-1]]
    ax = axes[row, 0]
    ax.imshow(s_["maps"][i], origin="lower", extent=ext, cmap="jet")
    ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_title("%s rate map (%.1f Hz peak)" % ("grid cell" if row == 0 else "non-grid cell",
                                                 np.nanmax(s_["maps"][i])))
    ax = axes[row, 1]
    n = ac.shape[0]
    half = (n - 1) / 2 * gl.BIN_CM
    ax.imshow(ac, origin="lower", cmap="jet", vmin=-0.5, vmax=1,
              extent=[-half, half, -half, half])
    spacing, ori, ell, pk = gl.autocorr_peaks(ac)
    ax.plot((pk[:, 0] - (n - 1) / 2) * gl.BIN_CM, (pk[:, 1] - (n - 1) / 2) * gl.BIN_CM,
            "o", mfc="none", mec="w", ms=6)
    ax.set_aspect("equal"); ax.set_xlabel("x lag (cm)"); ax.set_ylabel("y lag (cm)")
    ax.set_title("autocorrelogram, spacing %.0f cm" % spacing)
    ax = axes[row, 2]
    ang, corr, (ri, ro) = gl.rotational_correlation(ac)
    ax.plot(ang, corr, "k-")
    for a in (60, 120):
        ax.axvline(a, color="g", ls="--", lw=0.8)
    for a in (30, 90, 150):
        ax.axvline(a, color="r", ls=":", lw=0.8)
    g, _ = gl.gridness(ac)
    ax.set_xlabel("rotation (deg)"); ax.set_ylabel("correlation"); ax.set_xticks(range(0, 181, 30))
    ax.set_title("gridness = min(60,120) - max(30,90,150)\n= %.2f" % g)
fig.suptitle("Gridness: six-fold rotational symmetry of the spatial autocorrelogram", y=1.0)
fig.tight_layout()
fig.savefig("fig03_gridness_method.png", bbox_inches="tight")
plt.close(fig)
print("saved fig03_gridness_method.png")

# =============================================================================
# Figure 4 - shuffle test and classification
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axes[0]
bins = np.linspace(-0.9, 1.8, 60)
ax.hist(shuf_g.gridness.dropna(), bins=bins, density=True, color="0.75",
        label="shuffled (%d shifts)" % len(shuf_g))
ax.hist(units_df.gridness.dropna(), bins=bins, density=True, histtype="step",
        color="k", lw=1.5, label="observed (%d units)" % len(units_df))
ax.axvline(G_THR, color="r", ls="--", label="95th pct = %.2f" % G_THR)
ax.set_xlabel("gridness"); ax.set_ylabel("density"); ax.legend(fontsize=7)
ax.set_title("%d / %d units (%.0f%%) exceed the shuffle threshold"
             % (units_df.is_grid.sum(), len(units_df), 100 * units_df.is_grid.mean()))

ax = axes[1]
ax.scatter(units_df.gridness, units_df.stability, s=12,
           c=np.where(units_df.is_grid, "crimson", "0.6"), lw=0)
ax.axvline(G_THR, color="r", ls="--", lw=0.8)
ax.set_xlabel("gridness"); ax.set_ylabel("split-half map correlation")
ax.set_title("grid cells have stable maps")

ax = axes[2]
d = [units_df.loc[units_df.is_grid, "spatial_info"], units_df.loc[~units_df.is_grid, "spatial_info"]]
ax.boxplot(d, tick_labels=["grid", "non-grid"], showfliers=False)
for k, arr in enumerate(d):
    ax.plot(np.random.default_rng(k).normal(k + 1, 0.06, len(arr)), arr, ".",
            ms=3, color="crimson" if k == 0 else "0.5", alpha=0.6)
ax.set_ylabel("spatial information (bits/spike)")
ax.set_title("median %.2f vs %.2f bits/spike" % (d[0].median(), d[1].median()))
fig.tight_layout()
fig.savefig("fig04_shuffle_classification.png", bbox_inches="tight")
plt.close(fig)
print("saved fig04_shuffle_classification.png")

# =============================================================================
# Figure 5 - grid geometry
# =============================================================================
gdf = units_df[units_df.is_grid]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
ax = axes[0]
ax.hist(gdf.spacing_cm, bins=np.arange(25, 110, 5), color="steelblue", edgecolor="w")
ax.axvline(gdf.spacing_cm.median(), color="k", ls="--")
ax.set_xlabel("grid spacing (cm)"); ax.set_ylabel("grid cells")
ax.set_title("spacing: median %.0f cm\n(IQR %.0f-%.0f)" % (
    gdf.spacing_cm.median(), gdf.spacing_cm.quantile(0.25), gdf.spacing_cm.quantile(0.75)))

ax = axes[1]
ax.hist(gdf.orientation_deg, bins=np.arange(0, 61, 5), color="steelblue", edgecolor="w")
ax.set_xlabel("grid orientation (deg, mod 60)")
ax.set_title("orientation of individual grids\n(all rats pooled)")

ax = axes[2]
ax.hist(gdf.ellipticity, bins=np.arange(1, 2.05, 0.05), color="steelblue", edgecolor="w")
ax.set_xlabel("peak-distance ratio (max/min)")
ax.set_title("grids are close to regular\n(median %.2f)" % gdf.ellipticity.median())

ax = axes[3]
pairs = []
for path, d in gdf.groupby("path"):
    if len(d) >= 2:
        for a in range(len(d)):
            for b in range(a + 1, len(d)):
                pairs.append((abs(d.iloc[a].orientation_deg - d.iloc[b].orientation_deg),
                              abs(d.iloc[a].spacing_cm - d.iloc[b].spacing_cm)))
pairs = np.array(pairs)
within = np.minimum(pairs[:, 0], 60 - pairs[:, 0])
rng = np.random.default_rng(0)
o = gdf.orientation_deg.values
sess = gdf.path.values
a, b = rng.integers(0, len(o), 20000), rng.integers(0, len(o), 20000)
keep = sess[a] != sess[b]
across = np.abs(o[a][keep] - o[b][keep])
across = np.minimum(across, 60 - across)
bins = np.arange(0, 31, 2.5)
ax.hist(within, bins=bins, density=True, color="darkorange", edgecolor="w",
        label="same session (n=%d)" % len(within))
ax.hist(across, bins=bins, density=True, histtype="step", color="k", lw=1.4,
        label="different sessions")
ax.set_xlabel("|orientation difference| (deg)")
ax.set_ylabel("density"); ax.legend(fontsize=7)
ax.set_title("co-recorded grid cells share orientation\n(median %.1f vs %.1f deg)"
             % (np.median(within), np.median(across)))
fig.tight_layout()
fig.savefig("fig05_grid_geometry.png", bbox_inches="tight")
plt.close(fig)
print("saved fig05_grid_geometry.png")

# =============================================================================
# Figure 6 - layers and conjunctive grid x head-direction cells
# =============================================================================
hd_df = units_df[units_df.hd_mvl.notna()]
order = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]
fig = plt.figure(figsize=(12, 6.6))
gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.35)

ax = fig.add_subplot(gs[0, :2])
w = 0.38
xs = np.arange(len(order))
gr = [units_df[units_df.layer == L].is_grid.mean() * 100 for L in order]
hd = [hd_df[hd_df.layer == L].is_hd.mean() * 100 for L in order]
ax.bar(xs - w / 2, gr, w, label="grid", color="crimson")
ax.bar(xs + w / 2, hd, w, label="head-direction", color="steelblue")
for k, L in enumerate(order):
    ax.text(k - w / 2, gr[k] + 1.5, "%d/%d" % (units_df[units_df.layer == L].is_grid.sum(),
                                               (units_df.layer == L).sum()), ha="center", fontsize=7)
    ax.text(k + w / 2, hd[k] + 1.5, "%d/%d" % (hd_df[hd_df.layer == L].is_hd.sum(),
                                               (hd_df.layer == L).sum()), ha="center", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels([L.replace("MEC ", "") for L in order])
ax.set_ylabel("% of units"); ax.legend(fontsize=8)
ax.set_title("Grid cells occur in every layer; directional tuning is confined to deeper layers")

ax = fig.add_subplot(gs[0, 2:])
cats = ["grid only", "conjunctive", "HD only", "neither"]
bottom = np.zeros(len(order))
colors = ["crimson", "purple", "steelblue", "0.8"]
for c, col in zip(cats, colors):
    vals = []
    for L in order:
        d = hd_df[hd_df.layer == L]
        m = {"grid only": d.is_grid & ~d.is_hd, "conjunctive": d.is_grid & d.is_hd,
             "HD only": ~d.is_grid & d.is_hd, "neither": ~d.is_grid & ~d.is_hd}[c]
        vals.append(100 * m.mean())
    ax.bar(xs, vals, 0.6, bottom=bottom, label=c, color=col)
    bottom += np.array(vals)
ax.set_xticks(xs); ax.set_xticklabels([L.replace("MEC ", "") for L in order])
ax.set_ylabel("% of units"); ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Cell types by layer")

conj = (units_df[units_df.is_grid & units_df.is_hd].sort_values("hd_mvl", ascending=False).iloc[0])
pure = (units_df[units_df.is_grid & ~units_df.is_hd & units_df.hd_mvl.notna()]
        .sort_values("gridness", ascending=False).iloc[0])
for col, r, lab in [(0, pure, "pure grid cell"), (2, conj, "conjunctive grid x HD cell")]:
    s_ = session(r.path)
    i = unit_index(s_, r.unit)
    ext = [s_["edges"][0], s_["edges"][-1], s_["edges"][0], s_["edges"][-1]]
    ax = fig.add_subplot(gs[1, col])
    ax.imshow(s_["maps"][i], origin="lower", extent=ext, cmap="jet")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s\nrat %s %s, %s\ng = %.2f" % (lab, r.subject, r.unit,
                                                  r.layer.replace("MEC ", ""), r.gridness), fontsize=8)
    curves, centers, mvl = gl.hd_tuning(s_["units"], s_["hd"], s_["run_ep"])
    ax = fig.add_subplot(gs[1, col + 1], projection="polar")
    ax.plot(np.r_[centers, centers[0]], np.r_[curves[:, i], curves[0, i]], color="k")
    ax.fill(np.r_[centers, centers[0]], np.r_[curves[:, i], curves[0, i]], color="steelblue", alpha=0.4)
    ax.set_title("HD tuning (Hz), mvl = %.2f" % mvl[i], fontsize=8, pad=24)
    ax.tick_params(labelsize=6)
    rmax = np.nanmax(curves[:, i])
    ax.set_yticks([round(rmax / 2, 1), round(rmax, 1)])
fig.savefig("fig06_layers_and_conjunctive.png", bbox_inches="tight")
plt.close(fig)
print("saved fig06_layers_and_conjunctive.png")

# =============================================================================
# summary numbers for the README
# =============================================================================
summary = {
    "sessions": int(units_df.path.nunique()),
    "rats": int(units_df.subject.nunique()),
    "units": int(len(units_df)),
    "gridness_threshold": round(G_THR, 3),
    "hd_threshold": round(M_THR, 3),
    "grid_cells": int(units_df.is_grid.sum()),
    "grid_pct": round(100 * units_df.is_grid.mean(), 1),
    "spacing_median": round(float(gdf.spacing_cm.median()), 1),
    "spacing_iqr": [round(float(gdf.spacing_cm.quantile(0.25)), 1),
                    round(float(gdf.spacing_cm.quantile(0.75)), 1)],
    "stability_grid": round(float(gdf.stability.median()), 2),
    "stability_nongrid": round(float(units_df.loc[~units_df.is_grid, "stability"].median()), 2),
    "grid_pct_by_layer": {L: round(100 * units_df[units_df.layer == L].is_grid.mean(), 1) for L in order},
    "hd_pct_by_layer": {L: round(100 * hd_df[hd_df.layer == L].is_hd.mean(), 1) for L in order},
    "conjunctive_by_layer": {L: int((hd_df[hd_df.layer == L].is_grid & hd_df[hd_df.layer == L].is_hd).sum())
                             for L in order},
}
pd.Series(summary).to_json("summary.json", indent=2)
print(pd.Series(summary))
