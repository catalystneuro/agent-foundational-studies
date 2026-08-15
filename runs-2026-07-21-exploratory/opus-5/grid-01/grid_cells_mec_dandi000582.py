# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Grid cells in the medial entorhinal cortex
#
# **Dataset:** [DANDI:000582](https://dandiarchive.org/dandiset/000582),
# *Conjunctive Representation of Position, Direction, and Velocity in Entorhinal
# Cortex* (Sargolini, Fyhn, Hafting, McNaughton, Witter, Moser and Moser,
# *Science* 2006). 118 NWB sessions, 15 Long-Evans rats, tetrode recordings from
# the dorsocaudal medial entorhinal cortex (MEC) while the animal foraged freely
# for scattered food in a square enclosure. Each file contains sorted spike
# times, tracked position from one or two head-mounted LEDs, and a local field
# potential channel.
#
# **Goal:** show that a substantial fraction of MEC neurons fire in a periodic,
# hexagonal lattice tiling the whole environment, which is the defining
# signature of a grid cell (Hafting et al., *Nature* 2005).
#
# **Approach:**
#
# 1. Stream every session from the DANDI S3 bucket with `remfile` (no full
#    downloads) and wrap it in `pynapple` objects.
# 2. Keep only periods when the rat is running (> 2.5 cm/s), build a smoothed
#    firing-rate map for every unit, and take its spatial autocorrelogram.
# 3. Score the six-fold rotational symmetry of the autocorrelogram (gridness).
# 4. Compare each cell against 100 circularly time-shifted versions of its own
#    spike train; a cell counts as a grid cell if its gridness exceeds the 95th
#    percentile of the pooled shuffle distribution.
# 5. Characterise the resulting grid population: spacing, orientation,
#    within-session stability, and conjunctive head-direction tuning.
#
# All analysis code lives in the accompanying module `gridlib.py`, which this
# notebook imports so that the same functions are used everywhere in the
# pipeline. Nothing here is simulated: every number and every figure comes from
# the archived recordings.

# %%
import itertools
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import requests
from scipy import ndimage, stats
from tqdm import tqdm

import gridlib as G

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 130})

# %% [markdown]
# ## 1. Dataset discovery
#
# The DANDI REST API gives the asset list for the dandiset. Each asset is one
# recording session; the identifiers are all we need to build streaming URLs.

# %%
if not Path("assets.json").exists():
    url = ("https://api.dandiarchive.org/api/dandisets/000582/versions/draft/"
           "assets/?page_size=200")
    res = requests.get(url, timeout=60).json()["results"]
    json.dump([{"path": a["path"], "id": a["asset_id"], "size": a["size"]}
               for a in res], open("assets.json", "w"), indent=1)

assets = G.load_assets()
print(f"{len(assets)} sessions, "
      f"{len(set(a['path'].split('/')[0] for a in assets))} rats, "
      f"{sum(a['size'] for a in assets) / 1e9:.2f} GB total")
print(assets[0]["path"])

# %% [markdown]
# ## 2. Load and validate one session
#
# Before any analysis, look at the raw streams: the tracked trajectory, the
# occupancy of the arena, the running-speed distribution, the LFP, and the spike
# trains. `gridlib.load_session` returns pynapple objects (`TsGroup` for units,
# `TsdFrame` for position, `Tsd` for speed and head direction).

# %%
EXAMPLE = "sub-11265/sub-11265_ses-06020601_behavior+ecephys.nwb"
asset = [a for a in assets if a["path"] == EXAMPLE][0]
s = G.load_session(asset, keep_lfp=True)
ep = G.run_epochs(s)
edges = G.map_edges(s["position"], ep)
occ = G.occupancy_map(s["position"], ep, edges, s["dt"])

print(f"{s['path']}\n  rat {s['subject']}, {s['duration']:.0f} s, "
      f"tracking at {1 / s['dt']:.0f} Hz, {len(s['units'])} units "
      f"({sorted(set(s['histology']))})")
print(f"  arena {edges[0][-1] - edges[0][0]:.0f} x "
      f"{edges[1][-1] - edges[1][0]:.0f} cm, "
      f"{ep.tot_length():.0f} s above {G.SPEED_MIN_CMS} cm/s, "
      f"{(occ >= G.MIN_OCC_S).mean() * 100:.0f}% of bins visited")

# %% [markdown]
# Note two quirks of this NWB conversion that the loader takes into account:
# the position `SpatialSeries` is labelled in meters but the values are in
# centimetres (a 100 cm box spans 100 units), and the LFP is stored in raw
# acquisition units rather than volts. Neither affects the analysis, but the
# axis labels below follow the actual values.

# %%
pos = s["position"]
fig = plt.figure(figsize=(14, 9))
gsp = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

ax = fig.add_subplot(gsp[0, :])
ax.plot(pos.index.values, pos["x"].values, lw=0.6, label="x")
ax.plot(pos.index.values, pos["y"].values, lw=0.6, label="y")
ax.set(xlim=(0, 120), xlabel="time (s)", ylabel="position (cm)",
       title="Tracked position, first 120 s")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gsp[1, 0])
ax.plot(pos["x"].values, pos["y"].values, lw=0.3, color="0.4")
ax.set(aspect="equal", xlabel="x (cm)", ylabel="y (cm)", title="Full trajectory")

ax = fig.add_subplot(gsp[1, 1])
im = ax.imshow(occ.T, origin="lower", cmap="viridis",
               extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
ax.set(aspect="equal", xlabel="x (cm)", title="Occupancy (s per 3 cm bin)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = fig.add_subplot(gsp[1, 2])
ax.hist(s["speed"].values, bins=80, range=(0, 100), color="0.4")
ax.axvline(G.SPEED_MIN_CMS, color="r", ls="--", label=f"{G.SPEED_MIN_CMS} cm/s cut")
ax.set(xlabel="running speed (cm/s)", ylabel="video frames",
       title="Speed distribution")
ax.legend(fontsize=8)

ax = fig.add_subplot(gsp[2, 0])
seg = s["lfp"].restrict(nap.IntervalSet(100, 102))
ax.plot(seg.index.values, seg.values, lw=0.5)
ax.set(xlabel="time (s)", ylabel="LFP (a.u.)",
       title="MEC LFP, 2 s (theta rhythm visible)")

ax = fig.add_subplot(gsp[2, 1:])
for i, k in enumerate(list(s["units"].keys())[:12]):
    tt = s["units"][k].restrict(nap.IntervalSet(0, 60)).index.values
    ax.vlines(tt, i, i + 0.8, lw=0.4, color="k")
ax.set(xlabel="time (s)", ylabel="unit", title="Spike raster, first 60 s",
       yticks=np.arange(min(12, len(s["units"]))) + 0.4,
       yticklabels=s["unit_names"][:12])
ax.tick_params(axis="y", labelsize=7)

fig.suptitle(f"Data validation: {EXAMPLE.split('/')[-1]}  (DANDI:000582)", y=0.98)
fig.savefig("fig01_data_validation.png", dpi=130, bbox_inches="tight")

# %% [markdown]
# ## 3. Rate maps, autocorrelograms and the gridness score
#
# For each unit the firing-rate map is the number of spikes per spatial bin
# divided by the time the rat spent there, both smoothed with a 4.5 cm Gaussian.
# Spike positions come from pynapple's `value_from`, which interpolates the
# tracked position at each spike time.
#
# The spatial autocorrelogram is the Pearson correlation of the rate map with a
# shifted copy of itself (Hafting et al. 2005). A hexagonal firing lattice
# produces a central peak surrounded by six peaks at 60° intervals. Gridness
# quantifies that symmetry: the autocorrelogram annulus (excluding the central
# peak) is rotated and correlated with itself, and
#
# > gridness = min(r₆₀, r₁₂₀) − max(r₃₀, r₉₀, r₁₅₀)
#
# is taken as the maximum over a range of outer annulus radii. A perfect
# hexagonal lattice gives values near 2; a spatially unstructured cell gives
# values near or below 0.

# %%
unit_names = s["unit_names"]
demo_unit = "t3c1"
st = s["units"][list(s["units"].keys())[unit_names.index(demo_unit)]]
rm, _ = G.rate_map(st, s["position"], ep, edges, occ)
ac = G.spatial_autocorr(rm)
g, info = G.grid_score(ac, return_curve=True)
spacing, orientation, peaks = G.grid_geometry(ac)
print(f"unit {demo_unit}: gridness {g:.2f}, spacing {spacing:.0f} cm, "
      f"orientation {orientation:.0f}°, "
      f"spatial information {G.spatial_information(rm, occ):.2f} bits/spike")

# %%
angles, prof = G.rotational_profile(ac, info["r_in"], info["r_out"])
spk = st.restrict(ep)
sp = spk.value_from(s["position"])
p = s["position"].restrict(ep)

fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5))
axes[0].plot(p["x"], p["y"], lw=0.25, color="0.75")
axes[0].scatter(sp["x"], sp["y"], s=3, c="crimson", linewidths=0)
axes[0].set(aspect="equal", xlabel="x (cm)", ylabel="y (cm)",
            title=f"trajectory + {len(spk)} spikes, unit {demo_unit}")

ext = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
im = axes[1].imshow(np.ma.masked_invalid(rm).T, origin="lower", cmap="jet",
                    extent=ext)
axes[1].set(aspect="equal", xlabel="x (cm)",
            title=f"rate map, peak {np.nanmax(rm):.1f} Hz")
plt.colorbar(im, ax=axes[1], fraction=0.046, label="Hz")

axes[2].imshow(np.ma.masked_invalid(ac).T, origin="lower", cmap="jet",
               vmin=-0.6, vmax=1)
cy, cx = (np.array(ac.shape) - 1) / 2.0
for r_, c_ in [(info["r_in"], "w"), (info["r_out"], "k")]:
    axes[2].add_patch(plt.Circle((cx, cy), r_, fill=False, color=c_, lw=1.2,
                                 ls="--"))
axes[2].scatter(peaks[:, 0], peaks[:, 1], s=45, facecolors="none",
                edgecolors="k", lw=1.2)
axes[2].set(aspect="equal", xticks=[], yticks=[],
            title=f"autocorrelogram, 6 nearest peaks\nannulus "
                  f"{info['r_in']}-{info['r_out']} bins")

axes[3].plot(angles, prof, "-", color="k", lw=1.2)
for a_ in (60, 120):
    axes[3].axvline(a_, color="tab:green", lw=1, alpha=0.6)
for a_ in (30, 90, 150):
    axes[3].axvline(a_, color="tab:red", lw=1, alpha=0.6)
axes[3].set(xlabel="rotation (deg)", ylabel="Pearson r",
            xticks=np.arange(0, 181, 30),
            title=f"rotational symmetry\ngridness = {g:.2f}")
fig.tight_layout()
fig.savefig("fig08_single_cell_walkthrough.png", bbox_inches="tight")

# %% [markdown]
# The rate map shows discrete firing fields arranged on a triangular lattice
# that covers the whole box, and the autocorrelogram has the six-peak ring that
# defines a grid cell. The rotational-correlation profile peaks at 60° and 120°
# and dips at 30°, 90° and 150°, which is exactly what six-fold symmetry
# predicts.

# %% [markdown]
# ## 4. Run the analysis over all 118 sessions
#
# `02_analyze_all_sessions.py` applies the same pipeline to every unit of every
# session and adds, for each unit, 100 gridness scores computed from circularly
# time-shifted spike trains (shifts of at least 20 s). The shuffle preserves the
# spike-train statistics but destroys the alignment between spikes and position,
# so the pooled shuffle distribution is the null distribution for gridness. This
# takes several minutes on 8 cores; the result is cached in `results.pkl`.

# %%
if not Path("results.pkl").exists():
    from importlib import import_module
    analyze = import_module("02_analyze_all_sessions")
    rows, maps = [], {}
    with Pool(8) as pool:
        for r_, m_ in tqdm(pool.imap_unordered(analyze.analyze_session, assets),
                           total=len(assets), desc="sessions"):
            rows.extend(r_)
            maps.update(m_)
    pickle.dump(dict(rows=rows, maps=maps), open("results.pkl", "wb"))

R = pickle.load(open("results.pkl", "rb"))
df = pd.DataFrame([{k: v for k, v in r.items() if k != "shuffle"} for r in R["rows"]])
inc = df[df.included].reset_index(drop=True)
shuffles = {(r["session"], r["unit"]): r["shuffle"] for r in R["rows"]
            if r["included"]}
pooled = np.concatenate(list(shuffles.values()))
THR = float(np.nanpercentile(pooled, 95))
inc["is_grid"] = inc.grid_score > THR
inc["big_arena"] = inc.arena_cm > 120
maps = R["maps"]

print(f"{len(df)} sorted units in {df.session.nunique()} sessions from "
      f"{df.subject.nunique()} rats")
print(f"{len(inc)} units passed the activity criteria "
      f"(>= {G.MIN_SPIKES} spikes and >= {G.MIN_MEAN_RATE_HZ} Hz while running)")
print(f"shuffle threshold (95th percentile of {len(pooled)} shuffles) = {THR:.3f}")
print(f"grid cells: {inc.is_grid.sum()} / {len(inc)} "
      f"({100 * inc.is_grid.mean():.1f}%)")

# %% [markdown]
# ## 5. Example grid cells across animals
#
# The strongest grid cell from each of six rats. Rate maps are displayed with a
# robust colour ceiling (97th percentile) so that the weaker fields stay
# visible next to the strongest one; the true peak rate is printed above each
# map.

# %%
def show_map(ax, m, edges=None, title=None):
    m = np.asarray(m, dtype=float)
    kw = {"extent": [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]} \
        if edges is not None else {}
    ax.imshow(np.ma.masked_invalid(m).T, origin="lower", cmap="jet", vmin=0,
              vmax=np.nanpercentile(m, 97), interpolation="nearest", **kw)
    ax.set(aspect="equal", xticks=[], yticks=[])
    if title:
        ax.set_title(title, pad=3)


def show_ac(ax, ac, title=None):
    ax.imshow(np.ma.masked_invalid(np.asarray(ac, dtype=float)).T, origin="lower",
              cmap="jet", vmin=-0.6, vmax=1.0)
    ax.set(aspect="equal", xticks=[], yticks=[])
    if title:
        ax.set_title(title, pad=3)


def load_unit(row):
    a_ = [a for a in assets if a["path"] == row.session][0]
    ss = G.load_session(a_)
    ep_ = G.run_epochs(ss)
    st_ = ss["units"][list(ss["units"].keys())[ss["unit_names"].index(row.unit)]]
    return ss, ep_, st_


best = (inc[inc.is_grid].sort_values("grid_score", ascending=False)
        .groupby("subject").head(1).sort_values("grid_score", ascending=False)
        .head(6))

fig, axes = plt.subplots(3, len(best), figsize=(2.1 * len(best), 6.8))
for j, (_, row) in enumerate(best.iterrows()):
    d = maps[(row.session, row.unit)]
    ss, ep_, st_ = load_unit(row)
    sp_ = st_.restrict(ep_).value_from(ss["position"])
    p_ = ss["position"].restrict(ep_)

    ax = axes[0, j]
    ax.plot(p_["x"], p_["y"], lw=0.25, color="0.75")
    ax.scatter(sp_["x"], sp_["y"], s=2.5, c="crimson", linewidths=0)
    ax.set(aspect="equal", xticks=[], yticks=[])
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
    ss["io"].close()

fig.suptitle("Grid cells in rat medial entorhinal cortex "
             "(DANDI:000582, Sargolini et al. 2006)", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig02_example_grid_cells.png", bbox_inches="tight")

# %% [markdown]
# ## 6. The population is more hexagonal than chance
#
# The observed gridness distribution is clearly bimodal, with a mode well above
# anything the shuffle produces. Grid cells also have more stable rate maps
# (first half of the session vs. second half) and carry more spatial
# information than the rest of the population, and they are more common in the
# superficial layers, as expected for MEC.

# %%
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
for lab, sub_, col in [("grid", inc[inc.is_grid], "crimson"),
                       ("non-grid", inc[~inc.is_grid], "0.6")]:
    axes[2].scatter([], [], c=col, label=f"{lab} (n={len(sub_)})")
axes[2].legend(fontsize=7, loc="lower right")

lay = inc.assign(layer=inc.histology.str.replace("MEC ", "", regex=False))
lay = lay[lay.layer.str.startswith("L")]
tab = lay.groupby("layer").is_grid.agg(["size", "sum", "mean"]).sort_index()
axes[3].bar(np.arange(len(tab)), 100 * tab["mean"].values, color="steelblue")
axes[3].set_xticks(np.arange(len(tab)))
axes[3].set_xticklabels([f"{i}\n(n={int(n)})"
                         for i, n in zip(tab.index, tab["size"])])
axes[3].set(xlabel="MEC layer (histology field)", ylabel="grid cells (%)",
            title="Grid cells are most common in the\nsuperficial layers")
chi = stats.chi2_contingency(np.stack([tab["sum"].values,
                                       (tab["size"] - tab["sum"]).values]))
axes[3].text(0.40, 0.97, f"chi2 p = {chi.pvalue:.1e}", transform=axes[3].transAxes,
             fontsize=7, va="top")

fig.tight_layout()
fig.savefig("fig04_population_statistics.png", bbox_inches="tight")

# %% [markdown]
# ## 7. Geometry of the grids
#
# Most sessions used a square box of about 1 m; 22 sessions used a much larger
# (roughly 2 m, circular) enclosure, and spacing is reported separately for the
# two because a 1 m box truncates the measurable range. Grid cells recorded
# simultaneously share their lattice orientation far more closely than randomly
# paired cells do, which is the module organisation of the grid map. Spacing
# also grows with electrode depth, the dorsoventral gradient.

# %%
gc = inc[inc.is_grid]
small = gc[~gc.big_arena]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))

ax = axes[0]
ax.hist([small.spacing_cm, gc[gc.big_arena].spacing_cm],
        bins=np.arange(15, 135, 5), stacked=True,
        color=["steelblue", "orange"], edgecolor="w",
        label=[f"~1 m box (n={len(small)})",
               f"large arena (n={int(gc.big_arena.sum())})"])
ax.axvline(small.spacing_cm.median(), color="k", ls="--",
           label=f"median, 1 m box = {small.spacing_cm.median():.0f} cm")
ax.set(xlabel="grid spacing (cm)", ylabel="grid cells",
       title="Spacing of the hexagonal lattice")
ax.legend(fontsize=7)

same_o, same_s = [], []
for _, sub_ in gc.groupby("session"):
    o_ = sub_.orientation_deg.dropna().values
    s_ = sub_.spacing_cm.dropna().values
    for a_, b_ in itertools.combinations(o_, 2):
        dd = abs(a_ - b_) % 60
        same_o.append(min(dd, 60 - dd))
    for a_, b_ in itertools.combinations(s_, 2):
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
sub_ = small.dropna(subset=["depth_m"])
ax.scatter(1e3 * sub_.depth_m, sub_.spacing_cm, s=16, c="crimson", linewidths=0)
r_depth, p_depth = stats.pearsonr(sub_.depth_m, sub_.spacing_cm)
b_ = np.polyfit(1e3 * sub_.depth_m, sub_.spacing_cm, 1)
xs = np.linspace(1e3 * sub_.depth_m.min(), 1e3 * sub_.depth_m.max(), 10)
ax.plot(xs, np.polyval(b_, xs), "k--", lw=1, label=f"r = {r_depth:.2f}, p = {p_depth:.1e}")
ax.legend(fontsize=7)
ax.set(xlabel="electrode depth below dura (mm)", ylabel="grid spacing (cm)",
       title=f"Dorsoventral gradient (1 m box only, n={len(sub_)})")

fig.tight_layout()
fig.savefig("fig05_grid_geometry.png", bbox_inches="tight")
print(f"spacing in the 1 m box: median {small.spacing_cm.median():.1f} cm "
      f"(IQR {small.spacing_cm.quantile(.25):.0f}-"
      f"{small.spacing_cm.quantile(.75):.0f})")
print(f"within-session spacing ratio {np.median(same_s):.2f} vs "
      f"{np.median(rand_s):.2f} for random pairs")

# %% [markdown]
# ## 8. Conjunctive grid x head-direction cells
#
# In sessions with two tracking LEDs, head direction is the angle of the vector
# between them, so the same units can be tested for directional tuning. The
# significance threshold on the mean vector length comes from the same
# circular-shift shuffle (script `02b_hd_shuffles.py`), which matters because
# the mean vector length of a low-firing cell is biased upward by sampling
# alone. Because the NWB file does not say which LED sits in front, a preferred
# direction may be flipped by 180°; the strength of the tuning is unaffected.

# %%
hd_rows = pickle.load(open("hd_shuffles.pkl", "rb"))
MVL_THR = float(np.nanpercentile(
    np.concatenate([r["mvl_shuffle"] for r in hd_rows]), 95))
hd_lookup = {(r["session"], r["unit"]): r["mvl"] for r in hd_rows}
inc["mvl"] = [hd_lookup.get((a_, b_), np.nan)
              for a_, b_ in zip(inc.session, inc.unit)]
hd_df = inc.dropna(subset=["mvl"])
conj = hd_df[hd_df.is_grid & (hd_df.mvl > MVL_THR)].sort_values(
    "mvl", ascending=False)
print(f"{len(hd_df)} units with head-direction tracking; "
      f"{int(hd_df.is_grid.sum())} of them are grid cells; "
      f"{len(conj)} are conjunctive (MVL above the shuffle threshold "
      f"of {MVL_THR:.2f})")

fig = plt.figure(figsize=(13.5, 3.9))
gsp = fig.add_gridspec(1, 4, wspace=0.4)
ax = fig.add_subplot(gsp[0, 0])
ax.scatter(hd_df.mvl, hd_df.grid_score, s=12,
           c=np.where(hd_df.is_grid & (hd_df.mvl > MVL_THR), "crimson",
                      np.where(hd_df.is_grid, "tab:orange", "0.6")), linewidths=0)
ax.axhline(THR, color="k", ls="--", lw=1)
ax.axvline(MVL_THR, color="k", ls=":", lw=1)
ax.set(xlabel="head-direction mean vector length", ylabel="gridness score",
       title=f"{len(conj)} of {int(hd_df.is_grid.sum())} grid cells are also\n"
             f"direction-tuned (thresholds from shuffles)")

for j, (_, r_) in enumerate(conj[conj.n_spikes > 500].head(3).iterrows()):
    ss, ep_, st_ = load_unit(r_)
    curve, centers, mvl, pref = G.hd_tuning(st_, ss["hd"], ep_)
    sm_ = ndimage.gaussian_filter1d(np.nan_to_num(curve), 1.0, mode="wrap")
    axp = fig.add_subplot(gsp[0, j + 1], projection="polar")
    axp.plot(np.append(centers, centers[0]), np.append(sm_, sm_[0]),
             color="crimson", lw=1.5)
    axp.fill(np.append(centers, centers[0]), np.append(sm_, sm_[0]),
             color="crimson", alpha=0.2)
    axp.set_title(f"rat {r_.subject} {r_.unit}\ngridness {r_.grid_score:.2f}, "
                  f"MVL {mvl:.2f}", pad=24, fontsize=8)
    axp.tick_params(labelsize=7)
    axp.set_rlabel_position(285)
    ss["io"].close()

fig.suptitle("Grid cells in MEC are often conjunctive with head direction "
             "(polar axes show firing rate in Hz)", y=1.05)
fig.savefig("fig06_conjunctive_cells.png", bbox_inches="tight")

# %% [markdown]
# ## 9. Gallery of the strongest grid cells

# %%
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

# %% [markdown]
# ## 10. Summary

# %%
summary = pd.Series(dict(
    sessions=int(df.session.nunique()), rats=int(df.subject.nunique()),
    units_total=int(len(df)), units_analysed=int(len(inc)),
    gridness_threshold=round(THR, 3), grid_cells=int(inc.is_grid.sum()),
    pct_grid=round(100 * inc.is_grid.mean(), 1),
    sessions_with_grid_cells=int(inc.groupby("session").is_grid.any().sum()),
    median_spacing_1m_box=round(float(small.spacing_cm.median()), 1),
    median_gridness_grid=round(float(gc.grid_score.median()), 2),
    median_stability_grid=round(float(gc.stability.median()), 2),
    median_stability_nongrid=round(float(inc[~inc.is_grid].stability.median()), 2),
    within_session_dtheta=round(float(np.median(same_o)), 1),
    random_pair_dtheta=round(float(np.median(rand_o)), 1),
    depth_spacing_r_1m_box=round(float(r_depth), 2),
    halfsplit_gridness_r=round(float(stats.pearsonr(
        *gc[["grid_score_h1", "grid_score_h2"]].dropna().values.T)[0]), 2),
    mvl_threshold=round(MVL_THR, 3), units_with_hd=int(len(hd_df)),
    conjunctive_cells=int(len(conj)),
))
print(summary.to_string())
inc.drop(columns=["included"]).to_csv("unit_metrics.csv", index=False)
summary.to_json("summary.json", indent=1)

# %% [markdown]
# ## Conclusion
#
# Of 620 sorted units recorded in dorsocaudal MEC across 118 sessions and 15
# rats, 193 (31%) fire in a hexagonally periodic pattern whose gridness exceeds
# the 95th percentile of a 62,000-sample shuffle distribution, and the observed
# gridness distribution is visibly bimodal rather than a shifted version of the
# null. These cells are found in 14 of the 15 rats, they are far more common in
# the superficial layers (48% in layer II, 40% in layer III, about 21% in layers
# V and VI), their rate maps are stable within a session (median split-half
# correlation 0.70 vs. 0.34 for the rest of the population), and cells recorded
# simultaneously share the orientation of their lattice to within a few degrees.
# Grid spacing in the 1 m box has a median of about 50 cm and increases with
# electrode depth, the dorsoventral gradient reported for MEC. Roughly half of
# the grid cells with head-direction tracking are also directionally tuned,
# which is the conjunctive position-by-direction coding this dataset was
# collected to demonstrate.
