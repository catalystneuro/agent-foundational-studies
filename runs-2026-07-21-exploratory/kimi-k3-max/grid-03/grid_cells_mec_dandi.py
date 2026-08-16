# %% [markdown]
# # Grid Cells in the Medial Entorhinal Cortex (DANDI 000582)
#
# This notebook demonstrates grid cells in the medial entorhinal cortex (MEC)
# using data from the DANDI Archive. We use dandiset
# [000582](https://dandiarchive.org/dandiset/000582), the NWB conversion of
# **Sargolini et al. (2006), Science**, "Conjunctive Representation of Position,
# Direction, and Velocity in Entorhinal Cortex" (Moser lab). The dataset
# contains 118 tetrode recording sessions from 15 Long Evans rats foraging in
# square open-field boxes, with 50 Hz 2D position tracking and 620 sorted
# single units annotated by MEC layer.
#
# Grid cells fire in a hexagonal lattice of locations. The standard
# identification pipeline, which we follow here using the Moser-lab `opexebo`
# package, is:
#
# 1. **Occupancy map**: time the animal spent in each 2.5 cm bin (running
#    epochs only, speed >= 2.5 cm/s).
# 2. **Rate map**: spikes per bin divided by occupancy, smoothed with a
#    Gaussian (sigma = 2 bins = 5 cm).
# 3. **Spatial autocorrelation** of the rate map: a grid cell's autocorrelation
#    shows a central peak surrounded by six peaks on a hexagon.
# 4. **Grid score**: correlation of the autocorrelation annulus with itself
#    rotated by 60 deg and 120 deg (high for hexagonal structure) minus
#    correlations at 30/90/150 deg (low for hexagonal structure).
# 5. **Significance**: circular time-shift shuffle (100 shifts) of the spike
#    train against the position trace.
#
# The analysis streams NWB files directly from the archive with `remfile`
# (no full downloads) and uses `pynapple` for data access.

# %% [markdown]
# ## Setup

# %%
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from grid_utils import (
    MIN_SPIKES, N_SHUFFLES, SHUFFLE_GS_CANDIDATE,
    analyze_session_units, load_session,
)

RESULTS_DIR = "results"
MAPS_DIR = os.path.join(RESULTS_DIR, "maps")
P_SIG = 0.05
LAYER_ORDER = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]

# %% [markdown]
# ## Dataset Access
#
# We query the DANDI API for the 118 NWB assets of dandiset 000582 (published
# version 0.251111.2151) and stream each file through
# `https://api.dandiarchive.org/api/assets/{asset_id}/download/` with a
# disk-cached `remfile.File`. Position is read from
# `processing/behavior/Position/SpatialSeriesLED1`.
#
# Two dataset quirks worth knowing:
# - The position metadata says "meters" but the values are centimeters: the
#   1.5 m box spans +/-75. We treat them as cm throughout.
# - A handful of position frames per session are NaN (LED tracking dropouts);
#   we drop them before computing speed.

# %%
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("000582", "0.251111.2151")
manifest = {a.path: a.identifier for a in dandiset.get_assets() if a.path.endswith(".nwb")}
print(f"{len(manifest)} NWB sessions")

# %% [markdown]
# ## One Session Up Close
#
# We start with `sub-11265_ses-16030604` (20 min in a 1.5 m box, 14 MEC LII
# units) to validate the data streams and the analysis pipeline before scaling
# to the full dataset.

# %%
DEMO_PATH = "sub-11265/sub-11265_ses-16030604_behavior+ecephys.nwb"
nwb, io = load_session(manifest[DEMO_PATH])
print(nwb)

# %% [markdown]
# ### Raw data validation
#
# The position trace covers the whole box, and spike positions overlaid on the
# trajectory already hint at multiple firing fields for some units.

# %%
from grid_utils import get_position, compute_speed

t, xy = get_position(nwb)
speed = compute_speed(t, xy)
print(f"duration {t[-1] - t[0]:.0f} s, arena "
      f"{np.nanmax(xy[:,0]) - np.nanmin(xy[:,0]):.0f} x "
      f"{np.nanmax(xy[:,1]) - np.nanmin(xy[:,1]):.0f} cm, "
      f"running frames {(speed >= 2.5).sum()}/{len(speed)}")

units = nwb["units"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].plot(xy[:, 0], xy[:, 1], lw=0.2, color="0.4")
axes[0].set_title("Position trace")
axes[0].set_xlabel("x (cm)"); axes[0].set_ylabel("y (cm)"); axes[0].set_aspect("equal")
for k, uid in enumerate(list(units.keys())[:3]):
    spk = units[uid].t
    spk_xy = np.column_stack([np.interp(spk, t, xy[:, 0]), np.interp(spk, t, xy[:, 1])])
    ax = axes[k + 1]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.1, color="0.8", zorder=0)
    ax.scatter(spk_xy[:, 0], spk_xy[:, 1], s=2, c="r", zorder=1)
    ax.set_title(f"unit {uid} ({units['unit_name'][uid]}), {len(spk)} spikes", fontsize=9)
    ax.set_aspect("equal")
fig.suptitle("sub-11265_ses-16030604 — raw position and spike locations")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig_01_raw_data_validation.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Rate maps, autocorrelations, and grid scores
#
# `analyze_session_units` (in `grid_utils.py`) restricts position and spikes to
# running epochs (speed >= 2.5 cm/s), builds the occupancy map once per
# session, and computes the smoothed rate map, its autocorrelation, and the
# grid score for every unit with at least 200 spikes.

# %%
demo_results, demo_meta = analyze_session_units(nwb, run_shuffles=False)
io.close()
for uid, res in sorted(demo_results.items()):
    print(f"unit {uid:2d} ({res['unit_name']:6s}, {res['layer']}): "
          f"{res['n_spikes']:5d} spikes, grid score {res['grid_score']:+.3f}, "
          f"spacing {res['grid_spacing']:.0f} cm")

# %%
n = len(demo_results)
ncols = 7
nblocks = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nblocks * 2, ncols, figsize=(2.2 * ncols, 2.2 * nblocks * 2))
for i, (uid, res) in enumerate(sorted(demo_results.items())):
    block, col = divmod(i, ncols)
    ax = axes[block * 2, col]
    ax.imshow(res["rate_map"], origin="lower", cmap="jet")
    ax.set_title(f"u{uid} {res['unit_name']}, {res['n_spikes']} spk", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[block * 2 + 1, col]
    ax.imshow(res["acorr"], origin="lower", cmap="jet")
    ax.set_title(f"gs={res['grid_score']:.2f}, d={res['grid_spacing']:.0f} cm", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.flat[n * 2:]:
    ax.axis("off")
for block in range(nblocks):
    axes[block * 2, 0].set_ylabel("rate map", fontsize=9)
    axes[block * 2 + 1, 0].set_ylabel("autocorr", fontsize=9)
fig.suptitle("sub-11265_ses-16030604 — smoothed rate maps and spatial autocorrelations")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig_02_demo_session_gallery.png", dpi=150)
plt.close(fig)

# %% [markdown]
# Several units (e.g. u4-u6, u8) show the signature hexagonal autocorrelation
# of grid cells; others fire in a single field or in bands and score near or
# below zero.

# %% [markdown]
# ## All 118 Sessions
#
# We now run the same pipeline on every session. Units with an observed grid
# score above 0.3 are tested for significance with 100 circular time-shift
# shuffles: spike times are shifted by a random offset (>= 20 s, wrapping
# around the session), which destroys the spike-position relationship while
# preserving the temporal structure of the spike train and the behavioral
# statistics. The p-value is the fraction of shuffle scores at or above the
# observed score (with the +1 correction).
#
# This takes a few minutes with 8 parallel workers (each worker streams with
# its own disk cache). Results are cached to `results/grid_scores.csv` and
# per-session map archives under `results/maps/`; delete them to force a
# rerun.

# %%
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

spec = importlib.util.spec_from_file_location("batch", "03_run_all_sessions.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)

csv_path = os.path.join(RESULTS_DIR, "grid_scores.csv")
if not os.path.exists(csv_path):
    items = sorted(manifest.items())
    all_rows, failures = [], {}
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        futures = {ex.submit(batch.process_session, (p, a, i % 8)): p
                   for i, (p, a) in enumerate(items)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sessions"):
            path = futures[fut]
            try:
                all_rows.extend(fut.result())
            except Exception:
                import traceback
                failures[path] = traceback.format_exc()
                print(f"FAILED: {path}\n{failures[path]}")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} sessions failed: {list(failures)}")

df = pd.read_csv(csv_path)
n_sig = (df["shuffle_p"] < P_SIG).sum()
print(f"{len(df)} units with >= {MIN_SPIKES} spikes from {df['session'].nunique()} sessions")
print(f"{n_sig} significant grid cells (p < {P_SIG}, {N_SHUFFLES} shuffles)")

# %% [markdown]
# ## Population Results

# %%
def load_maps(session):
    safe = session.replace("/", "_").replace(".nwb", "")
    return np.load(os.path.join(MAPS_DIR, safe + ".npz"))

# %% [markdown]
# ### Example grid cells
#
# The twelve highest-scoring significant grid cells, spread across sessions.

# %%
sig = df[df["shuffle_p"] < P_SIG].sort_values("grid_score", ascending=False)
seen_sessions, picks = set(), []
for _, row in sig.iterrows():
    if row["session"] not in seen_sessions or len(picks) >= 8:
        picks.append(row)
        seen_sessions.add(row["session"])
    if len(picks) >= 12:
        break

fig, axes = plt.subplots(4, 6, figsize=(15, 10))
for i, row in enumerate(picks):
    maps = load_maps(row["session"])
    uid = row["unit_id"]
    r, c = divmod(i, 6)
    ax = axes[r * 2, c]
    ax.imshow(maps[f"rate_map_{uid}"], origin="lower", cmap="jet")
    ax.set_title(f"{row['session'].split('/')[0].replace('sub-', 'rat ')} "
                 f"{row['unit_name']} ({row['layer'].replace('MEC ', '')})", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[r * 2 + 1, c]
    ax.imshow(maps[f"acorr_{uid}"], origin="lower", cmap="jet")
    ax.set_title(f"gs={row['grid_score']:.2f}, d={row['grid_spacing']:.0f} cm, "
                 f"p={row['shuffle_p']:.3f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for ax in axes.flat[len(picks) * 2:]:
    ax.axis("off")
for r in range(2):
    axes[r * 2, 0].set_ylabel("rate map", fontsize=9)
    axes[r * 2 + 1, 0].set_ylabel("autocorr", fontsize=9)
fig.suptitle("Significant grid cells from DANDI 000582 (Sargolini et al. 2006)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig_03_example_grid_cells.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Grid score distributions and the shuffle null

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
bins = np.linspace(-1.5, 2.0, 60)
ax.hist(df["grid_score"], bins=bins, color="0.4", label=f"all units (n={len(df)})")
ax.hist(sig["grid_score"], bins=bins, color="tab:red", label=f"significant (n={len(sig)})")
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("grid score"); ax.set_ylabel("number of units")
ax.set_title("Grid score distribution"); ax.legend(fontsize=8)

ax = axes[1]
null_scores = []
for session in df.loc[df["shuffle_p"].notna(), "session"].unique():
    maps = load_maps(session)
    for k in maps.files:
        if k.startswith("shuffle_"):
            null_scores.append(maps[k])
null_scores = np.concatenate(null_scores)
null_scores = null_scores[~np.isnan(null_scores)]
ax.hist(null_scores, bins=np.linspace(-1.5, 2.0, 80), density=True, color="0.6",
        label=f"shuffle null (n={len(null_scores)})")
ax.hist(df.loc[df["shuffle_p"].notna(), "grid_score"], bins=np.linspace(-1.5, 2.0, 80),
        density=True, histtype="step", lw=2, color="tab:red", label="observed (candidates)")
ax.set_xlabel("grid score"); ax.set_ylabel("density")
ax.set_title("Observed vs circular-shift null"); ax.legend(fontsize=8)

ax = axes[2]
cand = df[df["shuffle_p"].notna()]
ax.scatter(cand["grid_score"], cand["shuffle_z"], s=8, alpha=0.4,
           c=np.where(cand["shuffle_p"] < P_SIG, "tab:red", "0.5"))
ax.axhline(1.645, color="k", ls="--", lw=0.8, label="z = 1.645 (p=0.05, one-sided)")
ax.set_xlabel("observed grid score"); ax.set_ylabel("shuffle z-score")
ax.set_title("Significance of candidate units"); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("fig_04_grid_score_distributions.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Grid cells across MEC layers
#
# Sargolini et al. reported grid cells throughout MEC layers II-VI, with the
# purest grid representation in layer II.

# %%
dfl = df.copy()
dfl["layer"] = dfl["layer"].replace("", np.nan)
dfl = dfl.dropna(subset=["layer"])
dfl = dfl[dfl["layer"].isin(LAYER_ORDER)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
frac = [((dfl["layer"] == layer) & (dfl["shuffle_p"] < P_SIG)).sum() /
        max((dfl["layer"] == layer).sum(), 1) for layer in LAYER_ORDER]
ns = [(dfl["layer"] == layer).sum() for layer in LAYER_ORDER]
bars = ax.bar(range(len(LAYER_ORDER)), frac, color="steelblue")
ax.set_xticks(range(len(LAYER_ORDER)))
ax.set_xticklabels([l.replace("MEC ", "") for l in LAYER_ORDER])
for rect, f, nn in zip(bars, frac, ns):
    ax.text(rect.get_x() + rect.get_width() / 2, f + 0.01, f"{f*100:.0f}%\n(n={nn})",
            ha="center", fontsize=9)
ax.set_ylabel("fraction significant grid cells")
ax.set_title("Grid-cell prevalence by layer")
ax.set_ylim(0, max(frac) * 1.35)

ax = axes[1]
data = [dfl.loc[dfl["layer"] == layer, "grid_score"].values for layer in LAYER_ORDER]
parts = ax.violinplot(data, showmedians=True)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
rng = np.random.default_rng(0)
for i, layer in enumerate(LAYER_ORDER):
    vals = dfl.loc[(dfl["layer"] == layer) & (dfl["shuffle_p"] < P_SIG), "grid_score"].values
    ax.scatter(np.full(len(vals), i + 1) + rng.normal(0, 0.04, len(vals)), vals,
               s=10, color="tab:red", zorder=3, label="significant" if i == 0 else None)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(range(1, len(LAYER_ORDER) + 1))
ax.set_xticklabels([l.replace("MEC ", "") for l in LAYER_ORDER])
ax.set_ylabel("grid score"); ax.set_title("Grid scores by layer"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig_05_layer_breakdown.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Grid spacing and orientation

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
ax.hist(sig["grid_spacing"], bins=np.arange(20, 200, 10), color="steelblue")
med = sig["grid_spacing"].median()
ax.axvline(med, color="tab:red", ls="--", label=f"median = {med:.0f} cm")
ax.set_xlabel("grid spacing (cm)"); ax.set_ylabel("number of units")
ax.set_title(f"Grid spacing of significant cells (n={len(sig)})"); ax.legend(fontsize=8)

ax = axes[1]
orient = sig["grid_orientation"].dropna().values % 60
ax.hist(orient, bins=np.arange(0, 65, 5), color="steelblue")
ax.set_xlabel("grid orientation (deg, modulo 60)"); ax.set_ylabel("number of units")
ax.set_title("Grid orientation distribution")
fig.tight_layout()
fig.savefig("fig_06_grid_spacing.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Summary
#
# - We analyzed all 118 sessions of DANDI 000582 (Sargolini et al. 2006):
#   620 tetrode units recorded from MEC of rats foraging in open-field boxes.
# - A substantial fraction of units with sufficient spiking show significant
#   hexagonal spatial periodicity (grid score above the 95th percentile of
#   circular time-shift shuffles), concentrated in layers II and III.
# - Grid spacings cluster around 40-80 cm, consistent with the original
#   report and with the dorsoventral spacing gradient of MEC.
#
# Exact counts depend on the shuffle seed; see `results/grid_scores.csv` for
# the full unit-level table and the README for headline numbers.
