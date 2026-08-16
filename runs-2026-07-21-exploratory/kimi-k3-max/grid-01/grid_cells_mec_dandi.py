# %% [markdown]
# # Grid cells in the medial entorhinal cortex
#
# **Data:** DANDI Archive dataset [000582](https://dandiarchive.org/dandiset/000582),
# *"Conjunctive Representation of Position, Direction, and Velocity in Entorhinal
# Cortex"* (Sargolini, Fyhn, Hafting, McNaughton, Witter, Moser & Moser, 2006,
# *Science*). Long Evans rats foraged in square open-field arenas while single
# units were recorded with tetrodes in the dorsocaudal medial entorhinal cortex
# (MEC). Each NWB file contains spike times for the isolated units and 50 Hz
# 2D position tracking of an LED on the head stage.
#
# **Question:** Do MEC neurons show the defining signature of grid cells:
# multiple firing fields arranged on a hexagonal lattice?
#
# **Approach:**
# 1. Stream one example session and inspect the raw behavioral and neural data.
# 2. Compute speed-filtered firing-rate maps, spatial autocorrelograms, and the
#    standard Moser-lab gridness score (via the `opexebo` package).
# 3. Scale to all 118 sessions (620 units) and validate candidate grid cells
#    with a circular time-shift shuffle test.
# 4. Summarize grid scores, grid spacing, and grid orientation across the
#    population and across MEC layers.

# %% [markdown]
# ## Setup

# %%
import json
import os
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py
import remfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from tqdm import tqdm

import opexebo.analysis as opa
import opexebo.general as opg

# Analysis parameters (standard for this dataset)
BIN_WIDTH = 2.5        # cm, spatial bin width
SMOOTH_SIGMA = 2.0     # bins, Gaussian smoothing of rate maps (5 cm)
SPEED_THRESH = 2.5     # cm/s, immobility threshold
MIN_SPIKES = 200       # min spikes (moving) for population statistics
N_SHUFFLES = 100       # circular-shift shuffles for significance
SHUFFLE_MIN_SHIFT = 20.0  # s

FIGDIR = "."
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LAYER_ORDER = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]
LAYER_COLORS = {"MEC LII": "#d62728", "MEC LIII": "#1f77b4",
                "MEC LV": "#2ca02c", "MEC LVI": "#9467bd", "": "0.5"}

# %% [markdown]
# ## Dataset survey
#
# We enumerate all assets of the published version of the dandiset through the
# DANDI REST API and keep sessions that contain both units and position data.

# %%
import requests

DANDISET = "000582"
VERSION = "0.251111.2151"

assets = []
url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/"
params = {"page_size": 200}
while url:
    r = requests.get(url, params=params, timeout=30).json()
    assets.extend(r["results"])
    url, params = r.get("next"), {}
rows = [(a["path"], a["asset_id"], a.get("size", 0))
        for a in assets if a["path"].endswith(".nwb")]
print(f"{len(rows)} NWB sessions in dandiset {DANDISET} v{VERSION}")

# %% [markdown]
# ## Core analysis functions
#
# Position in these files is labeled `meters` but the values are centimeters
# (a 1 m box spans -50..50; a 1.5 m box spans -75..75), consistent with the
# session descriptions. We therefore treat position as cm throughout.

# %%
def load_session(asset_id, cache_dir="/tmp/remfile_cache_grid"):
    """Stream one NWB session from DANDI; return position track and units table."""
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    h5 = h5py.File(remfile.File(url, disk_cache=disk_cache), "r")
    nwbfile = NWBHDF5IO(file=h5).read()
    pos = nwbfile.processing["behavior"]["Position"]["SpatialSeriesLED1"]
    t_pos = pos.timestamps[:]
    xy = pos.data[:].astype(float)
    good = ~np.isnan(xy).any(axis=1)
    t_pos, xy = t_pos[good], xy[good]
    units = nwbfile.units.to_dataframe()
    meta = dict(subject=nwbfile.subject.subject_id,
                session_id=nwbfile.session_id)
    return t_pos, xy, units, meta


def compute_speed(t_pos, xy, smooth_win_s=0.4):
    """Speed from forward differences, boxcar-smoothed, aligned to t_pos."""
    dt = np.median(np.diff(t_pos))
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t_pos)
    sp = np.concatenate([[sp[0]], sp])
    k = max(1, int(round(smooth_win_s / dt)))
    return np.convolve(sp, np.ones(k) / k, mode="same")


def session_occupancy(t_pos, xy, speed, bin_width=BIN_WIDTH):
    """Speed-filtered occupancy map (seconds per bin, masked where unvisited)."""
    move = speed > SPEED_THRESH
    t_mov, xy_mov = t_pos[move], xy[move]
    limits = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    arena_size = (limits[1] - limits[0], limits[3] - limits[2])
    occ, coverage, bin_edges = opa.spatial_occupancy(
        t_mov, xy_mov.T, arena_size, bin_width=bin_width, limits=limits)
    return occ, limits, arena_size, move, coverage


def rate_map_for_spikes(st, t_pos, xy, occ, limits, arena_size, bin_width=BIN_WIDTH):
    """Smoothed firing-rate map for one unit's spike times."""
    sx = np.interp(st, t_pos, xy[:, 0])
    sy = np.interp(st, t_pos, xy[:, 1])
    spikes_tracking = np.array([st, sx, sy])
    rmap = opa.rate_map(occ, spikes_tracking, arena_size,
                        bin_width=bin_width, limits=limits)
    return opg.smooth(rmap, sigma=SMOOTH_SIGMA)


def grid_stats_for_map(rmap, bin_width=BIN_WIDTH):
    """Spatial autocorrelogram and Moser-lab gridness score of a rate map."""
    acorr = opa.autocorrelation(rmap)
    gs, gstats = opa.grid_score(acorr, bin_width=bin_width)
    return acorr, gs, gstats


def spatial_information(rmap, occ):
    """Skaggs spatial information (bits/spike) and mean rate (Hz)."""
    r = np.asarray(rmap, dtype=float)
    o = np.asarray(occ, dtype=float)
    valid = ~np.ma.getmaskarray(rmap) & np.isfinite(r) & (o > 0)
    r, o = r[valid], o[valid]
    p = o / o.sum()
    mean_rate = np.sum(p * r)
    if mean_rate <= 0:
        return np.nan, np.nan
    nz = r > 0
    si = np.sum(p[nz] * (r[nz] / mean_rate) * np.log2(r[nz] / mean_rate))
    return si, mean_rate


def circular_shift_shuffle(st, t_end, rng):
    """Circularly shift spike times by a random offset (>= 20 s)."""
    shift = rng.uniform(SHUFFLE_MIN_SHIFT, t_end - SHUFFLE_MIN_SHIFT)
    return np.mod(st + shift, t_end)


def analyze_session(path, asset_id, n_shuffles=N_SHUFFLES,
                    shuffle_gs_thresh=0.3, seed0=0):
    """Full per-session pipeline: rate maps, grid scores, shuffle tests."""
    t_pos, xy, units, meta = load_session(
        asset_id, cache_dir=f"/tmp/remfile_cache_grid_{asset_id[:8]}")
    speed = compute_speed(t_pos, xy)
    occ, limits, arena_size, move, coverage = session_occupancy(t_pos, xy, speed)
    t_end = t_pos[-1]

    records = []
    for uid, u in units.iterrows():
        st_all = np.asarray(u["spike_times"], dtype=float)
        sp_speed = np.interp(st_all, t_pos, speed)
        st = st_all[sp_speed > SPEED_THRESH]
        rec = dict(session_path=path, asset_id=asset_id,
                   subject=meta["subject"], session_id=meta["session_id"],
                   unit_id=int(uid), unit_name=str(u["unit_name"]),
                   layer=str(u["histology"]), depth=float(u["depth"]),
                   n_spikes_total=int(len(st_all)), n_spikes=int(len(st)),
                   duration_s=float(t_end), coverage=float(coverage),
                   arena_size=tuple(float(a) for a in arena_size))
        if len(st) < 20:
            rec.update(grid_score=np.nan, si=np.nan, mean_rate=np.nan)
            records.append(rec)
            continue
        rmap = rate_map_for_spikes(st, t_pos, xy, occ, limits, arena_size)
        acorr, gs, gstats = grid_stats_for_map(rmap)
        si, mean_rate = spatial_information(rmap, occ)
        rec.update(grid_score=float(gs), si=float(si), mean_rate=float(mean_rate),
                   grid_spacing=float(gstats.get("grid_spacing", np.nan)),
                   grid_orientation=float(gstats.get("grid_orientation", np.nan)),
                   rmap=np.asarray(rmap, dtype=np.float32),
                   rmap_mask=np.ma.getmaskarray(rmap),
                   acorr=np.asarray(acorr, dtype=np.float32))
        if np.isfinite(gs) and gs > shuffle_gs_thresh and len(st) >= MIN_SPIKES and n_shuffles > 0:
            det_seed = zlib.crc32(f"{asset_id}|{int(uid)}".encode()) % 2**31
            rng = np.random.default_rng(seed0 + det_seed)
            sh_gs = np.empty(n_shuffles)
            for s in range(n_shuffles):
                st_sh = circular_shift_shuffle(st, t_end, rng)
                rmap_sh = rate_map_for_spikes(st_sh, t_pos, xy, occ, limits, arena_size)
                _, gs_sh, _ = grid_stats_for_map(rmap_sh)
                sh_gs[s] = gs_sh
            rec["shuffle_gs"] = sh_gs.astype(np.float32)
            rec["grid_p"] = float((np.sum(sh_gs >= gs) + 1) / (n_shuffles + 1))
            rec["shuffle_gs_p95"] = float(np.percentile(sh_gs, 95))
        records.append(rec)
    return records

# %% [markdown]
# ## Example session: raw data inspection
#
# We use session `sub-11265_ses-16030604` (14 units, all histologically assigned
# to MEC layer II, 20 min in a 1.5 m box) to develop and validate the pipeline.

# %%
example_path = "sub-11265/sub-11265_ses-16030604_behavior+ecephys.nwb"
example_row = next(r for r in rows if r[0] == example_path)
t_pos, xy, units_ex, meta_ex = load_session(example_row[1])
speed_ex = compute_speed(t_pos, xy)

print(f"subject {meta_ex['subject']}, session {meta_ex['session_id']}")
print(f"{len(units_ex)} units, {len(t_pos)} position samples, "
      f"{t_pos[-1]/60:.0f} min, arena {np.ptp(xy[:,0]):.0f} x {np.ptp(xy[:,1]):.0f} cm")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
ax.plot(xy[:, 0], xy[:, 1], lw=0.25, color="0.4")
ax.set_title("Trajectory (head LED, 50 Hz)")
ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
ax.set_aspect("equal")
ax = axes[1]
ax.hist(speed_ex, bins=np.arange(0, 80, 1), color="0.4")
ax.axvline(SPEED_THRESH, color="r", ls="--", label=f"{SPEED_THRESH} cm/s threshold")
ax.set_title("Running speed")
ax.set_xlabel("speed (cm/s)"); ax.set_ylabel("frames")
ax.legend()
fig.suptitle(f"Example session: {meta_ex['subject']} / {meta_ex['session_id']}")
fig.savefig(f"{FIGDIR}/fig1_example_session_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# The rat covers the arena thoroughly and runs most of the time (median speed
# ~22 cm/s). Position samples below 2.5 cm/s (immobility) are excluded from the
# rate maps.

# %% [markdown]
# ## Example session: rate maps, autocorrelograms, grid scores
#
# For each unit we compute:
# - the **firing-rate map** (spikes per bin / occupancy per bin, 2.5 cm bins,
#   5 cm Gaussian smoothing),
# - the **spatial autocorrelogram** of the rate map,
# - the **gridness score**: the minimum correlation of the autocorrelogram with
#   itself rotated by 60° and 120° minus the maximum correlation at 30°, 90°
#   and 150° (opexebo implementation of the standard Moser-lab metric). A
#   hexagonal grid gives high correlation at 60°/120° and low at 30°/90°/150°,
#   hence a strongly positive score.

# %%
occ_ex, limits_ex, arena_ex, move_ex, cov_ex = session_occupancy(t_pos, xy, speed_ex)
print(f"arena coverage: {cov_ex:.2%}")

example_units = []
for uid, u in units_ex.iterrows():
    st_all = np.asarray(u["spike_times"], dtype=float)
    st = st_all[np.interp(st_all, t_pos, speed_ex) > SPEED_THRESH]
    rmap = rate_map_for_spikes(st, t_pos, xy, occ_ex, limits_ex, arena_ex)
    acorr, gs, gstats = grid_stats_for_map(rmap)
    sx = np.interp(st, t_pos, xy[:, 0])
    sy = np.interp(st, t_pos, xy[:, 1])
    example_units.append(dict(name=u["unit_name"], st=st, sx=sx, sy=sy,
                              rmap=rmap, acorr=acorr, gs=gs))
    print(f"  {u['unit_name']}: {len(st)} spikes, grid score {gs:.2f}")

# %%
n = len(example_units)
fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.4))
for j, r in enumerate(example_units):
    ax = axes[0, j]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.2, color="0.8", zorder=1)
    ax.scatter(r["sx"], r["sy"], s=1, c="r", zorder=2)
    ax.set_title(r["name"], fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax = axes[1, j]
    ax.imshow(r["rmap"], origin="lower", cmap="jet")
    ax.set_title(f"{len(r['st'])} sp", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2, j]
    ax.imshow(r["acorr"], origin="lower", cmap="jet")
    ax.set_title(f"gs={r['gs']:.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for a, lab in zip(axes[:, 0], ["trajectory + spikes", "rate map", "autocorrelogram"]):
    a.set_ylabel(lab, fontsize=9)
fig.suptitle("Example session sub-11265 ses-16030604 (all units MEC layer II)", y=1.01)
fig.savefig(f"{FIGDIR}/fig2_example_session_units.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# Several layer II units (e.g. t1c1, t3c1, t3c2, t3c3, t3c5) show multiple
# firing fields whose autocorrelograms display the classic six-fold symmetric
# hexagonal pattern, with grid scores up to 1.16. Other units fire in single
# blobs or diffusely and score near or below zero.

# %% [markdown]
# ## Population analysis across all 118 sessions
#
# We now run the same pipeline on every session of the dandiset. For every unit
# with a grid score above 0.3 we additionally run 100 circular time-shift
# shuffles of the spike train (preserving its temporal structure while
# decoupling it from position) and compute an empirical p-value: the fraction
# of shuffles whose grid score reaches the observed one. Results are cached per
# session in `results/` so re-running this notebook is fast.

# %%
def session_tag(path):
    return os.path.basename(path).replace("_behavior+ecephys.nwb", "")


def work(path, asset_id):
    out = os.path.join(RESULTS_DIR, session_tag(path) + ".pkl")
    if os.path.exists(out):
        return path, "cached"
    recs = analyze_session(path, asset_id)
    with open(out, "wb") as f:
        pickle.dump(recs, f)
    return path, f"{len(recs)} units"


to_run = [(p, a) for (p, a, s) in rows]
done = sum(os.path.exists(os.path.join(RESULTS_DIR, session_tag(p) + ".pkl"))
           for p, a in to_run)
print(f"{len(to_run)} sessions, {done} already cached")
if done < len(to_run):
    # fork context so this works both as a script and inside Jupyter
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as ex:
        futs = {ex.submit(work, p, a): p for p, a in to_run}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="sessions"):
            tqdm.write(fut.result()[0])

# %%
# load all per-unit records
all_recs = []
for f in sorted(os.listdir(RESULTS_DIR)):
    if f.endswith(".pkl"):
        with open(os.path.join(RESULTS_DIR, f), "rb") as fh:
            all_recs.extend(pickle.load(fh))
print(f"{len(all_recs)} units from {len(set(r['session_path'] for r in all_recs))} sessions")

valid = [r for r in all_recs if r.get("n_spikes", 0) >= MIN_SPIKES
         and np.isfinite(r.get("grid_score", np.nan))]
print(f"{len(valid)} units with >= {MIN_SPIKES} spikes while moving")

# %% [markdown]
# ## Population results

# %%
gs_all = np.array([r["grid_score"] for r in valid])
layers_all = np.array([r["layer"] for r in valid])
tested = [r for r in valid if "grid_p" in r]
sig = [r for r in tested if r["grid_p"] < 0.05]
print(f"grid score > 0.4 (classic criterion): {np.mean(gs_all > 0.4):.1%} of units")
print(f"shuffle-tested: {len(tested)}; significant at p<0.05: {len(sig)} "
      f"({len(sig)/len(tested):.1%} of tested)")
for lay in LAYER_ORDER:
    m = layers_all == lay
    n_sig_lay = sum(1 for r in sig if r["layer"] == lay)
    print(f"  {lay}: n={m.sum()}, median gs={np.median(gs_all[m]):.2f}, "
          f"frac gs>0.4={np.mean(gs_all[m] > 0.4):.2f}, shuffle-sig={n_sig_lay}")

# %% [markdown]
# ### Figure 3: gallery of the strongest grid cells

# %%
gallery = sorted(sig, key=lambda r: -r["grid_score"])[:12]
fig, axes = plt.subplots(2, 12, figsize=(20, 4.4))
for j, r in enumerate(gallery):
    rm = np.ma.array(r["rmap"], mask=r["rmap_mask"])
    axes[0, j].imshow(rm, origin="lower", cmap="jet")
    axes[0, j].set_title(f"{r['subject']}\n{r['unit_name']} {r['layer'].replace('MEC ', '')}",
                         fontsize=7)
    axes[1, j].imshow(r["acorr"], origin="lower", cmap="jet")
    axes[1, j].set_title(f"gs={r['grid_score']:.2f}, p={r['grid_p']:.3f}", fontsize=7)
    for a in (axes[0, j], axes[1, j]):
        a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_ylabel("rate map", fontsize=9)
axes[1, 0].set_ylabel("autocorrelogram", fontsize=9)
fig.suptitle("Twelve strongest shuffle-significant grid cells (one column each)")
fig.savefig(f"{FIGDIR}/fig3_grid_cell_gallery.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 4: grid scores across the population and across layers

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
bins = np.arange(-1.2, 2.0, 0.1)
ax.hist(gs_all, bins=bins, color="0.4")
ax.axvline(0.4, color="r", ls="--", label="classic threshold 0.4")
ax.set_xlabel("grid score"); ax.set_ylabel("units")
ax.set_title(f"All units (n={len(valid)})")
ax.legend()

ax = axes[1]
for lay in LAYER_ORDER:
    m = layers_all == lay
    ax.hist(gs_all[m], bins=bins, histtype="step", lw=1.5,
            color=LAYER_COLORS[lay], label=f"{lay} (n={m.sum()})")
ax.axvline(0.4, color="0.5", ls=":", lw=1)
ax.set_xlabel("grid score"); ax.set_ylabel("units")
ax.set_title("By MEC layer")
ax.legend(fontsize=8)

ax = axes[2]
obs = np.array([r["grid_score"] for r in tested])
p95 = np.array([r["shuffle_gs_p95"] for r in tested])
ps = np.array([r["grid_p"] for r in tested])
ax.scatter(p95[ps >= 0.05], obs[ps >= 0.05], s=12, color="0.6",
           label=f"p>=0.05 (n={np.sum(ps >= 0.05)})")
ax.scatter(p95[ps < 0.05], obs[ps < 0.05], s=12, color="#d62728",
           label=f"p<0.05 (n={np.sum(ps < 0.05)})")
lim = (min(p95.min(), obs.min()) - 0.1, max(obs.max(), p95.max()) + 0.1)
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("shuffle 95th percentile"); ax.set_ylabel("observed grid score")
ax.set_title("Shuffle validation")
ax.legend(fontsize=8)

fig.savefig(f"{FIGDIR}/fig4_population_grid_scores.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 5: grid spacing and orientation of significant grid cells

# %%
sig_gs = np.array([r["grid_score"] for r in sig])
spacing = np.array([r.get("grid_spacing", np.nan) for r in sig])
orient = np.array([r.get("grid_orientation", np.nan) for r in sig])
ok = np.isfinite(spacing)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.hist(spacing[ok], bins=np.arange(0, 200, 5), color="0.4")
ax.set_xlabel("grid spacing (cm)"); ax.set_ylabel("grid cells")
ax.set_title(f"Grid spacing (n={ok.sum()})")

ax = axes[1]
ax.scatter(spacing[ok], sig_gs[ok], s=14, color="#1f77b4")
ax.set_xlabel("grid spacing (cm)"); ax.set_ylabel("grid score")
ax.set_title("Spacing vs. gridness")

ax = axes[2]
ok_o = np.isfinite(orient)
ax.hist(orient[ok_o] % 60, bins=np.arange(0, 61, 5), color="0.4")
ax.set_xlabel("grid orientation (deg, mod 60)"); ax.set_ylabel("grid cells")
ax.set_title("Grid orientation")

fig.savefig(f"{FIGDIR}/fig5_grid_spacing_orientation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### Figure 6: spatial information vs. grid score

# %%
si_all = np.array([r.get("si", np.nan) for r in valid])
ok = np.isfinite(si_all)
fig, ax = plt.subplots(figsize=(6.5, 5))
for lay in LAYER_ORDER:
    m = (layers_all == lay) & ok
    ax.scatter(si_all[m], gs_all[m], s=12, alpha=0.6,
               color=LAYER_COLORS[lay], label=lay)
ax.axhline(0.4, color="0.5", ls=":", lw=1)
ax.set_xlabel("spatial information (bits/spike)")
ax.set_ylabel("grid score")
ax.set_title("Spatial information vs. gridness")
ax.legend(fontsize=8)
fig.savefig(f"{FIGDIR}/fig6_spatial_info_vs_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# - We analyzed all 118 sessions (620 tetrode units) of DANDI 000582, the
#   Sargolini et al. (2006) MEC dataset, streaming NWB files with remfile.
# - A substantial fraction of MEC units, concentrated in layer II, show
#   hexagonal grid-like firing: multiple fields, six-fold symmetric
#   autocorrelograms, and grid scores far above a circular-shift shuffle null.
# - Grid spacing and orientation estimates for the significant cells are
#   consistent with the original report (tens of cm spacing in these arenas,
#   oriented lattices).
#
# Grid cells are therefore robustly present and straightforward to demonstrate
# in this public dataset with standard rate-map and gridness-score methods.
