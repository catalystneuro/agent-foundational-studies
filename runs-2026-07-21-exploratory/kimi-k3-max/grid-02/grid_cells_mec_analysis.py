# %% [markdown]
# # Grid Cells in the Medial Entorhinal Cortex
#
# This notebook demonstrates grid cells in the medial entorhinal cortex (MEC)
# using data from the DANDI Archive. Grid cells are neurons that fire when an
# animal passes through any vertex of a regular hexagonal lattice spanning the
# environment, and they are a signature cell type of the MEC (Hafting et al.,
# 2005, Nature).
#
# **Dataset**: DANDI dandiset
# [000582](https://dandiarchive.org/dandiset/000582), the NWB conversion of
# Sargolini et al. (2006, Science), "Conjunctive Representation of Position,
# Direction, and Velocity in Entorhinal Cortex" from the Moser lab. It contains
# 118 sessions from 15 Long Evans rats foraging in square or circular arenas,
# with tetrode recordings from MEC layers II, III, V, and VI, and 50 Hz 2D
# position tracking.
#
# **Approach**:
# 1. Stream one example session with remfile and inspect it with Pynapple.
# 2. Compute firing rate maps, spatial autocorrelograms, and grid scores with
#    opexebo (the Moser lab analysis package).
# 3. Establish statistical significance with circular time-shift shuffles.
# 4. Scale to all 118 sessions to quantify grid cell prevalence by cortical
#    layer and the distribution of grid spacing across the population.
#
# All data are streamed from the archive with local disk caching; nothing is
# downloaded in full. Runtime is roughly 7 minutes on 8 cores, most of it in
# the population stage.

# %% [markdown]
# ## Setup

# %%
import os
import json
import pickle
import time
import zlib
import warnings
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt

import h5py
import remfile
import requests
from pynwb import NWBHDF5IO
import pynapple as nap
import opexebo.analysis as opa
import opexebo.general as opg
from tqdm import tqdm

# Analysis parameters (standard for this literature)
BIN_WIDTH = 2.5        # cm, spatial bin width for rate maps
SMOOTH_SIGMA = 2       # bins, Gaussian smoothing of the rate map
MIN_SPIKES = 200       # minimum spike count for a unit to be analyzed
GS_CANDIDATE = 0.3     # grid score above which a unit is shuffle-tested
N_SHUFFLES = 100       # circular time shifts per candidate unit
MIN_SHIFT_S = 30.0     # minimum shift, so shuffles decorrelate spikes from position
N_WORKERS = min(8, os.cpu_count())
CACHE_ROOT = "/tmp/remfile_cache_grid_cells"
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# opexebo's grid_score calls scipy.stats.pearsonr on ring samples of the
# autocorrelogram; constant rings trigger a benign RuntimeWarning. NaN grid
# scores are handled explicitly below, so the warning is noise.
warnings.filterwarnings("ignore", category=RuntimeWarning)

DANDISET = "000582"
DANDISET_VERSION = "0.251111.2151"
DEMO_SESSION = "sub-11265/sub-11265_ses-16030604_behavior+ecephys.nwb"

# %% [markdown]
# ## Listing the Dataset Assets
#
# We query the DANDI REST API for the asset list of the published version of
# the dandiset. Each asset is one session's NWB file. The listing is cached to
# `assets.json` so re-runs do not hit the API again.

# %%
if os.path.exists("assets.json"):
    ASSETS = json.load(open("assets.json"))
else:
    url = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{DANDISET_VERSION}/assets/"
    ASSETS = []
    while url:
        r = requests.get(url, params={"page_size": 1000})
        r.raise_for_status()
        d = r.json()
        ASSETS.extend({"asset_id": a["asset_id"], "path": a["path"]} for a in d["results"])
        url = d["next"]
    json.dump(ASSETS, open("assets.json", "w"), indent=1)

print(f"{len(ASSETS)} sessions in dandiset {DANDISET} v{DANDISET_VERSION}")
ASSET_IDS = {a["path"]: a["asset_id"] for a in ASSETS}


# %% [markdown]
# ## Streaming One Session
#
# NWB files are opened through remfile, which performs HTTP range requests
# against the DANDI download URL and caches blocks on disk. Pynapple then wraps
# the NWB objects so spike trains and position behave as time series. The
# example session has 14 putative MEC layer II units and 20 minutes of
# tracking in a 1.5 m square box.

# %%
def load_session(session_path):
    """Open a session NWB file by streaming; return (pynapple NWB, pynwb file, h5 handle)."""
    url = f"https://api.dandiarchive.org/api/assets/{ASSET_IDS[session_path]}/download/"
    cache = remfile.DiskCache(CACHE_ROOT)
    h5 = h5py.File(remfile.File(url, disk_cache=cache), "r")
    nwbfile = NWBHDF5IO(file=h5).read()
    return nap.NWBFile(nwbfile), nwbfile, h5


def get_position(nwb):
    """Extract 2D position as (t, xy) with NaN frames removed.

    The NWB metadata claims meters, but values are centimeters: the 1.5 m box
    spans +/-75. A handful of frames per session are NaN (lost tracking) and
    are dropped here.
    """
    pos = nwb["SpatialSeriesLED1"]
    t = np.asarray(pos.index)
    xy = np.asarray(pos.values, dtype=float)
    good = ~np.isnan(xy).any(axis=1)
    return t[good], xy[good]


nwb, nwbfile, h5 = load_session(DEMO_SESSION)
print(nwb)
t, xy = get_position(nwb)
units = nwb["units"]
histology = [str(nwbfile.units["histology"][u]) for u in range(len(units))]
print(f"\nsession duration: {(t[-1]-t[0])/60:.1f} min, "
      f"position range x [{xy[:,0].min():.0f}, {xy[:,0].max():.0f}] cm, "
      f"y [{xy[:,1].min():.0f}, {xy[:,1].max():.0f}] cm")
print(f"{len(units)} units, histology labels: {sorted(set(histology))}")

# %% [markdown]
# ## Raw Data Overview
#
# Before any analysis we look at the raw streams: the foraging trajectory
# covers the box densely, running speed is in a plausible range for a rat
# (median ~22 cm/s, almost all frames above the 2.5 cm/s stillness threshold
# used in this literature), and all 14 units fire throughout the session.

# %%
dt = np.diff(t)
speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
speed_t = (t[:-1] + t[1:]) / 2

fig, axes = plt.subplots(3, 1, figsize=(11, 9),
                         gridspec_kw={"height_ratios": [1.2, 1, 1]})
ax = axes[0]
ax.plot(xy[:, 0], xy[:, 1], lw=0.3, color="0.4")
ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
ax.set_title("Trajectory (SpatialSeriesLED1, 50 Hz, 20 min, 1.5 m box)")
ax.set_aspect("equal")

ax = axes[1]
seg = (speed_t >= 100) & (speed_t <= 160)
ax.plot(speed_t[seg], speed[seg], lw=0.5, color="k")
ax.axhline(2.5, color="r", ls="--", lw=0.8, label="2.5 cm/s threshold")
ax.set_xlabel("time (s)"); ax.set_ylabel("speed (cm/s)")
ax.set_title("Running speed (60 s window)")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, 120)

ax = axes[2]
for i in range(len(units)):
    spk = np.asarray(units[i].index)
    spk = spk[(spk >= 100) & (spk <= 160)]
    ax.plot(spk, np.full_like(spk, i), "|", ms=4, color=f"C{i % 10}")
ax.set_yticks(range(len(units)))
ax.set_xlabel("time (s)"); ax.set_ylabel("unit #")
ax.set_title(f"Spike raster, all {len(units)} MEC LII units (same 60 s window)")
ax.set_ylim(-0.7, len(units) - 0.3)

fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig01_raw_data_overview.png", dpi=150)
plt.close(fig)
print(f"median speed {np.median(speed):.1f} cm/s; "
      f"{100*np.mean(speed > 2.5):.1f}% of frames above 2.5 cm/s")

# %% [markdown]
# ## Rate Maps, Autocorrelograms, and Grid Scores
#
# The standard grid cell pipeline (Hafting et al., 2005) has three steps.
# First, a firing rate map: the arena is divided into 2.5 cm bins, and each
# bin's rate is the number of spikes observed there divided by the time the
# animal spent there, smoothed with a Gaussian. Second, the spatial
# autocorrelogram of the rate map: a hexagonal firing pattern produces an
# autocorrelogram with a central peak surrounded by six peaks on a hexagon.
# Third, the grid score: the correlation of an annular crop of the
# autocorrelogram with itself rotated by 60 and 120 degrees (where a hexagonal
# pattern correlates positively) minus the correlation at 30, 90, and 150
# degrees (where it correlates negatively). We use the opexebo package, which
# implements the Moser lab's versions of these measures.

# %%
def session_occupancy(t, xy):
    """Occupancy map (seconds per bin) with data-driven arena limits."""
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    limits = (xmin, xmax, ymin, ymax)
    arena_size = float(max(xmax - xmin, ymax - ymin))
    occ, coverage, _ = opa.spatial_occupancy(t, xy.T, arena_size,
                                             bin_width=BIN_WIDTH, limits=limits)
    return occ, arena_size, limits, coverage


def unit_grid_score(spk, t, xy, occ, arena_size, limits):
    """Rate map -> smoothed map -> autocorrelogram -> grid score for one unit."""
    sx = np.interp(spk, t, xy[:, 0])
    sy = np.interp(spk, t, xy[:, 1])
    spikes_tracking = np.vstack([spk, sx, sy])
    rmap = opa.rate_map(occ, spikes_tracking, arena_size,
                        bin_width=BIN_WIDTH, limits=limits)
    srmap = opg.smooth(rmap, sigma=SMOOTH_SIGMA)
    acorr = opa.autocorrelation(srmap)
    gs, stats = opa.grid_score(acorr, bin_width=BIN_WIDTH)
    return gs, stats, srmap, acorr


occ, arena_size, limits, coverage = session_occupancy(t, xy)
print(f"occupancy map {occ.shape}, arena coverage {100*coverage:.0f}%")

demo_results = []
for u in range(len(units)):
    spk = np.asarray(units[u].index)
    spk = spk[(spk >= t[0]) & (spk <= t[-1])]
    gs, stats, srmap, acorr = unit_grid_score(spk, t, xy, occ, arena_size, limits)
    demo_results.append(dict(unit=u, n_spikes=len(spk), grid_score=gs,
                             spacing=stats["grid_spacing"], srmap=srmap, acorr=acorr))
    print(f"unit {u:2d}: {len(spk):5d} spikes, grid score {gs:+.3f}, "
          f"spacing {stats['grid_spacing']:.0f} cm")

# %% [markdown]
# ### All Units in the Example Session
#
# Even within one tetrode session the cell types are mixed: several units show
# the textbook hexagonal autocorrelogram (units 4, 5, 6, 8, 12), while others
# fire in single fields or without spatial structure. Grid score separates
# these groups cleanly.

# %%
n = len(demo_results)
fig, axes = plt.subplots(4, 7, figsize=(17, 10))
for i, r in enumerate(demo_results):
    ax = axes[(i // 7) * 2, i % 7]
    ax.imshow(r["srmap"], origin="lower", cmap="jet")
    ax.set_title(f"u{r['unit']}  gs={r['grid_score']:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if i % 7 == 0:
        ax.set_ylabel("rate map", fontsize=9)
    ax = axes[(i // 7) * 2 + 1, i % 7]
    ax.imshow(r["acorr"], origin="lower", cmap="jet")
    ax.set_title(f"{r['n_spikes']} spk, {r['spacing']:.0f} cm", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if i % 7 == 0:
        ax.set_ylabel("autocorr", fontsize=9)
fig.suptitle("Session sub-11265_ses-16030604: smoothed rate maps and spatial "
             "autocorrelograms (all 14 MEC LII units)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIGDIR}/fig02_demo_session_ratemaps.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Example Grid Cells in Detail
#
# The four highest-scoring units, shown as spikes superimposed on the
# trajectory, as smoothed rate maps, and as autocorrelograms. The hexagonal
# lattice is visible directly in the spike positions, which is the defining
# feature of a grid cell.

# %%
show = [6, 4, 5, 8]  # top four units by grid score in this session
fig, axes = plt.subplots(len(show), 3, figsize=(11, 3.4 * len(show)))
for row, u in enumerate(show):
    spk = np.asarray(units[u].index)
    spk = spk[(spk >= t[0]) & (spk <= t[-1])]
    sx = np.interp(spk, t, xy[:, 0])
    sy = np.interp(spk, t, xy[:, 1])
    r = demo_results[u]

    ax = axes[row, 0]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.25, color="0.6", zorder=1)
    ax.scatter(sx, sy, s=2, c="r", zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"unit {u}", fontsize=11)
    if row == 0:
        ax.set_title("spikes on trajectory", fontsize=11)

    ax = axes[row, 1]
    ax.imshow(r["srmap"], origin="lower", cmap="jet")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.95, f"peak {np.nanmax(r['srmap']):.1f} Hz",
            transform=ax.transAxes, color="w", fontsize=8, va="top")
    if row == 0:
        ax.set_title("smoothed rate map", fontsize=11)

    ax = axes[row, 2]
    ax.imshow(r["acorr"], origin="lower", cmap="jet")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.95, f"grid score {r['grid_score']:.2f}\nspacing {r['spacing']:.0f} cm",
            transform=ax.transAxes, color="w", fontsize=8, va="top")
    if row == 0:
        ax.set_title("spatial autocorrelogram", fontsize=11)

fig.suptitle("Grid cells from MEC layer II (sub-11265_ses-16030604)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIGDIR}/fig03_example_grid_cells.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Statistical Significance by Circular Time Shifting
#
# A high grid score can arise by chance in structured data, so each candidate
# is tested against a shuffle distribution. The spike train is shifted in time
# by a random offset (at least 30 s, wrapping around the end of the session),
# which destroys the relationship between spikes and position while preserving
# the spike train's temporal statistics and the trajectory's spatial
# statistics. The grid score is recomputed for each shift, and the p-value is
# the fraction of shifts whose score reaches the observed one. For the best
# unit in the example session the observed score sits far outside the shuffle
# distribution.

# %%
u = 6
spk = np.asarray(units[u].index)
spk = spk[(spk >= t[0]) & (spk <= t[-1])]
T = t[-1] - t[0]
gs_obs = demo_results[u]["grid_score"]

rng = np.random.default_rng(42)
N_SHUFFLES_DEMO = 500
sh = np.empty(N_SHUFFLES_DEMO)
for i in tqdm(range(N_SHUFFLES_DEMO), desc="shuffles"):
    off = rng.uniform(MIN_SHIFT_S, T - MIN_SHIFT_S)
    spk_s = (spk - t[0] + off) % T + t[0]
    gs_s, _, _, _ = unit_grid_score(spk_s, t, xy, occ, arena_size, limits)
    sh[i] = gs_s if np.isfinite(gs_s) else -np.inf
p_obs = (np.sum(sh >= gs_obs) + 1) / (N_SHUFFLES_DEMO + 1)
print(f"unit {u}: observed grid score {gs_obs:.3f}, "
      f"shuffle 95th percentile {np.percentile(sh, 95):.3f}, p = {p_obs:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(sh, bins=40, color="0.6", edgecolor="w", lw=0.5,
        label=f"{N_SHUFFLES_DEMO} circular time shifts")
ax.axvline(gs_obs, color="r", lw=2, label=f"observed grid score = {gs_obs:.2f}")
ax.axvline(np.percentile(sh, 95), color="k", ls="--", lw=1,
           label=f"shuffle 95th pct = {np.percentile(sh, 95):.2f}")
ax.set_xlabel("grid score"); ax.set_ylabel("count")
ax.set_title(f"Shuffle test, unit {u} (sub-11265_ses-16030604): p = {p_obs:.4f}")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig04_shuffle_test.png", dpi=150)
plt.close(fig)

h5.close()

# %% [markdown]
# ## Population Analysis Across All 118 Sessions
#
# The same pipeline now runs on every session in the dandiset: 620 tetrode
# units from 15 rats. Units with at least 200 spikes are scored; units scoring
# above 0.3 are shuffle-tested with 100 circular shifts each. Sessions are
# processed in parallel processes, each with its own remfile disk cache. This
# stage takes about 4 minutes on 8 cores and writes `population_results.pkl`;
# if that file already exists it is loaded instead of recomputed.

# %%
def init_worker(counter):
    global WORKER_ID
    with counter.get_lock():
        WORKER_ID = counter.value
        counter.value += 1


def process_session(args):
    """Score every unit in one session; shuffle-test the candidates."""
    idx, asset = args
    global WORKER_ID
    path = asset["path"]
    url = f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/download/"
    cache = remfile.DiskCache(os.path.join(CACHE_ROOT, f"w{WORKER_ID}"))
    h5 = h5py.File(remfile.File(url, disk_cache=cache), "r")
    nwbfile = NWBHDF5IO(file=h5).read()
    nwb = nap.NWBFile(nwbfile)

    t, xy = get_position(nwb)
    T = t[-1] - t[0]
    occ, arena_size, limits, _ = session_occupancy(t, xy)

    units = nwb["units"]
    seed = zlib.crc32(path.encode()) % (2**32)
    out = []
    for u in range(len(units)):
        layer = str(nwbfile.units["histology"][u]) or "unknown"
        spk = np.asarray(units[u].index)
        spk = spk[(spk >= t[0]) & (spk <= t[-1])]
        rec = dict(session=path, subject=path.split("/")[0], unit=u,
                   histology=layer, n_spikes=int(len(spk)), duration_s=float(T),
                   arena_cm=arena_size, grid_score=np.nan, spacing=np.nan,
                   orientation=np.nan, p_value=np.nan, n_shuffles=0)
        if len(spk) >= MIN_SPIKES:
            gs, stats, _, _ = unit_grid_score(spk, t, xy, occ, arena_size, limits)
            rec["grid_score"] = float(gs) if gs is not None else np.nan
            rec["spacing"] = float(stats.get("grid_spacing", np.nan))
            rec["orientation"] = float(stats.get("grid_orientation", np.nan))
            if np.isfinite(gs) and gs > GS_CANDIDATE and T > 2 * MIN_SHIFT_S + 60:
                rng = np.random.default_rng(seed + u)
                sh = np.empty(N_SHUFFLES)
                for i in range(N_SHUFFLES):
                    off = rng.uniform(MIN_SHIFT_S, T - MIN_SHIFT_S)
                    spk_s = (spk - t[0] + off) % T + t[0]
                    gs_s, _, _, _ = unit_grid_score(spk_s, t, xy, occ, arena_size, limits)
                    sh[i] = gs_s if (gs_s is not None and np.isfinite(gs_s)) else -np.inf
                rec["p_value"] = float((np.sum(sh >= gs) + 1) / (N_SHUFFLES + 1))
                rec["n_shuffles"] = N_SHUFFLES
        out.append(rec)
    h5.close()
    return path, out


USE_CACHE = True  # set False to force recomputation of the population stage
if USE_CACHE and os.path.exists("population_results.pkl"):
    results = pickle.load(open("population_results.pkl", "rb"))["results"]
    print(f"loaded {len(results)} cached unit records")
else:
    t0 = time.time()
    counter = mp.get_context("fork").Value("i", 0)
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(N_WORKERS, initializer=init_worker, initargs=(counter,)) as pool:
        for i, (path, recs) in enumerate(pool.imap_unordered(
                process_session, list(enumerate(ASSETS))), 1):
            results.extend(recs)
            print(f"[{i}/{len(ASSETS)}] {path}: {len(recs)} units "
                  f"({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(dict(results=results), open("population_results.pkl", "wb"))
    print(f"population stage took {time.time()-t0:.0f}s")

analyzed = [r for r in results if r["n_spikes"] >= MIN_SPIKES]
significant = [r for r in analyzed if r["p_value"] < 0.05]
print(f"\n{len(results)} units total, {len(analyzed)} with >= {MIN_SPIKES} spikes, "
      f"{len(significant)} significant grid cells (p < 0.05)")

# %% [markdown]
# ### Population Results
#
# Across the population, grid scores are bimodal: a large group of units near
# zero with no hexagonal structure, and a second group with scores above ~0.5.
# Grid cells are concentrated in the superficial layers: about half of layer II
# and layer III units pass the shuffle test, against about a fifth in layers V
# and VI, the pattern reported by Sargolini et al. (2006). Grid spacing is
# broadly distributed around a median near 60 cm, consistent with the dorso-
# ventral range of recording sites in this dataset.

# %%
LAYER_ORDER = ["MEC LII", "MEC LIII", "MEC LV", "MEC LVI"]
LAYER_COLORS = {"MEC LII": "#d62728", "MEC LIII": "#1f77b4",
                "MEC LV": "#2ca02c", "MEC LVI": "#9467bd"}

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

ax = axes[0, 0]
bins = np.arange(-1, 1.55, 0.1)
bottoms = np.zeros(len(bins) - 1)
for layer in LAYER_ORDER:
    v = np.array([r["grid_score"] for r in analyzed if r["histology"] == layer])
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=bins)
    ax.bar(bins[:-1], h, width=0.1, bottom=bottoms, color=LAYER_COLORS[layer],
           label=layer.replace("MEC ", ""), align="edge", edgecolor="w", lw=0.3)
    bottoms += h
ax.axvline(GS_CANDIDATE, color="k", ls=":", lw=1)
ax.text(GS_CANDIDATE + 0.03, bottoms.max() * 0.75,
        "shuffle-tested\ncandidates > 0.3", fontsize=7)
ax.set_xlabel("grid score"); ax.set_ylabel("number of units")
ax.set_title(f"a. Grid score distribution (n = {len(analyzed)} units, "
             f"{len(ASSETS)} sessions, 15 rats)")
ax.legend(frameon=False, title="layer", fontsize=9, title_fontsize=9)

ax = axes[0, 1]
fracs, ns, sigs = [], [], []
for layer in LAYER_ORDER:
    sub = [r for r in analyzed if r["histology"] == layer]
    s = [r for r in sub if r["p_value"] < 0.05]
    fracs.append(100 * len(s) / len(sub)); ns.append(len(sub)); sigs.append(len(s))
ax.bar(range(4), fracs, color=[LAYER_COLORS[l] for l in LAYER_ORDER],
       edgecolor="k", lw=0.5)
for i, (f, nn, s) in enumerate(zip(fracs, ns, sigs)):
    ax.text(i, f + 1, f"{s}/{nn}", ha="center", fontsize=9)
ax.set_xticks(range(4))
ax.set_xticklabels([l.replace("MEC ", "") for l in LAYER_ORDER])
ax.set_ylabel("% significant grid cells")
ax.set_title("b. Grid cell prevalence by MEC layer (p < 0.05, shuffle test)")
ax.set_ylim(0, 60)

ax = axes[1, 0]
sp = np.array([r["spacing"] for r in significant])
sp = sp[np.isfinite(sp)]
ax.hist(sp, bins=np.arange(20, 140, 5), color="0.45", edgecolor="w", lw=0.5)
ax.axvline(np.median(sp), color="r", lw=1.5, label=f"median = {np.median(sp):.0f} cm")
ax.set_xlabel("grid spacing (cm)"); ax.set_ylabel("number of grid cells")
ax.set_title(f"c. Grid spacing distribution (n = {len(significant)} significant cells)")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
gs_all = np.array([r["grid_score"] for r in analyzed])
pv = np.array([r["p_value"] for r in analyzed])
cand = np.isfinite(pv)
ax.scatter(gs_all[~cand], np.full((~cand).sum(), 0.9), s=6, c="0.7", alpha=0.5,
           label="not shuffle-tested (gs <= 0.3)")
colors = [LAYER_COLORS.get(r["histology"], "0.5")
          for r in np.array(analyzed, dtype=object)[cand]]
ax.scatter(gs_all[cand], pv[cand], s=8, c=colors, alpha=0.7)
ax.axhline(0.05, color="r", ls="--", lw=1, label="p = 0.05")
ax.set_yscale("log"); ax.set_ylim(0.005, 1.2)
ax.set_xlabel("observed grid score"); ax.set_ylabel("shuffle p-value")
ax.set_title("d. Observed grid score vs. shuffle significance")
ax.legend(frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig05_population_summary.png", dpi=150)
plt.close(fig)

for layer in LAYER_ORDER:
    sub = [r for r in analyzed if r["histology"] == layer]
    s = [r for r in sub if r["p_value"] < 0.05]
    print(f"{layer}: {len(s)}/{len(sub)} significant ({100*len(s)/len(sub):.0f}%)")
print(f"median grid spacing {np.median(sp):.0f} cm "
      f"(IQR {np.percentile(sp, 25):.0f}-{np.percentile(sp, 75):.0f} cm)")

# %% [markdown]
# ### The Best Grid Cell from Six Different Rats
#
# Grid cells are not a property of one animal or one session. Reloading the
# highest-scoring significant unit from each of six rats shows the same
# hexagonal structure across subjects, layers, and arena shapes (the dataset
# includes both square and circular arenas).

# %%
sig_sorted = sorted((r for r in significant if np.isfinite(r["grid_score"])),
                    key=lambda r: -r["grid_score"])
seen, picks = set(), []
for r in sig_sorted:
    if r["subject"] not in seen:
        seen.add(r["subject"]); picks.append(r)
    if len(picks) == 6:
        break

fig, axes = plt.subplots(6, 3, figsize=(10, 19))
for row, rec in enumerate(tqdm(picks, desc="example cells")):
    nwb, nwbfile, h5 = load_session(rec["session"])
    t, xy = get_position(nwb)
    occ, arena_size, limits, _ = session_occupancy(t, xy)
    spk = np.asarray(nwb["units"][rec["unit"]].index)
    spk = spk[(spk >= t[0]) & (spk <= t[-1])]
    sx = np.interp(spk, t, xy[:, 0])
    sy = np.interp(spk, t, xy[:, 1])
    _, _, srmap, acorr = unit_grid_score(spk, t, xy, occ, arena_size, limits)
    h5.close()

    ax = axes[row, 0]
    ax.plot(xy[:, 0], xy[:, 1], lw=0.2, color="0.6", zorder=1)
    ax.scatter(sx, sy, s=1.5, c="r", zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(f"{rec['subject'][-5:]} u{rec['unit']}\n"
                  f"{rec['histology'].replace('MEC ', '')}", fontsize=9)
    if row == 0:
        ax.set_title("spikes on trajectory", fontsize=11)

    ax = axes[row, 1]
    ax.imshow(srmap, origin="lower", cmap="jet")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.95, f"peak {np.nanmax(srmap):.1f} Hz",
            transform=ax.transAxes, color="w", fontsize=8, va="top")
    if row == 0:
        ax.set_title("smoothed rate map", fontsize=11)

    ax = axes[row, 2]
    ax.imshow(acorr, origin="lower", cmap="jet")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.95,
            f"gs {rec['grid_score']:.2f}, p={rec['p_value']:.3f}\n"
            f"spacing {rec['spacing']:.0f} cm",
            transform=ax.transAxes, color="w", fontsize=8, va="top")
    if row == 0:
        ax.set_title("spatial autocorrelogram", fontsize=11)

fig.suptitle("Best grid cell from six different rats (DANDI 000582)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(f"{FIGDIR}/fig06_population_examples.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## Summary
#
# This analysis demonstrates grid cells in the medial entorhinal cortex using
# publicly available data from the DANDI Archive. In the example session,
# individual MEC layer II units fire at the vertices of a hexagonal lattice,
# visible directly in the spike positions and quantified by grid scores up to
# 1.16, far outside the circular time-shift shuffle distribution (p = 0.002).
# Across all 118 sessions of Sargolini et al. (2006), 197 of 617 well-isolated
# units (32%) are significant grid cells. Prevalence follows the known
# anatomical gradient: roughly half of units in the superficial layers II
# (48%) and III (44%) are grid cells, compared with about a fifth in the deep
# layers V (19%) and VI (22%). Grid spacing across the population is broadly
# distributed with a median of 58 cm, consistent with the range of dorso-
# ventral recording positions in the original study.
#
# **References**
#
# - Sargolini, F. et al. (2006). Conjunctive representation of position,
#   direction, and velocity in entorhinal cortex. Science 312, 758-762.
# - Hafting, T. et al. (2005). Microstructure of a spatial map in the
#   entorhinal cortex. Nature 436, 801-806.
# - DANDI Archive dandiset 000582: https://dandiarchive.org/dandiset/000582
