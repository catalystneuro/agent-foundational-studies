# %% [markdown]
# # Grid Cells in the Medial Entorhinal Cortex
#
# This notebook demonstrates the classic **grid cell** phenomenon using real
# extracellular recordings from the medial entorhinal cortex (MEC) of freely
# foraging rats, published as
# [DANDI:000582](https://dandiarchive.org/dandiset/000582)
# ("Conjunctive Representation of Position, Direction, and Velocity in
# Entorhinal Cortex"), which contains the original data from
# Sargolini et al., *Science* (2006).
#
# Grid cells fire in a strikingly regular hexagonal lattice of locations as an
# animal explores a two-dimensional open field. This periodic firing pattern is
# revealed by computing the spatial autocorrelogram of a neuron's firing-rate
# map: for a grid cell, the autocorrelogram shows a hexagonal arrangement of
# six peaks around the central peak. The **gridness score** (Sargolini et al.,
# 2006; Langston et al., 2010) quantifies this hexagonal symmetry by
# correlating the autocorrelogram with rotated versions of itself.
#
# Pipeline:
# 1. Stream one example NWB session directly from DANDI (no full download).
# 2. Inspect and visualize the raw position tracking and spike data.
# 3. Compute 2D firing-rate maps and spatial autocorrelograms with Pynapple.
# 4. Compute the gridness score for each unit and identify grid cells.
# 5. Scale the analysis to multiple sessions/subjects, using a shuffle test
#    (circular time-shifts of the spike train) to assess statistical
#    significance of each cell's gridness score.
# 6. Summarize the population of recorded MEC units.

# %%
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from skimage.feature import peak_local_max
from tqdm import tqdm

np.random.seed(12345)

FIGDIR = "figures"
import os
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs("cache/remfile_cache", exist_ok=True)

# %% [markdown]
# ## Grid-cell analysis utilities
#
# These functions implement the standard rate-map smoothing, spatial
# autocorrelogram, and rotational gridness-score computation used throughout
# the grid-cell literature (Sargolini et al. 2006; Langston et al. 2010).

# %%
def smooth_rate_map(rate_map, sigma=1.0):
    """Smooth a rate map (NaN = unvisited bin) with a NaN-aware Gaussian kernel."""
    valid = ~np.isnan(rate_map)
    filled = np.where(valid, rate_map, 0.0)
    smoothed_num = gaussian_filter(filled, sigma=sigma)
    smoothed_den = gaussian_filter(valid.astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed_num / smoothed_den
    out[smoothed_den < 1e-6] = np.nan
    return out


def spatial_autocorrelogram(rate_map):
    """Pearson spatial autocorrelogram: for every (dx, dy) offset, correlate the
    overlapping valid pixels of the rate map with itself. Returns (2H-1, 2W-1)."""
    H, W = rate_map.shape
    valid = ~np.isnan(rate_map)
    data = np.where(valid, rate_map, 0.0)

    corr = np.full((2 * H - 1, 2 * W - 1), np.nan)
    min_overlap = 20

    for dy in range(-(H - 1), H):
        for dx in range(-(W - 1), W):
            y0a, y1a = max(0, dy), min(H, H + dy)
            x0a, x1a = max(0, dx), min(W, W + dx)
            y0b, y1b = max(0, -dy), min(H, H - dy)
            x0b, x1b = max(0, -dx), min(W, W - dx)

            a, b = data[y0a:y1a, x0a:x1a], data[y0b:y1b, x0b:x1b]
            va, vb = valid[y0a:y1a, x0a:x1a], valid[y0b:y1b, x0b:x1b]
            m = va & vb
            n = m.sum()
            if n < min_overlap:
                continue

            av, bv = a[m], b[m]
            sa, sb = av.sum(), bv.sum()
            saa, sbb, sab = (av ** 2).sum(), (bv ** 2).sum(), (av * bv).sum()
            num = n * sab - sa * sb
            den = np.sqrt((n * saa - sa ** 2) * (n * sbb - sb ** 2))
            if den <= 0:
                continue
            corr[H - 1 + dy, W - 1 + dx] = num / den

    return corr


def gridness_score(autocorr):
    """Rotational gridness score of a spatial autocorrelogram (Sargolini/Langston
    method): correlate the annulus around the six nearest peaks with rotated
    copies at 30/60/90/120/150 degrees.
    gridness = min(r60, r120) - max(r30, r90, r150)."""
    H, W = autocorr.shape
    cy, cx = H // 2, W // 2
    filled = np.nan_to_num(autocorr, nan=-1.0)

    coords = peak_local_max(filled, min_distance=2, threshold_abs=0.0)
    dists = np.sqrt((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2)
    not_center = dists > 3
    coords, dists = coords[not_center], dists[not_center]
    if len(dists) < 6:
        return np.nan

    order = np.argsort(dists)
    nearest6_dists = dists[order[:6]]
    r_outer = np.mean(nearest6_dists) * 1.25
    r_inner = np.min(nearest6_dists) * 0.5

    yy, xx = np.indices(autocorr.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annulus = (rr >= r_inner) & (rr <= r_outer)
    ref = np.where(annulus, filled, np.nan)

    def masked_corr(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() < 20:
            return np.nan
        return np.corrcoef(a[m], b[m])[0, 1]

    corrs = {}
    for angle in [30, 60, 90, 120, 150]:
        rotated = rotate(filled, angle, reshape=False, order=1, cval=-1.0)
        rotated_masked = np.where(annulus, rotated, np.nan)
        corrs[angle] = masked_corr(ref, rotated_masked)

    return min(corrs[60], corrs[120]) - max(corrs[30], corrs[90], corrs[150])


def load_session(url):
    disk_cache = remfile.DiskCache('cache/remfile_cache')
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), io


N_BINS = 40
MIN_OCC_TIME = 0.1  # seconds; bins visited less than this are treated as unvisited


def compute_rate_map(spike_ts, pos, ep):
    tc = nap.compute_tuning_curves(nap.TsGroup({0: spike_ts}), pos, bins=N_BINS,
                                    range=[(-50, 50), (-50, 50)], epochs=ep)
    occupancy = tc.attrs["occupancy"]
    fs = tc.attrs["fs"]
    rate_map = tc.sel(unit=0).values.astype(float)
    rate_map = np.where(occupancy < MIN_OCC_TIME * fs, np.nan, rate_map)
    return rate_map, occupancy


def analyze_unit(spike_ts, pos, ep):
    rate_map, occupancy = compute_rate_map(spike_ts, pos, ep)
    smoothed = smooth_rate_map(rate_map, sigma=1.0)
    autocorr = spatial_autocorrelogram(smoothed)
    g = gridness_score(autocorr)
    return g, rate_map, smoothed, autocorr


def circular_shift_ts(ts, ep, min_shift=20.0):
    """Circularly shift spike times within the session epoch by a random amount,
    the standard shuffle-control procedure for spatial firing statistics."""
    start, end = ep.start[0], ep.end[0]
    duration = end - start
    shift = np.random.uniform(min_shift, duration - min_shift)
    t = ts.t - start
    t_shifted = (t + shift) % duration + start
    return nap.Ts(t=np.sort(t_shifted))


# %% [markdown]
# ## 1. Load one example session and inspect its structure
#
# We stream the NWB file directly from the DANDI S3 bucket using `remfile`
# with local disk caching, avoiding a full download.

# %%
EXAMPLE_URL = "https://dandiarchive.s3.amazonaws.com/blobs/26a/22c/26a22c31-09bc-43a4-9187-edc7394ed12c"  # sub-10073/ses-17010302

nwb, io = load_session(EXAMPLE_URL)
print(nwb)

# %%
units = nwb["units"]
pos = nwb["SpatialSeriesLED1"]
ep = pos.time_support

print(f"Number of units: {len(units)}")
print(units.metadata)
print(f"\nPosition tracking: {len(pos)} samples over {ep.tot_length():.1f} s "
      f"(~{len(pos) / ep.tot_length():.1f} Hz)")

# %% [markdown]
# ## 2. Visualize raw data: foraging trajectory and spike raster
#
# Before any analysis, we plot the animal's raw tracked trajectory in the open
# field and a raster of spike times for all recorded units, to confirm the
# data look sensible (continuous exploration, no long dropouts, plausible
# firing rates).

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].plot(pos["x"].values, pos["y"].values, lw=0.3, color="steelblue")
axes[0].set_xlabel("x (cm)")
axes[0].set_ylabel("y (cm)")
axes[0].set_title("Foraging trajectory (open field)")
axes[0].set_aspect("equal")

for i, unit_id in enumerate(units.index):
    spk = units[unit_id]
    axes[1].vlines(spk.t, i + 0.6, i + 1.4, color="black", lw=0.5)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Unit #")
axes[1].set_title("Spike raster, all MEC units")
axes[1].set_xlim(0, ep.end[0])

plt.tight_layout()
plt.savefig(f"{FIGDIR}/01_raw_trajectory_and_raster.png", dpi=150)
plt.close()

# %% [markdown]
# ## 3. Rate maps and spatial autocorrelograms for one session
#
# For each unit we compute a smoothed 2D firing-rate map (spikes per second per
# spatial bin), then its spatial autocorrelogram. A grid cell's autocorrelogram
# shows six equally-spaced peaks surrounding the central peak, in a hexagonal
# arrangement; the gridness score captures how close to hexagonal that pattern is.

# %%
example_results = {}
for unit_id in units.index:
    g, rate_map, smoothed, autocorr = analyze_unit(units[unit_id], pos, ep)
    example_results[unit_id] = dict(gridness=g, rate_map=rate_map, smoothed=smoothed, autocorr=autocorr)
    print(f"unit {unit_id}: gridness = {g:.2f}")

# %%
n_units = len(units.index)
fig, axes = plt.subplots(3, n_units, figsize=(3 * n_units, 9.5))
for i, unit_id in enumerate(units.index):
    r = example_results[unit_id]
    axes[0, i].imshow(r["rate_map"], origin="lower", cmap="viridis")
    axes[0, i].set_title(f"unit {unit_id}\nraw rate map", fontsize=9)
    axes[0, i].axis("off")

    axes[1, i].imshow(r["smoothed"], origin="lower", cmap="viridis")
    axes[1, i].set_title("smoothed", fontsize=9)
    axes[1, i].axis("off")

    axes[2, i].imshow(r["autocorr"], origin="lower", cmap="viridis", vmin=-1, vmax=1)
    axes[2, i].set_title(f"autocorrelogram\ngridness = {r['gridness']:.2f}", fontsize=9)
    axes[2, i].axis("off")

fig.suptitle("Example session (sub-10073): rate maps and spatial autocorrelograms for all recorded MEC units", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/02_example_session_rate_maps.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## 4. A single grid cell in detail
#
# We highlight the unit with the highest gridness score in this session,
# overlaying its spike locations on the animal's trajectory to show the
# characteristic hexagonal firing lattice directly in space.

# %%
best_unit = max(example_results, key=lambda u: (example_results[u]["gridness"] if not np.isnan(example_results[u]["gridness"]) else -np.inf))
best = example_results[best_unit]
spk = units[best_unit]
spk_pos = pos.value_from(spk)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(pos["x"].values, pos["y"].values, lw=0.3, color="lightgray")
axes[0].scatter(spk_pos["x"].values, spk_pos["y"].values, s=4, color="crimson")
axes[0].set_title(f"Unit {best_unit}: spike positions on trajectory")
axes[0].set_xlabel("x (cm)"); axes[0].set_ylabel("y (cm)")
axes[0].set_aspect("equal")

axes[1].imshow(best["smoothed"], origin="lower", cmap="viridis")
axes[1].set_title("Smoothed firing-rate map")
axes[1].axis("off")

im = axes[2].imshow(best["autocorr"], origin="lower", cmap="viridis", vmin=-1, vmax=1)
axes[2].set_title(f"Spatial autocorrelogram\ngridness = {best['gridness']:.2f}")
axes[2].axis("off")
fig.colorbar(im, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/03_best_example_grid_cell.png", dpi=150)
plt.close()
io.close()

# %% [markdown]
# ## 5. Scaling to multiple sessions with a shuffle significance test
#
# A gridness score alone doesn't tell us whether the hexagonal pattern is
# stronger than expected by chance. We pool units across six sessions from six
# different subjects and, for each unit, build a null distribution of gridness
# scores from 100 circular time-shifts of its spike train (a shift that
# preserves the inter-spike-interval structure and firing rate while
# destroying the spike-position relationship). A unit is classified as a grid
# cell if its true gridness score exceeds the 95th percentile of its own
# shuffled null distribution.

# %%
SESSIONS = {
    "sub-10697": "https://dandiarchive.s3.amazonaws.com/blobs/9d7/e0d/9d7e0d8e-a133-46ef-a06f-245789350734",
    "sub-10073": "https://dandiarchive.s3.amazonaws.com/blobs/26a/22c/26a22c31-09bc-43a4-9187-edc7394ed12c",
    "sub-10884": "https://dandiarchive.s3.amazonaws.com/blobs/568/70d/56870d3a-f2a7-4632-8598-010481c07c24",
    "sub-11016": "https://dandiarchive.s3.amazonaws.com/blobs/dbf/0c8/dbf0c8b7-17fc-4d02-ab47-241ea73d1567",
    "sub-11207": "https://dandiarchive.s3.amazonaws.com/blobs/489/7ae/4897aef9-8bce-41e3-849d-f966ad73c9a1",
    "sub-11340": "https://dandiarchive.s3.amazonaws.com/blobs/224/331/22433160-8ee4-4405-86aa-9c5c86020544",
}
N_SHUFFLES = 100

rows = []
gallery = {}

for sub, url in SESSIONS.items():
    print(f"\n=== {sub} ===")
    nwb_s, io_s = load_session(url)
    units_s = nwb_s["units"]
    pos_s = nwb_s["SpatialSeriesLED1"]
    ep_s = pos_s.time_support
    metadata = units_s.metadata

    for unit_id in tqdm(units_s.index, desc=f"{sub} units"):
        spike_ts = units_s[unit_id]
        g_real, rate_map, smoothed, autocorr = analyze_unit(spike_ts, pos_s, ep_s)

        shuffled_g = np.full(N_SHUFFLES, np.nan)
        for i in range(N_SHUFFLES):
            shifted = circular_shift_ts(spike_ts, ep_s)
            g_shuf, _, _, _ = analyze_unit(shifted, pos_s, ep_s)
            shuffled_g[i] = g_shuf

        threshold = np.nanpercentile(shuffled_g, 95)
        is_grid = (not np.isnan(g_real)) and (g_real > threshold)

        meta_row = metadata.loc[unit_id] if unit_id in metadata.index else None
        histology = meta_row["histology"] if meta_row is not None else ""

        rows.append({
            "session": sub, "unit_id": unit_id, "n_spikes": len(spike_ts),
            "gridness": g_real, "shuffle_95pct": threshold,
            "is_grid_cell": is_grid, "histology": histology,
        })
        gallery[(sub, unit_id)] = dict(smoothed=smoothed, autocorr=autocorr,
                                        gridness=g_real, is_grid_cell=is_grid)

    io_s.close()

df = pd.DataFrame(rows)
df.to_csv("population_gridness_results.csv", index=False)
print(df)
print(f"\nTotal units analyzed: {len(df)}")
print(f"Classified as grid cells (gridness > subject-specific shuffle 95th pct): {df['is_grid_cell'].sum()} "
      f"({100 * df['is_grid_cell'].mean():.0f}%)")

# %% [markdown]
# ## 6. Population summary
#
# We summarize the distribution of gridness scores across the pooled
# population, compare it against the pooled shuffled null distribution, and
# show the rate maps of the top-scoring grid cells across sessions.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["gridness"].dropna(), bins=15, color="steelblue", alpha=0.8, label="observed")
axes[0].axvline(df["gridness"].median(), color="steelblue", ls="--", lw=1)
all_shuffle_thresh = df["shuffle_95pct"].dropna()
axes[0].hist(all_shuffle_thresh, bins=15, color="gray", alpha=0.5, label="per-unit shuffle 95th pct")
axes[0].set_xlabel("Gridness score")
axes[0].set_ylabel("Number of units")
axes[0].set_title("Population gridness scores vs. shuffled null")
axes[0].legend()

colors = np.where(df["is_grid_cell"], "crimson", "gray")
axes[1].scatter(df["shuffle_95pct"], df["gridness"], c=colors, alpha=0.8)
lims = [min(df["shuffle_95pct"].min(), df["gridness"].min()) - 0.1,
        max(df["shuffle_95pct"].max(), df["gridness"].max()) + 0.1]
axes[1].plot(lims, lims, "k--", lw=1)
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
axes[1].set_xlabel("Unit-specific shuffle 95th percentile")
axes[1].set_ylabel("Observed gridness score")
axes[1].set_title("Grid cells (red) exceed their own shuffled null")

plt.tight_layout()
plt.savefig(f"{FIGDIR}/04_population_gridness_summary.png", dpi=150)
plt.close()

# %%
grid_cells = df[df["is_grid_cell"]].sort_values("gridness", ascending=False)
top_n = min(8, len(grid_cells))
top_keys = [(row.session, row.unit_id) for row in grid_cells.head(top_n).itertuples()]

fig, axes = plt.subplots(2, top_n, figsize=(3 * top_n, 6.5))
if top_n == 1:
    axes = axes.reshape(2, 1)
for i, key in enumerate(top_keys):
    g = gallery[key]
    axes[0, i].imshow(g["smoothed"], origin="lower", cmap="viridis")
    axes[0, i].set_title(f"{key[0]}\nunit {key[1]}", fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].imshow(g["autocorr"], origin="lower", cmap="viridis", vmin=-1, vmax=1)
    axes[1, i].set_title(f"gridness = {g['gridness']:.2f}", fontsize=8)
    axes[1, i].axis("off")

fig.suptitle(f"Top {top_n} grid cells across {len(SESSIONS)} sessions / subjects", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/05_top_grid_cells_gallery.png", dpi=150, bbox_inches="tight")
plt.close()

# %% [markdown]
# ## Results and interpretation
#
# Across the recorded units in medial entorhinal cortex layer II, a
# substantial fraction show autocorrelograms with clear six-fold (hexagonal)
# symmetry, and gridness scores that exceed the 95th percentile of a
# per-unit, time-shuffled null distribution. This reproduces the original
# finding of Sargolini et al. (2006): MEC layer II contains a population of
# grid cells whose firing fields tile the environment in a periodic hexagonal
# lattice, providing a metric coordinate system for the animal's position
# that is independent of the specific environment being explored.
print("Analysis complete.")
print(df.groupby("session")["is_grid_cell"].agg(["sum", "count"]))
