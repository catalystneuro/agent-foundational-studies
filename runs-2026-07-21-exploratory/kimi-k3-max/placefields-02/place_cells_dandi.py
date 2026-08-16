# %% [markdown]
# # Hippocampal Place Cells on a Linear Track: A DANDI Archive Demonstration
#
# This notebook demonstrates hippocampal place cells using a public dataset from the
# DANDI Archive. We use dandiset **000044** ("Diversity in neural firing dynamics supports
# both rigid and learned hippocampal sequences", Buzsaki lab), specifically the session
# `sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb`: tetrode recordings from rat CA1
# (137 sorted units) while the animal runs back and forth on a 1.6 m linear track for
# water reward.
#
# Place cells, discovered by O'Keefe and Dostrovsky (1971), are hippocampal pyramidal
# neurons that fire when the animal occupies a specific location, the cell's "place field".
# On a linear track they have an additional property: most fields are strongly
# **directional**, appearing only during runs in one direction.
#
# The analysis proceeds in four steps:
#
# 1. Load the session by streaming from DANDI with remfile (no full download) and inspect
#    the behavioral data.
# 2. Extract run bouts from the linearized position and compute occupancy-normalized
#    firing-rate maps for every unit.
# 3. Quantify spatial tuning with Skaggs spatial information and assess significance with
#    circular time-shift shuffles, then classify place cells.
# 4. Fit Poisson GLM encoding models (nemos) to show that a smooth function of position
#    alone reproduces the binned rate maps.
#
# All data access is read-only streaming; results are cached under `/tmp` and figures are
# written to `figures/`.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Analysis parameters
NBINS = 50            # position bins over the 1.6 m track
TRACK_LEN = 1.6       # m
SMOOTH_BINS = 1.5     # Gaussian smoothing of spike/occupancy counts, in bins
NSHUF = 500           # circular time shifts for the SI null distribution
SEED = 7
MIN_BOUT_S = 1.0      # minimum run-bout duration
MERGE_GAP_S = 0.3     # merge bouts across shorter tracking dropouts
MIN_SPAN_M = 0.3      # minimum track distance covered per bout
MIN_SPEED = 0.15      # m/s, minimum median speed within a bout

rng = np.random.default_rng(SEED)

# %% [markdown]
# ## Streaming the NWB file from DANDI
#
# The file is 8.7 GB, but we only need the spike times, the position series, and the epoch
# table, which together are a few MB. remfile fetches only the required byte ranges and
# caches them on disk, so reruns are fast. The URL below is the public S3 blob for the
# asset `sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb` in
# dandiset 000044 (it can be regenerated through the DANDI API assets endpoint).

# %%
S3_URL = ("https://dandiarchive.s3.amazonaws.com/blobs/763/2d8/"
          "7632d81b-2819-473d-8946-34dc939e6028")
disk_cache = remfile.DiskCache("/tmp/remfile_cache_placecells_demo")
h5py_file = h5py.File(remfile.File(S3_URL, disk_cache=disk_cache), "r")
nwb = nap.NWBFile(NWBHDF5IO(file=h5py_file).read())
print(nwb)

units = nwb["units"]
cell_type = units.get_info("cell_type")
print(f"\n{len(units)} units:", dict(cell_type.value_counts()))

# %% [markdown]
# The session has PRE / Maze / POST epochs. We work only with the maze epoch. Two position
# streams are stored: the 2D camera position and a linearized coordinate (0 to 1.6 m) that
# is defined only while the animal is on the track arm. Both are sampled at 39.06 Hz.

# %%
pos2d = nwb["1.6mLinearMazeSpatialSeries"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t = lin.t
xy = np.asarray(pos2d.values)
x = np.asarray(lin.values).ravel()
dt = np.median(np.diff(t))
print(f"position: {len(t)} samples at {1/dt:.2f} Hz, "
      f"maze epoch {t[0]:.1f}-{t[-1]:.1f} s")
print(f"valid samples: 2D {100*np.mean(~np.isnan(xy[:,0])):.1f}%, "
      f"linearized {100*np.mean(~np.isnan(x)):.1f}%")

# %% [markdown]
# ## Figure 1: Session overview
#
# The rat shuttles between the two ends of the track, visible as a sawtooth in the
# linearized position. The linearized coordinate is NaN except during on-track runs
# (about 13% of the maze epoch); the animal spends the rest of the time at the reward
# platforms or off the track arm.

# %%
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3,
                      left=0.07, right=0.97, top=0.93, bottom=0.07)

ax = fig.add_subplot(gs[0, 0])
ax.plot(xy[:, 0], xy[:, 1], lw=0.3, color="0.4", alpha=0.6)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("A  2D trajectory on the 1.6 m linear maze", loc="left", fontsize=11)
ax.set_aspect("equal")

ax = fig.add_subplot(gs[0, 1])
m = (t >= t[0]) & (t <= t[0] + 400)
ax.plot(t[m] - t[0], x[m], lw=0.5, color="k")
ax.set_xlabel("time in maze epoch (s)"); ax.set_ylabel("linearized position (m)")
ax.set_title("B  Linearized position (first 400 s)", loc="left", fontsize=11)
ax.set_ylim(-0.05, 1.7)

ax = fig.add_subplot(gs[1, 0])
valid = ~np.isnan(x)
speed = np.abs(np.gradient(x, t))
ax.hist(speed[valid & ~np.isnan(speed)], bins=np.linspace(0, 1.5, 100), color="0.3")
ax.axvline(MIN_SPEED, color="r", ls="--", lw=1, label=f"{MIN_SPEED} m/s run threshold")
ax.set_xlabel("|d(lin)/dt| (m/s)"); ax.set_ylabel("samples")
ax.set_title("C  Speed distribution (valid samples)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)

ax = fig.add_subplot(gs[1, 1])
ax.bar(["2D position", "linearized"],
       [100*np.mean(~np.isnan(xy[:, 0])), 100*valid.mean()], color=["0.5", "0.2"])
ax.set_ylabel("% of maze-epoch samples valid"); ax.set_ylim(0, 100)
ax.set_title("D  Position data coverage", loc="left", fontsize=11)

ax = fig.add_subplot(gs[2, :])
t0, t1 = 18500.0, 18530.0
for i, k in enumerate(sorted(units.keys())):
    st = units[k].t
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st - t0, np.full_like(st, i), "|", color="k", ms=2)
ax.set_xlabel(f"time (s) from {t0:.0f} s"); ax.set_ylabel("unit index")
ax.set_title("E  Spike raster, 30 s of the maze epoch (137 units)", loc="left", fontsize=11)
ax.set_ylim(-1, len(units)); ax.set_xlim(0, 30)

fig.suptitle("DANDI 000044, sub-Achilles ses-Achilles-10252013, session overview",
             fontsize=13)
fig.savefig("figures/fig1_session_overview.png", dpi=150)
plt.close(fig)
print("saved figures/fig1_session_overview.png")

# %% [markdown]
# ## Run bouts
#
# Place fields are estimated from periods of actual running. We take the contiguous valid
# stretches of the linearized series, merge across tracking dropouts shorter than 300 ms,
# and keep bouts that last at least 1 s, cover at least 0.3 m of track, and have a median
# speed above 0.15 m/s. Each bout is also labeled by travel direction, which matters
# because linear-track place fields are typically direction selective.

# %%
def compute_bouts(t, x):
    valid = ~np.isnan(x)
    d = np.diff(valid.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if valid[0]:
        starts = [0] + starts
    if valid[-1]:
        ends = ends + [len(valid)]
    merged = []
    for s, e in zip(starts, ends):
        if merged and t[s] - t[merged[-1][1] - 1] < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    bouts = []
    for s, e in merged:
        m = ~np.isnan(x[s:e])  # merged-in gap samples are still NaN
        xb, tb = x[s:e][m], t[s:e][m]
        if t[e - 1] - t[s] < MIN_BOUT_S or len(xb) < 5:
            continue
        if xb.max() - xb.min() < MIN_SPAN_M:
            continue
        if np.median(np.abs(np.gradient(xb, tb))) < MIN_SPEED:
            continue
        bouts.append((s, e, 1 if xb[-1] > xb[0] else -1))
    return bouts

bouts = compute_bouts(t, x)
n_pos = sum(1 for b in bouts if b[2] == 1)
total_bout_s = sum(t[e - 1] - t[s] for s, e, _ in bouts)
print(f"{len(bouts)} run bouts ({n_pos} positive-direction, "
      f"{len(bouts) - n_pos} negative-direction), {total_bout_s:.0f} s total")

# %% [markdown]
# For the shuffle control below we need a time axis that contains only running. We
# concatenate the bouts into a single "run time" axis tau; every position sample inside a
# bout gets a tau value, and spikes are mapped onto tau by their bout membership. A
# circular shift of spike times on this axis destroys the spike-position relationship
# while preserving each spike's local temporal structure and the occupancy distribution.

# %%
tau = np.full(len(t), np.nan)
offsets = []
acc = 0.0
for s, e, _ in bouts:
    offsets.append(acc)
    tau[s:e] = acc + (t[s:e] - t[s])
    acc += tau[e - 1] - tau[s] + dt
offsets = np.array(offsets)
T_total = acc
ok = ~np.isnan(tau) & ~np.isnan(x)
tau_ok = tau[ok]
x_ok = x[ok]
order = np.argsort(tau_ok)
tau_ok, x_ok = tau_ok[order], x_ok[order]

b_starts = np.array([t[s] for s, e, _ in bouts])
b_ends = np.array([t[e - 1] for s, e, _ in bouts])
b_dirs = np.array([b[2] for b in bouts])

def spikes_to_run_time(st):
    """Spike times -> concatenated run-bout time; returns only in-bout spikes."""
    idx = np.clip(np.searchsorted(b_starts, st, side="right") - 1, 0, len(bouts) - 1)
    inb = (st >= b_starts[idx]) & (st <= b_ends[idx])
    return offsets[idx[inb]] + (st[inb] - b_starts[idx[inb]]), idx[inb]

# %% [markdown]
# ## Rate maps, spatial information, and shuffle statistics
#
# For each unit we build the occupancy-normalized rate map: spike counts and occupancy
# per 3.2 cm bin, each smoothed with a 1.5-bin Gaussian, then divided. Spatial tuning is
# quantified with Skaggs spatial information (bits/spike), and significance is assessed
# against 500 circular time shifts of the spike train on the run-time axis. A unit is
# called a place cell when the shuffle p-value is below 0.05, its mean in-run firing rate
# exceeds 0.1 Hz, and its peak field rate exceeds 1 Hz.

# %%
edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
occ_s = np.histogram(x_ok, bins=edges)[0].astype(float) * dt
occ_s_sm = gaussian_filter1d(occ_s, SMOOTH_BINS)
occ_mask = occ_s > 0

# per-direction occupancy
occ_d = {}
for dval in (1, -1):
    xs = np.concatenate([x[s:e][~np.isnan(x[s:e])] for s, e, dd in bouts if dd == dval])
    oc = np.histogram(xs, bins=edges)[0].astype(float) * dt
    occ_d[dval] = (oc, gaussian_filter1d(oc, SMOOTH_BINS), oc > 0)

def rate_map(spike_x, occ_sm, mask):
    counts = np.histogram(spike_x, bins=edges)[0].astype(float)
    rate = np.full(NBINS, np.nan)
    rate[mask] = gaussian_filter1d(counts, SMOOTH_BINS)[mask] / occ_sm[mask]
    return rate

def skaggs_si(rate, occ):
    p = occ / occ.sum()
    m = (p * rate).sum()
    if m <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rate / m
        return np.nansum(p * ratio * np.log2(ratio))

keys = sorted(units.keys())
ct = cell_type.loc[keys].values
n_units = len(keys)
rates = np.full((n_units, NBINS), np.nan)
rates_pos = np.full((n_units, NBINS), np.nan)
rates_neg = np.full((n_units, NBINS), np.nan)
si_real = np.zeros(n_units)
mean_rate = np.zeros(n_units)
p_si = np.ones(n_units)

for i, k in enumerate(tqdm(keys, desc="units")):
    st = units[k].t
    tau_s, bout_idx = spikes_to_run_time(st)
    if len(tau_s) < 10:
        continue
    x_s = np.interp(tau_s, tau_ok, x_ok)
    rates[i] = rate_map(x_s, occ_s_sm, occ_mask)
    si_real[i] = skaggs_si(np.nan_to_num(rates[i]), np.where(occ_mask, occ_s, 0))
    mean_rate[i] = len(tau_s) / T_total
    for dval, out in [(1, rates_pos), (-1, rates_neg)]:
        oc, oc_sm, om = occ_d[dval]
        out[i] = rate_map(x_s[b_dirs[bout_idx] == dval], oc_sm, om)
    si_null = np.empty(NSHUF)
    for j, sh in enumerate(rng.uniform(0, T_total, NSHUF)):
        x_sh = np.interp((tau_s + sh) % T_total, tau_ok, x_ok)
        si_null[j] = skaggs_si(np.nan_to_num(rate_map(x_sh, occ_s_sm, occ_mask)),
                               np.where(occ_mask, occ_s, 0))
    p_si[i] = (np.sum(si_null >= si_real[i]) + 1) / (NSHUF + 1)

peak_rate = np.nanmax(np.where(np.isnan(rates), -np.inf, rates), axis=1)
peak_rate[~np.isfinite(peak_rate)] = np.nan
is_place = (p_si < 0.05) & (mean_rate > 0.1) & (peak_rate > 1.0)
exc = ct == "excitatory"
print(f"\nplace cells: {np.sum(is_place & exc)}/{np.sum(exc)} excitatory units")
print(f"median SI: exc place {np.median(si_real[exc & is_place]):.2f}, "
      f"exc non-place {np.median(si_real[exc & ~is_place]):.2f}, "
      f"inhibitory {np.median(si_real[~exc]):.3f} bits/spike")

# direction-map correlation for place cells with both maps
dir_corr = np.full(n_units, np.nan)
for i in range(n_units):
    a, b = rates_pos[i], rates_neg[i]
    m2 = ~np.isnan(a) & ~np.isnan(b)
    if m2.sum() > 5 and np.std(a[m2]) > 0 and np.std(b[m2]) > 0:
        dir_corr[i] = np.corrcoef(a[m2], b[m2])[0, 1]
print(f"median direction-map correlation (place cells): "
      f"{np.nanmedian(dir_corr[is_place]):.2f}")

# %% [markdown]
# ## Figure 2: Example place cells
#
# Six excitatory place cells with high spatial information and fields spread across the
# track. Left: spike positions on the trajectory over the whole session, showing that the
# cell returns to the same location on lap after lap. Right: the rate map computed
# separately for the two travel directions. Most of these cells fire almost exclusively in
# one direction, a hallmark of CA1 place fields on linear tracks.

# %%
def spike_positions(st):
    """In-bout spike positions and bout directions for one unit."""
    tau_s, bout_idx = spikes_to_run_time(st)
    x_s = np.interp(tau_s, tau_ok, x_ok)
    return tau_s, x_s, b_dirs[bout_idx]

exc_place = np.where(is_place & exc)[0]
chosen = []
for c in exc_place[np.argsort(-si_real[exc_place])]:
    if all(abs(np.nanargmax(rates[c]) - np.nanargmax(rates[k2])) > 2 for k2 in chosen):
        chosen.append(c)
    if len(chosen) == 6:
        break
chosen = chosen[::-1]

fig, axes = plt.subplots(6, 2, figsize=(11, 13),
                         gridspec_kw=dict(hspace=0.55, wspace=0.25,
                                          left=0.07, right=0.97, top=0.95, bottom=0.05))
for row, ci in enumerate(chosen):
    k = keys[ci]
    tau_s, x_s, sdir = spike_positions(units[k].t)
    ax = axes[row, 0]
    okp = ~np.isnan(tau)
    ax.plot(tau[okp], x[okp], lw=0.4, color="0.75", zorder=1)
    ax.scatter(tau_s, x_s, s=4, c="crimson", zorder=2, rasterized=True)
    ax.set_ylabel("track position (m)", fontsize=9)
    ax.set_ylim(-0.05, 1.65); ax.set_xlim(0, T_total)
    ax.set_title(f"unit {k}, spikes on trajectory (SI={si_real[ci]:.2f} bits/spike)",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=8)
    ax = axes[row, 1]
    ax.plot(centers, rates_pos[ci], color="tab:blue", lw=1.5, label="pos. direction")
    ax.plot(centers, rates_neg[ci], color="tab:orange", lw=1.5, label="neg. direction")
    ax.set_ylabel("firing rate (Hz)", fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_title(f"unit {k}, tuning by direction (peak {np.nanmax(rates[ci]):.1f} Hz)",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=8)
    if row == 0:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    if row == 5:
        for c2 in range(2):
            axes[row, c2].set_xlabel(
                "time in run bouts (s)" if c2 == 0 else "track position (m)", fontsize=9)

fig.suptitle("Example CA1 place cells, DANDI 000044, Achilles 10252013", fontsize=13)
fig.savefig("figures/fig2_example_place_cells.png", dpi=150)
plt.close(fig)
print("saved figures/fig2_example_place_cells.png")

# %% [markdown]
# ## Figure 3: Place fields tile the track
#
# Normalized rate maps for all excitatory place cells, each row sorted by the position of
# its peak. The diagonal band shows that the population covers the whole track, both in
# the pooled maps and within each travel direction. Rows that are dark in one direction
# panel and bright in the other belong to direction-selective cells.

# %%
def norm_sorted_maps(idx, R):
    M = R[idx].copy()
    M = M[~np.isnan(M).all(axis=1)]
    pk = np.nanmax(M, axis=1, keepdims=True)
    pk[pk == 0] = np.nan
    M = M / pk
    M = M[~np.isnan(M).all(axis=1)]
    return M[np.argsort(np.nanargmax(M, axis=1))]

fig, axes = plt.subplots(1, 3, figsize=(13, 6.5),
                         gridspec_kw=dict(wspace=0.35, left=0.06, right=0.97,
                                          top=0.9, bottom=0.12))
for ax, R, title in [
    (axes[0], rates, "A  All runs (pooled)"),
    (axes[1], rates_pos, "B  Positive-direction runs"),
    (axes[2], rates_neg, "C  Negative-direction runs"),
]:
    M = norm_sorted_maps(exc_place, R)
    im = ax.imshow(M, aspect="auto", cmap="viridis", extent=[0, 1.6, 0, M.shape[0]],
                   origin="lower", vmin=0, vmax=1)
    ax.set_xlabel("track position (m)")
    ax.set_title(f"{title}, {M.shape[0]} place cells", fontsize=10, loc="left")
    if ax is axes[0]:
        ax.set_ylabel("place cells (sorted by peak)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="norm. rate")
fig.suptitle("Place fields tile the linear track: excitatory place cells, "
             "normalized rate maps", fontsize=12)
fig.savefig("figures/fig3_population_tiling.png", dpi=150)
plt.close(fig)
print("saved figures/fig3_population_tiling.png")

# %% [markdown]
# ## Figure 4: Population statistics
#
# Four views of the result. (A) Place cells carry much more spatial information per spike
# than inhibitory interneurons. (B) The shuffle separates most excitatory units from
# chance; note that the interneurons, despite tiny SI values, are also "significant"
# because their large spike counts make the null distribution very narrow. Weak but
# consistent spatial modulation of interneurons (through theta and speed modulation) is a
# real effect, and it is why an SI threshold matters in practice. (C) The rate criteria
# mostly remove low-firing units. (D) The correlation between the two direction-specific
# maps is low for most place cells (median about 0.25), quantifying the direction
# selectivity visible in Figure 2.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5),
                         gridspec_kw=dict(hspace=0.4, wspace=0.3,
                                          left=0.08, right=0.96, top=0.92, bottom=0.09))
ax = axes[0, 0]
bins = np.linspace(0, 3, 60)
ax.hist(si_real[exc & is_place], bins=bins, alpha=0.7, color="tab:blue",
        label=f"exc place cells (n={np.sum(exc & is_place)})")
ax.hist(si_real[exc & ~is_place], bins=bins, alpha=0.7, color="0.5",
        label=f"exc non-place (n={np.sum(exc & ~is_place)})")
ax.hist(si_real[~exc], bins=bins, alpha=0.7, color="tab:red",
        label=f"inhibitory (n={np.sum(~exc)})")
ax.set_xlabel("Skaggs spatial information (bits/spike)"); ax.set_ylabel("units")
ax.set_title("A  Spatial information by group", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
for m2, c, lab in [(exc & is_place, "tab:blue", "exc place"),
                   (exc & ~is_place, "0.5", "exc non-place"),
                   (~exc, "tab:red", "inhibitory")]:
    ax.scatter(si_real[m2], p_si[m2], s=12, alpha=0.7, c=c, label=lab)
ax.axhline(0.05, color="k", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel("Skaggs SI (bits/spike)"); ax.set_ylabel("shuffle p-value (log)")
ax.set_title("B  Shuffle significance vs SI", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
for m2, c, lab in [(exc & is_place, "tab:blue", "exc place"),
                   (exc & ~is_place, "0.5", "exc non-place"),
                   (~exc, "tab:red", "inhibitory")]:
    ax.scatter(mean_rate[m2], peak_rate[m2], s=12, alpha=0.7, c=c, label=lab)
ax.axhline(1.0, color="k", ls="--", lw=1); ax.axvline(0.1, color="k", ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("mean in-run firing rate (Hz)"); ax.set_ylabel("peak place-field rate (Hz)")
ax.set_title("C  Rate criteria", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 1]
dc = dir_corr[exc & is_place]
dc = dc[~np.isnan(dc)]
ax.hist(dc, bins=np.linspace(-1, 1, 40), color="tab:blue", alpha=0.8)
ax.axvline(np.median(dc), color="k", ls="--", lw=1, label=f"median = {np.median(dc):.2f}")
ax.axvline(0, color="0.5", lw=0.5)
ax.set_xlabel("correlation between direction-specific rate maps")
ax.set_ylabel("place cells")
ax.set_title("D  Directionality of place fields", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Place-cell identification and statistics: "
             f"{NSHUF} circular time-shifts per unit", fontsize=12)
fig.savefig("figures/fig4_place_cell_stats.png", dpi=150)
plt.close(fig)
print("saved figures/fig4_place_cell_stats.png")

# %% [markdown]
# ## Figure 5: A Poisson GLM recovers the same place fields (nemos)
#
# As an independent check, and to demonstrate a model-based encoding analysis, we fit each
# place cell with a Poisson GLM: position is expanded in 12 raised-cosine basis functions
# and passed through a log-link Poisson model with a small ridge penalty (nemos, JAX in
# float64 for stable convergence). Spikes are counted in the 25.6 ms position-sample bins
# of the run bouts. We then compare the GLM-predicted rate as a function of position with
# the binned tuning curve, and quantify fit quality with a pseudo-R2 against an
# intercept-only model, computed directly from Poisson log-likelihoods.

# %%
import jax
jax.config.update("jax_enable_x64", True)
import nemos as nmo

in_bout = ~np.isnan(tau) & ~np.isnan(x)
t_b = t[in_bout]
x_b = x[in_bout]
n_bins = len(t_b)
basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=12, bounds=(0.0, 1.6))
X = np.asarray(basis.compute_features(x_b))
print(f"design matrix: {X.shape} (run-bout time bins x basis functions)")

key_pos = {k: j for j, k in enumerate(keys)}
counts = np.zeros((n_bins, n_units))
for j, k in enumerate(keys):
    st = units[k].t
    st = st[(st >= t_b[0]) & (st <= t_b[-1])]
    ib = np.searchsorted(t_b, st, side="right") - 1
    okb = (ib >= 0) & (np.abs(t_b[np.clip(ib, 0, n_bins - 1)] - st) < dt)
    np.add.at(counts[:, j], ib[okb], 1)

def poisson_ll(count, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum(count * np.log(mu) - mu)  # log(count!) cancels in the ratio

pc_idx = np.where(is_place & exc)[0]
grid = np.linspace(0, 1.6, NBINS)
X_grid = np.asarray(basis.compute_features(grid))
glm_maps = np.full((len(pc_idx), NBINS), np.nan)
pseudo_r2 = np.zeros(len(pc_idx))

for ii, ci in enumerate(tqdm(pc_idx, desc="GLM fits")):
    y = counts[:, key_pos[keys[ci]]]
    model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=1e-4,
                        solver_name="LBFGS", solver_kwargs=dict(maxiter=5000))
    model.fit(X, y)
    mu = model.predict(X)
    glm_maps[ii] = model.predict(X_grid) / dt  # counts/bin -> Hz
    pseudo_r2[ii] = 1 - poisson_ll(y, np.clip(mu, 1e-12, None)) / \
                        poisson_ll(y, np.full_like(y, max(y.mean(), 1e-12)))

map_corr = np.array([
    np.corrcoef(np.nan_to_num(glm_maps[ii]), np.nan_to_num(rates[ci]))[0, 1]
    for ii, ci in enumerate(pc_idx)
])
print(f"median pseudo-R2: {np.median(pseudo_r2):.3f}")
print(f"median GLM-vs-binned map correlation: {np.median(map_corr):.3f}")

# %%
ex_order = np.argsort(-pseudo_r2)[:4]
fig, axes = plt.subplots(2, 4, figsize=(13, 6),
                         gridspec_kw=dict(hspace=0.5, wspace=0.3,
                                          left=0.06, right=0.97, top=0.88, bottom=0.12))
for col, ei in enumerate(ex_order):
    ci = pc_idx[ei]
    ax = axes[0, col]
    ax.plot(centers, rates[ci], color="k", lw=1.5, label="binned tuning curve")
    ax.plot(grid, glm_maps[ei], color="tab:red", lw=1.5, ls="--", label="Poisson GLM")
    ax.set_title(f"unit {keys[ci]}  (pseudo-$R^2$={pseudo_r2[ei]:.2f})",
                 fontsize=10, loc="left")
    ax.set_xlim(0, 1.6); ax.tick_params(labelsize=8)
    ax.set_xlabel("track position (m)", fontsize=9)
    if col == 0:
        ax.set_ylabel("firing rate (Hz)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
ax.hist(pseudo_r2, bins=np.linspace(0, 1, 30), color="tab:red", alpha=0.8)
ax.axvline(np.median(pseudo_r2), color="k", ls="--", lw=1,
           label=f"median={np.median(pseudo_r2):.2f}")
ax.set_xlabel("pseudo-$R^2$ (vs intercept-only)", fontsize=9)
ax.set_ylabel("place cells", fontsize=9)
ax.set_title("GLM goodness of fit", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8); ax.tick_params(labelsize=8)

ax = axes[1, 1]
ax.hist(map_corr, bins=np.linspace(-1, 1, 30), color="0.4")
ax.axvline(np.median(map_corr), color="k", ls="--", lw=1,
           label=f"median={np.median(map_corr):.2f}")
ax.set_xlabel("corr(GLM map, binned map)", fontsize=9)
ax.set_ylabel("place cells", fontsize=9)
ax.set_title("GLM vs binned agreement", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8); ax.tick_params(labelsize=8)

ax = axes[1, 2]
ax.scatter(si_real[pc_idx], pseudo_r2, s=14, alpha=0.7, color="tab:blue")
ax.set_xlabel("Skaggs SI (bits/spike)", fontsize=9)
ax.set_ylabel("pseudo-$R^2$", fontsize=9)
ax.set_title("SI vs GLM fit", fontsize=10, loc="left")
ax.tick_params(labelsize=8)

ax = axes[1, 3]
bg = np.asarray(basis.compute_features(np.linspace(0, 1.6, 200)))
ax.plot(np.linspace(0, 1.6, 200), bg, lw=1)
ax.set_xlabel("track position (m)", fontsize=9)
ax.set_title("12 raised-cosine basis functions", fontsize=10, loc="left")
ax.tick_params(labelsize=8)

fig.suptitle("Poisson GLM encoding of position (nemos): excitatory place cells",
             fontsize=12)
fig.savefig("figures/fig5_nemos_glm.png", dpi=150)
plt.close(fig)
print("saved figures/fig5_nemos_glm.png")

# %% [markdown]
# ## Summary
#
# Streaming a single 8.7 GB NWB file from DANDI 000044, we recover the classic properties
# of hippocampal place cells on a linear track:
#
# * **Most CA1 pyramidal cells are place cells.** About 70% of excitatory units (roughly
#   86 of 120) fire at a restricted location significantly more than circularly
#   time-shifted controls, with peak field rates of a few Hz to tens of Hz.
# * **Place fields tile the track.** Sorted population rate maps form a diagonal band from
#   one end of the 1.6 m track to the other, in each travel direction separately.
# * **Fields are direction selective.** The median correlation between a cell's two
#   direction-specific rate maps is only about 0.25; many cells fire in one direction only.
# * **Position alone explains the maps.** A Poisson GLM on a smooth position basis
#   reproduces the binned tuning curves (median map correlation about 0.99), with
#   pseudo-R2 values typical of single-bin spike prediction and tightly correlated with
#   Skaggs spatial information.
#
# Inhibitory interneurons carry an order of magnitude less spatial information per spike
# (median about 0.01 vs 0.5 bits/spike), consistent with their weak, largely
# non-spatial modulation during running.
