# %% [markdown]
# # Head Direction Cells in the Mouse Anterodorsal Thalamus
#
# **Dataset:** [DANDI 000056](https://dandiarchive.org/dandiset/000056) — Peyrache et al. (2015),
# *"Internally organized mechanisms of the head direction sense"* (Nature Neuroscience 18:569–575).
# Extracellular recordings from the anterodorsal nucleus of the thalamus (ADn) and postsubiculum
# of freely moving mice, together with dual-LED head tracking and scored behavioral states
# (Awake / REM / Non-REM).
#
# **Phenomenon:** Head direction (HD) cells fire selectively when the animal's head points in a
# particular direction in the horizontal plane, independent of position. They are the cellular
# basis of the internal compass.
#
# **What this notebook does:**
# 1. Streams one session from DANDI and reconstructs head direction from the two head-mounted LEDs.
# 2. Computes occupancy-corrected HD tuning curves and identifies HD cells with an
#    occupancy-matched randomization test (mean vector length).
# 3. Fits a Poisson GLM with a cyclic B-spline basis (NeMoS) as an encoding model of HD tuning.
# 4. Decodes HD from the HD-cell population with Bayesian decoding.
# 5. Reproduces the paper's signature result: pairwise HD-cell correlations are preserved
#    from wake to REM and Non-REM sleep.
# 6. Repeats the pipeline across five sessions (five mice) and pools the statistics.

# %% [markdown]
# ## Setup

# %%
import os

import h5py
import matplotlib
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import nemos as nmo

os.makedirs("figures", exist_ok=True)
os.makedirs("cache", exist_ok=True)

DANDI_API_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"

# Five sessions, one per mouse, chosen for modest file size. The three ~30 GB
# Mouse12 files and Mouse32-140820 (which lacks the states table) are excluded.
SESSIONS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}
PROTOTYPE = "Mouse17-130128"

RNG = np.random.default_rng(42)


# %% [markdown]
# ## Helper functions
#
# Streaming access via `remfile` with a disk cache, head direction from the LED
# difference vector, occupancy-corrected tuning curves, and an occupancy-matched
# randomization test for directional selectivity.

# %%
def load_session(asset_id, cache_dir="cache"):
    """Stream an NWB file from DANDI with remfile disk caching."""
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(DANDI_API_URL.format(asset_id=asset_id), disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    return nap.NWBFile(io.read()), io


def get_sorted_units(nwb):
    """Units TsGroup with each unit's spike times sorted.

    One unit in this dataset has unsorted spike times (pynapple warns at load);
    searchsorted-based operations assume sorted times, so sort defensively.
    Metadata is dropped: the only column ('rate') conflicts with the reserved
    TsGroup attribute and is not needed here.
    """
    units = nwb["units"]
    data = {k: nap.Ts(np.sort(units[k].t)) for k in units.keys()}
    return nap.TsGroup(data)


def compute_head_direction(nwb):
    """Head direction from the dual-LED tracking.

    HD = atan2(red_y - blue_y, red_x - blue_x) mapped to [0, 2*pi).
    Tracking failures are sentinel -1 values; those samples become NaN.
    """
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    rv = np.asarray(red.values, dtype=float)
    bv = np.asarray(blue.values, dtype=float)
    t = np.asarray(red.t, dtype=float)
    if np.any(np.diff(t) <= 0):  # enforce strictly increasing times
        order = np.argsort(t, kind="stable")
        t, rv, bv = t[order], rv[order], bv[order]
    valid = np.all(rv > 0, axis=1) & np.all(bv > 0, axis=1)
    d = rv - bv
    hd = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    hd[~valid] = np.nan
    return nap.Tsd(t=t, d=hd, time_support=red.time_support)


def get_state_epochs(nwb, label):
    """IntervalSet of epochs with the given state label (Awake / REM / Non-REM)."""
    states = nwb["states"]
    return states[states["label"] == label]


def tuning_curves_hd(units, hd, wake, bins=60):
    """Occupancy-corrected HD tuning curves.

    Returns (rates xarray in Hz, bin centers, occupancy in seconds).
    Bins with zero occupancy are NaN.
    """
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    counts = nap.compute_tuning_curves(
        units, hd, bins=[edges], epochs=wake, return_counts=True, return_pandas=False
    )
    occupancy = counts.attrs["occupancy"] / counts.attrs["fs"]  # seconds per bin
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = counts / occupancy
    rates = rates.where(occupancy > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return rates, centers, occupancy


def mean_vector_length(angles):
    """Mean resultant length of a set of angles (NaNs removed)."""
    angles = np.asarray(angles)
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan
    return np.abs(np.exp(1j * angles).mean())


def spike_angles(unit_ts, hd):
    """HD angle at each spike time of one unit (NaNs dropped)."""
    ang = np.asarray(unit_ts.value_from(hd))
    return ang[~np.isnan(ang)]


def random_time_null(n_spikes, valid_hd_samples, n_shuffles=1000, rng=None,
                     max_n=100_000, chunk=100):
    """Null distribution of MVL under random-time resampling.

    Redraws n_spikes angles uniformly from the valid wake HD samples and computes
    the MVL. This respects the (non-uniform) HD occupancy, unlike a circular
    time-shift shuffle, which is invalid here because wake is fragmented into
    short epochs of nearly constant HD. n_spikes is capped (the null only gets
    tighter with more spikes, so the cap is conservative) and sampling is
    chunked to bound memory.
    """
    rng = np.random.default_rng(rng)
    n = min(n_spikes, max_n)
    if n == 0 or len(valid_hd_samples) == 0:
        return np.full(n_shuffles, np.nan)
    out = np.empty(n_shuffles)
    for s in range(0, n_shuffles, chunk):
        e = min(s + chunk, n_shuffles)
        idx = rng.integers(0, len(valid_hd_samples), size=(e - s, n))
        out[s:e] = np.abs(np.exp(1j * valid_hd_samples[idx]).mean(axis=1))
    return out


def classify_hd_cells(units, hd, wake, n_shuffles=1000, mvl_floor=0.3, alpha=0.05, rng=None):
    """Classify units as HD cells: MVL above the random-time null AND MVL > floor.

    The effect-size floor matters: with hundreds of thousands of spikes, even
    trivial MVL deviations from the occupancy resultant pass p < 0.05.
    """
    rng = np.random.default_rng(rng)
    keys = list(units.keys())
    hd_wake = hd.restrict(wake)
    valid_hd = np.asarray(hd_wake.values)
    valid_hd = valid_hd[~np.isnan(valid_hd)]
    mvl = np.full(len(keys), np.nan)
    pval = np.full(len(keys), np.nan)
    pref = np.full(len(keys), np.nan)
    null_med = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        u_wake = units[k].restrict(wake)
        ang = spike_angles(u_wake, hd)
        if len(ang) < 50:
            continue
        mvl[i] = mean_vector_length(ang)
        pref[i] = np.angle(np.exp(1j * ang).mean()) % (2 * np.pi)
        null = random_time_null(len(u_wake), valid_hd, n_shuffles=n_shuffles, rng=rng)
        null_med[i] = np.nanmedian(null)
        pval[i] = (np.sum(null >= mvl[i]) + 1) / (np.sum(~np.isnan(null)) + 1)
    is_hd = (pval < alpha) & (mvl > mvl_floor)
    return {"keys": keys, "mvl": mvl, "p_value": pval, "pref_angle": pref,
            "null_median": null_med, "is_hd": is_hd}


def circ_interp(query_t, t, angles):
    """Circular-safe interpolation of an angle time series via sin/cos."""
    valid = ~np.isnan(angles)
    cos_i = np.interp(query_t, t[valid], np.cos(angles[valid]))
    sin_i = np.interp(query_t, t[valid], np.sin(angles[valid]))
    return np.arctan2(sin_i, cos_i) % (2 * np.pi)


def pair_correlations(group, epochs, bin_size=0.1, min_bins=50):
    """Upper-triangle Pearson correlations of binned spike counts."""
    cnt = group.count(bin_size, ep=epochs)
    X = cnt.values
    if X.shape[0] < min_bins:
        return None
    C = np.corrcoef(X.T)
    return C[np.triu_indices(X.shape[1], k=1)]


def pick_wide_coverage_window(hd, wake, win=60.0, stride=30.0):
    """Find a wake window of length win with the broadest HD coverage."""
    best_t0, best_cov = None, -1
    for t0 in np.arange(wake.start[0], wake.end[-1] - win, stride):
        ep = nap.IntervalSet(t0, t0 + win)
        if ep.intersect(wake).tot_length() < win - 1:
            continue
        v = hd.restrict(ep).values
        v = v[~np.isnan(v)]
        if len(v) < 500:
            continue
        occ, _ = np.histogram(v, bins=36, range=(0, 2 * np.pi))
        cov = np.mean(occ > 0)
        if cov > best_cov:
            best_cov, best_t0 = cov, t0
    return best_t0, best_cov


# %% [markdown]
# ## Load the prototype session and reconstruct head direction
#
# The animal wears two LEDs (red front, blue back). Head direction is the angle
# of the red-minus-blue vector. Samples where tracking failed are marked with
# -1 sentinels and set to NaN.

# %%
nwb, io = load_session(SESSIONS[PROTOTYPE])
print(nwb)

units = get_sorted_units(nwb)
hd = compute_head_direction(nwb)
wake = get_state_epochs(nwb, "Awake")
print(f"\n{len(units)} units; wake: {wake.tot_length():.0f} s in {len(wake)} epochs")
print(f"HD samples: {len(hd)} at ~{1 / np.median(np.diff(hd.t)):.1f} Hz; "
      f"NaN fraction: {np.mean(np.isnan(hd.values)):.3f}")

# %% [markdown]
# ### Raw tracking data
#
# One minute of LED tracking, the derived head direction, and the HD occupancy
# over all wake epochs. The occupancy is clearly non-uniform, which any
# statistical test of directional tuning must account for.

# %%
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
red_t = np.asarray(red.t)
red_v = np.asarray(red.values, dtype=float)
blue_v = np.asarray(blue.values, dtype=float)

fig = plt.figure(figsize=(11, 9))
ax0 = fig.add_subplot(3, 1, 1)
t0, t1 = 100.0, 160.0
sl = (red_t >= t0) & (red_t <= t1)
ax0.plot(red_t[sl], red_v[sl, 0], "r-", lw=0.6, label="Red LED x")
ax0.plot(red_t[sl], blue_v[sl, 0], "b-", lw=0.6, label="Blue LED x")
ax0.set_ylabel("x position (a.u.)")
ax0.legend(loc="upper right", fontsize=8)
ax0.set_title("Dual-LED head tracking (60 s snippet)")

ax1 = fig.add_subplot(3, 1, 2)
sl2 = (hd.t >= t0) & (hd.t <= t1)
ax1.plot(hd.t[sl2], hd.values[sl2], "k.", ms=1.5)
ax1.set_ylabel("Head direction (rad)")
ax1.set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
ax1.set_title("Head direction from the LED difference vector")

axp = fig.add_subplot(3, 1, 3, projection="polar")
hd_wake = hd.restrict(wake)
occ, edges = np.histogram(hd_wake.values[~np.isnan(hd_wake.values)], bins=60, range=(0, 2 * np.pi))
centers60 = 0.5 * (edges[:-1] + edges[1:])
axp.bar(centers60, occ / 60.0, width=2 * np.pi / 60, color="gray")
axp.set_title("Wake HD occupancy (s per 6° bin)", pad=32)
fig.tight_layout()
fig.savefig("figures/01_raw_tracking.png", dpi=150)
plt.close(fig)
print("saved figures/01_raw_tracking.png")

# %% [markdown]
# ## Tuning curves and HD-cell classification
#
# For each unit we compute the occupancy-corrected tuning curve (spike counts
# per HD bin divided by time spent in that bin) and the mean vector length
# (MVL) of the spike angles. Significance uses a random-time resampling null:
# redraw the same number of spike times uniformly from valid wake HD samples.
# A circular time-shift shuffle is *not* valid here — wake is fragmented into
# short epochs of nearly constant HD, so shifted spike trains keep artificially
# high MVL. A unit is an HD cell if p < 0.05 and MVL > 0.3 (effect-size floor).

# %%
rates, centers, occupancy = tuning_curves_hd(units, hd, wake, bins=60)
result = classify_hd_cells(units, hd, wake, n_shuffles=1000, rng=RNG)
keys = result["keys"]
is_hd = result["is_hd"]
hd_keys = [k for k, h in zip(keys, is_hd) if h]
print(f"HD cells: {is_hd.sum()}/{len(keys)} -> {hd_keys}")

order = np.argsort(result["pref_angle"][is_hd])
hd_keys_sorted = [hd_keys[i] for i in order]

# %% [markdown]
# ### Tuning curves of the identified HD cells

# %%
n_hd = len(hd_keys_sorted)
ncol = 4
nrow = int(np.ceil(n_hd / ncol))
fig = plt.figure(figsize=(3.0 * ncol, 3.0 * nrow))
for i, k in enumerate(hd_keys_sorted):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection="polar")
    r = rates.sel(unit=k).values
    ax.plot(np.append(centers, centers[0]), np.append(r, r[0]), "k-", lw=1.5)
    ax.fill(np.append(centers, centers[0]), np.append(r, r[0]), "r", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"unit {k}  MVL={result['mvl'][keys.index(k)]:.2f}", fontsize=9)
fig.suptitle(f"{PROTOTYPE}: HD-cell tuning curves (n={n_hd}, sorted by preferred direction)",
             y=1.02)
fig.savefig("figures/02_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/02_tuning_curves.png")

# %% [markdown]
# ### Selectivity versus the occupancy-matched null
#
# The null MVL median (~0.18) equals the resultant length of the wake HD
# occupancy — this is the value a completely untuned cell would be expected to
# reach just because the animal spent more time facing some directions.

# %%
fig, ax = plt.subplots(figsize=(6, 6))
mvl = result["mvl"]
null_med = result["null_median"]
ok = ~np.isnan(mvl)
ax.scatter(null_med[ok & ~is_hd], mvl[ok & ~is_hd], s=25, c="gray", alpha=0.7, label="not HD")
ax.scatter(null_med[ok & is_hd], mvl[ok & is_hd], s=35, c="crimson", label="HD cell")
lim = [0, max(1.0, np.nanmax(mvl) * 1.05)]
ax.plot(lim, lim, "k--", lw=0.8)
ax.axhline(0.3, color="crimson", ls=":", lw=1, label="MVL floor = 0.3")
ax.set_xlabel("Null MVL (median, random-time resampling)")
ax.set_ylabel("Observed MVL")
ax.set_title(f"{PROTOTYPE}: directional selectivity vs occupancy-matched null")
ax.legend(fontsize=9)
fig.savefig("figures/03_mvl_vs_null.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/03_mvl_vs_null.png")

# %% [markdown]
# ### Population raster in a wide-coverage window
#
# Spike rasters of the HD cells (sorted by preferred direction) under the HD
# trace, in the 60 s wake window with the broadest directional coverage.

# %%
best_t0, best_cov = pick_wide_coverage_window(hd, wake, win=60.0, stride=30.0)
print(f"display window: t0={best_t0:.0f} s, HD coverage={best_cov:.2f}")
ep = nap.IntervalSet(best_t0, best_t0 + 60.0)

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                         gridspec_kw={"height_ratios": [1, 2]})
hd_ep = hd.restrict(ep)
axes[0].plot(hd_ep.t, hd_ep.values, "k.", ms=1.5)
axes[0].set_ylabel("HD (rad)")
axes[0].set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
axes[0].set_title(f"{PROTOTYPE}: HD cells sorted by preferred direction (60 s of wake)")
for i, k in enumerate(hd_keys_sorted):
    spk = units[k].restrict(ep)
    axes[1].plot(spk.t, np.full(len(spk), i), "|", color="crimson", ms=6)
axes[1].set_ylabel("HD cell (sorted)")
axes[1].set_xlabel("Time (s)")
axes[1].set_yticks(range(n_hd), [str(k) for k in hd_keys_sorted], fontsize=7)
fig.tight_layout()
fig.savefig("figures/04_raster.png", dpi=150)
plt.close(fig)
print("saved figures/04_raster.png")

# %% [markdown]
# ## Encoding model: Poisson GLM with a cyclic B-spline basis (NeMoS)
#
# As a regression-based description of the tuning, we fit a Poisson GLM per HD
# cell with a cyclic B-spline basis over the HD angle (8 basis functions),
# using NeMoS. Spike counts in 50 ms bins are the target. The GLM-implied
# tuning curve (prediction on a grid of angles) is compared with the empirical
# occupancy-corrected tuning curve, and goodness of fit is quantified per cell
# with McFadden's pseudo-R² against a constant-rate null model.

# %%
bin_size = 0.05  # s
hd_group = units[hd_keys]
count = hd_group.count(bin_size, ep=wake)

hd_binned = circ_interp(count.t, hd.t, np.asarray(hd.values, dtype=float))
hd_tsd = nap.Tsd(t=count.t, d=hd_binned, time_support=count.time_support)

basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8, order=3, bounds=(0, 2 * np.pi))
X = basis.compute_features(hd_tsd)
print("design matrix:", X.shape, "counts:", count.shape)

model = nmo.glm.PopulationGLM(
    observation_model="Poisson",
    regularizer="Ridge",
    regularizer_strength=1e-5,
    solver_name="LBFGS",
    solver_kwargs={"tol": 1e-12, "maxiter": 5000},
)
model.fit(X, count)

# GLM-implied tuning curves on an angle grid
grid = np.linspace(0, 2 * np.pi, 60, endpoint=False)
rate_grid = model.predict(basis.compute_features(grid)) / bin_size  # Hz

# per-cell McFadden pseudo-R^2 (constant-rate null)
pred_counts = np.clip(np.asarray(model.predict(X)), 1e-12, None)  # counts/bin
n_obs = np.asarray(count.values, dtype=float)
ll_model = (n_obs * np.log(pred_counts) - pred_counts).sum(axis=0)
null_counts = np.full_like(pred_counts, n_obs.mean(axis=0))
ll_null = (n_obs * np.log(null_counts) - null_counts).sum(axis=0)
pseudo_r2 = 1 - ll_model / ll_null
for k, r2 in zip(hd_keys, pseudo_r2):
    print(f"unit {k}: pseudo-R2 = {r2:.3f}")

# %%
n_hd = len(hd_keys)
fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
for i, k in enumerate(hd_keys):
    ax = axes.flat[i]
    emp = rates.sel(unit=k).values
    ax.plot(np.rad2deg(centers), emp, "k-", lw=1.5, label="empirical")
    ax.plot(np.rad2deg(grid), rate_grid[:, i], "r-", lw=1.5, label="GLM")
    ax.set_title(f"unit {k}  (pseudo-$R^2$={pseudo_r2[i]:.2f})", fontsize=10)
    ax.set_xlabel("HD (deg)")
    if i % 4 == 0:
        ax.set_ylabel("Rate (Hz)")
    if i == 0:
        ax.legend(fontsize=8)
# rate snippet for the HD cell with the highest peak rate
peak_rates = [np.nanmax(rates.sel(unit=k).values) for k in hd_keys]
i_snip = int(np.argmax(peak_rates))
k_snip = hd_keys[i_snip]
ax = axes.flat[n_hd]
t0, t1 = best_t0, best_t0 + 30.0
sl = (count.t >= t0) & (count.t <= t1)
actual = count.values[sl, i_snip] / bin_size
kern = np.ones(20) / 20  # 1 s boxcar
ax.plot(count.t[sl], np.convolve(actual, kern, mode="same"), "k-", lw=1, label="actual (1 s smooth)")
ax.plot(count.t[sl], (pred_counts[sl, i_snip] / bin_size), "r-", lw=1, label="GLM prediction")
ax.set_title(f"unit {k_snip}: rate snippet", fontsize=10)
ax.set_xlabel("Time (s)")
ax.legend(fontsize=8)
for j in range(n_hd + 1, len(axes.flat)):
    axes.flat[j].axis("off")
fig.suptitle(f"{PROTOTYPE}: Poisson GLM with cyclic B-spline HD basis")
fig.tight_layout(h_pad=2.0)
fig.savefig("figures/05_glm_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/05_glm_tuning.png")

# %% [markdown]
# ## Bayesian decoding of head direction
#
# If the population genuinely encodes HD, the animal's momentary heading should
# be recoverable from the spike counts alone. We decode with `nap.decode_bayes`
# (100 ms bins, uniform prior) using the empirical tuning curves. Bins with
# fewer than 2 spikes carry no information and are excluded from the error
# statistics. Chance-level median absolute error for a uniform circular guess
# is 90°.

# %%
bin_size_dec = 0.1
count_dec = hd_group.count(bin_size_dec, ep=wake)
tc_filled = rates.sel(unit=hd_keys).fillna(0.0)
decoded, proba = nap.decode_bayes(tc_filled, count_dec, wake, bin_size_dec)

hd_actual = circ_interp(decoded.t, hd.t, np.asarray(hd.values, dtype=float))
n_spikes_bin = count_dec.values.sum(axis=1)
informative = n_spikes_bin >= 2
err = np.angle(np.exp(1j * (decoded.values - hd_actual)))
med_err = np.rad2deg(np.nanmedian(np.abs(err[informative])))
print(f"informative bins: {informative.sum()}/{len(err)} ({100 * informative.mean():.0f}%)")
print(f"median |error| = {med_err:.1f} deg; "
      f"within 30 deg: {100 * np.mean(np.abs(err[informative]) < np.deg2rad(30)):.0f}%")

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
t0, t1 = best_t0, best_t0 + 60.0
sl = (decoded.t >= t0) & (decoded.t <= t1) & informative
axes[0].plot(decoded.t[sl], hd_actual[sl], "k.", ms=3, label="actual HD")
axes[0].plot(decoded.t[sl], decoded.values[sl] % (2 * np.pi), "r.", ms=3, alpha=0.55,
             label="decoded HD")
axes[0].set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
axes[0].set_ylabel("HD (rad)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title(f"{PROTOTYPE}: Bayesian HD decoding from {len(hd_keys)} HD cells "
                  f"(100 ms bins, >=2 spikes)")
axes[1].hist(np.rad2deg(err[informative]), bins=72, range=(-180, 180), color="steelblue")
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_xlabel("Decoding error (deg)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Error distribution (median |err| = {med_err:.1f} deg; chance ~ 90 deg)")
fig.tight_layout()
fig.savefig("figures/06_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/06_decoding.png")

# %% [markdown]
# ## Correlation structure is preserved from wake to sleep
#
# The signature result of Peyrache et al. (2015): the pairwise correlation
# structure of HD cells during sleep (both REM and Non-REM) matches that of
# wake, as expected if the HD network is internally organized (a ring-like
# attractor) rather than driven by sensory input. We bin HD-cell spikes at
# 100 ms per state, correlate unit pairs within each state, then correlate the
# wake and sleep pairwise-correlation vectors across pairs. Significance is
# assessed by permuting pair identities between wake and sleep (10,000
# permutations).

# %%
C = {lab: pair_correlations(hd_group, get_state_epochs(nwb, lab) if lab != "Awake" else wake)
     for lab in ["Awake", "REM", "Non-REM"]}
n_pairs = len(C["Awake"])
print(f"{n_pairs} pairs from {len(hd_keys)} HD cells")

r_rem = np.corrcoef(C["Awake"], C["REM"])[0, 1]
r_nrem = np.corrcoef(C["Awake"], C["Non-REM"])[0, 1]

n_perm = 10_000
null_rem = np.empty(n_perm)
null_nrem = np.empty(n_perm)
for p in range(n_perm):
    perm = RNG.permutation(n_pairs)
    null_rem[p] = np.corrcoef(C["Awake"], C["REM"][perm])[0, 1]
    null_nrem[p] = np.corrcoef(C["Awake"], C["Non-REM"][perm])[0, 1]
p_rem = (np.sum(null_rem >= r_rem) + 1) / (n_perm + 1)
p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (n_perm + 1)
print(f"wake vs REM:  r = {r_rem:.3f} (p = {p_rem:.4f})")
print(f"wake vs NREM: r = {r_nrem:.3f} (p = {p_nrem:.4f})")

# correlation matrices
Cmat = {}
for lab in ["Awake", "REM", "Non-REM"]:
    ep = wake if lab == "Awake" else get_state_epochs(nwb, lab)
    cnt = hd_group.count(0.1, ep=ep)
    Cmat[lab] = np.corrcoef(cnt.values.T)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
vmin, vmax = -0.6, 0.6
for ax, lab in zip(axes, ["Awake", "REM", "Non-REM"]):
    im = ax.imshow(Cmat[lab], cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(f"{lab}")
    ax.set_xticks(range(n_hd), hd_keys, fontsize=7)
    ax.set_yticks(range(n_hd), hd_keys, fontsize=7)
    if ax is axes[0]:
        ax.set_ylabel("unit")
fig.suptitle(f"{PROTOTYPE}: HD-cell pairwise correlations by state")
fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r")
fig.savefig("figures/07_state_correlations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/07_state_correlations.png")

fig, ax = plt.subplots(figsize=(5.8, 5.8))
ax.scatter(C["Awake"], C["REM"], s=30, c="darkorange",
           label=f"REM (r={r_rem:.2f}, p={p_rem:.4f})")
ax.scatter(C["Awake"], C["Non-REM"], s=30, c="steelblue", alpha=0.7,
           label=f"NREM (r={r_nrem:.2f}, p={p_nrem:.4f})")
lims = [-0.6, 0.8]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("Wake pairwise correlation")
ax.set_ylabel("Sleep pairwise correlation")
ax.set_title(f"{PROTOTYPE}: HD-cell correlations preserved in sleep")
ax.legend(fontsize=9)
fig.savefig("figures/08_wake_sleep_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/08_wake_sleep_scatter.png")

io.close()

# %% [markdown]
# ## Scaling up: five sessions, five mice
#
# The same pipeline (HD reconstruction, tuning curves, occupancy-matched
# classification, state correlations) is run on one session from each of five
# mice. Per-session results are cached to `cache/` so re-runs are fast.

# %%
per_session = {}
for name, asset_id in SESSIONS.items():
    cache_file = f"cache/results_{name}.npz"
    if os.path.exists(cache_file):
        print(f"{name}: loading cached results")
        d = np.load(cache_file, allow_pickle=True)
        per_session[name] = {k: d[k] for k in d.files}
        continue
    print(f"\n=== {name} ===")
    nwb_s, io_s = load_session(asset_id)
    units_s = get_sorted_units(nwb_s)
    hd_s = compute_head_direction(nwb_s)
    wake_s = get_state_epochs(nwb_s, "Awake")
    print(f"{len(units_s)} units, wake {wake_s.tot_length():.0f} s")

    res_s = classify_hd_cells(units_s, hd_s, wake_s, n_shuffles=1000, rng=RNG)
    hd_keys_s = [k for k, h in zip(res_s["keys"], res_s["is_hd"]) if h]
    print(f"HD cells: {res_s['is_hd'].sum()}/{len(res_s['keys'])} -> {hd_keys_s}")

    out = {"session": name, "keys": np.array(res_s["keys"]), "mvl": res_s["mvl"],
           "p_value": res_s["p_value"], "pref_angle": res_s["pref_angle"],
           "null_median": res_s["null_median"], "is_hd": res_s["is_hd"],
           "occ_mvl": mean_vector_length(np.asarray(hd_s.restrict(wake_s).values))}
    if len(hd_keys_s) >= 4:
        group_s = units_s[hd_keys_s]
        for lab in ["Awake", "REM", "Non-REM"]:
            ep_s = wake_s if lab == "Awake" else get_state_epochs(nwb_s, lab)
            pc = pair_correlations(group_s, ep_s)
            if pc is not None:
                out[f"pairs_{lab}"] = pc
    np.savez(cache_file, **out)
    per_session[name] = out
    io_s.close()
    print(f"cached -> {cache_file}")

# %% [markdown]
# ### Population summary across sessions

# %%
n_units_all = [len(per_session[s]["keys"]) for s in SESSIONS]
n_hd_all = [int(per_session[s]["is_hd"].sum()) for s in SESSIONS]
mvl_all = np.concatenate([per_session[s]["mvl"] for s in SESSIONS])
is_hd_all = np.concatenate([per_session[s]["is_hd"] for s in SESSIONS])
pref_all = np.concatenate([per_session[s]["pref_angle"][per_session[s]["is_hd"]]
                           for s in SESSIONS])
pref_all = pref_all[~np.isnan(pref_all)]

fig = plt.figure(figsize=(14, 4.2))
ax = fig.add_subplot(1, 3, 1)
x = np.arange(len(SESSIONS))
ax.bar(x, n_units_all, color="lightgray", edgecolor="k", label="all units")
ax.bar(x, n_hd_all, color="crimson", edgecolor="k", label="HD cells")
ax.set_xticks(x, [s.replace("Mouse", "M") for s in SESSIONS], fontsize=8)
ax.set_ylabel("Unit count")
ax.set_title("HD cells per session")
ax.legend(fontsize=8)

ax = fig.add_subplot(1, 3, 2)
bins = np.linspace(0, 1, 41)
ax.hist(mvl_all[~is_hd_all & ~np.isnan(mvl_all)], bins=bins, color="gray", label="not HD")
ax.hist(mvl_all[is_hd_all], bins=bins, color="crimson", label="HD cells")
ax.axvline(0.3, color="k", ls=":", lw=1)
ax.set_xlabel("Mean vector length")
ax.set_ylabel("Unit count")
ax.set_title(f"MVL distribution (n={len(mvl_all)} units)")
ax.legend(fontsize=8)

axp = fig.add_subplot(1, 3, 3, projection="polar")
occ_p, edges_p = np.histogram(pref_all, bins=24, range=(0, 2 * np.pi))
axp.bar(0.5 * (edges_p[:-1] + edges_p[1:]), occ_p, width=2 * np.pi / 24, color="steelblue")
axp.set_title(f"Preferred directions (n={len(pref_all)} HD cells)", pad=22)
fig.tight_layout()
fig.savefig("figures/09_multisession_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/09_multisession_summary.png")

# Rayleigh-style Monte Carlo test of uniformity of preferred directions
R_obs = np.abs(np.exp(1j * pref_all).mean())
n_pref = len(pref_all)
null_R = np.abs(np.exp(1j * RNG.uniform(0, 2 * np.pi, size=(10_000, n_pref))).mean(axis=1))
p_unif = (np.sum(null_R >= R_obs) + 1) / (10_001)
print(f"preferred-direction resultant R = {R_obs:.3f}, uniformity p = {p_unif:.3f}")

# %% [markdown]
# ### Pooled wake-sleep correlation preservation
#
# Pairwise correlations pooled across sessions (sessions with >= 4 HD cells).
# The permutation null shuffles pair identities within each session.

# %%
wake_pairs, rem_pairs, nrem_pairs, sess_id = [], [], [], []
for i, s in enumerate(SESSIONS):
    d = per_session[s]
    if "pairs_Awake" in d and "pairs_REM" in d and "pairs_Non-REM" in d:
        w, r_, n_ = d["pairs_Awake"], d["pairs_REM"], d["pairs_Non-REM"]
        ok = ~(np.isnan(w) | np.isnan(r_) | np.isnan(n_))  # silent units -> NaN
        wake_pairs.append(w[ok])
        rem_pairs.append(r_[ok])
        nrem_pairs.append(n_[ok])
        sess_id.append(np.full(ok.sum(), i))
wake_pairs = np.concatenate(wake_pairs)
rem_pairs = np.concatenate(rem_pairs)
nrem_pairs = np.concatenate(nrem_pairs)
sess_id = np.concatenate(sess_id)

r_rem_pool = np.corrcoef(wake_pairs, rem_pairs)[0, 1]
r_nrem_pool = np.corrcoef(wake_pairs, nrem_pairs)[0, 1]

n_perm = 10_000
null_rem = np.empty(n_perm)
null_nrem = np.empty(n_perm)
for p in range(n_perm):
    perm = np.empty(len(wake_pairs), dtype=int)
    for i in np.unique(sess_id):
        idx = np.where(sess_id == i)[0]
        perm[idx] = idx[RNG.permutation(len(idx))]
    null_rem[p] = np.corrcoef(wake_pairs, rem_pairs[perm])[0, 1]
    null_nrem[p] = np.corrcoef(wake_pairs, nrem_pairs[perm])[0, 1]
p_rem_pool = (np.sum(null_rem >= r_rem_pool) + 1) / (n_perm + 1)
p_nrem_pool = (np.sum(null_nrem >= r_nrem_pool) + 1) / (n_perm + 1)
print(f"pooled across {len(np.unique(sess_id))} sessions, {len(wake_pairs)} pairs:")
print(f"wake vs REM:  r = {r_rem_pool:.3f} (p = {p_rem_pool:.5f})")
print(f"wake vs NREM: r = {r_nrem_pool:.3f} (p = {p_nrem_pool:.5f})")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(wake_pairs, rem_pairs, s=22, c="darkorange", alpha=0.8,
           label=f"REM (r={r_rem_pool:.2f}, p={p_rem_pool:.5f})")
ax.scatter(wake_pairs, nrem_pairs, s=22, c="steelblue", alpha=0.6,
           label=f"NREM (r={r_nrem_pool:.2f}, p={p_nrem_pool:.5f})")
lims = [-0.6, 0.9]
ax.plot(lims, lims, "k--", lw=0.8)
ax.set_xlabel("Wake pairwise correlation")
ax.set_ylabel("Sleep pairwise correlation")
ax.set_title("HD-cell correlations preserved in sleep (pooled sessions)")
ax.legend(fontsize=9)
fig.savefig("figures/10_pooled_wake_sleep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/10_pooled_wake_sleep.png")

# %% [markdown]
# ## Summary
#
# Across five sessions from five mice (DANDI 000056), roughly a third of the
# recorded units in the ADn/postsubiculum region are head direction cells:
# their firing is strongly modulated by the animal's heading (MVL well above an
# occupancy-matched null), with preferred directions tiling the circle. A
# Poisson GLM with a cyclic B-spline basis captures the tuning curves
# quantitatively, and the population supports accurate Bayesian decoding of
# heading (median error far below the 90° chance level). Finally, the pairwise
# correlation structure among HD cells is preserved from wakefulness into both
# REM and Non-REM sleep, reproducing the central finding of Peyrache et al.
# (2015) and supporting an internally organized (attractor-like) HD network.
