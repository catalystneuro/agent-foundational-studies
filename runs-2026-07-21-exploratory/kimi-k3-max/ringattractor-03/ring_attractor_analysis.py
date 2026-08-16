# %% [markdown]
# # The head-direction system is a continuous ring attractor, internally maintained during sleep
#
# **Dataset**: DANDI Archive dandiset
# [000056](https://dandiarchive.org/dandiset/000056) — Peyrache et al. (2015),
# *"Internally organized mechanisms of the head direction sense"* (Nature Neuroscience 18:569–575).
# Extracellular recordings from the anterodorsal thalamic nucleus (ADn) of freely moving mice,
# with dual-LED head tracking and scored sleep states (Awake / Non-REM / REM).
#
# **Question**: Head-direction (HD) cells each fire when the animal faces a preferred direction.
# Theory says these cells are organized as a *continuous ring attractor*: recurrent connectivity
# holds the population activity in a single "bump" on a ring, and the bump's position is the
# internally represented heading. The decisive test of *internal* organization is sleep: with no
# vestibular or visual input, an externally driven system should fall apart, whereas a ring
# attractor should keep its pairwise correlation structure and keep moving a coherent bump
# around the ring.
#
# **What this notebook shows**, on real data streamed from DANDI:
#
# 1. HD cells in ADn tile the circle of directions during wakefulness.
# 2. The population activity forms a single bump that tracks the actual head direction
#    (Bayesian decoding of HD at ~14° median error).
# 3. During REM sleep the same bump exists and *drifts continuously* around the ring
#    (median angular speed ~120°/s, far slower than a temporal shuffle), i.e. the network
#    autonomously maintains and moves a one-dimensional activity bump.
# 4. The pairwise correlation structure of HD cells — the signature of the ring topology —
#    is preserved during REM and Non-REM sleep (correlation-of-correlations ≈ 0.9 vs wake,
#    permutation p ≤ 0.001), replicated across 5 sessions from 5 mice.
#
# Analysis uses [Pynapple](https://pynapple.org) for data handling and computation; NWB files
# are streamed with `remfile` + disk caching (no bulk downloads).

# %% [markdown]
# ## Setup and shared functions

# %%
import h5py
import remfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

rng = np.random.default_rng(0)

# Published version 0.250624.0430 of dandiset 000056; one session per mouse.
ASSETS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}
PROTO = "Mouse28-140310"  # prototype session (richest HD-cell yield)
CACHE = "/tmp/remfile_cache_hd"
BIN = 0.1  # 100 ms count bins for decoding / correlations


def load_session(asset_id, cache_dir=CACHE):
    """Stream an NWB file from DANDI with a local disk cache."""
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    h5py_file = h5py.File(remfile.File(url, disk_cache=disk_cache), "r")
    return nap.NWBFile(NWBHDF5IO(file=h5py_file).read())


def compute_hd(nwb):
    """Head direction from the two LEDs; tracking failures are -1 sentinels -> NaN."""
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    rv, bv = np.asarray(red.values), np.asarray(blue.values)
    valid = (rv[:, 0] > 0) & (rv[:, 1] > 0) & (bv[:, 0] > 0) & (bv[:, 1] > 0)
    ang = np.full(rv.shape[0], np.nan)
    ang[valid] = np.arctan2(rv[valid, 1] - bv[valid, 1],
                            rv[valid, 0] - bv[valid, 0]) % (2 * np.pi)
    return nap.Tsd(t=red.t, d=ang, time_support=red.time_support)


def clean_units(units):
    """TsGroup of units with sorted spike times (one file has an unsorted unit)."""
    d = {k: nap.Ts(np.sort(units[k].t)) for k in units.keys() if len(units[k].t) > 0}
    return nap.TsGroup(d)


def identify_hd_cells(units, hd, wake, n_shuffle=1000, mvl_floor=0.3, alpha=0.05):
    """HD cells: mean vector length of spike angles, tested against a random-time
    resampling null (redraw spike times uniformly from valid wake HD samples), with an
    effect-size floor on the MVL."""
    hd_w = hd.restrict(wake)
    ok = ~np.isnan(hd_w.d)
    a_valid = hd_w.d[ok]
    mvl, pval = {}, {}
    for k in units.keys():
        ang = units[k].restrict(wake).value_from(hd)
        ang = ang[~np.isnan(ang)]
        n = len(ang)
        if n < 100:
            continue
        if n > 100_000:  # memory guard for very high-rate units
            ang = ang[rng.choice(n, 100_000, replace=False)]
            n = 100_000
        obs = np.abs(np.mean(np.exp(1j * ang)))
        null = np.empty(n_shuffle)
        for c in range(0, n_shuffle, 100):
            idx = rng.integers(0, len(a_valid), size=(100, n))
            null[c:c + 100] = np.abs(np.mean(np.exp(1j * a_valid[idx]), axis=1))
        mvl[k] = obs
        pval[k] = (np.sum(null >= obs) + 1) / (n_shuffle + 1)
    hd_cells = [k for k in mvl if pval[k] < alpha and mvl[k] > mvl_floor]
    return mvl, pval, hd_cells


def state_corr(units, ep, bin_size=BIN):
    """Pearson correlation matrix of binned spike counts within a state."""
    X = units.count(bin_size, ep).values
    with np.errstate(invalid="ignore"):
        return np.corrcoef(X.T)  # NaN row/col for units silent in this state


def corr_of_corr(C1, C2, iu):
    v1, v2 = C1[iu], C2[iu]
    ok = np.isfinite(v1) & np.isfinite(v2)
    return np.corrcoef(v1[ok], v2[ok])[0, 1]


def norm_rows(X, sigma):
    Xs = gaussian_filter1d(X.astype(float), sigma, axis=1)
    mx = np.nanmax(Xs, axis=1, keepdims=True)
    return np.nan_to_num(Xs / np.where(mx == 0, 1, mx))


def to_angle_grid(X_sorted, prefs_sorted, n_grid=60):
    """Interpolate a (units x time) matrix, with units sorted by preferred angle,
    onto a uniform angular grid so the y-axis is truly angular (circular interp)."""
    grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    ang_ext = np.concatenate([prefs_sorted - 2 * np.pi, prefs_sorted,
                              prefs_sorted + 2 * np.pi])
    out = np.empty((n_grid, X_sorted.shape[1]))
    for j in range(X_sorted.shape[1]):
        out[:, j] = np.interp(grid, ang_ext, np.concatenate([X_sorted[:, j]] * 3))
    return grid, out


def angular_speed(decoded, valid, bin_size=BIN):
    a = decoded.values[valid]
    t = decoded.t[valid]
    da = np.angle(np.exp(1j * np.diff(a)))
    dt = np.diff(t)
    ok = dt < 2 * bin_size  # consecutive bins only
    return np.degrees(np.abs(da[ok]) / dt[ok])


RAD_TICKS = ([0, np.pi, 2 * np.pi], ["0", "π", "2π"])

# %% [markdown]
# ## Load the prototype session and inspect the raw data
#
# Session Mouse28-140310: 45 well-isolated units, ~6 h of recording with 43 awake,
# 42 Non-REM and 15 REM epochs, and 39 Hz dual-LED tracking.

# %%
nwb = load_session(ASSETS[PROTO])
units_all = clean_units(nwb["units"])
states = nwb["states"]
wake = states[states["label"] == "Awake"]
rem = states[states["label"] == "REM"]
nrem = states[states["label"] == "Non-REM"]
hd = compute_hd(nwb)
print(nwb)
print(f"\n{len(units_all)} units | wake {float((wake.end-wake.start).sum()):.0f} s, "
      f"NREM {float((nrem.end-nrem.start).sum()):.0f} s, "
      f"REM {float((rem.end-rem.start).sum()):.0f} s")
print(f"HD valid fraction in wake: {np.mean(~np.isnan(hd.restrict(wake).d)):.3f}")

# %% [markdown]
# ### Figure 1 — session overview: head tracking, sleep states, and spike rasters

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1.2], hspace=0.5)

state_colors = {"Awake": "white", "Non-REM": "lightsteelblue", "REM": "mistyrose"}

# (a) full-session HD with state shading
ax = fig.add_subplot(gs[0])
for lab, c in state_colors.items():
    if lab == "Awake":
        continue
    for s, e in zip(states[states["label"] == lab].start, states[states["label"] == lab].end):
        ax.axvspan(s, e, color=c, zorder=0)
step = 20  # decimate for display
ax.plot(hd.t[::step], hd.d[::step], ".", ms=0.5, color="darkslategray", zorder=1)
ax.set_yticks(*RAD_TICKS)
ax.set_ylabel("HD (rad)")
ax.set_xlim(hd.t[0], hd.t[-1])
ax.set_title("Head direction across the full session (blue = Non-REM, red = REM)")
ax.legend(handles=[Patch(color="lightsteelblue", label="Non-REM"),
                   Patch(color="mistyrose", label="REM")], loc="upper right", fontsize=8)

# (b) HD over one awake minute + (c) raster of all units in the same window
t0, t1 = 3761, 3821
ax = fig.add_subplot(gs[1])
m = (hd.t >= t0) & (hd.t <= t1)
ax.plot(hd.t[m], hd.d[m], lw=1)
ax.set_yticks(*RAD_TICKS)
ax.set_ylabel("HD (rad)")
ax.set_title("One minute of wakefulness")

ax = fig.add_subplot(gs[2])
ep = nap.IntervalSet(t0, t1)
for i, k in enumerate(units_all.keys()):
    st = units_all[k].restrict(ep).t
    ax.plot(st, np.full_like(st, i), "|", ms=2, color="k")
ax.set_ylabel("unit #")
ax.set_xlabel("time (s)")
ax.set_title(f"Spike raster, all {len(units_all)} units")
fig.savefig("fig1_session_overview.png", dpi=150)
print("saved fig1_session_overview.png")

# %% [markdown]
# ## Head-direction tuning and HD-cell identification
#
# For each unit we compute the mean vector length (MVL) of the angles at which it spiked
# during wakefulness. Significance uses a random-time resampling null (spike counts matched,
# times redrawn from the actual wake HD samples, so the null inherits the true angular
# occupancy); we additionally require MVL > 0.3 as an effect-size floor.

# %%
mvl, pval, hd_cells = identify_hd_cells(units_all, hd, wake)
print(f"{len(hd_cells)}/{len(units_all)} units are HD cells: {hd_cells}")

units = units_all[hd_cells]
tc = nap.compute_tuning_curves(units, hd, bins=60, range=(0, 2 * np.pi),
                               epochs=wake, return_counts=True)
occ = tc.attrs["occupancy"] / tc.attrs["fs"]
tc_rate = tc / np.where(occ == 0, np.nan, occ)  # Hz
bin_centers = tc.coords[tc.dims[1]].values
prefs = np.array([bin_centers[np.nanargmax(tc_rate.values[i])]
                  for i in range(len(hd_cells))])
order = np.argsort(prefs)
prefs_sorted = prefs[order]

# %% [markdown]
# ### Figure 2 — HD cells tile the circle of directions

# %%
n = len(hd_cells)
ncol = 7
nrow = int(np.ceil(n / ncol)) + 1  # extra row for the MVL panel
fig = plt.figure(figsize=(15, 2.4 * nrow))
gs = fig.add_gridspec(nrow, ncol, hspace=0.7, wspace=0.5)
for i in range(n):
    axp = fig.add_subplot(gs[i // ncol, i % ncol], projection="polar")
    v = tc_rate.values[i]
    axp.plot(np.append(bin_centers, bin_centers[0]), np.append(v, v[0]), lw=1.2)
    axp.fill(np.append(bin_centers, bin_centers[0]), np.append(v, v[0]), alpha=0.3)
    axp.set_title(f"u{hd_cells[i]}  MVL={mvl[hd_cells[i]]:.2f}", fontsize=8)
    axp.set_xticks([])
    axp.set_yticks([])
# MVL distribution panel spanning the bottom row
axm = fig.add_subplot(gs[nrow - 1, :])
all_mvl = [mvl[k] for k in units_all.keys() if k in mvl]
axm.hist(all_mvl, bins=25, color="gray", alpha=0.7, label="all units")
axm.hist([mvl[k] for k in hd_cells], bins=25, color="steelblue", label="HD cells")
axm.axvline(0.3, color="r", ls="--", lw=1, label="MVL floor (0.3)")
axm.set_xlabel("mean vector length (wake)")
axm.set_ylabel("unit count")
axm.legend(fontsize=8)
fig.suptitle(f"{PROTO}: {n} HD cells — wake tuning curves (polar, Hz) and MVL selection")
fig.savefig("fig2_tuning_curves.png", dpi=150, bbox_inches="tight")
print("saved fig2_tuning_curves.png")

# %% [markdown]
# ## The population bump tracks the actual head direction
#
# Sorting the HD cells by preferred direction turns the population vector into an image of
# the ring: at every moment a single bump of activity sits at the represented angle. During
# wakefulness the bump tracks the LED-measured head direction, and a Bayesian decoder
# trained on the wake tuning curves recovers HD accurately.

# %%
counts_wake = units.count(BIN, wake)
decoded_wake, _ = nap.decode_bayes(tc_rate, units, wake, BIN)
n_spk = counts_wake.values.sum(axis=1)
valid_bins = n_spk >= 2
actual = counts_wake.value_from(hd)
err = np.angle(np.exp(1j * (decoded_wake.values - actual.values)))
err_valid = err[valid_bins & np.isfinite(actual.values)]
med_err = np.degrees(np.median(np.abs(err_valid)))
print(f"wake decode: median |error| = {med_err:.1f} deg over {valid_bins.sum()} bins")

# pick a 60 s wake window with full HD coverage for display
hdw = hd.restrict(wake)
best, best_t0 = -1, None
for cand in np.arange(wake.start[0], wake.end[-1] - 60, 20):
    mm = (hdw.t >= cand) & (hdw.t < cand + 60) & ~np.isnan(hdw.d)
    if mm.sum() < 100:
        continue
    cov = len(np.unique(np.floor(hdw.d[mm] / (2 * np.pi) * 36)))
    if cov > best:
        best, best_t0 = cov, cand
t0, t1 = best_t0, best_t0 + 60

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
m = (counts_wake.t >= t0) & (counts_wake.t <= t1)
Xs = norm_rows(counts_wake.values[m][:, order].T, 2)
grid, Xg = to_angle_grid(Xs, prefs_sorted)
ax.imshow(Xg, aspect="auto", origin="lower", cmap="viridis",
          extent=[counts_wake.t[m][0], counts_wake.t[m][-1], 0, 2 * np.pi])
hdm = (hd.t >= t0) & (hd.t <= t1)
ax.plot(hd.t[hdm], hd.d[hdm], "r.", ms=1.5, label="actual HD")
dm = (decoded_wake.t >= t0) & (decoded_wake.t <= t1)
ax.plot(decoded_wake.t[dm], decoded_wake.values[dm], "w.", ms=1.5, label="decoded HD")
ax.set_yticks(*RAD_TICKS)
ax.set_ylabel("angle (rad)")
ax.set_title(f"Population activity of {n} HD cells mapped onto the ring — wake")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.plot(actual.values[valid_bins], decoded_wake.values[valid_bins], ".", ms=1, alpha=0.3)
ax.plot([0, 2 * np.pi], [0, 2 * np.pi], "r-", lw=1)
ax.set_xlabel("actual HD (rad)")
ax.set_ylabel("decoded HD (rad)")
ax.set_xticks(*RAD_TICKS)
ax.set_yticks(*RAD_TICKS)
ax.set_title("Wake decoding")

ax = fig.add_subplot(gs[1, 1])
ax.hist(np.degrees(err_valid), bins=60, range=(-180, 180))
ax.set_xlabel("decode error (deg)")
ax.set_ylabel("bin count")
ax.set_title(f"Wake error, median |err| = {med_err:.0f}°")
fig.savefig("fig3_wake_bump_decode.png", dpi=150)
print("saved fig3_wake_bump_decode.png")

# %% [markdown]
# ## During REM sleep the bump persists and drifts around the ring
#
# The same wake-trained decoder is now applied to REM sleep, when the animal is immobile
# and the HD signal carries no sensory input. If the ring is internally maintained, the
# decoded angle should move *continuously* — the bump slides around the ring rather than
# jumping. We quantify continuity by the distribution of angular speeds and compare it to
# a control in which the time bins of the spike-count matrix are shuffled within each REM
# epoch (same spikes, same decoder, no temporal continuity).

# %%
decoded_rem, _ = nap.decode_bayes(tc_rate, units, rem, BIN)
counts_rem = units.count(BIN, rem)
valid_rem = counts_rem.values.sum(axis=1) >= 2
speed_rem = angular_speed(decoded_rem, valid_rem)

counts_shuf = counts_rem.values.copy()
for ep_i in range(len(rem)):
    idx = np.where((counts_rem.t >= rem.start[ep_i]) & (counts_rem.t <= rem.end[ep_i]))[0]
    counts_shuf[idx] = counts_shuf[idx][rng.permutation(len(idx))]
counts_shuf_tsd = nap.TsdFrame(t=counts_rem.t, d=counts_shuf,
                               columns=counts_rem.columns, time_support=rem)
decoded_shuf, _ = nap.decode_bayes(tc_rate, counts_shuf_tsd, rem, BIN)
speed_shuf = angular_speed(decoded_shuf, counts_shuf.sum(axis=1) >= 2)
print(f"REM angular speed: real median {np.median(speed_rem):.0f} deg/s, "
      f"bin-shuffle median {np.median(speed_shuf):.0f} deg/s")

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
iep = int(np.argmax(rem.end - rem.start))  # longest REM episode
t0r, t1r = float(rem.start[iep]), float(rem.end[iep])
m = (counts_rem.t >= t0r) & (counts_rem.t <= t1r)
Xs = norm_rows(counts_rem.values[m][:, order].T, 3)
grid, Xg = to_angle_grid(Xs, prefs_sorted)
ax.imshow(Xg, aspect="auto", origin="lower", cmap="viridis",
          extent=[counts_rem.t[m][0], counts_rem.t[m][-1], 0, 2 * np.pi])
dm = (decoded_rem.t >= t0r) & (decoded_rem.t <= t1r)
ax.plot(decoded_rem.t[dm], decoded_rem.values[dm], "w.", ms=2,
        label="decoded internal HD")
ax.set_yticks(*RAD_TICKS)
ax.set_ylabel("angle (rad)")
ax.set_title(f"REM sleep episode ({t1r - t0r:.0f} s): the bump drifts with no sensory input")
ax.legend(loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1, 0])
ax.plot(decoded_rem.t[dm], np.unwrap(decoded_rem.values[dm]), lw=1)
ax.set_xlabel("time (s)")
ax.set_ylabel("unwrapped decoded angle (rad)")
ax.set_title("Decoded angle rotates continuously during REM")

ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, 720, 60)
ax.hist(speed_rem, bins=bins, alpha=0.7, density=True, label="REM (real)")
ax.hist(speed_shuf, bins=bins, alpha=0.7, density=True, label="time-bin shuffle")
ax.set_xlabel("|angular speed| (deg/s)")
ax.set_ylabel("density")
ax.legend()
ax.set_title(f"median {np.median(speed_rem):.0f} vs {np.median(speed_shuf):.0f} deg/s")
fig.savefig("fig4_rem_bump_speed.png", dpi=150)
print("saved fig4_rem_bump_speed.png")

# %% [markdown]
# ## The ring's correlation structure is preserved in REM and Non-REM sleep
#
# The ring topology predicts a specific pairwise-correlation pattern: cells with similar
# preferred directions are correlated, cells ~180° apart are anti-correlated. We bin spikes
# at 100 ms within each state, correlate every pair of HD cells, and order the matrix by
# preferred direction. The wake pattern reappears in both sleep states; the
# correlation-of-correlations between the wake and sleep matrices quantifies it, with a
# label-permutation null (cell identities shuffled) for significance.

# %%
Cc = {name: state_corr(units, ep) for name, ep in
      [("wake", wake), ("REM", rem), ("NREM", nrem)]}
iu = np.triu_indices(n, k=1)
ang_dist = np.abs(np.angle(np.exp(1j * (prefs[iu[0]] - prefs[iu[1]]))))
r_rem = corr_of_corr(Cc["wake"], Cc["REM"], iu)
r_nrem = corr_of_corr(Cc["wake"], Cc["NREM"], iu)

n_perm = 1000
null_rem, null_nrem = np.empty(n_perm), np.empty(n_perm)
for p in tqdm(range(n_perm), desc="permutation null"):
    perm = rng.permutation(n)
    null_rem[p] = corr_of_corr(Cc["wake"], Cc["REM"][np.ix_(perm, perm)], iu)
    null_nrem[p] = corr_of_corr(Cc["wake"], Cc["NREM"][np.ix_(perm, perm)], iu)
p_rem = (np.sum(null_rem >= r_rem) + 1) / (n_perm + 1)
p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (n_perm + 1)
print(f"corr-of-corr wake-REM: {r_rem:.3f} (p={p_rem:.4f}); "
      f"wake-NREM: {r_nrem:.3f} (p={p_nrem:.4f})")

fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 4, wspace=0.4)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("lightgray")
mat_axes = []
for i, name in enumerate(["wake", "REM", "NREM"]):
    ax = fig.add_subplot(gs[0, i])
    Cs = np.ma.masked_invalid(Cc[name][np.ix_(order, order)])
    im = ax.imshow(Cs, cmap=cmap, vmin=-0.6, vmax=0.6, origin="lower")
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("HD cells (sorted by pref. dir.)")
    mat_axes.append(ax)
fig.colorbar(im, ax=mat_axes, fraction=0.03, pad=0.02, label="pairwise correlation")

ax = fig.add_subplot(gs[0, 3])
edges = np.linspace(0, np.pi, 13)
ctr = np.degrees(0.5 * (edges[:-1] + edges[1:]))
for name, c in [("wake", "k"), ("REM", "r"), ("NREM", "b")]:
    v = Cc[name][iu]
    mn = [np.nanmean(v[(ang_dist >= edges[j]) & (ang_dist < edges[j + 1])])
          for j in range(12)]
    ax.plot(ctr, mn, ".-", color=c, label=name, ms=4, lw=1)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ preferred direction (deg)")
ax.set_ylabel("pairwise correlation")
ax.legend(fontsize=8)
ax.set_title(f"corr-of-corr: REM {r_rem:.2f}, NREM {r_nrem:.2f}")
fig.savefig("fig5_corr_matrices.png", dpi=150)
print("saved fig5_corr_matrices.png")

# %% [markdown]
# ## Replication across five sessions (one per mouse)
#
# The same pipeline is run on four additional sessions from four other mice. For each
# session we compute the wake–REM and wake–NREM correlation-of-correlations with a
# 1000-fold label-permutation null, and we pool all cell pairs for the
# correlation-vs-angular-distance curves.

# %%
panel = {PROTO: dict(n_units=len(units_all), hd_cells=hd_cells, prefs=prefs,
                     Cw=Cc["wake"], Cr=Cc["REM"], Cn=Cc["NREM"],
                     r_rem=r_rem, r_nrem=r_nrem,
                     null_rem=null_rem, null_nrem=null_nrem,
                     p_rem=p_rem, p_nrem=p_nrem, iu=iu)}

for ses, aid in tqdm(list(ASSETS.items()), desc="sessions"):
    if ses == PROTO:
        continue
    nwb_s = load_session(aid)
    ua = clean_units(nwb_s["units"])
    st = nwb_s["states"]
    wk = st[st["label"] == "Awake"]
    rm = st[st["label"] == "REM"]
    nr = st[st["label"] == "Non-REM"]
    hd_s = compute_hd(nwb_s)
    mvl_s, pval_s, cells_s = identify_hd_cells(ua, hd_s, wk)
    us = ua[cells_s]
    tc_s = nap.compute_tuning_curves(us, hd_s, bins=60, range=(0, 2 * np.pi),
                                     epochs=wk, return_counts=True)
    occ_s = tc_s.attrs["occupancy"] / tc_s.attrs["fs"]
    tcr_s = tc_s / np.where(occ_s == 0, np.nan, occ_s)
    bc = tc_s.coords[tc_s.dims[1]].values
    prefs_s = np.array([bc[np.nanargmax(tcr_s.values[i])] for i in range(len(cells_s))])
    Cw = state_corr(us, wk)
    Cr = state_corr(us, rm)
    Cn = state_corr(us, nr)
    iu_s = np.triu_indices(len(cells_s), k=1)
    r_r = corr_of_corr(Cw, Cr, iu_s)
    r_n = corr_of_corr(Cw, Cn, iu_s)
    nl_r, nl_n = np.empty(n_perm), np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(len(cells_s))
        nl_r[p] = corr_of_corr(Cw, Cr[np.ix_(perm, perm)], iu_s)
        nl_n[p] = corr_of_corr(Cw, Cn[np.ix_(perm, perm)], iu_s)
    panel[ses] = dict(n_units=len(ua), hd_cells=cells_s, prefs=prefs_s,
                      Cw=Cw, Cr=Cr, Cn=Cn, r_rem=r_r, r_nrem=r_n,
                      null_rem=nl_r, null_nrem=nl_n,
                      p_rem=(np.sum(nl_r >= r_r) + 1) / (n_perm + 1),
                      p_nrem=(np.sum(nl_n >= r_n) + 1) / (n_perm + 1), iu=iu_s)
    print(f"{ses}: {len(cells_s)}/{len(ua)} HD cells, "
          f"REM r={r_r:.2f} (p={panel[ses]['p_rem']:.4f}), "
          f"NREM r={r_n:.2f} (p={panel[ses]['p_nrem']:.4f})")

# %% [markdown]
# ### Figure 6 — cross-session summary

# %%
sessions = list(panel.keys())
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
x = np.arange(len(sessions))
ax.bar(x, [panel[s]["n_units"] for s in sessions], color="lightgray", label="all units")
ax.bar(x, [len(panel[s]["hd_cells"]) for s in sessions], color="steelblue",
       label="HD cells")
ax.set_xticks(x, [s.replace("Mouse", "M") for s in sessions], fontsize=8)
ax.set_ylabel("unit count")
ax.legend()
ax.set_title("HD cell yield per session")

ax = axes[0, 1]
w = 0.35
for j, (key, nullkey, c, lab) in enumerate([("r_rem", "null_rem", "r", "wake–REM"),
                                            ("r_nrem", "null_nrem", "b", "wake–NREM")]):
    vals = [panel[s][key] for s in sessions]
    null_hi = [np.percentile(panel[s][nullkey], 99) for s in sessions]
    ax.bar(x + (j - 0.5) * w, vals, w, color=c, alpha=0.8, label=lab)
    ax.scatter(x + (j - 0.5) * w, null_hi, marker="_", color="k", s=60,
               label="99th pctile of permutation null" if j == 0 else None)
ax.set_xticks(x, [s.replace("Mouse", "M") for s in sessions], fontsize=8)
ax.set_ylabel("corr-of-corr (Pearson r)")
ax.set_ylim(0, 1.35)
ax.legend(fontsize=8, loc="upper center", ncol=3)
ax.set_title("Wake vs sleep correlation-structure preservation")

ax = axes[1, 0]
for key, c, lab in [("Cw", "k", "wake"), ("Cr", "r", "REM"), ("Cn", "b", "NREM")]:
    all_d, all_v = [], []
    for s in sessions:
        pr = panel[s]["prefs"]
        iu_s = panel[s]["iu"]
        d = np.abs(np.angle(np.exp(1j * (pr[iu_s[0]] - pr[iu_s[1]]))))
        v = panel[s][key][iu_s]
        ok = np.isfinite(v)
        all_d.append(d[ok])
        all_v.append(v[ok])
    all_d = np.concatenate(all_d)
    all_v = np.concatenate(all_v)
    mn = [np.mean(all_v[(all_d >= edges[j]) & (all_d < edges[j + 1])]) for j in range(12)]
    se = [np.std(all_v[(all_d >= edges[j]) & (all_d < edges[j + 1])]) /
          np.sqrt(np.sum((all_d >= edges[j]) & (all_d < edges[j + 1]))) for j in range(12)]
    ax.errorbar(ctr, mn, yerr=se, fmt=".-", color=c, label=lab, ms=5, lw=1, capsize=2)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("Δ preferred direction (deg)")
ax.set_ylabel("pairwise correlation")
ax.legend()
ax.set_title("Ring topology preserved in sleep (pooled pairs)")

ax = axes[1, 1]
for key, c, lab in [("Cr", "r", "REM"), ("Cn", "b", "NREM")]:
    vw, vs = [], []
    for s in sessions:
        iu_s = panel[s]["iu"]
        a, b = panel[s]["Cw"][iu_s], panel[s][key][iu_s]
        ok = np.isfinite(a) & np.isfinite(b)
        vw.append(a[ok])
        vs.append(b[ok])
    vw = np.concatenate(vw)
    vs = np.concatenate(vs)
    ax.scatter(vw, vs, s=3, alpha=0.3, color=c,
               label=f"{lab} (r={np.corrcoef(vw, vs)[0, 1]:.2f})")
ax.plot([-0.6, 1], [-0.6, 1], "k--", lw=0.8)
ax.set_xlabel("wake pairwise correlation")
ax.set_ylabel("sleep pairwise correlation")
ax.legend(fontsize=8, markerscale=3)
ax.set_title("Pair-level preservation")
fig.tight_layout()
fig.savefig("fig6_panel.png", dpi=150)
print("saved fig6_panel.png")

# %% [markdown]
# ## Summary
#
# On data from Peyrache et al. (2015, DANDI 000056):
#
# * **Ring structure during wakefulness.** ADn units are sharply tuned to head direction
#   (20/45 units in the prototype session pass an MVL > 0.3 floor against a random-time
#   null), their preferred directions tile the circle, and the population activity forms a
#   single bump on that ring that tracks the measured heading (Bayesian decode median
#   error ~14° at 100 ms resolution).
# * **Internal maintenance during sleep.** With the animal asleep and immobile, the same
#   bump persists and drifts continuously around the ring: during REM the decoded internal
#   heading rotates smoothly (median ~120°/s, versus ~660°/s after destroying temporal
#   continuity by shuffling time bins), completing multiple full revolutions within a
#   single REM episode.
# * **Preserved topology.** The pairwise correlation structure — positive between cells
#   with nearby preferred directions, negative near 180° separation, as a ring predicts —
#   is essentially unchanged in REM and Non-REM sleep (correlation-of-correlations ≈ 0.92
#   in the prototype session, permutation p ≤ 0.001), and this replicates across five
#   sessions from five mice.
#
# Together these are the defining signatures of a **continuous ring attractor**: a
# one-dimensional manifold of stable activity states, internally maintained by recurrent
# circuitry even when sensory input is absent, exactly as reported in the original
# publication.
