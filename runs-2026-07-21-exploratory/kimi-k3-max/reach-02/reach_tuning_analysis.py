# %% [markdown]
# # Reach Direction and Velocity Tuning in Macaque Motor Cortex
#
# **Dataset:** [DANDI 000128](https://dandiarchive.org/dandiset/000128): MC_Maze
# (Neural Latents Benchmark '21), macaque primary motor (M1) and dorsal premotor
# cortex (PMd) spiking activity during a delayed center-out reaching task with
# virtual barriers. Subject Jenkins, 182 sorted units, 2295 successful trials,
# hand position and velocity sampled at 1 kHz.
#
# **Question.** Two classical observations about motor cortex are demonstrated
# here on real data:
#
# 1. **Reach-direction tuning.** Single neurons fire most for reaches toward a
#    preferred direction and progressively less for directions further away from
#    it, a relationship well described by a cosine (Georgopoulos et al., 1982).
# 2. **Velocity tuning.** Instantaneous firing rate tracks the hand velocity
#    vector, with neural activity leading the hand by roughly 50–150 ms, and
#    rates typically growing with hand speed (Moran & Schwartz, 1999).
#
# The analysis streams the NWB file directly from the DANDI S3 bucket with
# `remfile` (no full download), uses Pynapple for time-series handling, fits
# cosine tuning curves with linear regression, quantifies velocity encoding with
# lagged regression, and finally fits per-unit Poisson GLMs with `nemos` as a
# principled encoding model of direction and speed.

# %% [markdown]
# ## 1. Setup and streaming data access

# %%
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless run: save figures, never plt.show()
import matplotlib.pyplot as plt
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# numpy 2.0.x with the macOS Accelerate BLAS emits spurious RuntimeWarnings from
# matmul ("divide by zero/overflow/invalid value encountered in matmul") even for
# finite, well-conditioned float64 inputs; the computed results are correct.
# Suppress only those matmul-sourced messages; every other warning still prints.
warnings.filterwarnings(
    "ignore",
    message=r"(divide by zero|overflow|invalid value) encountered in matmul",
    category=RuntimeWarning,
)

os.makedirs("cache", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Direct S3 blob URL for the MC_Maze train file (behavior + ecephys) on DANDI 000128.
S3_MCMAZE = "https://dandiarchive.s3.amazonaws.com/blobs/df3/e3f/df3e3f73-50ab-42b4-8827-82664ddd474a"

# %%
disk_cache = remfile.DiskCache("cache/remfile_cache")
rem_file = remfile.File(S3_MCMAZE, disk_cache=disk_cache)
h5py_file = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5py_file)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)

# %%
units = nwb["units"]          # TsGroup, 182 sorted units
trials_iset = nwb["trials"]   # IntervalSet with trial metadata
hand_pos = nwb["hand_pos"]    # TsdFrame, 1 kHz
hand_vel = nwb["hand_vel"]    # TsdFrame, 1 kHz

trials = nwbfile.intervals["trials"].to_dataframe()
print(f"{len(units)} units, {len(trials)} trials")
print("trial columns:", [c for c in trials.columns])

# Unit ids encode electrode channel as id // 10. Channels fall in two blocks
# (101-190 and 205-296) corresponding to the two 96-channel Utah arrays.
unit_ids = np.asarray(units.index)
channels = unit_ids // 10
array_block = np.where(channels < 200, "array 1 (ch 1xx)", "array 2 (ch 2xx)")

# %% [markdown]
# ## 2. Session overview: targets, kinematics, and spiking
#
# Each trial the monkey holds at the center, one or more targets appear, and
# after a delay and go cue the monkey reaches to the cued target, possibly
# steering around virtual barriers. Target positions are jittered from trial to
# trial, so rather than trusting nominal target angles I compute each trial's
# **reach direction** from the hand kinematics themselves: the direction of hand
# displacement between movement onset and the moment of peak speed. This is the
# initial, feed-forward phase of the reach.

# %%
pos = hand_pos.d
vel = hand_vel.d
ts = hand_pos.t

def sample_at(t):
    return np.searchsorted(ts, t)

move_on = trials["move_onset_time"].values
t_stop = trials["stop_time"].values
go_cue = trials["go_cue_time"].values
target_on = trials["target_on_time"].values

reach_dir = np.full(len(trials), np.nan)
peak_speed = np.full(len(trials), np.nan)
for i in range(len(trials)):
    i0, i1 = sample_at(move_on[i]), sample_at(t_stop[i])
    sp = np.linalg.norm(vel[i0:i1], axis=1)
    ipk = i0 + int(np.argmax(sp))
    disp = pos[ipk] - pos[i0]
    reach_dir[i] = np.arctan2(disp[1], disp[0])
    peak_speed[i] = sp.max()

trials["reach_dir"] = reach_dir
trials["peak_speed"] = peak_speed
straight = trials["trial_version"].values == 0  # barrier-free trials
print(f"straight (no-barrier) trials: {straight.sum()} / {len(trials)}")
print(f"median peak speed: {np.median(peak_speed):.0f} mm/s")

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

# (a) active target positions
ax = axes[0, 0]
active_pos = []
for _, row in trials.iterrows():
    p = np.asarray(row["target_pos"]).reshape(-1, 2)
    active_pos.append(p[int(row["active_target"])])
active_pos = np.array(active_pos)
ax.scatter(active_pos[:, 0], active_pos[:, 1], s=3, alpha=0.3, color="tab:blue")
ax.set_title("Cued target positions")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_aspect("equal")

# (b) reach direction histogram
ax = axes[0, 1]
ax.hist(np.degrees(reach_dir[straight]), bins=np.arange(-180, 181, 10), color="tab:blue")
ax.set_title("Reach direction (straight trials)")
ax.set_xlabel("direction (deg)"); ax.set_ylabel("trials")

# (c) hand position snippet
ax = axes[0, 2]
t0, t1 = 100.0, 112.0
hp = hand_pos.get(t0, t1)
ax.plot(hp.t, hp.d[:, 0], label="x", lw=1)
ax.plot(hp.t, hp.d[:, 1], label="y", lw=1)
ax.set_title("Hand position snippet")
ax.set_xlabel("time (s)"); ax.set_ylabel("position (mm)")
ax.legend(frameon=False)

# (d) speed snippet with movement onsets
ax = axes[1, 0]
hv = hand_vel.get(t0, t1)
sp = np.linalg.norm(hv.d, axis=1)
ax.plot(hv.t, sp, color="k", lw=1)
for m in move_on[(move_on >= t0) & (move_on <= t1)]:
    ax.axvline(m, color="r", ls="--", alpha=0.6, lw=0.8)
ax.set_title("Hand speed (red = move onset)")
ax.set_xlabel("time (s)"); ax.set_ylabel("speed (mm/s)")

# (e) spike raster snippet, 30 units
ax = axes[1, 1]
ep = nap.IntervalSet(t0, t1)
show_units = unit_ids[::6]
y = 0
for u in show_units:
    st = units[u].restrict(ep).t
    ax.plot(st, np.full_like(st, y), "|", color="k", ms=2)
    y += 1
ax.set_title(f"Spike raster ({len(show_units)} units)")
ax.set_xlabel("time (s)"); ax.set_ylabel("unit")
ax.set_yticks([])

# (f) firing-rate distribution across units
ax = axes[1, 2]
rates = units.get_info("rate").values
ax.hist(rates, bins=30, color="tab:blue")
ax.set_title("Mean firing rates")
ax.set_xlabel("rate (Hz)"); ax.set_ylabel("units")

fig.tight_layout()
fig.savefig("figures/fig01_session_overview.png", dpi=150)
plt.close(fig)
print("saved figures/fig01_session_overview.png")

# %%
# Trajectories of barrier-free trials, colored by reach direction.
fig, ax = plt.subplots(figsize=(6.5, 6))
cmap = plt.cm.hsv
idx = np.where(straight)[0]
for i in idx[::3]:
    i0, i1 = sample_at(move_on[i]), sample_at(t_stop[i])
    c = cmap((reach_dir[i] + np.pi) / (2 * np.pi))
    ax.plot(pos[i0:i1, 0], pos[i0:i1, 1], color=c, lw=0.5, alpha=0.5)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title("Straight-trial hand trajectories, colored by reach direction")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("figures/fig02_trajectories.png", dpi=150)
plt.close(fig)
print("saved figures/fig02_trajectories.png")

# %% [markdown]
# ## 3. Reach-direction tuning (trial-based analysis)
#
# For every unit I compute the firing rate in a perimovement window
# (movement onset −50 ms to +450 ms) on each trial, then regress those rates on
# the cosine and sine of the reach direction:
#
# $$ r(\theta) = b_0 + b_c \cos\theta + b_s \sin\theta = b_0 + A\cos(\theta - \phi) $$
#
# which is exactly the classical cosine tuning model. The fit yields a preferred
# direction $\phi$, a modulation amplitude $A$, and an $R^2$. Significance is
# assessed per unit with a permutation test (1000 shuffles of the direction
# labels, statistic = $R^2$).

# %%
MOVE_WIN = (-0.05, 0.45)      # perimovement rate window (s)
BASE_WIN = (-0.35, -0.05)     # pre-target baseline window relative to target_on
n_units = len(units)
n_trials = len(trials)

move_rate = np.zeros((n_trials, n_units), dtype=np.float64)
base_rate = np.zeros((n_trials, n_units), dtype=np.float64)
for j, u in enumerate(tqdm(unit_ids, desc="trial rates")):
    st = units[u].t
    i0 = np.searchsorted(st, move_on + MOVE_WIN[0])
    i1 = np.searchsorted(st, move_on + MOVE_WIN[1])
    move_rate[:, j] = (i1 - i0) / (MOVE_WIN[1] - MOVE_WIN[0])
    k0 = np.searchsorted(st, target_on + BASE_WIN[0])
    k1 = np.searchsorted(st, target_on + BASE_WIN[1])
    base_rate[:, j] = (k1 - k0) / (BASE_WIN[1] - BASE_WIN[0])

# %%
def cosine_design(theta):
    return np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)

def fit_cosine_all(theta, Y):
    """Least-squares cosine fit of every column of Y on theta. Returns R2, coefs."""
    X = cosine_design(theta)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ beta
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    ss_tot[ss_tot == 0] = np.nan
    return 1 - ss_res / ss_tot, beta

theta = reach_dir.astype(np.float64)
r2_dir, beta_dir = fit_cosine_all(theta, move_rate)
pref_dir = np.arctan2(beta_dir[2], beta_dir[1])          # preferred direction
mod_amp = np.hypot(beta_dir[1], beta_dir[2])             # A, in Hz
mod_depth = mod_amp / np.maximum(beta_dir[0], 1e-9)      # A / mean rate

# permutation test
rng = np.random.default_rng(0)
n_shuf = 1000
r2_null = np.zeros((n_shuf, n_units), dtype=np.float32)
for s in tqdm(range(n_shuf), desc="permutation test"):
    r2_null[s], _ = fit_cosine_all(theta[rng.permutation(n_trials)], move_rate)
p_dir = (1 + (r2_null >= r2_dir[None, :]).sum(axis=0)) / (n_shuf + 1)
p_dir[~np.isfinite(r2_dir)] = 1.0  # units with no spikes in the window are not tuned
sig_dir = p_dir < 0.01
print(f"direction-tuned units (permutation p<0.01): {sig_dir.sum()} / {n_units}")
print(f"median R2 (all units): {np.nanmedian(r2_dir):.3f}; "
      f"median R2 (tuned): {np.nanmedian(r2_dir[sig_dir]):.3f}")

# %% [markdown]
# ### Example units: rasters and per-direction PSTHs

# %%
edges8 = np.linspace(-np.pi, np.pi, 9)
sector = np.digitize(reach_dir, edges8) - 1
sector[sector == 8] = 0
sector_centers = (edges8[:-1] + edges8[1:]) / 2

# two strongly tuned example units
r2_dir_filled = np.nan_to_num(r2_dir, nan=-1.0)
tuned_order = np.argsort(np.where(sig_dir, r2_dir_filled, -1.0))[::-1]
example_units = tuned_order[:2]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
win = (-0.3, 0.6)
bins = np.arange(win[0], win[1] + 0.02, 0.02)
for col, j in enumerate(example_units):
    u = unit_ids[j]
    spk = units[u]
    ax = axes[0, col]
    y = 0
    yticks, ylabels = [], []
    for s in range(8):
        trs = np.where(sector == s)[0]
        pe = nap.compute_perievent(spk, nap.Ts(move_on[trs]), window=win)
        for k in range(len(trs)):
            st = pe[k].t
            ax.plot(st, np.full_like(st, y), "|", color=plt.cm.hsv(s / 8), ms=1.2)
            y += 1
        yticks.append(y - len(trs) / 2)
        ylabels.append(f"{int(np.degrees(sector_centers[s]))}")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_ylabel("reach direction (deg)")
    ax.set_title(f"unit {u}  (R$^2$={r2_dir[j]:.2f}, PD={np.degrees(pref_dir[j]):.0f}°)")

    ax = axes[1, col]
    for s in range(8):
        trs = np.where(sector == s)[0]
        pe = nap.compute_perievent(spk, nap.Ts(move_on[trs]), window=win)
        rel = np.concatenate([pe[k].t for k in range(len(trs))]) if len(trs) else np.array([])
        counts, _ = np.histogram(rel, bins=bins)
        ax.plot(bins[:-1] + 0.01, counts / (len(trs) * 0.02), color=plt.cm.hsv(s / 8), lw=1.2)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from movement onset (s)")
    ax.set_ylabel("rate (Hz)")
axes[0, 0].set_xlabel("time from movement onset (s)")
fig.suptitle("Spike rasters (top) and per-direction PSTHs (bottom), aligned to movement onset")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig03_direction_raster_psth.png", dpi=150)
plt.close(fig)
print("saved figures/fig03_direction_raster_psth.png")

# %% [markdown]
# ### Cosine tuning curves across the population

# %%
# mean rate per 45-degree sector, with the cosine fit overlaid
top12 = tuned_order[:12]
fig = plt.figure(figsize=(15, 10))
fine = np.linspace(-np.pi, np.pi, 200)
for k, j in enumerate(top12):
    ax = fig.add_subplot(3, 4, k + 1, projection="polar")
    means = [move_rate[sector == s, j].mean() for s in range(8)]
    th = np.concatenate([sector_centers, [sector_centers[0] + 2 * np.pi]])
    r = np.concatenate([means, [means[0]]])
    ax.plot(th, r, "o-", color="tab:blue", lw=1.5, ms=4, label="data")
    fit = beta_dir[0, j] + mod_amp[j] * np.cos(fine - pref_dir[j])
    ax.plot(fine, np.maximum(fit, 0), color="tab:red", lw=1.5, label="cosine fit")
    ax.set_title(f"unit {unit_ids[j]}\nR²={r2_dir[j]:.2f}, PD={np.degrees(pref_dir[j]):.0f}°",
                 fontsize=9, pad=26)
    ax.set_theta_zero_location("E")
    ax.tick_params(labelsize=7)
fig.suptitle("Reach-direction tuning curves (12 best-tuned units), perimovement rates")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig04_polar_tuning.png", dpi=150)
plt.close(fig)
print("saved figures/fig04_polar_tuning.png")

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) preferred-direction histogram for tuned units
axes[0, 0].remove()
ax = fig.add_subplot(2, 2, 1, projection="polar")
pdirs = pref_dir[sig_dir]
counts_pd, edges_pd = np.histogram(pdirs, bins=np.linspace(-np.pi, np.pi, 17))
centers_pd = (edges_pd[:-1] + edges_pd[1:]) / 2
ax.bar(centers_pd, counts_pd, width=np.diff(edges_pd), color="tab:blue", alpha=0.8)
ax.set_theta_zero_location("E")
ax.set_title(f"Preferred directions (n={sig_dir.sum()} tuned units)", pad=24)

# (b) R2 distribution vs shuffle
ax = axes[0, 1]
ax.hist(r2_null.flatten(), bins=60, color="gray", alpha=0.6, density=True, label="shuffled")
ax.hist(r2_dir, bins=60, color="tab:blue", alpha=0.7, density=True, label="observed")
ax.set_yscale("log")
ax.set_xlim(0, np.nanpercentile(r2_dir, 99.5) * 1.1)
ax.set_xlabel("cosine-fit $R^2$"); ax.set_ylabel("density (log scale)")
ax.set_title("Direction-tuning strength vs chance")
ax.legend(frameon=False)

# (c) baseline vs movement rate
ax = axes[1, 0]
bm = base_rate.mean(axis=0); mm = move_rate.mean(axis=0)
ax.scatter(bm[~sig_dir], mm[~sig_dir], s=12, color="gray", alpha=0.6, label="not tuned")
ax.scatter(bm[sig_dir], mm[sig_dir], s=12, color="tab:blue", alpha=0.8, label="tuned")
lim = max(bm.max(), mm.max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("baseline rate (Hz)"); ax.set_ylabel("perimovement rate (Hz)")
ax.set_title("Movement vs baseline firing")
ax.legend(frameon=False)

# (d) modulation depth by array block
ax = axes[1, 1]
for k, blk in enumerate(["array 1 (ch 1xx)", "array 2 (ch 2xx)"]):
    m = sig_dir & (array_block == blk)
    ax.hist(mod_depth[m], bins=np.arange(0, 1.5, 0.08), alpha=0.6,
            label=f"{blk} (n={m.sum()})")
ax.set_xlabel("modulation depth $A / b_0$")
ax.set_ylabel("units")
ax.set_title("Tuning depth by electrode array")
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig("figures/fig05_direction_population.png", dpi=150)
plt.close(fig)
print("saved figures/fig05_direction_population.png")

# %% [markdown]
# ### Population-vector decoding of reach direction
#
# Georgopoulos' population vector: each tuned unit votes for its preferred
# direction, weighted by its perimovement rate on that trial; the angle of the
# summed vector is the decoded direction. To keep the estimate honest, preferred
# directions are fit on one half of the trials and decoding is evaluated on the
# other half (and vice versa).

# %%
def population_vector_decode(train_idx, test_idx):
    r2_tr, b = fit_cosine_all(theta[train_idx], move_rate[train_idx])
    pd = np.arctan2(b[2], b[1])
    use = r2_tr > 0.05  # unit selection uses training half only
    V = np.stack([np.cos(pd[use]), np.sin(pd[use])], axis=0)  # 2 x n
    R = move_rate[np.ix_(test_idx, np.where(use)[0])]         # trials x n
    pv = R.astype(np.float64) @ V.T                           # trials x 2
    return np.arctan2(pv[:, 1], pv[:, 0])

half = n_trials // 2
idx_all = np.arange(n_trials)
dec_a = population_vector_decode(idx_all[:half], idx_all[half:])
dec_b = population_vector_decode(idx_all[half:], idx_all[:half])
decoded = np.concatenate([dec_b, dec_a])  # every trial decoded from the other half
ang_err = np.angle(np.exp(1j * (decoded - theta)))
abs_err = np.abs(np.degrees(ang_err))
print(f"median absolute decoding error: {np.median(abs_err):.1f} deg (chance ~ 90 deg)")

# sector accuracy
dec_sector = np.digitize(decoded, edges8) - 1
dec_sector[dec_sector == 8] = 0
acc = (dec_sector == sector).mean()
print(f"8-sector accuracy: {acc:.2%} (chance 12.5%)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.scatter(np.degrees(theta), np.degrees(decoded), s=4, alpha=0.3, color="tab:blue")
ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
ax.set_xlabel("actual reach direction (deg)")
ax.set_ylabel("decoded direction (deg)")
ax.set_title("Population-vector decoding")

ax = axes[1]
ax.hist(abs_err, bins=np.arange(0, 181, 10), color="tab:blue")
ax.axvline(90, color="r", ls="--", label="chance median (90°)")
ax.axvline(np.median(abs_err), color="k", ls="-", label=f"observed median ({np.median(abs_err):.0f}°)")
ax.set_xlabel("absolute angular error (deg)"); ax.set_ylabel("trials")
ax.set_title("Decoding error")
ax.legend(frameon=False)

ax = axes[2]
per_sector = [(dec_sector[sector == s] == s).mean() if (sector == s).any() else np.nan
              for s in range(8)]
ax.bar(np.degrees(sector_centers), per_sector, width=30, color="tab:blue")
ax.axhline(1 / 8, color="r", ls="--", label="chance (12.5%)")
ax.set_xlabel("reach direction (deg)"); ax.set_ylabel("fraction correct")
ax.set_title("Accuracy by direction")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig06_population_decoding.png", dpi=150)
plt.close(fig)
print("saved figures/fig06_population_decoding.png")

# %% [markdown]
# ## 4. Velocity tuning (continuous analysis)
#
# Trial-averaged direction tuning is only part of the story: motor cortex rates
# track the hand **velocity vector** continuously. I bin spikes into 25 ms bins,
# smooth with a 50 ms Gaussian, and regress each unit's rate on the two velocity
# components at a range of lags (−250 to +250 ms). A positive best lag means the
# neural activity leads the hand. The linear model
# $r(t) = b_0 + b_x v_x(t+\ell) + b_y v_y(t+\ell)$ is exactly cosine tuning to
# velocity direction with amplitude proportional to speed, so its coefficients
# give a velocity preferred direction directly. Only perimovement bins
# (movement onset −150 ms to +650 ms) enter the regression.

# %%
BIN = 0.025
bin_edges = np.arange(ts[0], ts[-1] + BIN, BIN)
bin_centers = bin_edges[:-1] + BIN / 2
nb = len(bin_centers)

counts = np.zeros((n_units, nb), dtype=np.float64)
for j, u in enumerate(tqdm(unit_ids, desc="binning spikes")):
    counts[j] = np.histogram(units[u].t, bins=bin_edges)[0]
rate_s = gaussian_filter1d(counts / BIN, sigma=2.0, axis=1)

vx, _ = np.histogram(hand_vel.t, bins=bin_edges, weights=hand_vel.d[:, 0])
vy, _ = np.histogram(hand_vel.t, bins=bin_edges, weights=hand_vel.d[:, 1])
n_in_bin, _ = np.histogram(hand_vel.t, bins=bin_edges)
vx = vx / np.maximum(n_in_bin, 1)
vy = vy / np.maximum(n_in_bin, 1)
speed = np.hypot(vx, vy)

pm_mask = np.zeros(nb, dtype=bool)
for m in move_on:
    i0 = int((m - 0.15 - ts[0]) / BIN)
    i1 = int((m + 0.65 - ts[0]) / BIN)
    pm_mask[max(i0, 0):min(i1, nb)] = True
print(f"perimovement bins: {pm_mask.sum()} ({pm_mask.sum() * BIN / 60:.1f} min)")

# %%
lags = np.arange(-0.25, 0.251, BIN)
lag_bins = (lags / BIN).astype(int)
V = np.stack([vx, vy], axis=1)

def fit_r2_all(Y, v1, v2, m):
    Xm = np.stack([np.ones(m.sum()), v1[m], v2[m]], axis=1)
    Ym = Y[:, m]
    beta, *_ = np.linalg.lstsq(Xm, Ym.T, rcond=None)
    pred = (Xm @ beta).T
    ss_res = ((Ym - pred) ** 2).sum(axis=1)
    ss_tot = ((Ym - Ym.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    ss_tot[ss_tot == 0] = np.nan
    return 1 - ss_res / ss_tot, beta

r2_vs_lag = np.zeros((n_units, len(lags)), dtype=np.float32)
for li, lb in enumerate(tqdm(lag_bins, desc="lag sweep")):
    Vs = np.roll(V, -lb, axis=0)
    r2_l, _ = fit_r2_all(rate_s, Vs[:, 0], Vs[:, 1], pm_mask)
    r2_vs_lag[:, li] = r2_l

best_li = np.argmax(np.nan_to_num(r2_vs_lag, nan=-np.inf), axis=1)
best_lag = lags[best_li]
best_r2_vel = r2_vs_lag[np.arange(n_units), best_li]
print(f"median best lag: {np.median(best_lag) * 1000:.0f} ms "
      f"(positive = spikes lead velocity)")
print(f"median velocity R2 at best lag: {np.nanmedian(best_r2_vel):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(lags * 1000, np.nanmean(r2_vs_lag, axis=0), "k")
ax.axvline(0, color="gray", ls=":")
ax.set_xlabel("lag (ms)  [+ = spikes lead velocity]")
ax.set_ylabel("population mean $R^2$")
ax.set_title("Velocity encoding vs lag")

ax = axes[1]
ax.hist(best_lag * 1000, bins=lags * 1000, color="tab:blue")
ax.axvline(0, color="gray", ls=":")
ax.set_xlabel("best lag per unit (ms)"); ax.set_ylabel("units")
ax.set_title("Best-lag distribution")

ax = axes[2]
ax.hist(best_r2_vel[np.isfinite(best_r2_vel)], bins=30, color="tab:blue")
ax.set_xlabel("velocity $R^2$ at best lag"); ax.set_ylabel("units")
ax.set_title("Velocity-encoding strength")
fig.tight_layout()
fig.savefig("figures/fig07_velocity_lag.png", dpi=150)
plt.close(fig)
print("saved figures/fig07_velocity_lag.png")

# %% [markdown]
# ### Example unit: velocity direction, speed, and lag

# %%
j = int(np.nanargmax(best_r2_vel))
u = unit_ids[j]
lb = lag_bins[best_li[j]]
Vs = np.roll(V, -lb, axis=0)
r2_ex, beta_ex = fit_r2_all(rate_s[[j]], Vs[:, 0], Vs[:, 1], pm_mask)
beta_ex = beta_ex[:, 0]
pref_vel = np.arctan2(beta_ex[2], beta_ex[1])
spd_s = np.hypot(Vs[:, 0], Vs[:, 1])
print(f"example unit {u}: R2={r2_ex[0]:.3f}, lag={best_lag[j]*1000:.0f} ms, "
      f"velocity PD={np.degrees(pref_vel):.0f} deg")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

ax = axes[0, 0]
ax.plot(lags * 1000, r2_vs_lag[j], "k")
ax.axvline(best_lag[j] * 1000, color="r", ls="--")
ax.set_xlabel("lag (ms) [+ = spikes lead velocity]"); ax.set_ylabel("$R^2$")
ax.set_title(f"unit {u}: lag sweep")

ax = axes[0, 1]
vdir = np.arctan2(Vs[:, 1], Vs[:, 0])
moving = pm_mask & (spd_s > 100)
edges16 = np.linspace(-np.pi, np.pi, 17)
centers16 = (edges16[:-1] + edges16[1:]) / 2
means16 = [rate_s[j][moving & (vdir >= edges16[k]) & (vdir < edges16[k + 1])].mean()
           for k in range(16)]
ax.plot(np.degrees(centers16), means16, "o-", color="tab:blue")
fine = np.linspace(-180, 180, 200)
mean_spd = spd_s[moving].mean()
fit_curve = beta_ex[0] + np.hypot(beta_ex[1], beta_ex[2]) * mean_spd * np.cos(np.radians(fine) - pref_vel)
ax.plot(fine, np.maximum(fit_curve, 0), color="tab:red", lw=1.5, label="cosine model")
ax.set_xlabel("velocity direction (deg)"); ax.set_ylabel("rate (Hz)")
ax.set_title(f"unit {u}: velocity-direction tuning (speed>100 mm/s)")
ax.legend(frameon=False)

ax = axes[1, 0]
sbins = np.arange(0, 1000, 75)
sc = (sbins[:-1] + sbins[1:]) / 2
smeans = [rate_s[j][pm_mask & (spd_s >= sbins[k]) & (spd_s < sbins[k + 1])].mean()
          for k in range(len(sbins) - 1)]
r_spd = np.corrcoef(rate_s[j][pm_mask], spd_s[pm_mask])[0, 1]
ax.plot(sc, smeans, "o-", color="tab:blue")
ax.set_xlabel("hand speed (mm/s)"); ax.set_ylabel("rate (Hz)")
ax.set_title(f"unit {u}: rate vs speed (r={r_spd:.2f})")

ax = axes[1, 1]
sel = (bin_centers > 500) & (bin_centers < 512)
r_trace = rate_s[j][sel]
s_trace = spd_s[sel]
ax.plot(bin_centers[sel], r_trace / r_trace.max(), "k", label="rate (norm)", lw=1.2)
ax.plot(bin_centers[sel], s_trace / s_trace.max(), "r", alpha=0.7, label="speed (norm)", lw=1.2)
for m in move_on[(move_on > 500) & (move_on < 512)]:
    ax.axvline(m, color="gray", ls=":", lw=0.8)
ax.set_xlabel("time (s)"); ax.set_ylabel("normalized")
ax.set_title(f"unit {u}: rate and lag-corrected speed")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig08_velocity_example.png", dpi=150)
plt.close(fig)
print("saved figures/fig08_velocity_example.png")

# %% [markdown]
# ### Speed tuning across the population, and trial-based vs velocity-based preferred directions
#
# If the same directional signal shows up in both analyses, the preferred
# direction from the trial-averaged cosine fit should match the preferred
# velocity direction from the lagged regression. I compare them for units that
# are direction-tuned in the trial analysis, using a circular–circular
# correlation (Fisher & Lee).

# %%
# per-unit speed correlation at best lag
speed_r = np.zeros(n_units)
pref_vel_all = np.zeros(n_units)
for j in range(n_units):
    lb = lag_bins[best_li[j]]
    Vs = np.roll(V, -lb, axis=0)
    spd = np.hypot(Vs[:, 0], Vs[:, 1])
    speed_r[j] = np.corrcoef(rate_s[j][pm_mask], spd[pm_mask])[0, 1]
    _, b = fit_r2_all(rate_s[[j]], Vs[:, 0], Vs[:, 1], pm_mask)
    pref_vel_all[j] = np.arctan2(b[2, 0], b[1, 0])

def circ_corr(a, b):
    """Fisher-Lee circular-circular correlation."""
    a = a - np.angle(np.exp(1j * a).mean())
    b = b - np.angle(np.exp(1j * b).mean())
    num = (np.sin(a[:, None] - a[None, :]) * np.sin(b[:, None] - b[None, :])).sum()
    den = np.sqrt((np.sin(a[:, None] - a[None, :]) ** 2).sum()
                  * (np.sin(b[:, None] - b[None, :]) ** 2).sum())
    return num / den

both = sig_dir & (best_r2_vel > np.nanmedian(best_r2_vel))
rc = circ_corr(pref_dir[both], pref_vel_all[both])
dphi = np.degrees(np.angle(np.exp(1j * (pref_dir[both] - pref_vel_all[both]))))
print(f"units in comparison: {both.sum()}, circular correlation = {rc:.2f}")
print(f"median |PD difference|: {np.median(np.abs(dphi)):.0f} deg")
fin = np.isfinite(speed_r)
print(f"speed tuning: {(speed_r[fin] > 0).mean():.0%} of units positively speed-modulated, "
      f"median r = {np.nanmedian(speed_r):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.hist(speed_r[np.isfinite(speed_r)], bins=40, color="tab:blue")
ax.axvline(0, color="k", ls=":")
ax.axvline(np.nanmedian(speed_r), color="r", ls="--", label=f"median r = {np.nanmedian(speed_r):.2f}")
ax.set_xlabel("Pearson r (rate vs speed)"); ax.set_ylabel("units")
ax.set_title("Speed tuning across units")
ax.legend(frameon=False)

ax = axes[1]
ax.scatter(np.degrees(pref_dir[both]), np.degrees(pref_vel_all[both]), s=14,
           alpha=0.7, color="tab:blue")
ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
ax.set_xlabel("trial-based preferred direction (deg)")
ax.set_ylabel("velocity-based preferred direction (deg)")
ax.set_title(f"Preferred directions agree (circular r = {rc:.2f})")

ax = axes[2]
ax.hist(dphi, bins=np.arange(-180, 181, 20), color="tab:blue")
ax.axvline(0, color="k", ls=":")
ax.set_xlabel("PD difference, trial minus velocity (deg)")
ax.set_ylabel("units")
ax.set_title("PD difference distribution")
fig.tight_layout()
fig.savefig("figures/fig09_speed_and_pd_comparison.png", dpi=150)
plt.close(fig)
print("saved figures/fig09_speed_and_pd_comparison.png")

# %% [markdown]
# ## 5. Poisson GLM encoding of velocity (nemos)
#
# Finally, a proper encoding model. For each unit I fit a Poisson GLM (nemos)
# on raw 25 ms spike counts with the velocity shifted by the unit's best lag.
# Three nested feature sets are compared:
#
# - **speed**: a 4-knot B-spline expansion of hand speed (direction-agnostic),
# - **velocity**: the two velocity components $v_x, v_y$ (linear velocity /
#   cosine-direction encoding with speed scaling),
# - **full**: velocity components plus the speed splines.
#
# Models are trained on the first 70% of perimovement bins and scored on the
# held-out remainder with Cohen's pseudo-$R^2$ (log-likelihood gain over the
# null model, normalized by the gap to a saturated model).

# %%
import nemos as nmo

speed_basis = nmo.basis.BSplineEval(n_basis_funcs=4, bounds=(0.0, 1.5))

pm_idx = np.where(pm_mask)[0]
n_train = int(0.7 * len(pm_idx))
train_idx, test_idx = pm_idx[:n_train], pm_idx[n_train:]

def build_features(j):
    lb = lag_bins[best_li[j]]
    Vs = np.roll(V, -lb, axis=0) / 1000.0  # m/s scale keeps features O(1)
    spd = np.hypot(Vs[:, 0], Vs[:, 1])
    S = speed_basis.compute_features(spd)
    return Vs, S

scores = {"speed": np.zeros(n_units), "velocity": np.zeros(n_units), "full": np.zeros(n_units)}
glm_full = {}
for j in tqdm(range(n_units), desc="GLM fits"):
    Vs, S = build_features(j)
    y = counts[j]
    feats = {
        "speed": S,
        "velocity": Vs,
        "full": np.concatenate([Vs, S], axis=1),
    }
    for name, X in feats.items():
        glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.01)
        glm.fit(X[train_idx], y[train_idx])
        scores[name][j] = glm.score(X[test_idx], y[test_idx], score_type="pseudo-r2-Cohen")
        if name == "full":
            glm_full[j] = glm

print(f"median test pseudo-R2  speed: {np.median(scores['speed']):.3f}, "
      f"velocity: {np.median(scores['velocity']):.3f}, full: {np.median(scores['full']):.3f}")

# %%
# example predictions on a continuous held-out stretch (display uses all bins,
# scores were computed on held-out perimovement bins only)
example_js = np.argsort(scores["full"])[::-1][:3]
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
t_test_start = bin_centers[test_idx[0]]
lo, hi = t_test_start + 10, t_test_start + 22
m = (bin_centers >= lo) & (bin_centers <= hi)
for row, j in enumerate(example_js):
    Vs, S = build_features(j)
    X = np.concatenate([Vs, S], axis=1)
    pred_rate = glm_full[j].predict(X[m]) / BIN
    obs_rate = rate_s[j][m]
    ax = axes[row]
    ax.plot(bin_centers[m], obs_rate, "k", lw=1.2, label="observed (smoothed)")
    ax.plot(bin_centers[m], pred_rate, "tab:red", lw=1.2, label="GLM prediction")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"unit {unit_ids[j]}  (test pseudo-$R^2$ = {scores['full'][j]:.2f})", fontsize=10)
    if row == 0:
        ax.legend(frameon=False, loc="upper left")
axes[-1].set_xlabel("time (s)")
fig.suptitle("Held-out encoding performance of the full velocity GLM")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/fig10_glm_examples.png", dpi=150)
plt.close(fig)
print("saved figures/fig10_glm_examples.png")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
data = [scores["speed"], scores["velocity"], scores["full"]]
bp = ax.boxplot(data, tick_labels=["speed only", "velocity ($v_x,v_y$)", "full"],
                showfliers=False, widths=0.5)
for k, d in enumerate(data):
    ax.scatter(np.full_like(d, k + 1) + np.random.default_rng(1).normal(0, 0.04, len(d)),
               d, s=6, alpha=0.35, color="k")
ax.set_ylabel("test pseudo-$R^2$")
ax.set_title("Encoding model comparison (182 units)")

ax = axes[1]
ax.scatter(scores["velocity"], scores["full"], s=10, alpha=0.6, color="tab:blue")
lim = max(scores["velocity"].max(), scores["full"].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8)
ax.set_xlabel("velocity model pseudo-$R^2$")
ax.set_ylabel("full model pseudo-$R^2$")
ax.set_title("Does nonlinear speed add beyond linear velocity?")
fig.tight_layout()
fig.savefig("figures/fig11_glm_population.png", dpi=150)
plt.close(fig)
print("saved figures/fig11_glm_population.png")

# %% [markdown]
# ## 6. Summary
#
# - **Reach-direction tuning.** Most units (see printed counts) are significantly
#   cosine-tuned to the direction of the upcoming reach in the perimovement
#   epoch, with preferred directions tiling the workspace. A Georgopoulos
#   population vector built from the tuned units decodes single-trial reach
#   direction far above chance.
# - **Velocity tuning.** Instantaneous rates are linearly tuned to the hand
#   velocity vector; the population lag distribution peaks with spikes leading
#   the hand by ~75–150 ms, matching the classic motor-cortex lead time. Rates
#   also grow with hand speed for the large majority of units.
# - **Consistency.** Preferred directions from the trial-averaged analysis and
#   from the continuous velocity regression agree (high circular correlation),
#   indicating both analyses tap the same underlying directional signal.
# - **Encoding model.** Poisson GLMs with velocity features predict held-out
#   spiking above chance for most units, and adding nonlinear speed features on
#   top of linear velocity improves the fit for a substantial fraction of the
#   population.
