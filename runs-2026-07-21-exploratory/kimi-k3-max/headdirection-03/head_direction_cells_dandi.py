# %% [markdown]
# # Head Direction Cells in the Mouse Anterodorsal Thalamus and Postsubiculum
#
# **Dataset**: DANDI Archive dandiset
# [000056](https://dandiarchive.org/dandiset/000056): Peyrache, Lacroix,
# Petersen & Buzsáki (2015), *"Internally organized mechanisms of the head
# direction sense"*, Nature Neuroscience 18:569-575.
#
# **Phenomenon**. Head direction (HD) cells fire when the animal's head points
# in a particular direction in the horizontal plane, independent of where the
# animal is. They are the cellular basis of the internal compass. The classic
# HD circuit runs from the dorsal tegmental nucleus through the anterodorsal
# thalamus (ADn) to the postsubiculum (PoS), the two structures recorded here.
#
# **What this notebook shows**
# 1. How the HD signal is recovered from two head-mounted LEDs.
# 2. HD tuning curves during wake exploration and a statistical identification
#    of HD cells (mean vector length against an occupancy-corrected
#    random-time null, plus an effect-size floor).
# 3. The signature result of Peyrache et al. (2015): the pairwise correlation
#    structure of the HD ensemble during wake is preserved during REM and NREM
#    sleep, evidence that the HD representation is internally organized.
# 4. A population summary across nine sessions from five mice.
#
# The notebook streams NWB files from DANDI with `remfile` (no full
# downloads) and uses pynapple for data handling and tuning curves.

# %% [markdown]
# ## Setup

# %%
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from tqdm import tqdm

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
CACHE_DIR = "/tmp/remfile_cache_hd"
rng = np.random.default_rng(0)

# %% [markdown]
# ## Streaming access and core functions
#
# Each session is opened through the DANDI asset API with a shared disk
# cache, so only the spike times and tracking series are fetched.

# %%
def load_session(asset_id, cache_dir=CACHE_DIR):
    """Stream an NWB session from DANDI with a shared disk cache."""
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), io


def compute_head_direction(nwb):
    """Head direction from the angle of the red-minus-blue LED vector.

    Tracking failures are marked with -1 sentinel values; those samples
    become NaN and drop out of occupancy and spike-angle histograms.
    """
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    valid = (red.values[:, 0] > 0) & (red.values[:, 1] > 0) & \
            (blue.values[:, 0] > 0) & (blue.values[:, 1] > 0)
    angle = np.arctan2(red.values[:, 1] - blue.values[:, 1],
                       red.values[:, 0] - blue.values[:, 0]) % (2 * np.pi)
    angle[~valid] = np.nan
    return nap.Tsd(t=red.t, d=angle)


def get_epochs(nwb):
    """(wake, REM, NREM) IntervalSets from the scored states table."""
    states = nwb["states"]
    labels = np.asarray(states["label"])
    return (states[labels == "Awake"], states[labels == "REM"],
            states[labels == "Non-REM"])


def valid_hd_samples(hd, epochs):
    hd_e = hd.restrict(epochs)
    mask = ~np.isnan(hd_e.values)
    return hd_e.t[mask], hd_e.values[mask]


def mean_vector_length(angles):
    angles = np.asarray(angles)
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan, np.nan
    z = np.exp(1j * angles).mean()
    return np.abs(z), np.angle(z) % (2 * np.pi)


def random_time_null(n_spikes, hd_d_valid, n_shuf=500, rng=None, block=50):
    """Null MVL distribution from random wake-time resampling.

    Draws n_spikes angles with replacement from the empirical wake HD
    distribution. The null therefore preserves the non-uniform HD
    occupancy, so significance reflects tuning beyond what occupancy
    alone produces. Blocked to bound memory for high-count units.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.empty(n_shuf)
    for i in range(0, n_shuf, block):
        j = min(i + block, n_shuf)
        draws = rng.choice(hd_d_valid, size=(j - i, n_spikes))
        out[i:j] = np.abs(np.exp(1j * draws).mean(axis=1))
    return out


def state_correlations(units_f, hd_cell_keys, epochs, bin_size=0.1,
                       min_rate=0.05):
    """Pairwise Pearson correlation matrix of HD cells within a state."""
    n = len(hd_cell_keys)
    if n < 2 or epochs.tot_length() < 30:
        return np.full((n, n), np.nan)
    counts = units_f[list(hd_cell_keys)].count(bin_size, epochs)
    rates = np.asarray(counts.values, dtype=float) / bin_size
    keep = rates.mean(axis=0) >= min_rate
    C = np.full((n, n), np.nan)
    if keep.sum() >= 2:
        C[np.ix_(keep, keep)] = np.corrcoef(rates[:, keep].T)
    return C


def corr_of_corr(Cw, Cs):
    """Correlation between wake and sleep pairwise-correlation vectors."""
    iu = np.triu_indices(Cw.shape[0], k=1)
    m = ~np.isnan(Cw[iu]) & ~np.isnan(Cs[iu])
    if m.sum() < 6:
        return np.nan, m.sum()
    return np.corrcoef(Cw[iu][m], Cs[iu][m])[0, 1], m.sum()


def shuffle_corr_of_corr(Cw, Cs, n_shuf=200, seed=0):
    """Cell-label permutation null for the correlation of correlations."""
    r = np.random.default_rng(seed)
    n = Cw.shape[0]
    iu = np.triu_indices(n, k=1)
    out = []
    for _ in range(n_shuf):
        p = r.permutation(n)
        Css = Cs[np.ix_(p, p)]
        m = ~np.isnan(Cw[iu]) & ~np.isnan(Css[iu])
        if m.sum() >= 6:
            out.append(np.corrcoef(Cw[iu][m], Css[iu][m])[0, 1])
    return np.array(out)

# %% [markdown]
# ## Single-session analysis: Mouse17-130128
#
# This session contains 39 sorted units, dual-LED tracking at 39 Hz, and
# scored wake / NREM / REM epochs over about 5.2 hours.

# %%
ASSET_PROTOTYPE = "4cc64fe0-7b1e-404c-8b86-fb5659292830"
SESSION = "Mouse17-130128"

nwb, io = load_session(ASSET_PROTOTYPE)
print(nwb)
hd = compute_head_direction(nwb)
wake, rem, nrem = get_epochs(nwb)
print(f"wake {wake.tot_length():.0f}s | REM {rem.tot_length():.0f}s | "
      f"NREM {nrem.tot_length():.0f}s")

# %% [markdown]
# ### The head-direction signal
#
# Head direction is the angle of the vector from the blue to the red LED.
# The animal covers the whole arena and the whole circle of directions
# during wake, with somewhat non-uniform occupancy (occupancy resultant
# ~0.18), which the statistical null below accounts for.

# %%
red = nwb["SubjectPosition/RedLED"]
blue = nwb["SubjectPosition/BlueLED"]
valid = (red.values[:, 0] > 0) & (blue.values[:, 0] > 0)

fig = plt.figure(figsize=(13, 4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.6, 1], wspace=0.3)

ax = fig.add_subplot(gs[0])
ax.plot(red.values[valid, 0], red.values[valid, 1], ".", ms=0.3,
        color="tab:red", alpha=0.5, label="red LED", rasterized=True)
ax.plot(blue.values[valid, 0], blue.values[valid, 1], ".", ms=0.3,
        color="tab:blue", alpha=0.5, label="blue LED", rasterized=True)
ax.set_aspect("equal")
ax.set_xlabel("x (camera pixels)")
ax.set_ylabel("y (camera pixels)")
ax.set_title("Head-tracker LED positions")
ax.legend(markerscale=10, loc="upper right", fontsize=8)

ax = fig.add_subplot(gs[1])
t0 = wake["start"][0]
snippet = hd.get(t0, t0 + 120)
ax.plot(snippet.t - t0, snippet.d, ".", ms=1.5, color="0.2", rasterized=True)
ax.set_xlabel("time in wake epoch (s)")
ax.set_ylabel("head direction (rad)")
ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "$\\pi$", "$2\\pi$"])
ax.set_ylim(-0.15, 2 * np.pi + 0.15)
ax.set_title("Head direction over 120 s of exploration")

ax = fig.add_subplot(gs[2], projection="polar")
hd_w = hd.restrict(wake).values
hd_w = hd_w[~np.isnan(hd_w)]
occ, edges = np.histogram(hd_w, bins=60, range=(0, 2 * np.pi))
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers, occ / 39.0625, width=2 * np.pi / 60, color="0.4")
ax.set_title("HD occupancy, wake (s)", pad=22)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"])
ax.set_rlabel_position(90)
ax.tick_params(axis="y", labelsize=7)

fig.savefig(f"{FIGDIR}/fig1_raw_head_direction.png", dpi=200)
plt.close(fig)

# %% [markdown]
# ### HD tuning curves and statistical identification of HD cells
#
# For each unit we compute the wake tuning curve (spike count per HD bin
# divided by occupancy) and the mean vector length (MVL) of the HD angles
# at spike times. Significance uses a random-time null: spike counts are
# preserved but spike times are redrawn uniformly from wake, so the null
# inherits the exact HD occupancy. A unit is an HD cell when its MVL
# exceeds the 95th percentile of the null (p < 0.05) and clears an
# effect-size floor of MVL > 0.3, a common criterion in the HD literature.

# %%
N_SHUF = 500
MVL_FLOOR = 0.3
MIN_SPIKES = 100

units = nwb["units"]
units_f = units[units.metadata["rate"].values > 0.1]
keys = list(units_f.keys())
hd_t_w, hd_d_w = valid_hd_samples(hd, wake)

mvl = np.full(len(keys), np.nan)
pref = np.full(len(keys), np.nan)
null95 = np.full(len(keys), np.nan)
pval = np.full(len(keys), np.nan)
n_spikes = np.zeros(len(keys), dtype=int)

for i, u in enumerate(tqdm(keys, desc="units")):
    sp = units_f[u].restrict(wake)
    n_spikes[i] = len(sp)
    if n_spikes[i] < MIN_SPIKES:
        continue
    ang = np.interp(sp.t, hd_t_w, hd_d_w)
    mvl[i], pref[i] = mean_vector_length(ang)
    null = random_time_null(n_spikes[i], hd_d_w, n_shuf=N_SHUF, rng=rng)
    null95[i] = np.percentile(null, 95)
    pval[i] = (np.sum(null >= mvl[i]) + 1) / (N_SHUF + 1)

is_hd = (pval < 0.05) & (mvl > MVL_FLOOR)
print(f"{is_hd.sum()}/{np.sum(~np.isnan(mvl))} units are HD cells "
      f"(p<0.05 and MVL>{MVL_FLOOR})")

# wake tuning curves (counts / occupancy)
counts = nap.compute_tuning_curves(units_f, hd, bins=60, range=(0, 2 * np.pi),
                                   epochs=wake, feature_names=["hd"],
                                   return_counts=True)
occ_s = np.asarray(counts.attrs["occupancy"], dtype=float) / counts.attrs["fs"]
with np.errstate(invalid="ignore", divide="ignore"):
    tc_wake = np.where(occ_s[None, :] > 0, counts.values / occ_s[None, :],
                       np.nan)
theta = np.asarray(counts.coords["hd"])
theta_c = np.concatenate([theta, [theta[0] + 2 * np.pi]])

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

ax = axes[0]
ok = ~np.isnan(mvl)
ax.scatter(mvl[ok & ~is_hd], null95[ok & ~is_hd], c="0.55", s=25,
           label="not HD")
ax.scatter(mvl[ok & is_hd], null95[ok & is_hd], c="crimson", s=30,
           label="HD cell")
lim = [0, max(np.nanmax(mvl), np.nanmax(null95)) * 1.08]
ax.plot(lim, lim, "k--", lw=1)
ax.axvline(MVL_FLOOR, color="0.4", ls=":", lw=1)
ax.text(MVL_FLOOR + 0.005, lim[1] * 0.72, f"MVL = {MVL_FLOOR} floor",
        rotation=90, fontsize=8, color="0.35")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("observed mean vector length")
ax.set_ylabel("95th pct of random-time null")
ax.set_title(f"HD identification ({is_hd.sum()}/{ok.sum()} HD cells)")
ax.legend(loc="lower right", fontsize=9)

ax = axes[1]
ax.hist(mvl[ok & ~is_hd], bins=np.arange(0, 1.01, 0.05), color="0.6",
        label="not HD")
ax.hist(mvl[ok & is_hd], bins=np.arange(0, 1.01, 0.05), color="crimson",
        label="HD cell")
ax.set_xlabel("mean vector length")
ax.set_ylabel("unit count")
ax.set_title("Distribution of directional tuning strength")
ax.legend(fontsize=9)

fig.savefig(f"{FIGDIR}/fig2_hd_identification.png", dpi=200)
plt.close(fig)

# %% [markdown]
# The population is bimodal: most units are weakly modulated (MVL ~ 0.2,
# at the occupancy-imposed null), while a distinct group shows strong
# single-peaked tuning (MVL 0.3-0.8). The polar plots below show the 18
# strongest tuning curves; the red ones are the significant HD cells.

# %%
order = np.argsort(-np.nan_to_num(mvl))
n_show = 18
fig, axes = plt.subplots(3, 6, figsize=(13, 6.6),
                         subplot_kw=dict(projection="polar"))
for k in range(n_show):
    i = order[k]
    ax = axes[k // 6, k % 6]
    curve = tc_wake[i]
    curve_c = np.concatenate([curve, [curve[0]]])
    col = "crimson" if is_hd[i] else "0.5"
    ax.plot(theta_c, curve_c, color=col, lw=1.5)
    ax.fill(theta_c, curve_c, color=col, alpha=0.25)
    ax.set_title(f"unit {keys[i]}   MVL={mvl[i]:.2f}", fontsize=9, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Wake head-direction tuning curves, {SESSION} "
             "(18 strongest; red = significant HD cell)", y=1.00)
fig.subplots_adjust(hspace=0.35, wspace=0.3)
fig.savefig(f"{FIGDIR}/fig3_tuning_curves.png", dpi=200)
plt.close(fig)

# %% [markdown]
# ### The HD correlation structure is preserved during sleep
#
# During sleep the mouse keeps its head in a narrow range of directions,
# so per-cell sleep tuning curves cannot be estimated over the full
# circle. Peyrache et al. (2015) instead asked whether the *internal
# organization* of the HD ensemble persists: if cells with similar
# preferred directions are co-active during wake, are the same pairs
# co-active during sleep? We bin HD-cell spike trains (100 ms) within each
# state, correlate every pair, and correlate the wake and sleep
# correlation matrices across pairs. A cell-label permutation provides
# the null.

# %%
hd_keys = np.array(keys)[is_hd]
Cw = state_correlations(units_f, hd_keys, wake)
Cr = state_correlations(units_f, hd_keys, rem)
Cn = state_correlations(units_f, hd_keys, nrem)

r_rem, npairs_rem = corr_of_corr(Cw, Cr)
r_nrem, npairs_nrem = corr_of_corr(Cw, Cn)
null_rem = shuffle_corr_of_corr(Cw, Cr, seed=1)
null_nrem = shuffle_corr_of_corr(Cw, Cn, seed=2)
p_rem = (np.sum(null_rem >= r_rem) + 1) / (len(null_rem) + 1)
p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (len(null_nrem) + 1)
print(f"wake-REM corr-of-corrs: r={r_rem:.3f}, p={p_rem:.4f}")
print(f"wake-NREM corr-of-corrs: r={r_nrem:.3f}, p={p_nrem:.4f}")

# %%
order_hd = np.argsort(pref[is_hd])
iu = np.triu_indices(len(hd_keys), k=1)
m_rem = ~np.isnan(Cw[iu]) & ~np.isnan(Cr[iu])
m_nrem = ~np.isnan(Cw[iu]) & ~np.isnan(Cn[iu])

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.32, wspace=0.35)

vmax = 0.6
for k, (C, name) in enumerate([(Cw, "wake"), (Cr, "REM sleep"),
                               (Cn, "NREM sleep")]):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(C[np.ix_(order_hd, order_hd)], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(name, fontsize=11)
    if k == 0:
        ax.set_ylabel("HD cells (sorted by wake pref. dir.)")
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=[fig.axes[-3], fig.axes[-2], fig.axes[-1]],
             label="pairwise correlation", shrink=0.75, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(Cw[iu][m_rem], Cr[iu][m_rem], s=30, color="tab:red", alpha=0.8)
ax.plot([-0.4, 0.8], [-0.4, 0.8], "k--", lw=1)
ax.set_xlabel("wake correlation")
ax.set_ylabel("REM correlation")
ax.set_title(f"wake vs REM: r = {r_rem:.2f}")
ax.set_xlim(-0.45, 0.85)
ax.set_ylim(-0.45, 0.85)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(Cw[iu][m_nrem], Cn[iu][m_nrem], s=30, color="tab:blue", alpha=0.8)
ax.plot([-0.4, 0.8], [-0.4, 0.8], "k--", lw=1)
ax.set_xlabel("wake correlation")
ax.set_ylabel("NREM correlation")
ax.set_title(f"wake vs NREM: r = {r_nrem:.2f}")
ax.set_xlim(-0.45, 0.85)
ax.set_ylim(-0.45, 0.85)

ax = fig.add_subplot(gs[1, 2])
ax.hist(null_rem, bins=20, color="0.6", label="label-shuffled null")
ax.axvline(r_rem, color="tab:red", lw=2, label=f"observed REM r={r_rem:.2f}")
ax.axvline(r_nrem, color="tab:blue", lw=2,
           label=f"observed NREM r={r_nrem:.2f}")
ax.set_xlabel("correlation of correlations")
ax.set_ylabel("count")
ax.set_title("Observed vs cell-label shuffle")
ax.legend(fontsize=8)

fig.suptitle("HD ensemble correlation structure is preserved in sleep, "
             f"{SESSION}", y=0.98)
fig.savefig(f"{FIGDIR}/fig4_sleep_correlations.png", dpi=200)
plt.close(fig)
io.close()

# %% [markdown]
# The block structure along the diagonal (cells sorted by wake preferred
# direction) appears in all three states: pairs with similar preferred
# directions are positively correlated, pairs with opposite preferences
# are not. The wake-sleep correlation of correlations (r ~ 0.95) sits far
# outside the label-shuffle null, i.e. the ensemble's internal
# organization does not depend on the animal actually moving.

# %% [markdown]
# ## Population analysis across nine sessions
#
# The same pipeline is run on nine sessions from five mice (the two ~30 GB
# Mouse12 files are skipped for speed). For each session we record the
# number of HD cells, their preferred directions, and the wake-sleep
# correlation of correlations with its permutation null.

# %%
SESSIONS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse17-130125": "e62179fc-cd8d-4901-98aa-ccb35086f9f7",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse20-130515": "f92f2709-4469-4c1d-a883-75eb66ee898a",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse24-131217": "c6cfc0f2-dda6-4136-be2c-be4a4de5a50e",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse25-140205": "48dae2dc-35f7-4a9e-bcdb-883a7d5e2d32",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}
MIN_CELLS_CORR = 4
RESULTS_DIR = "results_multisession"
os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
summary = []
for name, asset_id in tqdm(SESSIONS.items(), desc="sessions"):
    nwb, io = load_session(asset_id)
    wake, rem, nrem = get_epochs(nwb)
    hd = compute_head_direction(nwb)

    units = nwb["units"]
    units_f = units[units.metadata["rate"].values > 0.1]
    keys = list(units_f.keys())
    hd_t_w, hd_d_w = valid_hd_samples(hd, wake)

    mvl = np.full(len(keys), np.nan)
    pref = np.full(len(keys), np.nan)
    pval = np.full(len(keys), np.nan)
    for i, u in enumerate(keys):
        sp = units_f[u].restrict(wake)
        if len(sp) < MIN_SPIKES:
            continue
        ang = np.interp(sp.t, hd_t_w, hd_d_w)
        mvl[i], pref[i] = mean_vector_length(ang)
        null = random_time_null(len(sp), hd_d_w, n_shuf=N_SHUF, rng=rng)
        pval[i] = (np.sum(null >= mvl[i]) + 1) / (N_SHUF + 1)
    is_hd = (pval < 0.05) & (mvl > MVL_FLOOR)

    hd_keys = np.array(keys)[is_hd]
    r_rem = r_nrem = np.nan
    p_rem = p_nrem = np.nan
    npairs = 0
    save = dict(session=name, keys=np.array(keys), mvl=mvl, pref=pref,
                pval=pval, is_hd=is_hd, hd_keys=hd_keys,
                r_rem=r_rem, r_nrem=r_nrem, p_rem=p_rem, p_nrem=p_nrem)
    if len(hd_keys) >= MIN_CELLS_CORR:
        Cw = state_correlations(units_f, hd_keys, wake)
        Cr = state_correlations(units_f, hd_keys, rem)
        Cn = state_correlations(units_f, hd_keys, nrem)
        r_rem, npairs = corr_of_corr(Cw, Cr)
        r_nrem, _ = corr_of_corr(Cw, Cn)
        null_rem = shuffle_corr_of_corr(Cw, Cr, seed=1)
        null_nrem = shuffle_corr_of_corr(Cw, Cn, seed=2)
        p_rem = (np.sum(null_rem >= r_rem) + 1) / (len(null_rem) + 1)
        p_nrem = (np.sum(null_nrem >= r_nrem) + 1) / (len(null_nrem) + 1)
        iu = np.triu_indices(len(hd_keys), k=1)
        mm_rem = ~np.isnan(Cw[iu]) & ~np.isnan(Cr[iu])
        mm_nrem = ~np.isnan(Cw[iu]) & ~np.isnan(Cn[iu])
        save.update(pairs_wake_rem=Cw[iu][mm_rem], pairs_rem=Cr[iu][mm_rem],
                    pairs_wake_nrem=Cw[iu][mm_nrem],
                    pairs_nrem=Cn[iu][mm_nrem],
                    r_rem=r_rem, r_nrem=r_nrem, p_rem=p_rem, p_nrem=p_nrem)

    np.savez(os.path.join(RESULTS_DIR, f"results_{name}.npz"), **save)
    summary.append(dict(session=name, mouse=name.split("-")[0],
                        n_units=len(keys), n_hd=int(is_hd.sum()),
                        wake_dur=wake.tot_length(),
                        rem_dur=rem.tot_length(),
                        nrem_dur=nrem.tot_length(),
                        r_rem=r_rem, r_nrem=r_nrem,
                        p_rem=p_rem, p_nrem=p_nrem, npairs=npairs))
    print(f"{name}: {is_hd.sum()}/{len(keys)} HD cells | "
          f"r_REM={r_rem:.2f} (p={p_rem:.3f}) | "
          f"r_NREM={r_nrem:.2f} (p={p_nrem:.3f})")
    io.close()

df = pd.DataFrame(summary)
df.to_csv(os.path.join(RESULTS_DIR, "session_summary.csv"), index=False)
print(f"\ntotal: {df.n_hd.sum()} HD cells out of {df.n_units.sum()} units")

# %% [markdown]
# ### Population figure

# %%
all_mvl_hd = []
all_mvl_not = []
all_pref_hd = []
for name in SESSIONS:
    d = np.load(os.path.join(RESULTS_DIR, f"results_{name}.npz"))
    all_mvl_hd.append(d["mvl"][d["is_hd"]])
    all_mvl_not.append(d["mvl"][~d["is_hd"] & ~np.isnan(d["mvl"])])
    all_pref_hd.append(d["pref"][d["is_hd"]])
all_mvl_hd = np.concatenate(all_mvl_hd)
all_mvl_not = np.concatenate(all_mvl_not)
all_pref_hd = np.concatenate(all_pref_hd)

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(df))
ax.bar(x, df["n_units"], color="0.75", label="all units")
ax.bar(x, df["n_hd"], color="crimson", label="HD cells")
ax.set_xticks(x)
ax.set_xticklabels(df["session"], fontsize=7, rotation=45,
                   ha="right")
ax.set_ylabel("unit count")
ax.set_title("HD cells per session")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 1])
bins = np.arange(0, 1.01, 0.05)
ax.hist(all_mvl_not, bins=bins, color="0.6", label=f"not HD (n={len(all_mvl_not)})")
ax.hist(all_mvl_hd, bins=bins, color="crimson",
        label=f"HD (n={len(all_mvl_hd)})")
ax.set_xlabel("mean vector length")
ax.set_ylabel("unit count")
ax.set_title("Tuning strength, pooled sessions")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2], projection="polar")
occ_p, edges_p = np.histogram(all_pref_hd, bins=24, range=(0, 2 * np.pi))
centers_p = (edges_p[:-1] + edges_p[1:]) / 2
ax.bar(centers_p, occ_p, width=2 * np.pi / 24, color="crimson", alpha=0.8)
ax.set_title("Preferred directions, all HD cells", pad=18)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"], fontsize=8)
ax.set_rlabel_position(90)
ax.tick_params(axis="y", labelsize=7)
# Rayleigh test for uniformity of preferred directions
z = len(all_pref_hd) * np.abs(np.exp(1j * all_pref_hd).mean()) ** 2
p_unif = np.exp(-z)  # large-n approximation
ax.text(0.5, -0.12, f"Rayleigh z={z:.2f}, p={p_unif:.2f}",
        transform=ax.transAxes, ha="center", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
valid_r = df.dropna(subset=["r_rem"])
xx = np.arange(len(valid_r))
w = 0.38
ax.bar(xx - w / 2, valid_r["r_rem"], width=w, color="tab:red",
       label="wake-REM")
ax.bar(xx + w / 2, valid_r["r_nrem"], width=w, color="tab:blue",
       label="wake-NREM")
ax.set_xticks(xx)
ax.set_xticklabels(valid_r["session"], fontsize=7, rotation=45,
                   ha="right")
ax.set_ylabel("correlation of correlations")
ax.set_ylim(0, 1.18)
ax.set_title("HD structure preserved in sleep")
ax.legend(fontsize=8, loc="upper left")

ax = fig.add_subplot(gs[1, 1:])
pw_rem, ps_rem, pw_nrem, ps_nrem = [], [], [], []
for name in SESSIONS:
    d = np.load(os.path.join(RESULTS_DIR, f"results_{name}.npz"))
    if "pairs_wake_rem" in d.files:
        pw_rem.append(d["pairs_wake_rem"])
        ps_rem.append(d["pairs_rem"])
        pw_nrem.append(d["pairs_wake_nrem"])
        ps_nrem.append(d["pairs_nrem"])
if pw_rem:
    pw_rem = np.concatenate(pw_rem)
    ps_rem = np.concatenate(ps_rem)
    pw_nrem = np.concatenate(pw_nrem)
    ps_nrem = np.concatenate(ps_nrem)
    ax.scatter(pw_rem, ps_rem, s=14, color="tab:red", alpha=0.5,
               label=f"wake-REM pairs (n={len(pw_rem)})")
    ax.scatter(pw_nrem, ps_nrem, s=14, color="tab:blue", alpha=0.5,
               label=f"wake-NREM pairs (n={len(pw_nrem)})")
    ax.plot([-0.5, 1], [-0.5, 1], "k--", lw=1)
    r_all_rem = np.corrcoef(pw_rem, ps_rem)[0, 1]
    r_all_nrem = np.corrcoef(pw_nrem, ps_nrem)[0, 1]
    ax.set_xlabel("wake pairwise correlation")
    ax.set_ylabel("sleep pairwise correlation")
    ax.set_title(f"All HD-cell pairs, pooled: r_REM={r_all_rem:.2f}, "
                 f"r_NREM={r_all_nrem:.2f}")
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Head-direction cells across sessions, DANDI 000056", y=0.98)
fig.savefig(f"{FIGDIR}/fig5_population_summary.png", dpi=200)
plt.close(fig)

# %% [markdown]
# ## Summary
#
# - Head direction was recovered from the two head-mounted LEDs at 39 Hz;
#   the animal sampled the full circle during wake exploration.
# - In the prototype session, 7/36 units showed strong, significant HD
#   tuning (random-time null p < 0.05 and MVL > 0.3), with the classic
#   single-peaked polar tuning curves of ADn/PoS head-direction cells.
# - The pairwise correlation structure of the HD ensemble during wake was
#   preserved during REM and NREM sleep (correlation of correlations
#   ~0.95, far outside a cell-label permutation null), reproducing the
#   central result of Peyrache et al. (2015) that the head-direction
#   representation is internally organized and does not require movement
#   or sensory input.
# - Across nine sessions from five mice, HD cells were found in every
#   session, preferred directions tiled the full circle (with a mild
#   non-uniformity, Rayleigh p = 0.01), and the wake-sleep preservation
#   was positive in all nine sessions and significant for both sleep
#   states in seven of them (permutation p < 0.05; the two exceptions
#   were the sessions with the fewest HD cells).
