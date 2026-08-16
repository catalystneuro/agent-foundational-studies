"""Compute wake HD tuning curves; identify HD cells with MVL + random-time null."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from tqdm import tqdm
from common import SESSIONS, load_session, get_hd_angle, get_states, sorted_units, mean_vector_length

rng = np.random.default_rng(42)
name = "Mouse28-140310"
nwb = load_session(SESSIONS[name])
units = sorted_units(nwb)
hd = get_hd_angle(nwb)
states = get_states(nwb)
wake = states["Awake"]

# valid wake HD samples (tracking good)
hd_wake = hd.restrict(wake)
valid_mask = ~np.isnan(hd_wake.values)
hd_valid_t = hd_wake.t[valid_mask]
hd_valid_v = hd_wake.values[valid_mask]
print(f"valid wake HD samples: {len(hd_valid_t)}")

# ---- tuning curves (pynapple) ----
counts_da = nap.compute_tuning_curves(
    units, hd, bins=60, range=[(0, 2 * np.pi)], epochs=wake,
    return_counts=True, feature_names=["hd"],
)
counts = counts_da.values  # units x bins spike counts
occupancy = counts_da.attrs["occupancy"] / counts_da.attrs["fs"]  # seconds per bin
rates = np.where(occupancy[None, :] > 0, counts / occupancy[None, :], np.nan)
bin_centers = counts_da.coords["hd"].values
np.savez("data/mouse28_tuning.npz", rates=rates, bin_centers=bin_centers,
         occupancy=occupancy, counts=counts)

# ---- HD cell statistics: MVL of spike angles + random-time null ----
# For each unit: spike-time HD values via value_from; MVL observed.
# Null: redraw the same number of spike times uniformly from valid wake HD
# samples (preserves wake occupancy), recompute MVL. Chunked to bound memory.
N_SHUFFLE = 1000
MVL_FLOOR = 0.3
SPIKE_CAP = 100_000

mvl_obs = np.zeros(len(units))
p_val = np.zeros(len(units))
pref_ang = np.zeros(len(units))
n_spikes = np.zeros(len(units), dtype=int)

for i, k in enumerate(tqdm(units.keys(), desc="HD stats")):
    spk = units[k].restrict(wake)
    if len(spk) == 0:
        mvl_obs[i] = 0
        p_val[i] = 1.0
        continue
    ang = spk.value_from(hd)  # HD at each spike time (39 Hz samples)
    ang = ang[~np.isnan(ang)]
    if len(ang) > SPIKE_CAP:
        ang = ang[rng.choice(len(ang), SPIKE_CAP, replace=False)]
    n_spikes[i] = len(ang)
    mvl_obs[i] = mean_vector_length(ang)
    pref_ang[i] = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2 * np.pi)
    # random-time null, chunked
    n = len(ang)
    null = np.zeros(N_SHUFFLE)
    for chunk_start in range(0, N_SHUFFLE, 100):
        m = min(100, N_SHUFFLE - chunk_start)
        idx = rng.integers(0, len(hd_valid_v), size=(m, n))
        samp = hd_valid_v[idx]
        s = np.sin(samp).mean(axis=1)
        c = np.cos(samp).mean(axis=1)
        null[chunk_start:chunk_start + m] = np.sqrt(s**2 + c**2)
    p_val[i] = (np.sum(null >= mvl_obs[i]) + 1) / (N_SHUFFLE + 1)

is_hd = (p_val < 0.05) & (mvl_obs > MVL_FLOOR)
print(f"\nHD cells: {is_hd.sum()}/{len(units)} (MVL>{MVL_FLOOR}, random-time null p<0.05)")
np.savez("data/mouse28_hd_stats.npz", mvl=mvl_obs, p=p_val, pref=pref_ang,
         is_hd=is_hd, n_spikes=n_spikes)

# ---- figures ----
fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# MVL vs p-value scatter
ax = fig.add_subplot(gs[0, 0])
ax.scatter(mvl_obs[~is_hd], -np.log10(p_val[~is_hd]), s=25, c="0.6", label="non-HD")
ax.scatter(mvl_obs[is_hd], -np.log10(p_val[is_hd]), s=25, c="crimson", label="HD cell")
ax.axhline(-np.log10(0.05), ls="--", c="k", lw=0.8)
ax.axvline(MVL_FLOOR, ls="--", c="k", lw=0.8)
ax.set_xlabel("mean vector length")
ax.set_ylabel("-log10(p) vs random-time null")
ax.legend(fontsize=8)
ax.set_title("HD cell selection")

# occupancy-normalized tuning curves of HD cells, sorted by preferred angle
ax = fig.add_subplot(gs[0, 1:])
order = np.argsort(pref_ang[is_hd])
hd_idx = np.where(is_hd)[0][order]
tc = rates[hd_idx]
tc_norm = tc / np.nanmax(tc, axis=1, keepdims=True)
im = ax.imshow(tc_norm, aspect="auto", cmap="viridis",
               extent=[0, 360, len(hd_idx) - 0.5, -0.5])
ax.set_xlabel("head direction (deg)")
ax.set_ylabel("HD cell (sorted by pref. dir.)")
ax.set_title("Normalized wake tuning curves")
fig.colorbar(im, ax=ax, label="norm. rate", shrink=0.8)

# polar tuning curves for all HD cells
n_hd = is_hd.sum()
ncols = 5
nrows = int(np.ceil(n_hd / ncols))
for j, u in enumerate(hd_idx):
    axp = fig.add_subplot(gs[1, j % 3], polar=True) if False else None
# separate polar grid figure
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(13, 2.2 * nrows),
                           subplot_kw=dict(polar=True))
axes2 = np.atleast_1d(axes2).ravel()
for j, u in enumerate(hd_idx):
    axp = axes2[j]
    th = np.concatenate([bin_centers, bin_centers[:1]])
    r = np.concatenate([rates[u], rates[u][:1]])
    axp.plot(th, r, color="crimson")
    axp.fill(th, r, color="crimson", alpha=0.25)
    axp.set_title(f"u{u} MVL={mvl_obs[u]:.2f}", fontsize=8)
    axp.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    axp.set_xticklabels(["0", "90", "180", "270"], fontsize=7)
    axp.set_yticks([])
for j in range(len(hd_idx), len(axes2)):
    axes2[j].axis("off")
fig2.suptitle(f"{name}: wake HD tuning curves of {n_hd} HD cells", fontsize=13)
fig2.tight_layout()
fig2.savefig("figures/02b_tuning_curves_polar.png", dpi=150, bbox_inches="tight")

# preferred direction distribution
ax = fig.add_subplot(gs[1, :])
ax.hist(np.degrees(pref_ang[is_hd]), bins=18, range=(0, 360), color="crimson", alpha=0.7)
ax.set_xlabel("preferred direction (deg)")
ax.set_ylabel("count")
ax.set_title("Preferred-direction distribution of HD cells")

fig.suptitle(f"{name}: HD cell identification", fontsize=13)
fig.savefig("figures/02a_hd_cell_selection.png", dpi=150, bbox_inches="tight")
print("saved figures/02a_hd_cell_selection.png, figures/02b_tuning_curves_polar.png")
