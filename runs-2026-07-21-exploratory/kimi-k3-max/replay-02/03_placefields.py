"""Step 3: run bouts on the linear track + place-field template with SI shuffle.

Run bouts: contiguous valid stretches of the linearized position, merged over
<0.3 s gaps, min duration 1 s, span >0.3 m, median |dl/dt| > 0.15 m/s.

Place cells: excitatory units with mean rate >= 0.1 Hz and peak >= 1 Hz during
run bouts, and Skaggs spatial information significant vs 500 circular time
shifts on the concatenated run-bout time axis (shifting over wall-clock piles
shuffled spikes at track ends and inflates the null).

Saves cache/placefields.npz and figures/fig02_place_fields.png,
figures/fig01_behavior.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

RNG = np.random.default_rng(42)
TRACK_LEN = 1.6
N_BINS = 50
DT_POS = 0.0256  # 39.0625 Hz

b = np.load("cache/behavior.npz")
pos_t, lin = b["pos_t"], b["lin"]
pos_xy = b["pos_xy"]
s = np.load("cache/spikes.npz")
unit_ids, cell_type = s["unit_ids"], s["cell_type"]
spikes = {k: s[f"spikes_{k}"] for k in unit_ids}

# --- run bouts ---------------------------------------------------------------
valid = ~np.isnan(lin)
# contiguous valid stretches
d = np.diff(valid.astype(int))
starts = np.where(d == 1)[0] + 1
ends = np.where(d == -1)[0]
if valid[0]:
    starts = np.r_[0, starts]
if valid[-1]:
    ends = np.r_[ends, len(valid) - 1]
# merge over gaps < 0.3 s
gap_t = pos_t[starts[1:]] - pos_t[ends[:-1]]
merge = gap_t < 0.3
bouts = []
i = 0
while i < len(starts):
    j = i
    while j < len(starts) - 1 and merge[j]:
        j += 1
    bouts.append((pos_t[starts[i]], pos_t[ends[j]]))
    i = j + 1
# quality criteria
kept = []
for t0, t1 in bouts:
    dur = t1 - t0
    if dur < 1.0:
        continue
    m = (pos_t >= t0) & (pos_t <= t1) & valid
    x = lin[m]
    if len(x) < 5 or (x.max() - x.min()) < 0.3:
        continue
    speed = np.abs(np.diff(x)) / DT_POS
    if np.median(speed) < 0.15:
        continue
    kept.append((t0, t1))
print(f"run bouts: {len(kept)}, total {sum(e - s for s, e in kept):.1f} s")

# --- concatenated bout time axis ----------------------------------------------
bout_starts = np.array([k[0] for k in kept])
bout_ends = np.array([k[1] for k in kept])
bout_durs = bout_ends - bout_starts
tau_offset = np.r_[0, np.cumsum(bout_durs)[:-1]]
T_total = bout_durs.sum()

# position samples inside bouts (mask NaN inside merged gaps!)
in_bout = np.zeros(len(pos_t), bool)
for t0, t1 in kept:
    in_bout |= (pos_t >= t0) & (pos_t <= t1)
samp = in_bout & valid
x_samp = lin[samp]
t_samp = pos_t[samp]
bout_idx_samp = np.searchsorted(bout_starts, t_samp, side="right") - 1
tau_samp = tau_offset[bout_idx_samp] + (t_samp - bout_starts[bout_idx_samp])

edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
bin_centers = (edges[:-1] + edges[1:]) / 2
occ_samp = np.bincount(np.clip((x_samp / TRACK_LEN * N_BINS).astype(int), 0, N_BINS - 1),
                       minlength=N_BINS)
occ_sec = occ_samp * DT_POS

# --- per-cell rate maps --------------------------------------------------------
exc_ids = unit_ids[cell_type == "excitatory"]

def spike_positions(st):
    """Positions of spikes inside run bouts, via nearest position sample."""
    if len(st) == 0:
        return np.array([])
    bidx = np.searchsorted(bout_starts, st, side="right") - 1
    ok = (bidx >= 0) & (bidx < len(kept))
    ok[ok] &= st[ok] <= bout_ends[bidx[ok]]
    st = st[ok]
    if len(st) == 0:
        return np.array([])
    idx = np.searchsorted(pos_t, st)
    idx = np.clip(idx, 1, len(pos_t) - 1)
    closer_right = (pos_t[idx] - st) < (st - pos_t[idx - 1])
    idx = np.where(closer_right, idx, idx - 1)
    x = lin[idx]
    return x[~np.isnan(x)]

def rate_map_from_x(x):
    counts = np.bincount(np.clip((x / TRACK_LEN * N_BINS).astype(int), 0, N_BINS - 1),
                         minlength=N_BINS)
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = counts / occ_sec
    rm[occ_sec < 0.1] = 0.0  # unvisited bins
    return rm, counts

def skaggs_si(rm):
    p = occ_sec / occ_sec.sum()
    R = (rm * p).sum()
    if R <= 0:
        return 0.0
    m = (rm > 0) & (p > 0)
    return float((p[m] * rm[m] / R * np.log2(rm[m] / R)).sum())

rate_maps, spike_counts, sis = {}, {}, {}
spike_x = {}
for k in exc_ids:
    x = spike_positions(spikes[k])
    spike_x[k] = x
    rm, counts = rate_map_from_x(x)
    rate_maps[k] = rm
    spike_counts[k] = counts
    sis[k] = skaggs_si(rm)

# --- SI shuffle: circular time shifts on the concatenated bout axis ------------
NSHUF = 500
# per-cell spike taus
spike_tau = {}
for k in exc_ids:
    st = spikes[k]
    bidx = np.searchsorted(bout_starts, st, side="right") - 1
    ok = (bidx >= 0) & (bidx < len(kept))
    ok[ok] &= st[ok] <= bout_ends[bidx[ok]]
    spike_tau[k] = tau_offset[bidx[ok]] + (st[ok] - bout_starts[bidx[ok]])

si_null = {k: np.zeros(NSHUF) for k in exc_ids}
shifts = RNG.uniform(0, T_total, NSHUF)
for k in tqdm(list(exc_ids), desc="SI shuffle"):
    tau = spike_tau[k]
    if len(tau) < 10:
        continue
    for i, sh in enumerate(shifts):
        tau_sh = (tau + sh) % T_total
        x_sh = np.interp(tau_sh, tau_samp, x_samp)
        rm_sh, _ = rate_map_from_x(x_sh)
        si_null[k][i] = skaggs_si(rm_sh)

si_val = np.array([sis[k] for k in exc_ids])
si_p = np.array([(np.sum(si_null[k] >= sis[k]) + 1) / (NSHUF + 1) for k in exc_ids])
mean_rate = np.array([spike_counts[k].sum() / occ_sec.sum() for k in exc_ids])
peak_rate = np.array([rate_maps[k].max() for k in exc_ids])

is_place = (si_p < 0.05) & (mean_rate >= 0.1) & (peak_rate >= 1.0)
place_ids = exc_ids[is_place]
print(f"place cells: {is_place.sum()}/{len(exc_ids)} excitatory "
      f"(SI p<0.05, mean>=0.1 Hz, peak>=1 Hz)")
print(f"median SI: place {np.median(si_val[is_place]):.3f}, "
      f"non-place {np.median(si_val[~is_place]):.3f} bits/spike")

np.savez(
    "cache/placefields.npz",
    place_ids=place_ids,
    exc_ids=exc_ids,
    rate_maps=np.array([rate_maps[k] for k in place_ids]),
    all_rate_maps=np.array([rate_maps[k] for k in exc_ids]),
    si=si_val, si_p=si_p, is_place=is_place,
    bin_centers=bin_centers, occ_sec=occ_sec,
    bout_starts=bout_starts, bout_ends=bout_ends,
)

# --- figures -------------------------------------------------------------------
# fig01: behavior / raw data check
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
ax = axes[0]
ax.plot(pos_t[::10], pos_xy[::10, 0], lw=0.3)
ax.set_ylabel("x (m)")
ax.set_title("2D position (x) over the maze epoch")
ax = axes[1]
ax.plot(pos_t[::10], lin[::10], lw=0.3)
for t0, t1 in kept:
    ax.axvspan(t0, t1, color="orange", alpha=0.15, lw=0)
ax.set_ylabel("linearized (m)")
ax.set_title("Linearized position with run bouts highlighted")
ax = axes[2]
example = place_ids[np.argsort([rate_maps[k].max() for k in place_ids])[-1]]
st = spikes[example]
m = (st > pos_t[0]) & (st < pos_t[0] + 120)
ax.eventplot([st[m] - pos_t[0]], lineoffsets=0, color="k")
ax.set_xlim(0, 120)
ax.set_yticks([])
ax.set_xlabel("time in maze epoch (s)")
ax.set_title(f"Example place cell {example} spikes, first 120 s of maze")
fig.tight_layout()
fig.savefig("figures/fig01_behavior.png", dpi=150)
plt.close(fig)

# fig02: place fields
order = np.argsort(np.argmax([rate_maps[k] for k in place_ids], axis=1))
rm_sorted = np.array([rate_maps[k] for k in place_ids])[order]
rm_norm = rm_sorted / rm_sorted.max(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
ax = axes[0]
im = ax.imshow(rm_norm, aspect="auto", cmap="viridis",
               extent=[0, TRACK_LEN, len(place_ids), 0])
ax.set_xlabel("track position (m)")
ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"Place-field template ({len(place_ids)} cells)")
fig.colorbar(im, ax=ax, label="normalized rate")
ax = axes[1]
for k in place_ids[:6]:
    ax.plot(bin_centers, rate_maps[k], lw=1)
ax.set_xlabel("track position (m)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("Example rate maps (6 cells)")
ax = axes[2]
ax.hist(si_val[is_place], bins=30, alpha=0.7, label=f"place (n={is_place.sum()})")
ax.hist(si_val[~is_place], bins=30, alpha=0.7, label=f"non-place (n={(~is_place).sum()})")
ax.set_xlabel("Skaggs SI (bits/spike)")
ax.set_ylabel("count")
ax.set_title("Spatial information")
ax.legend()
fig.tight_layout()
fig.savefig("figures/fig02_place_fields.png", dpi=150)
plt.close(fig)
print("DONE")
