# 03: place fields on the 1.6 m linear track during run bouts
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from common import open_nwb, get_epochs

N_BINS = 50
TRACK_LEN = 1.6
SMOOTH_BINS = 1.5
MIN_OCC_S = 0.1
N_SHUFFLE = 200
RNG = np.random.default_rng(42)

nwb, h5 = open_nwb()
epochs = get_epochs(nwb)
maze = epochs["MazeEpoch"]
units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
print("lin:", lin.shape, lin.t[0], lin.t[-1])

# --- run bouts: contiguous valid stretches of the linearized series ---
t, x = lin.t, np.asarray(lin.values).ravel()
valid = ~np.isnan(x)
d = np.diff(valid.astype(int))
starts = list(np.where(d == 1)[0] + 1)
ends = list(np.where(d == -1)[0] + 1)
if valid[0]:
    starts = [0] + starts
if valid[-1]:
    ends = ends + [len(valid)]
bouts = []
for s, e in zip(starts, ends):
    bouts.append([t[s], t[e - 1]])
# merge over gaps < 0.3 s
merged = []
for b in bouts:
    if merged and b[0] - merged[-1][1] < 0.3:
        merged[-1][1] = b[1]
    else:
        merged.append(list(b))
# criteria: >=1 s, span > 0.3 m, median speed > 0.15 m/s
run_bouts = []
for a, b in merged:
    if b - a < 1.0:
        continue
    m = (t >= a) & (t <= b) & valid
    if m.sum() < 5:
        continue
    xs = x[m]
    if xs.max() - xs.min() < 0.3:
        continue
    dt = np.median(np.diff(t[m]))
    spd = np.abs(np.diff(xs)) / np.diff(t[m])
    if np.median(spd) < 0.15:
        continue
    run_bouts.append((a, b))
run_iset = nap.IntervalSet(start=[a for a, b in run_bouts], end=[b for a, b in run_bouts])
total_run = np.sum([b - a for a, b in run_bouts])
print(f"{len(run_bouts)} run bouts, {total_run:.0f} s total")

# direction per bout (median diff of x)
bout_dir = []
for a, b in run_bouts:
    m = (t >= a) & (t <= b) & valid
    bout_dir.append(np.sign(np.nanmedian(np.diff(x[m]))))
bout_dir = np.array(bout_dir)
print(f"directions: {(bout_dir > 0).sum()} positive, {(bout_dir < 0).sum()} negative")

# --- excitatory units ---
cell_type = units.get_info("cell_type")
exc_keys = cell_type[cell_type == "excitatory"].index.to_numpy()
print(f"{len(exc_keys)} excitatory units")
exc_units = units[exc_keys]

# --- tuning curves with pynapple (pooled over direction) ---
lin_maze = lin.restrict(maze)
lin_frame = nap.TsdFrame(t=lin_maze.t, d=np.asarray(lin_maze.values),
                         columns=["lin"], time_support=lin_maze.time_support)
edges = np.linspace(0, TRACK_LEN, N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
tc = nap.compute_tuning_curves(
    exc_units, lin_frame, bins=[edges], epochs=run_iset)
print("tuning curves:", tc.shape, "dims:", tc.dims)
# occupancy in seconds, computed directly on the restricted feature
fs_pos = 1.0 / np.median(np.diff(t))
lin_run = lin_frame.restrict(run_iset)
x_run_all = np.asarray(lin_run.values).ravel()
occ_counts, _ = np.histogram(x_run_all[~np.isnan(x_run_all)], bins=edges)
occ_s = occ_counts / fs_pos
rate = np.asarray(tc).copy()  # (n_units, n_bins) in Hz
low_occ = occ_s < MIN_OCC_S
rate[:, low_occ] = np.nan
rate_sm = np.array([gaussian_filter1d(np.nan_to_num(r), SMOOTH_BINS) for r in rate])
rate_sm[:, low_occ] = np.nan

# --- Skaggs spatial information + circular time-shift shuffle ---
# spike positions via searchsorted (value_from fails on Ts targets)
def spike_positions(unit_ts, t_pos, x_pos):
    idx = np.searchsorted(t_pos, unit_ts)
    idx = np.clip(idx, 0, len(x_pos) - 1)
    return x_pos[idx]

def skaggs_si(rate_map, occ_map):
    r = np.nan_to_num(rate_map)
    o = occ_map / np.nansum(occ_map)
    mean_r = np.nansum(o * r)
    if mean_r <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        si = np.nansum(o * r * np.log2(r / mean_r)) / mean_r
    return float(si)

# --- concatenated run-bout time axis for circular time shifts ---
# map wall-clock times inside run bouts onto a contiguous axis tau in
# [0, total_run); circular shifts on tau keep spike counts fixed and every
# shifted spike lands on a valid on-track position sample.
bout_starts = np.array([a for a, b in run_bouts])
bout_ends = np.array([b for a, b in run_bouts])
bout_durs = bout_ends - bout_starts
cum_dur = np.concatenate([[0], np.cumsum(bout_durs)])[:-1]
T_run = bout_durs.sum()

def wall_to_tau(ts):
    i = np.searchsorted(bout_starts, ts, side="right") - 1
    i = np.clip(i, 0, len(bout_starts) - 1)
    return ts - bout_starts[i] + cum_dur[i]

pos_run = lin_maze.restrict(run_iset)
t_run, x_run = pos_run.t, np.asarray(pos_run.values).ravel()
valid_run = ~np.isnan(x_run)
tau_pos = wall_to_tau(t_run[valid_run])
x_run_v = x_run[valid_run]
order_tau = np.argsort(tau_pos)
tau_pos, x_run_v = tau_pos[order_tau], x_run_v[order_tau]

si_real = np.zeros(len(exc_keys))
spk_tau_all = []
for i, k in enumerate(exc_keys):
    sp = exc_units[k].restrict(run_iset).t
    spk_tau_all.append(wall_to_tau(sp))
    spos = spike_positions(spk_tau_all[i], tau_pos, x_run_v)
    h, _ = np.histogram(spos, bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = h / occ_s
    si_real[i] = skaggs_si(gaussian_filter1d(np.nan_to_num(rm), SMOOTH_BINS), occ_s)

# shuffle: circularly shift spike times on the concatenated run axis
si_null = np.zeros((len(exc_keys), N_SHUFFLE))
for j in range(N_SHUFFLE):
    shift = RNG.uniform(0, T_run)
    for i in range(len(exc_keys)):
        sp_sh = np.sort((spk_tau_all[i] + shift) % T_run)
        spos = spike_positions(sp_sh, tau_pos, x_run_v)
        h, _ = np.histogram(spos, bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = h / occ_s
        si_null[i, j] = skaggs_si(gaussian_filter1d(np.nan_to_num(rm), SMOOTH_BINS), occ_s)

si_p = (np.sum(si_null >= si_real[:, None], axis=1) + 1) / (N_SHUFFLE + 1)

# --- place cell criteria ---
peak_rate = np.nanmax(rate_sm, axis=1)
n_spikes = np.array([len(exc_units[k].restrict(run_iset)) for k in exc_keys])
mean_rate_run = n_spikes / total_run
is_place = (si_p < 0.05) & (peak_rate >= 1.0) & (mean_rate_run >= 0.1)
print(f"place cells: {is_place.sum()}/{len(exc_keys)} "
      f"(SI p<0.05, peak>=1 Hz, mean>=0.1 Hz)")
print(f"median SI: exc {np.median(si_real):.3f}, null {np.median(si_null):.3f} bits/spike")

np.savez("place_fields.npz", rate_sm=rate_sm, occ_s=occ_s, edges=edges,
         centers=centers, exc_keys=exc_keys, si_real=si_real, si_p=si_p,
         is_place=is_place, peak_rate=peak_rate, mean_rate_run=mean_rate_run,
         run_bouts=np.array(run_bouts), bout_dir=bout_dir)

# ---------- figures ----------
# 1) example place fields: top 12 by SI
order = np.argsort(si_real)[::-1]
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, i in zip(axes.ravel(), order[:12]):
    ax.plot(centers, rate_sm[i], color="navy")
    ax.fill_between(centers, 0, rate_sm[i], color="navy", alpha=0.2)
    ax.set_title(f"unit {exc_keys[i]}: SI {si_real[i]:.2f}, p={si_p[i]:.3f}", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 0.8, 1.6])
for ax in axes[-1]:
    ax.set_xlabel("track position (m)")
for ax in axes[:, 0]:
    ax.set_ylabel("rate (Hz)")
plt.suptitle("Example place fields (top 12 by spatial information)")
plt.tight_layout()
plt.savefig("fig_placefields_examples.png", dpi=150)

# 2) population rate map, place cells sorted by peak position
pc_idx = np.where(is_place)[0]
pk = np.array([np.nanargmax(rate_sm[i]) for i in pc_idx])
pc_order = pc_idx[np.argsort(pk)]
norm = np.array([r / np.nanmax(r) if np.nanmax(r) > 0 else r for r in rate_sm[pc_order]])
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(norm, aspect="auto", extent=[0, TRACK_LEN, len(pc_order), 0],
          cmap="viridis", interpolation="nearest")
ax.set_xlabel("track position (m)")
ax.set_ylabel("place cell (sorted by peak)")
ax.set_title(f"{len(pc_order)} place cells, direction-pooled rate maps")
plt.tight_layout()
plt.savefig("fig_placefields_population.png", dpi=150)

# 3) SI distribution vs shuffle
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(si_null.ravel(), bins=60, density=True, alpha=0.6, color="gray", label="time-shift null")
ax.hist(si_real, bins=60, density=True, alpha=0.6, color="seagreen", label="excitatory units")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("density")
ax.set_title(f"Spatial information: {is_place.sum()} place cells")
ax.legend()
plt.tight_layout()
plt.savefig("fig_placefields_si.png", dpi=150)
print("saved place field figures")
