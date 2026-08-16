# 02_place_fields.py — direction-specific linear-track tuning curves and
# place-field detection.
#
# Significance of spatial information is assessed with a TIME-shift shuffle:
# spike times are circularly shifted relative to the trajectory (preserving
# spike-train structure and occupancy). Shifting spike POSITIONS in space
# instead would leave SI invariant under uniform occupancy and produce
# p ~ 0.75 everywhere — a silent failure mode.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
from achilles_loader import open_nwb

NBINS = 50
TRACK_LEN = 1.6
SMOOTH_SD = 1.5  # bins
N_SHUFFLE = 500
RNG = np.random.default_rng(42)

C = np.load("cache_preproc.npz")
pos_t, lin_v = C["pos_t"], C["lin"]
bouts = np.column_stack([C["bout_starts"], C["bout_ends"]])
bout_dir = C["bout_dir"]
maze_start, maze_end = float(C["maze_start"]), float(C["maze_end"])

nwbfile, nwb, h5 = open_nwb()
units = nwb["units"]
cell_type = units.get_info("cell_type")
unit_ids = list(units.keys())
exc_ids = [u for u in unit_ids if cell_type[u] == "excitatory"]
print(f"{len(unit_ids)} units, {len(exc_ids)} excitatory")

edges = np.linspace(0, TRACK_LEN, NBINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
binw = TRACK_LEN / NBINS
gw = signal.windows.gaussian(int(round(SMOOTH_SD * 6)) | 1, std=SMOOTH_SD)
gw /= gw.sum()


def spikes_lin_pos(spk_t):
    """linearized position at each spike time (NaN if position invalid)"""
    idx = np.searchsorted(pos_t, spk_t) - 1
    idx = np.clip(idx, 0, len(pos_t) - 1)
    return lin_v[idx]


def in_bouts(t, d_bouts):
    """vectorized membership test of times t in bouts (N,2 array)"""
    starts, ends = d_bouts[:, 0], d_bouts[:, 1]
    idx = np.searchsorted(starts, t, side="right") - 1
    inb = np.zeros(len(t), dtype=bool)
    ok = idx >= 0
    inb[ok] = t[ok] <= ends[idx[ok]]
    return inb


def rate_map(spk_pos, occ):
    """smoothed firing-rate map given spike positions and occupancy (s per bin)"""
    cnt = np.histogram(spk_pos[np.isfinite(spk_pos)], bins=edges)[0].astype(float)
    cnt_s = np.convolve(cnt, gw, mode="same")
    occ_s = np.convolve(occ, gw, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = cnt_s / occ_s
    rate[occ_s < 0.05] = np.nan  # under-occupied bins
    return rate, cnt


def skaggs_si(rate, occ):
    """spatial information (bits/spike), Skaggs et al. 1993"""
    p = occ / occ.sum()
    ok = np.isfinite(rate) & (rate > 0) & (p > 0)
    mean_rate = np.nansum(rate[ok] * p[ok])
    if mean_rate <= 0:
        return 0.0, 0.0
    si = np.nansum(p[ok] * (rate[ok] / mean_rate) * np.log2(rate[ok] / mean_rate))
    return si, mean_rate


results = {}  # (unit_id, direction) -> dict
occ_by_dir = {}
for d in (+1, -1):
    d_bouts = bouts[bout_dir == d]
    occ = np.zeros(NBINS)
    for s, e in d_bouts:
        m = (pos_t >= s) & (pos_t <= e) & np.isfinite(lin_v)
        occ += np.histogram(lin_v[m], bins=edges)[0] * np.median(np.diff(pos_t))
    occ_by_dir[d] = occ
    print(f"direction {d:+d}: {len(d_bouts)} bouts, occupancy {occ.sum():.0f} s, "
          f"min occ {occ.min():.2f} s")

for uid in tqdm(exc_ids, desc="place fields"):
    spk = units[uid]
    spk_t = np.asarray(spk.t)
    spk_pos = spikes_lin_pos(spk_t)
    for d in (+1, -1):
        d_bouts = bouts[bout_dir == d]
        mask = in_bouts(spk_t, d_bouts)
        sp = spk_pos[mask]
        sp = sp[np.isfinite(sp)]
        occ = occ_by_dir[d]
        rate, cnt = rate_map(sp, occ)
        si, mean_rate = skaggs_si(rate, occ)
        peak = np.nanmax(rate) if np.isfinite(rate).any() else 0.0
        n_spikes = len(sp)
        # shuffle: circularly shift spike TIMES relative to the trajectory
        si_null = np.empty(N_SHUFFLE)
        epoch_len = maze_end - maze_start
        for i in range(N_SHUFFLE):
            tau = RNG.uniform(20, epoch_len - 20)
            t_sh = maze_start + (spk_t - maze_start + tau) % epoch_len
            pos_sh = spikes_lin_pos(t_sh)
            m_sh = in_bouts(t_sh, d_bouts)
            sp_sh = pos_sh[m_sh]
            sp_sh = sp_sh[np.isfinite(sp_sh)]
            r_sh, _ = rate_map(sp_sh, occ)
            si_null[i], _ = skaggs_si(r_sh, occ)
        p_si = float(np.mean(si_null >= si))
        # field: contiguous bins > 20% of peak containing the peak bin
        field = None
        if np.isfinite(rate).any() and peak >= 1.0:
            pk_bin = int(np.nanargmax(rate))
            thr = 0.2 * peak
            lo, hi = pk_bin, pk_bin
            while lo > 0 and np.isfinite(rate[lo - 1]) and rate[lo - 1] > thr:
                lo -= 1
            while hi < NBINS - 1 and np.isfinite(rate[hi + 1]) and rate[hi + 1] > thr:
                hi += 1
            width = (hi - lo + 1) * binw
            field = dict(lo=lo * binw, hi=(hi + 1) * binw, width=width,
                         peak_bin=pk_bin, peak_pos=centers[pk_bin])
        results[(uid, d)] = dict(rate=rate, si=si, p_si=p_si, mean_rate=mean_rate,
                                 peak=peak, n_spikes=n_spikes, field=field)

# --- select place cells ---
place = {}
for (uid, d), r in results.items():
    f = r["field"]
    if (f is not None and r["p_si"] < 0.05 and r["n_spikes"] >= 30
            and 0.2 <= f["width"] <= 1.2):
        place[(uid, d)] = r
print(f"\nplace cell-directions: {len(place)} "
      f"(unique cells: {len(set(u for u, _ in place))})")

np.savez_compressed("cache_fields.npz",
                    results_keys=np.array([(u, d) for (u, d) in results.keys()]),
                    place_keys=np.array([(u, d) for (u, d) in place.keys()]),
                    **{f"rate_{u}_{d}": r["rate"] for (u, d), r in results.items()},
                    **{f"meta_{u}_{d}": np.array([r["si"], r["p_si"], r["mean_rate"],
                                                  r["peak"], r["n_spikes"]])
                       for (u, d), r in results.items()},
                    **{f"field_{u}_{d}": np.array([r["field"]["lo"], r["field"]["hi"],
                                                   r["field"]["width"],
                                                   r["field"]["peak_pos"]])
                       for (u, d), r in results.items() if r["field"] is not None},
                    centers=centers)

# --- figure: population rate maps sorted by peak, per direction ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6), width_ratios=[1, 1, 1.2])
for ax, d, ttl in zip(axes[:2], (+1, -1), ["+ direction", "- direction"]):
    keys = [(u, dd) for (u, dd) in place.keys() if dd == d]
    keys.sort(key=lambda k: place[k]["field"]["peak_pos"])
    if keys:
        M = np.array([place[k]["rate"] / np.nanmax(place[k]["rate"]) for k in keys])
        im = ax.imshow(M, aspect="auto", cmap="viridis",
                       extent=[0, TRACK_LEN, len(keys), 0])
        ax.set_title(f"{ttl}: {len(keys)} place fields")
        ax.set_xlabel("linearized position (m)")
        ax.set_ylabel("cell (sorted by peak)")
        fig.colorbar(im, ax=ax, label="norm. rate", shrink=0.8)

ax = axes[2]
si_all = np.array([results[k]["si"] for k in results])
is_place = np.array([k in place for k in results])
ax.hist(si_all[is_place], bins=30, alpha=0.7, label=f"place (n={is_place.sum()})")
ax.hist(si_all[~is_place], bins=30, alpha=0.7,
        label=f"other (n={len(si_all) - is_place.sum()})")
ax.set_xlabel("Skaggs spatial information (bits/spike)")
ax.set_ylabel("cell-directions")
ax.set_title("Place-cell selection (SI shuffle p<0.05)")
ax.legend()
fig.tight_layout()
fig.savefig("fig02_place_fields.png", dpi=150)
print("saved fig02_place_fields.png")
