"""
Linearize position on the circular/oval maze, restrict to one consistent
running direction (the 'Right' pre-cooling trials, which are near-complete
laps around the loop), build a Pynapple TsGroup of units, and identify
place cells from 1D occupancy-normalized tuning curves.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

CACHE_DIR = "cache"
FIG_DIR = "figures"


def main():
    d = np.load(f"{CACHE_DIR}/session_data.npz")
    trials = pd.read_pickle(f"{CACHE_DIR}/trials.pkl")
    with open(f"{CACHE_DIR}/units.pkl", "rb") as fh:
        spike_times = pickle.load(fh)

    pos_xyz = d["pos_xyz"]
    pos_t = d["pos_t"]
    speed = d["speed"]
    speed_t = d["speed_t"]

    valid = ~np.isnan(pos_xyz[:, 0]) & ~np.isnan(pos_xyz[:, 1])
    position = nap.TsdFrame(
        t=pos_t[valid], d=pos_xyz[valid, :2], columns=["x", "y"]
    )
    speed_tsd = nap.Tsd(t=speed_t, d=speed)

    # --- raw data validation plot -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sc = axes[0].scatter(
        position["x"].values, position["y"].values, c=position.index.values,
        s=1, cmap="viridis",
    )
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Full session maze trajectory")
    axes[0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0], label="time (s)")

    pre_right = trials[(trials["cooling state"] == "Pre-Cooling") & (trials["condition"] == "Right")]
    t0, t1 = pre_right.iloc[0][["start_time", "stop_time"]]
    m = (pos_t >= t0) & (pos_t <= t1)
    axes[1].plot(pos_xyz[m, 0], pos_xyz[m, 1], "-o", ms=3, color="C1")
    axes[1].scatter(pos_xyz[m, 0][:1], pos_xyz[m, 1][:1], color="green", s=80, label="start", zorder=5)
    axes[1].scatter(pos_xyz[m, 0][-1:], pos_xyz[m, 1][-1:], color="red", s=80, label="end", zorder=5)
    axes[1].set_title(f"Single 'Right' trial (id={pre_right.index[0]})")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_aspect("equal")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_maze_trajectory.png", dpi=150)
    plt.close()
    print("Saved figures/01_maze_trajectory.png")

    # --- linearize: unwrapped angle around the loop centroid, per trial --
    cx, cy = np.nanmean(pos_xyz[:, 0]), np.nanmean(pos_xyz[:, 1])
    print(f"Loop centroid: ({cx:.3f}, {cy:.3f})")

    run_starts, run_stops, lin_t, lin_pos = [], [], [], []
    for _, row in pre_right.iterrows():
        t0, t1 = row["start_time"], row["stop_time"]
        m = (pos_t >= t0) & (pos_t <= t1) & valid
        if m.sum() < 5:
            continue
        tt = pos_t[m]
        xx = pos_xyz[m, 0]
        yy = pos_xyz[m, 1]
        ang = np.unwrap(np.arctan2(yy - cy, xx - cx))
        linpos = -(ang - ang[0])  # increasing distance traveled around the loop
        if linpos[-1] < 2.0:  # drop incomplete/aborted laps
            continue
        run_starts.append(tt[0])
        run_stops.append(tt[-1])
        lin_t.append(tt)
        lin_pos.append(linpos)

    run_epochs = nap.IntervalSet(start=run_starts, end=run_stops)
    linpos_tsd = nap.Tsd(t=np.concatenate(lin_t), d=np.concatenate(lin_pos), time_support=run_epochs)
    print(f"{len(run_epochs)} complete 'Right' laps retained, "
          f"linpos range [{linpos_tsd.values.min():.2f}, {linpos_tsd.values.max():.2f}] rad")

    # --- units & speed-thresholded run epochs -------------------------
    t_session0, t_session1 = float(pos_t[0]), float(pos_t[-1])
    units = nap.TsGroup(
        {uid: nap.Ts(t=st[(st >= t_session0) & (st <= t_session1)]) for uid, st in spike_times.items()}
    )
    rates = units.restrict(run_epochs).rate
    active = units[rates > 0.5]
    print(f"{len(active)}/{len(units)} units fire at >0.5 Hz during runs")

    speed_run = speed_tsd.restrict(run_epochs)
    fast_epochs = speed_run.threshold(0.05).time_support  # >5 cm/s
    run_fast = run_epochs.intersect(fast_epochs)

    # --- 1D tuning curves & place-cell selection -----------------------
    tc = nap.compute_1d_tuning_curves(active, linpos_tsd, nb_bins=40, ep=run_fast)

    occupancy = np.histogram(
        linpos_tsd.restrict(run_fast).values, bins=tc.index.size,
        range=(linpos_tsd.values.min(), linpos_tsd.values.max()),
    )[0]
    occ_p = occupancy / occupancy.sum()
    rate_vals = np.nan_to_num(tc.values, nan=0.0)  # unvisited bins contribute nothing
    mean_rate = (rate_vals * occ_p[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rate_vals > 0, rate_vals / mean_rate[None, :], 1.0)
        info_terms = np.where(rate_vals > 0, occ_p[:, None] * rate_vals * np.log2(ratio), 0.0)
    spatial_info = info_terms.sum(axis=0) / mean_rate  # bits/spike
    spatial_info = pd.Series(spatial_info, index=tc.columns)
    peak_rate = pd.Series(rate_vals.max(axis=0), index=tc.columns)

    summary = pd.DataFrame({
        "peak_rate": peak_rate,
        "mean_rate": mean_rate,
        "spatial_info_bits_per_spike": spatial_info,
    }).sort_values("spatial_info_bits_per_spike", ascending=False)
    print("\nTop candidate place cells:")
    print(summary.head(10))

    place_cells = summary[(summary["peak_rate"] > 2.0) & (summary["spatial_info_bits_per_spike"] > 0.5)]
    place_cells = place_cells.sort_values("spatial_info_bits_per_spike", ascending=False)
    print(f"\n{len(place_cells)} units pass place-cell criteria (peak>2Hz, SI>0.5 bits/spike)")
    print(place_cells.head(15))

    # save
    tc.to_pickle(f"{CACHE_DIR}/tuning_curves.pkl")
    summary.to_pickle(f"{CACHE_DIR}/place_cell_summary.pkl")
    place_cells.to_pickle(f"{CACHE_DIR}/place_cells.pkl")
    run_fast.save(f"{CACHE_DIR}/run_epochs.npz")
    np.savez(
        f"{CACHE_DIR}/linpos.npz",
        t=linpos_tsd.index.values,
        d=linpos_tsd.values,
        run_starts=run_fast.start,
        run_stops=run_fast.end,
    )
    with open(f"{CACHE_DIR}/units_active.pkl", "wb") as fh:
        pickle.dump({uid: np.asarray(active[uid].t) for uid in active.index}, fh)

    print("\nSaved cache/tuning_curves.pkl, place_cell_summary.pkl, place_cells.pkl, linpos.npz, units_active.pkl")


if __name__ == "__main__":
    main()
