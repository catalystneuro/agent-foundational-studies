"""
Plot linearized-position tuning curves (place fields) for the candidate
place cells and a spike raster over the maze trajectory for the exemplar
precessing cell, for visual QC and to contextualize the phase-precession
result.
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
    lin = np.load(f"{CACHE_DIR}/linpos.npz")
    tc = pd.read_pickle(f"{CACHE_DIR}/tuning_curves.pkl")
    place_cells = pd.read_pickle(f"{CACHE_DIR}/place_cells.pkl")
    precession = pd.read_pickle(f"{CACHE_DIR}/precession_results.pkl")
    with open(f"{CACHE_DIR}/units_active.pkl", "rb") as fh:
        units_active = pickle.load(fh)

    pos_xyz = d["pos_xyz"]
    pos_t = d["pos_t"]
    run_epochs = nap.IntervalSet(start=lin["run_starts"], end=lin["run_stops"])

    fig, axes = plt.subplots(1, len(place_cells.index), figsize=(3.2 * len(place_cells.index), 3), sharey=False)
    for ax, uid in zip(axes, place_cells.index):
        curve = np.nan_to_num(tc[uid].values, nan=0.0)
        ax.plot(tc.index.values, curve, color="C0")
        ax.set_title(f"unit {uid}")
        ax.set_xlabel("linearized position (rad)")
        if ax is axes[0]:
            ax.set_ylabel("firing rate (Hz)")
    plt.suptitle("Linearized place fields (candidate place cells, 'Right'-lap runs)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_place_field_tuning_curves.png", dpi=150)
    plt.close()
    print("Saved figures/05_place_field_tuning_curves.png")

    # spike raster over the maze trajectory for the best precessing cell
    best_uid = precession["rho"].abs().idxmax()
    st = units_active[best_uid]
    spk = nap.Ts(t=st).restrict(run_epochs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pos_xyz[:, 0], pos_xyz[:, 1], color="lightgray", lw=0.5, zorder=1)
    valid = ~np.isnan(pos_xyz[:, 0])
    idx = np.searchsorted(pos_t, spk.index.values)
    idx = np.clip(idx, 0, len(pos_t) - 1)
    ax.scatter(pos_xyz[idx, 0], pos_xyz[idx, 1], color="C3", s=15, zorder=2, label=f"unit {best_uid} spikes")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Spike locations for exemplar precessing cell (unit {best_uid})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_exemplar_spike_locations.png", dpi=150)
    plt.close()
    print("Saved figures/06_exemplar_spike_locations.png")


if __name__ == "__main__":
    main()
