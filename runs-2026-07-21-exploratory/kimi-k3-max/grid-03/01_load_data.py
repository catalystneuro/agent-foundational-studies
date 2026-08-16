"""01_load_data.py — prototype loading of one Sargolini MEC session from DANDI 000582.

Streams sub-11265_ses-16030604 (20 min, 1.5 m box, 14 MEC LII units) with remfile,
inspects the NWB structure, and plots the raw position trace plus spike positions
for a few units to validate the data streams before analysis.
"""

import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

DANDI_API_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"
DEMO_PATH = "sub-11265/sub-11265_ses-16030604_behavior+ecephys.nwb"


def load_session(asset_id, cache_dir="/tmp/remfile_cache_grid"):
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(DANDI_API_URL.format(asset_id=asset_id), disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, io


if __name__ == "__main__":
    with open("asset_manifest.json") as f:
        manifest = json.load(f)
    asset_id = manifest[DEMO_PATH]

    nwb, io = load_session(asset_id)
    print(nwb)
    units = nwb["units"]
    print("\nUnits metadata columns:", list(units.metadata_columns))

    # Position stream: processing/behavior/Position/SpatialSeriesLED1 (50 Hz, values in cm)
    pos = nwb["SpatialSeriesLED1"]
    print("\nPosition:", pos)
    print("Position shape:", pos.shape, "rate:", pos.rate)
    print("Position range x:", np.nanmin(pos[:, 0]), np.nanmax(pos[:, 0]))
    print("Position range y:", np.nanmin(pos[:, 1]), np.nanmax(pos[:, 1]))
    print("NaN frames:", np.isnan(pos[:, 0].values).sum(), "of", pos.shape[0])

    print("\nN units:", len(units))
    print(units[list(units.keys())[:3]])

    # --- Validation figure: position trace + spike positions for 4 units ---
    t = pos.t
    xy = pos.values
    valid = ~np.isnan(xy[:, 0]) & ~np.isnan(xy[:, 1])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ax = axes[0, 0]
    ax.plot(xy[valid, 0], xy[valid, 1], lw=0.2, color="0.4")
    ax.set_title("Position trace (whole session)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")

    ax = axes[1, 0]
    sl = slice(0, 3000)  # first 60 s at 50 Hz
    ax.plot(t[sl], xy[sl, 0], lw=0.5, label="x")
    ax.plot(t[sl], xy[sl, 1], lw=0.5, label="y")
    ax.set_title("First 60 s of position")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (cm)")
    ax.legend()

    for k in range(3):
        uid = list(units.keys())[k]
        spk = units[uid]
        # spike positions by interpolation onto the position timeline
        spk_xy = np.column_stack(
            [np.interp(spk.t, t[valid], xy[valid, 0]),
             np.interp(spk.t, t[valid], xy[valid, 1])]
        )
        ax = axes[0, k + 1]
        ax.plot(xy[valid, 0], xy[valid, 1], lw=0.1, color="0.8", zorder=0)
        ax.scatter(spk_xy[:, 0], spk_xy[:, 1], s=2, c="r", zorder=1)
        layer = units["histology"][uid] if "histology" in units.metadata_columns else "?"
        ax.set_title(f"unit {uid} ({layer}), {len(spk)} spikes")
        ax.set_aspect("equal")

        ax = axes[1, k + 1]
        ax.plot(t[sl], xy[sl, 0], lw=0.5, color="0.6")
        in_win = spk.t[(spk.t >= t[sl][0]) & (spk.t <= t[sl][-1])]
        ax.scatter(in_win, np.interp(in_win, t[valid], xy[valid, 0]), s=4, c="r")
        ax.set_title(f"unit {uid} spikes on x(t)")
        ax.set_xlabel("time (s)")

    fig.suptitle("DANDI 000582 — sub-11265_ses-16030604 raw data validation")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fig_01_raw_data_validation.png", dpi=150)
    print("\nsaved fig_01_raw_data_validation.png")
    io.close()
