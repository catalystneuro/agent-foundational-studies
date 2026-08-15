"""Second pass: per-trial mean running speed for each analysed session.

Kept separate from run_sessions.py because it only needs the (small) running
module of each NWB file, which the remfile cache already holds.
"""

import os
import pickle

import numpy as np
from tqdm import tqdm

import orilib as O
import analysis as A

OUT = "session_results"


def trial_mean_speed(running, starts, stops):
    """Mean running speed within each trial window."""
    t, v = running.t, running.d
    csum = np.concatenate([[0.0], np.cumsum(v)])
    lo = np.searchsorted(t, starts, "left")
    hi = np.searchsorted(t, stops, "right")
    n = np.maximum(hi - lo, 1)
    return (csum[hi] - csum[lo]) / n


def main():
    assets = {a["path"].split("ses-")[1].replace(".nwb", ""): a for a in O.get_assets()}
    for sid in tqdm(sorted(os.listdir(OUT)), desc="running speed"):
        if not sid.endswith(".pkl"):
            continue
        sid = sid[:-4]
        fp = os.path.join(OUT, f"{sid}_running.npz")
        if os.path.exists(fp):
            continue
        nwbfile = O.open_nwb(assets[sid]["url"])
        run = O.load_running_speed(nwbfile)
        dg = A._clean_trials(
            O.load_stim_table(nwbfile, "drifting_gratings_presentations"), A.DIRECTIONS
        )
        speed = trial_mean_speed(run, dg["start_time"].values, dg["stop_time"].values)
        np.savez(fp, speed=speed, start_time=dg["start_time"].values,
                 stop_time=dg["stop_time"].values)
        print("wrote", fp, "median speed", np.median(speed), flush=True)


if __name__ == "__main__":
    main()
