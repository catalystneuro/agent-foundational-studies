"""Preprocess the Achilles session: build run epoch, extract spikes + position, cache to npz."""
import sys
sys.path.insert(0, ".")
from importlib import import_module
import numpy as np
import pynapple as nap
from tqdm import tqdm

load_mod = import_module("01_load_data")

OUT = "achilles_maze_cache.npz"


def main():
    nwb = load_mod.load_nwb()
    ep = nwb["epochs"]
    maze = ep[ep["label"] == "MazeEpoch"]

    pos = nwb["1.6mLinearMazeSpatialSeries"].restrict(maze)
    lin = nwb["1.6mLinearMazeLinearizedTimeSeries"].restrict(maze)
    t = pos.t
    xy = pos.values
    lin_v = lin.values[:, 0]
    fs = lin.rate
    print(f"position samples: {len(t)}, fs={fs:.2f} Hz")

    # Speed from linearized position (smoothed with a 5-sample boxcar ~ 128 ms)
    dlin = np.gradient(lin_v, t)
    kernel = np.ones(5) / 5
    speed = np.convolve(dlin, kernel, mode="same")

    valid = ~np.isnan(lin_v)
    run_mask = valid & (np.abs(speed) > 0.05)  # > 5 cm/s
    print(f"run samples: {run_mask.sum()} ({run_mask.mean()*100:.1f}% of maze epoch)")

    # Build run IntervalSet from contiguous True bouts (gap tolerance 0.5 s, min 1 s)
    run_ep = mask_to_intervalset(t, run_mask, gap_tol=0.5, min_dur=1.0)
    print(f"run epoch: {len(run_ep)} bouts, total {run_ep.tot_length():.1f} s")

    # Direction per sample: sign of linearized velocity
    direction = np.sign(speed)  # +1 = increasing x, -1 = decreasing x

    # Extract spikes restricted to run epoch
    units = nwb["units"]
    unit_ids = np.array(list(units.keys()))
    cell_type = np.array([units.get_info("cell_type")[u] for u in unit_ids])
    location = np.array([units.get_info("location")[u] for u in unit_ids])
    spikes_run = {}
    n_spikes_run = np.zeros(len(unit_ids), dtype=int)
    for i, u in enumerate(tqdm(unit_ids, desc="extracting spikes")):
        spk = units[u].restrict(run_ep).t
        spikes_run[str(u)] = spk
        n_spikes_run[i] = len(spk)

    np.savez_compressed(
        OUT,
        t=t, xy=xy, lin=lin_v, speed=speed, direction=direction, fs=fs,
        run_starts=run_ep.start, run_ends=run_ep.end,
        unit_ids=unit_ids, cell_type=cell_type, location=location,
        n_spikes_run=n_spikes_run,
        **{f"spk_{u}": spikes_run[str(u)] for u in unit_ids},
    )
    print(f"saved {OUT}")
    print("spikes in run epoch per unit: median", np.median(n_spikes_run),
          "min", n_spikes_run.min(), "max", n_spikes_run.max())


def mask_to_intervalset(t, mask, gap_tol=0.5, min_dur=1.0):
    """Convert a boolean mask on timebase t to an IntervalSet, bridging short gaps."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return nap.IntervalSet(start=[], end=[])
    splits = np.where(np.diff(t[idx]) > gap_tol)[0]
    starts = np.concatenate([[idx[0]], idx[splits + 1]])
    ends = np.concatenate([idx[splits], [idx[-1]]])
    s = t[starts]
    e = t[ends]
    keep = (e - s) >= min_dur
    return nap.IntervalSet(start=s[keep], end=e[keep])


if __name__ == "__main__":
    main()
