"""Stream the Achilles-10252013 session (DANDI:000044) and cache the maze-epoch
data needed for a theta phase-precession analysis to a local .npz file.

We extract, restricted to the MAZE (linear-track) epoch:
  - spike times for excitatory CA1 units
  - linearized position (0-1.6 m) and its timestamps
  - one CA1 LFP channel selected for strong theta, at 1250 Hz
"""
import numpy as np
import time
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
from scipy.signal import butter, filtfilt, welch

CACHE = "achilles_maze_cache.npz"
ASSET = "Achilles-10252013"


def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def main():
    client = DandiAPIClient()
    d = client.get_dandiset("000044", "draft")
    asset = [x for x in d.get_assets() if ASSET in x.path][0]
    print("streaming", asset.path)
    rf = remfile.File(asset.download_url)
    h = h5py.File(rf, "r")

    # --- epochs: identify the MAZE epoch (the one that contains position data) ---
    ep = h["intervals/epochs"]
    ep_start = ep["start_time"][:]
    ep_stop = ep["stop_time"][:]
    print("epochs:", list(zip(ep_start, ep_stop)))
    # The maze epoch is the middle one (index 1) per inspection
    maze_start, maze_stop = float(ep_start[1]), float(ep_stop[1])
    print(f"MAZE epoch: {maze_start:.1f} - {maze_stop:.1f} s  ({maze_stop-maze_start:.0f} s)")

    # --- linearized position ---
    lin = h["processing/behavior/1.6mLinearMazeLinearizedPosition/1.6mLinearMazeLinearizedTimeSeries"]
    pos_rate = float(lin["starting_time"].attrs["rate"])
    pos_t0 = float(lin["starting_time"][()])
    pos_data = lin["data"][:, 0].astype(float)
    pos_t = pos_t0 + np.arange(pos_data.shape[0]) / pos_rate
    print("position: n=", pos_data.shape[0], "rate=", round(pos_rate, 3),
          "t=[%.1f, %.1f]" % (pos_t[0], pos_t[-1]))

    # --- units: excitatory CA1 cells ---
    u = h["units"]
    st_index = u["spike_times_index"][:]
    st_all = u["spike_times"]
    cell_type = u["cell_type"][:]
    location = u["location"][:]
    n_units = len(st_index)
    starts = np.concatenate([[0], st_index[:-1]])
    spike_trains = []
    keep_meta = []
    for i in range(n_units):
        s = st_all[starts[i]:st_index[i]]
        ct = cell_type[i].decode() if isinstance(cell_type[i], bytes) else cell_type[i]
        loc = location[i].decode() if isinstance(location[i], bytes) else location[i]
        # restrict to maze epoch
        s_maze = s[(s >= maze_start) & (s <= maze_stop)]
        if ct == "excitatory" and s_maze.size > 20:
            spike_trains.append(s_maze)
            keep_meta.append((i, ct, loc, s_maze.size))
    print(f"kept {len(spike_trains)} excitatory units with >20 maze spikes")

    # --- LFP channel selection: sample channels, pick strongest theta during maze ---
    es = h["processing/ecephys/LFP/LFP/data"]
    lfp_rate = float(h["processing/ecephys/LFP/LFP/starting_time"].attrs["rate"])
    conv = float(es.attrs["conversion"])
    i0 = int(round(maze_start * lfp_rate))
    i1 = int(round(maze_stop * lfp_rate))
    print("LFP maze samples:", i1 - i0, "at", lfp_rate, "Hz")

    # candidate channels: one representative per shank (14 shanks -> sample 24 chans)
    cand = np.arange(4, 128, 5)
    best_chan, best_ratio = None, -1
    t = time.time()
    # use a 200 s sub-window for speed
    w0 = i0
    w1 = min(i0 + int(200 * lfp_rate), i1)
    for ch in cand:
        x = es[w0:w1, ch].astype(float) * conv
        f, p = welch(x, fs=lfp_rate, nperseg=int(4 * lfp_rate))
        theta = p[(f >= 6) & (f <= 12)].mean()
        delta = p[(f >= 1) & (f <= 4)].mean()
        ratio = theta / delta
        if ratio > best_ratio:
            best_ratio, best_chan = ratio, int(ch)
    print(f"selected LFP channel {best_chan} (theta/delta={best_ratio:.2f}) in {time.time()-t:.0f}s")

    # read the full maze-epoch LFP for the chosen channel
    t = time.time()
    lfp = es[i0:i1, best_chan].astype(float) * conv
    lfp_t = i0 / lfp_rate + np.arange(lfp.size) / lfp_rate
    print(f"read LFP channel: {lfp.size} samples in {time.time()-t:.0f}s")

    np.savez_compressed(
        CACHE,
        maze_start=maze_start, maze_stop=maze_stop,
        pos_t=pos_t, pos=pos_data, pos_rate=pos_rate,
        lfp=lfp.astype(np.float32), lfp_t=lfp_t.astype(np.float64), lfp_rate=lfp_rate,
        best_chan=best_chan,
        spike_trains=np.array(spike_trains, dtype=object),
        unit_meta=np.array(keep_meta, dtype=object),
    )
    print("wrote", CACHE)


if __name__ == "__main__":
    main()
