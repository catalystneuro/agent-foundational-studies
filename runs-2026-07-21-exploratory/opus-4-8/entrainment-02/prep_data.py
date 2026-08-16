"""
Stream the required data from DANDI:000044 (Grosmark & Buzsaki, hc-11) session
sub-Buddy and cache locally as an .npz so the analysis scripts do not re-stream.

We only pull what we need:
  - LFP for one candidate channel per shank (16 channels) restricted to the
    MazeEpoch, to select the channel with the strongest theta.
  - The full MazeEpoch LFP for the selected channel (single channel = cheap,
    chunks are (102807, 1)).
  - Spike times per unit (small), unit metadata, position, and epoch bounds.
"""
import numpy as np
from dandi.dandiapi import DandiAPIClient
import h5py, remfile
from pynwb import NWBHDF5IO
from scipy.signal import welch

CACHE = "cache/buddy_prep.npz"

def load_nwb():
    client = DandiAPIClient()
    ds = client.get_dandiset("000044", "draft")
    a = [x for x in ds.get_assets() if x.path.endswith(".nwb")]
    a.sort(key=lambda x: x.size)
    asset = a[0]  # sub-Buddy, smallest (5.2 GB)
    rem = remfile.File(asset.download_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5)
    nwb = io.read()
    return nwb, asset.path

def main():
    nwb, path = load_nwb()
    print("session:", path)

    # --- epochs ---
    edf = nwb.epochs.to_dataframe()
    maze = edf[edf["label"] == "MazeEpoch"].iloc[0]
    t0, t1 = float(maze["start_time"]), float(maze["stop_time"])
    print(f"MazeEpoch: {t0:.1f} - {t1:.1f} s ({t1-t0:.1f} s)")

    # --- LFP handle ---
    lfp = nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series["LFP"]
    rate = float(lfp.rate)
    n_t, n_ch = lfp.data.shape
    conv = float(lfp.conversion)  # to volts
    i0, i1 = int(t0 * rate), int(t1 * rate)
    print(f"LFP rate={rate} Hz, shape={lfp.data.shape}, conversion={conv}")
    maze_t = i0 + np.arange(i1 - i0) / rate * rate  # sample indices
    tvec = (np.arange(i0, i1)) / rate

    # --- candidate channels: one mid channel per shank (8 ch/shank) ---
    cand = list(range(3, n_ch, 8))  # channel 3 of each 8-ch shank
    print("candidate channels:", cand)
    theta_ratio = {}
    cand_lfp = {}
    for ch in cand:
        x = lfp.data[i0:i1, ch].astype(np.float32) * conv * 1e6  # microvolts
        cand_lfp[ch] = x
        f, p = welch(x, fs=rate, nperseg=int(rate * 4))
        theta = p[(f >= 6) & (f <= 10)].mean()
        broad = p[(f >= 2) & (f <= 40)].mean()
        theta_ratio[ch] = theta / broad
        print(f"  ch {ch:3d}: theta/broad power ratio = {theta_ratio[ch]:.3f}")

    best_ch = max(theta_ratio, key=theta_ratio.get)
    print("BEST theta channel:", best_ch, "ratio", theta_ratio[best_ch])
    lfp_best = cand_lfp[best_ch]

    # --- position (linearized) restricted later; grab full maze position ---
    beh = nwb.processing["behavior"].data_interfaces
    pos_key = [k for k in beh if "LinearizedPosition" in k][0]
    spatial = beh[pos_key].spatial_series
    ss = list(spatial.values())[0]
    pos_data = ss.data[:]
    if ss.timestamps is not None:
        pos_t = ss.timestamps[:]
    else:
        pos_t = ss.starting_time + np.arange(len(pos_data)) / ss.rate
    pos_data = np.asarray(pos_data).squeeze()
    print("position:", pos_data.shape, "t range", pos_t.min(), pos_t.max())

    # --- units ---
    udf = nwb.units.to_dataframe()
    spike_times = [np.asarray(udf.iloc[i]["spike_times"]) for i in range(len(udf))]
    locations = udf["location"].astype(str).values
    cell_types = udf["cell_type"].astype(str).values
    shank_ids = udf["shank_id"].values
    print("units:", len(udf))

    np.savez_compressed(
        CACHE,
        session=path,
        rate=rate,
        t0=t0, t1=t1,
        tvec=tvec.astype(np.float32),
        best_ch=best_ch,
        cand_channels=np.array(cand),
        cand_theta_ratio=np.array([theta_ratio[c] for c in cand]),
        lfp_best=lfp_best.astype(np.float32),
        pos_data=pos_data.astype(np.float32),
        pos_t=pos_t.astype(np.float64),
        spike_times=np.array(spike_times, dtype=object),
        locations=locations,
        cell_types=cell_types,
        shank_ids=shank_ids,
    )
    print("saved", CACHE)

if __name__ == "__main__":
    main()
