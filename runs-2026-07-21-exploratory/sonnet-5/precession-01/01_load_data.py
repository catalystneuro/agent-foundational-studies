"""
Stream the processed behavior+ecephys NWB file (units, position, speed, trials)
and the theta-reference LFP channel from the raw ecephys NWB file for one
session of DANDI:000059 (Cooling of Medial Septum Reveals Theta Phase Lag
Coordination of Hippocampal Cell Assemblies), subject MS10, session
170317-153237.

Caches everything to ./cache/session_data.npz and ./cache/units.pkl so that
downstream scripts do not need to re-stream from S3.
"""
import pickle
import time

import h5py
import numpy as np
import remfile
from pynwb import NWBHDF5IO
from scipy.signal import resample_poly
from tqdm import tqdm

CACHE_DIR = "cache"
DISK_CACHE_DIR = "/tmp/remfile_cache"

PROCESSED_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/093/2c2/"
    "0932c245-ac35-4dfd-be76-20ae328f43a4"
)
RAW_URL = (
    "https://dandiarchive.s3.amazonaws.com/blobs/ef5/164/"
    "ef516442-7328-44a4-bee2-cf8758077a46"
)

THETA_ELECTRODE = 46  # electrodes table: theta_reference == True
RAW_FS = 20000.0
LFP_TARGET_FS = 1250.0
DECIMATION = int(RAW_FS // LFP_TARGET_FS)  # 16


def load_processed(url):
    disk_cache = remfile.DiskCache(DISK_CACHE_DIR)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f)
    nwbfile = io.read()
    return nwbfile, io


def main():
    print("Loading processed behavior+ecephys NWB file (units, position, trials)...")
    nwbfile, io = load_processed(PROCESSED_URL)
    print(nwbfile)

    trials = nwbfile.trials.to_dataframe()
    print(f"\n{len(trials)} trials, cooling states: {trials['cooling state'].value_counts().to_dict()}")

    units_df = nwbfile.units.to_dataframe()
    spike_times = {uid: np.asarray(row["spike_times"]) for uid, row in units_df.iterrows()}
    print(f"{len(spike_times)} units, quality labels: {units_df['quality'].value_counts().to_dict()}")

    pos_series = nwbfile.processing["behavior"]["SubjectPosition"]["SpatialSeries"]
    pos_xyz = np.asarray(pos_series.data) * pos_series.conversion  # -> meters
    pos_t = np.asarray(pos_series.timestamps)
    print(f"Position: {pos_xyz.shape}, time range {pos_t[0]:.1f}-{pos_t[-1]:.1f} s")

    speed_series = nwbfile.processing["behavior"]["SubjectSpeed"]
    speed = np.asarray(speed_series.data)
    speed_t = np.asarray(speed_series.timestamps) if speed_series.timestamps is not None else None
    if speed_t is None:
        speed_t = speed_series.starting_time + np.arange(len(speed)) / speed_series.rate
    print(f"Speed: {speed.shape}")

    electrodes = nwbfile.electrodes.to_dataframe()
    theta_row = electrodes[electrodes["theta_reference"] == True]
    print(f"\nTheta reference electrode:\n{theta_row}")

    io.close()

    # Restrict LFP streaming to the Pre-Cooling epoch (baseline, no septal
    # cooling) so theta and precession reflect normal hippocampal dynamics.
    pre = trials[trials["cooling state"] == "Pre-Cooling"]
    t_start, t_stop = pre["start_time"].min() - 5.0, pre["stop_time"].max() + 5.0
    print(f"\nStreaming theta channel {THETA_ELECTRODE} for Pre-Cooling window "
          f"{t_start:.1f}-{t_stop:.1f} s ({t_stop - t_start:.1f} s) from raw ecephys file...")

    disk_cache = remfile.DiskCache(DISK_CACHE_DIR)
    rem_file = remfile.File(RAW_URL, disk_cache=disk_cache)
    raw_h5 = h5py.File(rem_file, "r")
    raw_data = raw_h5["acquisition"]["ElectricalSeries"]["data"]

    i0 = int(t_start * RAW_FS)
    i1 = int(t_stop * RAW_FS)
    n_total = i1 - i0
    chunk_samples = 20 * int(RAW_FS)  # stream in 20-s blocks with a progress bar
    blocks = []
    t0 = time.time()
    for start in tqdm(range(i0, i1, chunk_samples), desc="Streaming LFP"):
        stop = min(start + chunk_samples, i1)
        blocks.append(raw_data[start:stop, THETA_ELECTRODE])
    lfp_raw = np.concatenate(blocks).astype(np.float64)
    print(f"Streamed {n_total} samples in {time.time() - t0:.1f} s")
    raw_h5.close()

    lfp_ds = resample_poly(lfp_raw, up=1, down=DECIMATION)
    lfp_t = t_start + np.arange(len(lfp_ds)) / LFP_TARGET_FS
    print(f"Downsampled LFP: {lfp_ds.shape} at {LFP_TARGET_FS} Hz")

    np.savez(
        f"{CACHE_DIR}/session_data.npz",
        pos_xyz=pos_xyz,
        pos_t=pos_t,
        speed=speed,
        speed_t=speed_t,
        lfp=lfp_ds,
        lfp_t=lfp_t,
        lfp_fs=LFP_TARGET_FS,
    )
    trials.to_pickle(f"{CACHE_DIR}/trials.pkl")
    with open(f"{CACHE_DIR}/units.pkl", "wb") as fh:
        pickle.dump(spike_times, fh)

    print("\nSaved cache/session_data.npz, cache/trials.pkl, cache/units.pkl")


if __name__ == "__main__":
    main()
