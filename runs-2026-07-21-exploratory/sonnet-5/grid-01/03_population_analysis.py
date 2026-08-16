import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
import pandas as pd
from tqdm import tqdm

from gridcell_utils import smooth_rate_map, spatial_autocorrelogram, gridness_score

SESSIONS = {
    "sub-10697": "https://dandiarchive.s3.amazonaws.com/blobs/9d7/e0d/9d7e0d8e-a133-46ef-a06f-245789350734",
    "sub-10073": "https://dandiarchive.s3.amazonaws.com/blobs/26a/22c/26a22c31-09bc-43a4-9187-edc7394ed12c",
    "sub-10884": "https://dandiarchive.s3.amazonaws.com/blobs/568/70d/56870d3a-f2a7-4632-8598-010481c07c24",
    "sub-11016": "https://dandiarchive.s3.amazonaws.com/blobs/dbf/0c8/dbf0c8b7-17fc-4d02-ab47-241ea73d1567",
    "sub-11207": "https://dandiarchive.s3.amazonaws.com/blobs/489/7ae/4897aef9-8bce-41e3-849d-f966ad73c9a1",
    "sub-11340": "https://dandiarchive.s3.amazonaws.com/blobs/224/331/22433160-8ee4-4405-86aa-9c5c86020544",
}

N_BINS = 40
N_SHUFFLES = 100
MIN_OCC_TIME = 0.1  # seconds


def load_session(url):
    disk_cache = remfile.DiskCache('cache/remfile_cache')
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, io


def rate_map_and_gridness(spike_ts, pos, ep):
    tc = nap.compute_tuning_curves(nap.TsGroup({0: spike_ts}), pos, bins=N_BINS,
                                    range=[(-50, 50), (-50, 50)], epochs=ep)
    occupancy = tc.attrs["occupancy"]
    fs = tc.attrs["fs"]
    rate_map = tc.sel(unit=0).values.astype(float)
    rate_map = np.where(occupancy < MIN_OCC_TIME * fs, np.nan, rate_map)
    smoothed = smooth_rate_map(rate_map, sigma=1.0)
    autocorr = spatial_autocorrelogram(smoothed)
    g, r_in, r_out, peaks = gridness_score(autocorr)
    return g, smoothed, autocorr


def circular_shift_ts(ts, ep, min_shift=20.0):
    start, end = ep.start[0], ep.end[0]
    duration = end - start
    shift = np.random.uniform(min_shift, duration - min_shift)
    t = ts.t - start
    t_shifted = (t + shift) % duration + start
    return nap.Ts(t=np.sort(t_shifted))


rows = []
example_maps = {}

for sub, url in SESSIONS.items():
    print(f"\n=== Loading {sub} ===")
    nwb, io = load_session(url)
    units = nwb["units"]
    pos = nwb["SpatialSeriesLED1"]
    ep = pos.time_support

    metadata = units.metadata

    for unit_id in tqdm(units.index, desc=f"{sub} units"):
        spike_ts = units[unit_id]
        g_real, smoothed, autocorr = rate_map_and_gridness(spike_ts, pos, ep)

        shuffled_g = np.full(N_SHUFFLES, np.nan)
        for i in range(N_SHUFFLES):
            shifted = circular_shift_ts(spike_ts, ep)
            g_shuf, _, _ = rate_map_and_gridness(shifted, pos, ep)
            shuffled_g[i] = g_shuf

        threshold = np.nanpercentile(shuffled_g, 95)
        is_grid = (not np.isnan(g_real)) and (g_real > threshold)

        meta_row = metadata.loc[unit_id] if unit_id in metadata.index else None
        histology = meta_row["histology"] if meta_row is not None else ""
        depth = meta_row["depth"] if meta_row is not None else np.nan

        rows.append({
            "session": sub,
            "unit_id": unit_id,
            "n_spikes": len(spike_ts),
            "gridness": g_real,
            "shuffle_95pct": threshold,
            "is_grid_cell": is_grid,
            "histology": histology,
            "depth": depth,
        })

        example_maps[(sub, unit_id)] = {
            "smoothed": smoothed,
            "autocorr": autocorr,
            "gridness": g_real,
            "is_grid_cell": is_grid,
        }

    io.close()

df = pd.DataFrame(rows)
df.to_csv("population_gridness_results.csv", index=False)
print(df)
print(f"\nTotal units: {len(df)}")
print(f"Grid cells (gridness > shuffle 95th pct): {df['is_grid_cell'].sum()}")

np.save("example_maps.npy", example_maps, allow_pickle=True)
