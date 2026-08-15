"""Run the grid-cell rate-map / gridness-score pipeline across multiple
sessions of DANDI:000582 (Sargolini et al. 2006) and save results to disk
for later plotting."""

import pickle
import numpy as np
import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
from tqdm import tqdm
from dandi.dandiapi import DandiAPIClient

from grid_utils import smooth_ratemap, autocorrelogram, gridness_score, shuffled_gridness_null

SESSION_PATHS = [
    "sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb",
    "sub-10697/sub-10697_ses-02030402_behavior+ecephys.nwb",
    "sub-10697/sub-10697_ses-24020402_behavior+ecephys.nwb",
    "sub-11278/sub-11278_ses-30080505_behavior+ecephys.nwb",
    "sub-11278/sub-11278_ses-31080502_behavior+ecephys.nwb",
    "sub-10704/sub-10704_ses-06070402_behavior+ecephys.nwb",
    "sub-10884/sub-10884_ses-01080402_behavior+ecephys.nwb",
    "sub-10938/sub-10938_ses-08100401_behavior+ecephys.nwb",
    "sub-10962/sub-10962_ses-27110403_behavior+ecephys.nwb",
    "sub-11016/sub-11016_ses-02020502_behavior+ecephys.nwb",
    "sub-11025/sub-11025_ses-01060511_behavior+ecephys.nwb",
    "sub-11084/sub-11084_ses-01030503_behavior+ecephys.nwb",
    "sub-11138/sub-11138_ses-05040502_behavior+ecephys.nwb",
    "sub-11207/sub-11207_ses-03060501_behavior+ecephys.nwb",
    "sub-11265/sub-11265_ses-01020602_behavior+ecephys.nwb",
    "sub-11340/sub-11340_ses-01120501_behavior+ecephys.nwb",
    "sub-11343/sub-11343_ses-08120502_behavior+ecephys.nwb",
]

BINS = 30
POS_RANGE = [(-50, 50), (-50, 50)]
N_SHUFFLES = 100

rng = np.random.default_rng(42)


def load_session(path, disk_cache):
    client = DandiAPIClient()
    ds = client.get_dandiset("000582", "draft")
    asset = ds.get_asset_by_path(path)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    return nwb, nwbfile, io


def analyze_session(session_path, disk_cache):
    nwb, nwbfile, io = load_session(session_path, disk_cache)
    position = nwb["SpatialSeriesLED1"]
    units = nwb["units"]
    ep = position.time_support
    histology_by_idx = nwbfile.units.to_dataframe()["histology"].to_dict()

    results = []
    tc = nap.compute_tuning_curves(units, position, bins=BINS, range=POS_RANGE, epochs=ep)
    unit_table = nwb["units"]

    for uid in tc.coords["unit"].values:
        ratemap = tc.sel(unit=uid).values
        n_spikes = len(unit_table[uid])
        mean_rate = unit_table[uid].rate

        smooth = smooth_ratemap(ratemap, sigma=1.0)
        if np.all(np.isnan(smooth)):
            continue
        corr = autocorrelogram(smooth)
        score, rot_corrs, annulus = gridness_score(corr)

        null = shuffled_gridness_null(
            unit_table[uid], position, ep, BINS, rng, n_shuffles=N_SHUFFLES
        )
        thresh = np.nanpercentile(null, 95)
        is_grid_cell = (not np.isnan(score)) and (score > thresh)

        results.append(
            {
                "session": session_path,
                "unit_id": uid,
                "histology": histology_by_idx.get(uid, "unknown"),
                "n_spikes": n_spikes,
                "mean_rate": mean_rate,
                "gridness": score,
                "null_95th": thresh,
                "is_grid_cell": is_grid_cell,
                "ratemap": ratemap,
                "smoothed_ratemap": smooth,
                "autocorr": corr,
                "null_scores": null,
            }
        )

    io.close()
    return results


def main():
    disk_cache = remfile.DiskCache("/tmp/remfile_cache")
    all_results = []
    for path in tqdm(SESSION_PATHS, desc="sessions"):
        try:
            res = analyze_session(path, disk_cache)
        except Exception as e:
            print(f"FAILED on {path}: {e}")
            continue
        all_results.extend(res)
        print(f"{path}: {len(res)} units processed")

    with open("population_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"Total units analyzed: {len(all_results)}")
    n_grid = sum(r["is_grid_cell"] for r in all_results)
    print(f"Grid cells (gridness > shuffled 95th pct): {n_grid}")


if __name__ == "__main__":
    main()
