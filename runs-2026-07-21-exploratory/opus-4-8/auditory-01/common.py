"""Shared loading helpers for DANDI:000986 auditory cortex Neuropixels data."""
import json, os, urllib.request
import h5py, remfile
import numpy as np
import pynapple as nap
from pynwb import NWBHDF5IO

DANDISET = "000986"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000986")
_ASSET_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset_urls.json")


def get_asset_urls():
    """Map asset path -> download URL for every NWB file in the dandiset (cached to disk)."""
    if os.path.exists(_ASSET_JSON):
        return json.load(open(_ASSET_JSON))
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
           "/versions/draft/assets/?page_size=200")
    results = json.load(urllib.request.urlopen(url))["results"]
    urls = {a["path"]: (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
                        f"/versions/draft/assets/{a['asset_id']}/download/")
            for a in results}
    json.dump(urls, open(_ASSET_JSON, "w"), indent=1)
    return urls


def open_nwb(asset_path):
    """Stream an NWB file from the DANDI S3 bucket with an on-disk cache."""
    url = get_asset_urls()[asset_path]
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True)
    return io.read()


def load_session(asset_path):
    """Return a dict of pynapple objects plus session metadata for one recording."""
    nwbfile = open_nwb(asset_path)
    trials = nwbfile.trials.to_dataframe()

    units = nap.TsGroup(
        {i: nap.Ts(t) for i, t in enumerate(nwbfile.units["spike_times"][:])}
    )
    tone_on = nap.Ts(trials["start_time"].values)
    beh = nwbfile.processing["behavior"]
    pupil_ts = beh["PupilTracking"]["pupil_diameter"]
    run_ts = beh["running_speed"]

    return dict(
        path=asset_path,
        subject=nwbfile.subject.subject_id,
        session_id=str(nwbfile.session_id),
        units=units,
        trials=trials,
        tone_on=tone_on,
        frequency=trials["stim_frequency"].values,
        amplitude=trials["stim_amplitude"].values,
        duration=float(trials["stim_duration"].iloc[0]),
        pupil=nap.Tsd(t=pupil_ts.timestamps[:], d=pupil_ts.data[:]),
        running=nap.Tsd(t=run_ts.timestamps[:], d=run_ts.data[:]),
        spont=nap.IntervalSet(
            start=nwbfile.intervals["spontaneous_blocks"].to_dataframe()["start_time"].values,
            end=nwbfile.intervals["spontaneous_blocks"].to_dataframe()["stop_time"].values,
        ),
    )


SESSIONS = [
    "sub-LA3/sub-LA3_ses-3_behavior.nwb",
    "sub-LA8/sub-LA8_ses-1_behavior.nwb",
    "sub-LA8/sub-LA8_ses-2_behavior.nwb",
    "sub-LA9/sub-LA9_ses-1_behavior.nwb",
    "sub-LA9/sub-LA9_ses-3_behavior.nwb",
    "sub-LA9/sub-LA9_ses-4_behavior.nwb",
    "sub-LA9/sub-LA9_ses-5_behavior.nwb",
    "sub-LA11/sub-LA11_ses-1_behavior.nwb",
    "sub-LA11/sub-LA11_ses-2_behavior.nwb",
    "sub-LA11/sub-LA11_ses-3_behavior.nwb",
    "sub-LA11/sub-LA11_ses-4_behavior.nwb",
    "sub-LA12/sub-LA12_ses-1_behavior.nwb",
    "sub-LA12/sub-LA12_ses-2_behavior.nwb",
    "sub-LA12/sub-LA12_ses-3_behavior.nwb",
    "sub-LA12/sub-LA12_ses-4_behavior.nwb",
]
