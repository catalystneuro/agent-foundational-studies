"""Shared streaming loader for DANDI:000044 (Grosmark & Buzsaki 2016, hc-11).

Only the small NWB objects (units table, behavior/position) are actually read
over the network; the multi-GB raw ephys datasets are never touched.
"""

import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap

CACHE_DIR = "/tmp/remfile_cache_000044"

# asset path -> S3 blob URL (resolved from the DANDI API, dandiset 000044 draft)
SESSIONS = {
    "Achilles_10252013": "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae",
    "Achilles_11012013": "https://dandiarchive.s3.amazonaws.com/blobs/75a/14b/75a14b2b-05f2-4bb1-abcb-e2af85579af2",
    "Buddy_06272013":    "https://dandiarchive.s3.amazonaws.com/blobs/98b/25c/98b25cb1-310c-45f7-97cc-669fce2057b7",
    "Cicero_09012014":   "https://dandiarchive.s3.amazonaws.com/blobs/ced/326/ced32609-16ab-4c4c-991f-8fa9f40e9401",
    "Cicero_09102014":   "https://dandiarchive.s3.amazonaws.com/blobs/6b8/206/6b8206bd-2829-48d9-95fc-e713cb3d363a",
    "Cicero_09172014":   "https://dandiarchive.s3.amazonaws.com/blobs/12f/dc6/12fdc641-306b-4c83-8112-8244e4a846e1",
    "Gatsby_08022013":   "https://dandiarchive.s3.amazonaws.com/blobs/810/e9d/810e9d83-a8f3-475a-9a71-634659b1c690",
    "Gatsby_08282013":   "https://dandiarchive.s3.amazonaws.com/blobs/560/a77/560a7710-50a3-4a16-9a5b-d079310eaa1f",
}


def open_session(name):
    """Stream one session. Returns (pynapple NWBFile, pynwb NWBFile, io)."""
    url = SESSIONS[name]
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5f = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5f, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


if __name__ == "__main__":
    nwb, nwbfile, io = open_session("Achilles_10252013")
    print("=" * 78)
    print("session_description:", nwbfile.session_description)
    print("identifier         :", nwbfile.identifier)
    print("session_start_time :", nwbfile.session_start_time)
    print("subject            :", nwbfile.subject)
    print("=" * 78)
    print(nwb)
    print("=" * 78)
    print("units colnames:", nwbfile.units.colnames)
    print("n units:", len(nwbfile.units))
    print("epochs:", nwbfile.epochs.to_dataframe() if nwbfile.epochs is not None else None)
    print("=" * 78)
    for mod_name, mod in nwbfile.processing.items():
        print("processing module:", mod_name)
        for k, v in mod.data_interfaces.items():
            print("   ", k, type(v).__name__)
            if hasattr(v, "spatial_series"):
                for sk, ss in v.spatial_series.items():
                    print("        series:", sk, ss.data.shape, "unit=", ss.unit,
                          "| ts:", None if ss.timestamps is None else ss.timestamps.shape)
