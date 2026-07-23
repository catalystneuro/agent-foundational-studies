"""Streaming access helpers for DANDI:000021 (Allen Visual Coding - Neuropixels).

The session NWB files are ~2-3 GB each and are read over HTTP with remfile +
a local disk cache, so only the byte ranges we actually touch are transferred.
"""

import os

import h5py
import numpy as np
import remfile

DANDISET = "000021"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000021")

# Session NWB assets (probe/LFP files excluded). asset_id -> path
SESSION_ASSETS = [
    ("58703c97-c0a9-4736-b684-73c85c1a444a", "sub-699733573_ses-715093703"),
    ("02291b99-e583-498b-9929-b68bba2c50e2", "sub-703279277_ses-719161530"),
    ("224b57e5-c9a3-46ef-85db-966713f3ccbe", "sub-707296975_ses-721123822"),
    ("c7f32379-adce-4961-9211-d07790be9cab", "sub-716813540_ses-739448407"),
    ("b4aeeb19-cdc6-4895-ab7b-bc8a688cf6f5", "sub-717038285_ses-732592105"),
    ("106e84a7-1c41-43f1-a675-7df17e4aba69", "sub-718643564_ses-737581020"),
    ("eb36f94f-d6e7-45c6-aa02-7d4ed23453d3", "sub-719817799_ses-744228101"),
    ("5a58bf3d-a1b9-444b-8ab0-ef5478aa42a6", "sub-719828686_ses-754312389"),
    ("522e8054-34ca-4579-ae80-350d0b24e0f4", "sub-722882751_ses-743475441"),
    ("96c200cf-29c2-457a-b2f3-99f11de5b039", "sub-723627600_ses-742951821"),
]


def asset_url(asset_id):
    return (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
        f"/versions/draft/assets/{asset_id}/download/"
    )


def open_h5(asset_id):
    """Open a DANDI asset as an h5py.File via cached HTTP range reads."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    disk_cache = remfile.DiskCache(CACHE_DIR)
    rem = remfile.File(asset_url(asset_id), disk_cache=disk_cache)
    return h5py.File(rem, "r")


def read_ragged(group, name, rows):
    """Read selected rows of an NWB ragged (VectorIndex-backed) column.

    Avoids pulling the entire spike_times dataset (~10^8 values per session)
    when only a few hundred units are of interest.
    """
    index = group[f"{name}_index"][:]
    data = group[name]
    starts = np.concatenate([[0], index[:-1]])
    out = {}
    for r in rows:
        out[r] = data[starts[r] : index[r]]
    return out
