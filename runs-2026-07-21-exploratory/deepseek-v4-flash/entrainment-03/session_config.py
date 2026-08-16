"""Session configuration for the IBL (DANDI 000149) theta entrainment analysis.

Each session is one large ecephys+behavior NWB file on DANDI. We stream the
small tables (units, trials, lfp metadata) via the neurosift LINDI index and
read only the chunks of LFP we need.
"""

import lindi

# asset_id -> short label, subject, session
SESSIONS = {
    "f791a116-1e6c-4d6a-a9eb-fe3644737be2": dict(
        label="sub-c6e8125f", subject="c6e8125f-b6c7-4349-b74d-32e8bd606f63",
        ses="aad23144-0e52-4eac-80c5-c4ee2decb198"),
    "31f22c47-1512-4293-b19f-6fa5bd9b7cbf": dict(
        label="sub-92130c1b", subject="92130c1b-4fdb-4acc-86e0-1853d429c41a",
        ses="c7bd79c9-c47e-4ea5-aea3-74dda991b48e"),
    "81169999-c697-4eca-a635-2fd994ac183f": dict(
        label="sub-70bf8cbd", subject="70bf8cbd-d312-4654-a4ea-3a21ea2f541b",
        ses="4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a"),
    "e7fa5ae0-b957-4b24-aa40-fb4c3276d331": dict(
        label="sub-9bebfe0b", subject="9bebfe0b-082e-4d66-aca7-fae29317f708",
        ses="4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b"),
}


def lindi_url(asset_id: str) -> str:
    return (f"https://lindi.neurosift.org/dandi/dandisets/000149/"
            f"assets/{asset_id}/nwb.lindi.json")


_CACHE = None


def local_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = lindi.LocalCache(cache_dir=".lindi_cache")
    return _CACHE


def open_h5py(asset_id: str):
    """Open the LINDI-backed file for an asset id (dixtures h5py API)."""
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url(asset_id),
                                            local_cache=local_cache())
    return f


def open_nwb(asset_id: str):
    """Read the NWB file with pynwb."""
    from pynwb import NWBHDF5IO
    f = lindi.LindiH5pyFile.from_lindi_file(lindi_url(asset_id),
                                            local_cache=local_cache())
    io = NWBHDF5IO(file=f, mode="r")
    nwbfile = io.read()
    return nwbfile, io