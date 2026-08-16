"""Load one session of DANDI:000021 over remfile and print its structure."""
import json
import sys

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

SESSIONS = {s["session"]: s["url"] for s in json.load(open("sessions_000021.json"))}
CACHE = remfile.DiskCache("/tmp/remfile_cache")


def open_session(session_id):
    rf = remfile.File(SESSIONS[session_id], disk_cache=CACHE)
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    return io.read()


if __name__ == "__main__":
    ses = sys.argv[1] if len(sys.argv) > 1 else "715093703"
    nwbfile = open_session(ses)
    print("session_id:", nwbfile.session_id, "| description:", nwbfile.session_description)
    print("subject:", nwbfile.subject)

    print("\n=== intervals ===")
    for name, iv in nwbfile.intervals.items():
        print(f"  {name:45s} n={len(iv):6d} cols={list(iv.colnames)}")

    print("\n=== units table ===")
    units = nwbfile.units
    print("  n units:", len(units))
    print("  colnames:", list(units.colnames))

    print("\n=== electrodes ===")
    print("  colnames:", list(nwbfile.electrodes.colnames))
    locs = np.asarray(nwbfile.electrodes["location"][:])
    u, c = np.unique(locs, return_counts=True)
    print("  locations:", dict(zip(u.tolist(), c.tolist())))

    print("\n=== processing / acquisition ===")
    for name, mod in nwbfile.processing.items():
        print("  processing:", name, list(mod.data_interfaces))
    print("  acquisition:", list(nwbfile.acquisition))

    print("\n=== drifting gratings table head ===")
    dg = nwbfile.intervals["drifting_gratings_presentations"]
    df = dg.to_dataframe()
    print(df.head(8).to_string())
    for col in ("orientation", "temporal_frequency", "contrast", "spatial_frequency"):
        if col in df.columns:
            print(f"  unique {col}:", sorted(set(map(str, df[col].values))))

    print("\n=== pynapple view ===")
    nwb = nap.NWBFile(nwbfile)
    print(nwb)
