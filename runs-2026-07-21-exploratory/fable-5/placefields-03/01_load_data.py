"""Load and inspect one session from DANDI:000044 (Grosmark & Buzsaki 2016).

Streams the NWB file via LINDI with a local chunk cache so that only the
units table and behavioral data are actually downloaded.
"""

import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

DANDISET = "000044"
ASSET_ID = "5349c68b-c0a7-46c0-9900-cda050722fa4"  # sub-Achilles ses-Achilles-10252013
LINDI_URL = (
    f"https://lindi.neurosift.org/dandi/dandisets/{DANDISET}/assets/{ASSET_ID}/nwb.lindi.json"
)

local_cache = lindi.LocalCache(cache_dir="./lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(LINDI_URL, local_cache=local_cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()

print("=" * 80)
print("RAW NWB")
print("=" * 80)
print("session_description:", nwbfile.session_description)
print("session_id         :", nwbfile.session_id)
print("subject            :", nwbfile.subject)
print("lab / institution  :", nwbfile.lab, "/", nwbfile.institution)
print()
print("acquisition        :", list(nwbfile.acquisition.keys()))
print("processing         :", list(nwbfile.processing.keys()))
for mod_name, mod in nwbfile.processing.items():
    print(f"  [{mod_name}]", list(mod.data_interfaces.keys()))
    for di_name, di in mod.data_interfaces.items():
        print(f"     - {di_name}: {type(di).__name__}")
        if hasattr(di, "spatial_series"):
            for ss_name, ss in di.spatial_series.items():
                print(
                    f"         * {ss_name}: shape={ss.data.shape} unit={ss.unit} "
                    f"ref={ss.reference_frame!r}"
                )
        if hasattr(di, "time_series"):
            for ts_name, ts in di.time_series.items():
                print(f"         * {ts_name}: shape={ts.data.shape} unit={ts.unit}")
print()
print("intervals          :", list(nwbfile.intervals.keys()) if nwbfile.intervals else None)
if nwbfile.intervals:
    for name, tbl in nwbfile.intervals.items():
        print(f"  [{name}] n={len(tbl)} cols={tbl.colnames}")
print()
print("units n =", len(nwbfile.units))
print("units colnames =", nwbfile.units.colnames)
print()
print("electrode groups:", list(nwbfile.electrode_groups.keys())[:10])
print("electrodes cols :", nwbfile.electrodes.colnames if nwbfile.electrodes is not None else None)

print()
print("=" * 80)
print("PYNAPPLE VIEW")
print("=" * 80)
nwb = nap.NWBFile(nwbfile)
print(nwb)
