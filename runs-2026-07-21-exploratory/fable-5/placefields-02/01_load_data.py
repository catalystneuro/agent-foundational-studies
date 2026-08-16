"""Load one session of DANDI:000044 via LINDI and inspect its structure."""
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET = "5349c68b-c0a7-46c0-9900-cda050722fa4"  # sub-Achilles ses-Achilles-10252013
URL = f"https://lindi.neurosift.org/dandi/dandisets/000044/assets/{ASSET}/nwb.lindi.json"

cache = lindi.LocalCache(cache_dir="/tmp/lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()

print("session_description:", nwbfile.session_description)
print("session_start_time :", nwbfile.session_start_time)
print("subject            :", nwbfile.subject)
print()
print("=== acquisition ===");         print(list(nwbfile.acquisition))
print("=== processing ===")
for name, mod in nwbfile.processing.items():
    print(f"  [{name}]", list(mod.data_interfaces))
    for dname, di in mod.data_interfaces.items():
        print(f"     - {dname}: {type(di).__name__}")
        if hasattr(di, "spatial_series"):
            for sname, ss in di.spatial_series.items():
                print(f"         * {sname}: data{ss.data.shape} unit={ss.unit} "
                      f"rate={getattr(ss,'rate',None)} ts={ss.timestamps is not None}")
print("=== intervals ===")
for name, iv in (nwbfile.intervals or {}).items():
    print(" ", name, iv.colnames, len(iv))
print("=== units ===")
print("n units:", len(nwbfile.units))
print("cols:", nwbfile.units.colnames)
print("=== electrodes ===")
print(nwbfile.electrodes.colnames)

nwb = nap.NWBFile(nwbfile)
print()
print(nwb)
