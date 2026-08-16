import lindi, pynapple as nap
from pynwb import NWBHDF5IO

URL = "https://lindi.neurosift.org/dandi/dandisets/000044/assets/5349c68b-c0a7-46c0-9900-cda050722fa4/nwb.lindi.json"
cache = lindi.LocalCache(cache_dir="/tmp/lindi_cache")
f = lindi.LindiH5pyFile.from_lindi_file(URL, local_cache=cache)
io = NWBHDF5IO(file=f, mode="r")
nwbfile = io.read()
print(nwbfile.session_description)
print("subject:", nwbfile.subject)
print("\n--- acquisition:", list(nwbfile.acquisition))
print("--- processing:", list(nwbfile.processing))
for k, m in nwbfile.processing.items():
    print(" module", k, list(m.data_interfaces))
print("--- intervals:", list(nwbfile.intervals))
print("--- units colnames:", nwbfile.units.colnames, len(nwbfile.units.id))
print("--- electrodes colnames:", nwbfile.electrodes.colnames)
print("--- epochs:", nwbfile.epochs.to_dataframe() if nwbfile.epochs is not None else None)
