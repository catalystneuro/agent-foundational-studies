import lindi, pynapple as nap
from pynwb import NWBHDF5IO

asset_id = "b8d3abca-0e78-4df1-9a51-d122a383be63"  # sub-LA11 ses-2
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{asset_id}/nwb.lindi.json"
local_cache = lindi.LocalCache()
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url, local_cache=local_cache)
io = NWBHDF5IO(file=f)
nwbfile = io.read()
nwb = nap.NWBFile(nwbfile)
print(nwb)
print()
print("trials columns:", nwbfile.trials.colnames)
tr = nwbfile.trials.to_dataframe()
print(tr.head())
print("n trials:", len(tr))
print("freqs:", sorted(tr['stim_frequency'].unique()))
print("n units:", len(nwbfile.units))
