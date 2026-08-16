"""Load each session, compute peri-onset spike-count tensor + trial table, cache to disk."""
import numpy as np, pandas as pd, os
from tqdm import tqdm
import db_pipeline as P

SESSIONS = {
    "NYU-11": "sub-NYU-11/sub-NYU-11_ses-6713a4a7-faed-4df2-acab-ee4e63326f8d_desc-processed_behavior+ecephys.nwb",
    "NYU-30": "sub-NYU-30/sub-NYU-30_ses-77e6dc6a-66ed-433c-b1a2-778c914f523c_desc-processed_behavior+ecephys.nwb",
    "NYU-37": "sub-NYU-37/sub-NYU-37_ses-21d21fc3-4201-4edc-802a-c67b61952548_desc-processed_behavior+ecephys.nwb",
    "NYU-46": "sub-NYU-46/sub-NYU-46_ses-d32876dd-8303-4720-8e7e-20678dc2fd71_desc-processed_behavior+ecephys.nwb",
    "NYU-40": "sub-NYU-40/sub-NYU-40_ses-8ca740c5-e7fe-430a-aa10-e74e9c3cbbe8_desc-processed_behavior+ecephys.nwb",
    "NYU-39": "sub-NYU-39/sub-NYU-39_ses-91a3353a-2da1-420d-8c7c-fad2fedfdd18_desc-processed_behavior+ecephys.nwb",
}
# Peri-onset bins: 100 ms bins from -1.0 to +0.6 s relative to stimulus onset.
EDGES = np.round(np.arange(-1.0, 0.601, 0.1), 3)
OUT = "cache"

for name, path in SESSIONS.items():
    fT = f"{OUT}/{name}_T.npy"
    if os.path.exists(fT):
        print(name, "cached, skip"); continue
    print("Loading", name)
    nwbfile, df, units = P.load_session(path)
    d = P.clean_trials(df)
    dur = float(units.time_support.tot_length())
    us = P.select_units(units, min_rate=1.0, session_dur=dur)
    print(f"  {name}: {len(d)} trials, {len(us.index)} units (>=1Hz)")
    T = P.peri_onset_tensor(us, d.onset.values, EDGES)
    np.save(fT, T)
    d.to_parquet(f"{OUT}/{name}_d.parquet")
    print(f"  saved T {T.shape}")
np.save(f"{OUT}/edges.npy", EDGES)
print("ALL DONE")
