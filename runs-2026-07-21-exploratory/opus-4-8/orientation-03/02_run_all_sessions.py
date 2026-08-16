"""Run the per-unit orientation analysis across multiple sessions of DANDI:000021."""
import os, time
import pandas as pd
from tqdm import tqdm
import orientation_lib as ol

SESSIONS = ["715093703", "750749662", "754829445", "756029989",
            "763673393", "799864342", "751348571", "755434585"]

screen = pd.read_csv("session_screen.csv").set_index("session_id")
os.makedirs("results", exist_ok=True)
for sid in tqdm(SESSIONS, desc="sessions"):
    out = f"results/units_{sid}.csv"
    if os.path.exists(out):
        continue
    t = time.time()
    df, _ = ol.analyze_session(screen.loc[int(sid), "s3_url"], sid)
    df.to_csv(out, index=False)
    print(f"{sid}: {len(df)} units, {time.time()-t:.0f}s", flush=True)

all_units = pd.concat([pd.read_csv(f"results/units_{s}.csv") for s in SESSIONS], ignore_index=True)
all_units.to_csv("results/units_all_sessions.csv", index=False)
print(all_units.groupby("group").responsive.agg(["sum", "count"]))
