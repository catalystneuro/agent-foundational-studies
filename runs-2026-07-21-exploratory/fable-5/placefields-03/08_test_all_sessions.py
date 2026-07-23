"""Run the pipeline across all eight sessions of DANDI:000044."""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import place_cell_lib as pcl

rows = []
for asset_id, subject, label, maze in tqdm(pcl.SESSIONS, desc="sessions"):
    t0 = time.time()
    a = pcl.analyze_session(asset_id, n_shuffles=300, progress=False)
    sess = a["session"]
    dec = pcl.decoding_error(sess, a["exc"], a["moving"]["both"], bin_size=0.25)
    rows.append(dict(
        subject=subject,
        session=label,
        maze=maze,
        extent_m=round(sess["extent"], 2),
        laps=a["laps"]["n_laps"]["rightward"] + a["laps"]["n_laps"]["leftward"],
        run_s=round(a["moving"]["both"].tot_length()),
        n_exc=len(a["exc"]),
        n_inh=len(a["inh"]),
        n_place=int(a["is_place_cell_any"].sum()),
        pct_place=round(100 * a["is_place_cell_any"].mean()),
        si_med=round(float(np.median(a["directions"]["rightward"]["si"])), 2),
        si_inh_med=round(float(np.median(a["inhibitory"]["si"])), 3),
        dec_err_cm=round(float(np.median(dec["error"])) * 100, 1),
        chance_cm=round(float(np.median(dec["chance"])) * 100, 1),
        secs=round(time.time() - t0, 1),
    ))
    a["session"]["io"].close()

df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
df.to_csv("session_summary_prototype.csv", index=False)
