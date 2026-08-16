"""Survey DANDI:000939 and record what each session contains.

Writes `session_survey.csv`: number of units, number of flagged head-direction cells,
and the amount of REM / NREM scored inside the home cage (the sleep we can analyse).
"""
import numpy as np
import pandas as pd
import h5py
import remfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import hd_io

OUT = "session_survey.csv"


def _overlap(a_start, a_end, b_start, b_end):
    tot = 0.0
    for s, e in zip(a_start, a_end):
        tot += np.maximum(0, np.minimum(e, b_end) - np.maximum(s, b_start)).sum()
    return float(tot)


def _survey_one(item):
    path, url = item
    h5 = h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(hd_io.CACHE_DIR)), "r")
    ep = h5["intervals/epochs"]
    tags = [t.decode() if isinstance(t, bytes) else str(t) for t in ep["tags"][:]]
    home = np.array(["home" in t for t in tags])
    hs, he = ep["start_time"][:], ep["stop_time"][:]
    ss = h5["intervals/sleep_states"]
    states = [x.decode() if isinstance(x, bytes) else str(x) for x in ss["state"][:]]
    st, en = ss["start_time"][:].astype(float), ss["stop_time"][:].astype(float)
    row = dict(path=path, n_units=len(h5["units/id"]),
               n_hd=int(h5["units/is_head_direction"][:].sum()))
    for s in ("rem", "nrem"):
        m = np.array([x == s for x in states])
        row[f"{s}_home"] = round(_overlap(st[m], en[m], hs[home], he[home]), 1) if m.any() else 0.0
    return row


def main(out=OUT):
    assets = hd_io.get_assets()
    with ThreadPoolExecutor(8) as ex:
        rows = list(tqdm(ex.map(_survey_one, sorted(assets.items())), total=len(assets),
                         desc="survey"))
    df = pd.DataFrame(rows).sort_values("n_hd", ascending=False)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    print(main().to_string())
