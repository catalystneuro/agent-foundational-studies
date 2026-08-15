"""Run the per-session pipeline over several DANDI:000021 sessions and cache results."""

import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

import orilib as O
import analysis as A

OUT = "session_results"
N_SESSIONS = int(os.environ.get("N_SESSIONS", "8"))


def main():
    os.makedirs(OUT, exist_ok=True)
    assets = O.get_assets()[:N_SESSIONS]
    for a in tqdm(assets, desc="sessions"):
        sid = a["path"].split("ses-")[1].replace(".nwb", "")
        fp = os.path.join(OUT, f"{sid}.pkl")
        if os.path.exists(fp):
            print("skip", sid, flush=True)
            continue
        print("=== ", a["path"], flush=True)
        res = A.analyze_session(a["url"])
        keep = {
            k: res[k]
            for k in (
                "session_id",
                "units",
                "tc_dg",
                "tc_sg",
                "dg_rates",
                "dg_labels",
                "dg_tf",
                "sg_rates",
                "sg_labels",
            )
        }
        keep["dg_blocks"] = res["dg_table"]["stimulus_block"].values
        with open(fp, "wb") as fh:
            pickle.dump(keep, fh)
        print("wrote", fp, len(res["units"]), "units", flush=True)


if __name__ == "__main__":
    main()
