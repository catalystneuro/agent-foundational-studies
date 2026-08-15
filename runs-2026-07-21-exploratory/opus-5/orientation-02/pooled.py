"""Pool per-session results into a single analysis-ready structure."""

import glob
import os
import pickle

import numpy as np
import pandas as pd


def load_sessions(folder="session_results"):
    out = []
    for fp in sorted(glob.glob(os.path.join(folder, "*.pkl"))):
        with open(fp, "rb") as fh:
            res = pickle.load(fh)
        sp_fp = fp.replace(".pkl", "_running.npz")
        if os.path.exists(sp_fp):
            res["speed"] = np.load(sp_fp)["speed"]
        out.append(res)
    return out


def pool(sessions):
    units = pd.concat([s["units"] for s in sessions])
    tc_dg = np.concatenate([s["tc_dg"] for s in sessions], axis=1)
    tc_sg = np.concatenate([s["tc_sg"] for s in sessions], axis=1)
    assert tc_dg.shape[1] == len(units)
    return dict(units=units.reset_index(), tc_dg=tc_dg, tc_sg=tc_sg,
                n_sessions=len(sessions))
