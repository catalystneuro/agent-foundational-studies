"""Stream each selected session once and cache everything the analysis needs.

Downloading is by far the slow step, so trial tables, pre-stimulus spike counts,
sliding-window counts and the pre-stimulus behavioural regressors are all written
to ``cache/`` and every later script works from those.
"""
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import ibl_io as io
import analysis_lib as al

CENTERS = np.round(np.arange(-1.0, 0.61, 0.1), 2)
WIDTH = 0.2
os.makedirs("cache", exist_ok=True)


def extract(sel):
    assets = {a[0]: a for a in io.list_processed_assets()}
    path = sel["path"]
    eid = path.split("_ses-")[1][:36]
    out = f"cache/{sel['subject']}_{eid[:8]}.pkl"
    if os.path.exists(out):
        return out
    hf = io.open_h5(assets[path][2])
    df = al.build_trials(hf)
    tsg = al.load_spikes(hf)
    conv, n_easy = al.check_choice_convention(df)

    ok = al.valid_trials(df)
    d = df[ok].reset_index(drop=True)
    t = d["stim_on"].values
    X_pre = al.window_counts(tsg, t, al.PRE_WIN)
    X_time = np.stack([al.window_counts(tsg, t, (c - WIDTH / 2, c + WIDTH / 2))
                       for c in CENTERS])

    beh = {}
    wt, wv = io.read_wheel(hf)
    beh["wheel_speed"] = al.behaviour_regressor(d, (wt, np.abs(wv)), al.PRE_WIN)
    for key, grp, name in [("pupil", "pupil", "LeftPupilDiameterSmoothed"),
                           ("motion_energy", "motion_energy", "LeftCameraMotionEnergy")]:
        g = hf.get(f"/processing/{grp}/{name}")
        if g is not None and "timestamps" in g:
            beh[key] = al.behaviour_regressor(
                d, (g["timestamps"][:], g["data"][:]), al.PRE_WIN)

    # A short stretch of raw spike times for the raw-data figure.
    keys = list(tsg.keys())
    t0 = float(d["stim_on"].iloc[10]) - 6.0
    raw = {int(k): np.asarray(tsg[k].t)[(np.asarray(tsg[k].t) > t0)
                                        & (np.asarray(tsg[k].t) < t0 + 26)]
           for k in keys[:60]}
    best_unit = None
    wt_seg = (wt > t0 - 2) & (wt < t0 + 30)

    # Peri-stimulus spike times for every unit, for rasters/PSTHs (-1.0 to +0.3 s).
    peri = []
    for k in keys:
        s = np.asarray(tsg[k].t)
        idx = np.searchsorted(s, np.column_stack([t - 1.0, t + 0.3]))
        peri.append([s[a:b] - ti for (a, b), ti in zip(idx, t)])

    rec = dict(
        path=path, eid=eid, subject=sel["subject"], size=sel["size"],
        conv=conv, n_easy=n_easy, n_all_trials=int(len(df)),
        trials=d, X_pre=X_pre, X_time=X_time, centers=CENTERS, width=WIDTH,
        regions=np.asarray(tsg.get_info("region")),
        probes=np.asarray(tsg.get_info("probe")),
        beh=beh, raw_spikes=raw, raw_t0=t0,
        wheel=(wt[wt_seg], wv[wt_seg]),
        peri=peri,
        wheel_profile=al.wheel_profile(d, wt, wv),
    )
    pd.to_pickle(rec, out)
    return out


if __name__ == "__main__":
    sel = json.load(open("selected_sessions.json"))
    for s in tqdm(sel, desc="extracting"):
        p = extract(s)
        r = pd.read_pickle(p)
        print(f"{r['subject']:>14s} trials={len(r['trials']):4d} "
              f"units={r['X_pre'].shape[1]:4d} blocks={r['trials'].block_id.nunique()} "
              f"conv={r['conv']:.2f}", flush=True)
