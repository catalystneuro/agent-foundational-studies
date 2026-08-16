"""Split-half tuning curves: align on one half of the trials, read out on the other.

Aligning a tuning curve to its own peak guarantees a peak even for pure noise, so
the population-average "tuning curve" of an untuned region looks tuned. Choosing
the preferred orientation on half the trials and measuring the curve on the held-out
half removes that bias and makes the region comparison interpretable.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import dandi_io as dio
import tuning as tn
from run_analysis import SESSIONS, SG_WINDOW

RNG = np.random.default_rng(1)


def split_half_curves(session, tsgroup):
    sg = session["static_gratings"]
    keep = sg[["orientation", "spatial_frequency"]].notna().all(axis=1).values
    tab = sg.loc[keep].reset_index(drop=True)
    rates = tn.trial_rates(tsgroup, tab, window=SG_WINDOW)
    angles = np.sort(tab["orientation"].unique())

    # split trials of each orientation into two halves
    half = np.zeros(len(tab), dtype=int)
    for a in angles:
        idx = np.where(tab["orientation"].values == a)[0]
        half[RNG.permutation(idx)[: len(idx) // 2]] = 1

    def mean_curve(sel):
        return np.stack([rates[sel & (tab["orientation"].values == a)].mean(axis=0) for a in angles])

    c_a = mean_curve(half == 0)
    c_b = mean_curve(half == 1)
    pref = np.argmax(c_a, axis=0)  # preferred orientation from half A
    # roll half B so the half-A preferred orientation is at index 0, then
    # normalise by the unit's mean rate so untuned units sit flat at 1
    rolled = np.stack([np.roll(c_b[:, u], -pref[u]) for u in range(c_b.shape[1])])
    denom = rolled.mean(axis=1, keepdims=True)
    return angles, np.divide(rolled, denom, out=np.full_like(rolled, np.nan), where=denom > 0)


if __name__ == "__main__":
    rows, curves = [], []
    for path in tqdm(SESSIONS, desc="split-half"):
        s = dio.extract_session(path)
        tsg, meta = tn.make_tsgroup(s)
        angles, c = split_half_curves(s, tsg)
        curves.append(c)
        rows.append(meta.assign(session_id=s["session_id"]))
    out = dict(angles=angles, curves=np.concatenate(curves, axis=0),
               meta=pd.concat(rows, ignore_index=True))
    pd.to_pickle(out, "results_split_half.pkl")
    m = out["meta"]
    c = out["curves"]
    for grp in ["visual cortex", "visual thalamus", "hippocampus"]:
        sel = (m.region_group == grp).values
        mod = np.nanmax(c[sel], axis=1) - np.nanmin(c[sel], axis=1)
        print(f"{grp:16s} n={sel.sum():5d}  held-out modulation depth "
              f"(median) {np.nanmedian(mod):.3f}")
