"""Compute orientation/direction selectivity for every session and pool the results."""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import dandi_io as dio
import tuning as tn

SESSIONS = [
    "sub-699733573/sub-699733573_ses-715093703.nwb",
    "sub-703279277/sub-703279277_ses-719161530.nwb",
    "sub-716813540/sub-716813540_ses-739448407.nwb",
    "sub-718643564/sub-718643564_ses-737581020.nwb",
    "sub-719828686/sub-719828686_ses-754312389.nwb",
    "sub-723627600/sub-723627600_ses-742951821.nwb",
    "sub-726141242/sub-726141242_ses-750332458.nwb",
    "sub-726298249/sub-726298249_ses-754829445.nwb",
    "sub-738651046/sub-738651046_ses-760693773.nwb",
    "sub-744915196/sub-744915196_ses-762602078.nwb",
]

# Response windows relative to stimulus onset. Drifting gratings last 2 s and are
# separated by 1 s of grey; static gratings run back-to-back at 250 ms, so their
# window is shifted past the ~50 ms cortical response latency.
DG_WINDOW = (0.05, 2.0)
SG_WINDOW = (0.05, 0.30)
N_SHUFFLES = 1000
OUT = "results_selectivity.pkl"


def analyze_session(path, n_shuffles=N_SHUFFLES):
    s = dio.extract_session(path)
    tsg, meta = tn.make_tsgroup(s)
    out = dict(session_id=s["session_id"], subject_id=s["subject_id"], genotype=s["genotype"])

    specs = [
        ("dg", s["drifting_gratings"], "temporal_frequency", DG_WINDOW, False),
        ("sg", s["static_gratings"], "spatial_frequency", SG_WINDOW, True),
    ]
    frames = []
    for tag, table, nuisance, window, ori_only in specs:
        rates = tn.trial_rates(tsg, table, window=window)
        angles, pref_level, curves, grid, tab, r = tn.tuning_by_preferred_condition(
            rates, table, "orientation", nuisance
        )
        sel = tn.selectivity_table(
            angles, curves, n_shuffles=n_shuffles, orientation_only=ori_only,
            trial_data=(tab["orientation"].values, r, tab[nuisance].values, pref_level),
        )
        sel["pref_level"] = pref_level
        sel["blank_rate"] = tn.blank_rate(tsg, table, window)
        # von Mises width of the orientation-folded curve
        widths, fit_r2 = np.full(len(sel), np.nan), np.full(len(sel), np.nan)
        for u in range(curves.shape[1]):
            fit = tn.fit_von_mises(angles, curves[:, u])
            if fit is not None:
                widths[u], fit_r2[u] = fit["hwhm"], fit["r2"]
        sel["hwhm"] = widths
        sel["fit_r2"] = fit_r2
        sel = sel.add_prefix(f"{tag}_")
        frames.append(sel)
        out[f"{tag}_angles"] = angles
        out[f"{tag}_curves"] = curves
        out[f"{tag}_grid"] = grid
        out[f"{tag}_nuisance_levels"] = np.sort(tab[nuisance].unique())

    units = pd.concat([meta.reset_index(drop=True)] + frames, axis=1)
    units["session_id"] = s["session_id"]
    units["subject_id"] = s["subject_id"]
    out["units"] = units
    return out


if __name__ == "__main__":
    results = {}
    if os.path.exists(OUT):
        with open(OUT, "rb") as fh:
            results = pickle.load(fh)
    for path in tqdm(SESSIONS, desc="sessions"):
        sid = path.split("ses-")[1].split(".")[0]
        if sid in results:
            continue
        results[sid] = analyze_session(path)
        with open(OUT, "wb") as fh:
            pickle.dump(results, fh)
        print(f"done {sid}: {len(results[sid]['units'])} units", flush=True)

    pooled = pd.concat([r["units"] for r in results.values()], ignore_index=True)
    pooled.to_pickle("results_pooled_units.pkl")
    print(f"\npooled: {len(pooled)} units from {len(results)} sessions")
    print(pooled.groupby("region_group")[["dg_gOSI", "sg_gOSI", "dg_gDSI"]].median().round(3))
    print(pooled.groupby("region_group")[["dg_p_gOSI", "sg_p_gOSI"]].apply(
        lambda d: (d < 0.01).mean()).round(3))
