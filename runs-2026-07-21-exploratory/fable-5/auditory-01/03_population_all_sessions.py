"""Run the tuning analysis on all 15 sessions of DANDI 000986 and pool the units."""

import numpy as np
import pandas as pd
from tqdm import tqdm

import dandi_auditory as da
from analysis_core import analyze_session

OUT = "population_results.npz"


if __name__ == "__main__":
    assets = da.list_assets()
    per_session = []
    for _, row in tqdm(list(assets.iterrows()), desc="sessions"):
        r = analyze_session(row["asset_id"])
        r["path"] = row["path"]
        per_session.append(r)
        print(f"  {row['path']}: {r['n_units']} units, "
              f"{np.sum(r['responsive_p'] < 0.01)} responsive, "
              f"{np.sum(r['tuned_p'] < 0.01)} tuned")

    # pool units across sessions
    def cat(key, axis=-1):
        return np.concatenate([r[key] for r in per_session], axis=axis)

    pooled = dict(
        freqs=per_session[0]["freqs"],
        tuning=cat("tuning"), tuning_sem=cat("tuning_sem"), tuning_even=cat("tuning_even"),
        tuning_abs=cat("tuning_abs"),
        psth=cat("psth"), bins=da.psth_bin_centers(),
        responsive_p=cat("responsive_p", 0), tuned_p=cat("tuned_p", 0),
        bf=cat("bf", 0), bf_odd=cat("bf_odd", 0), bf_even=cat("bf_even", 0),
        baseline=cat("baseline", 0), evoked=cat("evoked", 0), sparse=cat("sparseness", 0),
        subject=np.concatenate([[r["subject"]] * r["n_units"] for r in per_session]),
        session=np.concatenate([[r["path"]] * r["n_units"] for r in per_session]),
    )
    np.savez(OUT, **pooled)

    summary = pd.DataFrame([
        {"path": r["path"], "subject": r["subject"], "n_units": r["n_units"],
         "n_responsive": int(np.sum(r["responsive_p"] < 0.01)),
         "n_tuned": int(np.sum(r["tuned_p"] < 0.01))}
        for r in per_session
    ])
    summary.to_csv("session_summary.csv", index=False)
    print(summary.to_string())
    print(f"\npooled: {len(pooled['bf'])} units, "
          f"{np.sum(pooled['responsive_p'] < 0.01)} responsive, "
          f"{np.sum(pooled['tuned_p'] < 0.01)} frequency-tuned")
