"""Run the phase-entrainment analysis over every selected session and cache results."""

import os
import pickle

import numpy as np
import pandas as pd

import analysis as an
import theta_lib as tl

OUT = "results"
os.makedirs(OUT, exist_ok=True)

rows, phases_normal, phases_cooled, freqs = [], {}, {}, []

for session in tl.SESSIONS:
    cache = f"{OUT}/{session}.pkl"
    if os.path.exists(cache):
        payload = pickle.load(open(cache, "rb"))
        print(f"{session}: loaded from cache ({len(payload['normal'])} units)")
    else:
        s = an.load_session(session)
        normal = an.phase_lock_stats(s["units"], s["phase_at"], s["run_normal"],
                                     session=session, seed=1)
        cooled = an.phase_lock_stats(s["units"], s["phase_at"], s["run_cooled"],
                                     session=session, seed=2, with_burst=False,
                                     verbose=False) if s["run_cooled"].tot_length() > 60 else []
        payload = dict(
            normal=normal, cooled=cooled,
            freq_normal=an.theta_frequency(s["phase"], s["run_normal"], s["fs"]),
            freq_cooled=an.theta_frequency(s["phase"], s["run_cooled"], s["fs"]),
            run=float(s["run"].tot_length()),
            run_normal=float(s["run_normal"].tot_length()),
            run_cooled=float(s["run_cooled"].tot_length()),
            n_units=len(s["units"]), theta_ref=s["theta_ref_row"], align=s["align"],
        )
        with open(cache, "wb") as f:
            pickle.dump(payload, f)

    cooled_by_unit = {r["unit"]: r for r in payload["cooled"]}
    for r in payload["normal"]:
        row = {k: v for k, v in r.items() if k != "phases"}
        row["cell_type"] = an.classify(r)
        c = cooled_by_unit.get(r["unit"])
        if c is not None:
            row.update(mrl_cooled=c["mrl"], pref_phase_cooled=c["pref_phase"],
                       n_spikes_cooled=c["n_spikes"], rate_cooled=c["rate"],
                       mrl_null_cooled=c["mrl_null_mean"], shuffle_p_cooled=c["shuffle_p"])
            phases_cooled[f"{session}|{r['unit']}"] = c["phases"].astype(np.float32)
        rows.append(row)
        phases_normal[f"{session}|{r['unit']}"] = r["phases"].astype(np.float32)
    freqs.append(dict(session=session, freq_normal=payload["freq_normal"],
                      freq_cooled=payload["freq_cooled"],
                      run_normal=payload["run_normal"], run_cooled=payload["run_cooled"]))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/unit_stats.csv", index=False)
np.savez_compressed(f"{OUT}/spike_phases_normal.npz", **phases_normal)
np.savez_compressed(f"{OUT}/spike_phases_cooled.npz", **phases_cooled)
with open(f"{OUT}/theta_freq.pkl", "wb") as f:
    pickle.dump(freqs, f)

print(f"\n{len(df)} units from {df.session.nunique()} sessions")
print(df.groupby("cell_type").agg(n=("mrl", "size"), mrl=("mrl", "median"),
                                  rate=("rate", "median"), burst=("burst", "median")))
print("significant (shuffle p<0.05):", int((df.shuffle_p < 0.05).sum()), "/", len(df))
