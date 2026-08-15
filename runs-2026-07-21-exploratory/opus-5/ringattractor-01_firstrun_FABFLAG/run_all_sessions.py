"""Run the ring-attractor analysis over several DANDI:000056 sessions.

Writes `results_multisession.pkl`, a compact per-session summary used by the
consolidated notebook. Streaming plus analysis takes a few minutes per session.
"""
import pickle
import sys

import numpy as np
from tqdm import tqdm

import analysis as A

SESSIONS = [
    "Mouse12-120807",
    "Mouse17-130130",
    "Mouse20-130515",
    "Mouse24-131216",
    "Mouse25-140124",
    "Mouse28-140313",
    "Mouse28-140317",
    "Mouse25-140130",
]

# Not every asset in the dandiset carries both LED tracking and sleep scoring.
# Sessions missing either cannot support this analysis and are listed here so the
# exclusion is explicit rather than silent.
EXCLUDED = {
    "Mouse32-140820": "no behavior processing module: no tracking, no sleep scoring",
    "Mouse17-130201": "behavior module has sleep scoring but no SubjectPosition",
}


def summarise(res):
    """Keep only the small arrays needed for the population figures."""
    if not res.get("ok"):
        return dict(name=res["name"], ok=False, n_units=res["n_units"], n_hd=res["n_hd"])
    out = dict(name=res["name"], ok=True, n_units=res["n_units"], n_hd=res["n_hd"],
               epoch_dur=res["epoch_dur"], corr_similarity=res["corr_similarity"],
               pd_hd=res["sel"]["pd_hd"], mvl=res["sel"]["mvl"], states={})
    for st, d in res["states"].items():
        out["states"][st] = dict(
            r_cos=d["r_cos"], profile=d["profile"], dpd=d["dpd"], cvals=d["cvals"],
            align_R=d["align"]["R"], align_shuffle=d["align_shuffle"],
            ring_index=d["ring_score"]["ring_index"],
            ring_index_shuffle=d["ring_score_shuffle"],
            conc_mean=d["conc_mean"], conc_shuffle=d["conc_shuffle"],
            conc_hist=np.histogram(d["conc"], bins=50, range=(0, 1))[0],
            drift_speed=d["drift_speed"],
            split_half=d["split_half"], split_half_shuffle=d["split_half_shuffle"],
            ang_vel_hist=np.histogram(np.degrees(d["ang_vel"]), bins=101,
                                      range=(-1500, 1500))[0],
            mvl_int=d["mvl_int"], pd_int=d["pd_int"],
            decode_err_med_deg=d.get("decode_err_med_deg"),
        )
    return out


def main(sessions=SESSIONS, out="results_multisession.pkl"):
    allres = []
    for s in tqdm(sessions, desc="sessions"):
        res, _ = A.analyse_session(s, n_shuffle_mvl=100, n_emb_shuffle=5)
        allres.append(summarise(res))
        with open(out, "wb") as f:
            pickle.dump(allres, f)
    return allres


if __name__ == "__main__":
    main(sys.argv[1:] or SESSIONS)
