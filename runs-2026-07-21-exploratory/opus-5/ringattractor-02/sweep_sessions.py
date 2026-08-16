"""Run the ring-attractor metrics on every usable session of DANDI:000939.

Produces `cross_session_metrics.csv`, one row per (session, state, condition),
where condition is 'real' or 'shuffle' (per-cell circular shift of spike times).
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import hd_io
import hd_core as hc

OUT = "cross_session_metrics.csv"
MIN_HD_CELLS = 20
MIN_REM = 100.0     # seconds of REM inside the home cage


def analyse_session(path, n_shuffle=1, seed=0):
    d = hd_io.load_session(path)
    beh = hc.behaviour_epoch(d)
    sleep = hc.sleep_epochs(d)
    tc, st = hc.select_hd_cells(d, beh)
    keep = st.index[st["keep"]].values
    if len(keep) < MIN_HD_CELLS:
        return []
    units = d["units"][list(keep)]
    tck = tc[keep]
    pref = st.loc[keep, "pref"].values
    hd = hc.clean_hd(d["hd"])
    rng = np.random.default_rng(seed)

    states = [("wake", beh, 0)] + [(s, sleep[s], hc.EMBED["nrem_activity_pct"] if s == "nrem" else 0)
                                   for s in ("rem", "nrem") if s in sleep]
    rows = []
    for name, ep, pct in states:
        if float(ep.tot_length()) < 60:
            continue
        for cond in ["real"] + [f"shuffle{i}" for i in range(n_shuffle)]:
            U = units if cond == "real" else hc.shift_shuffle(units, ep, rng)
            Y, cnt, mask = hc.population_matrix(U, ep, activity_pct=pct)
            emb, sel = hc.isomap_embed(Y, n_components=2,
                                       n_neighbors=hc.EMBED["n_neighbors"],
                                       max_points=hc.EMBED["max_points"])
            rm = hc.ring_metrics(emb)
            ph = hc.ph_h1(Y)
            dec, prob = hc.decode(tck, U, ep, hc.EMBED["bin_size"])
            dv = np.mod(dec.values[mask], hc.TWOPI)
            cc_iso = hc.best_circ_align(rm["angle"], dv[sel])[0]
            sh = hc.split_half_decode(tck, U, ep, hc.EMBED["bin_size"], activity_mask=mask)
            speed = hc.angular_speed(dec, hc.EMBED["bin_size"])
            # internal (behaviour-free) tuning curves against the manifold angle
            counts = cnt.values[mask][sel]
            itc = hc.internal_tuning(counts, rm["angle"])
            ipref = hc.tuning_stats(itc.rename(columns=dict(enumerate(tck.columns))))["pref"].values
            cc_pref = hc.best_circ_align(ipref, pref)[0]
            _, corroff, _ = hc.pair_corr_vs_offset(Y, pref)
            row = dict(session=d["session"], subject=d["subject"], state=name, cond=cond,
                       n_cells=len(keep), dur=float(ep.tot_length()), n_bins=len(Y),
                       ring_score=rm["ring_score"], hole=rm["hole"], ang_unif=rm["ang_unif"],
                       h1_top=ph["top"], h1_ratio=ph["ratio"],
                       cc_iso_decoded=cc_iso, cc_internal_pref=cc_pref,
                       split_cc=sh["cc"], split_err_deg=np.degrees(sh["median_abs_err"]),
                       med_speed=float(np.median(speed)),
                       corr_near=float(np.nanmean(corroff[:3])),
                       corr_far=float(np.nanmean(corroff[-3:])))
            if name == "wake" and cond == "real":
                true = np.mod(np.angle(np.interp(dec.t, hd.t, np.cos(hd.d))
                                       + 1j * np.interp(dec.t, hd.t, np.sin(hd.d))), hc.TWOPI)
                row["decode_err_deg"] = float(np.degrees(np.median(np.abs(hc.circ_dist(
                    np.mod(dec.values, hc.TWOPI), true)))))
            rows.append(row)
    d["io"].close()
    return rows


def session_list():
    if not os.path.exists("session_survey.csv"):
        import survey_sessions
        survey_sessions.main()
    surv = pd.read_csv("session_survey.csv")
    ok = surv[(surv["n_hd"] >= 25) & (surv["rem_home"] >= MIN_REM)]
    return list(ok.sort_values("n_hd", ascending=False)["path"])


def main(paths=None, out=OUT):
    paths = paths or session_list()
    rows = []
    for p in tqdm(paths, desc="sessions"):
        rows += analyse_session(p)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}: {len(rows)} rows, {len(set(r['session'] for r in rows))} sessions")


if __name__ == "__main__":
    main()
