"""Per-session orientation analysis, returning a tidy per-unit table."""

import warnings

import numpy as np
import pandas as pd

import analysis as A


def analyze_session(sess, n_perm=1000, seed=0):
    """Run the full single-session analysis and return (table, intermediates)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsg = A.build_tsgroup(sess)
        dg_trials, dg_ori, dg_blanks = A.grating_epochs(sess, "dg")
        sg_trials, sg_ori, _ = A.grating_epochs(sess, "sg")

    dg_rates = A.trial_rates(tsg, dg_trials)
    blank_rates = A.trial_rates(tsg, dg_blanks)
    sg_rates = A.trial_rates(tsg, sg_trials)

    tc, sem = A.condition_means(dg_rates, dg_ori, A.DIRECTIONS)
    tc_sg, _ = A.condition_means(sg_rates, sg_ori, A.SG_ORIENTATIONS)

    gosi, p_osi, null_med, null_95 = A.permutation_osi(
        dg_rates, dg_ori, A.DIRECTIONS, n_perm, seed
    )
    tc_a, tc_b = A.split_half_tuning(dg_rates, dg_ori, A.DIRECTIONS, seed)
    pref_a = A.preferred_orientation(tc_a)
    pref_b = A.preferred_orientation(tc_b)

    table = pd.DataFrame(
        {
            "session": str(sess["session"]),
            "unit_id": sess["unit_id"],
            "location": sess["location"],
            "depth": sess["depth"],
            "mean_rate": dg_rates.mean(axis=1),
            "blank_rate": blank_rates.mean(axis=1),
            "gOSI": gosi,
            "gDSI": A.global_dsi(tc),
            "OSI": A.classic_osi(tc),
            "DSI": A.classic_dsi(tc),
            "pref_ori": A.preferred_orientation(tc),
            "pref_dir": A.preferred_direction(tc),
            "pref_ori_half_a": pref_a,
            "pref_ori_half_b": pref_b,
            "p_osi": p_osi,
            "gOSI_null": null_med,
            "gOSI_null95": null_95,
            "p_resp": A.responsiveness(dg_rates, blank_rates),
            "gOSI_sg": A.global_osi(tc_sg, A.SG_ORIENTATIONS),
            "pref_ori_sg": A.preferred_orientation(tc_sg, A.SG_ORIENTATIONS),
        }
    )
    intermediates = dict(
        tsg=tsg,
        dg_trials=dg_trials,
        dg_ori=dg_ori,
        dg_rates=dg_rates,
        dg_counts=np.round(dg_rates * (dg_trials.end - dg_trials.start)[None, :]).astype(int),
        blank_rates=blank_rates,
        tc=tc,
        sem=sem,
        tc_a=tc_a,
        tc_b=tc_b,
        tc_sg=tc_sg,
    )
    return table, intermediates
