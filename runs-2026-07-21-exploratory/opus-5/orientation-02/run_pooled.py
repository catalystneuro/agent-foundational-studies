"""Pooled multi-session analysis: figures 3-8, decoding, GLM. Writes summary stats."""

import json
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import orilib as O
import pooled as P
import figures as F
import decoding as D
import glm as G
from analysis import DIRECTIONS

sessions = P.load_sessions()
pool = P.pool(sessions)
u = pool["units"]
print(f"{len(sessions)} sessions, {len(u)} QC units")

# ---------------------------------------------------------------- figures 3-5
print(F.fig_population_tuning(pool))
print(F.fig_osi_by_area(pool))
print(F.fig_cross_stimulus(pool))

# ---------------------------------------------------------------- decoding
dec = D.run_decoding(sessions)
dec.to_csv("decoding_results.csv", index=False)
print(F.fig_decoding(dec))

# ---------------------------------------------------------------- GLM
glm_rows, examples = [], []
for res in sessions:
    df, dir_basis = G.fit_session(res, res["speed"])
    glm_rows.append(df)
    if not examples:
        Y = np.rint(res["dg_rates"] * 2.0)
        X_dir, _, _ = G.build_design(res["dg_labels"], res["speed"])
        grid, rate = G.predicted_tuning(Y, X_dir, dir_basis)
        cand = df[df.area.isin(O.VISUAL_CORTEX)].sort_values(
            "d_dir_given_speed", ascending=False)
        for j in cand.index[:4]:
            col = int(np.where(res["units"].index.values == df.loc[j, "unit_id"])[0][0])
            emp = res["tc_dg"][:, col]
            sem = np.array([
                res["dg_rates"][res["dg_labels"] == d, col].std(ddof=1)
                / np.sqrt((res["dg_labels"] == d).sum())
                for d in DIRECTIONS
            ])
            examples.append(dict(unit_id=df.loc[j, "unit_id"], area=df.loc[j, "area"],
                                 emp=emp, sem=sem, grid=grid, fit=rate[:, col],
                                 d=df.loc[j, "d_dir_given_speed"]))
    print("glm done", res["session_id"], flush=True)
glm_df = pd.concat(glm_rows, ignore_index=True)
glm_df.to_csv("glm_results.csv", index=False)
print(F.fig_glm(glm_df, examples))

# ---------------------------------------------------------------- figure 8 + stats
print(F.fig_summary(pool))
u.to_csv("unit_metrics.csv", index=False)

vis = u[u.area.isin(O.VISUAL_CORTEX)]
ctl = u[u.area.isin(O.CONTROL)]
mw = stats.mannwhitneyu(vis.dg_osi.dropna(), ctl.dg_osi.dropna(), alternative="greater")
tuned = vis[vis.dg_p_perm < 0.05]
both = vis[(vis.dg_p_perm < 0.05) & (vis.sg_p_perm < 0.05)]
delta = O.circ_dist_ori(both.dg_pref_ori.values, both.sg_pref_ori.values)
rng = np.random.default_rng(0)
delta_shuf = O.circ_dist_ori(both.dg_pref_ori.values, rng.permutation(both.sg_pref_ori.values))

# cardinal vs oblique preference among tuned cortical units
d_card = np.minimum(O.circ_dist_ori(tuned.dg_pref_ori.values, 0.0),
                    O.circ_dist_ori(tuned.dg_pref_ori.values, 90.0))
frac_cardinal = float((d_card < 22.5).mean())

dec_best = dec[(dec.task == "orientation") & (dec.n_units == dec.n_units.max())]
summary = dict(
    n_sessions=len(sessions),
    n_units=int(len(u)),
    n_visual_cortex=int(len(vis)),
    frac_tuned_visual_cortex=float((vis.dg_p_perm < 0.05).mean()),
    frac_tuned_control=float((ctl.dg_p_perm < 0.05).mean()),
    median_osi_visual=float(vis.dg_osi.median()),
    median_osi_null_visual=float(vis.dg_osi_null_median.median()),
    median_osi_control=float(ctl.dg_osi.median()),
    mannwhitney_U=float(mw.statistic),
    mannwhitney_p=float(mw.pvalue),
    n_both_stimuli=int(len(both)),
    median_delta_pref_ori=float(np.median(delta)),
    median_delta_pref_ori_shuffled=float(np.median(delta_shuf)),
    frac_delta_under_30deg=float((delta < 30).mean()),
    frac_cardinal_preference=frac_cardinal,
    osi_by_area={a: float(u.loc[u.area == a, "dg_osi"].median())
                 for a in O.VISUAL_CORTEX + O.THALAMUS + O.CONTROL
                 if (u.area == a).sum() >= 20},
    frac_tuned_by_area={a: float((u.loc[u.area == a, "dg_p_perm"] < 0.05).mean())
                        for a in O.VISUAL_CORTEX + O.THALAMUS + O.CONTROL
                        if (u.area == a).sum() >= 20},
    decoding_orientation_best={
        a: float(g.acc.mean()) for a, g in
        dec_best[dec_best.kind == "observed"].groupby("area")},
    decoding_orientation_shuffled={
        a: float(g.acc.mean()) for a, g in
        dec_best[dec_best.kind == "shuffled"].groupby("area")},
    glm_frac_positive_dir_given_speed={
        a: float((glm_df.loc[glm_df.area == a, "d_dir_given_speed"] > 0).mean())
        for a in O.VISUAL_CORTEX + O.CONTROL if (glm_df.area == a).sum() >= 20},
    glm_median_d_dir_given_speed={
        a: float(glm_df.loc[glm_df.area == a, "d_dir_given_speed"].median())
        for a in O.VISUAL_CORTEX + O.CONTROL if (glm_df.area == a).sum() >= 20},
)
json.dump(summary, open("summary_stats.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
