"""
Compute orientation-tuning metrics and their significance for every unit, for both the
drifting-grating and static-grating stimuli.
"""
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from analysis_common import (
    condition_means, gosi_from_means, permutation_test, two_point_indices,
    fit_double_von_mises, N_PERM,
)


# --------------------------------------------------------------------------------------
# Per-session analysis
# --------------------------------------------------------------------------------------
def analyse_session(res, seed=0):
    meta = res["meta"].copy()
    n_units = len(res["unit_ids"])
    assert list(meta["unit_id"]) == list(res["unit_ids"])

    out = {c: meta[c].values for c in ["uid", "session", "unit_id", "area", "area_group", "snr", "waveform_duration"]}

    # ---------------- drifting gratings: direction tuning at the preferred TF ----------
    dg, rates = res["dg_table"], res["dg_rates"]
    dirs = np.sort(dg["orientation"].unique())
    tfs = np.sort(dg["temporal_frequency"].unique())
    tf_lab, dir_lab = dg["temporal_frequency"].values, dg["orientation"].values

    tf_marginal = np.stack([rates[tf_lab == tf].mean(axis=0) for tf in tfs])  # (n_tf, n_units)
    pref_tf_i = np.argmax(tf_marginal, axis=0)
    out["dg_pref_tf"] = tfs[pref_tf_i]
    out["dg_blank_rate"] = res["dg_blank_rates"].mean(axis=0)

    dir_means = np.full((len(dirs), n_units), np.nan)
    gosi = np.full(n_units, np.nan); gp = np.full(n_units, np.nan)
    gz = np.full(n_units, np.nan); gnull = np.full(n_units, np.nan)
    gdsi = np.full(n_units, np.nan); dp = np.full(n_units, np.nan)
    kruskal_p = np.full(n_units, np.nan)
    resp_p = np.full(n_units, np.nan)
    mean_rate = np.full(n_units, np.nan)

    for k, tf in enumerate(tfs):
        cols = np.flatnonzero(pref_tf_i == k)
        if len(cols) == 0:
            continue
        sel = tf_lab == tf
        r_sub, l_sub = rates[np.ix_(sel, cols)], dir_lab[sel]
        m, _ = condition_means(r_sub, l_sub, dirs)
        dir_means[:, cols] = m
        gosi[cols], gp[cols], gz[cols], gnull[cols] = permutation_test(r_sub, l_sub, dirs, 2, seed=seed + k)
        gdsi[cols], dp_k, _ = permutation_test(r_sub, l_sub, dirs, 1, seed=seed + 100 + k)[0:3]
        dp[cols] = dp_k
        groups = [r_sub[l_sub == d] for d in dirs]
        mean_rate[cols] = r_sub.mean(axis=0)
        for jj, c in enumerate(cols):
            kruskal_p[c] = stats.kruskal(*[g[:, jj] for g in groups]).pvalue
            # Responsiveness pools over all directions rather than using each unit's best
            # direction.  Testing the best direction would pre-select units with a large
            # response contrast across directions, which is the very thing the
            # orientation-selectivity test goes on to measure.
            resp_p[c] = stats.mannwhitneyu(
                r_sub[:, jj], res["dg_blank_rates"][:, c], alternative="greater"
            ).pvalue

    out["dg_dir_means"] = dir_means.T          # (n_units, n_dirs)
    out["dg_gOSI"] = gosi
    out["dg_gOSI_p"] = gp
    out["dg_gOSI_z"] = gz
    out["dg_gOSI_null"] = gnull
    out["dg_gDSI"] = gdsi
    out["dg_gDSI_p"] = dp
    out["dg_kruskal_p"] = kruskal_p
    out["dg_resp_p"] = resp_p
    out["dg_mean_rate"] = mean_rate
    osi, pref_ori, peak = two_point_indices(dir_means, dirs, 180)
    dsi, pref_dir, _ = two_point_indices(dir_means, dirs, 360)
    out["dg_OSI"] = osi
    out["dg_DSI"] = dsi
    out["dg_pref_ori"] = pref_ori
    out["dg_pref_dir"] = pref_dir
    out["dg_peak_rate"] = peak

    # Robustness check: same index computed on all temporal frequencies pooled.
    m_all, _ = condition_means(rates, dir_lab, dirs)
    out["dg_gOSI_allTF"], _ = gosi_from_means(m_all, dirs, 2)

    # ---------------- static gratings: orientation tuning at the preferred SF ---------
    sg, srates = res["sg_table"], res["sg_rates"]
    oris = np.sort(sg["orientation"].unique())
    sfs = np.sort(sg["spatial_frequency"].unique())
    sf_lab, ori_lab = sg["spatial_frequency"].values, sg["orientation"].values

    sf_marginal = np.stack([srates[sf_lab == sf].mean(axis=0) for sf in sfs])
    pref_sf_i = np.argmax(sf_marginal, axis=0)
    out["sg_pref_sf"] = sfs[pref_sf_i]
    out["sg_blank_rate"] = res["sg_blank_rates"].mean(axis=0)

    ori_means = np.full((len(oris), n_units), np.nan)
    s_gosi = np.full(n_units, np.nan); s_gp = np.full(n_units, np.nan)
    s_gz = np.full(n_units, np.nan); s_kp = np.full(n_units, np.nan)
    s_resp = np.full(n_units, np.nan)
    s_mean_rate = np.full(n_units, np.nan)

    for k, sf in enumerate(sfs):
        cols = np.flatnonzero(pref_sf_i == k)
        if len(cols) == 0:
            continue
        sel = sf_lab == sf
        r_sub, l_sub = srates[np.ix_(sel, cols)], ori_lab[sel]
        m, _ = condition_means(r_sub, l_sub, oris)
        ori_means[:, cols] = m
        s_gosi[cols], s_gp[cols], s_gz[cols], _ = permutation_test(r_sub, l_sub, oris, 2, seed=seed + 200 + k)
        groups = [r_sub[l_sub == o] for o in oris]
        s_mean_rate[cols] = r_sub.mean(axis=0)
        for jj, c in enumerate(cols):
            s_kp[c] = stats.kruskal(*[g[:, jj] for g in groups]).pvalue
            s_resp[c] = stats.mannwhitneyu(
                r_sub[:, jj], res["sg_blank_rates"][:, c], alternative="greater"
            ).pvalue

    out["sg_ori_means"] = ori_means.T
    out["sg_gOSI"] = s_gosi
    out["sg_gOSI_p"] = s_gp
    out["sg_gOSI_z"] = s_gz
    out["sg_kruskal_p"] = s_kp
    out["sg_resp_p"] = s_resp
    out["sg_mean_rate"] = s_mean_rate
    s_osi, s_pref, s_peak = two_point_indices(ori_means, oris, 180)
    out["sg_OSI"] = s_osi
    out["sg_pref_ori"] = s_pref
    out["sg_peak_rate"] = s_peak

    # ---------------- von Mises tuning width from the drifting-grating curves ---------
    hwhm = np.full(n_units, np.nan)
    vm_r2 = np.full(n_units, np.nan)
    vm_params = np.full((n_units, 5), np.nan)
    for j in range(n_units):
        y = dir_means[:, j]
        if np.all(np.isfinite(y)) and y.max() > 0.5:
            popt, hw, r2 = fit_double_von_mises(dirs, y)
            if popt is not None:
                vm_params[j] = popt
                hwhm[j] = hw
                vm_r2[j] = r2
    out["vm_hwhm"] = hwhm
    out["vm_r2"] = vm_r2

    df = pd.DataFrame({k: v for k, v in out.items() if np.ndim(v) == 1})
    arrays = dict(
        dg_dir_means=out["dg_dir_means"], sg_ori_means=out["sg_ori_means"],
        vm_params=vm_params, dirs=dirs, oris=oris, tfs=tfs, sfs=sfs,
    )
    return df, arrays


if __name__ == "__main__":
    with open("responses.pkl", "rb") as f:
        results = pickle.load(f)

    dfs, arr_by_session = [], {}
    for res in tqdm(results, desc="analysing"):
        df, arrays = analyse_session(res)
        dfs.append(df)
        arr_by_session[res["session"]] = arrays
    units = pd.concat(dfs, ignore_index=True)

    # Response and selectivity criteria (Benjamini-Hochberg not needed: permutation
    # p-values are already conservative and we report proportions, not per-unit claims).
    units["dg_responsive"] = (units["dg_resp_p"] < 0.01) & (units["dg_mean_rate"] > 1.0)
    units["sg_responsive"] = (units["sg_resp_p"] < 0.01) & (units["sg_mean_rate"] > 1.0)
    units["dg_orientation_selective"] = units["dg_responsive"] & (units["dg_gOSI_p"] < 0.05)
    units["sg_orientation_selective"] = units["sg_responsive"] & (units["sg_gOSI_p"] < 0.05)
    units["dg_direction_selective"] = units["dg_responsive"] & (units["dg_gDSI_p"] < 0.05)

    units.to_parquet("unit_metrics.parquet")
    with open("tuning_arrays.pkl", "wb") as f:
        pickle.dump(arr_by_session, f)

    pd.set_option("display.width", 220)
    print("\n", units.groupby("area_group")[["dg_responsive", "dg_orientation_selective",
                                             "sg_responsive", "sg_orientation_selective"]].mean())
    print("\n", units.groupby("area")[["dg_gOSI", "dg_gOSI_z", "sg_gOSI", "sg_gOSI_z"]].mean())
    print("\nn units per area:\n", units["area"].value_counts())
