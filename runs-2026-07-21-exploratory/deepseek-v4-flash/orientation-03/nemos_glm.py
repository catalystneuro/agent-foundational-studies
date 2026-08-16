"""nemos GLM encoding of orientation.

For each VISp good unit, fit a Poisson GLM predicting each drifting-grating
presentation's spike count from orientation features [cos2theta, sin2theta]
(full model) and compare held-out deviance against an intercept-only null
model. Pseudo-R2 = 1 - LL_full/LL_null quantifies how much single-trial firing
is explained by orientation. We also save the fitted orientation coefficient
vector beta=(beta_cos, beta_sin), whose magnitude (tuning "gain") is a
regression-based measure of orientation encoding.
"""
import numpy as np
import pynapple as nap
from sklearn.model_selection import KFold
import nemos as nm
from analysis_core import load_nwb, count_in_intervals

ASSET = "58703c97-c0a9-4736-b684-73c85c1a444a"


def main():
    nwbfile = load_nwb(ASSET)
    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']
    el = nwbfile.electrodes
    loc_arr = np.asarray(el['location'][:])
    loc_series = {int(i): (loc_arr[idx] if loc_arr[idx] else 'unknown')
                  for idx, i in enumerate(el.id[:])}
    peak = np.asarray(units['peak_channel_id'].values, dtype=int)
    unit_area = np.array([loc_series.get(int(p), 'unknown') for p in peak])
    quality = np.asarray(units['quality'].values)

    dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
    starts = dg['start_time'].values; stops = dg['stop_time'].values
    ori_raw = dg['orientation'].values.astype(float)
    inv = nwbfile.intervals['invalid_times'].to_dataframe()
    overlap = np.zeros(len(dg), dtype=bool)
    for _, r in inv.iterrows():
        overlap |= (starts < r['stop_time']) & (stops > r['start_time'])
    valid = ~overlap
    vs = starts[valid]; vst = stops[valid]
    vori = ori_raw[valid]
    nb = ~np.isnan(vori)
    vori_f = np.mod(vori[nb], 180.0)
    dur = (vst - vs)[nb]

    th = np.deg2rad(vori_f)
    X = np.stack([np.cos(2 * th), np.sin(2 * th)], axis=1).astype(np.float32)
    X0 = np.ones((X.shape[0], 1), dtype=np.float32)   # intercept-only null design
    unit_ids = np.array(list(units.keys()))
    visp_idx = np.where((unit_area == 'VISp') & (quality == 'good'))[0]

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    r2_pseudo = {}
    betas = {}
    n_analyzed = 0
    for ui in visp_idx:
        grp = unit_ids[ui]
        ts = units[grp].t
        y = count_in_intervals(ts, vs[nb], vst[nb]) * dur  # count per trial
        y = np.asarray(y, dtype=np.float32)
        if y.sum() < 50:   # need enough spikes for a stable fit
            continue
        ll_full = ll_null = 0.0
        n_obs = 0
        for tr, te in kf.split(X):
            gl = nm.glm.GLM(regularizer='Ridge', regularizer_strength=1e-5)
            gl.fit(X[tr], y[tr])
            ll_full += gl.score(X[te], y[te], score_type='log-likelihood') * len(te)
            gl0 = nm.glm.GLM(regularizer='Ridge', regularizer_strength=1e-5)
            gl0.fit(X0[tr], y[tr])
            ll_null += gl0.score(X0[te], y[te], score_type='log-likelihood') * len(te)
            n_obs += len(te)
        ll_full /= n_obs; ll_null /= n_obs
        r2 = 1.0 - ll_full / ll_null if ll_null < 0 else 0.0
        r2_pseudo[ui] = r2
        betas[ui] = np.asarray(gl.coef_, dtype=float)
        n_analyzed += 1
    print("analyzed", n_analyzed, "VISp good units")

    res = np.load('results.npz')
    gosi = res['gosi']; peak_ab = res['peak_above_blank']; pv = res['p_value']
    selective = (pv < 0.05) & (peak_ab >= 1.0)
    sel_r2 = np.array([r2_pseudo[ui] for ui in visp_idx if ui in r2_pseudo and selective[ui]])
    non_r2 = np.array([r2_pseudo[ui] for ui in visp_idx if ui in r2_pseudo and not selective[ui]])
    print("selective n", len(sel_r2), "pseudoR2 mean", sel_r2.mean() if len(sel_r2) else np.nan)
    print("non-selective n", len(non_r2), "pseudoR2 mean", non_r2.mean() if len(non_r2) else np.nan)

    # regression-based orientation gain |beta| and its correlation with gOSI
    beta_arr = np.array([betas.get(i, [np.nan, np.nan]) for i in visp_idx])
    gain = np.linalg.norm(beta_arr, axis=1)
    nbet = ~np.isnan(gain) & ~np.isnan(gosi[visp_idx])
    from scipy.stats import pearsonr
    rho, pcorr = pearsonr(gain[nbet], gosi[visp_idx][nbet])
    print(f"corr(|beta|, gOSI) among VISp good: r={rho:.3f}, p={pcorr:.2e}")

    np.savez_compressed('nemos_r2.npz', visp_idx=visp_idx,
                        r2=np.array([r2_pseudo.get(i, np.nan) for i in visp_idx]),
                        beta=beta_arr,
                        selective=selective[visp_idx],
                        corr_gain_gosi=rho)
    print("saved nemos_r2.npz")


if __name__ == "__main__":
    main()
