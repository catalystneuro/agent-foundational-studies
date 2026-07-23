"""Core computations for orientation/direction tuning in Allen Visual Coding sessions."""

import jax

jax.config.update("jax_enable_x64", True)  # GLM fits need float64 to converge

import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats
from tqdm.auto import tqdm

VIS_AREAS = ["VISp", "VISl", "VISrl", "VISal", "VISam", "VISpm"]

QC = dict(snr=1.0, isi_violations=0.5, presence_ratio=0.9, amplitude_cutoff=0.1, firing_rate=0.5)


def unit_table(nwb):
    """Unit metadata joined to the electrode table (brain area of the peak channel)."""
    cols = ["peak_channel_id", "snr", "isi_violations", "presence_ratio", "firing_rate",
            "amplitude_cutoff", "quality", "waveform_duration"]
    df = pd.DataFrame({c: nwb.units[c].data[:] for c in cols}, index=nwb.units.id.data[:])
    df["quality"] = df["quality"].astype(str)
    df["row"] = np.arange(len(df))
    el = nwb.electrodes.to_dataframe()
    df["location"] = el["location"].reindex(df["peak_channel_id"]).values
    df["probe"] = el["group_name"].reindex(df["peak_channel_id"]).values
    return df


def select_units(units, areas=VIS_AREAS):
    keep = (
        (units["quality"] == "good")
        & (units["snr"] > QC["snr"])
        & (units["isi_violations"] < QC["isi_violations"])
        & (units["presence_ratio"] > QC["presence_ratio"])
        & (units["amplitude_cutoff"] < QC["amplitude_cutoff"])
        & (units["firing_rate"] > QC["firing_rate"])
        & units["location"].isin(areas)
    )
    return units[keep]


def load_spikes(nwb, sel, desc="loading spikes"):
    """Stream spike times for the selected units into a pynapple TsGroup."""
    spikes = {}
    for uid, row in tqdm(list(zip(sel.index, sel["row"])), desc=desc):
        spikes[int(uid)] = nap.Ts(t=np.asarray(nwb.units["spike_times"][int(row)]))
    tsg = nap.TsGroup(spikes)
    tsg.set_info(area=sel["location"].values, probe=sel["probe"].values)
    return tsg


def stim_table(nwb, name):
    """Stimulus presentation table with numeric orientation / TF / SF columns."""
    tbl = nwb.intervals[name]
    out = pd.DataFrame({
        "start_time": tbl["start_time"].data[:],
        "stop_time": tbl["stop_time"].data[:],
    })
    for c in ["orientation", "temporal_frequency", "spatial_frequency", "phase", "contrast"]:
        if c in tbl.colnames:
            out[c] = pd.to_numeric(pd.Series(np.asarray(tbl[c].data[:], dtype=object)
                                             .astype(str)), errors="coerce").values
    out["duration"] = out["stop_time"] - out["start_time"]
    return out


def trial_rates(tsgroup, trials):
    """Firing rate (Hz) of every unit on every trial. Rows = trials, columns = unit ids."""
    ep = nap.IntervalSet(start=trials["start_time"].values, end=trials["stop_time"].values)
    counts = np.asarray(tsgroup.count(ep=ep))
    return pd.DataFrame(counts / trials["duration"].values[:, None],
                        index=trials.index, columns=list(tsgroup.keys()))


def circular_selectivity(theta_deg, rates, harmonic):
    """Vector-strength selectivity index. harmonic=2 -> gOSI (orientation), 1 -> gDSI."""
    theta = np.deg2rad(theta_deg)
    r = np.clip(np.asarray(rates, dtype=float), 0, None)
    denom = r.sum(axis=-1)
    vec = (r * np.exp(1j * harmonic * theta)).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = np.abs(vec) / denom
        ang = np.rad2deg(np.angle(vec) / harmonic) % (360 / harmonic)
    return np.where(denom > 0, mag, np.nan), np.where(denom > 0, ang, np.nan)


def ratio_index(theta_deg, rates, harmonic):
    """Classical (R_pref - R_opposite) / (R_pref + R_opposite).

    harmonic=2 compares the preferred direction with the orthogonal orientation (OSI);
    harmonic=1 compares it with the opposite direction (DSI).
    """
    rates = np.asarray(rates, dtype=float)
    period = 180.0 / harmonic  # 90 deg away for OSI, 180 deg away for DSI
    full = 360.0 if theta_deg.max() > 180 else 180.0  # sampled range of the stimulus
    idx = np.nanargmax(rates, axis=-1)
    pref = theta_deg[idx]
    other = (pref + period) % full
    j = np.array([int(np.argmin(np.abs(theta_deg - o))) for o in np.atleast_1d(other)])
    r_pref = np.take_along_axis(rates, np.atleast_1d(idx)[:, None], axis=-1)[:, 0]
    r_other = np.take_along_axis(rates, j[:, None], axis=-1)[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r_pref - r_other) / (r_pref + r_other)
    return np.where((r_pref + r_other) > 0, out, np.nan), pref


def direction_tuning(rate_by_trial, trials, cond_col="orientation", n_shuffle=1000, seed=0):
    """Per-unit tuning summary over a categorical stimulus column.

    Returns (tuning_mean, tuning_sem, summary) where the tuning frames are
    conditions x units and `summary` holds selectivity indices and statistics.
    """
    rng = np.random.default_rng(seed)
    ok = trials[cond_col].notna().values
    cond = trials.loc[ok, cond_col].values
    R = rate_by_trial.loc[ok].values
    thetas = np.sort(np.unique(cond))
    groups = [R[cond == th] for th in thetas]

    mean = np.stack([g.mean(axis=0) for g in groups])           # conditions x units
    sem = np.stack([g.std(axis=0, ddof=1) / np.sqrt(len(g)) for g in groups])

    gosi, pref_ori = circular_selectivity(thetas, mean.T, harmonic=2)
    osi, _ = ratio_index(thetas, mean.T, harmonic=2)
    has_dir = thetas.max() > 180
    if has_dir:
        gdsi, pref_dir = circular_selectivity(thetas, mean.T, harmonic=1)
        dsi, _ = ratio_index(thetas, mean.T, harmonic=1)
    else:
        gdsi = dsi = pref_dir = np.full(mean.shape[1], np.nan)

    f, p = stats.f_oneway(*groups)

    # Permutation null for gOSI: shuffle the condition labels across trials.
    null = np.empty((n_shuffle, mean.shape[1]))
    for i in tqdm(range(n_shuffle), desc=f"gOSI permutation ({cond_col})", leave=False):
        perm = rng.permutation(cond)
        m = np.stack([R[perm == th].mean(axis=0) for th in thetas])
        null[i] = circular_selectivity(thetas, m.T, harmonic=2)[0]
    p_gosi = (1 + (null >= gosi[None, :]).sum(axis=0)) / (n_shuffle + 1)

    summary = pd.DataFrame({
        "gOSI": gosi, "OSI": osi, "pref_ori": pref_ori,
        "gDSI": gdsi, "DSI": dsi, "pref_dir": pref_dir,
        "anova_F": f, "anova_p": p, "gOSI_p": p_gosi,
        "gOSI_null_mean": null.mean(axis=0),
        "peak_rate": mean.max(axis=0), "mean_rate": mean.mean(axis=0),
    }, index=rate_by_trial.columns)
    idx = pd.Index(thetas, name=cond_col)
    return (pd.DataFrame(mean, index=idx, columns=rate_by_trial.columns),
            pd.DataFrame(sem, index=idx, columns=rate_by_trial.columns),
            summary)


def circ_dist_deg(a, b, period=180.0):
    """Smallest absolute difference between two angles on a circle of given period."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % period)
    return np.minimum(d, period - d)


def split_half_reliability(rate_by_trial, trials, cond_col="orientation"):
    """Correlate tuning curves built from alternating repeats of each condition."""
    ok = trials[cond_col].notna().values
    cond = trials.loc[ok, cond_col].values
    R = rate_by_trial.loc[ok].values
    thetas = np.sort(np.unique(cond))
    halves = []
    for parity in (0, 1):
        rows = []
        for th in thetas:
            idx = np.where(cond == th)[0]
            rows.append(R[idx[parity::2]].mean(axis=0))
        halves.append(np.stack(rows))
    a, b = halves
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    denom = np.sqrt((a_c ** 2).sum(axis=0) * (b_c ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a_c * b_c).sum(axis=0) / denom
    pref_a = circular_selectivity(thetas, a.T, harmonic=2)[1]
    pref_b = circular_selectivity(thetas, b.T, harmonic=2)[1]
    return pd.DataFrame({
        "split_half_r": np.where(denom > 0, r, np.nan),
        "split_half_pref_diff": circ_dist_deg(pref_a, pref_b, 180.0),
    }, index=rate_by_trial.columns)


def poisson_loglik(counts, rate):
    """Mean per-trial Poisson log-likelihood for each unit (columns)."""
    from scipy.special import gammaln
    rate = np.clip(rate, 1e-9, None)
    return (counts * np.log(rate) - rate - gammaln(counts + 1)).mean(axis=0)


def glm_direction_model(counts, dirs, n_basis=8, n_splits=5, seed=0, reg=1e-4):
    """Cross-validated Poisson GLM of trial spike counts on drift direction.

    Uses a NeMoS PopulationGLM with a cyclic B-spline basis over direction, and
    compares held-out log-likelihood with an intercept-only (mean-rate) model.
    """
    import nemos as nmo
    from sklearn.model_selection import KFold

    solver_kwargs = {"maxiter": 10000, "tol": 1e-9}

    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=n_basis, bounds=(0.0, 1.0),
                                        label="direction")
    X = np.asarray(basis.compute_features(dirs / 360.0))
    y = np.asarray(counts, dtype=float)

    ll_model = np.zeros((n_splits, y.shape[1]))
    ll_null = np.zeros_like(ll_model)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(tqdm(list(kf.split(X)), desc="GLM cross-validation")):
        glm = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=reg,
                                    solver_name="LBFGS", solver_kwargs=solver_kwargs)
        glm.fit(X[tr], y[tr])
        ll_model[k] = poisson_loglik(y[te], np.asarray(glm.predict(X[te])))
        ll_null[k] = poisson_loglik(y[te], np.tile(y[tr].mean(axis=0), (len(te), 1)))

    full = nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=reg,
                                 solver_name="LBFGS", solver_kwargs=solver_kwargs)
    full.fit(X, y)
    grid = np.arange(0, 360, 1.0)
    curves = np.asarray(full.predict(np.asarray(basis.compute_features(grid / 360.0))))
    return dict(ll_gain=ll_model.mean(axis=0) - ll_null.mean(axis=0),
                ll_model=ll_model.mean(axis=0), ll_null=ll_null.mean(axis=0),
                grid=grid, curves=curves)


def analyze_session(path, n_shuffle=500, cache_dir="results", areas=VIS_AREAS):
    """Full drifting- and static-grating tuning analysis for one session (cached)."""
    import os
    import pickle

    os.makedirs(cache_dir, exist_ok=True)
    tag = path.split("/")[-1].replace(".nwb", "")
    cache = os.path.join(cache_dir, f"{tag}_tuning.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    import dandi_io

    nwb, io = dandi_io.open_session(path)
    units = unit_table(nwb)
    sel = select_units(units, areas)
    tsg = load_spikes(nwb, sel, desc=f"{tag}: spikes")

    dg = stim_table(nwb, "drifting_gratings_presentations")
    sg = stim_table(nwb, "static_gratings_presentations")
    dg_rates = trial_rates(tsg, dg)
    sg_rates = trial_rates(tsg, sg)

    blank = dg["orientation"].isna().values
    tfs = np.sort(dg.loc[~blank, "temporal_frequency"].unique())
    per_tf = pd.DataFrame({tf: dg_rates[dg["temporal_frequency"] == tf].mean(axis=0)
                           for tf in tfs})
    pref_tf = per_tf.idxmax(axis=1)

    tun, sem, summ, half = {}, {}, {}, {}
    for tf in tfs:
        m = (dg["temporal_frequency"] == tf).values
        tun[tf], sem[tf], summ[tf] = direction_tuning(dg_rates.loc[m], dg.loc[m],
                                                      n_shuffle=n_shuffle)
        half[tf] = split_half_reliability(dg_rates.loc[m], dg.loc[m])

    cols = dg_rates.columns
    dg_tuning = pd.DataFrame({u: tun[pref_tf[u]][u] for u in cols})
    dg_sem = pd.DataFrame({u: sem[pref_tf[u]][u] for u in cols})
    dg_summary = pd.DataFrame({u: summ[pref_tf[u]].loc[u] for u in cols}).T
    dg_summary = dg_summary.join(pd.DataFrame({u: half[pref_tf[u]].loc[u] for u in cols}).T)
    dg_summary["pref_tf"] = pref_tf.values
    dg_summary["baseline_rate"] = dg_rates[blank].mean(axis=0).values
    dg_summary["area"] = sel["location"].values
    dg_summary["session"] = tag

    sg_tuning, sg_sem, sg_summary = direction_tuning(sg_rates, sg, n_shuffle=n_shuffle)
    sg_summary["area"] = sel["location"].values
    sg_summary["session"] = tag

    out = dict(session=tag, units=sel, dg_tuning=dg_tuning, dg_sem=dg_sem,
               dg_summary=dg_summary, sg_tuning=sg_tuning, sg_sem=sg_sem,
               sg_summary=sg_summary, n_dg_trials=int((~blank).sum()),
               n_sg_trials=int(sg["orientation"].notna().sum()))
    io.close()
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out
