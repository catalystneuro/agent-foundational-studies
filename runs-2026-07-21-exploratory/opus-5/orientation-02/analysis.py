"""
Per-session orientation-selectivity analysis for DANDI:000021.

Everything here operates on one session and returns plain arrays/frames so that
the multi-session driver can pool results.
"""

import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

import orilib as O

DIRECTIONS = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
STATIC_ORIS = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
N_PERM = 1000


def _clean_trials(df, ori_values):
    """Drop blank-sweep trials and keep only the canonical stimulus values."""
    ok = df["orientation"].notna() & df["orientation"].isin(ori_values)
    return df[ok].reset_index(drop=True)


def trial_rate_matrix(tsgroup, df):
    """(n_trials, n_units) firing rates in the presentation window."""
    starts = df["start_time"].values
    stops = df["stop_time"].values
    dur = stops - starts
    counts = np.zeros((len(starts), len(tsgroup)), dtype=float)
    for j, u in enumerate(tsgroup.index):
        t = tsgroup[u].t
        counts[:, j] = np.searchsorted(t, stops, "right") - np.searchsorted(t, starts, "left")
    return counts / dur[:, None], counts


def _design(labels, values):
    """Trial-averaging matrix D with D[i, k] = 1/n_k if trial i had value k."""
    D = np.zeros((len(labels), len(values)))
    for k, v in enumerate(values):
        m = labels == v
        D[m, k] = 1.0 / m.sum()
    return D


def condition_means(rates, labels, values):
    """Mean and SEM rate per stimulus value -> (n_values, n_units) each."""
    m = np.stack([rates[labels == v].mean(0) for v in values])
    s = np.stack(
        [rates[labels == v].std(0, ddof=1) / np.sqrt((labels == v).sum()) for v in values]
    )
    return m, s


def selectivity_table(rates, labels, values, kind="direction", blocks=None, n_perm=N_PERM, seed=0):
    """Per-unit selectivity metrics with a label-shuffle null.

    kind="direction": labels are 0..315 drift directions; OSI uses 2*theta, DSI theta.
    kind="orientation": labels are 0..150 static orientations; angles are doubled
        internally so the same circular-variance formula applies.
    """
    rng = np.random.default_rng(seed)
    n_units = rates.shape[1]
    D = _design(labels, values)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        tc = D.T @ rates  # (n_val, n_units) mean rate per stimulus value

    # Doubling the stimulus angle maps orientation onto the full circle, so the
    # same resultant-vector formula serves drift directions and static orientations.
    v2 = np.exp(1j * 2.0 * np.deg2rad(values))
    tot = tc.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z2 = (tc * v2[:, None]).sum(0) / tot
    osi = np.abs(z2)
    pref_ori = (0.5 * np.rad2deg(np.angle(z2))) % 180.0
    if kind == "direction":
        v1 = np.exp(1j * np.deg2rad(values))
        with np.errstate(invalid="ignore", divide="ignore"):
            z1 = (tc * v1[:, None]).sum(0) / tot
        dsi = np.abs(z1)
        pref_dir = np.rad2deg(np.angle(z1)) % 360.0
    else:
        dsi = np.full(n_units, np.nan)
        pref_dir = np.full(n_units, np.nan)

    # classic ratio index: (R_pref - R_orth) / (R_pref + R_orth) on the orientation curve
    if kind == "direction":
        ori_vals = np.array([0.0, 45.0, 90.0, 135.0])
        ori_tc = np.stack(
            [tc[np.isin(values, [o, o + 180.0])].mean(0) for o in ori_vals]
        )
    else:
        ori_vals = values
        ori_tc = tc
    ipref = np.argmax(ori_tc, axis=0)
    n_ori = len(ori_vals)
    iorth = (ipref + n_ori // 2) % n_ori
    rp = ori_tc[ipref, np.arange(n_units)]
    ro = ori_tc[iorth, np.arange(n_units)]
    with np.errstate(invalid="ignore", divide="ignore"):
        osi_ratio = (rp - ro) / (rp + ro)

    # one-way ANOVA across stimulus values
    groups = [rates[labels == v] for v in values]
    fstat, pval_anova = stats.f_oneway(*groups, axis=0)

    # Permutation null on OSI. Labels are shuffled *within* each stimulus block so
    # that slow drift in firing rate across the ~2.5 h session cannot masquerade as
    # tuning: the null preserves each block's rate level.
    if blocks is None:
        blocks = np.zeros(len(labels))
    block_idx = [np.where(blocks == b)[0] for b in np.unique(blocks)]
    null = np.zeros((n_perm, n_units))
    for k in range(n_perm):
        perm = np.arange(len(labels))
        for idx in block_idx:
            perm[idx] = rng.permutation(idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            tcp = D[perm].T @ rates
            null[k] = np.abs((tcp * v2[:, None]).sum(0) / tcp.sum(0))
    pval_perm = (1.0 + (null >= osi[None, :]).sum(0)) / (n_perm + 1.0)
    osi_null_median = np.nanmedian(null, axis=0)

    out = pd.DataFrame(
        dict(
            osi=osi,
            dsi=dsi,
            osi_ratio=osi_ratio,
            pref_ori=pref_ori,
            pref_dir=pref_dir,
            mean_rate=rates.mean(0),
            max_rate=tc.max(0),
            p_anova=pval_anova,
            p_perm=pval_perm,
            osi_null_median=osi_null_median,
        ),
        index=list(tc_index(rates)),
    )
    return out, tc


def tc_index(rates):
    return range(rates.shape[1])


def split_half_reliability(rates, labels, values, kind, n_rep=50, seed=1):
    """Correlation between tuning curves from independent random trial halves."""
    rng = np.random.default_rng(seed)
    n_units = rates.shape[1]
    r = np.zeros((n_rep, n_units))
    idx_by_val = [np.where(labels == v)[0] for v in values]
    for k in range(n_rep):
        a, b = [], []
        for idx in idx_by_val:
            p = rng.permutation(idx)
            h = len(p) // 2
            a.append(rates[p[:h]].mean(0))
            b.append(rates[p[h : 2 * h]].mean(0))
        A, B = np.stack(a), np.stack(b)
        A = A - A.mean(0)
        B = B - B.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[k] = (A * B).sum(0) / np.sqrt((A**2).sum(0) * (B**2).sum(0))
    return np.nanmean(r, axis=0)


def analyze_session(url, verbose=True):
    """Full per-session pipeline. Returns a dict of results."""
    nwbfile = O.open_nwb(url)
    tsg, meta = O.load_units(nwbfile)
    keep = O.passes_qc(meta)
    tsg = tsg[list(meta.index[keep])]
    meta = meta[keep].copy()

    dg = _clean_trials(O.load_stim_table(nwbfile, "drifting_gratings_presentations"), DIRECTIONS)
    sg = _clean_trials(O.load_stim_table(nwbfile, "static_gratings_presentations"), STATIC_ORIS)

    dg_rates, dg_counts = trial_rate_matrix(tsg, dg)
    sg_rates, sg_counts = trial_rate_matrix(tsg, sg)

    dg_labels = dg["orientation"].values
    sg_labels = sg["orientation"].values

    tab_dg, tc_dg = selectivity_table(
        dg_rates, dg_labels, DIRECTIONS, kind="direction", blocks=dg["stimulus_block"].values
    )
    tab_sg, tc_sg = selectivity_table(
        sg_rates, sg_labels, STATIC_ORIS, kind="orientation", blocks=sg["stimulus_block"].values
    )

    tab_dg.index = meta.index
    tab_sg.index = meta.index
    tab_dg["reliability"] = split_half_reliability(dg_rates, dg_labels, DIRECTIONS, "direction")
    tab_sg["reliability"] = split_half_reliability(sg_rates, sg_labels, STATIC_ORIS, "orientation")

    res = pd.concat(
        [
            meta[["area", "session_id", "firing_rate", "snr", "waveform_duration"]],
            tab_dg.add_prefix("dg_"),
            tab_sg.add_prefix("sg_"),
        ],
        axis=1,
    )
    res["ori_agreement_deg"] = O.circ_dist_ori(res["dg_pref_ori"], res["sg_pref_ori"])

    if verbose:
        print(
            f"session {meta['session_id'].iloc[0]}: {len(meta)} QC units, "
            f"{len(dg)} drifting-grating trials, {len(sg)} static-grating trials"
        )

    return dict(
        session_id=str(nwbfile.session_id),
        units=res,
        tc_dg=tc_dg,  # (8 directions, n_units) mean rate
        tc_sg=tc_sg,  # (6 orientations, n_units)
        dg_rates=dg_rates,
        dg_labels=dg_labels,
        dg_tf=dg["temporal_frequency"].values,
        sg_rates=sg_rates,
        sg_labels=sg_labels,
        dg_table=dg,
        tsgroup=tsg,
        nwbfile=nwbfile,
    )
