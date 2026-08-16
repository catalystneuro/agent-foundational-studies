"""Extraction and per-unit analysis for the orientation-selectivity pipeline.

`extract_session` streams one session of DANDI:000021 and caches everything the
analysis needs as a compressed .npz. `analyze_session` turns that cache into one
row of tuning metrics per unit. Both are imported by the numbered development
scripts and by the consolidated notebook, so there is one implementation.
"""
import os

import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

import oslib
from analyze_helpers import responsive  # noqa: F401  (re-exported for convenience)


OUT = os.path.join(oslib.HERE, "extracted")
os.makedirs(OUT, exist_ok=True)

# Counting windows. The 30 ms offset skips the visual response latency of mouse
# visual cortex; the window then covers the rest of the presentation.
DG_OFFSET, DG_WINDOW = 0.03, 1.95   # drifting gratings, 2.00 s presentations
SG_OFFSET, SG_WINDOW = 0.03, 0.22   # static gratings, 0.25 s back-to-back presentations

# Session used for the single-session prototype figures; its drifting-grating
# block spike times are saved as well so rasters can be drawn without restreaming.
PROTOTYPE = "715093703"


def extract_session(session_id):
    out = os.path.join(OUT, f"{session_id}.npz")
    if os.path.exists(out):
        return out
    nwbfile = oslib.open_session(session_id)
    tsg, meta = oslib.good_units(nwbfile)

    dg = oslib.stim_table(nwbfile, "drifting_gratings_presentations")
    sg = oslib.stim_table(nwbfile, "static_gratings_presentations")
    dg_blank = oslib.blank_table(nwbfile, "drifting_gratings_presentations")

    dg_rates = oslib.trial_rates(tsg, dg["start_time"].values, dg["stop_time"].values, DG_OFFSET, DG_WINDOW)
    sg_rates = oslib.trial_rates(tsg, sg["start_time"].values, sg["stop_time"].values, SG_OFFSET, SG_WINDOW)
    blank_rates = oslib.trial_rates(
        tsg, dg_blank["start_time"].values, dg_blank["stop_time"].values, DG_OFFSET, DG_WINDOW
    )

    # Trials overlapping a probe's invalid-data epochs are set to NaN for the
    # units on that probe only.
    probes = meta["probe_id"].values
    dg_bad = oslib.invalid_trial_mask(nwbfile, probes, dg["start_time"].values, dg["stop_time"].values)
    sg_bad = oslib.invalid_trial_mask(nwbfile, probes, sg["start_time"].values, sg["stop_time"].values)
    dg_rates[dg_bad] = np.nan
    sg_rates[sg_bad] = np.nan
    print(f"  invalid trials: drifting {dg_bad.mean():.2%}, static {sg_bad.mean():.2%}", flush=True)

    # Mean running speed on each drifting-grating trial, for the locomotion control.
    run = nwbfile.processing["running"]["running_speed"]
    rt, rv = np.asarray(run.timestamps[:]), np.asarray(run.data[:])
    order = np.argsort(rt)
    rt, rv = rt[order], rv[order]
    lo = np.searchsorted(rt, dg["start_time"].values)
    hi = np.searchsorted(rt, dg["stop_time"].values)
    dg_speed = np.array([np.nanmean(rv[a:b]) if b > a else np.nan for a, b in zip(lo, hi)])

    np.savez_compressed(
        out,
        session=str(session_id),
        subject=str(nwbfile.subject.subject_id),
        genotype=str(nwbfile.subject.genotype),
        unit_ids=np.asarray(list(tsg.keys())),
        area=meta["area"].values.astype(str),
        snr=meta["snr"].values,
        firing_rate=meta["firing_rate"].values,
        waveform_duration=meta["waveform_duration"].values,
        probe_id=meta["probe_id"].values.astype(str),
        dg_rates=dg_rates.astype(np.float32),
        dg_ori=dg["orientation"].values,
        dg_tf=dg["temporal_frequency"].values,
        dg_start=dg["start_time"].values,
        dg_stop=dg["stop_time"].values,
        dg_speed=dg_speed,
        blank_rates=blank_rates.astype(np.float32),
        sg_rates=sg_rates.astype(np.float32),
        sg_ori=sg["orientation"].values,
        sg_sf=sg["spatial_frequency"].values,
        sg_phase=sg["phase"].values,
        sg_start=sg["start_time"].values,
    )

    if str(session_id) == PROTOTYPE:
        # Spike times inside the drifting-grating block, for raster figures.
        t0, t1 = dg["start_time"].min() - 2, dg["stop_time"].max() + 2
        sub = tsg.restrict(nap.IntervalSet(start=t0, end=t1))
        np.savez_compressed(
            os.path.join(OUT, f"{session_id}_dg_spikes.npz"),
            unit_ids=np.asarray(list(sub.keys())),
            **{f"u{k}": np.asarray(v.t, dtype=np.float32) for k, v in sub.items()},
        )
    return out




N_PERM = 1000
RUN_THRESHOLD = 1.0   # cm/s; below this the mouse is treated as stationary


def analyze_session(path):
    d = np.load(path, allow_pickle=True)
    ses = str(d["session"])
    n_units = len(d["unit_ids"])

    dg, dg_ori, dg_tf, dg_speed = d["dg_rates"].astype(float), d["dg_ori"], d["dg_tf"], d["dg_speed"]
    sg, sg_ori = d["sg_rates"].astype(float), d["sg_ori"]
    blank = d["blank_rates"].astype(float)
    base = np.nanmean(blank, axis=1)

    dirs, tuning, best_tf = oslib.dir_tuning(dg, dg_ori, dg_tf, base)
    # Raw (not baseline-subtracted) mean and SEM at the same preferred temporal
    # frequency, kept for plotting: the metrics use the evoked response, but a
    # tuning curve is only readable next to the spontaneous rate it sits on.
    tfs_all = np.array(sorted(np.unique(dg_tf)))
    tuning_raw = np.full_like(tuning, np.nan)
    tuning_sem = np.full_like(tuning, np.nan)
    for i in range(n_units):
        at_tf = dg_tf == tfs_all[best_tf[i]]
        for j, dd in enumerate(dirs):
            v = dg[i, at_tf & (dg_ori == dd)]
            v = v[np.isfinite(v)]
            if len(v):
                tuning_raw[i, j] = v.mean()
                tuning_sem[i, j] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    gosi, gdsi, pref_ori, pref_dir = oslib.osi_dsi(tuning, dirs)
    osi_c = oslib.classic_osi(tuning, dirs)
    _, p_perm, gosi_null = oslib.perm_test_osi(dg, dg_ori, dg_tf, base, N_PERM, seed=int(ses) % 2**31)

    # Visual responsiveness, tested at the trial level: the rates on the ~15 trials
    # of the best (direction, temporal frequency) condition against the 30 blank
    # sweeps of the same block. The blank sweeps are the matched baseline (same grey
    # screen, interleaved), so this is not confounded by slow drift in firing rate.
    # A trial-level test is needed rather than a threshold on the mean: units firing
    # a couple of spikes per trial can otherwise reach gOSI near 1 purely by chance,
    # because a single positive direction in a curve of zeros is maximally selective.
    peak = np.nanmax(tuning, axis=1)
    tfs = np.array(sorted(np.unique(dg_tf)))
    best_dir = np.array(dirs)[np.nanargmax(tuning, axis=1)]
    p_resp = np.ones(n_units)
    for i in range(n_units):
        sel = (dg_ori == best_dir[i]) & (dg_tf == tfs[best_tf[i]])
        a = dg[i, sel]
        b = blank[i]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) >= 5 and len(b) >= 5 and (a.std() + b.std()) > 0:
            p_resp[i] = stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue

    # Locomotion control.
    still = np.isfinite(dg_speed) & (dg_speed < RUN_THRESHOLD)
    moving = np.isfinite(dg_speed) & (dg_speed >= RUN_THRESHOLD)
    gosi_still = np.full(n_units, np.nan)
    pref_still = np.full(n_units, np.nan)
    gosi_move = np.full(n_units, np.nan)
    if still.sum() >= 80:
        _, t, _ = oslib.dir_tuning(dg[:, still], dg_ori[still], dg_tf[still], base, dirs)
        gosi_still, _, pref_still, _ = oslib.osi_dsi(t, dirs)
    if moving.sum() >= 80:
        _, t, _ = oslib.dir_tuning(dg[:, moving], dg_ori[moving], dg_tf[moving], base, dirs)
        gosi_move, _, _, _ = oslib.osi_dsi(t, dirs)

    # Split-half: repeats of each (direction, TF) condition split odd/even.
    rep = np.zeros(len(dg_ori), dtype=int)
    for o, f in set(zip(dg_ori, dg_tf)):
        idx = np.where((dg_ori == o) & (dg_tf == f))[0]
        rep[idx] = np.arange(len(idx))
    half = rep % 2 == 0
    _, t1, _ = oslib.dir_tuning(dg[:, half], dg_ori[half], dg_tf[half], base, dirs)
    _, t2, _ = oslib.dir_tuning(dg[:, ~half], dg_ori[~half], dg_tf[~half], base, dirs)
    g1, _, pref1, _ = oslib.osi_dsi(t1, dirs)
    g2, _, pref2, _ = oslib.osi_dsi(t2, dirs)

    # Static gratings: independent block, 6 orientations x 5 spatial frequencies x
    # 4 phases, 0.25 s each. Orientation here already has period 180 deg.
    sg_base = np.nanpercentile(sg, 10, axis=1)   # no blank sweeps in this block
    sg_levels, sg_by_ori, _ = oslib.tuning_by_group(sg, sg_ori)
    r = np.clip(sg_by_ori - sg_base[:, None], 0, None)
    th = np.deg2rad(sg_levels)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (r * np.exp(2j * th)).sum(axis=-1) / r.sum(axis=-1)
    sg_gosi = np.abs(v)
    sg_pref = np.rad2deg(np.angle(v) / 2) % 180

    df = pd.DataFrame(dict(
        session=ses, subject=str(d["subject"]), unit_id=d["unit_ids"], area=d["area"],
        snr=d["snr"], pref_tf=np.array(sorted(np.unique(dg_tf)))[best_tf],
        baseline_rate=base, peak_evoked=peak, p_resp=p_resp,
        gosi=gosi, gosi_null=gosi_null, p_perm=p_perm, osi_classic=osi_c, gdsi=gdsi,
        pref_ori=pref_ori, pref_dir=pref_dir,
        gosi_still=gosi_still, gosi_move=gosi_move, pref_ori_still=pref_still,
        gosi_half1=g1, gosi_half2=g2, pref_half1=pref1, pref_half2=pref2,
        sg_gosi=sg_gosi, sg_pref_ori=sg_pref,
    ))
    df["gosi_corrected"] = (df["gosi"] - df["gosi_null"]).clip(lower=0)
    df["region"] = np.where(df["area"].isin(oslib.VISUAL_CORTEX), "cortex", "thalamus")
    np.save(f"extracted/{ses}_tuning.npy", tuning)
    np.save(f"extracted/{ses}_tuningraw.npy", tuning_raw)
    np.save(f"extracted/{ses}_tuningsem.npy", tuning_sem)
    np.save(f"extracted/{ses}_half1.npy", t1)
    np.save(f"extracted/{ses}_half2.npy", t2)
    np.save(f"extracted/{ses}_dirs.npy", dirs)
    return df


