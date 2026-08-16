"""Per-session frequency-tuning pipeline for DANDI:000986 (mouse A1, pure tones)."""
import numpy as np
import pandas as pd
import pynapple as nap

import analysis_core as ac
import loaders

DANDISET = '000986'
WIN_EV = (0.005, 0.105)     # tone-evoked window (25 ms tone, cortical onset ~15-40 ms)
WIN_BL = (-0.105, -0.005)   # matched pre-tone baseline (inter-tone interval is 805 ms)
PSTH_BIN = 0.005
PSTH_RANGE = (-0.1, 0.3)


def onset_latency(t, rate, bl_lo=-0.1, bl_hi=-0.005, n_sd=3.0, n_consec=2):
    """First post-onset time at which the PSTH exceeds baseline + n_sd SD for n_consec bins."""
    bl = rate[(t >= bl_lo) & (t < bl_hi)]
    thr = bl.mean() + n_sd * bl.std()
    post = np.where(t >= 0)[0]
    run = 0
    for i in post:
        if rate[i] > thr:
            run += 1
            if run >= n_consec:
                return t[i - n_consec + 1]
        else:
            run = 0
    return np.nan


def analyze_session(path, asset_id=None):
    """Load one session and compute tuning curves, PSTHs and per-unit statistics."""
    assets = loaders.list_assets(DANDISET)
    nwbfile = loaders.open_nwb(DANDISET, asset_id or assets[path])
    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']
    trials = nwbfile.trials.to_dataframe()
    onsets = trials.start_time.values
    freqs = trials.stim_frequency.values
    ufreq = np.sort(np.unique(freqs))
    uid = np.asarray(units.index)

    ev = ac.trial_counts(units, onsets, WIN_EV)
    bl = ac.trial_counts(units, onsets, WIN_BL)
    curves, sems, stats_df = ac.tuning_table(ev, bl, freqs, WIN_EV, WIN_BL, uid)

    # PSTH per frequency: (n_units, n_freqs, n_bins)
    psths = None
    for i, f in enumerate(ufreq):
        t, r = ac.psth(units, onsets[freqs == f], *PSTH_RANGE, PSTH_BIN)
        if psths is None:
            psths = np.zeros((len(uid), len(ufreq), len(t)))
        psths[:, i, :] = r

    lat = [onset_latency(t, psths[j, list(ufreq).index(stats_df.best_frequency.iloc[j])])
           for j in range(len(uid))]
    stats_df['latency_s'] = lat
    stats_df['session'] = path
    stats_df['subject'] = path.split('/')[0].replace('sub-', '')
    stats_df['responsive'] = ac.fdr(stats_df.p_responsive.values)
    stats_df['tuned'] = ac.fdr(stats_df.p_tuned.values)

    return dict(session=path, ufreq=ufreq, curves=curves, sems=sems, stats=stats_df,
                psth_t=t, psths=psths, counts_ev=ev, counts_bl=bl, freqs=freqs,
                onsets=onsets, unit_ids=uid, n_trials=len(trials))
