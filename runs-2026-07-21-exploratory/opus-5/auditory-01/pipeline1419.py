"""Per-session frequency-tuning pipeline for DANDI:001419 (mouse A1 linear probe, 10 tones)."""
import numpy as np
import pandas as pd
import pynapple as nap

import analysis_core as ac
import loaders

DANDISET = '001419'
WIN_EV = (0.005, 0.105)
WIN_BL = (-0.105, -0.005)


def analyze_session(path, asset_id=None):
    """Tuning curves from the LED-off (control) trials of one pure-tone session."""
    assets = loaders.list_assets(DANDISET)
    nwbfile = loaders.open_nwb(DANDISET, asset_id or assets[path])
    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']
    udf = nwbfile.units.to_dataframe()
    trials = nwbfile.trials.to_dataframe()

    ctrl = trials[trials.led_on_time.isna()]          # no optogenetic stimulation
    onsets, freqs = ctrl.start_time.values, ctrl.frequency.values
    uid = np.asarray(units.index)

    ev = ac.trial_counts(units, onsets, WIN_EV)
    bl = ac.trial_counts(units, onsets, WIN_BL)
    curves, sems, stats_df = ac.tuning_table(ev, bl, freqs, WIN_EV, WIN_BL, uid)

    # peak-channel depth and cortical layer for each unit
    depth, layer = [], []
    for i in range(len(udf)):
        e = udf['electrodes'].iloc[i]
        depth.append(float(np.mean(e['y'].values)))
        loc = [s for s in e['location'].values if s.strip()]
        layer.append(loc[0] if loc else 'unknown')
    stats_df['depth_um'] = depth
    stats_df['layer'] = layer
    stats_df['quality'] = udf['quality'].values
    stats_df['session'] = path
    stats_df['subject'] = path.split('/')[0].replace('sub-', '')
    stats_df['responsive'] = ac.fdr(stats_df.p_responsive.values)
    stats_df['tuned'] = ac.fdr(stats_df.p_tuned.values)
    return dict(session=path, ufreq=np.sort(np.unique(freqs)), curves=curves, sems=sems,
                stats=stats_df, n_trials=len(ctrl), counts_ev=ev, freqs=freqs, unit_ids=uid)
