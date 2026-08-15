"""Extract per-session decoding features from the selected IBL sessions.

For each session we save
  * the trial table (with derived choice / block / history columns),
  * a (n_trials, n_units, n_windows) tensor of spike counts in 200 ms windows
    stepped by 50 ms around Gabor onset,
  * per-unit coarse brain region,
  * per-trial behavioural covariates measured in the same pre-stimulus window
    (wheel movement, whisker-pad motion energy, pupil diameter), which are the
    obvious confounds for any "pre-stimulus decoding" claim.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import ibl_common as ic

OUT = "session_data"
WIN_HALF = 0.10
CENTERS = np.round(np.arange(-1.10, 0.4001, 0.05), 3)
# Cap on units per coarse region.  Some sessions yield >1500 units, which makes
# the permutation tests very slow for no gain in decoding accuracy.
MAX_UNITS_PER_REGION = 120


def subsample_units(u, seed=0):
    """Drop unlocalised units and cap each coarse region, without regard to activity."""
    rng = np.random.default_rng(seed)
    keep = []
    for reg, idx in u.groupby("region").groups.items():
        if reg == "unknown":
            continue
        idx = np.asarray(idx)
        if len(idx) > MAX_UNITS_PER_REGION:
            idx = rng.choice(idx, MAX_UNITS_PER_REGION, replace=False)
        keep.append(idx)
    return u.loc[np.sort(np.concatenate(keep))]


def interval_mean(timestamps, values, starts, stops):
    """Mean of an irregularly sampled signal inside each [start, stop) window."""
    order = np.argsort(timestamps)
    t, v = np.asarray(timestamps)[order], np.asarray(values, dtype=float)[order]
    good = np.isfinite(v)
    csum = np.concatenate([[0.0], np.cumsum(np.where(good, v, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(good.astype(int))])
    i0, i1 = np.searchsorted(t, starts), np.searchsorted(t, stops)
    n = cnt[i1] - cnt[i0]
    s = csum[i1] - csum[i0]
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def behavioural_covariates(nwbfile, stim_on, window):
    """Movement and arousal measures in the pre-stimulus window."""
    starts, stops = stim_on + window[0], stim_on + window[1]
    cov = {}

    wheel = nwbfile.processing["wheel"]
    vel = wheel["WheelVelocitySmoothed"]
    t_vel = vel.starting_time + np.arange(vel.data.shape[0]) / vel.rate
    v = np.abs(np.asarray(vel.data[:], dtype=np.float32))
    cov["wheel_abs_vel"] = interval_mean(t_vel, v, starts, stops)

    pos = wheel["WheelPosition"]
    t_pos = np.asarray(pos.timestamps[:])
    p = np.asarray(pos.data[:], dtype=np.float32)
    i0, i1 = np.searchsorted(t_pos, starts), np.searchsorted(t_pos, stops)
    cov["wheel_range"] = np.array(
        [np.ptp(p[a:b]) if b > a else np.nan for a, b in zip(i0, i1)]
    )

    for mod, key, name in [
        ("motion_energy", "LeftCameraMotionEnergy", "motion_energy"),
        ("pupil", "LeftPupilDiameterSmoothed", "pupil"),
    ]:
        if mod in nwbfile.processing and key in nwbfile.processing[mod].data_interfaces:
            ts = nwbfile.processing[mod][key]
            cov[name] = interval_mean(
                np.asarray(ts.timestamps[:]), np.asarray(ts.data[:]), starts, stops
            )
        else:
            cov[name] = np.full(len(stim_on), np.nan)
    return pd.DataFrame(cov)


def extract(asset_id, session_id):
    nwbfile = ic.open_nwb(asset_id)
    tr = ic.get_trials(nwbfile)
    v = ic.valid_trials(tr, require_biased=True)
    u = subsample_units(ic.get_units(nwbfile), seed=abs(hash(session_id)) % 10000)

    all_spikes = ic.all_spike_times(nwbfile)
    spikes = [all_spikes[i] for i in u.index]
    stim_on = v["stim_on"].to_numpy()

    counts = np.stack(
        [
            ic.window_counts(spikes, stim_on, (c - WIN_HALF, c + WIN_HALF))
            for c in CENTERS
        ],
        axis=-1,
    ).astype(np.int16)

    cov = behavioural_covariates(nwbfile, stim_on, ic.PRE_WINDOW)
    cov.index = v.index

    np.savez_compressed(
        f"{OUT}/{session_id}.npz",
        counts=counts,
        centers=CENTERS,
        win_half=WIN_HALF,
        regions=u["region"].to_numpy().astype(str),
        locations=u["location"].to_numpy().astype(str),
        firing_rate=u["firing_rate"].to_numpy(),
    )
    keep = [
        "stim_on", "signed_contrast", "abs_contrast", "choice_right", "block_right",
        "probability_left", "is_mouse_rewarded", "gabor_stimulus_side",
        "prev_choice_right", "prev_rewarded", "prev_stim_right", "quiescence_period",
        "wheel_movement_onset_time", "feedback_time", "block_index",
    ]
    pd.concat([v[keep].reset_index(drop=True), cov.reset_index(drop=True)], axis=1).to_csv(
        f"{OUT}/{session_id}_trials.csv", index=False
    )
    return len(v), len(u)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sel = pd.read_csv("selected_sessions.csv")
    for _, s in tqdm(list(sel.iterrows()), desc="extracting"):
        if os.path.exists(f"{OUT}/{s.session_id}.npz"):
            continue
        n_tr, n_u = extract(s.asset_id, s.session_id)
        print(f"{s.session_id}  {n_tr} trials  {n_u} units", flush=True)
