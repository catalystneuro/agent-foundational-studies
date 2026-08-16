"""Trial-level preprocessing: reach direction, epochs, movement periods."""
import numpy as np, pandas as pd, pynapple as nap

N_DIR_BINS = 8

def trial_geometry(trials, cursor_pos):
    """Add active target position, start position, reach direction (rad) to trials frame."""
    tgt = np.array([np.asarray(p)[a] for p, a in zip(trials['target_pos'], trials['active_target'])],
                   dtype=float)
    i_mo = np.searchsorted(cursor_pos.index, trials['move_onset_time'].values)
    p0 = cursor_pos.values[i_mo].astype(float)
    disp = tgt - p0
    tr = trials.copy()
    tr['target_x'], tr['target_y'] = tgt[:, 0], tgt[:, 1]
    tr['start_x'], tr['start_y'] = p0[:, 0], p0[:, 1]
    tr['reach_angle'] = np.arctan2(disp[:, 1], disp[:, 0])
    tr['reach_dist'] = np.hypot(disp[:, 0], disp[:, 1])
    edges = np.linspace(-np.pi, np.pi, N_DIR_BINS + 1)
    tr['dir_bin'] = np.clip(np.digitize(tr['reach_angle'], edges) - 1, 0, N_DIR_BINS - 1)
    tr['is_straight'] = tr['num_barriers'] == 0
    return tr

def dir_bin_centers(n=N_DIR_BINS):
    edges = np.linspace(-np.pi, np.pi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])

def epochs(tr, t_start, t_stop, ref='move_onset_time'):
    """IntervalSet of [ref+t_start, ref+t_stop] per trial."""
    r = tr[ref].values
    return nap.IntervalSet(start=r + t_start, end=r + t_stop)

def movement_epochs(hand_vel, thresh=100.0, min_dur=0.15):
    """Epochs where hand speed exceeds `thresh` mm/s for at least `min_dur` s."""
    speed = nap.Tsd(t=hand_vel.index, d=np.hypot(hand_vel.values[:, 0], hand_vel.values[:, 1]))
    ep = speed.threshold(thresh, method='above').time_support
    return ep[(ep.end - ep.start) >= min_dur]

def perievent_continuous(tsd, events, window=(-0.4, 0.8), dt=0.001):
    """Align a regularly-sampled-within-trial TsdFrame/Tsd to events.

    Behavior in MC_Maze is 1 kHz within trials with small gaps between trials, so
    pynapple's compute_perievent (which requires a global uniform grid) cannot be used.
    Returns (lags, array of shape (n_lags, n_events, n_cols)) with NaN where samples are
    missing or non-contiguous.
    """
    lags = np.arange(round(window[0] / dt), round(window[1] / dt) + 1) * dt
    t = np.asarray(tsd.index)
    v = np.atleast_2d(np.asarray(tsd.values).T).T
    out = np.full((len(lags), len(events), v.shape[1]), np.nan)
    for j, e in enumerate(events):
        idx = np.searchsorted(t, e + lags)
        ok = (idx < len(t)) & (np.abs(t[np.clip(idx, 0, len(t) - 1)] - (e + lags)) < dt)
        out[ok, j] = v[idx[ok]]
    return lags, out
