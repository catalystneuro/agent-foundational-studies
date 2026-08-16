"""Build pynapple objects from the cached MC_Maze arrays and validate the behavior streams."""
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s01_load import load_cache

nap.nap_config.suppress_conversion_warnings = True

MOVE_WIN = (0.0, 0.20)  # window after move onset used to define reach direction


def build(d=None):
    if d is None:
        d = load_cache()

    obs = nap.IntervalSet(start=d["obs_intervals"][:, 0], end=d["obs_intervals"][:, 1])

    spikes = nap.TsGroup(
        {i: nap.Ts(t=s) for i, s in enumerate(d["spike_times"])},
        time_support=obs,
    )

    t = d["t"]
    pos = nap.TsdFrame(t=t, d=d["pos"].astype(np.float64), columns=["x", "y"], time_support=obs)
    vel = nap.TsdFrame(t=t, d=d["vel"].astype(np.float64), columns=["vx", "vy"], time_support=obs)
    speed = nap.Tsd(t=t, d=np.linalg.norm(d["vel"], axis=1).astype(np.float64), time_support=obs)

    trials = pd.DataFrame(
        dict(
            start=d["trial_start"],
            stop=d["trial_stop"],
            target_on=d["trial_target_on"],
            go_cue=d["trial_go_cue"],
            move_onset=d["trial_move_onset"],
            num_barriers=d["trial_num_barriers"],
            maze_id=d["trial_maze_id"],
            split=d["trial_split"],
            target_x=d["trial_target_xy"][:, 0] / 10.0,  # mm -> cm
            target_y=d["trial_target_xy"][:, 1] / 10.0,
        )
    )

    # Reach direction: direction of mean hand velocity in the first 200 ms of movement.
    mo = trials["move_onset"].values
    i0 = np.searchsorted(t, mo + MOVE_WIN[0])
    i1 = np.searchsorted(t, mo + MOVE_WIN[1])
    mv = np.array([d["vel"][a:b].mean(0) for a, b in zip(i0, i1)])
    trials["reach_angle"] = np.arctan2(mv[:, 1], mv[:, 0])
    trials["reach_speed"] = np.linalg.norm(mv, axis=1)
    # Peak speed reached in the 600 ms after movement onset
    sp = np.linalg.norm(d["vel"], axis=1)
    i2 = np.searchsorted(t, mo + 0.6)
    trials["peak_speed"] = np.array([sp[a:b].max() for a, b in zip(i0, i2)])
    trials["hand_x0"] = d["pos"][i0, 0]
    trials["hand_y0"] = d["pos"][i0, 1]

    return dict(spikes=spikes, pos=pos, vel=vel, speed=speed, trials=trials, obs=obs, raw=d)


def figure_behavior(S, path="figures/fig01_behavior_overview.png"):
    d, trials, pos = S["raw"], S["trials"], S["pos"]
    t = d["t"]
    free = trials["num_barriers"].values == 0

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

    cmap = plt.get_cmap("hsv")
    ang = trials["reach_angle"].values

    # (a,b) hand paths, straight (barrier-free) vs maze trials
    for k, (mask, title) in enumerate(
        [(free, "Barrier-free reaches (n=%d)" % free.sum()),
         (~free, "Maze reaches (n=%d)" % (~free).sum())]
    ):
        ax = fig.add_subplot(gs[0, k])
        idx = np.where(mask)[0][:250]
        for i in idx:
            a = np.searchsorted(t, trials["move_onset"].values[i])
            b = np.searchsorted(t, trials["move_onset"].values[i] + 0.7)
            ax.plot(d["pos"][a:b, 0], d["pos"][a:b, 1], lw=0.5,
                    color=cmap((ang[i] + np.pi) / (2 * np.pi)), alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("hand x (cm)")
        ax.set_ylabel("hand y (cm)")
        ax.set_aspect("equal")
        ax.set_xlim(-16, 16)
        ax.set_ylim(-19, 12)

    # (c) speed profiles aligned to movement onset
    ax = fig.add_subplot(gs[0, 2])
    lags = np.arange(-400, 801)  # ms, 1 kHz behavior
    prof = np.full((len(trials), len(lags)), np.nan)
    sp = np.linalg.norm(d["vel"], axis=1)
    for i, m in enumerate(trials["move_onset"].values):
        a = np.searchsorted(t, m) + lags[0]
        if a < 0 or a + len(lags) > len(t):
            continue
        seg_t = t[a:a + len(lags)]
        if seg_t[-1] - seg_t[0] > 1.5 * (len(lags) / 1000.0):
            continue  # crosses a gap between trials
        prof[i] = sp[a:a + len(lags)]
    mp = np.nanmean(prof, 0)
    sd = np.nanstd(prof, 0)
    ax.plot(lags, mp, color="k")
    ax.fill_between(lags, mp - sd, mp + sd, color="k", alpha=0.2)
    ax.axvline(0, color="crimson", ls="--", label="movement onset")
    ax.set_xlabel("time from movement onset (ms)")
    ax.set_ylabel("hand speed (cm/s)")
    ax.set_title("Speed profiles (mean $\\pm$ SD)", fontsize=11)
    ax.legend(fontsize=9)

    # (d) reach direction / speed distributions
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    ax.hist(ang, bins=36, color="steelblue")
    ax.set_title("Reach direction distribution", fontsize=11, pad=18)
    ax.set_yticklabels([])

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(trials["peak_speed"], bins=40, color="slategray")
    ax.set_xlabel("peak hand speed (cm/s)")
    ax.set_ylabel("trials")
    ax.set_title("Peak speed per trial", fontsize=11)

    # (e) raw traces + spike raster for one stretch of trials
    ax = fig.add_subplot(gs[1, 2])
    t0 = trials["start"].values[10]
    t1 = trials["stop"].values[14]
    m = (t >= t0) & (t <= t1)
    tw, pw, sw = t[m], d["pos"][m].astype(float).copy(), sp[m].astype(float).copy()
    gap = np.where(np.diff(tw) > 0.005)[0]      # break the line across inter-trial gaps
    pw[gap] = np.nan
    sw[gap] = np.nan
    ax.plot(tw - t0, pw[:, 0], label="hand x", color="tab:blue", lw=1)
    ax.plot(tw - t0, pw[:, 1], label="hand y", color="tab:orange", lw=1)
    ax.plot(tw - t0, sw / 10.0, label="speed/10", color="k", lw=0.8, alpha=0.7)
    for u in range(0, 60):
        s = S["spikes"][u].t
        s = s[(s >= t0) & (s <= t1)] - t0
        ax.plot(s, np.full_like(s, -14 - 0.16 * u), "|", color="0.3", ms=2, mew=0.5)
    for m_on in trials["move_onset"].values[10:15]:
        ax.axvline(m_on - t0, color="crimson", ls="--", lw=0.8)
    ax.set_xlabel("time from trial 10 onset (s)")
    ax.set_ylabel("cm  /  (cm/s)/10")
    ax.set_title("Raw hand kinematics + spikes (60 units)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.set_ylim(-24, 20)

    fig.suptitle("MC_Maze (DANDI:000128), monkey Jenkins - behavior and raw neural data", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    S = build()
    print(S["spikes"])
    print(S["trials"].describe().T)
    print("mean rate over observed intervals (Hz):", float(np.mean(S["spikes"].rates)))
    print("total observed time (s):", float(S["obs"].tot_length()))
    figure_behavior(S)
