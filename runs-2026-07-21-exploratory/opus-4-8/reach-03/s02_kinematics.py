"""Per-trial reach kinematics: onset, peak speed, direction, endpoint."""
import numpy as np, os
import matplotlib; import matplotlib.pyplot as plt
from s01_load import load_all

CACHE = "cache"; FIG = "figures"; os.makedirs(FIG, exist_ok=True)


def trial_kinematics(d, pre=0.1, post=0.6):
    t, hv, hp = d['t'], d['hand_vel'], d['hand_pos']
    dt = np.median(np.diff(t[:1000]))
    speed = np.hypot(hv[:, 0], hv[:, 1])
    mo = d['move_onset_time']
    i0 = np.searchsorted(t, mo)
    npost = int(round(post / dt))
    rows = []
    for k, i in enumerate(i0):
        sl = slice(i, i + npost)
        sp = speed[sl]
        ipk = int(np.argmax(sp))
        pk = sp[ipk]
        # reach end: first drop below 20% of peak after the peak
        after = np.where(sp[ipk:] < 0.2 * pk)[0]
        iend = ipk + (after[0] if len(after) else npost - 1)
        disp = hp[i + iend] - hp[i]
        rows.append(dict(
            idx=k, t_onset=t[i], peak_speed=pk, t_peak=t[i + ipk],
            dur=iend * dt, dist=np.hypot(*disp),
            dir_end=np.arctan2(disp[1], disp[0]),
            dir_pkvel=np.arctan2(hv[i + ipk, 1], hv[i + ipk, 0]),
            t_end=t[i + iend],
        ))
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load_all()
    K = trial_kinematics(d)
    K['num_barriers'] = d['num_barriers']; K['num_targets'] = d['num_targets']
    K.to_pickle(os.path.join(CACHE, "kinematics.pkl"))
    print(K.describe().to_string())

    t, hp, hv = d['t'], d['hand_pos'], d['hand_vel']
    speed = np.hypot(hv[:, 0], hv[:, 1])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # (a) raw traces, 20 s
    m = (t > 100) & (t < 120)
    ax[0, 0].plot(t[m], hp[m, 0], lw=.8, label='x')
    ax[0, 0].plot(t[m], hp[m, 1], lw=.8, label='y')
    ax[0, 0].plot(t[m], speed[m] / 10, lw=.8, color='k', alpha=.6, label='speed/10')
    for mo in d['move_onset_time'][(d['move_onset_time'] > 100) & (d['move_onset_time'] < 120)]:
        ax[0, 0].axvline(mo, color='r', ls=':', lw=.8)
    ax[0, 0].legend(fontsize=8); ax[0, 0].set_xlabel('time (s)'); ax[0, 0].set_ylabel('mm  /  mm/s')
    ax[0, 0].set_title('hand position & speed (red = move onset)')
    # (b) hand paths coloured by direction
    sel = np.arange(0, len(K), 4)
    cm = plt.get_cmap('hsv')
    for k in sel:
        i = np.searchsorted(t, K.t_onset[k]); j = np.searchsorted(t, K.t_end[k])
        p = hp[i:j] - hp[i]
        ax[0, 1].plot(p[:, 0], p[:, 1], lw=.4, color=cm((K.dir_end[k] % (2*np.pi)) / (2*np.pi)), alpha=.6)
    ax[0, 1].set_aspect('equal'); ax[0, 1].set_title('reach paths (aligned to onset), coloured by endpoint direction')
    ax[0, 1].set_xlabel('x (mm)'); ax[0, 1].set_ylabel('y (mm)')
    # (c) direction distribution
    ax[1, 0].hist(np.degrees(K.dir_end), bins=72, color='steelblue')
    ax[1, 0].set_xlabel('reach direction (deg)'); ax[1, 0].set_ylabel('# trials')
    ax[1, 0].set_title('endpoint direction distribution')
    # (d) speed distribution
    ax[1, 1].hist(K.peak_speed, bins=60, color='indianred')
    ax[1, 1].set_xlabel('peak speed (mm/s)'); ax[1, 1].set_ylabel('# trials')
    ax[1, 1].set_title('peak reach speed distribution')
    fig.tight_layout(); fig.savefig(f"{FIG}/fig01_kinematics.png", dpi=140)
    print("saved")
