"""Load cached maze data into pynapple objects, compute running speed and
movement-direction run epochs, extract theta phase from the LFP, and produce
validation plots of the raw streams.
"""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert

CACHE = "achilles_maze_cache.npz"


def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def load():
    z = np.load(CACHE, allow_pickle=True)
    maze = nap.IntervalSet(start=float(z["maze_start"]), end=float(z["maze_stop"]))
    pos = nap.Tsd(t=z["pos_t"], d=z["pos"], time_support=maze)
    lfp = nap.Tsd(t=z["lfp_t"], d=z["lfp"].astype(float), time_support=maze)
    trains = z["spike_trains"]
    meta = z["unit_meta"]
    spikes = nap.TsGroup(
        {i: nap.Ts(t=np.asarray(trains[i], dtype=float)) for i in range(len(trains))},
        time_support=maze,
    )
    spikes.set_info(
        orig_id=np.array([m[0] for m in meta]),
        location=np.array([m[2] for m in meta]),
        n_spk=np.array([m[3] for m in meta]),
    )
    return z, maze, pos, lfp, spikes


def compute_speed(pos, sigma_s=0.25):
    """Smoothed absolute running speed (m/s) on the position grid."""
    t = pos.index.values
    x = pos.values.copy()
    # fill occasional NaNs by interpolation
    good = ~np.isnan(x)
    x = np.interp(t, t[good], x[good])
    dt = np.median(np.diff(t))
    # gaussian smooth position then differentiate
    xs = nap.Tsd(t=t, d=x).smooth(sigma_s).values
    v = np.gradient(xs, t)
    speed = nap.Tsd(t=t, d=np.abs(v), time_support=pos.time_support)
    vel = nap.Tsd(t=t, d=v, time_support=pos.time_support)
    return speed, vel, dt


def theta_phase(lfp):
    fs = 1.0 / np.median(np.diff(lfp.index.values))
    filt = bandpass(lfp.values, 6, 12, fs)
    analytic = hilbert(filt)
    phase = np.angle(analytic)  # -pi..pi, 0=peak
    amp = np.abs(analytic)
    theta = nap.Tsd(t=lfp.index.values, d=filt, time_support=lfp.time_support)
    ph = nap.Tsd(t=lfp.index.values, d=phase, time_support=lfp.time_support)
    am = nap.Tsd(t=lfp.index.values, d=amp, time_support=lfp.time_support)
    return theta, ph, am, fs


if __name__ == "__main__":
    z, maze, pos, lfp, spikes = load()
    print("maze", maze, "n_spikes_groups", len(spikes))
    print("pos range", np.nanmin(pos.values), np.nanmax(pos.values))
    speed, vel, dt = compute_speed(pos)
    theta, ph, amp, fs = theta_phase(lfp)
    print("LFP fs", round(fs, 1), "theta chan", int(z["best_chan"]))

    # ---- validation figure ----
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    t0 = maze.start[0]
    # (a) position over a 60 s window + speed
    w = (pos.index.values >= t0 + 100) & (pos.index.values <= t0 + 160)
    ax[0].plot(pos.index.values[w] - t0, pos.values[w], "k", lw=1)
    ax[0].set_ylabel("linear pos (m)")
    ax[0].set_title("(a) Linearized position on 1.6 m track (60 s window)")
    axb = ax[0].twinx()
    axb.plot(speed.index.values[w] - t0, speed.values[w], "tab:red", lw=0.8, alpha=0.6)
    axb.set_ylabel("speed (m/s)", color="tab:red")

    # (b) LFP raw + theta band, 2 s
    wl = (lfp.index.values >= t0 + 120) & (lfp.index.values <= t0 + 122)
    ax[1].plot(lfp.index.values[wl] - t0, lfp.values[wl] * 1e3, color="0.6", lw=0.8, label="raw LFP")
    ax[1].plot(theta.index.values[wl] - t0, theta.values[wl] * 1e3, "tab:blue", lw=1.5, label="6-12 Hz")
    ax[1].set_ylabel("LFP (mV)")
    ax[1].set_title(f"(b) CA1 LFP and theta band (channel {int(z['best_chan'])})")
    ax[1].legend(loc="upper right", fontsize=8)

    # (c) raster of first 30 units over the same 60 s
    order = np.argsort(spikes.get_info("n_spk"))[::-1][:30]
    for row, ui in enumerate(order):
        st = spikes[int(ui)].index.values
        st = st[(st >= t0 + 100) & (st <= t0 + 160)]
        ax[2].plot(st - t0, np.full_like(st, row), "|", color="k", ms=3, mew=0.5)
    ax[2].set_ylabel("unit (by rate)")
    ax[2].set_xlabel("time from window start (s)")
    ax[2].set_title("(c) Spike raster, 30 most active excitatory units")
    plt.tight_layout()
    plt.savefig("fig1_raw_data_validation.png", dpi=130)
    print("saved fig1_raw_data_validation.png")
