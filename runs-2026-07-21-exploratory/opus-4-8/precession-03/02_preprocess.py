"""Preprocess: running speed/epochs, lap direction, and LFP theta phase.

Produces preprocessed.npz and fig_01_raw_streams.png for validation.
"""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert

THETA_BAND = (6, 12)
SPEED_THRESH = 0.03   # m/s; running when faster than this
MIN_RUN_DUR = 0.5     # s

def theta_phase(lfp, fs):
    b, a = butter(3, [THETA_BAND[0]/(fs/2), THETA_BAND[1]/(fs/2)], btype='band')
    filt = filtfilt(b, a, lfp)
    analytic = hilbert(filt)
    return filt, np.angle(analytic), np.abs(analytic)

def main():
    d = np.load('loaded_data.npz', allow_pickle=True)
    pos_data, pos_t = d['pos_data'], d['pos_t']
    lfp, lfp_t, fs = d['lfp'], d['lfp_t'], float(d['lfp_rate'])

    # --- position as a pynapple Tsd; smooth and compute speed ---
    # handle NaNs by interpolation
    good = ~np.isnan(pos_data)
    pos_data = np.interp(pos_t, pos_t[good], pos_data[good])
    pos = nap.Tsd(t=pos_t, d=pos_data)
    # smooth position a little (std ~ 100 ms) then differentiate for speed
    pos_smooth = pos.smooth(0.1)
    dt = np.median(np.diff(pos_t))
    speed_vals = np.abs(np.gradient(pos_smooth.values, pos_t))
    speed = nap.Tsd(t=pos_t, d=speed_vals)
    print(f"Speed: median={np.median(speed_vals):.3f}, 90th pct={np.percentile(speed_vals,90):.3f} m/s")

    # --- running epochs (speed threshold) ---
    run_ep = speed.threshold(SPEED_THRESH, method='above').time_support
    run_ep = run_ep.drop_short_intervals(MIN_RUN_DUR).merge_close_intervals(0.2)
    print(f"Running epochs: {len(run_ep)} intervals, {run_ep.tot_length():.0f} s total")

    # --- lap direction: sign of smoothed position derivative ---
    vel = np.gradient(pos_smooth.values, pos_t)   # signed velocity
    direction = nap.Tsd(t=pos_t, d=np.sign(vel))

    # --- theta phase from LFP ---
    filt, phase, amp = theta_phase(lfp, fs)
    theta_filt = nap.Tsd(t=lfp_t, d=filt)
    theta_ph = nap.Tsd(t=lfp_t, d=phase)      # radians in (-pi, pi]
    theta_amp = nap.Tsd(t=lfp_t, d=amp)

    # --- validation figure ---
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    tsel = (pos_t > pos_t[0] + 300) & (pos_t < pos_t[0] + 330)   # 30 s window
    ax[0].plot(pos_t[tsel], pos_data[tsel], 'k')
    ax[0].set_ylabel('Position (m)')
    ax[0].set_title('Linearized position on 1.6 m maze (30 s window)')
    ax[1].plot(pos_t[tsel], speed_vals[tsel], 'b')
    ax[1].axhline(SPEED_THRESH, color='r', ls='--', label=f'{SPEED_THRESH} m/s')
    ax[1].set_ylabel('Speed (m/s)'); ax[1].legend(loc='upper right')
    # LFP + theta over a 2 s window
    lsel = (lfp_t > pos_t[0] + 300) & (lfp_t < pos_t[0] + 302)
    ax[2].plot(lfp_t[lsel], lfp[lsel]*1e3, color='0.6', lw=0.8, label='LFP')
    ax[2].plot(lfp_t[lsel], filt[lsel]*1e3, 'r', lw=1.2, label='theta 6-12 Hz')
    ax[2].set_ylabel('LFP (mV)'); ax[2].set_xlabel('Time (s)')
    ax[2].legend(loc='upper right'); ax[2].set_xlim(pos_t[0]+300, pos_t[0]+302)
    ax[2].set_title('CA1 LFP and theta-band filtered signal (2 s window)')
    plt.tight_layout()
    plt.savefig('fig_01_raw_streams.png', dpi=130)
    print("Saved fig_01_raw_streams.png")

    np.savez_compressed(
        'preprocessed.npz',
        pos_t=pos_t, pos_data=pos_data, speed=speed_vals, direction=direction.values,
        run_starts=run_ep.start, run_ends=run_ep.end,
        lfp_t=lfp_t, theta_filt=filt, theta_phase=phase, theta_amp=amp,
    )
    print("Saved preprocessed.npz")

if __name__ == '__main__':
    main()
