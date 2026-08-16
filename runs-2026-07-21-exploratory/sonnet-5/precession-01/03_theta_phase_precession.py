"""
Extract theta phase from the hippocampal LFP, validate the theta rhythm
during running, and test for phase precession in the place cells
identified in 02_preprocess_and_placefields.py: does spike theta phase
advance (precess) as the animal moves through a cell's place field?
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt, hilbert, welch

CACHE_DIR = "cache"
FIG_DIR = "figures"

THETA_BAND = (5.0, 11.0)  # Hz


def bandpass(x, fs, low, high, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def circular_linear_regression(pos, phase_rad, slope_range=(-2 * np.pi, 2 * np.pi)):
    """Kempter et al. (2012) circular-linear regression: find the slope `a`
    (rad per unit position) maximizing the resultant vector length of
    phase - 2*pi*a*pos, then report the circular-linear correlation."""

    def neg_R(a):
        ang = phase_rad - 2 * np.pi * a * pos
        return -np.sqrt(np.mean(np.cos(ang)) ** 2 + np.mean(np.sin(ang)) ** 2)

    res = minimize_scalar(neg_R, bounds=slope_range, method="bounded",
                           options={"xatol": 1e-4})
    a = res.x
    residual = phase_rad - 2 * np.pi * a * pos
    phi0 = np.arctan2(np.mean(np.sin(residual)), np.mean(np.cos(residual)))

    # Jammalamadaka & Sarma circular-circular correlation between the spike
    # phase and the position mapped onto the circle by the fitted slope.
    theta_pos = 2 * np.pi * np.mod(a * pos, 1.0)
    phase_mean = np.angle(np.mean(np.exp(1j * phase_rad)))
    pos_mean = np.angle(np.mean(np.exp(1j * theta_pos)))
    num = np.sum(np.sin(phase_rad - phase_mean) * np.sin(theta_pos - pos_mean))
    den = np.sqrt(np.sum(np.sin(phase_rad - phase_mean) ** 2) * np.sum(np.sin(theta_pos - pos_mean) ** 2))
    rho = num / den
    return a, phi0, rho, -res.fun


def permutation_pvalue(pos, phase_rad, rho_obs, n_perm=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    rhos = np.empty(n_perm)
    for i in range(n_perm):
        shuff = rng.permutation(pos)
        _, _, rho_i, _ = circular_linear_regression(shuff, phase_rad)
        rhos[i] = rho_i
    return (np.sum(np.abs(rhos) >= np.abs(rho_obs)) + 1) / (n_perm + 1)


def main():
    d = np.load(f"{CACHE_DIR}/session_data.npz")
    lin = np.load(f"{CACHE_DIR}/linpos.npz")
    tc = pd.read_pickle(f"{CACHE_DIR}/tuning_curves.pkl")
    place_cells = pd.read_pickle(f"{CACHE_DIR}/place_cells.pkl")
    with open(f"{CACHE_DIR}/units_active.pkl", "rb") as fh:
        units_active = pickle.load(fh)

    lfp, lfp_t, fs = d["lfp"], d["lfp_t"], float(d["lfp_fs"])
    run_epochs = nap.IntervalSet(start=lin["run_starts"], end=lin["run_stops"])
    linpos_tsd = nap.Tsd(t=lin["t"], d=lin["d"], time_support=run_epochs)

    # --- theta band validation: PSD + example trace -------------------
    run_mask = np.zeros_like(lfp_t, dtype=bool)
    for s, e in zip(lin["run_starts"], lin["run_stops"]):
        run_mask |= (lfp_t >= s) & (lfp_t <= e)
    f_psd, Pxx_run = welch(lfp[run_mask], fs=fs, nperseg=2048)
    f_psd2, Pxx_still = welch(lfp[~run_mask], fs=fs, nperseg=2048)

    theta_filt = bandpass(lfp, fs, *THETA_BAND)
    analytic = hilbert(theta_filt)
    theta_phase = np.angle(analytic)  # radians, -pi..pi
    theta_phase_tsd = nap.Tsd(t=lfp_t, d=theta_phase, time_support=nap.IntervalSet(lfp_t[0], lfp_t[-1]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].semilogy(f_psd, Pxx_run, label="running (Pre-Cooling laps)")
    axes[0].semilogy(f_psd2, Pxx_still, label="non-running", alpha=0.7)
    axes[0].axvspan(*THETA_BAND, color="gold", alpha=0.3, label="theta filter band")
    axes[0].set_xlim(0, 20)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power")
    axes[0].set_title("LFP power spectrum (theta reference electrode)")
    axes[0].legend()

    t0 = lin["run_starts"][0]
    m = (lfp_t >= t0) & (lfp_t < t0 + 3.0)
    axes[1].plot(lfp_t[m], lfp[m] - np.mean(lfp[m]), label="raw LFP", color="gray", lw=0.8)
    axes[1].plot(lfp_t[m], theta_filt[m], label=f"{THETA_BAND[0]:.0f}-{THETA_BAND[1]:.0f} Hz filtered", color="C0")
    ax2 = axes[1].twinx()
    ax2.plot(lfp_t[m], np.degrees(theta_phase[m]), color="C3", lw=0.8, alpha=0.6, label="theta phase")
    ax2.set_ylabel("theta phase (deg)", color="C3")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("LFP (a.u.)")
    axes[1].set_title("Example raw vs. theta-filtered LFP with instantaneous phase")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_theta_validation.png", dpi=150)
    plt.close()
    print("Saved figures/02_theta_validation.png")

    # --- phase precession per place cell --------------------------------
    results = []
    example_units = place_cells.index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for i, uid in enumerate(example_units[:8]):
        curve = np.nan_to_num(tc[uid].values, nan=0.0)  # unvisited bins -> 0, not NaN
        bin_centers = tc.index.values
        peak_bin = np.argmax(curve)
        half_max = curve[peak_bin] * 0.2
        # field = contiguous bins around peak above 20% of peak rate
        lo = peak_bin
        while lo > 0 and curve[lo - 1] > half_max:
            lo -= 1
        hi = peak_bin
        while hi < len(curve) - 1 and curve[hi + 1] > half_max:
            hi += 1
        field_lo, field_hi = bin_centers[lo], bin_centers[hi]

        st = units_active[uid]
        st = st[(st >= run_epochs.start.min()) & (st <= run_epochs.end.max())]
        spk = nap.Ts(t=st).restrict(run_epochs)
        spk_pos = spk.value_from(linpos_tsd).values
        spk_phase = spk.value_from(theta_phase_tsd).values

        infield = (spk_pos >= field_lo) & (spk_pos <= field_hi)
        pos_f, phase_f = spk_pos[infield], spk_phase[infield]

        ax = axes[i]
        if len(pos_f) < 15:
            ax.set_title(f"unit {uid}: too few in-field spikes ({len(pos_f)})")
            ax.axis("off")
            continue

        # Bound the fitted slope to at most 2 theta cycles across the field
        # width -- classic precession spans well under one cycle per field
        # pass, and an unconstrained search can overfit noisy, weakly
        # correlated cells with implausibly steep multi-cycle slopes.
        field_width = field_hi - field_lo
        slope_bound = 2.0 / field_width
        a, phi0, rho, R = circular_linear_regression(
            pos_f, phase_f, slope_range=(-slope_bound, slope_bound)
        )
        pval = permutation_pvalue(pos_f, phase_f, rho, n_perm=500)

        phase_deg = np.degrees(phase_f) % 360
        ax.scatter(pos_f, phase_deg, s=10, color="C0", alpha=0.7)
        ax.scatter(pos_f, phase_deg + 360, s=10, color="C0", alpha=0.7)
        xs = np.linspace(field_lo, field_hi, 50)
        line = np.degrees(2 * np.pi * a * xs + phi0) % 360
        # unwrap the fitted line for a clean plotted trend across the doubled y-range
        order = np.argsort(xs)
        line_unwrapped = np.degrees(np.unwrap(np.radians(line[order])))
        ax.plot(xs[order], line_unwrapped, color="C3", lw=2)
        ax.plot(xs[order], line_unwrapped + 360, color="C3", lw=2)
        ax.plot(xs[order], line_unwrapped - 360, color="C3", lw=2)
        ax.set_ylim(0, 720)
        ax.set_xlabel("linearized position (rad)")
        ax.set_ylabel("theta phase (deg)")
        ax.set_title(f"unit {uid}: slope={np.degrees(2*np.pi*a):.0f} deg/rad\nrho={rho:.2f}, p={pval:.3f}, n={len(pos_f)}")

        results.append(dict(unit=uid, slope_rad_per_rad=2 * np.pi * a, rho=rho, R=R, p=pval, n_spikes=len(pos_f),
                             field_lo=field_lo, field_hi=field_hi))

    for j in range(len(example_units), 8):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_phase_precession_per_cell.png", dpi=150)
    plt.close()
    print("Saved figures/03_phase_precession_per_cell.png")

    results_df = pd.DataFrame(results).set_index("unit")
    print("\nPhase precession results:")
    print(results_df)
    results_df.to_pickle(f"{CACHE_DIR}/precession_results.pkl")

    # --- population summary -------------------------------------------
    sig = results_df[results_df["p"] < 0.05]
    print(f"\n{len(sig)}/{len(results_df)} place cells show significant phase-position "
          f"circular-linear correlation (p<0.05, permutation test)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ["C2" if p < 0.05 else "gray" for p in results_df["p"]]
    axes[0].bar(results_df.index.astype(str), results_df["slope_rad_per_rad"], color=colors)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("unit id")
    axes[0].set_ylabel("precession slope (rad phase / rad position)")
    axes[0].set_title("Precession slope per place cell\n(green = p<0.05)")

    axes[1].bar(results_df.index.astype(str), results_df["rho"], color=colors)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("unit id")
    axes[1].set_ylabel("circular-linear correlation (rho)")
    axes[1].set_title("Phase-position correlation per place cell")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_population_summary.png", dpi=150)
    plt.close()
    print("Saved figures/04_population_summary.png")


if __name__ == "__main__":
    main()
