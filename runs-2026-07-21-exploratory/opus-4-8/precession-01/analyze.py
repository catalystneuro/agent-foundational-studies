"""Core phase-precession analysis.

Pipeline:
  1. Build clean position, running speed, and directional run epochs.
  2. Compute 1-D place fields (firing rate vs position) per movement direction.
  3. Select place cells (peak rate, spatial information, single field).
  4. For place-field spikes, pair theta phase with within-field position.
  5. Fit circular-linear regression (Kempter et al. 2012) to quantify precession.
"""
import numpy as np
import pynapple as nap
from preprocess import load, compute_speed, theta_phase

SPEED_TH = 0.10      # m/s, running threshold
MIN_PEAK_RATE = 3.0  # Hz, place-field peak
MIN_SI = 0.3         # bits/spike spatial information
N_POS_BINS = 60
FIELD_FRAC = 0.20    # field = bins where rate > 20% of peak


# ---------------------------------------------------------------- circular-linear
def circ_lin_fit(x, phi, slope_range=(-2.0, 2.0), n=2001):
    """Kempter et al. (2012) circular-linear regression.
    x normalized to [0,1]; returns slope (cycles/unit-x), phase0 (rad), rho, p."""
    x = np.asarray(x); phi = np.asarray(phi)
    slopes = np.linspace(*slope_range, n)
    R = np.array([np.abs(np.mean(np.exp(1j * (phi - 2 * np.pi * s * x)))) for s in slopes])
    a = slopes[np.argmax(R)]
    phase0 = np.angle(np.mean(np.exp(1j * (phi - 2 * np.pi * a * x))))
    # circular-linear correlation coefficient (Kempter 2012, eq. for rho_cl)
    theta = 2 * np.pi * np.abs(a) * x
    sx = np.sin(x * 2 * np.pi * a - np.mean(x * 2 * np.pi * a))  # not used directly
    phi_bar = np.angle(np.sum(np.exp(1j * phi)))
    theta_bar = np.angle(np.sum(np.exp(1j * theta)))
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta - theta_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta - theta_bar) ** 2))
    # theta = 2*pi*|a|*x increases with distance travelled, so this correlation
    # is negative for a precessing cell (phase falls as the animal crosses the field)
    rho = num / den if den > 0 else 0.0
    # significance via permutation
    nperm = 500
    cnt = 0
    for _ in range(nperm):
        pp = np.random.permutation(phi)
        pb = np.angle(np.sum(np.exp(1j * pp)))
        nm = np.sum(np.sin(pp - pb) * np.sin(theta - theta_bar))
        rp = nm / den if den > 0 else 0.0
        if abs(rp) >= abs(rho):
            cnt += 1
    p = (cnt + 1) / (nperm + 1)
    return a, phase0, rho, p


def spatial_information(rate, occ):
    """Skaggs spatial information in bits/spike."""
    occ = occ / occ.sum()
    mean_rate = np.sum(rate * occ)
    if mean_rate <= 0:
        return 0.0
    r = rate / mean_rate
    nz = rate > 0
    return float(np.sum(occ[nz] * r[nz] * np.log2(r[nz])))


def build(z, maze, pos, lfp, spikes):
    speed, vel, dt = compute_speed(pos)
    theta, ph, amp, fs = theta_phase(lfp)

    # clean interpolated position (fill NaN) as a Tsd for downstream use
    t = pos.index.values
    x = pos.values.copy()
    good = ~np.isnan(x)
    x = np.interp(t, t[good], x[good])
    cpos = nap.Tsd(t=t, d=x, time_support=maze)

    # directional run epochs
    run_r = vel.threshold(SPEED_TH).time_support        # rightward (increasing pos)
    run_l = (vel * -1).threshold(SPEED_TH).time_support  # leftward
    run_r = run_r.drop_short_intervals(0.3)
    run_l = run_l.drop_short_intervals(0.3)
    print("rightward run time %.0f s, leftward %.0f s" %
          (run_r.tot_length(), run_l.tot_length()))
    return dict(speed=speed, vel=vel, theta=theta, ph=ph, amp=amp, fs=fs,
                cpos=cpos, run_r=run_r, run_l=run_l)


def place_fields(spikes, cpos, run_ep):
    """1-D tuning curves + occupancy restricted to run epochs of one direction."""
    bins = np.linspace(0, 1.6, N_POS_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    p = cpos.restrict(run_ep)
    dt = np.median(np.diff(cpos.index.values))
    occ, _ = np.histogram(p.values, bins=bins)
    occ = occ * dt  # seconds per bin
    tc = {}
    si = {}
    for u in spikes.keys():
        s = spikes[u].restrict(run_ep)
        if len(s) < 20:
            continue
        sp_pos = np.interp(s.index.values, cpos.index.values, cpos.values)  # pos at spike
        cnt, _ = np.histogram(sp_pos, bins=bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(occ > 0.05, cnt / occ, 0.0)
        # light smoothing
        rate = np.convolve(rate, np.ones(3) / 3, mode="same")
        tc[u] = rate
        si[u] = spatial_information(rate, occ + 1e-9)
    return centers, occ, tc, si


def select_place_cells(centers, tc, si):
    keep = {}
    for u, rate in tc.items():
        peak = rate.max()
        if peak < MIN_PEAK_RATE or si[u] < MIN_SI:
            continue
        thr = FIELD_FRAC * peak
        above = rate >= thr
        # find the contiguous field containing the peak
        pk = int(np.argmax(rate))
        lo = pk
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = pk
        while hi < len(rate) - 1 and above[hi + 1]:
            hi += 1
        width = centers[hi] - centers[lo]
        if width < 0.10 or width > 0.9:  # reject too-narrow / whole-track
            continue
        keep[u] = dict(peak=peak, si=si[u], pk_pos=centers[pk],
                       lo=centers[lo], hi=centers[hi], width=width)
    return keep


if __name__ == "__main__":
    z, maze, pos, lfp, spikes = load()
    B = build(z, maze, pos, lfp, spikes)
    for name, ep in [("rightward", B["run_r"]), ("leftward", B["run_l"])]:
        centers, occ, tc, si = place_fields(spikes, B["cpos"], ep)
        pc = select_place_cells(centers, tc, si)
        print(f"{name}: {len(pc)} place cells (of {len(tc)} active)")
