"""Theta-band signal processing: channel selection, filtering, phase, statistics."""

import numpy as np
import pynapple as nap
from scipy.signal import hilbert, welch

import hc11_io as io

THETA_BAND = (6.0, 10.0)
DELTA_BAND = (1.0, 4.0)
RUN_SPEED = 0.10  # m/s; standard threshold for locomotor theta


def run_epochs(speed, maze, thresh=RUN_SPEED, min_dur=0.5):
    """Periods of sustained running inside the maze epoch."""
    ep = speed.restrict(maze).threshold(thresh, "above").time_support
    ep = ep.merge_close_intervals(0.25)
    return ep[(ep.end - ep.start) >= min_dur]


def theta_delta_ratio(sig, fs=io.LFP_RATE):
    """Ratio of 6-10 Hz to 1-4 Hz power in a raw LFP segment (Welch PSD)."""
    f, p = welch(sig, fs=fs, nperseg=int(4 * fs))
    th = p[(f >= THETA_BAND[0]) & (f <= THETA_BAND[1])].mean()
    de = p[(f >= DELTA_BAND[0]) & (f <= DELTA_BAND[1])].mean()
    return th / de, f, p


def select_theta_channel(h5, run_ep, n_channels=128, probe_seconds=120):
    """Pick the LFP channel with the strongest theta/delta ratio during running.

    A contiguous probe window drawn from the running epochs is read for every
    channel; the channel maximising theta/delta power is used for all downstream
    phase estimates (in CA1 this lands in or near the pyramidal layer/fissure).
    """
    # Assemble a probe window that overlaps as much running as possible.
    t_start = float(run_ep.start[0])
    probe = nap.IntervalSet(start=t_start, end=t_start + probe_seconds)
    block = io.load_lfp_block(h5, probe, channels=np.arange(n_channels))
    block_run = block.restrict(run_ep)
    ratios = np.array(
        [theta_delta_ratio(block_run[:, i].values)[0] for i in range(n_channels)]
    )
    return int(np.argmax(ratios)), ratios


def theta_phase(lfp, band=THETA_BAND, fs=io.LFP_RATE):
    """Band-pass filter the LFP and return (filtered Tsd, phase Tsd, amplitude Tsd).

    Phase is in radians on [0, 2*pi) with 0 = peak of the filtered theta cycle
    (the convention used throughout this analysis; trough = pi).
    """
    filt = nap.apply_bandpass_filter(lfp, band, fs=fs)
    # Pynapple filters each interval of the time support separately; the Hilbert
    # transform has to respect the same boundaries or phase would wrap across gaps.
    phase = np.empty(len(filt))
    amp = np.empty(len(filt))
    for ep in filt.time_support:
        slc = filt.get_slice(start=ep.start[0], end=ep.end[0])
        analytic = hilbert(filt.values[slc])
        phase[slc] = np.mod(np.angle(analytic), 2 * np.pi)
        amp[slc] = np.abs(analytic)
    ts = filt.time_support
    return (filt,
            nap.Tsd(t=filt.t, d=phase, time_support=ts),
            nap.Tsd(t=filt.t, d=amp, time_support=ts))


def circular_stats(phases):
    """Mean resultant length, preferred phase, Rayleigh p, and n."""
    n = len(phases)
    if n == 0:
        return dict(n=0, mrl=np.nan, pref=np.nan, p=np.nan, z=np.nan)
    r = np.exp(1j * phases).mean()
    mrl = np.abs(r)
    pref = np.mod(np.angle(r), 2 * np.pi)
    z = n * mrl**2
    # Zar (1999) approximation to the Rayleigh test p-value.
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * mrl) ** 2)) - (1 + 2 * n))
    return dict(n=n, mrl=mrl, pref=pref, p=p, z=z)


def spike_phases(unit_ts, phase_tsd, ep):
    """Theta phase at each spike time of one unit, restricted to `ep`.

    `value_from` takes the phase sample nearest each spike rather than
    interpolating, which avoids the artefact linear interpolation would
    introduce at the 2*pi wrap. At 1250 Hz the residual error is under 0.03 rad
    for an 8 Hz oscillation.
    """
    ts = unit_ts.restrict(ep)
    if len(ts) == 0:
        return np.array([])
    ph = ts.value_from(phase_tsd).values
    return ph[~np.isnan(ph)]


class PhaseLookup:
    """Concatenated theta-phase series used for fast spike-phase shuffling.

    The intervals of `ep` are laid end to end, so a random circular rotation of
    the phase series relative to the spike train destroys the spike-phase
    relationship while preserving spike count, spike-train autocorrelation, and
    the marginal distribution of theta phase. This is the null used to test
    phase locking; a plain Rayleigh test would be anti-conservative here because
    consecutive spikes within a burst are not independent.
    """

    def __init__(self, phase_tsd, ep):
        self.ep = ep
        self.phase_t = phase_tsd.t
        self.phase_v = np.exp(1j * phase_tsd.values)
        self.m = len(self.phase_t)

    def spike_index(self, unit_ts):
        ts = unit_ts.restrict(self.ep).t
        if len(ts) == 0:
            return np.array([], dtype=int)
        idx = np.searchsorted(self.phase_t, ts)
        return np.clip(idx, 0, self.m - 1)

    def null_mrl(self, unit_ts, n_shuffle=200, rng=None):
        rng = rng or np.random.default_rng(0)
        idx = self.spike_index(unit_ts)
        if len(idx) == 0:
            return np.full(n_shuffle, np.nan)
        shifts = rng.integers(0, self.m, size=n_shuffle)
        return np.array(
            [np.abs(self.phase_v[(idx + k) % self.m].mean()) for k in shifts]
        )
