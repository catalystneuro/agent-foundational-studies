"""Per-session theta phase-entrainment analysis for DANDI:000059.

The measurement: for every curated unit, take the theta phase of the LFP at the
time of each spike (during locomotion), then test whether that circular
distribution departs from uniform. Chance level for the mean resultant length is
set by a spike-count-matched shuffle in which each spike train is circularly
shifted within the analysis-epoch timeline, which preserves each unit's spike
count and its inter-spike-interval structure but destroys its alignment to the LFP.
"""

import numpy as np
import pynapple as nap
from tqdm import tqdm

import theta_lib as tl

N_SHUFFLE = 200
NORMAL_TEMP = 34.0   # °C, septum at physiological temperature
COOLED_TEMP = 30.0   # °C, septum cooled


# ---------------------------------------------------------------------------
# Circular time-shift shuffle
# ---------------------------------------------------------------------------

def _compress_times(t, ep):
    """Map times inside an IntervalSet onto a contiguous timeline starting at 0."""
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    durs = ends - starts
    offsets = np.concatenate([[0], np.cumsum(durs)[:-1]])
    i = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(starts) - 1)
    return offsets[i] + (t - starts[i]), float(durs.sum())


def _decompress_times(tc, ep):
    """Inverse of ``_compress_times`` (works on arrays of any shape)."""
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    durs = ends - starts
    offsets = np.concatenate([[0], np.cumsum(durs)]).astype(float)
    i = np.clip(np.searchsorted(offsets, tc, side="right") - 1, 0, len(starts) - 1)
    return starts[i] + (tc - offsets[i])


def shuffled_mrl(spike_times, phase_at, ep, n=N_SHUFFLE, rng=None, chunk=25):
    """Null distribution of the mean resultant length under circular time shifts."""
    rng = rng or np.random.default_rng(0)
    tc, total = _compress_times(spike_times, ep)
    shifts = rng.uniform(1.0, total - 1.0, size=n)
    out = np.empty(n)
    for a in range(0, n, chunk):
        s = shifts[a:a + chunk][:, None]
        t_shift = _decompress_times(np.mod(tc[None, :] + s, total), ep)
        ph = phase_at(t_shift)
        out[a:a + len(s)] = np.abs(np.mean(np.exp(1j * ph), axis=1))
    return out


# ---------------------------------------------------------------------------
# Per-unit statistics
# ---------------------------------------------------------------------------

def burst_index(spikes_tsgroup, ep):
    """ACG-based burstiness: 3-5 ms mass relative to the 40-50 ms baseline.

    Used, together with firing rate, to separate putative pyramidal cells from
    putative interneurons (no spike waveforms are distributed with this dandiset).
    """
    acg = nap.compute_autocorrelogram(spikes_tsgroup, binsize=0.001, windowsize=0.05, ep=ep, norm=True)
    lags = np.asarray(acg.index)
    v = np.asarray(acg.values)
    near = (lags >= 0.003) & (lags <= 0.005)
    far = (lags >= 0.040) & (lags <= 0.050)
    return v[near].mean(axis=0) / np.maximum(v[far].mean(axis=0), 1e-9)


def phase_lock_stats(units, phase_at, ep, session="", n_shuffle=N_SHUFFLE, seed=0,
                     min_spikes=tl.MIN_SPIKES, with_burst=True, verbose=True, keep_phases=True):
    """Phase-locking statistics for every unit with enough spikes inside ``ep``."""
    rng = np.random.default_rng(seed)
    rec = []
    dur = ep.tot_length()
    for uid in tqdm(list(units.index), desc=f"{session} phase locking", disable=not verbose):
        sp = units[uid].restrict(ep)
        n = len(sp)
        if n < min_spikes:
            continue
        ph = phase_at(sp.t)
        r, z, p = tl.rayleigh(ph)
        null = shuffled_mrl(sp.t, phase_at, ep, n=n_shuffle, rng=rng)
        d = dict(
            session=session, unit=int(uid), n_spikes=n, rate=n / dur,
            mrl=r, mrl_unbiased=tl.circ_r_unbiased(ph),
            pref_phase=tl.circ_mean(ph), rayleigh_z=z, rayleigh_p=p,
            mrl_null_mean=float(null.mean()), mrl_null_p95=float(np.percentile(null, 95)),
            mrl_z=float((r - null.mean()) / max(null.std(), 1e-12)),
            shuffle_p=float((np.sum(null >= r) + 1) / (len(null) + 1)),
        )
        if with_burst:
            d["burst"] = float(burst_index(nap.TsGroup({0: sp}), ep)[0])
        if keep_phases:
            d["phases"] = ph
        rec.append(d)
    return rec


def classify(rec):
    """Putative interneuron vs pyramidal cell from firing rate and ACG burstiness."""
    return "interneuron" if (rec["rate"] > 8.0 and rec.get("burst", 99) < 1.6) else "pyramidal"


# ---------------------------------------------------------------------------
# Session loading + derived signals
# ---------------------------------------------------------------------------

def load_session(session, verbose=True):
    """Load spikes, behaviour and the theta-reference LFP, and derive theta phase."""
    fp, fr = tl.open_session_files(session)
    align = tl.check_alignment(fp, fr)
    assert 0.9 < align["ratio"] <= 1.0, f"{session}: clocks disagree ({align})"

    units = tl.load_units(fp)
    speed, pos, temp = tl.load_behavior(fp)
    trials, cooling, condition = tl.load_trials(fp)
    col, row = tl.theta_reference_column(fr)
    t0, t1 = float(speed.t[0]), float(speed.t[-1])
    lfp, fs = tl.load_lfp(fr, t0, t1, column=col)
    filt, phase = tl.theta_phase(lfp, fs)
    env = nap.compute_hilbert_envelope(filt)
    run = tl.running_epochs(speed).intersect(nap.IntervalSet(lfp.t[0], lfp.t[-1]))

    # The LFP slice is contiguous and uniformly sampled, so the phase at an arbitrary
    # time is a direct index lookup (at 1250 Hz one sample is 0.8 ms, i.e. <0.5 deg of
    # theta). That is far cheaper than np.interp, which matters for the shuffles.
    phase_d = phase.d
    lfp_t0 = float(phase.t[0])
    n_phase = phase_d.size

    def phase_at(t):
        idx = np.clip(np.rint((np.asarray(t) - lfp_t0) * fs).astype(np.int64), 0, n_phase - 1)
        return phase_d[idx]

    # Septal temperature splits the session into normal and cooled theta.
    if temp is not None:
        normal = temp.threshold(NORMAL_TEMP, method="above").time_support
        cooled = temp.threshold(COOLED_TEMP, method="below").time_support
    else:
        normal = nap.IntervalSet(trials.start[cooling != "Cooling on"], trials.end[cooling != "Cooling on"])
        cooled = nap.IntervalSet(trials.start[cooling == "Cooling on"], trials.end[cooling == "Cooling on"])
    normal = normal.drop_short_intervals(1.0)
    cooled = cooled.drop_short_intervals(1.0)

    out = dict(session=session, units=units, speed=speed, pos=pos, temp=temp,
               trials=trials, cooling=cooling, condition=condition,
               lfp=lfp, filt=filt, phase=phase, env=env, fs=fs, phase_at=phase_at,
               run=run, run_normal=run.intersect(normal), run_cooled=run.intersect(cooled),
               align=align, theta_ref_row=row, theta_ref_col=col, t0=t0, t1=t1)
    if verbose:
        print(f"{session}: {len(units)} curated units, run {run.tot_length():.0f} s "
              f"(normal {out['run_normal'].tot_length():.0f} s, cooled {out['run_cooled'].tot_length():.0f} s)")
    return out


def theta_frequency(phase, ep, fs):
    """Instantaneous theta frequency (Hz) from the derivative of the unwrapped phase."""
    freqs = []
    for s, e in zip(np.asarray(ep.start), np.asarray(ep.end)):
        seg = phase.restrict(nap.IntervalSet(s, e))
        if len(seg) < int(fs):
            continue
        f = np.diff(np.unwrap(seg.d)) * fs / (2 * np.pi)
        freqs.append(f[(f > 2) & (f < 15)])
    return np.concatenate(freqs) if freqs else np.array([])


def cycle_frequency(phase, ep, fs, fmin=4.0, fmax=13.0):
    """Theta frequency measured cycle by cycle, as 1 / (duration of each 2*pi cycle).

    Sample-by-sample differentiation of the phase is quantised at the LFP sampling
    rate and produces a comb-like frequency histogram; one estimate per cycle does not.
    """
    out = []
    for s, e in zip(np.asarray(ep.start), np.asarray(ep.end)):
        seg = phase.restrict(nap.IntervalSet(s, e))
        if len(seg) < int(fs):
            continue
        wraps = np.where(np.diff(seg.d) < -np.pi)[0]  # 2*pi -> 0 transitions
        if len(wraps) < 2:
            continue
        f = fs / np.diff(wraps)
        out.append(f[(f > fmin) & (f < fmax)])
    return np.concatenate(out) if out else np.array([])


def phase_histogram(phases, nbins=36):
    """Normalised spike-phase histogram (sums to 1) and its bin centres."""
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    h, _ = np.histogram(phases, bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), h / max(h.sum(), 1)
