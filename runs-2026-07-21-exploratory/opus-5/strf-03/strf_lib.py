"""Shared loading and analysis helpers for the DANDI:000986 STRF analysis.

DANDI:000986 -- "Auditory cortex Neuropixels recordings and pupil diameter traces
from mice during passive exposure to pure tones" (Jo & McCormick, U. Oregon).

Each session presents a randomized sequence of 25 ms pure tones (2, 4, 8, 16, 32 kHz,
60 dB SPL) at a fixed 0.805 s inter-onset interval, in ~5 min tone blocks that
alternate with 5 min blocks of silence.  Spike-sorted units, pupil diameter and
running speed are stored in the NWB file.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000986"
VERSION = "0.251031.1939"
API = "https://api.dandiarchive.org/api"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")
FIGDIR = os.path.dirname(os.path.abspath(__file__))

# Stimulus set (Hz), fixed across every session in the dandiset.
FREQS = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])
TONE_DUR = 0.025  # s


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def list_assets():
    """Return {path: asset_id} for every NWB file in the dandiset."""
    r = requests.get(
        f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/",
        params={"page_size": 100},
    )
    r.raise_for_status()
    return {a["path"]: a["asset_id"] for a in r.json()["results"]}


def asset_url(asset_id):
    return f"{API}/dandisets/{DANDISET}/versions/{VERSION}/assets/{asset_id}/download/"


def open_nwb(asset_id):
    """Stream an NWB file from the DANDI S3 bucket with a local disk cache."""
    rem = remfile.File(asset_url(asset_id), disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    return io.read()


def tone_blocks(trial_starts, max_gap=5.0, pad=1.0):
    """IntervalSet covering the tone-presentation blocks (gaps > `max_gap` split blocks)."""
    breaks = np.flatnonzero(np.diff(trial_starts) > max_gap)
    starts = np.concatenate([[trial_starts[0]], trial_starts[breaks + 1]])
    ends = np.concatenate([trial_starts[breaks], [trial_starts[-1]]]) + pad
    return nap.IntervalSet(start=starts, end=ends)


def load_session(asset_id, with_behavior=True):
    """Load one session into pynapple objects.

    Returns a dict with:
        units    : TsGroup of spike trains
        trials   : DataFrame of tone presentations (start_time, stim_frequency, ...)
        blocks   : IntervalSet of tone-presentation blocks
        events   : {frequency: nap.Ts of tone onsets}
        pupil    : Tsd of pupil diameter (or None)
        running  : Tsd of running speed (or None)
        meta     : session metadata
    """
    nwbfile = open_nwb(asset_id)
    nwb = nap.NWBFile(nwbfile)

    units = nwb["units"]
    trials = nwbfile.trials.to_dataframe()
    blocks = tone_blocks(trials.start_time.values)
    events = {
        f: nap.Ts(t=trials.start_time.values[trials.stim_frequency.values == f])
        for f in FREQS
    }

    pupil = running = None
    if with_behavior:
        beh = nwbfile.processing["behavior"]
        p = beh["PupilTracking"]["pupil_diameter"]
        pupil = nap.Tsd(t=p.timestamps[:], d=p.data[:])
        rs = beh["running_speed"]
        running = nap.Tsd(t=rs.timestamps[:], d=rs.data[:])

    meta = dict(
        subject=nwbfile.subject.subject_id,
        session=str(nwbfile.session_id),
        n_units=len(units),
        n_trials=len(trials),
        duration=float(trials.start_time.max()),
    )
    return dict(
        units=units,
        trials=trials,
        blocks=blocks,
        events=events,
        pupil=pupil,
        running=running,
        meta=meta,
        nwbfile=nwbfile,
    )


# --------------------------------------------------------------------------- #
# Stimulus representation
# --------------------------------------------------------------------------- #
def stimulus_tsdframe(trials, blocks, binsize=0.005):
    """Binary spectrogram of the tone sequence: one column per frequency channel.

    Sampled on a regular grid at `binsize`, restricted to the tone blocks.  A bin is
    1 when a tone of that frequency is playing (25 ms boxcar from onset).
    """
    t0, t1 = blocks.start[0], blocks.end[-1]
    t = np.arange(t0, t1, binsize)
    S = np.zeros((t.size, FREQS.size))
    n_bins_on = max(1, int(round(TONE_DUR / binsize)))
    for j, f in enumerate(FREQS):
        onsets = trials.start_time.values[trials.stim_frequency.values == f]
        idx = np.searchsorted(t, onsets)
        idx = idx[(idx >= 0) & (idx < t.size)]
        for k in range(n_bins_on):
            ii = np.clip(idx + k, 0, t.size - 1)
            S[ii, j] = 1.0
    stim = nap.TsdFrame(t=t, d=S, columns=[f"{f/1000:g}kHz" for f in FREQS])
    return stim.restrict(blocks)


# --------------------------------------------------------------------------- #
# STRF estimation
# --------------------------------------------------------------------------- #
def strf_psth(units, events, blocks, binsize=0.005, window=(-0.15, 0.30)):
    """Frequency-resolved PSTH -> STRF.

    Returns
    -------
    centers  : (n_lags,) time relative to tone onset (s)
    R        : (n_units, n_freqs, n_lags) firing rate in spikes/s
    n_trials : (n_freqs,) number of presentations of each frequency
    """
    edges = np.arange(window[0], window[1] + binsize / 2, binsize)
    centers = edges[:-1] + binsize / 2
    keys = list(units.index)
    R = np.zeros((len(keys), len(FREQS), centers.size))
    n_trials = np.zeros(len(FREQS))
    for j, f in enumerate(FREQS):
        ev = events[f].restrict(blocks).t
        n_trials[j] = ev.size
        for i, k in enumerate(keys):
            R[i, j] = _peri_counts(units[k].t, ev, edges) / (ev.size * binsize)
    return centers, R, n_trials


def _peri_counts(spike_times, event_times, lag_edges):
    """Histogram of spike times relative to a set of events."""
    if event_times.size == 0:
        return np.zeros(lag_edges.size - 1)
    lo = np.searchsorted(spike_times, event_times + lag_edges[0])
    hi = np.searchsorted(spike_times, event_times + lag_edges[-1])
    rel = np.concatenate([spike_times[a:b] - e for a, b, e in zip(lo, hi, event_times)])
    counts, _ = np.histogram(rel, bins=lag_edges)
    return counts.astype(float)


def _forward_sum(x, K):
    """sum_{k=0}^{K-1} x[i+k] for every i (zero-padded at the end)."""
    xp = np.concatenate([x, np.zeros(K - 1)])
    c = np.cumsum(np.concatenate([[0.0], xp]))
    return c[K:] - c[:-K]


def strf_sta(R, n_trials, n_spikes, block_duration, binsize=0.005):
    """Reverse-correlation STRF: mean-subtracted spike-triggered average.

    For a binary stimulus the STA has a closed form.  With ``n(t)`` the binned spike
    train, ``S_f(t)`` the stimulus channel, ``N`` the total spike count and ``K`` the
    tone duration in bins,

        STA_f(u) = (1/N) sum_t n(t) S_f(t-u) = (1/N) sum_{k=0}^{K-1} C_f(u+k),

    where ``C_f(v)`` is the total peri-tone spike count at lag ``v``.  The STA is the
    frequency-resolved peri-tone histogram summed over the tone boxcar and normalised
    by the spike count, which avoids materialising a 10^6 x 10^2 Hankel matrix.
    Subtracting the marginal probability P(S_f = 1) makes 0 mean "no preference".
    """
    K = max(1, int(round(TONE_DUR / binsize)))
    total = R * binsize * n_trials[None, :, None]  # spikes per bin summed over trials
    sm = np.empty_like(total)
    for i in range(total.shape[0]):
        for j in range(total.shape[1]):
            sm[i, j] = _forward_sum(total[i, j], K)
    sta = sm / n_spikes[:, None, None]
    p_marg = n_trials * TONE_DUR / block_duration  # fraction of time channel f is on
    return sta - p_marg[None, :, None]


# --------------------------------------------------------------------------- #
# Response metrics
# --------------------------------------------------------------------------- #
def response_metrics(centers, R, base_win=(-0.1, 0.0), resp_win=(0.0, 0.1)):
    """Per-unit tone responsiveness, best frequency, latency and tuning width."""
    bi = (centers >= base_win[0]) & (centers < base_win[1])
    ri = (centers >= resp_win[0]) & (centers < resp_win[1])
    base = R[:, :, bi].mean(axis=(1, 2))  # spontaneous rate (Hz)
    per_freq = R[:, :, ri].mean(axis=2)  # (n_units, n_freqs) evoked rate
    evoked = per_freq - base[:, None]
    peak_evoked = np.abs(evoked).max(axis=1)
    bf_idx = np.argmax(evoked, axis=1)
    # latency: first lag bin in [0, 100) ms where the BF response crosses
    # baseline + 3 SD of the pre-stimulus fluctuation
    lat = np.full(R.shape[0], np.nan)
    for i in range(R.shape[0]):
        trace = R[i, bf_idx[i]]
        sd = trace[bi].std()
        thr = trace[bi].mean() + 3 * sd
        cand = np.flatnonzero((centers >= 0) & (centers < 0.1) & (trace > thr))
        if cand.size:
            lat[i] = centers[cand[0]]
    return pd.DataFrame(
        dict(
            baseline_hz=base,
            bf_hz=FREQS[bf_idx],
            bf_idx=bf_idx,
            peak_evoked_hz=peak_evoked,
            evoked_at_bf_hz=evoked[np.arange(len(bf_idx)), bf_idx],
            latency_s=lat,
        )
    ), evoked


def permutation_pvalues(units, trials, blocks, n_perm=500, resp_win=(0.0, 0.1),
                        base_win=(-0.1, 0.0), seed=0, progress=True):
    """Shuffle test for tone responsiveness.

    The null circularly shifts the tone onsets within each tone block, which keeps
    both the spike trains and the tone sequence intact but destroys the temporal
    locking between them.  Statistic: |evoked rate - baseline rate| pooled over all
    tones, where "evoked" is 0-100 ms after onset and "baseline" is the 100 ms before.
    """
    from tqdm.auto import tqdm

    rng = np.random.default_rng(seed)
    onsets = trials.start_time.values
    starts, ends = np.asarray(blocks.start), np.asarray(blocks.end)
    blk = np.searchsorted(ends, onsets)  # which block each tone belongs to
    blk = np.clip(blk, 0, len(starts) - 1)
    spans = ends - starts

    def shifted(shift):
        s0 = starts[blk]
        return np.sort(s0 + np.mod(onsets - s0 + shift, spans[blk]))

    def window_rate(st, ev, win):
        n = (np.searchsorted(st, ev + win[1]) - np.searchsorted(st, ev + win[0])).sum()
        return n / (ev.size * (win[1] - win[0]))

    def stat(st, ev):
        return abs(window_rate(st, ev, resp_win) - window_rate(st, ev, base_win))

    perm_events = [shifted(rng.uniform(1.0, spans.min() - 1.0)) for _ in range(n_perm)]
    keys = list(units.index)
    obs = np.zeros(len(keys))
    null = np.zeros((n_perm, len(keys)))
    for i, k in enumerate(tqdm(keys, desc="permutation test", disable=not progress)):
        st = units[k].restrict(blocks).t
        obs[i] = stat(st, onsets)
        for p, ev in enumerate(perm_events):
            null[p, i] = stat(st, ev)
    pvals = (1 + (null >= obs[None, :]).sum(axis=0)) / (n_perm + 1)
    return pvals, obs, null


def save_json(obj, name):
    with open(os.path.join(FIGDIR, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
