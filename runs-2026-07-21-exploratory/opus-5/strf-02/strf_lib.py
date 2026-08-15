"""Shared helpers for the DANDI spectrotemporal-receptive-field analysis.

Two dandisets are used:

* 000986 - mouse auditory cortex, Neuropixels, passive pure tones (2-32 kHz,
  25 ms, 60 dB SPL, ~0.8 s inter-onset interval).
* 001262 - Mongolian gerbil auditory nerve fibres, single-unit glass-electrode
  recordings with a fine best-frequency tone sweep per fibre.

Everything is streamed from the DANDI S3 bucket with remfile + a local disk
cache, so no whole-file downloads happen.
"""

import json
import os
import re
import urllib.parse
import urllib.request

import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

CACHE_DIR = os.environ.get("STRF_REMFILE_CACHE", "/tmp/remfile_cache")
_DISK_CACHE = remfile.DiskCache(CACHE_DIR)

DANDI_API = "https://api.dandiarchive.org/api"


# --------------------------------------------------------------------------
# DANDI asset resolution
# --------------------------------------------------------------------------
def _blob_url(blob_id):
    return f"https://dandiarchive.s3.amazonaws.com/blobs/{blob_id[:3]}/{blob_id[3:6]}/{blob_id}"


def list_assets(dandiset_id, version="draft"):
    """Return [(path, s3_url, size_bytes), ...] for every asset in a dandiset."""
    url = f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}/assets/?page_size=1000"
    out = []
    while url:
        payload = json.load(urllib.request.urlopen(url))
        for a in payload["results"]:
            out.append((a["path"], _blob_url(a["blob"]), a["size"]))
        url = payload.get("next")
    return sorted(out)


def asset_url(dandiset_id, path, version="draft"):
    url = (
        f"{DANDI_API}/dandisets/{dandiset_id}/versions/{version}/assets/"
        f"?path={urllib.parse.quote(path)}"
    )
    res = json.load(urllib.request.urlopen(url))["results"]
    match = [r for r in res if r["path"] == path]
    return _blob_url(match[0]["blob"])


def open_h5(s3_url):
    """Stream an NWB file from S3 and return an open h5py.File."""
    return h5py.File(remfile.File(s3_url, disk_cache=_DISK_CACHE), "r")


def open_nwb(s3_url):
    """Stream an NWB file and return (pynapple NWBFile, pynwb NWBFile, io)."""
    h5 = open_h5(s3_url)
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


# --------------------------------------------------------------------------
# Dandiset 000986 - mouse auditory cortex
# --------------------------------------------------------------------------
TONE_FREQS_HZ = np.array([2000.0, 4000.0, 8000.0, 16000.0, 32000.0])


def load_ac_session(s3_url):
    """Load one 000986 session.

    Returns a dict with a pynapple TsGroup of units, a pynapple IntervalSet of
    tone trials plus the per-trial stimulus parameters, and the pupil trace.
    """
    nwb, nwbfile, io = open_nwb(s3_url)

    units = nwb["units"]  # pynapple TsGroup
    trials = nwbfile.intervals["trials"].to_dataframe()
    spont = nwbfile.intervals["spontaneous_blocks"].to_dataframe()

    pupil = nwbfile.processing["behavior"]["PupilTracking"]["pupil_diameter"]
    pupil_ts = nap.Tsd(t=pupil.timestamps[:], d=pupil.data[:])

    return dict(
        units=units,
        trials=trials,
        spont=spont,
        pupil=pupil_ts,
        session_id=nwbfile.session_id,
        subject_id=nwbfile.subject.subject_id,
        io=io,
    )


# --------------------------------------------------------------------------
# Stimulus design matrix and STRF estimation
# --------------------------------------------------------------------------
def onset_bin(onsets, t_bins):
    """Index of the bin that *contains* each tone onset.

    ``searchsorted`` on its own would round the onset up to the next bin edge,
    which shortens every measured latency by up to one bin.
    """
    return np.searchsorted(t_bins, onsets, side="right") - 1


def build_tone_stimulus(onsets, freq_index, n_freq, t_bins, tone_dur):
    """Binary stimulus matrix S[time, frequency] for a pure-tone-pip ensemble.

    Each tone contributes 1.0 to its own frequency channel for the duration of
    the pip.  This is the (log-)spectrogram of the stimulus discretised on the
    same clock as the spike counts, and is what the STRF is defined against.
    """
    dt = t_bins[1] - t_bins[0]
    S = np.zeros((len(t_bins), n_freq), dtype=np.float32)
    n_on = max(1, int(round(tone_dur / dt)))
    idx = onset_bin(onsets, t_bins)
    for i, k in zip(idx, freq_index):
        if 0 <= i < len(t_bins):
            S[i : i + n_on, k] = 1.0
    return S


def lagged_design(S, n_lags):
    """Time-embed a stimulus matrix: X[t, lag, freq] flattened to [t, lag*freq]."""
    n_t, n_f = S.shape
    X = np.zeros((n_t, n_lags, n_f), dtype=np.float32)
    for L in range(n_lags):
        X[L:, L, :] = S[: n_t - L, :]
    return X.reshape(n_t, n_lags * n_f)


def sta_strf(counts, S, n_lags):
    """Spike-triggered-average STRF, in units of spikes/s above baseline.

    For a sparse tone ensemble the STA reduces to the frequency-conditioned
    PSTH minus the mean rate, which is exactly the tone-evoked receptive field.
    """
    n_t, n_f = S.shape
    strf = np.zeros((n_lags, n_f))
    mean_rate = counts.mean()
    for k in range(n_f):
        on = np.flatnonzero(np.diff(np.r_[0, S[:, k]]) > 0)
        on = on[on < n_t - n_lags]
        if len(on) == 0:
            continue
        seg = np.stack([counts[i : i + n_lags] for i in on])
        strf[:, k] = seg.mean(axis=0) - mean_rate
    return strf


def bin_spikes(spike_times, t_bins):
    dt = t_bins[1] - t_bins[0]
    edges = np.r_[t_bins, t_bins[-1] + dt]
    return np.histogram(spike_times, bins=edges)[0].astype(np.float64)


# --------------------------------------------------------------------------
# Dandiset 001262 - gerbil auditory nerve
# --------------------------------------------------------------------------
_BF_TAG = re.compile(r"^BF_FREQ(\d+)_rep(\d+)$")


def load_anf_fibre(s3_url):
    """Load one 001262 auditory-nerve-fibre file.

    Each 'unit' row in this NWB file is one stimulus presentation; the ``tag``
    column names the stimulus and ``spike_times`` are relative to trial onset.
    Returns the per-frequency spike times of the best-frequency tone sweep,
    together with the fibre metadata published with the dataset.
    """
    f = open_h5(s3_url)

    tags = [t.decode() for t in f["units/tag"][:]]
    st = f["units/spike_times"][:]
    idx = f["units/spike_times_index"][:]
    starts = np.r_[0, idx[:-1]]

    trials = {}
    for i, tag in enumerate(tags):
        m = _BF_TAG.match(tag)
        if m:
            trials.setdefault(float(m.group(1)), []).append(st[starts[i] : idx[i]])

    # Not every fibre in the archive was held long enough to run the
    # best-frequency sweep; those files have no BF_FREQ acquisitions.
    bf_traces = [k for k in f["acquisition"] if k.startswith("BF_FREQ")]
    if not bf_traces or not trials:
        f.close()
        return None

    # trace duration and sampling rate of the recorded sweeps
    ds = f["acquisition"][bf_traces[0]]
    rate = ds["starting_time"].attrs["rate"]
    dur = ds["data"].shape[0] / rate

    at = f["analysis/analysis_table"]
    exps = [e.decode() for e in at["experiment"][:]]
    meta = {}
    for col in ("results_bf", "results_sr", "results_threshold"):
        vals = at[col][:]
        for e, v in zip(exps, vals):
            if np.isfinite(v):
                meta[f"{col[8:]}_{e}"] = float(v)

    notes = f["general/notes"][()] if "notes" in f["general"] else b""
    return dict(
        freqs=np.array(sorted(trials)),
        spikes={k: v for k, v in trials.items()},
        trial_dur=dur,
        bf_published=meta.get("bf_BF", np.nan),
        sr_published=meta.get("sr_BF", meta.get("sr_RLF", np.nan)),
        threshold_published=meta.get("threshold_RLF", np.nan),
        notes=notes.decode(errors="replace") if isinstance(notes, bytes) else str(notes),
        h5=f,
    )


def anf_strf(fibre, bin_ms=2.0, t_max=None):
    """Frequency x time response field for one auditory-nerve fibre (spikes/s)."""
    dt = bin_ms / 1000.0
    t_max = fibre["trial_dur"] if t_max is None else t_max
    edges = np.arange(0, t_max + dt, dt)
    freqs = fibre["freqs"]
    R = np.zeros((len(edges) - 1, len(freqs)))
    for j, fq in enumerate(freqs):
        reps = fibre["spikes"][fq]
        h = np.zeros(len(edges) - 1)
        for r in reps:
            h += np.histogram(r, bins=edges)[0]
        R[:, j] = h / (len(reps) * dt)
    return R, edges[:-1], freqs
