"""Streaming access helpers for the two DANDI dandisets used in this analysis.

DANDI:001419 - mouse auditory cortex (A1/A2) linear-probe recordings during pure-tone
               presentation (10 half-octave frequencies, 4-80 kHz, 70 dB SPL).
DANDI:000986 - mouse auditory cortex Neuropixels recordings during passive pure-tone
               exposure (5 octave-spaced frequencies, 2-32 kHz, 60 dB SPL).

Files are streamed from the DANDI S3 bucket with remfile and a local disk cache; nothing
is downloaded in full.
"""

import json
import os
import re

import h5py
import numpy as np
import remfile
import requests

API = "https://api.dandiarchive.org/api"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")

DANDISETS = {
    "001419": "0.250521.2343",  # auditory cortex linear probe, tones + clicks
    "000986": "0.251031.1939",  # auditory cortex Neuropixels, pure tones + pupil
}


def list_assets(dandiset_id, version=None, contains=None):
    """Return [(path, download_url), ...] for a dandiset, optionally filtered by substring."""
    version = version or DANDISETS[dandiset_id]
    assets, url = [], f"{API}/dandisets/{dandiset_id}/versions/{version}/assets/?page_size=100"
    while url:
        page = requests.get(url).json()
        assets += page["results"]
        url = page["next"]
    if contains is not None:
        assets = [a for a in assets if contains in a["path"]]
    assets.sort(key=lambda a: a["path"])
    return [
        (
            a["path"],
            f"{API}/dandisets/{dandiset_id}/versions/{version}/assets/{a['asset_id']}/download/",
        )
        for a in assets
    ]


def open_stream(url):
    """Open a remote NWB file as an h5py.File via remfile with disk caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return h5py.File(remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR)), "r")


def _ragged(group, name):
    """Read an NWB ragged (VectorIndex-backed) column into a list of arrays."""
    data = group[name][:]
    idx = group[f"{name}_index"][:].astype(np.int64)
    starts = np.concatenate([[0], idx[:-1]]).astype(np.int64)
    return [data[s:e] for s, e in zip(starts, idx)]


def load_tone_session_001419(url):
    """Load one DANDI:001419 tone session.

    Returns a dict with spike times per unit, unit metadata (quality, cortical layer,
    electrode depth) and the tone trial table.
    """
    f = open_stream(url)
    units = f["units"]
    spikes = _ragged(units, "spike_times")
    unit_ids = units["id"][:]
    elec_idx = units["electrodes"][:]

    etab = f["general/extracellular_ephys/electrodes"]
    loc = np.array([s.decode() if isinstance(s, bytes) else s for s in etab["location"][:]])
    depth = etab["y"][:]  # distance from the top of the probe, microns

    trials = f["intervals/trials"]
    out = dict(
        session=f["general/session_id"][()].decode(),
        subject=f["general/subject/subject_id"][()].decode(),
        spikes={int(u): np.asarray(s, dtype=float) for u, s in zip(unit_ids, spikes)},
        quality=np.array([q.decode() for q in units["quality"][:]]),
        unit_ids=unit_ids,
        layer=loc[elec_idx],
        depth=depth[elec_idx],
        tone_onset=trials["start_time"][:],
        tone_offset=trials["stop_time"][:],
        frequency=trials["frequency"][:],
        intensity=trials["intensity"][:],
    )
    out["file"] = f
    return out


def load_tone_session_000986(url):
    """Load one DANDI:000986 session (spike times + tone trial table only)."""
    f = open_stream(url)
    units = f["units"]
    spikes = _ragged(units, "spike_times")
    trials = f["intervals/trials"]
    return dict(
        session=f["general/session_id"][()].decode(),
        subject=f["general/subject/subject_id"][()].decode(),
        spikes={int(u): np.asarray(s, dtype=float) for u, s in zip(units["id"][:], spikes)},
        tone_onset=trials["start_time"][:],
        tone_offset=trials["stop_time"][:],
        frequency=trials["stim_frequency"][:],
        intensity=trials["stim_amplitude"][:],
        file=f,
    )


def read_behavior_window(f, t_start, t_stop):
    """Read pupil diameter and running speed over [t_start, t_stop] from a 000986 file.

    The traces have ~7.5 million samples each, so we locate the window by assuming a
    near-constant sampling rate, then read only that index range and check the timestamps
    we actually got.
    """
    out = {}
    for name, path in [
        ("pupil", "processing/behavior/PupilTracking/pupil_diameter"),
        ("running", "processing/behavior/running_speed"),
    ]:
        ts, data = f[f"{path}/timestamps"], f[f"{path}/data"]
        n = ts.shape[0]
        t0, t1 = float(ts[0]), float(ts[-1])
        rate = (n - 1) / (t1 - t0)
        i0 = int(np.clip((t_start - t0) * rate - rate, 0, n - 2))
        i1 = int(np.clip((t_stop - t0) * rate + rate, i0 + 2, n))
        t, d = ts[i0:i1], data[i0:i1]
        m = (t >= t_start) & (t <= t_stop)
        assert m.sum() > 0, f"{name}: index estimate missed the requested window"
        out[name] = (t[m], d[m])
    return out


GRID_RE = re.compile(r"^CF_FREQ([\d.]+)_ABI([-\d.]+)_rep(\d+)$")   # frequency x level grid
ISO_RE = re.compile(r"^BF_FREQ([\d.]+)_rep(\d+)$")                 # frequency sweep, one level


def load_fibre_001262(url):
    """Load one DANDI:001262 auditory-nerve-fibre file.

    Each NWB file holds a single fibre. The `units` table has one row per stimulus
    sweep, tagged with the stimulus name; `CF_FREQ<f>_ABI<level>_rep<n>` sweeps form the
    frequency x level grid used to measure the fibre's frequency response area.
    Sweeps are 200 ms long and spike times are relative to sweep onset.

    Returns a dict with per-sweep spike times plus the frequency/level of each sweep.
    """
    f = open_stream(url)
    units = f["units"]
    tags = np.array([t.decode() if isinstance(t, bytes) else t for t in units["tag"][:]])
    sweeps = _ragged(units, "spike_times")

    freq, level, keep = [], [], []
    iso_freq, iso_keep = [], []
    for i, t in enumerate(tags):
        m = GRID_RE.match(t)
        if m:
            freq.append(float(m.group(1)))
            level.append(float(m.group(2)))
            keep.append(i)
            continue
        m = ISO_RE.match(t)
        if m:
            iso_freq.append(float(m.group(1)))
            iso_keep.append(i)

    # a few sweeps carry NaN padding in place of spike times
    def clean(a):
        a = np.asarray(a, dtype=float)
        return np.sort(a[np.isfinite(a)])

    silent = [clean(sweeps[i]) for i, t in enumerate(tags) if "_silent_" in t]

    subj = f["general/subject"]
    return dict(
        silent_sweeps=silent,
        fibre=f["general/session_id"][()].decode(),
        subject=subj["subject_id"][()].decode(),
        age_days=int(subj["age"][()].decode().strip("PD")),
        sweeps=[clean(sweeps[i]) for i in keep],
        frequency=np.array(freq),
        level=np.array(level),
        iso_sweeps=[clean(sweeps[i]) for i in iso_keep],
        iso_frequency=np.array(iso_freq),
        sweep_duration=9765 / 48828.125,
        file=f,
    )


if __name__ == "__main__":
    tone_files = list_assets("001419", contains="tones")
    print(f"DANDI:001419 tone sessions: {len(tone_files)}")
    s = load_tone_session_001419(tone_files[0][1])
    print(s["session"], s["subject"], len(s["spikes"]), "units")
    print("frequencies (kHz):", np.unique(s["frequency"]) / 1e3)
    print("layers:", sorted(set(s["layer"])))
    json.dump([p for p, _ in tone_files], open("tone_sessions_001419.json", "w"), indent=1)
