"""Streaming access helpers for the two DANDI dandisets used in this analysis.

DANDI:000986 - Neuropixels recordings from mouse auditory cortex during passive
               exposure to pure tones (5 frequencies, 60 dB SPL, 25 ms).
DANDI:001419 - Linear-probe recordings from mouse auditory cortex with a
               half-octave pure-tone series (10 frequencies, 70 dB SPL, 300 ms)
               and laminar electrode positions.

Files are read over HTTP with remfile + a local disk cache; nothing is
downloaded in full.
"""

import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache")

DANDISET_TONES = "000986"
DANDISET_TONES_VERSION = "0.251031.1939"
DANDISET_LAMINAR = "001419"
DANDISET_LAMINAR_VERSION = "0.250521.2343"

_API = "https://api.dandiarchive.org/api"


def list_assets(dandiset_id, version):
    """Return [(path, asset_id), ...] for every asset in a published dandiset version."""
    out = []
    url = f"{_API}/dandisets/{dandiset_id}/versions/{version}/assets/"
    params = {"page_size": 200}
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        out.extend((a["path"], a["asset_id"]) for a in d["results"])
        url, params = d.get("next"), None
    return sorted(out)


def asset_s3_url(asset_id):
    """Resolve a DANDI asset id to its direct S3 URL."""
    r = requests.get(f"{_API}/assets/{asset_id}/info/", timeout=60)
    r.raise_for_status()
    urls = r.json()["metadata"]["contentUrl"]
    return next(u for u in urls if "s3" in u)


def open_nwb(asset_id):
    """Stream an NWB asset and return (nwbfile, pynapple NWBFile, io handle)."""
    url = asset_s3_url(asset_id)
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    io = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile), io


def spikes_as_tsgroup(nwbfile, metadata=None):
    """Build a pynapple TsGroup from the NWB units table."""
    units = nwbfile.units.to_dataframe()
    spikes = {i: np.asarray(st) for i, st in enumerate(units["spike_times"].values)}
    return nap.TsGroup(spikes, metadata=metadata)


# ---------------------------------------------------------------------------
# DANDI:000986 - Neuropixels, 5 tone frequencies, thousands of repeats
# ---------------------------------------------------------------------------


def load_tone_session(asset_id):
    """Load one DANDI:000986 session.

    Returns a dict with the spike TsGroup, the trial table, an IntervalSet of
    the tone-presentation block, and the pupil / running behavioural traces.
    """
    nwbfile, _, io = open_nwb(asset_id)
    trials = nwbfile.trials.to_dataframe()

    spikes = spikes_as_tsgroup(nwbfile)
    tone_onsets = nap.Ts(t=trials["start_time"].values)
    frequency = trials["stim_frequency"].values

    behavior = nwbfile.processing["behavior"]
    pupil_ts = behavior["PupilTracking"]["pupil_diameter"]
    running_ts = behavior["running_speed"]

    return dict(
        nwbfile=nwbfile,
        io=io,
        subject=nwbfile.subject.subject_id,
        session=nwbfile.identifier,
        spikes=spikes,
        trials=trials,
        tone_onsets=tone_onsets,
        frequency=frequency,
        frequencies=np.unique(frequency),
        pupil=pupil_ts,
        running=running_ts,
        spontaneous=nap.IntervalSet(
            start=nwbfile.intervals["spontaneous_blocks"].to_dataframe()["start_time"].values,
            end=nwbfile.intervals["spontaneous_blocks"].to_dataframe()["stop_time"].values,
        ),
    )


# ---------------------------------------------------------------------------
# DANDI:001419 - laminar probe, 10 half-octave frequencies
# ---------------------------------------------------------------------------


def load_laminar_session(asset_id):
    """Load one DANDI:001419 'tones' session (10 frequencies, laminar probe).

    Only trials without optogenetic LED stimulation are returned, so the tuning
    measured here reflects the acoustic response alone.
    """
    nwbfile, _, io = open_nwb(asset_id)
    trials = nwbfile.trials.to_dataframe()
    units = nwbfile.units.to_dataframe()

    # depth of the electrode each unit was assigned to (y in microns from tip)
    depth, quality = [], []
    for _, row in units.iterrows():
        depth.append(float(np.asarray(row["electrodes"]["y"])[0]))
        quality.append(row["quality"])
    depth = np.asarray(depth)

    # laminar label of the assigned electrode
    layer = []
    for _, row in units.iterrows():
        layer.append(str(np.asarray(row["electrodes"]["location"])[0]).strip())

    spikes = spikes_as_tsgroup(
        nwbfile, metadata={"depth": depth, "quality": np.asarray(quality), "layer": np.asarray(layer)}
    )

    led_on = np.isfinite(trials["led_on_time"].values)
    acoustic = trials.loc[~led_on]

    return dict(
        nwbfile=nwbfile,
        io=io,
        subject=nwbfile.subject.subject_id,
        session=nwbfile.identifier,
        spikes=spikes,
        trials=acoustic,
        tone_onsets=nap.Ts(t=acoustic["start_time"].values),
        frequency=acoustic["frequency"].values,
        frequencies=np.unique(trials["frequency"].values),
        n_led_trials=int(led_on.sum()),
    )
