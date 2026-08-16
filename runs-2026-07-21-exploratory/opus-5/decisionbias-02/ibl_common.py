"""Shared loading / preprocessing helpers for DANDI 000409 (IBL Brain Wide Map).

The task is the IBL two-alternative visual contrast discrimination task with
biased prior blocks: after an initial unbiased (0.5/0.5) block, the probability
that the Gabor appears on the left alternates in blocks between 0.2 and 0.8.
The block is never cued, so the animal has to infer it; the inferred prior shows
up as a lateral bias in upcoming choices.  That bias is the quantity we try to
decode from pre-stimulus spiking.
"""

import numpy as np
import pandas as pd
import h5py
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000409"
CACHE_DIR = "/tmp/remfile_cache_ibl"

# Pre-stimulus window (seconds relative to Gabor onset).  The IBL trial engine
# enforces a quiescence period of 0.4-0.7 s with no wheel movement immediately
# before stimulus onset, so this window is guaranteed movement-free.
PRE_WINDOW = (-0.4, 0.0)


def asset_url(asset_id):
    return (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
        f"/versions/draft/assets/{asset_id}/download/"
    )


def list_processed_assets():
    """All non-raw processed NWB assets in the dandiset, sorted by size."""
    assets, url = [], (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
        "/versions/draft/assets/?page_size=1000"
    )
    while url:
        r = requests.get(url).json()
        assets += r["results"]
        url = r["next"]
    nwb = [
        {"asset_id": a["asset_id"], "path": a["path"], "size": a["size"]}
        for a in assets
        if a["path"].endswith(".nwb") and "desc-processed" in a["path"]
    ]
    nwb.sort(key=lambda a: a["size"])
    return pd.DataFrame(nwb)


def open_nwb(asset_id, threads=8):
    """Stream an NWB file from the archive with on-disk byte caching.

    remfile defaults to a single download thread, which makes reading a
    150 MB spike-time array from S3 painfully slow; a handful of threads gets
    most of the available bandwidth.
    """
    f = remfile.File(
        asset_url(asset_id),
        disk_cache=remfile.DiskCache(CACHE_DIR),
        _max_threads=threads,
    )
    io = NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True)
    return io.read()


def all_spike_times(nwbfile):
    """Spike times for every unit, read as one sequential pass over the file.

    Indexing ``units['spike_times']`` unit by unit issues hundreds of small
    random reads, each of which pays the full round-trip latency to S3.  Reading
    the ragged array's backing dataset once and slicing it locally turns that
    into a single streaming read.
    """
    vi = nwbfile.units["spike_times"]
    ends = np.asarray(vi.data[:], dtype=np.int64)
    flat = np.asarray(vi.target.data[:])
    starts = np.concatenate([[0], ends[:-1]])
    return [flat[s:e] for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------

def get_trials(nwbfile):
    """Trial table with derived signed-contrast / choice-side / prior columns.

    ``mouse_wheel_choice`` is a wheel-turn direction.  We convert it to the
    *reported side* by using the fact that on rewarded trials the reported side
    equals the stimulus side, which fixes the mapping without assuming a
    convention.
    """
    tr = nwbfile.trials.to_dataframe().reset_index(drop=True)

    rewarded = tr["is_mouse_rewarded"].astype(bool)
    # Which wheel direction corresponds to "stimulus was on the left"?
    ref = tr.loc[rewarded & (tr["gabor_stimulus_side"] == "left"), "mouse_wheel_choice"]
    left_turn = ref.mode().iloc[0]
    consistency = (ref == left_turn).mean()
    assert consistency > 0.95, f"choice/side mapping is ambiguous ({consistency:.2f})"

    tr["choice_side"] = np.where(tr["mouse_wheel_choice"] == left_turn, "left", "right")
    tr["choice_right"] = (tr["choice_side"] == "right").astype(int)

    contrast = tr["gabor_stimulus_contrast"].to_numpy() / 100.0
    sign = np.where(tr["gabor_stimulus_side"] == "right", 1.0, -1.0)
    # 0 % contrast trials are nominally assigned a side but carry no evidence.
    tr["signed_contrast"] = np.where(contrast == 0, 0.0, sign * contrast)
    tr["abs_contrast"] = contrast

    tr["prior_right"] = 1.0 - tr["probability_left"]  # P(stimulus on right)
    tr["biased"] = tr["block_type"] != "unbiased"
    tr["block_right"] = np.where(tr["probability_left"] == 0.2, 1, 0)  # 0.2 left => right block

    # Behavioural history regressors (previous trial), used as a control.
    tr["prev_choice_right"] = tr["choice_right"].shift(1)
    tr["prev_rewarded"] = tr["is_mouse_rewarded"].astype(float).shift(1)
    tr["prev_stim_right"] = (tr["gabor_stimulus_side"] == "right").astype(float).shift(1)

    tr["stim_on"] = tr["gabor_stimulus_onset_time"]
    return tr


def valid_trials(tr, require_biased=True):
    """Trials with a usable stimulus-onset time, a choice, and (optionally) a block."""
    ok = tr["stim_on"].notna() & tr["choice_side"].notna() & tr["quiescence_period"].notna()
    if require_biased:
        ok &= tr["biased"]
    return tr[ok].copy()


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

def get_units(nwbfile, min_rate=1.0, quality="mua"):
    """Unit table with the anatomical location of the peak-amplitude channel.

    ``quality='good'`` keeps only Kilosort "good" units; ``'mua'`` keeps
    everything above the firing-rate floor (the IBL BWM decoding analyses use
    all units passing basic QC, which is what we do here).
    """
    # Read only the columns we need.  ``units.to_dataframe()`` would also pull
    # the mean waveforms and per-spike amplitudes, which are hundreds of MB.
    units = nwbfile.units
    u = pd.DataFrame(
        {
            "unit_name": np.asarray(units["unit_name"].data[:]),
            "firing_rate": np.asarray(units["firing_rate"].data[:], dtype=float),
            "kilosort2_label": [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in units["kilosort2_label"].data[:]
            ],
            "probe_name": [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in units["probe_name"].data[:]
            ],
        }
    )
    # ``max_electrode`` is a DynamicTableRegion; resolve it to raw row indices
    # rather than letting pynwb expand it into a DataFrame per unit.
    max_elec = np.asarray(units["max_electrode"].data[:], dtype=int)
    loc = nwbfile.electrodes["location"].data[:]
    loc = [l.decode() if isinstance(l, bytes) else str(l) for l in loc]
    u["location"] = [loc[e] if 0 <= e < len(loc) else "unknown" for e in max_elec]
    u["region"] = [acronym_of(l) for l in u["location"]]

    keep = u["firing_rate"] >= min_rate
    if quality == "good":
        keep &= u["kilosort2_label"] == "good"
    return u[keep]


# Coarse Allen-CCF grouping.  The NWB files store full region names, so we map
# them onto a handful of coarse areas that are large enough to decode from.
_COARSE = [
    ("visual_ctx", ("visual area", "visual cortex")),
    ("motor_ctx", ("motor area",)),
    ("frontal_ctx", ("prelimbic", "infralimbic", "orbital area", "anterior cingulate")),
    ("somatosens_ctx", ("somatosensory area",)),
    ("hippocampus", ("field ca", "dentate gyrus", "subiculum", "ammon")),
    ("thalamus", ("thalamus", "nucleus of the thalamus", "geniculate", "lateral posterior",
                  "ventral anterior-lateral", "ventral medial nucleus", "mediodorsal",
                  "posterior complex", "reticular nucleus of the thalamus")),
    ("striatum", ("caudoputamen", "nucleus accumbens", "striatum", "olfactory tubercle")),
    ("midbrain", ("superior colliculus", "midbrain", "substantia nigra", "periaqueductal",
                  "ventral tegmental", "inferior colliculus", "pretectal", "red nucleus")),
    ("hypothalamus", ("hypothalam", "zona incerta")),
    ("amygdala", ("amygdal",)),
    ("olfactory", ("piriform", "olfactory areas", "taenia tecta", "endopiriform")),
    ("hindbrain", ("pons", "medulla", "cerebell", "tegmental nucleus")),
    ("cortex_other", ("area", "cortex", "layer", "retrosplenial", "temporal association",
                      "perirhinal", "ectorhinal", "entorhinal")),
]


def acronym_of(location):
    l = str(location).lower()
    if l in ("unknown", "root", "void", "nan"):
        return "unknown"
    for name, keys in _COARSE:
        if any(k in l for k in keys):
            return name
    return "other"


# --------------------------------------------------------------------------
# Spike counts
# --------------------------------------------------------------------------

def window_counts(spike_times_list, event_times, window):
    """(n_trials, n_units) spike counts in ``window`` around ``event_times``.

    ``spike_times_list`` must contain sorted arrays.  Uses searchsorted, which
    is O(log n) per edge and fast enough for thousands of units.
    """
    ev = np.asarray(event_times, dtype=float)
    starts, stops = ev + window[0], ev + window[1]
    counts = np.empty((len(ev), len(spike_times_list)), dtype=np.float32)
    for j, st in enumerate(spike_times_list):
        counts[:, j] = np.searchsorted(st, stops) - np.searchsorted(st, starts)
    return counts
