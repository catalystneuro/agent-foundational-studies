"""Shared loading helper for DANDI 000986 (Jaramillo auditory cortex, pure tones)."""
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
import pandas as pd
import numpy as np

LINDI_PREFIX = "https://lindi.neurosift.org/dandi/dandisets/000986/assets/{asset_id}/nwb.lindi.json"


def load_session(asset_id, local_cache=None):
    """Stream-load an NWB session via LINDI and return a nap.NWBFile."""
    url = LINDI_PREFIX.format(asset_id=asset_id)
    if local_cache is None:
        local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile


def build_trials(nwbfile):
    """IntervalSet of tone trials with stim_frequency metadata (Hz)."""
    tr = nwbfile.trials
    df = pd.DataFrame({
        "start": np.asarray(tr.start_time[:], dtype=float),
        "end": np.asarray(tr.stop_time[:], dtype=float),
        "stim_frequency": np.asarray(tr["stim_frequency"][:], dtype=float),
    })
    # keep metadata except start/end
    iv = nap.IntervalSet(
        start=df["start"], end=df["end"],
        metadata=df[["stim_frequency"]],
    )
    return iv
