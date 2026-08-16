"""Shared loading + analysis helpers for reach direction / velocity tuning."""
import numpy as np
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap

ASSETS = {
    "MC_Maze": "https://api.dandiarchive.org/api/assets/26e85f09-39b7-480f-b337-278a8f034007/download/",
    "MC_RTT":  "https://api.dandiarchive.org/api/assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/",
}
CACHE = "/tmp/remfile_cache"


def open_nwb(key):
    f = remfile.File(ASSETS[key], disk_cache=remfile.DiskCache(CACHE))
    io = NWBHDF5IO(file=h5py.File(f, "r"), load_namespaces=True)
    return io.read()


def obs_intervals_set(nwbfile):
    """IntervalSet of epochs where spiking was observed (identical across units here)."""
    oi = nwbfile.units["obs_intervals"][0]
    return nap.IntervalSet(start=oi[:, 0], end=oi[:, 1])


def load_units(nwbfile, ep):
    u = nwbfile.units
    spikes = {i: np.asarray(u["spike_times"][i]) for i in range(len(u.id))}
    # NOTE: in this NWB file every unit's electrode reference resolves into the
    # PMd block of the electrode table (87 distinct rows for 182 units), which is
    # not consistent with the documented dual-array M1+PMd recording. We therefore
    # do not split units by area and analyse them as one motor-cortical population.
    md = {"heldout": np.asarray(u["heldout"][:]).astype(bool)}
    return nap.TsGroup(spikes, time_support=ep, **md)
