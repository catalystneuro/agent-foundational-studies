"""Shared utilities for the sharp-wave ripple / replay analysis on DANDI:000044.

Dandiset 000044 -- Grosmark & Buzsaki (2016), "Diversity in neural firing dynamics
supports both rigid and learned hippocampal sequences". Bilateral silicon-probe
recordings from rat dorsal CA1 with a PRE sleep epoch, a novel-maze running epoch,
and a POST sleep epoch.
"""

import os

import h5py
import numpy as np
import pynapple as nap
import remfile
import scipy.signal as sig
from pynwb import NWBHDF5IO
from scipy.fft import next_fast_len

DANDISET = "000044"
ASSETS = {
    # session name -> DANDI asset id
    "Achilles_10252013": "c0ac352b-9da5-44b0-b73d-41a9ee3c3b1d",
    "Achilles_11012013": "080a847a-5211-4101-bc05-fd45b7e77dbd",
    "Cicero_09012014": "97252767-5e90-45cf-a1b8-8cbff2f2a3a3",
    "Cicero_09102014": "d6b8092b-94e2-4e37-8660-86f9e856aec8",
    "Cicero_09172014": "d9a1e010-af4a-4fba-812d-66e2d1a71e35",
    "Gatsby_08022013": "93569d6c-781f-4422-938e-e935a62863de",
    "Gatsby_08282013": "402f78e1-9e7d-486c-8822-93d538ed6ccc",
    "Buddy_06272013": "185b8a36-d671-4688-ba05-9e89a902c486",
}

CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000044")
LFP_FS = 1250.0  # Hz, stated in the ElectricalSeries starting_time.rate attribute


def asset_url(session):
    return (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}"
        f"/versions/draft/assets/{ASSETS[session]}/download/"
    )


def open_session(session):
    """Open one NWB file by streaming, returning (nwbfile, h5py handle)."""
    rem = remfile.File(asset_url(session), disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, mode="r", load_namespaces=False)
    nwbfile = io.read()
    return nwbfile, h5


def decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])


def load_core(session):
    """Load spikes, epochs, brain states and maze position with pynapple.

    Returns a dict; the LFP is left out because it needs channel selection first.
    """
    nwbfile, h5 = open_session(session)
    nwb = nap.NWBFile(nwbfile)

    units = nwb["units"]
    utab = nwbfile.units.to_dataframe()
    units.set_info(
        cell_type=utab["cell_type"].values,
        shank_id=utab["shank_id"].values,
        location=utab["location"].values,
    )

    ep = nwbfile.epochs.to_dataframe()
    epochs = {
        str(r.label): nap.IntervalSet(start=r.start_time, end=r.stop_time)
        for r in ep.itertuples()
    }

    st = nwbfile.processing["behavior"]["states"].to_dataframe()
    labels = decode(st["label"].values)
    states = {
        lab: nap.IntervalSet(
            start=st["start_time"].values[labels == lab].astype(float),
            end=st["stop_time"].values[labels == lab].astype(float),
        )
        for lab in np.unique(labels)
    }

    beh = nwbfile.processing["behavior"]
    lin_key = [k for k in beh.data_interfaces if "Linearized" in k][0]
    lin_ss = list(beh[lin_key].spatial_series.values())[0]
    t0 = lin_ss.starting_time
    rate = lin_ss.rate
    lin = np.asarray(lin_ss.data[:]).squeeze()
    t = t0 + np.arange(lin.size) / rate
    position = nap.Tsd(t=t, d=lin.astype(float))

    return dict(
        session=session,
        nwbfile=nwbfile,
        h5=h5,
        units=units,
        epochs=epochs,
        states=states,
        position=position,
        maze_name=lin_key,
    )


def lfp_dataset(h5):
    return h5["processing/ecephys/LFP/LFP/data"]


def lfp_conversion(h5):
    return float(h5["processing/ecephys/LFP/LFP/data"].attrs["conversion"])



def band_envelope(x, low, high, fs=LFP_FS, order=4, chunk_s=600.0, pad_s=2.0):
    """Band-pass filter and Hilbert-envelope a long 1-D signal in overlapping chunks.

    Returns (filtered, envelope) as float32 arrays the same length as ``x``.
    """
    b, a = sig.butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    n = x.size
    step = int(chunk_s * fs)
    pad = int(pad_s * fs)
    filt = np.empty(n, dtype=np.float32)
    env = np.empty(n, dtype=np.float32)
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        j0, j1 = max(0, i0 - pad), min(n, i1 + pad)
        seg = sig.filtfilt(b, a, x[j0:j1].astype(np.float64))
        e = np.abs(sig.hilbert(seg, N=next_fast_len(seg.size)))[: seg.size]
        filt[i0:i1] = seg[i0 - j0 : i1 - j0]
        env[i0:i1] = e[i0 - j0 : i1 - j0]
    return filt, env


def gaussian_smooth(x, sigma_s, fs=LFP_FS):
    n = int(np.ceil(4 * sigma_s * fs))
    w = sig.windows.gaussian(2 * n + 1, sigma_s * fs)
    w /= w.sum()
    return np.convolve(x, w, mode="same")


