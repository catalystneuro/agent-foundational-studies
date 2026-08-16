"""Shared loading and analysis helpers for the head-direction cell demonstration.

Dataset: DANDI:000939, "Large-scale recordings of head direction cells in mouse
postsubiculum" (Duszkiewicz et al.), streamed from S3 with remfile disk caching.
"""
import os

import h5py
import numpy as np
import pandas as pd
import pynapple as nap
import remfile
import requests
from pynwb import NWBHDF5IO

DANDISET = "000939"
CACHE_DIR = os.environ.get("REMFILE_CACHE", "/tmp/remfile_cache_000939")
API = "https://api.dandiarchive.org/api/dandisets"


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def list_assets(dandiset=DANDISET):
    """Return a DataFrame of (path, asset_id, size) for every NWB asset."""
    r = requests.get(
        f"{API}/{dandiset}/versions/draft/assets/", params={"page_size": 200}
    )
    r.raise_for_status()
    rows = [
        {"path": a["path"], "asset_id": a["asset_id"], "size_gb": a["size"] / 1e9}
        for a in r.json()["results"]
        if a["path"].endswith(".nwb")
    ]
    return pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def asset_s3_url(asset_id, dandiset=DANDISET):
    r = requests.get(f"{API}/{dandiset}/versions/draft/assets/{asset_id}/")
    r.raise_for_status()
    return [u for u in r.json()["contentUrl"] if "s3.amazonaws" in u][0]


def load_session(asset_id, dandiset=DANDISET):
    """Stream one NWB session and return (nap.NWBFile, pynwb NWBFile, io handle).

    Only the byte ranges actually touched are fetched, so the multi-GB raw
    ElectricalSeries in these files is never downloaded.
    """
    url = asset_s3_url(asset_id, dandiset)
    rf = remfile.File(url, disk_cache=remfile.DiskCache(CACHE_DIR))
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), nwbfile, io


def session_bundle(asset_id, dandiset=DANDISET):
    """Load a session and package the streams this analysis needs.

    Returns a dict with:
      name       session identifier
      units      TsGroup of spike trains, with per-unit metadata
      hd         Tsd of head direction in radians, wrapped to [0, 2*pi)
      position   TsdFrame of x/y position (cm)
      epochs     IntervalSet of behavioural epochs, with an 'epoch' label column
      wake       IntervalSet covering the open-field foraging epochs
      sleep_states  IntervalSet with a 'state' column (wake / nrem / rem)
    """
    nwb, nwbfile, io = load_session(asset_id, dandiset)

    units = nwb["units"]
    epochs = nwb["epochs"]

    # Head direction: wrap to [0, 2*pi) and drop tracking dropouts (NaNs).
    hd_raw = nwb["head-direction"]
    good = ~np.isnan(hd_raw.d)
    hd = nap.Tsd(t=hd_raw.t[good], d=np.mod(hd_raw.d[good], 2 * np.pi))
    n_dropped = int((~good).sum())

    pos_raw = nwb["position"]
    pgood = ~np.any(np.isnan(pos_raw.values), axis=1)
    position = nap.TsdFrame(
        t=pos_raw.t[pgood], d=pos_raw.values[pgood], columns=list(pos_raw.columns)
    )

    # Electrode location per unit, via the unit's electrode_group.
    elec = nwbfile.electrodes.to_dataframe()
    udf = nwbfile.units.to_dataframe()
    grp2loc = (
        elec.groupby("group_name")["location"].agg(lambda s: s.mode().iloc[0]).to_dict()
    )
    locations = [grp2loc.get(g.name, "unknown") for g in udf["electrode_group"]]
    units.set_info(location=np.array(locations))

    # Epoch names live in the NWB 'tags' column (home_cage, wake_square, ...),
    # stored as a one-element array per interval.
    ep_labels = np.asarray([np.asarray(t).ravel()[0] for t in epochs.tags])
    wake_idx = [i for i, l in enumerate(ep_labels) if l.startswith("wake")]
    wake = nap.IntervalSet(start=epochs.start[wake_idx], end=epochs.end[wake_idx])

    return {
        "name": nwbfile.identifier,
        "subject": nwbfile.subject.subject_id,
        "units": units,
        "hd": hd,
        "position": position,
        "epochs": epochs,
        "epoch_labels": ep_labels,
        "n_hd_nan": n_dropped,
        "wake": wake,
        "sleep_states": nwb["sleep_states"] if "sleep_states" in nwb.keys() else None,
        "nwbfile": nwbfile,
        "io": io,
    }


# --------------------------------------------------------------------------- #
# Circular statistics
# --------------------------------------------------------------------------- #
def mean_vector(tuning_curve):
    """Resultant vector (length, angle) of a tuning curve indexed by angle."""
    ang = np.asarray(tuning_curve.index)
    rate = np.asarray(tuning_curve.values, dtype=float)
    rate = np.clip(rate, 0, None)
    if rate.sum() == 0:
        return 0.0, np.nan
    z = (rate * np.exp(1j * ang)).sum() / rate.sum()
    return float(np.abs(z)), float(np.mod(np.angle(z), 2 * np.pi))


def hd_information(tuning_curve, occupancy):
    """Directional information in bits/spike (Skaggs et al.).

    occupancy: probability of each angular bin (sums to 1).
    """
    rate = np.clip(np.asarray(tuning_curve.values, dtype=float), 0, None)
    p = np.asarray(occupancy, dtype=float)
    p = p / p.sum()
    mean_rate = np.sum(p * rate)
    if mean_rate <= 0:
        return 0.0
    nz = rate > 0
    return float(np.sum(p[nz] * (rate[nz] / mean_rate) * np.log2(rate[nz] / mean_rate)))


def angular_occupancy(hd, ep, nb_bins=120):
    """Probability of each head-direction bin, matching pynapple's binning."""
    hd_ep = hd.restrict(ep)
    bins = np.linspace(0, 2 * np.pi, nb_bins + 1)
    counts, _ = np.histogram(hd_ep.values, bins)
    return counts / counts.sum()


def circ_diff(a, b):
    """Signed circular difference a - b wrapped to (-pi, pi]."""
    return np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Tuning curves and shuffling
# --------------------------------------------------------------------------- #
def compute_tuning(units, hd, ep, nb_bins=120):
    """Head-direction tuning curves (Hz) for a TsGroup over an IntervalSet."""
    return nap.compute_tuning_curves(
        data=units, features=hd, bins=nb_bins, range=(0, 2 * np.pi),
        epochs=ep, return_pandas=True,
    )


def compute_tuning_xr(units, hd, ep, nb_bins=60):
    """Same tuning curves as an xarray.DataArray, which nap.decode_bayes wants."""
    return nap.compute_tuning_curves(
        data=units, features=hd, bins=nb_bins, range=(0, 2 * np.pi), epochs=ep
    )


def shuffle_null(units, hd, ep, n_shuffles=200, nb_bins=120, seed=0, tqdm_desc=None):
    """Null distribution of mean-vector length and HD information.

    The head-direction time series is circularly shifted in time relative to the
    spikes by a random offset (>= 20 s), which preserves both the spike-train
    autocorrelation and the temporal structure of the behaviour while destroying
    the spike-to-direction pairing.
    """
    from tqdm.auto import tqdm

    rng = np.random.default_rng(seed)
    occ = angular_occupancy(hd, ep, nb_bins)
    hd_ep = hd.restrict(ep)
    vals = hd_ep.values
    n = len(vals)
    min_shift = max(1, int(20 / np.median(np.diff(hd_ep.t))))

    mvl = np.zeros((n_shuffles, len(units)))
    info = np.zeros((n_shuffles, len(units)))
    it = range(n_shuffles)
    if tqdm_desc:
        it = tqdm(it, desc=tqdm_desc, ncols=70, mininterval=10.0)
    for i in it:
        shift = rng.integers(min_shift, n - min_shift)
        hd_shuf = nap.Tsd(t=hd_ep.t, d=np.roll(vals, shift))
        tc = compute_tuning(units, hd_shuf, ep, nb_bins)
        for j, u in enumerate(tc.columns):
            mvl[i, j] = mean_vector(tc[u])[0]
            info[i, j] = hd_information(tc[u], occ)
    return mvl, info, list(units.keys())


def tuning_stats(units, hd, ep, nb_bins=120):
    """Per-unit tuning curve summary: rate, mean vector, HD information."""
    tc = compute_tuning(units, hd, ep, nb_bins)
    occ = angular_occupancy(hd, ep, nb_bins)
    rows = []
    for u in tc.columns:
        r, theta = mean_vector(tc[u])
        rows.append(
            {
                "unit": u,
                "mean_rate": float(len(units[u].restrict(ep)) / ep.tot_length()),
                "peak_rate": float(np.max(tc[u].values)),
                "mvl": r,
                "pref_dir": theta,
                "hd_info": hd_information(tc[u], occ),
            }
        )
    return tc, pd.DataFrame(rows).set_index("unit")
