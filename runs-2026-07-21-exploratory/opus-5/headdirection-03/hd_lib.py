"""
Shared loading + analysis helpers for the head-direction demonstration.

Dataset: DANDI:000939 -- "Large-scale recordings of head direction cells in
mouse postsubiculum" (Duszkiewicz et al.), 64-channel silicon probes in mouse
postsubiculum (PoSub) during free foraging in two environments plus home-cage
sleep.

Files are ~20-30 GB each but we only ever touch a few tens of MB (spike times,
head-direction time series, position, epoch tables), so everything is streamed
with remfile + a local disk cache. The small extracted arrays are additionally
cached as .npz so that repeated analysis runs do not re-hit the network.
"""

import json
import os

import numpy as np
import pynapple as nap

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
REMFILE_CACHE = "/tmp/remfile_cache"
DANDISET = "000939"

# Sessions without optogenetic manipulation (one session per animal).
SESSIONS = [
    "sub-A3701_ses-191119",
    "sub-A3702_ses-191126",
    "sub-A3703_ses-191215",
    "sub-A3705_ses-200306",
    "sub-A3706_ses-200313",
    "sub-A3707_ses-200317",
    "sub-A3709_ses-200601",
    "sub-A3710_ses-200609",
    "sub-A5505_ses-200831",
    "sub-A5506_ses-200914a",
]


def _asset_map():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets_000939.json")
    if not os.path.exists(path):
        import requests

        r = requests.get(
            f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/assets/",
            params={"page_size": 100},
        ).json()
        m = {
            x["path"].split("/")[-1].replace(".nwb", ""): x["asset_id"]
            for x in r["results"]
        }
        json.dump(m, open(path, "w"), indent=1)
    return json.load(open(path))


def asset_url(session):
    m = _asset_map()
    key = [k for k in m if k.startswith(session)]
    if len(key) != 1:
        raise KeyError(f"{session} -> {key}")
    return (
        f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/draft/"
        f"assets/{m[key[0]]}/download/"
    )


def extract_session(session, force=False):
    """Stream one NWB file and cache the small arrays we need as .npz."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz = os.path.join(CACHE_DIR, f"{session}.npz")
    if os.path.exists(npz) and not force:
        return npz

    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    rem = remfile.File(asset_url(session), disk_cache=remfile.DiskCache(REMFILE_CACHE))
    nwbfile = NWBHDF5IO(file=h5py.File(rem, "r"), load_namespaces=True).read()

    hd_ts = nwbfile.processing["behavior"]["CompassDirection"]["head-direction"]
    pos = nwbfile.processing["behavior"]["Position"]["position"]

    units = nwbfile.units
    spike_times = units["spike_times"][:]  # list of arrays
    counts = np.array([len(s) for s in spike_times])
    flat = np.concatenate(spike_times)
    starts = np.concatenate([[0], np.cumsum(counts)])

    ep = nwbfile.intervals["epochs"].to_dataframe()
    ss = nwbfile.intervals["sleep_states"].to_dataframe()

    elec = nwbfile.electrodes.to_dataframe()
    udf = units.to_dataframe()

    np.savez_compressed(
        npz,
        session=session,
        subject=nwbfile.subject.subject_id,
        spikes_flat=flat,
        spikes_index=starts,
        is_head_direction=udf["is_head_direction"].values.astype(int),
        is_excitatory=udf["is_excitatory"].values.astype(int),
        is_fast_spiking=udf["is_fast_spiking"].values.astype(int),
        trough_to_peak=udf["trough_to_peak"].values.astype(float),
        electrode_index=udf["electrode_index"].values.astype(int),
        unit_depth=elec["rel_y"].values[udf["electrode_index"].values.astype(int)],
        hd=np.asarray(hd_ts.data[:], dtype=float),
        hd_t=np.asarray(hd_ts.timestamps[:], dtype=float),
        pos=np.asarray(pos.data[:], dtype=float),
        pos_t=np.asarray(pos.timestamps[:], dtype=float)
        if pos.timestamps is not None
        else np.arange(pos.data.shape[0]) / pos.rate,
        ep_start=ep["start_time"].values,
        ep_stop=ep["stop_time"].values,
        ep_tag=np.array([t[0] for t in ep["tags"].values]),
        ss_start=ss["start_time"].values.astype(float),
        ss_stop=ss["stop_time"].values.astype(float),
        ss_state=ss["state"].values.astype(str),
        location=str(elec["location"].iloc[0]),
    )
    return npz


def load_session(session):
    """Return a dict of pynapple objects for one session (from the .npz cache)."""
    d = np.load(extract_session(session), allow_pickle=False)

    flat, idx = d["spikes_flat"], d["spikes_index"]
    spikes = {i: flat[idx[i] : idx[i + 1]] for i in range(len(idx) - 1)}
    metadata = {
        "is_hd_author": d["is_head_direction"].astype(bool),
        "is_excitatory": d["is_excitatory"].astype(bool),
        "is_fast_spiking": d["is_fast_spiking"].astype(bool),
        "trough_to_peak": d["trough_to_peak"],
        "depth_um": d["unit_depth"],
    }
    units = nap.TsGroup(spikes, metadata=metadata)

    hd = nap.Tsd(t=d["hd_t"], d=d["hd"])
    position = nap.TsdFrame(t=d["pos_t"], d=d["pos"], columns=["x", "y"])

    epochs = {}
    for s, e, tag in zip(d["ep_start"], d["ep_stop"], d["ep_tag"]):
        epochs.setdefault(str(tag), []).append((s, e))
    epochs = {k: nap.IntervalSet(start=[a for a, _ in v], end=[b for _, b in v])
              for k, v in epochs.items()}

    states = {}
    for s, e, st in zip(d["ss_start"], d["ss_stop"], d["ss_state"]):
        states.setdefault(str(st), []).append((s, e))
    states = {k: nap.IntervalSet(start=[a for a, _ in v], end=[b for _, b in v])
              for k, v in states.items()}

    return dict(
        session=session,
        subject=str(d["subject"]),
        location=str(d["location"]),
        units=units,
        hd=hd,
        position=position,
        epochs=epochs,
        states=states,
    )


# ---------------------------------------------------------------- metrics ---

def circular_mean_vector(bin_centers, rates):
    """Mean resultant vector (length, angle) of a circular tuning curve."""
    rates = np.asarray(rates, dtype=float)
    if rates.sum() <= 0:
        return 0.0, np.nan
    z = np.sum(rates * np.exp(1j * bin_centers)) / rates.sum()
    return np.abs(z), np.mod(np.angle(z), 2 * np.pi)


def hd_information(bin_centers, rates, occupancy):
    """Skaggs spatial (here directional) information in bits/spike."""
    rates = np.asarray(rates, dtype=float)
    p = np.asarray(occupancy, dtype=float)
    p = p / p.sum()
    mean_rate = np.sum(p * rates)
    if mean_rate <= 0:
        return 0.0
    ok = rates > 0
    return float(np.sum(p[ok] * (rates[ok] / mean_rate) * np.log2(rates[ok] / mean_rate)))


def tuning_curves(units, hd, ep, nb_bins=60):
    """Pynapple 1-D tuning curves over head direction, plus occupancy."""
    tc = nap.compute_1d_tuning_curves(units, hd, nb_bins=nb_bins,
                                      minmax=(0, 2 * np.pi), ep=ep)
    hd_ep = hd.restrict(ep)
    edges = np.linspace(0, 2 * np.pi, nb_bins + 1)
    occ, _ = np.histogram(hd_ep.values, edges)
    return tc, occ.astype(float), tc.index.values


class FastTuning:
    """Precomputed head-direction binning, for cheap tuning curves in shuffles.

    Equivalent to ``nap.compute_1d_tuning_curves`` (nearest-sample assignment of
    each spike to a head-direction bin, divided by the occupancy in seconds);
    validated against it in the analysis script.
    """

    def __init__(self, hd, ep, nb_bins=60):
        h = hd.restrict(ep)
        t, v = h.times(), h.values
        ok = ~np.isnan(v)
        self.t, v = t[ok], v[ok]
        self.nb_bins = nb_bins
        self.edges = np.linspace(0, 2 * np.pi, nb_bins + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.bin_of_sample = np.clip(np.digitize(v, self.edges) - 1, 0, nb_bins - 1)
        self.dt = float(np.median(np.diff(self.t)))
        self.occupancy = np.bincount(self.bin_of_sample, minlength=nb_bins) * self.dt
        self.ep = ep

    def curve(self, spike_times):
        s = np.asarray(spike_times)
        s = s[(s >= self.t[0]) & (s <= self.t[-1])]
        if s.size == 0:
            return np.zeros(self.nb_bins)
        i = np.searchsorted(self.t, s)
        i = np.clip(i, 1, len(self.t) - 1)
        left = np.abs(s - self.t[i - 1]) < np.abs(self.t[i] - s)
        i = np.where(left, i - 1, i)
        # drop spikes that fall in a tracking gap (no sample within one frame)
        i = i[np.abs(s - self.t[i]) < 5 * self.dt]
        cnt = np.bincount(self.bin_of_sample[i], minlength=self.nb_bins)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.occupancy > 0, cnt / self.occupancy, 0.0)


def circ_shift_shuffle(unit_ts, ep, rng, n_shuffles=200):
    """Yield circularly time-shifted copies of a spike train within an epoch.

    Shifting preserves each cell's autocorrelation/burst structure while
    destroying its temporal relationship to head direction, which is the
    standard null for directional tuning.
    """
    t = unit_ts.restrict(ep).times()
    # Work in "epoch time": concatenate the epoch's intervals.
    starts, ends = ep.start, ep.end
    durs = ends - starts
    total = durs.sum()
    offs = np.concatenate([[0], np.cumsum(durs)])
    seg = np.searchsorted(ends, t, side="left")
    seg = np.clip(seg, 0, len(starts) - 1)
    tc = t - starts[seg] + offs[seg]
    for _ in range(n_shuffles):
        shift = rng.uniform(0.1 * total, 0.9 * total)
        ts = np.mod(tc + shift, total)
        seg2 = np.clip(np.searchsorted(offs, ts, side="right") - 1, 0, len(starts) - 1)
        yield np.sort(ts - offs[seg2] + starts[seg2])


def circ_tuning_width(bin_centers, rates):
    """Full width at half maximum of a circular tuning curve, in radians."""
    rates = np.asarray(rates, dtype=float)
    if rates.max() <= 0:
        return np.nan
    half = 0.5 * (rates.max() + rates.min())
    # rotate so the peak sits at the centre, then measure the contiguous run
    k = int(np.argmax(rates))
    rr = np.roll(rates, len(rates) // 2 - k)
    c = len(rr) // 2
    lo = c
    while lo > 0 and rr[lo - 1] >= half:
        lo -= 1
    hi = c
    while hi < len(rr) - 1 and rr[hi + 1] >= half:
        hi += 1
    return (hi - lo + 1) * (2 * np.pi / len(rates))


def angdiff(a, b):
    """Signed circular difference wrapped to (-pi, pi]."""
    return np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi
