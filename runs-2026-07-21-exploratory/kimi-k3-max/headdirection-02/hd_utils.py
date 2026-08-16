"""Shared utilities for head-direction analysis of DANDI 000056 (Peyrache et al. 2015)."""
import h5py
import numpy as np
import pynapple as nap
import remfile
from pynwb import NWBHDF5IO

DANDI_API_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"

# Sessions used in this study (asset id, name). Chosen to span 5 mice with
# modest file sizes; the three ~30 GB Mouse12 files and Mouse32-140820
# (no states table) are excluded.
SESSIONS = {
    "Mouse17-130128": "4cc64fe0-7b1e-404c-8b86-fb5659292830",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse25-140123": "bdb30f7d-ba69-4d2e-8504-2efab69cd8d7",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
}


def load_session(asset_id, cache_dir="cache"):
    """Stream an NWB file from DANDI with remfile disk caching; return pynapple NWBFile."""
    disk_cache = remfile.DiskCache(cache_dir)
    rem_file = remfile.File(DANDI_API_URL.format(asset_id=asset_id), disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile), io


def get_sorted_units(nwb):
    """Return the units TsGroup with each unit's spike times sorted.

    At least one unit in this dataset has unsorted spike times (pynapple warns
    at load); searchsorted-based operations assume sorted times, so sort
    defensively. Metadata columns are preserved.
    """
    units = nwb["units"]
    data = {k: nap.Ts(np.sort(units[k].t)) for k in units.keys()}
    return nap.TsGroup(data)  # metadata dropped: 'rate' conflicts with TsGroup.rate


def compute_head_direction(nwb):
    """Head direction from the dual-LED tracking.

    HD = atan2(red_y - blue_y, red_x - blue_x) mapped to [0, 2*pi).
    Tracking failures are sentinel -1 values; those samples become NaN.
    Returns a pynapple Tsd (radians).
    """
    red = nwb["SubjectPosition/RedLED"]
    blue = nwb["SubjectPosition/BlueLED"]
    rv = np.asarray(red.values, dtype=float)
    bv = np.asarray(blue.values, dtype=float)
    t = np.asarray(red.t, dtype=float)
    if np.any(np.diff(t) <= 0):  # defensive: enforce strictly increasing times
        order = np.argsort(t, kind="stable")
        t, rv, bv = t[order], rv[order], bv[order]
    valid = np.all(rv > 0, axis=1) & np.all(bv > 0, axis=1)
    d = rv - bv
    hd = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    hd[~valid] = np.nan
    return nap.Tsd(t=t, d=hd, time_support=red.time_support)


def get_wake_epochs(nwb, label="Awake"):
    """IntervalSet of epochs with the given state label."""
    states = nwb["states"]
    return states[states["label"] == label]


def tuning_curves_hd(units, hd, wake, bins=60):
    """Occupancy-corrected HD tuning curves.

    Returns (tuning xarray in Hz, bin centers, occupancy in seconds).
    Bins with zero occupancy are NaN.
    """
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    counts = nap.compute_tuning_curves(
        units, hd, bins=[edges], epochs=wake, return_counts=True, return_pandas=False
    )
    occupancy = counts.attrs["occupancy"] / counts.attrs["fs"]  # seconds per bin
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = counts / occupancy
    rates = rates.where(occupancy > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return rates, centers, occupancy


def mean_vector_length(angles):
    """Mean resultant length of a set of angles (NaNs removed)."""
    angles = np.asarray(angles)
    angles = angles[~np.isnan(angles)]
    if len(angles) == 0:
        return np.nan
    return np.abs(np.exp(1j * angles).mean())


def spike_angles(unit_ts, hd):
    """HD angle at each spike time of one unit (NaNs dropped)."""
    ang = unit_ts.value_from(hd)
    ang = np.asarray(ang)
    return ang[~np.isnan(ang)]


def random_time_null(n_spikes, valid_hd_samples, n_shuffles=1000, rng=None,
                     max_n=100_000, chunk=100):
    """Null distribution of MVL under random-time resampling.

    Redraw n_spikes angles uniformly from the valid wake HD samples and
    compute the MVL. This respects the (non-uniform) HD occupancy, unlike a
    circular time-shift shuffle, which is invalid here because wake is
    fragmented into short epochs of nearly constant HD.

    n_spikes is capped at max_n (the null only gets tighter with more spikes,
    so the cap is conservative for the p-value) and sampling is chunked to
    bound memory.
    """
    rng = np.random.default_rng(rng)
    n = min(n_spikes, max_n)
    if n == 0 or len(valid_hd_samples) == 0:
        return np.full(n_shuffles, np.nan)
    out = np.empty(n_shuffles)
    for s in range(0, n_shuffles, chunk):
        e = min(s + chunk, n_shuffles)
        idx = rng.integers(0, len(valid_hd_samples), size=(e - s, n))
        out[s:e] = np.abs(np.exp(1j * valid_hd_samples[idx]).mean(axis=1))
    return out


def classify_hd_cells(units, hd, wake, n_shuffles=1000, mvl_floor=0.3, alpha=0.05, rng=None):
    """Classify units as HD cells: MVL above the random-time null AND MVL > floor.

    Returns a dict with per-unit arrays: mvl, p_value, preferred angle, is_hd.
    """
    rng = np.random.default_rng(rng)
    keys = list(units.keys())
    hd_wake = hd.restrict(wake)
    valid_hd = np.asarray(hd_wake.values)
    valid_hd = valid_hd[~np.isnan(valid_hd)]
    mvl = np.full(len(keys), np.nan)
    pval = np.full(len(keys), np.nan)
    pref = np.full(len(keys), np.nan)
    null_med = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        u_wake = units[k].restrict(wake)
        ang = spike_angles(u_wake, hd)
        if len(ang) < 50:
            continue
        mvl[i] = mean_vector_length(ang)
        pref[i] = np.angle(np.exp(1j * ang).mean()) % (2 * np.pi)
        null = random_time_null(len(u_wake), valid_hd, n_shuffles=n_shuffles, rng=rng)
        null_med[i] = np.nanmedian(null)
        pval[i] = (np.sum(null >= mvl[i]) + 1) / (np.sum(~np.isnan(null)) + 1)
    is_hd = (pval < alpha) & (mvl > mvl_floor)
    return {
        "keys": keys,
        "mvl": mvl,
        "p_value": pval,
        "pref_angle": pref,
        "null_median": null_med,
        "is_hd": is_hd,
    }
