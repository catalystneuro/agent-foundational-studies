import h5py
from pynwb import NWBHDF5IO
import remfile
import pynapple as nap
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from grid_lib import spatial_autocorrelogram, gridness_score
import pickle

SESSIONS = {
    "ses-03020601": "https://dandiarchive.s3.amazonaws.com/blobs/275/de6/275de639-185d-433c-866b-edad9bfcdcc8",
    "ses-01020602": "https://dandiarchive.s3.amazonaws.com/blobs/62f/e98/62fe9813-a6f2-46bf-bdad-08918c23eae1",
    "ses-02020601": "https://dandiarchive.s3.amazonaws.com/blobs/fd3/e8f/fd3e8f90-9ac0-4a47-a487-8587759ee92e",
    "ses-06020601": "https://dandiarchive.s3.amazonaws.com/blobs/94a/c41/94ac4119-1a14-42cb-a2e4-9a8d0ac9950d",
}

N_SHUFFLES = 100
N_BINS = 30
MIN_SHIFT = 20.0  # seconds, minimum circular shift for shuffle test


def load_session(url):
    disk_cache = remfile.DiskCache('remfile_cache')
    rem_file = remfile.File(url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file, load_namespaces=True)
    nwbfile = io.read()
    nwb = nap.NWBFile(nwbfile)
    pos = nwb['SpatialSeriesLED1']
    position = nap.TsdFrame(t=pos.t, d=pos.values, columns=['x', 'y']).dropna()
    units = nwb['units']
    histology = nwbfile.units.histology[:]
    return position, units, histology


def rate_map_for_unit(spike_ts, position, bins=N_BINS):
    tc = nap.compute_tuning_curves(spike_ts, position, bins=bins, epochs=position.time_support)
    return tc.values.T if tc.ndim == 2 else tc.squeeze().values.T


def gridness_for_ts(spike_ts, position, bins=N_BINS):
    rate_map = rate_map_for_unit(spike_ts, position, bins=bins)
    valid = ~np.isnan(rate_map)
    smoothed = gaussian_filter(np.nan_to_num(rate_map), sigma=1.0)
    smoothed[~valid] = np.nan
    ac = spatial_autocorrelogram(smoothed)
    return gridness_score(ac), rate_map, ac


def shuffle_null_distribution(spike_train, position, n_shuffles=N_SHUFFLES, seed=0):
    rng = np.random.default_rng(seed)
    t0, t1 = position.time_support.start[0], position.time_support.end[0]
    duration = t1 - t0
    spike_times = spike_train.index.values
    null_scores = np.full(n_shuffles, np.nan)
    for i in range(n_shuffles):
        shift = rng.uniform(MIN_SHIFT, duration - MIN_SHIFT)
        shifted = (spike_times - t0 + shift) % duration + t0
        shifted_ts = nap.Ts(t=np.sort(shifted))
        score, _, _ = gridness_for_ts(shifted_ts, position)
        null_scores[i] = score
    return null_scores


def analyze_session(session_name, url):
    print(f"\n=== Loading {session_name} ===")
    position, units, histology = load_session(url)
    duration = position.time_support.end[0] - position.time_support.start[0]
    print(f"  duration={duration:.1f}s, n_units={len(units)}, n_pos_samples={len(position)}")

    results = []
    for uid in tqdm(units.index, desc=f"{session_name} units"):
        spike_train = units[uid]
        score, rate_map, ac = gridness_for_ts(spike_train, position)
        null_scores = shuffle_null_distribution(spike_train, position)
        pctile_95 = np.nanpercentile(null_scores, 95)
        is_grid_cell = (not np.isnan(score)) and score > pctile_95
        results.append({
            'session': session_name,
            'unit_id': int(uid),
            'histology': histology[uid] if uid < len(histology) else '',
            'n_spikes': len(spike_train),
            'gridness': score,
            'shuffle_95pct': pctile_95,
            'shuffle_null': null_scores,
            'is_grid_cell': is_grid_cell,
            'rate_map': rate_map,
            'autocorr': ac,
        })
        print(f"    unit {uid}: gridness={score:.3f}, shuffle_95%={pctile_95:.3f}, grid_cell={is_grid_cell}")

    return {'position': position, 'results': results}


if __name__ == '__main__':
    all_data = {}
    for name, url in SESSIONS.items():
        all_data[name] = analyze_session(name, url)

    with open('grid_analysis_results.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print("\nSaved grid_analysis_results.pkl")

    total_units = sum(len(d['results']) for d in all_data.values())
    total_grid = sum(sum(r['is_grid_cell'] for r in d['results']) for d in all_data.values())
    print(f"\nTOTAL: {total_grid}/{total_units} units classified as grid cells across {len(SESSIONS)} sessions")
