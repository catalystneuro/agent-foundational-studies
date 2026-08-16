"""Core computation for orientation selectivity (DANDI 000021, Allen Visual Coding NP).

Computes per-unit: orientation tuning at preferred temporal frequency, gOSI
(global orientation selectivity index), peak response above blank, preferred
orientation, and a permutation-test p-value for orientation selectivity.
Saves a results npz for the figure stage.
"""
import numpy as np
import requests
import h5py
import remfile
import pynapple as nap
from pynwb import NWBHDF5IO

ASSET = "58703c97-c0a9-4736-b684-73c85c1a444a"
ORIS_DEG = np.array([0, 45, 90, 135], dtype=float)
TFS = np.array([1.0, 2.0, 4.0, 8.0, 15.0])
THETA = np.deg2rad(ORIS_DEG)


def resolve_asset_url(asset_id):
    r = requests.get(f"https://api.dandiarchive.org/api/assets/{asset_id}/")
    r.raise_for_status()
    urls = r.json()["contentUrl"]
    return [u for u in urls if "dandiarchive.s3" in u][0]


def load_nwb(asset):
    url = resolve_asset_url(asset)
    dc = remfile.DiskCache('/tmp/remfile_cache_orientation')
    nf = h5py.File(remfile.File(url, disk_cache=dc), "r")
    io = NWBHDF5IO(file=nf)
    nwbfile = io.read()
    return nwbfile


def gosi_from_resp(resp, blank):
    """gOSI of a length-4 orientation response vector (baseline-subtracted)."""
    r = np.asarray(resp, dtype=float) - blank
    num = np.sqrt((np.sum(r * np.cos(2 * THETA)) ** 2) +
                  (np.sum(r * np.sin(2 * THETA)) ** 2))
    den = np.sum(r)
    return num / den if den > 0 else np.nan


def count_in_intervals(spikes, starts, stops):
    i0 = np.searchsorted(spikes, starts, side='left')
    i1 = np.searchsorted(spikes, stops, side='left')
    return (i1 - i0).astype(float) / (stops - starts)


def run():
    nwbfile = load_nwb(ASSET)
    nwb = nap.NWBFile(nwbfile)
    units = nwb['units']

    # ---- unit -> brain area via peak channel -> electrode location ----
    el = nwbfile.electrodes
    loc_arr = np.asarray(el['location'][:])
    loc_series = {int(i): (loc_arr[idx] if loc_arr[idx] else 'unknown')
                  for idx, i in enumerate(el.id[:])}
    peak = np.asarray(units['peak_channel_id'].values, dtype=int)
    unit_area = np.array([loc_series.get(int(p), 'unknown') for p in peak])
    quality = np.asarray(units['quality'].values)

    # ---- drifting gratings tables ----
    dg = nwbfile.intervals['drifting_gratings_presentations'].to_dataframe()
    starts = dg['start_time'].values
    stops = dg['stop_time'].values
    dur = stops - starts
    ori_raw = dg['orientation'].values.astype(float)
    tf = dg['temporal_frequency'].values.astype(float)

    # ---- exclude presentations overlapping invalid_times ----
    inv = nwbfile.intervals['invalid_times'].to_dataframe()
    overlap = np.zeros(len(dg), dtype=bool)
    for _, r in inv.iterrows():
        overlap |= (starts < r['stop_time']) & (stops > r['start_time'])
    valid = ~overlap
    print(f"total presentations: {len(dg)}, overlap invalid: {overlap.sum()}, valid: {valid.sum()}")

    vs = starts[valid]; vst = stops[valid]; vd = dur[valid]
    vt = tf[valid]
    vori = ori_raw[valid]
    blank_mask = np.isnan(vori)

    # folded orientation 0..135
    vori_f = np.mod(vori, 180.0)
    ori_idx = np.array([np.argmin(np.abs(ORIS_DEG - o)) for o in vori_f])
    tf_idx = np.array([np.argmin(np.abs(TFS - t)) for t in vt])

    # ---- per-unit firing rates over presentations ----
    n_tri = vs.shape[0]
    t0 = vs - 0.08   # account for 0.08 s latency/delay window
    fr = np.zeros((len(units), n_tri))
    unit_ids = np.array(list(units.keys()))
    for ui, grp in enumerate(units):
        ts = units[grp].t
        fr[ui] = count_in_intervals(ts, t0, vst)

    # blank rate per unit (mean over blank presentations)
    blank = fr[:, blank_mask].mean(axis=1)

    # non-blank trial set
    nb = ~blank_mask
    F = fr[:, nb]
    g = tf_idx[nb] * 4 + ori_idx[nb]
    cnt = np.bincount(g, minlength=20)

    # ---- observed tuning + gOSI ----
    n_tri_nb = nb.sum()
    n_units = F.shape[0]
    M = np.zeros((n_tri_nb, 20))
    M[np.arange(n_tri_nb), g] = 1.0 / cnt[g]
    means = (F @ M).reshape(n_units, 5, 4)         # n_units x tf x ori
    mean_over_ori = np.nanmean(means, axis=2)     # n_units x tf
    pref_tf_idx = np.argmax(mean_over_ori, axis=1)
    resp = means[np.arange(F.shape[0]), pref_tf_idx, :]   # n_units x 4
    r_baseline = resp - blank[:, None]
    peak_above_blank = np.nanmax(r_baseline, axis=1)
    pref_ori_idx = np.nanargmax(r_baseline, axis=1)

    # standard Allen gOSI on raw mean responses (bounded ~[0,1])
    num = np.sqrt((np.sum(resp * np.cos(2 * THETA), axis=1)) ** 2 +
                  (np.sum(resp * np.sin(2 * THETA), axis=1)) ** 2)
    den = np.sum(resp, axis=1)
    gosi = num / den
    gosi[den <= 0] = np.nan

    print("observed done", gosi.shape)

    # ---- permutation test (vectorized across units) ----
    rng = np.random.default_rng(0)
    NPERM = 200
    n_units = F.shape[0]
    tfnb = tf_idx[nb]
    orinb = ori_idx[nb]
    null_gosi = np.zeros((n_units, NPERM))
    for s in range(NPERM):
        # permute orientation within each tf for this shuffle
        ori_perm = orinb.copy()
        for t in range(5):
            sel = np.where(tfnb == t)[0]
            ori_perm[sel] = rng.permutation(orinb[sel])
        gp = tfnb * 4 + ori_perm
        Mp2 = np.zeros((n_tri_nb, 20))
        Mp2[np.arange(n_tri_nb), gp] = 1.0 / cnt[gp]
        Mp = (F @ Mp2).reshape(n_units, 5, 4)
        pref_tfp = np.argmax(np.nanmean(Mp, axis=2), axis=1)
        resp_p = Mp[np.arange(n_units), pref_tfp, :]
        denp = np.sum(resp_p, axis=1)
        nump = np.sqrt((np.sum(resp_p * np.cos(2 * THETA), axis=1)) ** 2 +
                       (np.sum(resp_p * np.sin(2 * THETA), axis=1)) ** 2)
        gpv = nump / denp
        gpv[denp <= 0] = np.nan
        null_gosi[:, s] = gpv
    # p-value (count null >= observed); NaN observed -> NaN p
    obs = gosi[:, None]
    p_val = (np.sum(null_gosi >= obs, axis=1) + 1) / (NPERM + 1)
    p_val = np.asarray(p_val, dtype=float)
    p_val[np.isnan(gosi)] = np.nan

    np.savez_compressed(
        'results.npz',
        unit_ids=unit_ids, unit_area=unit_area, quality=quality,
        gosi=gosi, peak_above_blank=peak_above_blank,
        pref_ori_deg=ORIS_DEG[pref_ori_idx], pref_tf_deg=TFS[pref_tf_idx],
        p_value=p_val, blank=blank,
        means_response=means,  # n_units x tf x ori
    )
    print("saved results.npz")

    # quick validation summary
    visp = (unit_area == 'VISp') & (quality == 'good')
    sel = (p_val < 0.05) & (peak_above_blank >= 1.0)
    for area in ['VISp', 'VISl', 'VISam', 'VISpm', 'LGd', 'CA1']:
        m = (unit_area == area) & (quality == 'good')
        n = m.sum()
        frac = (sel & m).sum() / n if n else np.nan
        print(f"{area}: good units {n}, selective {frac*100:.1f}%")


if __name__ == "__main__":
    run()
