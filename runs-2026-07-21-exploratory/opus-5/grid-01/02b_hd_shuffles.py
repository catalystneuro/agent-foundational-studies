"""Shuffle control for head-direction tuning.

The mean vector length of a low-firing cell is biased upward by sampling alone,
so a cell only counts as directionally tuned if its MVL beats the 95th
percentile of MVLs obtained from circularly time-shifted copies of its own
spike train. Same shuffle procedure as the gridness test.
"""

import pickle
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import gridlib as G

N_SHUFFLE = 100
MIN_SHIFT_S = 20.0
NBINS = 36


def mvl_from_spikes(spike_t, hd_t, hd_v, occ, edges, centers):
    """Occupancy-normalised HD tuning curve and its mean vector length."""
    hd_at_spike = np.interp(spike_t, hd_t, np.unwrap(hd_v)) % (2 * np.pi)
    cnt, _ = np.histogram(hd_at_spike, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(occ > 0, cnt / np.maximum(occ, 1e-9), np.nan)
    w = np.nan_to_num(curve)
    if w.sum() <= 0:
        return curve, 0.0
    z = np.sum(w * np.exp(1j * centers)) / w.sum()
    return curve, float(np.abs(z))


def analyze_session(asset):
    s = G.load_session(asset)
    if s["hd"] is None:
        s["io"].close()
        return []
    ep = G.run_epochs(s)
    hd = s["hd"].restrict(ep)
    hd_t, hd_v = hd.index.values, hd.values
    edges = np.linspace(0, 2 * np.pi, NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occ, _ = np.histogram(hd_v, bins=edges)
    occ = occ * s["dt"]

    t0, t1 = float(s["position"].index[0]), float(s["position"].index[-1])
    rng = np.random.default_rng(abs(hash(asset["path"] + "hd")) % (2 ** 31))
    out = []
    for i, k in enumerate(s["units"].keys()):
        st = s["units"][k]
        spk = st.restrict(ep).index.values
        if len(spk) < G.MIN_SPIKES:
            continue
        _, mvl = mvl_from_spikes(spk, hd_t, hd_v, occ, edges, centers)
        offs = rng.uniform(MIN_SHIFT_S, (t1 - t0) - MIN_SHIFT_S, N_SHUFFLE)
        sh = np.array([mvl_from_spikes(
            G.shifted_spikes(st, t0, t1, o).restrict(ep).index.values,
            hd_t, hd_v, occ, edges, centers)[1] for o in offs])
        out.append(dict(session=s["path"], unit=s["unit_names"][i],
                        mvl=mvl, mvl_shuffle=sh))
    s["io"].close()
    return out


if __name__ == "__main__":
    assets = G.load_assets()
    rows = []
    with Pool(8) as pool:
        for r in tqdm(pool.imap_unordered(analyze_session, assets),
                      total=len(assets), desc="HD shuffles"):
            rows.extend(r)
    pickle.dump(rows, open("hd_shuffles.pkl", "wb"))
    pooled = np.concatenate([r["mvl_shuffle"] for r in rows])
    thr = float(np.nanpercentile(pooled, 95))
    mvls = np.array([r["mvl"] for r in rows])
    print(f"{len(rows)} units with head-direction tracking")
    print(f"MVL shuffle threshold = {thr:.3f}; "
          f"{(mvls > thr).sum()} directionally tuned "
          f"({100 * (mvls > thr).mean():.1f}%)")
