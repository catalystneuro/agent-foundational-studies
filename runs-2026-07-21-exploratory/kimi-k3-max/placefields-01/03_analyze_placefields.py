"""Core place-field analysis on the cached Achilles maze data.

Computes, per unit:
  - 1D rate map over the linearized track position (80 x 2 cm bins)
  - Skaggs spatial information (bits/spike)
  - shuffle significance (circular time-shift within the run epoch)
  - direction-specific rate maps (positive vs negative velocity)
Saves results to achilles_placefields.npz
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

CACHE = "achilles_maze_cache.npz"
OUT = "achilles_placefields.npz"

NB_BINS = 80               # 2 cm bins over the 1.6 m track
TRACK_RANGE = (0.0, 1.6)
SMOOTH_SIGMA_BINS = 2.0    # 4 cm Gaussian smoothing of rate maps
MIN_OCC_S = 0.05           # ignore bins with < 50 ms occupancy in SI computation
N_SHUFFLE = 200
RNG_SEED = 42


def load_cache():
    d = np.load(CACHE, allow_pickle=False)
    unit_ids = d["unit_ids"]
    spikes = {u: d[f"spk_{u}"] for u in unit_ids}
    return dict(
        t=d["t"], xy=d["xy"], lin=d["lin"], speed=d["speed"],
        direction=d["direction"], fs=float(d["fs"]),
        run_starts=d["run_starts"], run_ends=d["run_ends"],
        unit_ids=unit_ids, cell_type=d["cell_type"], location=d["location"],
        n_spikes_run=d["n_spikes_run"], spikes=spikes,
    )


def bin_edges():
    return np.linspace(TRACK_RANGE[0], TRACK_RANGE[1], NB_BINS + 1)


def smooth_ratemap(counts, occ_s):
    """Occupancy-normalized, Gaussian-smoothed rate map. Zero-occupancy bins -> NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(occ_s > 0, counts / occ_s, np.nan)
    # smooth counts and occupancy separately then divide (standard approach)
    c_s = gaussian_filter1d(np.nan_to_num(counts), SMOOTH_SIGMA_BINS)
    o_s = gaussian_filter1d(np.nan_to_num(occ_s), SMOOTH_SIGMA_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_s = np.where(o_s > 0, c_s / o_s, np.nan)
    return rate, rate_s


def spatial_information(rate, occ_s):
    """Skaggs spatial information in bits/spike."""
    m = np.isfinite(rate) & (occ_s >= MIN_OCC_S)
    if m.sum() < 5:
        return np.nan
    p = occ_s[m] / occ_s[m].sum()
    r = rate[m]
    rbar = (p * r).sum()
    if rbar <= 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        si = np.nansum(p * (r / rbar) * np.log2(r / rbar))
    return si


def main():
    D = load_cache()
    t, lin, fs = D["t"], D["lin"], D["fs"]
    edges = bin_edges()
    centers = 0.5 * (edges[:-1] + edges[1:])

    # --- run-epoch sample mask and run-time coordinate (continuous across bouts) ---
    in_run = np.zeros(len(t), dtype=bool)
    for s, e in zip(D["run_starts"], D["run_ends"]):
        in_run[(t >= s) & (t <= e)] = True
    # drop samples with NaN position inside bouts
    in_run &= ~np.isnan(lin)
    idx_run = np.where(in_run)[0]
    t_run = t[idx_run]
    lin_run = lin[idx_run]
    dir_run = D["direction"][idx_run]
    dt = np.median(np.diff(t))
    # cumulative run time (seconds within the concatenated run epoch)
    rt = np.arange(len(t_run)) * dt

    # --- occupancy ---
    occ_counts, _ = np.histogram(lin_run, bins=edges)
    occ_s = occ_counts / fs
    # direction-specific occupancy
    occ_s_posdir = np.histogram(lin_run[dir_run > 0], bins=edges)[0] / fs
    occ_s_negdir = np.histogram(lin_run[dir_run < 0], bins=edges)[0] / fs

    print(f"run samples kept: {in_run.sum()}, total run time {rt[-1]:.1f} s")
    print(f"occupancy: median {np.median(occ_s):.2f} s/bin, "
          f"min {occ_s.min():.3f}, max {occ_s.max():.2f}")
    print(f"pos-dir time {occ_s_posdir.sum():.1f} s, neg-dir time {occ_s_negdir.sum():.1f} s")

    # --- per-unit rate maps ---
    unit_ids = D["unit_ids"]
    n_units = len(unit_ids)
    ratemaps = np.full((n_units, NB_BINS), np.nan)
    ratemaps_sm = np.full((n_units, NB_BINS), np.nan)
    ratemaps_pos = np.full((n_units, NB_BINS), np.nan)
    ratemaps_neg = np.full((n_units, NB_BINS), np.nan)
    si = np.full(n_units, np.nan)
    si_shuffle = np.full((n_units, N_SHUFFLE), np.nan)
    mean_rate = np.full(n_units, np.nan)
    peak_rate = np.full(n_units, np.nan)

    # position lookup as a function of run-time for shuffle shifts
    rng = np.random.default_rng(RNG_SEED)

    for i, u in enumerate(tqdm(unit_ids, desc="rate maps + shuffles")):
        spk = D["spikes"][u]
        if len(spk) < 20:
            continue
        # map spike times to run-time coordinate via interpolation onto sample grid
        spk_idx = np.searchsorted(t_run, spk)
        spk_idx = np.clip(spk_idx, 0, len(t_run) - 1)
        rt_spk = rt[spk_idx]
        lin_spk = lin_run[spk_idx]
        dir_spk = dir_run[spk_idx]

        counts, _ = np.histogram(lin_spk, bins=edges)
        rate, rate_s = smooth_ratemap(counts, occ_s)
        ratemaps[i] = rate
        ratemaps_sm[i] = rate_s
        si[i] = spatial_information(rate_s, occ_s)
        mean_rate[i] = len(spk) / rt[-1]
        peak_rate[i] = np.nanmax(rate_s)

        c_pos, _ = np.histogram(lin_spk[dir_spk > 0], bins=edges)
        c_neg, _ = np.histogram(lin_spk[dir_spk < 0], bins=edges)
        ratemaps_pos[i] = smooth_ratemap(c_pos, occ_s_posdir)[1]
        ratemaps_neg[i] = smooth_ratemap(c_neg, occ_s_negdir)[1]

        # --- circular time-shift shuffle within the run epoch ---
        total_rt = rt[-1] + dt
        for j in range(N_SHUFFLE):
            shift = rng.uniform(dt, total_rt)
            rt_sh = (rt_spk + shift) % total_rt
            lin_sh = np.interp(rt_sh, rt, lin_run)
            c_sh, _ = np.histogram(lin_sh, bins=edges)
            _, rate_sh = smooth_ratemap(c_sh, occ_s)
            si_shuffle[i, j] = spatial_information(rate_sh, occ_s)

    si_thresh_95 = np.nanpercentile(si_shuffle, 95, axis=1)
    is_place = (
        (D["cell_type"] == "excitatory")
        & (si > si_thresh_95)
        & (peak_rate >= 1.0)
        & (mean_rate >= 0.05)
    )
    n_place = int(is_place.sum())
    print(f"\nplace cells: {n_place}/{int((D['cell_type']=='excitatory').sum())} excitatory units")
    print(f"SI excitatory: median {np.nanmedian(si[D['cell_type']=='excitatory']):.3f} bits/spike")
    print(f"SI inhibitory: median {np.nanmedian(si[D['cell_type']=='inhibitory']):.3f} bits/spike")

    np.savez_compressed(
        OUT,
        centers=centers, edges=edges, occ_s=occ_s,
        occ_s_posdir=occ_s_posdir, occ_s_negdir=occ_s_negdir,
        ratemaps=ratemaps, ratemaps_sm=ratemaps_sm,
        ratemaps_pos=ratemaps_pos, ratemaps_neg=ratemaps_neg,
        si=si, si_shuffle=si_shuffle, si_thresh_95=si_thresh_95,
        mean_rate=mean_rate, peak_rate=peak_rate, is_place=is_place,
        unit_ids=unit_ids, cell_type=D["cell_type"], location=D["location"],
        n_spikes_run=D["n_spikes_run"],
    )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
