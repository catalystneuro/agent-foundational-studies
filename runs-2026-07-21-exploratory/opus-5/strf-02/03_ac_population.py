"""Cortical STRFs across all 15 sessions of dandiset 000986.

For every unit we estimate the tone-pip STRF by reverse correlation, test it
against a circular-shift null, and extract best frequency, onset latency and a
suppression index.  Results are cached to ac_population.npz.
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm

import strf_lib as S

DT = 0.005
N_LAGS = 50          # 0-250 ms
N_SHUFFLE = 50
MIN_RATE = 0.5       # Hz, over the tone-presentation epoch
RNG = np.random.default_rng(0)
OUT = "ac_population.npz"


def onset_index_matrix(onsets, fidx, n_freq, t_bins, n_lags):
    """Per-frequency arrays of [onset, lag] indices into the binned spike train."""
    i0 = S.onset_bin(onsets, t_bins)
    lags = np.arange(n_lags)
    mats = []
    for k in range(n_freq):
        sel = i0[fidx == k]
        sel = sel[(sel >= 0) & (sel < len(t_bins) - n_lags)]
        mats.append(sel[:, None] + lags[None, :])
    return mats


def strf_from_index(counts, mats):
    """STRF (lag x freq) in spikes/s above the unit's mean rate."""
    m = counts.mean()
    return np.stack([counts[mm].mean(axis=0) - m for mm in mats], axis=1) / DT


def analyse_session(path):
    url = S.asset_url("000986", path)
    d = S.load_ac_session(url)
    units, trials = d["units"], d["trials"]
    freqs = S.TONE_FREQS_HZ

    onsets = trials["start_time"].values
    fidx = np.searchsorted(freqs, trials["stim_frequency"].values)
    t0, t1 = onsets[0] - 1.0, onsets[-1] + 2.0
    t_bins = np.arange(t0, t1, DT)
    n_t = len(t_bins)
    mats = onset_index_matrix(onsets, fidx, len(freqs), t_bins, N_LAGS)

    # arousal split: pupil diameter at each tone onset, lowest vs highest third
    pup = d["pupil"]
    p_at_onset = np.interp(onsets, pup.t, pup.d, left=np.nan, right=np.nan)
    lo_c, hi_c = np.nanpercentile(p_at_onset, [33.3, 66.7])
    lo = p_at_onset <= lo_c
    hi = p_at_onset >= hi_c
    mats_lo = onset_index_matrix(onsets[lo], fidx[lo], len(freqs), t_bins, N_LAGS)
    mats_hi = onset_index_matrix(onsets[hi], fidx[hi], len(freqs), t_bins, N_LAGS)

    rows = []
    for uid in tqdm(units.index, desc=path.split("/")[-1], leave=False):
        st = units[uid].t
        st = st[(st >= t0) & (st < t1)]
        rate = len(st) / (t1 - t0)
        if rate < MIN_RATE:
            continue
        counts = S.bin_spikes(st, t_bins)
        strf = strf_from_index(counts, mats)

        # circular-shift null: shift the spike train, keep the stimulus fixed
        null = np.empty((N_SHUFFLE, N_LAGS, len(freqs)))
        for s in range(N_SHUFFLE):
            shift = RNG.integers(int(5 / DT), n_t - int(5 / DT))
            null[s] = strf_from_index(np.roll(counts, shift), mats)
        mu, sd = null.mean(0), null.std(0)
        z = (strf - mu) / np.where(sd > 0, sd, np.inf)

        rows.append(dict(session=path, unit=int(uid), rate=rate,
                         strf=strf.astype(np.float32), z=z.astype(np.float32),
                         strf_lo=strf_from_index(counts, mats_lo).astype(np.float32),
                         strf_hi=strf_from_index(counts, mats_hi).astype(np.float32)))
    d["io"].close()
    return rows


def main():
    assets = [p for p, _, _ in S.list_assets("000986") if p.endswith(".nwb")]
    print(f"{len(assets)} sessions")
    all_rows = []
    for p in tqdm(assets, desc="sessions"):
        all_rows += analyse_session(p)

    strfs = np.stack([r["strf"] for r in all_rows])
    zs = np.stack([r["z"] for r in all_rows])
    np.savez_compressed(
        OUT,
        strf=strfs, z=zs,
        strf_lo=np.stack([r["strf_lo"] for r in all_rows]),
        strf_hi=np.stack([r["strf_hi"] for r in all_rows]),
        rate=np.array([r["rate"] for r in all_rows]),
        unit=np.array([r["unit"] for r in all_rows]),
        session=np.array([r["session"] for r in all_rows]),
        freqs=S.TONE_FREQS_HZ, dt=DT, n_lags=N_LAGS,
    )
    print("saved", OUT, strfs.shape)


if __name__ == "__main__":
    main()
