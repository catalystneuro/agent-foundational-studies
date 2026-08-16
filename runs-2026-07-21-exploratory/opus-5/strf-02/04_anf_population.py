"""Auditory-nerve-fibre receptive fields across dandiset 001262.

Each NWB file is one single fibre.  The BF protocol sweeps tone frequency on a
fine grid centred near the fibre's characteristic frequency, 5 repetitions per
frequency, so a frequency x time response field can be measured directly.
Results are cached to anf_population.npz.
"""

import sys

import numpy as np
from tqdm import tqdm

import strf_lib as S

BIN_MS = 1.0
N_FIBRES = int(sys.argv[1]) if len(sys.argv) > 1 else 160
N_GRID = 41           # common CF-relative frequency grid (octaves)
N_TIME = 80           # ms of trial time kept
OUT = "anf_population.npz"
RNG = np.random.default_rng(1)


def resample_to_octave_grid(R, freqs, bf, grid):
    """Interpolate a fibre's response field onto a common log-frequency grid
    expressed in octaves relative to that fibre's best frequency."""
    oct_axis = np.log2(freqs / bf)
    out = np.full((R.shape[0], len(grid)), np.nan)
    inside = (grid >= oct_axis[0]) & (grid <= oct_axis[-1])
    for i in range(R.shape[0]):
        out[i, inside] = np.interp(grid[inside], oct_axis, R[i])
    return out


def main():
    assets = [(p, u) for p, u, _ in S.list_assets("001262") if p.endswith(".nwb")]
    sel = [assets[i] for i in RNG.permutation(len(assets))[:N_FIBRES]]

    grid = np.linspace(-1.0, 1.0, N_GRID)
    n_bins = int(N_TIME / BIN_MS)

    recs = []
    n_no_bf = 0
    for path, url in tqdm(sel, desc="fibres"):
        fb = S.load_anf_fibre(url)
        if fb is None:
            n_no_bf += 1
            continue
        if len(fb["freqs"]) < 8 or not np.isfinite(fb["bf_published"]):
            fb["h5"].close()
            n_no_bf += 1
            continue
        R, t, fq = S.anf_strf(fb, bin_ms=BIN_MS, t_max=N_TIME / 1000.0)
        if R.shape[0] < n_bins:
            continue
        R = R[:n_bins]

        # driven response measured over the tone (10-60 ms) against the
        # pre-onset window (0-8 ms) which is spontaneous activity
        drive = R[10:60].mean(0) - R[:8].mean()
        if np.all(~np.isfinite(drive)) or drive.max() <= 0:
            continue
        bf_measured = fq[int(np.argmax(drive))]

        aligned = resample_to_octave_grid(R, fq, bf_measured, grid)
        prof = R[:, int(np.argmax(drive))]          # time course at measured BF

        recs.append(dict(
            path=path,
            bf_pub=fb["bf_published"], bf_meas=bf_measured,
            sr=fb["sr_published"], thr=fb["threshold_published"],
            n_freq=len(fq), span_oct=np.log2(fq[-1] / fq[0]),
            aligned=aligned.astype(np.float32),
            prof=prof.astype(np.float32),
            spont=float(R[:8].mean()),
        ))
        fb["h5"].close()

    print(f"kept {len(recs)} fibres; {n_no_bf} files had no usable BF sweep")
    np.savez_compressed(
        OUT,
        aligned=np.stack([r["aligned"] for r in recs]),
        prof=np.stack([r["prof"] for r in recs]),
        bf_pub=np.array([r["bf_pub"] for r in recs]),
        bf_meas=np.array([r["bf_meas"] for r in recs]),
        sr=np.array([r["sr"] for r in recs]),
        thr=np.array([r["thr"] for r in recs]),
        spont=np.array([r["spont"] for r in recs]),
        span_oct=np.array([r["span_oct"] for r in recs]),
        n_freq=np.array([r["n_freq"] for r in recs]),
        path=np.array([r["path"] for r in recs]),
        grid=grid, bin_ms=BIN_MS, n_time=N_TIME,
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()
