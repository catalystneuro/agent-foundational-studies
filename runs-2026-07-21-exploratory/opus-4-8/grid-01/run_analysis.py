"""Multi-session grid-cell analysis over DANDI dandiset 000582.

Pools units from the richest sessions, computes rate maps, autocorrelograms,
gridness scores, and a shuffle-based significance threshold, then caches all
results to disk for the figure-generation step.
"""
import numpy as np
import pickle
from tqdm import tqdm
import gridcells as gc

BINS = 40
BOX = (-50.0, 50.0)
BIN_SIZE = (BOX[1] - BOX[0]) / BINS   # 2.5 cm
SIGMA = 1.0
N_SHUFFLE = 30
MIN_SPIKES = 100

# Richest sessions from the survey (unit counts in parentheses); all 2-LED.
SESSIONS = [
    "sub-11265/sub-11265_ses-06020601_behavior+ecephys.nwb",  # 19
    "sub-11207/sub-11207_ses-18060501_behavior+ecephys.nwb",  # 18
    "sub-11207/sub-11207_ses-27060501_behavior+ecephys.nwb",  # 14
    "sub-11265/sub-11265_ses-16030604_behavior+ecephys.nwb",  # 14
    "sub-11207/sub-11207_ses-21060503_behavior+ecephys.nwb",  # 13
    "sub-11265/sub-11265_ses-09020601_behavior+ecephys.nwb",  # 13
    "sub-11207/sub-11207_ses-11060501_behavior+ecephys.nwb",  # 12
    "sub-11265/sub-11265_ses-16030601_behavior+ecephys.nwb",  # 12
]


def process_session(path):
    nwb = gc.load_session(path)
    t, xy = gc.extract_position(nwb)
    # drop any NaN position samples (interpolate small gaps)
    good = np.isfinite(xy).all(axis=1)
    if not good.all():
        xy = xy[good]
        t = t[good]
    units = gc.extract_units(nwb)
    occ, ex, ey = gc.occupancy_map(t, xy, BINS, BOX, sigma=SIGMA)
    recs = []
    for u in units:
        st = u["spike_times"]
        st = st[(st >= t[0]) & (st <= t[-1])]
        if len(st) < MIN_SPIKES:
            continue
        rm, ac, g, si = gc.analyze_unit(st, t, xy, occ, ex, BIN_SIZE, SIGMA)
        null = gc.shuffle_gridness(st, t, xy, occ, ex, BIN_SIZE,
                                   n_shuffles=N_SHUFFLE, seed=len(recs))
        recs.append(dict(
            session=path.split("/")[-1].replace("_behavior+ecephys.nwb", ""),
            subject=path.split("/")[0],
            name=u["name"], histology=u["histology"], depth=u["depth"],
            n_spikes=int(len(st)), mean_rate=len(st) / (t[-1] - t[0]),
            gridness=g["gridness"], spacing_cm=g["spacing_cm"],
            orientation_deg=g.get("orientation_deg", np.nan),
            spatial_info=si, peak_rate=float(np.nanmax(rm)),
            rate_map=rm, autocorr=ac, null_gridness=null,
            spike_x=np.interp(st, t, xy[:, 0]),
            spike_y=np.interp(st, t, xy[:, 1]),
        ))
    return dict(path=path, t=t, xy=xy, occ=occ, edges=ex, records=recs)


def main():
    sessions = []
    all_recs = []
    for path in tqdm(SESSIONS, desc="sessions"):
        s = process_session(path)
        sessions.append(s)
        all_recs.extend(s["records"])
        print(f"  {path.split('/')[-1]}: {len(s['records'])} units "
              f"(>={MIN_SPIKES} spikes)")

    # Shuffle-based grid-cell threshold: 95th percentile of the pooled null.
    pooled_null = np.concatenate([r["null_gridness"] for r in all_recs])
    pooled_null = pooled_null[np.isfinite(pooled_null)]
    thr = float(np.nanpercentile(pooled_null, 95))
    grid = np.array([r["gridness"] for r in all_recs], float)
    n_grid = int(np.nansum(grid > thr))
    print(f"\nPooled {len(all_recs)} units from {len(SESSIONS)} sessions")
    print(f"Shuffle 95th-pctile threshold = {thr:.3f}")
    print(f"Grid cells (gridness > threshold): {n_grid} / {len(all_recs)} "
          f"({100 * n_grid / len(all_recs):.0f}%)")
    print(f"Median gridness (all): {np.nanmedian(grid):.2f}")

    out = dict(sessions=sessions, records=all_recs, pooled_null=pooled_null,
               threshold=thr, n_grid=n_grid, BINS=BINS, BOX=BOX,
               BIN_SIZE=BIN_SIZE)
    with open("results.pkl", "wb") as f:
        pickle.dump(out, f)
    print("\nSaved results.pkl")


if __name__ == "__main__":
    main()
