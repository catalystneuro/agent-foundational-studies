"""Step 3: theta phase precession within CA1 place fields (single session)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import precession_lib as pl

SESSION = "Achilles-10252013"
BIN_SIZE = 0.04  # m
MIN_PEAK_RATE = 1.0
MIN_SPATIAL_INFO = 0.5
MIN_FIELD_SPIKES = 50
EDGE_FRAC = 0.09  # require the field peak to be this far from the track ends
N_SHUFFLE = 500


def analyse_session(session, n_shuffle=N_SHUFFLE, verbose=True):
    """Run the full pipeline on one session; returns (results_df, context dict)."""
    f, nwbfile = pl.open_session(session)
    nwb = nap.NWBFile(nwbfile)
    exc = nwb["units"].getby_category("cell_type")["excitatory"]

    pos, fs_pos, track_len, maze_name = pl.load_position(f)
    vel, speed = pl.compute_speed(pos, fs_pos)
    runs = dict(zip(("right", "left"), pl.find_runs(pos, vel, fs_pos, track_len)))
    n_bins = int(round(track_len / BIN_SIZE))
    maze = pl.maze_epoch(f)

    chan, ratios = pl.pick_theta_channel(f, runs["right"])
    lfp = pl.read_lfp_channel(f, chan, maze.start[0], maze.end[0])
    filt, phase, amp = pl.theta_phase(lfp)
    if verbose:
        print(f"[{session}] {len(exc)} excitatory units, "
              f"{len(runs['right'])}+{len(runs['left'])} laps on {maze_name}, "
              f"track {track_len:.1f} m, theta channel {chan}")

    bins = np.linspace(0, track_len, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    tcs, occs = {}, {}
    for d, ep in runs.items():
        tc = nap.compute_tuning_curves(exc, pos, bins=n_bins, range=[(0, track_len)], epochs=ep,
                                       fs=fs_pos, return_pandas=True)
        tcs[d] = tc.apply(lambda c: gaussian_filter1d(c.values, 1.0, mode="nearest"))
        occ, _ = np.histogram(pos.restrict(ep).values, bins=bins)
        occs[d] = occ / fs_pos

    rng = np.random.default_rng(1)
    rows, per_field = [], {}
    todo = [(u, d) for d in runs for u in tcs[d].columns]
    for uid, d in tqdm(todo, desc=f"{session}: fields", disable=not verbose):
        v = tcs[d][uid].values
        if not np.isfinite(v).any() or np.nanmax(v) < MIN_PEAK_RATE:
            continue
        si = pl.spatial_info(v, occs[d])
        lo, hi, pk = pl.field_bounds(v, centers)
        width = hi - lo
        x, phi, t = pl.field_spike_phase_position(exc[uid], runs[d], pos, phase, lo, hi, d)
        row = dict(session=session, maze=maze_name, track_len=track_len, unit=uid, direction=d, peak_rate=float(np.nanmax(v)),
                   spatial_info=si, field_lo=lo, field_hi=hi, field_peak=pk,
                   field_width=width, n_field_spikes=len(x))
        row["is_place_cell"] = bool(
            si >= MIN_SPATIAL_INFO and 0.10 <= width <= 1.0
            and EDGE_FRAC * track_len <= pk <= (1 - EDGE_FRAC) * track_len
            and len(x) >= MIN_FIELD_SPIKES
        )
        if row["is_place_cell"]:
            a, phi0, R, _ = pl.circlin_regress(x, phi)
            rho = pl.circlin_corr(x, phi, a)
            p, null = pl.circlin_pvalue(x, phi, rho, n_shuffle=n_shuffle, rng=rng)
            row.update(slope_cycles=a, slope_deg=a * 360.0, phi0=phi0, rho=rho, pval=p)
            per_field[(uid, d)] = dict(x=x, phi=phi, tc=v, lo=lo, hi=hi, null=null, **row)
        rows.append(row)

    df = pd.DataFrame(rows)
    ctx = dict(pos=pos, vel=vel, runs=runs, track_len=track_len, lfp=lfp, filt=filt, phase=phase, exc=exc,
               tcs=tcs, centers=centers, per_field=per_field, chan=chan, maze=maze)
    return df, ctx


if __name__ == "__main__":
    df, ctx = analyse_session(SESSION)
    df.to_csv("precession_single_session.csv", index=False)
    pc = df[df.is_place_cell]
    sig = pc[pc.pval < 0.05]
    print(f"\n{len(pc)} place fields analysed; {len(sig)} with significant "
          f"circular-linear correlation (p < 0.05) = {100*len(sig)/len(pc):.0f}%")
    print(f"median slope {pc.slope_deg.median():.0f} deg/field; "
          f"{(pc.slope_cycles < 0).mean()*100:.0f}% negative")
    print(f"median rho {pc.rho.median():.3f}")
