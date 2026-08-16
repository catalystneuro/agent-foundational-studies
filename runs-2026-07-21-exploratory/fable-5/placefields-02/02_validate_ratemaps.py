"""Cross-check the fast rate-map implementation against pynapple's tuning curves.

The shuffle test needs thousands of rate maps, so we use a compressed-time
implementation that is ~50x faster than calling pynapple per shuffle. This
script confirms that the fast implementation returns the same rate maps, and
the same spatial-information values, as `nap.compute_tuning_curves`.
"""
import numpy as np
import pynapple as nap
import pf_lib

nwbfile = pf_lib.open_session(pf_lib.PROTOTYPE)
maze_dur, track_dur, _ = pf_lib.check_timebase(nwbfile)
print(f"timebase check OK: MazeEpoch {maze_dur:.1f} s, position series {track_dur:.1f} s")

pos2d, lin, fs, maze = pf_lib.load_behavior(nwbfile)
epochs = pf_lib.load_epochs(nwbfile)
lin = lin.restrict(epochs["MazeEpoch"])
lin_valid, vel, run_eps = pf_lib.make_run_epochs(lin, fs)
units = pf_lib.load_units(nwbfile)
exc = units[np.where(units.cell_type == "excitatory")[0]]
rng_track = pf_lib.track_range(lin_valid)
print(f"maze={maze}  fs={fs:.3f} Hz  track range={rng_track[0]:.3f}-{rng_track[1]:.3f} m")
print(f"units={len(units)}  excitatory={len(exc)}")

for direction in ("rightward", "leftward"):
    ep = run_eps[direction]
    # smoothing off, so the comparison is like for like
    rm = pf_lib.DirectionalRateMaps(lin_valid, ep, fs, n_bins=40,
                                    track_range=rng_track, smooth_bins=0.0)
    mine = rm.rate_maps(exc)
    tc = nap.compute_tuning_curves(exc, lin_valid, bins=40, range=rng_track,
                                   epochs=ep, feature_names=["position"])
    theirs = np.asarray(tc.values)

    ok = rm.valid_bins
    a, b = mine[:, ok], theirs[:, ok]
    f = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a - b)[f]
    print(f"\n[{direction}] {len(ep)} laps, {ep.tot_length():.0f} s")
    print(f"  rate map: median|diff|={np.median(d):.2e} Hz  "
          f"mean|diff|={d.mean():.2e} Hz  max|diff|={d.max():.3f} Hz")
    print(f"  mean rate mine={a[f].mean():.4f} Hz  theirs={b[f].mean():.4f} Hz  "
          f"corr={np.corrcoef(a[f], b[f])[0, 1]:.6f}")

    # pynapple's `occupancy` attribute is in samples, not seconds
    occ_p = np.asarray(tc.attrs["occupancy"])[ok] / fs
    print(f"  occupancy: mine={rm.occupancy[ok].sum():.1f} s  "
          f"pynapple={occ_p.sum():.1f} s  epoch={ep.tot_length():.1f} s")

    si_mine = rm.si(mine)[0]
    si_theirs = pf_lib.spatial_information(np.nan_to_num(b), occ_p)[0]
    print(f"  spatial info (bits/spike): median mine={np.median(si_mine):.4f} "
          f"theirs={np.median(si_theirs):.4f}  "
          f"max|diff|={np.nanmax(np.abs(si_mine - si_theirs)):.4f}  "
          f"corr={np.corrcoef(si_mine, si_theirs)[0, 1]:.6f}")

# odd/even lap split should return two maps of the same shape
ep = run_eps["rightward"]
rm = pf_lib.DirectionalRateMaps(lin_valid, ep, fs, n_bins=40,
                                track_range=rng_track, smooth_bins=1.0)
odd, even = rm.lap_split_maps(exc)
print(f"\nlap split: odd {odd.shape}, even {even.shape}")
pr, pp, w = pf_lib.field_metrics(rm.rate_maps(exc), rm.centers, rm.bin_size)
print(f"field metrics: peak rate median {np.nanmedian(pr):.2f} Hz, "
      f"width median {np.nanmedian(w):.3f} m")
