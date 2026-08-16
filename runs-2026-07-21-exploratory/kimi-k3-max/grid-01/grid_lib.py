"""Shared library for grid cell analysis of DANDI 000582 (Sargolini et al. 2006)."""
import numpy as np

BIN_WIDTH = 2.5          # cm
SMOOTH_SIGMA = 2.0       # bins (=> 5 cm Gaussian)
SPEED_THRESH = 2.5       # cm/s
MIN_SPIKES = 200         # min spikes (after speed filter) for population stats
N_SHUFFLES = 100
SHUFFLE_MIN_SHIFT = 20.0  # s, minimum circular shift for shuffles


def load_session(asset_id, cache_dir="/tmp/remfile_cache_grid"):
    """Stream one NWB session from DANDI; return position track and units table."""
    import h5py
    import remfile
    from pynwb import NWBHDF5IO
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    disk_cache = remfile.DiskCache(cache_dir)
    h5 = h5py.File(remfile.File(url, disk_cache=disk_cache), "r")
    nwbfile = NWBHDF5IO(file=h5).read()
    pos = nwbfile.processing["behavior"]["Position"]["SpatialSeriesLED1"]
    t_pos = pos.timestamps[:]
    xy = pos.data[:].astype(float)  # values are cm despite 'meters' label
    good = ~np.isnan(xy).any(axis=1)
    t_pos, xy = t_pos[good], xy[good]
    units = nwbfile.units.to_dataframe()
    meta = dict(subject=nwbfile.subject.subject_id,
                session_id=nwbfile.session_id,
                identifier=str(nwbfile.identifier))
    return t_pos, xy, units, meta


def compute_speed(t_pos, xy, smooth_win_s=0.4):
    """Speed from forward differences, boxcar-smoothed, aligned to t_pos."""
    dt = np.median(np.diff(t_pos))
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t_pos)
    sp = np.concatenate([[sp[0]], sp])
    k = max(1, int(round(smooth_win_s / dt)))
    return np.convolve(sp, np.ones(k) / k, mode="same")


def session_occupancy(t_pos, xy, speed, bin_width=BIN_WIDTH):
    """Speed-filtered occupancy map; returns (occ, limits, arena_size, move_mask)."""
    import opexebo.analysis as opa
    move = speed > SPEED_THRESH
    t_mov, xy_mov = t_pos[move], xy[move]
    limits = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    arena_size = (limits[1] - limits[0], limits[3] - limits[2])
    occ, coverage, bin_edges = opa.spatial_occupancy(
        t_mov, xy_mov.T, arena_size, bin_width=bin_width, limits=limits)
    return occ, limits, arena_size, move, coverage


def rate_map_for_spikes(st, t_pos, xy, occ, limits, arena_size, bin_width=BIN_WIDTH):
    """Smoothed rate map for one unit's spike times."""
    import opexebo.analysis as opa
    import opexebo.general as opg
    sx = np.interp(st, t_pos, xy[:, 0])
    sy = np.interp(st, t_pos, xy[:, 1])
    spikes_tracking = np.array([st, sx, sy])
    rmap = opa.rate_map(occ, spikes_tracking, arena_size, bin_width=bin_width, limits=limits)
    return opg.smooth(rmap, sigma=SMOOTH_SIGMA)


def grid_stats_for_map(rmap, bin_width=BIN_WIDTH):
    """Autocorrelation, grid score and grid stats for a rate map."""
    import opexebo.analysis as opa
    acorr = opa.autocorrelation(rmap)
    gs, gstats = opa.grid_score(acorr, bin_width=bin_width)
    return acorr, gs, gstats


def spatial_information(rmap, occ):
    """Skaggs spatial information (bits/spike) from a rate map and occupancy (s)."""
    r = np.asarray(rmap, dtype=float)
    o = np.asarray(occ, dtype=float)
    valid = ~np.ma.getmaskarray(rmap) & np.isfinite(r) & (o > 0)
    r, o = r[valid], o[valid]
    p = o / o.sum()
    mean_rate = np.sum(p * r)
    if mean_rate <= 0:
        return np.nan, np.nan
    nz = r > 0
    si = np.sum(p[nz] * (r[nz] / mean_rate) * np.log2(r[nz] / mean_rate))
    return si, mean_rate


def circular_shift_shuffle(st, t_end, rng):
    """Circularly shift spike times by a random offset >= SHUFFLE_MIN_SHIFT."""
    shift = rng.uniform(SHUFFLE_MIN_SHIFT, t_end - SHUFFLE_MIN_SHIFT)
    return np.mod(st + shift, t_end)


def analyze_session(row, n_shuffles=N_SHUFFLES, shuffle_gs_thresh=0.3, seed0=0):
    """Full per-session pipeline. Returns list of per-unit records (picklable)."""
    import warnings
    warnings.filterwarnings("ignore")
    t_pos, xy, units, meta = load_session(row["asset_id"],
                                          cache_dir=f"/tmp/remfile_cache_grid_{row['asset_id'][:8]}")
    speed = compute_speed(t_pos, xy)
    occ, limits, arena_size, move, coverage = session_occupancy(t_pos, xy, speed)
    t_end = t_pos[-1]
    dur_h = (t_pos[-1] - t_pos[0]) / 3600.0

    records = []
    for uid, u in units.iterrows():
        st_all = np.asarray(u["spike_times"], dtype=float)
        sp_speed = np.interp(st_all, t_pos, speed)
        st = st_all[sp_speed > SPEED_THRESH]
        rec = dict(session_path=row["path"], asset_id=row["asset_id"],
                   subject=meta["subject"], session_id=meta["session_id"],
                   unit_id=int(uid), unit_name=str(u["unit_name"]),
                   layer=str(u["histology"]), depth=float(u["depth"]),
                   n_spikes_total=int(len(st_all)), n_spikes=int(len(st)),
                   duration_s=float(t_end), coverage=float(coverage),
                   arena_size=tuple(float(a) for a in arena_size))
        if len(st) < 20:
            rec.update(grid_score=np.nan, si=np.nan, mean_rate=np.nan)
            records.append(rec)
            continue
        rmap = rate_map_for_spikes(st, t_pos, xy, occ, limits, arena_size)
        acorr, gs, gstats = grid_stats_for_map(rmap)
        si, mean_rate = spatial_information(rmap, occ)
        rec.update(grid_score=float(gs), si=float(si), mean_rate=float(mean_rate),
                   grid_spacing=float(gstats.get("grid_spacing", np.nan)),
                   grid_orientation=float(gstats.get("grid_orientation", np.nan)),
                   rmap=np.asarray(rmap, dtype=np.float32),
                   rmap_mask=np.ma.getmaskarray(rmap),
                   acorr=np.asarray(acorr, dtype=np.float32))
        # shuffle test for candidate grid cells
        if np.isfinite(gs) and gs > shuffle_gs_thresh and len(st) >= MIN_SPIKES and n_shuffles > 0:
            import zlib
            det_seed = zlib.crc32(f"{row['asset_id']}|{int(uid)}".encode()) % 2**31
            rng = np.random.default_rng(seed0 + det_seed)
            sh_gs = np.empty(n_shuffles)
            for s in range(n_shuffles):
                st_sh = circular_shift_shuffle(st, t_end, rng)
                rmap_sh = rate_map_for_spikes(st_sh, t_pos, xy, occ, limits, arena_size)
                _, gs_sh, _ = grid_stats_for_map(rmap_sh)
                sh_gs[s] = gs_sh
            rec["shuffle_gs"] = sh_gs.astype(np.float32)
            rec["grid_p"] = float((np.sum(sh_gs >= gs) + 1) / (n_shuffles + 1))
            rec["shuffle_gs_p95"] = float(np.percentile(sh_gs, 95))
        records.append(rec)
    return records
