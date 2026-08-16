"""Per-session pipeline: laps -> place fields -> theta phase -> entrainment /
precession.  Results are cached to an .npz so figure scripts can be re-run
cheaply."""

import os
import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm
from scipy.signal import welch

from hc11_io import (
    open_session, get_maze_epoch, get_position, get_units, SESSIONS,
    maze_type, is_linear_track,
)
from theta_analysis import (
    FS_LFP,
    THETA_BAND,
    lap_intervals,
    running_speed,
    band_power_ratio,
    theta_phase,
    rayleigh,
    circ_lin_fit,
    circ_lin_shuffle_p,
    smooth_tc,
    spatial_information,
    find_field,
)

N_POS_BINS = 40
MIN_SPEED = 5.0  # cm/s
MIN_FIELD_PEAK = 1.0  # Hz
MIN_SPATIAL_INFO = 0.3  # bits/spike
MIN_SPIKES_IN_FIELD = 40
N_SHUFFLE = 1000


def select_theta_channel(h5file, maze, laps, n_channels=128, n_chunks=4, cache=None):
    """Rank LFP channels by theta/delta ratio during track running.

    The LFP dataset is chunked one channel at a time (170221 samples per
    chunk, about 136 s), so the scan reads a few whole chunks per channel
    rather than the whole maze epoch: that is enough running data to rank
    channels and keeps the streamed volume to a few hundred megabytes.
    """
    if cache is not None and os.path.exists(cache):
        out = np.load(cache)
        return int(out[np.argmax(out[:, 2]), 0]), out

    dset = h5file["/processing/ecephys/LFP/LFP/data"]
    csize = dset.chunks[0]
    i0, i1 = int(maze.start[0] * FS_LFP), int(maze.end[0] * FS_LFP)

    # mask of running samples over the whole recording, then per-chunk coverage
    run = np.zeros(dset.shape[0], bool)
    for s, e in zip(laps.start, laps.end):
        run[int(s * FS_LFP) : int(e * FS_LFP)] = True
    cand = np.arange(i0 // csize, i1 // csize + 1)
    cover = np.array([run[c * csize : (c + 1) * csize].sum() for c in cand])
    cand = cand[cover > 10 * FS_LFP]
    if cand.size == 0:  # fall back to the chunks that overlap the epoch at all
        cand = np.arange(i0 // csize, i1 // csize + 1)
    take = cand[np.linspace(0, cand.size - 1, min(n_chunks, cand.size)).astype(int)]
    slices = [(int(c * csize), int(min((c + 1) * csize, dset.shape[0]))) for c in take]
    n_run = sum(run[a:b].sum() for a, b in slices)
    print(f"  scanning {len(slices)} LFP chunks, {n_run / FS_LFP:.0f} s of running")

    out = []
    for ch in tqdm(range(n_channels), desc="  theta channel scan"):
        seg = np.concatenate(
            [np.asarray(dset[a:b, ch], float)[run[a:b]] for a, b in slices]
        )
        th, ratio = band_power_ratio(seg)
        out.append((ch, th, ratio))
    out = np.array(out)
    if cache is not None:
        np.save(cache, out)
    best = int(out[np.argmax(out[:, 2]), 0])
    return best, out


def analyze_session(asset_id, outdir="results", n_channels=128):
    label = SESSIONS[asset_id]
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== {label} ===")
    h5, nwbfile = open_session(asset_id)

    if not is_linear_track(nwbfile):
        print(f"  skipping: {maze_type(nwbfile)} is not a linear track")
        return None
    maze = get_maze_epoch(nwbfile)
    pos = get_position(nwbfile).restrict(maze)
    track_len = float(np.ceil(np.nanmax(pos.d) / 10.0) * 10.0)
    units = get_units(nwbfile)
    laps, lap_dir = lap_intervals(
        pos, min_duration=0.5, min_extent=0.6 * track_len
    )
    posf = pos[np.isfinite(pos.d)]
    speed = running_speed(posf, laps)
    print(f"  {len(units)} units, {len(laps)} laps, {laps.tot_length():.0f} s running")

    # keep only samples during laps where the animal is actually moving
    run_ep = laps.intersect(speed.threshold(MIN_SPEED).time_support)
    ep_dir = {
        +1: laps[lap_dir == 1].intersect(run_ep),
        -1: laps[lap_dir == -1].intersect(run_ep),
    }

    # ---- theta channel and phase -----------------------------------------
    best_ch, chan_scan = select_theta_channel(
        h5, maze, laps, n_channels, cache=os.path.join(outdir, f"{label}_chanscan.npy")
    )
    print(f"  best theta channel {best_ch} (theta/delta {chan_scan[:,2].max():.1f})")
    es = nwbfile.processing["ecephys"]["LFP"].electrical_series["LFP"]
    dset = h5["/processing/ecephys/LFP/LFP/data"]
    conv = float(es.conversion)  # int16 -> volts
    i0, i1 = int(maze.start[0] * FS_LFP), int(maze.end[0] * FS_LFP)
    raw = np.asarray(dset[i0:i1, best_ch], float) * conv * 1e3  # mV
    lfp = nap.Tsd(t=np.arange(i0, i1) / FS_LFP, d=raw)
    theta, phase = theta_phase(lfp)

    # ---- tuning curves per direction --------------------------------------
    bins = np.linspace(0, track_len, N_POS_BINS + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    tcs, occs = {}, {}
    for dsign, ep in ep_dir.items():
        tc = nap.compute_1d_tuning_curves(
            units, posf, nb_bins=N_POS_BINS, ep=ep, minmax=(0, track_len)
        )
        tcs[dsign] = smooth_tc(tc, sigma_bins=1.5)
        p = posf.restrict(ep)
        occs[dsign] = np.histogram(p.d, bins)[0] * np.median(np.diff(posf.t))

    # ---- per unit / direction stats ---------------------------------------
    rows = []
    examples = {}
    rng = np.random.default_rng(1)
    meta = units.metadata
    ph_t, ph_d = np.asarray(phase.t), np.asarray(phase.d)
    for dsign, ep in ep_dir.items():
        tc = tcs[dsign]
        occ = occs[dsign]
        for uid in tqdm(list(units.keys()), desc=f"  units dir {dsign:+d}"):
            if meta.loc[uid, "cell_type"] != "excitatory":
                continue
            rate = tc[uid].values
            si = spatial_information(rate, occ)
            fld = find_field(rate, centers, peak_frac=0.25, min_peak=MIN_FIELD_PEAK)
            if fld is None or not np.isfinite(si) or si < MIN_SPATIAL_INFO:
                continue
            f0, f1, fpk, pk = fld

            spk = units[uid].restrict(ep)
            if len(spk) == 0:
                continue
            spk_pos = spk.value_from(posf)
            spk_ph = spk.value_from(phase)
            inside = (spk_pos.d >= f0) & (spk_pos.d <= f1)
            n_in = int(inside.sum())

            # entrainment uses all in-field spikes of this cell/direction
            ph_rad = np.deg2rad(spk_ph.d[inside])
            mu, R, p_ray, n = rayleigh(ph_rad)

            # jitter control: shifting spikes by up to +/- 400 ms destroys any
            # theta-cycle relationship while preserving rate and field position
            t_in = np.asarray(spk.t)[inside]
            r_jit = []
            for _ in range(20):
                tj = t_in + rng.uniform(-0.4, 0.4, t_in.size)
                j = np.clip(np.searchsorted(ph_t, tj), 0, ph_t.size - 1)
                r_jit.append(rayleigh(np.deg2rad(ph_d[j]))[1])
            mrl_jitter = float(np.mean(r_jit))

            row = dict(
                session=label,
                maze=maze_type(nwbfile),
                unit=uid,
                direction=dsign,
                location=meta.loc[uid, "location"],
                peak_rate=pk,
                spatial_info=si,
                field_start=f0,
                field_stop=f1,
                field_peak=fpk,
                field_width=f1 - f0,
                n_spikes_field=n_in,
                pref_phase=np.rad2deg(mu) % 360,
                mrl=R,
                mrl_jitter=mrl_jitter,
                p_rayleigh=p_ray,
            )

            if n_in >= MIN_SPIKES_IN_FIELD:
                xnorm = (spk_pos.d[inside] - f0) / max(f1 - f0, 1e-9)
                if dsign < 0:  # travel is in the -x direction; flip to travel order
                    xnorm = 1.0 - xnorm
                fit = circ_lin_shuffle_p(
                    xnorm, ph_rad, n_shuffle=N_SHUFFLE, rng=rng
                )
                if fit is not None:
                    row.update(
                        slope=fit["slope"],
                        phi0=fit["phi0"],
                        rho=fit["rho"],
                        p_circlin=fit["p"],
                        p_shuffle=fit["p_shuffle"],
                    )
                    examples[(uid, dsign)] = dict(
                        x=xnorm, phase=np.rad2deg(ph_rad), fit=fit,
                        pos=spk_pos.d[inside], field=(f0, f1),
                    )
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  {len(df)} place fields, {df['slope'].notna().sum()} with enough spikes")

    # ---- spectra for the overview figure ----------------------------------
    run_mask = np.zeros(len(lfp), bool)
    for s, e in zip(run_ep.start, run_ep.end):
        run_mask[int((s - maze.start[0]) * FS_LFP) : int((e - maze.start[0]) * FS_LFP)] = True
    fr, psd_run = welch(lfp.d[run_mask], fs=FS_LFP, nperseg=int(4 * FS_LFP))
    fr, psd_still = welch(lfp.d[~run_mask], fs=FS_LFP, nperseg=int(4 * FS_LFP))

    np.savez_compressed(
        os.path.join(outdir, f"{label}.npz"),
        stats=df.to_records(index=False),
        tc_fwd=tcs[+1].values,
        tc_bwd=tcs[-1].values,
        tc_units=np.array(list(tcs[+1].columns)),
        centers=centers,
        occ_fwd=occs[+1],
        occ_bwd=occs[-1],
        best_ch=best_ch,
        chan_scan=chan_scan,
        freqs=fr,
        psd_run=psd_run,
        psd_still=psd_still,
        lap_start=laps.start,
        lap_end=laps.end,
        lap_dir=lap_dir,
        maze_epoch=np.array([maze.start[0], maze.end[0]]),
        track_len=track_len,
        run_time=laps.tot_length(),
        n_units=len(units),
    )
    df.to_csv(os.path.join(outdir, f"{label}_fields.csv"), index=False)
    return dict(
        label=label, df=df, tcs=tcs, occs=occs, centers=centers, units=units,
        pos=posf, laps=laps, lap_dir=lap_dir, ep_dir=ep_dir, lfp=lfp,
        theta=theta, phase=phase, examples=examples, best_ch=best_ch, chan_scan=chan_scan,
        freqs=fr, psd_run=psd_run, psd_still=psd_still, maze=maze, speed=speed,
        track_len=track_len,
    )


if __name__ == "__main__":
    import sys

    aid = sys.argv[1] if len(sys.argv) > 1 else "5349c68b-c0a7-46c0-9900-cda050722fa4"
    analyze_session(aid)
