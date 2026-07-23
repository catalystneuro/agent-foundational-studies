"""Full theta entrainment + phase precession analysis, run over two sessions.

Saves a pickle of per-cell results plus the spike-level phase/position table used
for the population figures.
"""
import pickle
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.signal import welch
from tqdm import tqdm

import theta_lib as T
import theta_analysis as A

SESSIONS = list(T.ASSETS)
BIN_SIZE = 0.04            # m, spatial bin for rate maps
MIN_SPIKES_FIELD = 50
MIN_PEAK_RATE = 1.0
MIN_SI = 0.4
MIN_FIELD_W, MAX_FIELD_W = 0.12, 0.90
N_PERM = 500


def pick_theta_channel(nwbfile, maze, n_probe=128):
    """Rank channels by theta/delta power over a 300 s slice of the maze epoch."""
    es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
    fs = es.rate
    t0 = maze.start[0] + 120.0
    i0, i1 = int(t0 * fs), int((t0 + 300.0) * fs)
    ratios = np.zeros(n_probe)
    for ch in tqdm(range(n_probe), desc="theta/delta scan", leave=False):
        x = es.data[i0:i1, ch].astype(np.float64)
        f, p = welch(x, fs=fs, nperseg=int(4 * fs))
        ratios[ch] = p[(f >= 6) & (f <= 10)].mean() / p[(f >= 2) & (f <= 4)].mean()
    return int(np.argmax(ratios)), ratios


def _theta_freq(phase, ep):
    """Mean theta frequency (Hz) from the rate of phase advance during running."""
    ph = phase.restrict(ep)
    d = np.mod(np.diff(ph.values) + np.pi, 2 * np.pi) - np.pi
    dt_ = np.diff(ph.t)
    ok = dt_ < 0.01
    return float(np.mean(d[ok]) / np.mean(dt_[ok]) / (2 * np.pi))


def analyze_session(session):
    print(f"\n===== {session} =====")
    nwbfile = T.open_session(session)
    maze = T.get_epochs(nwbfile)["MazeEpoch"]
    ch, ratios = pick_theta_channel(nwbfile, maze)
    print(f"theta reference channel: {ch} (theta/delta = {ratios[ch]:.2f})")

    lfp, fs = T.get_lfp(nwbfile, ch, maze)
    phase, amp, filt = T.theta_phase_amp(lfp, fs)

    position, dt, track_len = T.get_position(nwbfile)
    track = (0.0, track_len)
    n_bins = int(round(track_len / BIN_SIZE))
    print(f"track length {track_len:.1f} m -> {n_bins} bins")
    runs, direction = T.run_epochs(position, dt, track_len)
    speed = T.speed_tsd(position, dt)
    print(f"{len(runs)} laps ({np.sum(direction > 0)} R / {np.sum(direction < 0)} L), "
          f"{runs.tot_length():.0f} s of running")

    # moving-only epochs, split by running direction
    moving = speed.threshold(T.SPEED_THRESH).time_support
    dir_eps = {}
    for lbl, sgn in [("R", 1), ("L", -1)]:
        idx = np.where(direction == sgn)[0]
        e = nap.IntervalSet(start=runs.start[idx], end=runs.end[idx])
        dir_eps[lbl] = e.intersect(moving).drop_short_intervals(0.2)
    all_run = dir_eps["R"].union(dir_eps["L"])
    print(f"moving-only: R {dir_eps['R'].tot_length():.0f} s, "
          f"L {dir_eps['L'].tot_length():.0f} s")

    units = T.get_units(nwbfile).restrict(maze)

    # mean band-passed LFP as a function of theta phase (defines the convention)
    ph_run = phase.restrict(all_run)
    fl_run = filt.restrict(all_run)
    wave_edges = np.linspace(0, 2 * np.pi, 73)
    wbin = np.digitize(ph_run.values, wave_edges) - 1
    mean_wave = np.array([fl_run.values[wbin == b].mean() for b in range(72)])
    wave_centers = (wave_edges[:-1] + wave_edges[1:]) / 2

    # ---------------- theta phase entrainment (all units, all running) --------
    PHASE_BINS = 72
    phase_edges = np.linspace(0, 2 * np.pi, PHASE_BINS + 1)
    entrain, phase_hists = [], {}
    for uid in units.index:
        st = units[uid].restrict(all_run)
        if len(st) < 30:
            continue
        ph = st.value_from(phase).values
        ph = ph[np.isfinite(ph)]
        phase_hists[uid] = np.histogram(ph, bins=phase_edges)[0]
        p, z, r = A.rayleigh_test(ph)
        entrain.append(dict(session=session, unit=uid,
                            cell_type=units.cell_type[uid],
                            location=units.location[uid],
                            n_spikes=len(ph), mrl=r, rayleigh_z=z, rayleigh_p=p,
                            pref_phase=A.circ_mean(ph),
                            rate=len(st) / all_run.tot_length()))
    entrain = pd.DataFrame(entrain)
    print(f"entrainment: {len(entrain)} units, "
          f"{np.sum(entrain.rayleigh_p < 0.01)} significant at p<0.01")

    # ---------------- place fields + phase precession -------------------------
    cells, spike_rows, rate_maps = [], [], {}
    for lbl in ["R", "L"]:
        ep = dir_eps[lbl]
        tc = A.tuning_curves(units, position, ep, n_bins, track, smooth_bins=1.0)
        centers = tc.index.values.astype(float)
        occ, _ = A.occupancy(position, ep, n_bins, track, dt)
        rate_maps[lbl] = tc

        for uid in units.index:
            if units.cell_type[uid] != "excitatory":
                continue
            rm = tc[uid].values
            fld = A.find_field(rm, centers, frac=0.2)
            if fld is None:
                continue
            f0, f1, pk_pos, pk_rate = fld
            width = f1 - f0
            si = A.spatial_information(rm, occ)
            st = units[uid].restrict(ep)
            pos_at_spike = st.value_from(position)
            m = (pos_at_spike.values >= f0) & (pos_at_spike.values <= f1)
            n_in = int(m.sum())

            rec = dict(session=session, unit=uid, direction=lbl,
                       peak_rate=pk_rate, peak_pos=pk_pos, field_start=f0,
                       field_end=f1, field_width=width, spatial_info=si,
                       n_spikes_field=n_in, is_place_cell=False,
                       slope=np.nan, phase0=np.nan, rho=np.nan, perm_p=np.nan)

            ok = (pk_rate >= MIN_PEAK_RATE and si >= MIN_SI
                  and MIN_FIELD_W <= width <= MAX_FIELD_W
                  and n_in >= MIN_SPIKES_FIELD)
            if ok:
                rec["is_place_cell"] = True
                xs = pos_at_spike.values[m]
                ts = pos_at_spike.t[m]
                phs = nap.Ts(t=ts).value_from(phase).values
                good = np.isfinite(phs) & np.isfinite(xs)
                xs, ts, phs = xs[good], ts[good], phs[good]
                # normalized position within the field, oriented along travel
                xn = (xs - f0) / width
                if lbl == "L":
                    xn = 1.0 - xn
                a, phi0, rho = A.circlin_fit(xn, phs)
                pp, obs, _ = A.circlin_permutation_p(xn, phs, n_perm=N_PERM, rng=uid)
                rec.update(slope=a, phase0=phi0, rho=rho, perm_p=pp)
                spike_rows.append(pd.DataFrame(
                    dict(session=session, unit=uid, direction=lbl,
                         t=ts, x_norm=xn, phase=phs)))
            cells.append(rec)

    cells = pd.DataFrame(cells)
    spikes_df = (pd.concat(spike_rows, ignore_index=True)
                 if spike_rows else pd.DataFrame())
    pc = cells[cells.is_place_cell]
    print(f"place fields: {len(pc)} cell-direction pairs pass criteria "
          f"(from {cells.unit.nunique()} excitatory units)")
    if len(pc):
        print(f"  precession significant (perm p<0.05): "
              f"{int((pc.perm_p < 0.05).sum())} / {len(pc)}; "
              f"negative slope: {int((pc.slope < 0).sum())}")

    return dict(session=session, channel=ch, ratios=ratios, entrain=entrain,
                cells=cells, spikes=spikes_df, n_laps=len(runs),
                run_time=all_run.tot_length(), fs=fs, track_len=track_len,
                n_units=len(units), rate_maps=rate_maps,
                phase_hists=phase_hists, phase_edges=phase_edges,
                dir_eps={k: np.c_[v.start, v.end] for k, v in dir_eps.items()},
                mean_theta_freq=_theta_freq(phase, all_run),
                mean_wave=mean_wave, wave_centers=wave_centers)


if __name__ == "__main__":
    results = [analyze_session(s) for s in SESSIONS]
    with open("results.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print("\nwrote results.pkl")
