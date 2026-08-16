"""Core single-session analysis: place fields, theta phase locking, phase precession."""
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
import theta_lib as tl

NB_BINS = 50
TRACK_LEN = 1.6
TC_SMOOTH_BINS = 1.5
N_SHUFFLE = 200
MIN_OCCUPANCY = 0.5  # seconds per spatial bin
RNG = np.random.default_rng(0)


def analyze_session(subject, label, asset_id, verbose=True):
    nwbfile = tl.open_session(asset_id)
    maze = tl.get_maze_epoch(nwbfile)
    pos, pos_fs = tl.get_position(nwbfile)
    runs, direction = tl.get_run_epochs(pos, pos_fs)
    units = tl.get_units(nwbfile)

    # ---- pick the LFP channel with the strongest theta/delta ratio during running
    probe = nap.IntervalSet(start=runs.start[0], end=min(runs.start[0] + 60, maze.end[-1]))
    x = tl.read_lfp_channels(nwbfile, np.arange(128), probe).values
    theta_pow = tl.bandpower(x, *tl.THETA_BAND, tl.LFP_FS)
    ratio = theta_pow / tl.bandpower(x, *tl.DELTA_BAND, tl.LFP_FS)
    best_ch = int(np.argmax(ratio))

    # ---- theta phase across the whole maze epoch
    lfp = tl.read_lfp_channels(nwbfile, [best_ch], maze)[:, 0]
    filt, phase = tl.theta_phase(lfp)

    run_ep = {+1: runs[direction > 0], -1: runs[direction < 0]}
    all_runs = runs

    # ---- place fields, per running direction
    bins = np.linspace(0, TRACK_LEN, NB_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    dt_pos = 1.0 / pos_fs
    tcs, occ = {}, {}
    for d in (+1, -1):
        tcs[d] = nap.compute_1d_tuning_curves(units, pos, nb_bins=NB_BINS,
                                              minmax=(0, TRACK_LEN), ep=run_ep[d])
        occ[d] = np.histogram(pos.restrict(run_ep[d]).values, bins=bins)[0] * dt_pos

    # ---- LFP theta peak frequency while the animal is running
    from scipy.signal import welch
    fr, P = welch(np.concatenate([lfp.restrict(all_runs[i:i + 1]).values
                                  for i in range(len(all_runs))]),
                  fs=tl.LFP_FS, nperseg=int(4 * tl.LFP_FS))
    band = (fr >= 5) & (fr <= 12)
    lfp_theta_freq = fr[band][np.argmax(P[band])]

    # ---- spike theta phases during running
    spk_phase = units.restrict(all_runs).value_from(phase)
    spk_pos = units.restrict(all_runs).value_from(pos)

    rows, precession, examples = [], [], {}
    for u in units.index:
        ph_all = np.asarray(spk_phase[u].values)
        mrl, mu, p_ray = tl.rayleigh(ph_all)
        rate_track = len(ph_all) / all_runs.tot_length()
        row = dict(session=label, subject=subject, unit=int(u),
                   cell_type=units.metadata["cell_type"][u],
                   location=units.metadata["location"][u],
                   n_spikes_run=len(ph_all), rate_track=rate_track,
                   mrl=mrl, pref_phase=mu, p_rayleigh=p_ray, lfp_channel=best_ch)

        # place-field / precession analysis per direction
        best = None
        for d in (+1, -1):
            tc = np.nan_to_num(tcs[d][u].values)
            tc = gaussian_filter1d(tc, TC_SMOOTH_BINS, mode="nearest")
            if np.nanmax(tc) < 1.0:
                continue
            si = tl.spatial_information(tc, occ[d])
            # the animal barely dwells in the outermost bins, so rates there are
            # poorly estimated; exclude them as candidate field peaks
            ok_occ = occ[d] >= MIN_OCCUPANCY
            if not ok_occ.any():
                continue
            pk = int(np.argmax(np.where(ok_occ, tc, -np.inf)))
            thr = 0.2 * tc[pk]
            lo = pk
            while lo > 0 and tc[lo - 1] >= thr:
                lo -= 1
            hi = pk
            while hi < len(tc) - 1 and tc[hi + 1] >= thr:
                hi += 1
            f0, f1 = bins[lo], bins[hi + 1]
            width = f1 - f0
            info = dict(direction=d, si=si, peak_rate=tc[pk], peak_pos=centers[pk],
                        field=(f0, f1), width=width, tc=tc)
            if best is None or info["si"] * info["peak_rate"] > best["si"] * best["peak_rate"]:
                best = info
        if best is None:
            rows.append(row)
            continue

        d = best["direction"]
        row.update(direction=d, spatial_info=best["si"], peak_rate=best["peak_rate"],
                   peak_pos=best["peak_pos"], field_width=best["width"])

        # spikes inside the field on runs of the preferred direction
        pu = np.asarray(spk_pos[u].restrict(run_ep[d]).values)
        phu = np.asarray(spk_phase[u].restrict(run_ep[d]).values)
        f0, f1 = best["field"]
        m = (pu >= f0) & (pu <= f1) & np.isfinite(pu)
        xin = (pu[m] - f0) / (f1 - f0)          # normalized position in field, 0..1
        if d < 0:
            xin = 1 - xin                        # orient along direction of travel
        phin = phu[m]
        slope, phi0, rho, p_prec = tl.circ_lin_regression(xin, phin)

        # shuffle control: break the position-phase pairing, keep both marginals
        if m.sum() >= 20:
            shuf_rho = tl.circ_lin_shuffle(xin, phin, N_SHUFFLE, RNG)
            p_shuf = (np.sum(shuf_rho >= abs(rho)) + 1) / (N_SHUFFLE + 1)
        else:
            shuf_rho = np.full(N_SHUFFLE, np.nan)
            p_shuf = np.nan
        row.update(n_field_spikes=int(m.sum()), prec_slope=slope, prec_phi0=phi0,
                   prec_rho=rho, prec_p=p_prec, prec_p_shuffle=p_shuf,
                   shuf_rho95=np.nanpercentile(shuf_rho, 95) if m.sum() >= 20 else np.nan)
        rows.append(row)
        precession.append(dict(unit=int(u), x=xin, phase=phin, slope=slope, phi0=phi0,
                               rho=rho, p=p_prec, direction=d, field=(f0, f1)))

    df = pd.DataFrame(rows)
    out = dict(df=df, precession=precession, tcs=tcs, occ=occ, centers=centers,
               units=units, runs=runs, direction=direction, run_ep=run_ep, pos=pos,
               phase=phase, filt=filt, lfp=lfp, best_ch=best_ch, label=label,
               subject=subject, maze=maze, spk_phase=spk_phase, spk_pos=spk_pos,
               lfp_theta_freq=lfp_theta_freq, psd=(fr, P),
               channel_ratio=ratio, channel_theta_pow=theta_pow)
    if verbose:
        print(f"{label}: {len(units)} units, {len(runs)} runs, ch {best_ch}, "
              f"LFP theta {lfp_theta_freq:.2f} Hz, "
              f"{(df.p_rayleigh < 0.05).sum()} phase-locked")
    return out


def place_cell_mask(df, si_min=0.5, peak_min=1.0, nspk_min=50):
    return ((df.cell_type == "excitatory") & (df.spatial_info >= si_min)
            & (df.peak_rate >= peak_min) & (df.n_field_spikes >= nspk_min)
            & (df.rate_track < 8.0))
