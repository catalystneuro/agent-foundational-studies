"""Place fields, theta phase locking and theta phase precession for one session.

Writes a pickle of per-unit results that 04_visualize.py turns into figures.
"""
import pickle
import numpy as np
import pynapple as nap
from tqdm import tqdm

import hc11

SESSION = "Achilles_10252013"


def spatial_info(tc, occ):
    """Skaggs spatial information in bits/spike."""
    p = occ / occ.sum()
    r = np.asarray(tc, float)
    rbar = np.nansum(p * r)
    ok = (r > 0) & (p > 0)
    if rbar <= 0:
        return np.nan
    return float(np.sum(p[ok] * (r[ok] / rbar) * np.log2(r[ok] / rbar)))


def find_field(tc, bins, frac=0.2, min_w=0.15, max_w=1.0):
    """Contiguous field around the peak, thresholded at `frac` of peak rate."""
    r = np.asarray(tc, float)
    pk = int(np.nanargmax(r))
    thr = frac * r[pk]
    lo = pk
    while lo > 0 and r[lo - 1] >= thr:
        lo -= 1
    hi = pk
    while hi < len(r) - 1 and r[hi + 1] >= thr:
        hi += 1
    width = bins[hi] - bins[lo]
    if not (min_w <= width <= max_w):
        return None
    return dict(lo=bins[lo], hi=bins[hi], peak_pos=bins[pk], peak_rate=r[pk],
                width=width)


def analyze_session(session, n_bins=hc11.N_POS_BINS, verbose=True):
    nwbfile, io = hc11.open_session(session)
    maze = hc11.maze_epoch(nwbfile)
    pos_all = hc11.load_position(nwbfile)
    tracked = maze.intersect(pos_all.time_support)
    pos = pos_all.restrict(tracked)
    vel = hc11.compute_velocity(pos)
    speed, run, run_r, run_l = hc11.run_epochs(vel)
    units = hc11.load_units(nwbfile)

    # --- LFP channel with the strongest theta/delta ratio during the maze ----
    from scipy.signal import welch
    el = nwbfile.electrodes.to_dataframe()
    cand = [int(sub.index.to_numpy()[len(sub) // 2])
            for _, sub in el.groupby("group_name", sort=False)]
    best, best_score = None, -np.inf
    for ch in tqdm(cand, desc=f"{session}: LFP channels", disable=not verbose):
        lfp_c = hc11.load_lfp_channel(nwbfile, ch, maze)
        fr, pxx = welch(lfp_c.values, fs=hc11.LFP_FS, nperseg=int(4 * hc11.LFP_FS))
        score = (pxx[(fr >= 6) & (fr <= 10)].mean()
                 / pxx[(fr >= 1) & (fr <= 4)].mean())
        if score > best_score:
            best, best_score, lfp = ch, score, lfp_c
    filt, phase = hc11.theta_phase(lfp)

    # --- place fields per running direction ---------------------------------
    edges = np.linspace(0, float(pos.max()), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    results = []
    for label, ep in [("right", run_r), ("left", run_l)]:
        p_ep = pos.restrict(ep)
        occ, _ = np.histogram(p_ep.values, bins=edges)
        occ = occ * np.median(np.diff(pos.t))            # seconds per bin
        tc = nap.compute_1d_tuning_curves(units, p_ep, nb_bins=n_bins,
                                          minmax=(0, float(pos.max())), ep=ep)
        for uid in units.index:
            rates = tc[uid].values
            results.append(dict(session=session, unit=int(uid), direction=label,
                                tc=rates, centers=centers, occ=occ,
                                cell_type=units.cell_type[uid],
                                location=units.location[uid],
                                si=spatial_info(rates, occ),
                                n_spikes=len(units[uid].restrict(ep)),
                                field=find_field(rates, centers)))

    # --- theta phase locking during running ---------------------------------
    locking = []
    for uid in units.index:
        st = units[uid].restrict(run)
        ph = st.value_from(phase).values if len(st) else np.array([])
        R, mu, p = hc11.rayleigh(ph)
        locking.append(dict(session=session, unit=int(uid), n=len(ph),
                            mrl=R, pref_phase=mu, p=p,
                            cell_type=units.cell_type[uid],
                            phases=ph))

    return dict(session=session, maze=maze, pos=pos, speed=speed, run=run,
                run_right=run_r, run_left=run_l, units=units, lfp=lfp,
                filt=filt, phase=phase, best_ch=best, theta_delta=best_score,
                place=results, locking=locking, centers=centers, edges=edges)


def precession(res, min_spikes=40):
    """Circular-linear regression of theta phase on in-field position."""
    units, phase, pos = res["units"], res["phase"], res["pos"]
    out = []
    for rec in res["place"]:
        f = rec["field"]
        if f is None or rec["cell_type"] != "excitatory":
            continue
        if f["peak_rate"] < hc11.MIN_PEAK_RATE or rec["si"] < 0.5:
            continue
        ep = res["run_right"] if rec["direction"] == "right" else res["run_left"]
        st = units[rec["unit"]].restrict(ep)
        if len(st) < min_spikes:
            continue
        spos = st.value_from(pos).values
        sph = st.value_from(phase).values
        inside = (spos >= f["lo"]) & (spos <= f["hi"])
        if inside.sum() < min_spikes:
            continue
        # Distance travelled into the field, 0 at entry and 1 at exit. On
        # leftward runs the animal enters at the high-position edge, so the
        # coordinate has to be reversed relative to the linearized axis.
        if rec["direction"] == "right":
            x = (spos[inside] - f["lo"]) / f["width"]
        else:
            x = (f["hi"] - spos[inside]) / f["width"]
        phi = sph[inside]
        rho, p, slope, phi0 = hc11.circ_lin_corr(x, phi)
        out.append(dict(session=rec["session"], unit=rec["unit"],
                        direction=rec["direction"], rho=rho, p=p,
                        slope=slope, phi0=phi0, n=int(inside.sum()),
                        x=x, phi=phi, field=f, si=rec["si"],
                        tc=rec["tc"], centers=rec["centers"]))
    return out


if __name__ == "__main__":
    res = analyze_session(SESSION)
    prec = precession(res)
    print(f"best LFP channel {res['best_ch']} (theta/delta {res['theta_delta']:.2f})")

    lock = [l for l in res["locking"] if l["n"] >= 100]
    sig = [l for l in lock if l["p"] < 0.01]
    print(f"phase locking: {len(sig)}/{len(lock)} units significant (Rayleigh p<0.01)")
    exc = [l for l in lock if l["cell_type"] == "excitatory"]
    print(f"  pyramidal mean MRL {np.mean([l['mrl'] for l in exc]):.3f}")

    print(f"place cells with fields: {len(prec)} cell-direction pairs")
    neg = [q for q in prec if q["p"] < 0.05 and q["slope"] < 0]
    print(f"  significant precession (p<0.05, negative slope): {len(neg)}")
    print(f"  median slope: {np.median([q['slope'] for q in prec]):.2f} rad/field")

    with open(f"results_{SESSION}.pkl", "wb") as fh:
        pickle.dump(dict(res={k: v for k, v in res.items()
                              if k in ("session", "best_ch", "theta_delta",
                                       "place", "locking", "centers")},
                         prec=prec), fh)
    print("wrote results pickle")
