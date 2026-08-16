"""Run the full per-session analysis over all sessions and cache the results."""
import os
import pickle

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import hd_lib

NB = 60
N_SHUFFLE = 200
BIN_DECODE = 0.2
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def analyse(session, seed=0):
    d = hd_lib.load_session(session)
    hd, units = d["hd"], d["units"]
    sq = d["epochs"]["wake_square"]
    rng = np.random.default_rng(seed)

    ft = hd_lib.FastTuning(hd, sq, NB)
    C = ft.centers
    tc = np.array([ft.curve(units[c].t) for c in units.keys()])
    mvl = np.zeros(len(tc)); pref = np.zeros(len(tc)); info = np.zeros(len(tc))
    for i, r in enumerate(tc):
        mvl[i], pref[i] = hd_lib.circular_mean_vector(C, r)
        info[i] = hd_lib.hd_information(C, r, ft.occupancy)

    # circular-shift null, pooled across cells
    null = np.zeros((len(tc), N_SHUFFLE))
    for i, c in enumerate(units.keys()):
        for j, s in enumerate(hd_lib.circ_shift_shuffle(units[c], sq, rng, N_SHUFFLE)):
            null[i, j] = hd_lib.circular_mean_vector(C, ft.curve(s))[0]
    p_cell = (1 + (null >= mvl[:, None]).sum(1)) / (N_SHUFFLE + 1)
    thr = np.percentile(null.ravel(), 99)

    # split-half stability
    mid = sq.start[0] + (sq.end[0] - sq.start[0]) / 2
    e1, e2 = nap.IntervalSet(sq.start[0], mid), nap.IntervalSet(mid, sq.end[0])
    f1, f2 = hd_lib.FastTuning(hd, e1, NB), hd_lib.FastTuning(hd, e2, NB)
    tc1 = np.array([f1.curve(units[c].t) for c in units.keys()])
    tc2 = np.array([f2.curve(units[c].t) for c in units.keys()])
    stab = np.array([np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
                     for a, b in zip(tc1, tc2)])

    is_hd = (mvl > thr) & (p_cell < 0.01) & (stab > 0.5)
    hd_idx = np.where(is_hd)[0]

    # cross-environment
    cross = None
    if "wake_triangle" in d["epochs"]:
        tri = d["epochs"]["wake_triangle"]
        ftt = hd_lib.FastTuning(hd, tri, NB)
        tct = np.array([ftt.curve(units[c].t) for c in units.keys()])
        pref_t = np.array([hd_lib.circular_mean_vector(C, r)[1] for r in tct])
        mvl_t = np.array([hd_lib.circular_mean_vector(C, r)[0] for r in tct])
        dphi = hd_lib.angdiff(pref_t[hd_idx], pref[hd_idx])
        z = np.mean(np.exp(1j * dphi))
        cross = dict(pref_tri=pref_t, mvl_tri=mvl_t, dphi=dphi,
                     rotation=float(np.mod(np.angle(z), 2 * np.pi)), R=float(np.abs(z)))

    # cross-validated Bayesian decoding of head direction
    sub = units[list(hd_idx)]
    tcdf1 = pd.DataFrame(index=C, data={c: f1.curve(units[c].t) for c in hd_idx})
    dec, _ = nap.decode_1d(tuning_curves=tcdf1, group=sub, ep=e2, bin_size=BIN_DECODE)
    true = hd.restrict(e2).interpolate(dec)
    derr = np.degrees(np.abs(hd_lib.angdiff(dec.values, true.values)))
    derr = derr[~np.isnan(derr)]

    # decoding accuracy as a function of ensemble size
    curve_n = []
    for n in [2, 5, 10, 20, 40, 80]:
        if n > len(hd_idx):
            break
        pick = np.sort(rng.permutation(hd_idx)[:n])
        tcn = pd.DataFrame(index=C, data={c: f1.curve(units[c].t) for c in pick})
        dn, _ = nap.decode_1d(tuning_curves=tcn, group=units[list(pick)], ep=e2,
                              bin_size=BIN_DECODE)
        tn = hd.restrict(e2).interpolate(dn)
        e = np.degrees(np.abs(hd_lib.angdiff(dn.values, tn.values)))
        curve_n.append((n, float(np.nanmedian(e))))

    # pairwise correlation structure in wake / REM / nREM
    home = d["epochs"]["home_cage"]
    pw = {}
    iu = np.triu_indices(len(hd_idx), 1)
    offs = np.abs(hd_lib.angdiff(pref[hd_idx][:, None], pref[hd_idx][None, :]))[iu]
    for name, ep, bs in [("wake", sq, 0.5),
                         ("REM", d["states"].get("rem", nap.IntervalSet([], [])).intersect(home), 0.5),
                         ("nREM", d["states"].get("nrem", nap.IntervalSet([], [])).intersect(home), 0.1)]:
        if ep.tot_length() < 100:
            pw[name] = None
            continue
        z = np.asarray(sub.count(bs, ep).values, dtype=float)
        pw[name] = np.corrcoef(z.T)[iu]

    return dict(
        session=session, subject=d["subject"], centers=C, tc=tc, tc1=tc1, tc2=tc2,
        occupancy=ft.occupancy, mvl=mvl, pref=pref, info=info, stab=stab,
        p_cell=p_cell, null=null, thr=thr, is_hd=is_hd,
        rates=np.array([units.rates[c] for c in units.keys()]),
        is_hd_author=units.metadata["is_hd_author"].values,
        is_excitatory=units.metadata["is_excitatory"].values,
        is_fast_spiking=units.metadata["is_fast_spiking"].values,
        cross=cross, decode_err=derr, decode_vs_n=curve_n,
        pw=pw, pw_offsets=offs,
        rem_s=float(d["states"].get("rem", nap.IntervalSet([], [])).intersect(home).tot_length()),
        wake_s=float(sq.tot_length()),
    )


if __name__ == "__main__":
    out = {}
    for s in tqdm(hd_lib.SESSIONS, desc="sessions"):
        out[s] = analyse(s)
        r = out[s]
        print("%s  n=%3d  HD=%3d (%.0f%%)  author=%3d  median|err|=%.1f deg"
              % (s, len(r["mvl"]), r["is_hd"].sum(), 100 * r["is_hd"].mean(),
                 r["is_hd_author"].sum(), np.median(r["decode_err"])), flush=True)
    pickle.dump(out, open(os.path.join(OUT, "all_sessions.pkl"), "wb"))
    print("saved", os.path.join(OUT, "all_sessions.pkl"))
