"""Stage 5: run the full replay pipeline over several sessions and cache the results."""
import os, sys, numpy as np, pandas as pd, pickle
import pynapple as nap
import pipeline as pl

SESSIONS = ["Achilles_10252013", "Achilles_11012013", "Cicero_09102014", "Gatsby_08282013"]


def run_session(session):
    out = f"cache/session_{session}.pkl"
    if os.path.exists(out):
        print(f"[{session}] cached"); return pickle.load(open(out, "rb"))
    print(f"=== {session}")
    d = pl.load_session(session)
    ep, states = d["epochs"], d["states"]
    nrem = states["Non-REM"]
    eps = {"PRE": nrem.intersect(ep["PREEpoch"]), "POST": nrem.intersect(ep["POSTEpoch"])}
    print(f"  NREM: PRE {eps['PRE'].tot_length():.0f} s, POST {eps['POST'].tot_length():.0f} s")

    ch, rip_prof = pl.pick_ripple_channel(d, eps["POST"])
    print(f"  ripple channel {ch}")
    rip, lfp_raw, lfp_filt, zenv = pl.detect_ripples(d, ch, nrem, eps)
    for k in rip:
        print(f"  {k}: {len(rip[k]['ep'])} ripples ({len(rip[k]['ep'])/eps[k].tot_length():.2f} Hz)")

    pos, laps, dt, track = pl.linear_position(d)
    dirs, ok, direction, speed = pl.split_laps(pos, laps, track)
    print(f"  linearised track length {track:.2f} m ({pl.find_linearized(d)})")
    print(f"  {ok.sum()} full traversals ({len(dirs['rightward'][1])} R / {len(dirs['leftward'][1])} L), "
          f"mean speed {speed[ok].mean():.2f} m/s")

    pyr = d["units"][list(d["pyr"])]
    bins = np.linspace(0, track, pl.NBINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    tc = {k: pl.tuning_curves(pyr, pos, dirs[k][0], track) for k in dirs}
    occ = {k: np.histogram(pos.restrict(dirs[k][0]).d, bins)[0] * dt for k in dirs}
    si = {k: np.array([pl.spatial_info(tc[k][:, i], occ[k]) for i in range(len(d["pyr"]))]) for k in dirs}
    pk = {k: tc[k].max(0) for k in dirs}
    is_pc = ((pk["rightward"] > 1) & (si["rightward"] > 0.5)) | ((pk["leftward"] > 1) & (si["leftward"] > 0.5))
    pc_idx = np.where(is_pc)[0]
    pc_ids = d["pyr"][pc_idx]
    print(f"  {len(pc_ids)}/{len(d['pyr'])} place cells")
    pc_units = d["units"][list(pc_ids)]
    TC = {k: tc[k][:, pc_idx] for k in tc}

    # --- cross-validated decoder check on running laps
    errs = []
    for k in dirs:
        ids = dirs[k][1]
        tr = nap.IntervalSet(start=laps.start[ids[0::2]], end=laps.end[ids[0::2]])
        te = nap.IntervalSet(start=laps.start[ids[1::2]], end=laps.end[ids[1::2]])
        tct = np.maximum(pl.tuning_curves(pc_units, pos, tr, track), 1e-3)
        cnt = pc_units.count(0.25, ep=te)
        N = np.asarray(cnt.values, float)
        P = pl._norm(N @ np.log(tct).T - 0.25 * tct.sum(1)[None, :])
        errs.append(centers[P.argmax(1)] - np.interp(cnt.t, pos.t, pos.d))
    med_err = float(np.median(np.abs(np.concatenate(errs))))
    print(f"  cross-validated decoding error: {med_err*100:.1f} cm")

    res = dict(session=session, channel=ch, track=track, ripple_profile=rip_prof, centers=centers,
               tc=TC, si=si, pc_idx=pc_idx, pc_ids=pc_ids, n_pyr=len(d["pyr"]),
               med_err=med_err, nrem={k: (v.start, v.end) for k, v in eps.items()},
               laps=(laps.start, laps.end), lap_dir=direction, lap_ok=ok,
               mean_run_speed=float(speed[ok].mean()), events={}, posteriors={}, pbe_frac={})

    for k in ["POST", "PRE"]:
        pbe, mua_z, frac = pl.detect_pbes(d, eps[k], rip[k])
        res["pbe_frac"][k] = frac
        print(f"  {k}: {len(pbe)} ripple-coincident population bursts")
        df, post = pl.decode_events(pc_units, TC, pbe, centers, seed=hash(k) % 1000,
                                    progress=f"  decoding {k}")
        res["events"][k] = df
        if k == "POST":
            res["posteriors"] = post
            res["pbe"] = (pbe.start, pbe.end)
        print(f"    {len(df)} analysable events, {df.sig.sum()} significant ({100*df.sig.mean():.1f}%)")
    dfc, _ = pl.decode_events(pc_units, TC,
                              nap.IntervalSet(start=res["pbe"][0], end=res["pbe"][1]), centers,
                              seed=99, shuffle_observed=True, progress="  decoding CONTROL")
    res["events"]["CONTROL"] = dfc
    print(f"    control: {len(dfc)} events, {dfc.sig.sum()} significant ({100*dfc.sig.mean():.1f}%)")

    res["rip_summary"] = {k: dict(n=len(rip[k]["ep"]), rate=len(rip[k]["ep"]) / eps[k].tot_length(),
                                  dur=float(np.median(rip[k]["ep"].end - rip[k]["ep"].start)))
                          for k in rip}
    # keep a small LFP excerpt around the strongest POST ripple for plotting
    j = int(np.argmax(rip["POST"]["peak_z"]))
    c = rip["POST"]["peak_t"][j]
    res["example_ripple"] = dict(t=lfp_raw.get(c - .3, c + .3).t - c,
                                 raw=lfp_raw.get(c - .3, c + .3).d,
                                 filt=lfp_filt.get(c - .3, c + .3).d,
                                 z=zenv.get(c - .3, c + .3).d,
                                 start=rip["POST"]["ep"].start[j] - c, end=rip["POST"]["ep"].end[j] - c)
    idx = (rip["POST"]["peak_t"] * d["fs"]).astype(int)
    w = int(0.1 * d["fs"])
    idx = idx[(idx > w) & (idx < len(lfp_raw) - w)][:2000]
    res["rip_trig"] = dict(raw=np.stack([lfp_raw.d[i - w:i + w] for i in idx]).mean(0),
                           filt=np.stack([lfp_filt.d[i - w:i + w] for i in idx]).mean(0),
                           t=np.arange(-w, w) / d["fs"])
    pickle.dump(res, open(out, "wb"))
    return res


if __name__ == "__main__":
    for s in (sys.argv[1:] or SESSIONS):
        run_session(s)
