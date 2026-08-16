"""Per-session ring-attractor pipeline. Saves a results dict to <sid>_results.npz."""
import warnings, sys, json
warnings.filterwarnings("ignore")
import numpy as np
import pynapple as nap

from loader import load_session, SESSIONS
from hdcells import screen_hd_cells, mvl_pref, ANG
import ringanalysis as ra

BIN = 0.1          # s
SMOOTH_SD = 2.0    # bins -> 200 ms Gaussian, used for correlations and PCA
MANIFOLD_SD = 2.0  # bins, smoothing used for the Isomap embedding
ACT_PCT = 20       # drop the least active 20% of bins before embedding
STATES = ["wake", "REM", "nREM"]


def nearest_hd(hd, t, tol):
    """Head direction sampled at times `t`, NaN where the nearest sample is > tol away."""
    ht, hv = hd.index.values, hd.values
    i = np.clip(np.searchsorted(ht, t), 1, len(ht) - 1)
    left = np.abs(t - ht[i - 1]) < np.abs(ht[i] - t)
    j = np.where(left, i - 1, i)
    v = hv[j].astype(float)
    v[np.abs(t - ht[j]) > tol] = np.nan
    return v


def analyze(sid, seed=0, n_null=20, verbose=True):
    rng = np.random.default_rng(seed)
    s = load_session(SESSIONS[sid])
    spikes, hd, ep = s["spikes"], s["hd"], s["epochs"]
    wake = ep["wake_tracked"]
    eps = {"wake": wake, "REM": ep["REM"], "nREM": ep["Non-REM"]}

    df, tc = screen_hd_cells(spikes, hd, wake, n_shuffle=200, seed=seed)
    hd_units = df.loc[df.is_hd, "unit"].values
    pref = df.loc[df.is_hd, "pref"].values
    order = np.argsort(pref)
    if verbose:
        print(f"{sid}: {len(spikes)} units, {len(hd_units)} HD cells")
    if len(hd_units) < 8:
        raise RuntimeError(f"{sid}: too few HD cells ({len(hd_units)})")

    hd_spikes = spikes[list(hd_units)]
    tc_hd = np.nan_to_num(tc[hd_units].values)          # (nbins, n_hd), Hz

    # control ensemble: non-HD units with a comparable firing-rate range
    ctrl = df.loc[~df.is_hd & (df.rate > 0.5) & (df.rate < 50), "unit"].values
    ctrl_spikes = spikes[list(ctrl)] if len(ctrl) >= 8 else None

    out = dict(sid=sid, subject=s["subject"], n_units=len(spikes),
               hd_units=hd_units, pref=pref, order=order, tc_hd=tc_hd,
               screen=df.to_dict("list"), bin=BIN,
               ep_len={k: float(v.tot_length()) for k, v in eps.items()})

    for st in STATES:
        e = eps[st]
        t, x = ra.population_matrix(hd_spikes, e, BIN, SMOOTH_SD)
        keep = x.sum(1) > 0                              # drop wholly silent bins
        t, x = t[keep], x[keep]
        z = ra.zscore_cols(x)

        # --- pairwise correlation structure ---
        d, r, cmat = ra.pairwise_corr_vs_angle(x, pref)
        a, b, r2 = ra.cosine_fit(d, r)

        # --- Bayesian decoding against the wake tuning curves ---
        counts = np.asarray(hd_spikes.count(BIN, e).values, dtype=float)[keep]
        dec, post = ra.bayesian_decode(counts, tc_hd, BIN)
        da, db = ra.split_half_decode(counts, tc_hd, BIN, np.random.default_rng(seed))
        coh = ra.agreement(da, db)

        # --- manifold ---
        mkeep, mx = ra.manifold_input(counts, MANIFOLD_SD, ACT_PCT)
        sel, emb = ra.ring_embedding(mx, seed=seed)
        rs = ra.ring_stats(emb)
        theta_emb = rs.pop("theta")
        dec_m = dec[mkeep][sel]
        map_conc = ra.map_concentration(theta_emb, dec_m)
        null_map = np.mean([ra.map_concentration(theta_emb, rng.permutation(dec_m))
                            for _ in range(10)])
        ev1d = ra.cv_one_d_explained_variance(counts, z, tc_hd, BIN,
                                              np.random.default_rng(seed))
        dz = ra.dimensionality_input(counts, SMOOTH_SD)
        spec = ra.pca_spectrum(dz, 10)
        pr = ra.participation_ratio(dz)

        # angular drift speed of the internal bump (rad/s), across contiguous bins
        gap = np.diff(t) > 1.5 * BIN
        step = np.abs(ra.circ_diff(dec[1:], dec[:-1])) / BIN
        step = step[~gap]

        # --- shuffle null: independent circular time shifts per neuron ---
        null_r2, null_coh, null_holl, null_ev = [], [], [], []
        for _ in range(n_null):
            xs = ra.shift_shuffle(x, rng)
            ds, rs_, _ = ra.pairwise_corr_vs_angle(xs, pref)
            null_r2.append(ra.cosine_fit(ds, rs_)[2])
            cs = ra.shift_shuffle(counts, rng)
            na, nb = ra.split_half_decode(cs, tc_hd, BIN, np.random.default_rng(seed))
            null_coh.append(ra.agreement(na, nb))
            null_ev.append(ra.cv_one_d_explained_variance(
                cs, ra.zscore_cols(xs), tc_hd, BIN, np.random.default_rng(seed), n_rep=1))
            _, mxs = ra.manifold_input(cs, MANIFOLD_SD, ACT_PCT)
            _, semb = ra.ring_embedding(mxs, max_points=1500, seed=seed)
            null_holl.append(ra.ring_stats(semb)["hollowness"])

        res = dict(t=t, x=x, z=z, counts=counts, corr_d=d, corr_r=r, cmat=cmat,
                   cos_a=a, cos_b=b, cos_r2=r2,
                   emb=emb, emb_keep=mkeep, emb_sel=sel, theta_emb=theta_emb, ev1d=ev1d,
                   map_conc=map_conc, null_map_conc=null_map,
                   pca_spectrum=spec, participation_ratio=pr,
                   hollowness=rs["hollowness"], cv_r=rs["cv_r"], coverage=rs["coverage"],
                   decoded=dec, decoded_emb=dec_m, split_coh=coh, drift=step,
                   null_cos_r2=np.array(null_r2), null_coh=np.array(null_coh),
                   null_hollowness=np.array(null_holl), null_ev1d=np.array(null_ev))

        if st == "wake":
            true_hd = nearest_hd(hd, t, BIN)
            ok = np.isfinite(true_hd)
            res["true_hd"] = true_hd
            res["decode_err"] = np.abs(ra.circ_diff(dec[ok], true_hd[ok]))
            res["decode_circ_corr"] = ra.circ_corr(dec[ok], true_hd[ok])
            hd_m = true_hd[mkeep][sel]
            m = np.isfinite(hd_m)
            res["true_hd_emb"] = hd_m
            res["map_conc_true"] = ra.map_concentration(theta_emb[m], hd_m[m])

        # control ensemble: simultaneously recorded non-HD units, same pipeline
        if ctrl_spikes is not None:
            tc_ctrl = np.nan_to_num(tc[ctrl].values)
            pref_c = mvl_pref(tc_ctrl)[1]
            _, xc = ra.population_matrix(ctrl_spikes, e, BIN, SMOOTH_SD)
            cc = np.asarray(ctrl_spikes.count(BIN, e).values, dtype=float)
            m = xc.sum(1) > 0
            xc, cc = xc[m], cc[m]
            dc, rc, _ = ra.pairwise_corr_vs_angle(xc, pref_c)
            _, mxc = ra.manifold_input(cc, MANIFOLD_SD, ACT_PCT)
            csel, cemb = ra.ring_embedding(mxc, seed=seed)
            res["ctrl"] = dict(cos_r2=ra.cosine_fit(dc, rc)[2],
                               hollowness=ra.ring_stats(cemb)["hollowness"],
                               emb=cemb,
                               participation_ratio=ra.participation_ratio(
                                   ra.dimensionality_input(cc, SMOOTH_SD)),
                               pca_spectrum=ra.pca_spectrum(
                                   ra.dimensionality_input(cc, SMOOTH_SD), 10),
                               n=xc.shape[1])
        out[st] = res
        if verbose:
            print(f"  {st}: {x.shape[0]} bins  cos R2={r2:.2f} (null {np.mean(null_r2):.2f})  "
                  f"hollow={rs['hollowness']:.2f} (null {np.mean(null_holl):.2f})  "
                  f"cv-1D-EV={ev1d:.2f} (null {np.mean(null_ev):.2f})  "
                  f"map-conc={map_conc:.2f} (null {null_map:.2f})  "
                  f"split-half coh={coh:.2f} (null {np.mean(null_coh):.2f})", flush=True)
    return out


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        r = analyze(sid)
        np.save(f"{sid}_results.npy", r, allow_pickle=True)
