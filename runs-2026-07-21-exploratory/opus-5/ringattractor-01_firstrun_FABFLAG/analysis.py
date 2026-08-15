"""Per-session ring-attractor analysis for DANDI:000056.

For each session this computes
  1. wake head-direction tuning curves and HD-cell selection,
  2. pairwise correlation structure vs difference in preferred direction,
     during wake / REM / non-REM,
  3. an unsupervised 2-D embedding of the population activity and the
     internal ring coordinate derived from it,
  4. Bayesian decoding of the internal head direction with wake tuning curves,
     with a cell-identity-shuffle control.
"""
import numpy as np
import pynapple as nap
from scipy.stats import pearsonr
from sklearn.manifold import Isomap, SpectralEmbedding

import hd_lib as H
import ring_lib as R

# binning per brain state: sleep sequences run on a compressed timescale, so
# non-REM is analysed with shorter bins than wake / REM.
BINS = {"wake": 0.2, "rem": 0.2, "sws": 0.05}
SMOOTH = {"wake": 1.0, "rem": 1.0, "sws": 2.0}
DRIFT_BIN = 0.1  # common bin size for the bump-drift comparison
CTRL_MAX_DUR = 2500.0  # cap (s) on the epoch used for the resampling controls
STATES = ["wake", "rem", "sws"]


def epoch_of(S, state):
    return S["epochs"]["explore" if state == "wake" else state]


def cap_epoch(ep, max_dur):
    """Take intervals from the start of `ep` until `max_dur` seconds accumulate.

    Used for the resampling controls, which are run on a bounded slice of each
    epoch so that observed and shuffled statistics are computed on exactly the
    same data without the cost of the full multi-hour non-REM epoch.
    """
    import pynapple as _nap

    dur = np.cumsum(ep.end - ep.start)
    n = int(np.searchsorted(dur, max_dur) + 1)
    n = min(n, len(ep))
    return _nap.IntervalSet(start=ep.start[:n], end=ep.end[:n])


# ---------------------------------------------------------------------------
def select_hd_cells(S, mvl_thresh=0.3, rate_thresh=0.5, n_shuffle=100, seed=0):
    ep = S["epochs"]["explore"]
    tc, occ = H.compute_hd_tuning(S["spikes"], S["angle"], ep)
    pd_, mvl = H.circ_mean_vector(tc)
    rate = np.asarray(S["spikes"].restrict(ep).rates)
    info = H.hd_info(tc, occ)

    null = H.shuffle_mvl(S["spikes"], S["angle"], ep, n_shuffle=n_shuffle, seed=seed)
    thresh95 = np.percentile(null, 95, axis=0)
    sig = mvl > thresh95
    mask = sig & (mvl > mvl_thresh) & (rate > rate_thresh)
    return dict(
        tc=tc, occupancy=occ, pd=pd_, mvl=mvl, rate=rate, info=info,
        null_mvl=null, sig=sig, mask=mask,
        ids=list(np.asarray(tc.columns)[mask]), pd_hd=pd_[mask],
    )


# ---------------------------------------------------------------------------
def population_matrix(spikes, ep, bin_size, smooth, drop_quantile=25):
    """Binned, smoothed population matrix with the quietest bins removed.

    Dropping the lowest-activity bins excludes non-REM DOWN states, in which
    almost no cell fires and the population vector carries no direction.
    """
    Rr = R.binned_rates(spikes, ep, bin_size, smooth)
    X = np.asarray(Rr.values, dtype=float)
    t = np.asarray(Rr.index.values)
    tot = X.sum(1)
    keep = tot >= np.percentile(tot, drop_quantile)
    return X[keep], t[keep]


def ring_coordinate(X, n_neighbors=30, max_bins=2500, seed=1, with_isomap=True):
    """Unsupervised circular coordinate from Laplacian eigenmaps.

    Population vectors are square-root transformed (Poisson variance
    stabilisation) and normalised to unit length, which removes the overall
    population-gain dimension so the embedding reflects only the *pattern*
    of activity across cells.
    """
    Xs = np.sqrt(np.maximum(X, 0))
    n = np.linalg.norm(Xs, axis=1, keepdims=True)
    ok = n[:, 0] > 0
    Xn = Xs[ok] / n[ok]
    idx = np.flatnonzero(ok)
    if len(Xn) > max_bins:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(Xn), max_bins, replace=False))
        Xn, idx = Xn[sel], idx[sel]
    lap = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=0)
    emb_lap = lap.fit_transform(Xn)
    theta = np.arctan2(emb_lap[:, 1], emb_lap[:, 0])
    iso = None
    if with_isomap:
        iso = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(Xn)
        iso = iso - iso.mean(0)
    return dict(theta=theta, emb_lap=emb_lap, emb_iso=iso, idx=idx)


def internal_tuning(X, theta, nb=40, min_count=3):
    """Tuning of each cell against the internally derived ring coordinate."""
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    ctr = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(theta, edges) - 1, 0, nb - 1)
    rows, keep = [], []
    for i in range(nb):
        s = idx == i
        if s.sum() >= min_count:
            rows.append(X[s].mean(0))
            keep.append(i)
    W = np.array(rows)
    C = ctr[keep]
    z = (W * np.exp(1j * C)[:, None]).sum(0) / np.maximum(W.sum(0), 1e-12)
    return dict(itc=W, bins=C, pd_int=np.angle(z), mvl_int=np.abs(z))


def align_to_wake(pd_int, pd_wake):
    """Best rigid map (rotation, optional reflection) from internal to wake PDs."""
    best = None
    for s in (1, -1):
        z = np.mean(np.exp(1j * (pd_int - s * pd_wake)))
        if best is None or abs(z) > best["R"]:
            best = dict(R=float(abs(z)), sign=s, offset=float(np.angle(z)))
    return best


# ---------------------------------------------------------------------------
def decode_state(tc_hd, spikes, ep, bin_size):
    dec, prob = nap.decode_1d(
        tuning_curves=tc_hd, group=spikes, ep=ep, bin_size=bin_size, feature=None
    )
    conc, ang = R.posterior_concentration(prob)
    return dec, prob, conc


def split_half_coherence(tc_hd, spikes, ep, bin_size, n_splits=10, seed=0, shuffle=False):
    """Decode head direction from two disjoint halves of the HD population.

    If the population holds a single coherent bump on the ring, the two
    independent read-outs agree bin by bin. The statistic is the mean resultant
    length of the angular difference between them, which is 0 for independent
    read-outs and 1 for perfect agreement. With `shuffle=True` the cell->tuning
    assignment is permuted first, which is the matched null.
    """
    rng = np.random.default_rng(seed)
    cols = list(tc_hd.columns)
    out = []
    for _ in range(n_splits):
        tcs = tc_hd
        if shuffle:
            perm = rng.permutation(len(cols))
            tcs = tc_hd.copy()
            tcs.columns = [cols[p] for p in perm]
            tcs = tcs[cols]
        idx = rng.permutation(len(cols))
        h1 = sorted(cols[i] for i in idx[: len(cols) // 2])
        h2 = sorted(cols[i] for i in idx[len(cols) // 2:])
        g1 = nap.TsGroup({k: spikes[k] for k in h1}, time_support=spikes.time_support)
        g2 = nap.TsGroup({k: spikes[k] for k in h2}, time_support=spikes.time_support)
        d1, _ = nap.decode_1d(tuning_curves=tcs[h1], group=g1, ep=ep,
                              bin_size=bin_size, feature=None)
        d2, _ = nap.decode_1d(tuning_curves=tcs[h2], group=g2, ep=ep,
                              bin_size=bin_size, feature=None)
        n = min(len(d1), len(d2))
        diff = R.wrap_pi(np.asarray(d1.values)[:n] - np.asarray(d2.values)[:n])
        out.append(float(abs(np.mean(np.exp(1j * diff)))))
    return np.array(out)


def cell_shuffle_concentration(tc_hd, spikes, ep, bin_size, n=10, seed=0):
    """Permute which tuning curve belongs to which cell.

    This preserves every single-cell property (rate, tuning shape, temporal
    statistics) and destroys only the correspondence between cells and ring
    positions, so it isolates the population-level ring structure.
    """
    rng = np.random.default_rng(seed)
    cols = list(tc_hd.columns)
    out = []
    for _ in range(n):
        perm = rng.permutation(len(cols))
        tcs = tc_hd.copy()
        tcs.columns = [cols[p] for p in perm]
        tcs = tcs[cols]
        _, prob = nap.decode_1d(
            tuning_curves=tcs, group=spikes, ep=ep, bin_size=bin_size, feature=None
        )
        c, _ = R.posterior_concentration(prob)
        out.append(np.mean(c))
    return np.array(out)


# ---------------------------------------------------------------------------
def analyse_session(session_key, n_shuffle_mvl=100, n_emb_shuffle=5, verbose=True):
    import time as _time

    _t0 = _time.time()

    def _tick(msg):
        if verbose:
            print(f"    [{_time.time() - _t0:6.1f}s] {msg}", flush=True)

    S = H.load_session(session_key)
    _tick(f"loaded {session_key}")
    sel = select_hd_cells(S, n_shuffle=n_shuffle_mvl)
    _tick("HD-cell selection")
    ids, pdh = sel["ids"], sel["pd_hd"]
    res = dict(name=S["name"], n_units=len(S["spikes"]), n_hd=len(ids),
               led=S["led_alignment"], sel=sel,
               epoch_dur={k: float(epoch_of(S, k).tot_length()) for k in STATES})
    if verbose:
        print(f"{S['name']}: {len(ids)}/{len(S['spikes'])} HD cells; "
              + ", ".join(f"{k}={res['epoch_dur'][k]:.0f}s" for k in STATES))
    if len(ids) < 8:
        res["ok"] = False
        return res, S
    res["ok"] = True
    spk = S["spikes"][ids]
    tc_hd = sel["tc"][ids]

    per_state = {}
    for st in STATES:
        ep = epoch_of(S, st)
        bs, sm = BINS[st], SMOOTH[st]
        d = {}

        # --- pairwise correlation structure --------------------------------
        C = R.pairwise_corr(spk, ep, bs, sm)
        dpd, cvals = R.corr_vs_dpd(C, pdh)
        d["corr_matrix"] = C
        d["dpd"], d["cvals"] = dpd, cvals
        d["r_cos"] = float(pearsonr(np.cos(dpd), cvals)[0])
        d["profile"] = R.binned_profile(dpd, cvals, 12)

        # --- unsupervised ring coordinate ----------------------------------
        X, t = population_matrix(spk, ep, bs, sm)
        rc = ring_coordinate(X)
        it = internal_tuning(X[rc["idx"]], rc["theta"])
        al = align_to_wake(it["pd_int"], pdh)
        d.update(emb_iso=rc["emb_iso"], emb_lap=rc["emb_lap"], theta=rc["theta"],
                 idx=rc["idx"], times=t, pd_int=it["pd_int"], mvl_int=it["mvl_int"],
                 itc=it["itc"], itc_bins=it["bins"], align=al)
        _tick(f"{st}: embedding")
        d["ring_score"] = R.ring_score(rc["emb_lap"])
        d["ring_score_iso"] = R.ring_score(rc["emb_iso"])

        # shuffle control for the ring: independent circular time shifts
        alsh, rsh = [], []
        for k in range(n_emb_shuffle):
            sh = R.shuffle_population(spk, ep, seed=100 + k)
            Xs, _ = population_matrix(sh, ep, bs, sm)
            rcs = ring_coordinate(Xs, seed=1 + k, with_isomap=False)
            its = internal_tuning(Xs[rcs["idx"]], rcs["theta"])
            alsh.append(align_to_wake(its["pd_int"], pdh)["R"])
            rsh.append(R.ring_score(rcs["emb_lap"])["ring_index"])
        d["align_shuffle"] = np.array(alsh)
        d["ring_score_shuffle"] = np.array(rsh)

        # --- decoding ------------------------------------------------------
        dec, prob, conc = decode_state(tc_hd, spk, ep, bs)
        d["decoded"] = dec
        d["conc"] = conc
        d["conc_mean"] = float(np.mean(conc))
        _tick(f"{st}: embedding shuffles")
        ep_ctrl = cap_epoch(ep, CTRL_MAX_DUR)
        d["conc_shuffle"] = cell_shuffle_concentration(tc_hd, spk, ep_ctrl, bs, n=5)
        d["split_half"] = split_half_coherence(tc_hd, spk, ep_ctrl, bs, n_splits=6)
        d["split_half_shuffle"] = split_half_coherence(
            tc_hd, spk, ep_ctrl, bs, n_splits=6, seed=7, shuffle=True
        )

        _tick(f"{st}: controls")
        # drift of the internal bump, measured at a bin size common to all
        # states so that the three are directly comparable
        dec_c, _, _ = decode_state(tc_hd, spk, ep, DRIFT_BIN)
        tt = np.asarray(dec_c.index.values)
        av, _ = R.angular_velocity(np.asarray(dec_c.values), tt)
        ok = np.diff(tt) < DRIFT_BIN * 1.5
        d["ang_vel"] = av[ok]
        d["drift_speed"] = float(np.median(np.abs(av[ok])))
        if st == "wake":
            true = dec.value_from(S["angle"])
            n = min(len(true), len(dec))
            err = R.wrap_pi(np.asarray(dec.values)[:n] - np.asarray(true.values)[:n])
            d["decode_err_med_deg"] = float(np.degrees(np.median(np.abs(err))))
            d["decode_err_R"] = float(abs(np.mean(np.exp(1j * err))))
            ta = np.asarray(true.index.values)
            avt, _ = R.angular_velocity(np.asarray(true.values), ta)
            d["ang_vel_true"] = avt[np.diff(ta) < DRIFT_BIN * 1.5]
        _tick(f"{st} done")
        per_state[st] = d

    # cross-state similarity of the pairwise correlation structure
    iu = np.triu_indices(len(ids), 1)
    res["corr_similarity"] = {
        f"wake_vs_{st}": float(pearsonr(per_state["wake"]["corr_matrix"][iu],
                                        per_state[st]["corr_matrix"][iu])[0])
        for st in ("rem", "sws")
    }
    res["states"] = per_state
    return res, S
