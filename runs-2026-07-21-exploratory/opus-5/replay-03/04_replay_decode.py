"""Stage 4: Bayesian decoding of the trajectory represented inside each SWR.

Candidate events are population-burst events that overlap a detected ripple
(the standard definition in the replay literature).  Each event is decoded in
20 ms bins against the two direction-specific place-field templates, scored by
the posterior-weighted correlation between decoded position and time, and
tested against two shuffles.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import replaylib as R

SESSION = os.environ.get("SESSION", "Achilles-10252013")
FIG, CACHE = "figures", "cache"
rng = np.random.default_rng(0)

DEC_BIN = 0.020        # s, decoding bin
MUA_BIN = 0.001        # s
MUA_SIGMA = 0.010      # s, Gaussian smoothing of the MUA rate
MUA_HIGH = 3.0         # SD, burst detection threshold
MUA_LOW = 0.0          # SD, burst boundary
EVT_MIN, EVT_MAX = 0.10, 0.50   # s
MIN_ACTIVE = 5         # place cells that must fire in the event
MIN_BINS = 5           # decoding bins
N_SHUFFLE = 500

pf = np.load(f"{CACHE}/place_fields.npz")
place_ids = pf["place_ids"]
centers = pf["centers"]
templates = {"right": pf["tc_right"], "left": pf["tc_left"]}
rip = np.load(f"{CACHE}/ripples.npz")

h5, nwbfile, nwb = R.open_session(SESSION)
epochs = {row.label: nap.IntervalSet(start=row.start_time, end=row.stop_time)
          for row in nwbfile.epochs.to_dataframe().itertuples()}
units = nwb["units"][list(place_ids)]
print(f"{len(place_ids)} place cells, {len(centers)} spatial bins")


# ------------------------------------------------------------------ events
def population_burst_events(ep, ripple_iv):
    """Population-burst events within `ep` that overlap a detected ripple."""
    start, stop = float(ep.start[0]), float(ep.end[0])
    spikes = np.concatenate([units[i].t for i in units.index])
    spikes = np.sort(spikes[(spikes >= start) & (spikes <= stop)])
    edges = np.arange(start, stop + MUA_BIN, MUA_BIN)
    mua, _ = np.histogram(spikes, bins=edges)
    mua = gaussian_filter1d(mua.astype(np.float32), MUA_SIGMA / MUA_BIN)
    z = (mua - mua.mean()) / mua.std()
    t = edges[:-1] + MUA_BIN / 2

    above = z > MUA_LOW
    s_i = np.where(np.diff(np.r_[0, above.astype(np.int8)]) == 1)[0]
    e_i = np.where(np.diff(np.r_[above.astype(np.int8), 0]) == -1)[0]
    peak = np.array([z[s:e + 1].max() for s, e in zip(s_i, e_i)])
    ok = peak >= MUA_HIGH
    s, e = t[s_i[ok]], t[e_i[ok]]
    dur = e - s
    ok2 = (dur >= EVT_MIN) & (dur <= EVT_MAX)
    s, e, peak = s[ok2], e[ok2], peak[ok][ok2]

    # Keep only bursts that contain a ripple peak: this is what makes them SWRs.
    keep = np.array([np.any((ripple_iv >= a) & (ripple_iv <= b)) for a, b in zip(s, e)])
    print(f"  population bursts {len(s)} -> {keep.sum()} coincident with a ripple")
    return nap.IntervalSet(start=s[keep], end=e[keep]), peak[keep]


# ------------------------------------------------------------------ scoring
def weighted_fit(post, pos, tt):
    """Posterior-weighted correlation and regression slope (m/s)."""
    r = R.weighted_correlation(post, pos, tt)
    w = post
    sw = w.sum()
    mt = (w * tt[:, None]).sum() / sw
    mx = (w * pos[None, :]).sum() / sw
    cov = (w * (tt[:, None] - mt) * (pos[None, :] - mx)).sum() / sw
    vt = (w * (tt[:, None] - mt) ** 2).sum() / sw
    return r, (cov / vt if vt > 0 else np.nan)


def shuffle_pvals(post, pos, tt, r_obs):
    """Column-cycle and time-bin-permutation null distributions."""
    n_t, n_x = post.shape
    cols = np.arange(n_x)
    shift = rng.integers(0, n_x, size=(N_SHUFFLE, n_t))
    idx = (cols[None, None, :] + shift[:, :, None]) % n_x
    cyc = np.take_along_axis(np.broadcast_to(post, (N_SHUFFLE, n_t, n_x)), idx, axis=2)
    r_cyc = R.weighted_correlation_stack(cyc, pos, tt)

    order = np.argsort(rng.random((N_SHUFFLE, n_t)), axis=1)
    perm = post[order]
    r_perm = R.weighted_correlation_stack(perm, pos, tt)
    a = np.abs(r_obs)
    return (np.mean(np.abs(r_cyc) >= a), np.mean(np.abs(r_perm) >= a))


def analyse_epoch(name, tmpl=None, tag=""):
    ep = epochs[name]
    print(f"\n=== {name}{tag} ===")
    tmpl = templates if tmpl is None else tmpl
    ripple_peaks = rip[f"{name}_peak_t"]
    events, burst_peak = population_burst_events(ep, ripple_peaks)

    counts = units.count(DEC_BIN, events)
    which = np.searchsorted(events.end, counts.t)  # event index for every bin
    out = []
    for ei in tqdm(range(len(events)), desc=f"decoding {name}"):
        sel = which == ei
        c = counts.values[sel]
        if c.shape[0] < MIN_BINS or (c.sum(axis=0) > 0).sum() < MIN_ACTIVE:
            continue
        tt = counts.t[sel]
        tt = tt - tt[0]
        rec = dict(event=ei, t_start=float(events.start[ei]),
                   t_end=float(events.end[ei]), n_bins=c.shape[0],
                   n_active=int((c.sum(axis=0) > 0).sum()),
                   n_spikes=int(c.sum()), burst_z=float(burst_peak[ei]))
        for d in ("right", "left"):
            post = R.bayesian_decode(c, tmpl[d], DEC_BIN)
            r, slope = weighted_fit(post, centers, tt)
            p_cyc, p_perm = shuffle_pvals(post, centers, tt, r)
            rec[f"r_{d}"], rec[f"slope_{d}"] = r, slope
            rec[f"pcyc_{d}"], rec[f"pperm_{d}"] = p_cyc, p_perm
            rec[f"maxjump_{d}"] = float(np.abs(np.diff(centers[np.argmax(post, 1)])).max())
        out.append(rec)
    print(f"  {len(out)} events passed the spiking criteria "
          f"(>= {MIN_BINS} bins, >= {MIN_ACTIVE} active cells)")
    return out, events


results = {}
event_sets = {}
for name in ("POSTEpoch", "PREEpoch"):
    results[name], event_sets[name] = analyse_epoch(name)

# Cell-identity control: reassign place fields to random cells.  This destroys
# the spatial code while preserving every spike time, every event boundary and
# every population statistic, so the fraction of "significant" events it yields
# is the empirical false-positive rate of the whole pipeline.
perm = rng.permutation(len(place_ids))
tmpl_shuf = {d: templates[d][perm] for d in templates}
results["POSTEpoch_idshuffle"], _ = analyse_epoch("POSTEpoch", tmpl_shuf,
                                                  tag=" (cell-ID shuffled)")

# ------------------------------------------------------------------ summary
import pandas as pd

frames = {}
for name, recs in results.items():
    df = pd.DataFrame(recs)
    # Best-supported direction template per event, with a Bonferroni correction
    # for having tested two templates.
    best = np.where(np.abs(df.r_right) >= np.abs(df.r_left), "right", "left")
    df["direction"] = best
    for col in ("r", "slope", "pcyc", "pperm", "maxjump"):
        df[col] = [df[f"{col}_{d}"].iloc[i] for i, d in enumerate(best)]
    df["significant"] = (df.pcyc < 0.025) & (df.pperm < 0.025)
    frames[name] = df
    df.to_csv(f"{CACHE}/replay_events_{name}.csv", index=False)
    print(f"\n{name}: {len(df)} candidate events, "
          f"{df.significant.sum()} significant "
          f"({100 * df.significant.mean():.1f}%), "
          f"median |r| = {np.abs(df.r).median():.3f}")

post_df, pre_df = frames["POSTEpoch"], frames["PREEpoch"]
ctrl_df = frames["POSTEpoch_idshuffle"]
print(f"\nempirical false-positive rate (cell-ID shuffled templates): "
      f"{100 * ctrl_df.significant.mean():.1f}%")

from scipy.stats import chi2_contingency, mannwhitneyu

tab = np.array([[post_df.significant.sum(), (~post_df.significant).sum()],
                [pre_df.significant.sum(), (~pre_df.significant).sum()]])
chi2, p_chi, _, _ = chi2_contingency(tab)
u, p_u = mannwhitneyu(np.abs(post_df.r), np.abs(pre_df.r), alternative="greater")
print(f"\nPOST vs PRE significant fraction: chi2 = {chi2:.1f}, p = {p_chi:.2g}")
print(f"POST vs PRE |weighted r|: Mann-Whitney U = {u:.0f}, p = {p_u:.2g}")

tab2 = np.array([[post_df.significant.sum(), (~post_df.significant).sum()],
                 [ctrl_df.significant.sum(), (~ctrl_df.significant).sum()]])
chi2_c, p_ctrl, _, _ = chi2_contingency(tab2)
tab3 = np.array([[pre_df.significant.sum(), (~pre_df.significant).sum()],
                 [ctrl_df.significant.sum(), (~ctrl_df.significant).sum()]])
chi2_p, p_prectrl, _, _ = chi2_contingency(tab3)
_, p_u_ctrl = mannwhitneyu(np.abs(post_df.r), np.abs(ctrl_df.r), alternative="greater")
_, p_u_prectrl = mannwhitneyu(np.abs(pre_df.r), np.abs(ctrl_df.r), alternative="greater")
print(f"POST vs cell-ID-shuffled control: chi2 = {chi2_c:.1f}, p = {p_ctrl:.2g} "
      f"| |r| Mann-Whitney p = {p_u_ctrl:.2g}")
print(f"PRE  vs cell-ID-shuffled control: chi2 = {chi2_p:.1f}, p = {p_prectrl:.2g} "
      f"| |r| Mann-Whitney p = {p_u_prectrl:.2g}")

np.savez(f"{CACHE}/replay_stats.npz", chi2=chi2, p_chi=p_chi, u=u, p_u=p_u,
         n_post=len(post_df), n_pre=len(pre_df), n_ctrl=len(ctrl_df),
         sig_post=int(post_df.significant.sum()), sig_pre=int(pre_df.significant.sum()),
         sig_ctrl=int(ctrl_df.significant.sum()), p_ctrl=p_ctrl,
         p_prectrl=p_prectrl, p_u_ctrl=p_u_ctrl, p_u_prectrl=p_u_prectrl)
print("\nwrote cache/replay_events_*.csv")
