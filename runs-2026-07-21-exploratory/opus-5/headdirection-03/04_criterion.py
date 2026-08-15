"""Explore how to define an HD cell: pooled shuffle threshold, stability, ROC."""
import numpy as np
import pynapple as nap
import hd_lib

S = "sub-A3701_ses-191119"
d = hd_lib.load_session(S)
hd, units, sq = d["hd"], d["units"], d["epochs"]["wake_square"]
ft = hd_lib.FastTuning(hd, sq, nb_bins=60)
C = ft.centers
auth = units.metadata["is_hd_author"].values

rng = np.random.default_rng(1)
n_sh = 100
obs, null_all = [], []
for c in units.keys():
    r = ft.curve(units[c].t)
    obs.append(hd_lib.circular_mean_vector(C, r)[0])
    null_all.append([hd_lib.circular_mean_vector(C, ft.curve(s))[0]
                     for s in hd_lib.circ_shift_shuffle(units[c], sq, rng, n_sh)])
obs = np.array(obs)
null_all = np.array(null_all)

thr99 = np.percentile(null_all.ravel(), 99)
thr95 = np.percentile(null_all.ravel(), 95)
print("pooled-null MVL: mean %.3f  95th %.3f  99th %.3f  max %.3f"
      % (null_all.mean(), thr95, thr99, null_all.max()))
for name, thr in [("pooled 95th", thr95), ("pooled 99th", thr99),
                  ("pooled max", null_all.max())]:
    sel = obs > thr
    print("  %-12s thr=%.3f -> %2d cells, agreement w/ author %.1f%% (TP %d, FP %d, FN %d)"
          % (name, thr, sel.sum(), 100 * np.mean(sel == auth),
             (sel & auth).sum(), (sel & ~auth).sum(), (~sel & auth).sum()))

n_half = sq.start[0] + (sq.end[0] - sq.start[0]) / 2
h1 = nap.IntervalSet(sq.start[0], n_half)
h2 = nap.IntervalSet(n_half, sq.end[0])
f1, f2 = hd_lib.FastTuning(hd, h1, 60), hd_lib.FastTuning(hd, h2, 60)
stab = np.array([np.corrcoef(f1.curve(units[c].t), f2.curve(units[c].t))[0, 1]
                 for c in units.keys()])
print("\nsplit-half r: author-HD median %.2f | non-HD median %.2f"
      % (np.nanmedian(stab[auth]), np.nanmedian(stab[~auth])))


def auc(score, label):
    o = np.argsort(score)
    ranks = np.empty(len(score)); ranks[o] = np.arange(1, len(score) + 1)
    n1, n0 = label.sum(), (~label).sum()
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print("AUC(MVL vs author label)      = %.3f" % auc(obs, auth))
print("AUC(stability vs author)      = %.3f" % auc(np.nan_to_num(stab), auth))

conj = (obs > thr99) & (stab > 0.5)
print("\nconjunctive (MVL>%.2f & stability>0.5): %d cells, agreement %.1f%% "
      "(TP %d, FP %d, FN %d)" % (thr99, conj.sum(), 100 * np.mean(conj == auth),
                                 (conj & auth).sum(), (conj & ~auth).sum(),
                                 (~conj & auth).sum()))
fs = units.metadata["is_fast_spiking"].values
print("of the %d false positives, %d are fast-spiking (putative interneurons)"
      % ((conj & ~auth).sum(), (conj & ~auth & fs).sum()))
print("author labels among fast-spiking cells:", auth[fs].sum(), "/", fs.sum())
print("MVL of author-HD cells missed:", np.round(obs[~conj & auth], 3))
print("MVL/stab of false positives:")
for i in np.where(conj & ~auth)[0]:
    print("   u%-3d mvl %.2f stab %.2f rate %.1f fs %s exc %s"
          % (i, obs[i], stab[i], units.rates[i], fs[i],
             units.metadata["is_excitatory"].values[i]))
