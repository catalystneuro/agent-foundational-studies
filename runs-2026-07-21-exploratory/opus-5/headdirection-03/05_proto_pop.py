"""Prototype decoding, cross-environment stability and sleep correlation structure."""
import numpy as np
import pandas as pd
import pynapple as nap
import hd_lib

S = "sub-A3701_ses-191119"
d = hd_lib.load_session(S)
hd, units = d["hd"], d["units"]
sq, tri = d["epochs"]["wake_square"], d["epochs"]["wake_triangle"]
ft = hd_lib.FastTuning(hd, sq, nb_bins=60)
C = ft.centers

mvl, pref = [], []
for c in units.keys():
    m, a = hd_lib.circular_mean_vector(C, ft.curve(units[c].t))
    mvl.append(m); pref.append(a)
mvl, pref = np.array(mvl), np.array(pref)
hdcells = np.where(mvl > 0.3)[0]
print("cells with MVL>0.3:", len(hdcells))

# ---- Bayesian decoding ----------------------------------------------------
tc = pd.DataFrame(index=C, data={c: ft.curve(units[c].t) for c in hdcells})
sub = units[list(hdcells)]
for bs in [0.1, 0.2, 0.5]:
    dec, prob = nap.decode_1d(tuning_curves=tc, group=sub, ep=sq, bin_size=bs)
    true = hd.restrict(sq).interpolate(dec)
    err = np.degrees(np.abs(hd_lib.angdiff(dec.values, true.values)))
    err = err[~np.isnan(err)]
    print("bin %.1f s: median |error| %.1f deg, mean %.1f, frac<45deg %.2f"
          % (bs, np.median(err), err.mean(), np.mean(err < 45)))

# decoding on held-out half
half = sq.start[0] + (sq.end[0] - sq.start[0]) / 2
e1, e2 = nap.IntervalSet(sq.start[0], half), nap.IntervalSet(half, sq.end[0])
f1 = hd_lib.FastTuning(hd, e1, 60)
tc1 = pd.DataFrame(index=C, data={c: f1.curve(units[c].t) for c in hdcells})
dec, _ = nap.decode_1d(tuning_curves=tc1, group=sub, ep=e2, bin_size=0.2)
true = hd.restrict(e2).interpolate(dec)
err = np.degrees(np.abs(hd_lib.angdiff(dec.values, true.values)))
err = err[~np.isnan(err)]
print("cross-validated (train 1st half, test 2nd): median %.1f deg" % np.median(err))

# ---- cross-environment ----------------------------------------------------
ftt = hd_lib.FastTuning(hd, tri, nb_bins=60)
pref_t, mvl_t = [], []
for c in units.keys():
    m, a = hd_lib.circular_mean_vector(C, ftt.curve(units[c].t))
    mvl_t.append(m); pref_t.append(a)
pref_t, mvl_t = np.array(pref_t), np.array(mvl_t)
dphi = hd_lib.angdiff(pref_t[hdcells], pref[hdcells])
R = np.abs(np.mean(np.exp(1j * dphi)))
print("\nsquare->triangle: mean rotation %.1f deg, concentration of offsets R=%.2f, "
      "circular SD %.1f deg" % (np.degrees(np.angle(np.mean(np.exp(1j * dphi)))), R,
                                np.degrees(np.sqrt(-2 * np.log(R)))))

# ---- sleep correlation structure -----------------------------------------
home = d["epochs"]["home_cage"]
states = {k: d["states"][k].intersect(home) for k in ["rem", "nrem"]}
print("\nhome-cage REM %.0f s, nREM %.0f s" % (states["rem"].tot_length(),
                                               states["nrem"].tot_length()))
offs = hd_lib.angdiff(pref[hdcells][:, None], pref[hdcells][None, :])
iu = np.triu_indices(len(hdcells), 1)

def pairwise(ep, bs):
    cnt = sub.count(bs, ep)
    z = np.asarray(cnt.values, dtype=float)
    return np.corrcoef(z.T)

res = {}
for name, ep, bs in [("wake", sq, 0.5), ("REM", states["rem"], 0.5),
                     ("nREM", states["nrem"], 0.1)]:
    res[name] = pairwise(ep, bs)[iu]
print("corr(wake pairwise, REM pairwise)  = %.3f" % np.corrcoef(res["wake"], res["REM"])[0, 1])
print("corr(wake pairwise, nREM pairwise) = %.3f" % np.corrcoef(res["wake"], res["nREM"])[0, 1])
ao = np.abs(offs[iu])
for name in res:
    near, far = ao < np.pi / 4, ao > 3 * np.pi / 4
    print("  %-5s: mean r near-pref %.3f, opposite-pref %.3f"
          % (name, res[name][near].mean(), res[name][far].mean()))
