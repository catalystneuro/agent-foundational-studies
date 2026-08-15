"""Medial-septal cooling: a causal manipulation of the theta rhythm.

In this dandiset the septum is cooled on a subset of trials, which slows the theta
oscillation. That gives an internal control for the entrainment measurement: if the
spike-phase distributions really track the LFP theta rhythm, they should survive a
change in its frequency.

Comparing cooled against normal epochs naively is not safe. Cooled running time is
shorter than normal running time, and the mean resultant length is biased upward both
by small spike counts and by short observation windows (a preferred phase that drifts
slowly over the session averages away over a long window but not a short one). Every
comparison here is therefore matched on both: the normal-condition statistic is
computed inside contiguous blocks of normal running epochs whose total duration equals
the cooled duration, and then subsampled to the same spike count.
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy.stats import wilcoxon
from tqdm import tqdm

import analysis as an
import theta_lib as tl

N_MATCH = 20
rng = np.random.default_rng(0)


def contiguous_block(ep, duration, rng):
    """A contiguous run of intervals from ``ep`` with total length >= ``duration``."""
    st, en = np.asarray(ep.start), np.asarray(ep.end)
    durs = en - st
    cum = np.concatenate([[0], np.cumsum(durs)])
    total = cum[-1]
    if total <= duration:
        return ep
    offset = rng.uniform(0, total - duration)
    i0 = np.searchsorted(cum, offset, side="right") - 1
    i1 = np.searchsorted(cum, offset + duration, side="right")
    return nap.IntervalSet(st[i0:i1], en[i0:i1])


rows, freqs = [], []
for session in tl.SESSIONS:
    s = an.load_session(session)
    cooled, normal = s["run_cooled"], s["run_normal"]
    if cooled.tot_length() < 60:
        continue
    d_cool = float(cooled.tot_length())

    freqs.append(dict(session=session,
                      normal=an.cycle_frequency(s["phase"], normal, s["fs"]),
                      cooled=an.cycle_frequency(s["phase"], cooled, s["fs"]),
                      env_normal=float(np.median(s["env"].restrict(normal).d)),
                      env_cooled=float(np.median(s["env"].restrict(cooled).d))))

    for uid in tqdm(list(s["units"].index), desc=f"{session} cooling"):
        t_cool = s["units"][uid].restrict(cooled).t
        if len(t_cool) < tl.MIN_SPIKES:
            continue
        ph_cool_all = s["phase_at"](t_cool)
        mrl_c, mrl_n = [], []
        for _ in range(N_MATCH):
            block = contiguous_block(normal, d_cool, rng)
            t_norm = s["units"][uid].restrict(block).t
            n = min(len(t_norm), len(t_cool))
            if n < tl.MIN_SPIKES // 2:
                continue
            mrl_n.append(tl.circ_r(s["phase_at"](rng.choice(t_norm, n, replace=False))))
            mrl_c.append(tl.circ_r(rng.choice(ph_cool_all, n, replace=False)))
        if not mrl_n:
            continue
        ph_norm_all = s["phase_at"](s["units"][uid].restrict(normal).t)
        rows.append(dict(session=session, unit=int(uid),
                         n_cooled=len(t_cool), n_normal=len(ph_norm_all),
                         rate_cooled=len(t_cool) / d_cool,
                         rate_normal=len(ph_norm_all) / float(normal.tot_length()),
                         mrl_cooled_matched=float(np.mean(mrl_c)),
                         mrl_normal_matched=float(np.mean(mrl_n)),
                         mrl_cooled=tl.circ_r(ph_cool_all), mrl_normal=tl.circ_r(ph_norm_all),
                         pref_cooled=tl.circ_mean(ph_cool_all), pref_normal=tl.circ_mean(ph_norm_all),
                         rayleigh_p_cooled=tl.rayleigh(ph_cool_all)[2]))

df = pd.DataFrame(rows)
df.to_csv("results/cooling_units.csv", index=False)
with open("results/cooling_freq.pkl", "wb") as f:
    pickle.dump(freqs, f)
print(f"{len(df)} units with enough spikes in both conditions")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))

ax = axes[0, 0]
fn = np.concatenate([f["normal"] for f in freqs])
fc = np.concatenate([f["cooled"] for f in freqs])
bins = np.linspace(4, 12, 60)
ax.hist(fn, bins=bins, density=True, color="0.4", alpha=0.75, label=f"normal ({np.median(fn):.2f} Hz)")
ax.hist(fc, bins=bins, density=True, color="steelblue", alpha=0.75, label=f"cooled ({np.median(fc):.2f} Hz)")
ax.set_xlabel("theta frequency (Hz, per cycle)")
ax.set_ylabel("density")
ax.set_title("septal cooling slows theta", fontsize=10)
ax.legend(fontsize=8)

ax = axes[0, 1]
for f in freqs:
    ax.plot([0, 1], [np.median(f["normal"]), np.median(f["cooled"])], "-o", color="0.3", ms=5)
ax.set_xticks([0, 1], ["normal", "cooled"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylabel("median theta frequency (Hz)")
delta = np.mean([np.median(f["cooled"]) - np.median(f["normal"]) for f in freqs])
ax.set_title(f"per session (n={len(freqs)}): Δ = {delta:+.2f} Hz", fontsize=10)

ax = axes[0, 2]
for f in freqs:
    ax.plot([0, 1], [f["env_normal"] * 1e3, f["env_cooled"] * 1e3], "-o", color="0.3", ms=5)
ax.set_xticks([0, 1], ["normal", "cooled"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylabel("median theta envelope (µV)")
ax.set_title("theta amplitude under cooling", fontsize=10)

ax = axes[0, 3]
ax.scatter(df.mrl_normal_matched, df.mrl_cooled_matched, s=16, c="crimson", alpha=0.7)
lim = [0, max(df.mrl_normal_matched.max(), df.mrl_cooled_matched.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim), ax.set_ylim(lim)
stat, pval = wilcoxon(df.mrl_normal_matched, df.mrl_cooled_matched)
ax.set_xlabel("MRL, normal (matched)")
ax.set_ylabel("MRL, cooled (matched)")
frac = float((df.mrl_cooled_matched > df.mrl_normal_matched).mean())
ax.set_title(f"locking under cooling\n{frac:.0%} of units above unity, Wilcoxon p={pval:.2g}", fontsize=10)

ax = axes[1, 0]
strong = df[df.rayleigh_p_cooled < 0.01]
ax.scatter(np.degrees(strong.pref_normal), np.degrees(strong.pref_cooled), s=18, c="crimson", alpha=0.75)
ax.plot([0, 360], [0, 360], "k--", lw=1)
ax.set_xlabel("preferred phase, normal (deg)")
ax.set_ylabel("preferred phase, cooled (deg)")
dphi = np.angle(np.exp(1j * (strong.pref_cooled - strong.pref_normal)))
ax.set_title(f"phase preference is preserved\nmedian |Δ| = {np.degrees(np.median(np.abs(dphi))):.0f}° "
             f"(n={len(strong)})", fontsize=10)

ax = axes[1, 1]
ax.hist(np.degrees(dphi), bins=np.linspace(-180, 180, 37), color="steelblue", alpha=0.85)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Δ preferred phase, cooled − normal (deg)")
ax.set_ylabel("units")
ax.set_title("phase shift induced by cooling", fontsize=10)

ax = axes[1, 2]
ax.scatter(df.rate_normal, df.rate_cooled, s=16, c="0.4", alpha=0.75)
lim = [df[["rate_normal", "rate_cooled"]].min().min() * 0.8, df[["rate_normal", "rate_cooled"]].max().max() * 1.2]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xscale("log"), ax.set_yscale("log")
ax.set_xlabel("firing rate, normal (Hz)")
ax.set_ylabel("firing rate, cooled (Hz)")
ax.set_title(f"firing rates\nmedian ratio {np.median(df.rate_cooled / df.rate_normal):.2f}", fontsize=10)

ax = axes[1, 3]
per_session = df.groupby("session")[["mrl_normal_matched", "mrl_cooled_matched"]].median()
x = np.arange(len(per_session))
ax.bar(x - 0.2, per_session.mrl_normal_matched, 0.4, color="0.5", label="normal")
ax.bar(x + 0.2, per_session.mrl_cooled_matched, 0.4, color="steelblue", label="cooled")
ax.set_xticks(x, [i.split("-")[1] + "\n" + i.split("-")[2] for i in per_session.index], fontsize=8)
ax.set_ylabel("median matched MRL")
ax.set_title("per-session medians", fontsize=10)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("fig07_cooling.png", dpi=150)
print("wrote fig07_cooling.png")

pd.Series(dict(
    n_units=len(df),
    freq_normal=float(np.median(fn)), freq_cooled=float(np.median(fc)), freq_delta=float(delta),
    env_ratio=float(np.mean([f["env_cooled"] / f["env_normal"] for f in freqs])),
    mrl_normal_matched=float(df.mrl_normal_matched.median()),
    mrl_cooled_matched=float(df.mrl_cooled_matched.median()),
    frac_above_unity=frac, wilcoxon_p=float(pval),
    median_abs_phase_shift_deg=float(np.degrees(np.median(np.abs(dphi)))),
    rate_ratio=float(np.median(df.rate_cooled / df.rate_normal)),
)).to_json("results/cooling_summary.json", indent=1)
print(open("results/cooling_summary.json").read())
