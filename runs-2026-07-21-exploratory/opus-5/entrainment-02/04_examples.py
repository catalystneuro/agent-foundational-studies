"""Spike-triggered LFP averages and within-session stability of the phase preference."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import analysis as an
import theta_lib as tl

SESSION = tl.SESSIONS[0]

df = pd.read_csv("results/unit_stats.csv")
d = df[df.session == SESSION].sort_values("mrl", ascending=False)
# Control units are chosen among those with enough spikes for the STA to be meaningful,
# otherwise a handful of clustered spikes produces a spuriously structured average.
locked_ids = [int(u) for u in d[d.n_spikes > 5000].unit[:3]]
unlocked_ids = [int(u) for u in d[d.n_spikes > 5000].sort_values("mrl_z").unit[:2]]

s = an.load_session(SESSION)
units, ep = s["units"], s["run_normal"]

# ---------------------------------------------------------------------------
# Spike-triggered average of the wideband LFP, with a circular-shift null band
# ---------------------------------------------------------------------------
# A spike-triggered average is not by itself evidence of entrainment: units that fire
# in a few clustered bouts average over very few independent theta cycles, so their
# STA oscillates at chance. The shaded band is the 5-95% range of STAs computed after
# circularly shifting the spike train within the running epochs.
sel = nap.TsGroup({u: units[u] for u in locked_ids + unlocked_ids})
sta_pynap = nap.compute_event_triggered_average(s["lfp"], sel, binsize=1 / s["fs"],
                                                window=(-0.25, 0.25), epochs=ep)
lags = np.asarray(sta_pynap.index)

lfp_d, lfp_t0, fs = s["lfp"].d, float(s["lfp"].t[0]), s["fs"]
W = int(0.25 * fs)
rng = np.random.default_rng(0)


def sta_of(times):
    idx = np.rint((times - lfp_t0) * fs).astype(int)
    idx = idx[(idx > W) & (idx < len(lfp_d) - W)]
    seg = np.stack([lfp_d[i - W:i + W] for i in idx])
    m = seg.mean(0) * 1e3
    return m - m.mean()


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
for u, style in [(u, "-") for u in locked_ids] + [(u, "--") for u in unlocked_ids]:
    t = units[u].restrict(ep).t
    obs = sta_of(t)
    tc, total = an._compress_times(t, ep)
    null = np.stack([sta_of(an._decompress_times(np.mod(tc + sh, total), ep))
                     for sh in rng.uniform(1, total - 1, 100)])
    lo, hi = np.percentile(null, [5, 95], axis=0)
    ratio = np.ptp(obs) / np.median(np.ptp(null, axis=1))
    line, = ax.plot(lags[:len(obs)], obs, lw=1.6, ls=style,
                    label=f"u{u}: MRL={float(d.mrl[d.unit == u].iloc[0]):.2f}, "
                          f"STA {ratio:.1f}x chance")
    ax.fill_between(lags[:len(obs)], lo, hi, color=line.get_color(), alpha=0.18, lw=0)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("time from spike (s)")
ax.set_ylabel("LFP (µV)")
ax.set_title("spike-triggered LFP average\n(bands: 5-95% of circular-shift shuffles)", fontsize=10)
ax.legend(fontsize=7, ncol=2)

# ---------------------------------------------------------------------------
# Stability: odd vs even running epochs
# ---------------------------------------------------------------------------
# Splitting the session into first and second halves would confound stability with
# the cooling manipulation and with slow drift, so the split is interleaved instead.
starts, ends = np.asarray(ep.start), np.asarray(ep.end)
ep1 = nap.IntervalSet(starts[0::2], ends[0::2])
ep2 = nap.IntervalSet(starts[1::2], ends[1::2])

ax = axes[1]
u = int(d[d.shuffle_p < 0.05].sort_values("n_spikes", ascending=False).unit.iloc[0])
for e, label, color in [(ep1, "odd epochs", "crimson"), (ep2, "even epochs", "navy")]:
    p = s["phase_at"](units[u].restrict(e).t)
    c, h = an.phase_histogram(p, nbins=24)
    x = np.degrees(np.concatenate([c, c + 2 * np.pi]))
    ax.plot(x, np.concatenate([h, h]) * len(h), color=color, lw=1.8,
            label=f"{label} (MRL={tl.circ_r(p):.3f}, {len(p)} spikes)")
ax.set_xlim(0, 720)
ax.set_xticks([0, 180, 360, 540, 720])
ax.set_xlabel("theta phase (deg)")
ax.set_ylabel("norm. rate")
ax.set_title(f"unit {u}: the phase preference repeats\nacross interleaved epochs", fontsize=10)
ax.legend(fontsize=8)

# Split-half consistency of the preferred phase, pooled over all sessions.
rows = []
for session in tl.SESSIONS:
    ss = s if session == SESSION else an.load_session(session, verbose=False)
    e = ss["run_normal"]
    st, en = np.asarray(e.start), np.asarray(e.end)
    e1, e2 = nap.IntervalSet(st[0::2], en[0::2]), nap.IntervalSet(st[1::2], en[1::2])
    for uid in ss["units"].index:
        p1 = ss["phase_at"](ss["units"][uid].restrict(e1).t)
        p2 = ss["phase_at"](ss["units"][uid].restrict(e2).t)
        if min(len(p1), len(p2)) < tl.MIN_SPIKES // 2:
            continue
        rows.append((tl.circ_mean(p1), tl.circ_mean(p2), tl.circ_r(np.concatenate([p1, p2]))))
rows = np.array(rows)

ax = axes[2]
sc = ax.scatter(np.degrees(rows[:, 0]), np.degrees(rows[:, 1]), c=rows[:, 2],
                cmap="viridis", s=22)
ax.plot([0, 360], [0, 360], "k--", lw=1)
ax.set_xlabel("preferred phase, odd epochs (deg)")
ax.set_ylabel("preferred phase, even epochs (deg)")
diff = np.angle(np.exp(1j * (rows[:, 0] - rows[:, 1])))
strong = rows[:, 2] > np.median(rows[:, 2])
ax.set_title(f"split-half consistency, {len(rows)} units\n"
             f"median |Δphase| = {np.degrees(np.median(np.abs(diff[strong]))):.0f}° "
             f"(better-locked half), {np.degrees(np.median(np.abs(diff))):.0f}° overall", fontsize=10)
fig.colorbar(sc, ax=ax, label="MRL")

fig.tight_layout()
fig.savefig("fig06_sta_and_stability.png", dpi=150)
print("wrote fig06_sta_and_stability.png")
