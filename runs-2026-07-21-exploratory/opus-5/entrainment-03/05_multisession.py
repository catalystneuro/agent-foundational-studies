"""Stage 5: repeat the phase-locking analysis across all five sessions and pool."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import theta_utils as tu

MIN_SPIKES = 100
N_SHUFFLES = 200
SPEED_THRESH = 10.0


def analyse_session(name, url, rng):
    h = tu.open_session(url)
    meta = tu.load_metadata(h)
    units = meta["units"]
    maze = meta["maze_epoch"]

    # theta channel: best theta/delta ratio over 100 s in the middle of the maze
    t0 = maze.start[0] + 0.5 * maze.tot_length()
    ratios = [tu.theta_delta_ratio(tu.load_lfp_channel(h, ch, t0, t0 + 100).d)
              for ch in tqdm(range(meta["n_lfp_channels"]),
                             desc=f"{name}: channels", leave=False)]
    best_ch = int(np.argmax(ratios))

    speed = tu.compute_speed(meta["position"].restrict(maze),
                             rate=meta["position_rate"])
    run = speed.threshold(SPEED_THRESH).time_support.drop_short_intervals(1.0)
    phase, amp, raw, kept = tu.phase_by_interval(h, best_ch, run, max_total=1500.0)
    lookup = tu.PhaseLookup(phase)

    rows = []
    cell_type = units.cell_type.values
    for i, uid in enumerate(units.index):
        ph = lookup(units[uid].t)
        n = len(ph)
        if n < MIN_SPIKES:
            continue
        r, z, p = tu.rayleigh_test(ph)
        null = tu.jitter_null_mrl(units[uid].t, lookup, n_shuffles=N_SHUFFLES,
                                  rng=rng)
        rows.append(dict(session=name, unit=int(uid), cell_type=cell_type[i],
                         channel=best_ch, theta_delta=float(np.max(ratios)),
                         n_spikes=n, rate=float(np.asarray(units.rates)[i]),
                         mrl=r, ppc=tu.ppc(ph), pref=tu.circ_mean(ph), p_rayleigh=p,
                         mrl_null_mean=null.mean(),
                         p_shuffle=(np.sum(null >= r) + 1) / (N_SHUFFLES + 1)))
    df = pd.DataFrame(rows)
    df["sig"] = tu.benjamini_hochberg(df["p_rayleigh"].values)
    print(f"{name}: ch{best_ch} theta/delta={max(ratios):.1f}, "
          f"{run.tot_length():.0f}s running, {len(df)} units, "
          f"{df.sig.mean()*100:.0f}% locked")
    return df


rng = np.random.default_rng(2)
all_df = pd.concat([analyse_session(k, v, rng)
                    for k, v in tqdm(tu.SESSIONS.items(), desc="sessions")],
                   ignore_index=True)
all_df.to_csv("phase_locking_all_sessions.csv", index=False)

# ------------------------------------------------------------------ summary
print("\n=== pooled across %d sessions ===" % all_df.session.nunique())
print("units: %d (%d excitatory, %d inhibitory)"
      % (len(all_df), (all_df.cell_type == "excitatory").sum(),
         (all_df.cell_type == "inhibitory").sum()))
print("significantly theta-locked (Rayleigh, FDR q<0.05 within session): "
      "%d/%d = %.0f%%" % (all_df.sig.sum(), len(all_df), 100 * all_df.sig.mean()))
print("exceeds jitter null at p<0.05: %d/%d"
      % ((all_df.p_shuffle < 0.05).sum(), len(all_df)))
for ct in ["excitatory", "inhibitory"]:
    m = all_df.cell_type == ct
    mu = np.degrees(tu.circ_mean(all_df.loc[m & all_df.sig, "pref"]))
    R = tu.circ_r(all_df.loc[m & all_df.sig, "pref"])
    print(f"  {ct:11s} n={m.sum():3d}  locked={100*all_df.loc[m,'sig'].mean():3.0f}%"
          f"  median MRL={all_df.loc[m,'mrl'].median():.3f}"
          f"  mean preferred phase={mu:6.1f} deg (R={R:.2f})")

# ------------------------------------------------------------------ figure 8
fig = plt.figure(figsize=(13.5, 7.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.33)
sessions = list(all_df.session.unique())

ax = fig.add_subplot(gs[0, 0])
frac = [100 * all_df.loc[all_df.session == s, "sig"].mean() for s in sessions]
ax.bar(range(len(sessions)), frac, color="C0")
for i, f in enumerate(frac):
    ax.text(i, f + 1.5, f"{f:.0f}%", ha="center", fontsize=8)
ax.set_xticks(range(len(sessions)))
ax.set_xticklabels(sessions, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("% units significantly locked"); ax.set_ylim(0, 110)
ax.set_title("phase locking replicates in every session")

ax = fig.add_subplot(gs[0, 1])
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    v = np.sort(all_df.loc[all_df.cell_type == ct, "mrl"].values)
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c,
            label=f"{ct} (n={len(v)})")
v = np.sort(all_df["mrl_null_mean"].values)
ax.plot(v, np.arange(1, len(v) + 1) / len(v), color="k", ls="--",
        label="jitter null")
ax.set_xlabel("mean resultant length"); ax.set_ylabel("cumulative fraction")
ax.legend(fontsize=8); ax.set_title("pooled locking strength")

ax = fig.add_subplot(gs[0, 2], projection="polar")
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = (all_df.cell_type == ct) & all_df.sig
    cnt, edges = np.histogram(all_df.loc[m, "pref"], bins=24, range=(0, 2 * np.pi))
    ax.bar(edges[:-1] + np.pi / 24, cnt, width=2 * np.pi / 24, color=c, alpha=0.6,
           label=ct)
    ax.annotate("", xy=(tu.circ_mean(all_df.loc[m, "pref"]), cnt.max()),
                xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
ax.set_theta_zero_location("E")
ax.set_rlabel_position(285)
ax.tick_params(axis="y", labelsize=7)
ax.set_title("preferred phase, all locked units\n(0 deg = LFP theta peak)",
             fontsize=10, pad=22)
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.3, -0.1))

ax = fig.add_subplot(gs[1, 0])
for i, s in enumerate(sessions):
    m = all_df.session == s
    ax.scatter(np.full(m.sum(), i) + np.random.uniform(-0.18, 0.18, m.sum()),
               all_df.loc[m, "mrl"], s=10,
               c=["C3" if t == "inhibitory" else "C0"
                  for t in all_df.loc[m, "cell_type"]])
    ax.plot([i - 0.3, i + 0.3], [all_df.loc[m, "mrl"].median()] * 2, "k-", lw=2)
ax.set_xticks(range(len(sessions)))
ax.set_xticklabels(sessions, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("MRL"); ax.set_title("per-session distributions (black = median)")

ax = fig.add_subplot(gs[1, 1])
ax.scatter(all_df["mrl_null_mean"], all_df["mrl"], s=10,
           c=["C3" if t == "inhibitory" else "C0" for t in all_df.cell_type])
lim = [0, all_df["mrl"].max() * 1.05]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(0, 0.09); ax.set_ylim(*lim)
ax.set_xlabel("mean MRL of jittered spikes"); ax.set_ylabel("observed MRL")
ax.set_title("observed vs jitter null (pooled)")

ax = fig.add_subplot(gs[1, 2])
for ct, c in [("excitatory", "C0"), ("inhibitory", "C3")]:
    m = (all_df.cell_type == ct) & all_df.sig
    for s, mark in zip(sessions, "os^Dv"):
        mm = m & (all_df.session == s)
        ax.scatter(np.degrees(all_df.loc[mm, "pref"]), all_df.loc[mm, "mrl"],
                   s=16, color=c, marker=mark,
                   label=s if ct == "excitatory" else None)
ax.set_xlabel("preferred phase (deg)"); ax.set_ylabel("MRL")
ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 90))
ax.legend(fontsize=6.5, loc="upper right")
ax.set_title("phase vs strength, by session")
for a in fig.axes:
    if a.name != "polar":
        a.spines[["top", "right"]].set_visible(False)
fig.suptitle("CA1 theta phase entrainment during running, five sessions from "
             "DANDI:000044", y=0.99)
fig.savefig("fig08_multisession_summary.png", dpi=150, bbox_inches="tight")
print("\nsaved fig08_multisession_summary.png and phase_locking_all_sessions.csv")
