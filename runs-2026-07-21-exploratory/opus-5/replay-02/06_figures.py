"""Stage 6: publication-style figures for the replay analysis."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import pynapple as nap
import pipeline as pl

SESSION = "Achilles_10252013"
res = pickle.load(open(f"cache/session_{SESSION}.pkl", "rb"))
centers = res["centers"]
d = pl.load_session(SESSION)
pc_units = d["units"][list(res["pc_ids"])]
post = res["events"]["POST"]; pre = res["events"]["PRE"]; ctrl = res["events"]["CONTROL"]

# ---------------------------------------------------------------- FIG A: example replay events
sig = post[post.sig].copy()
sig["score"] = sig.wcorr.abs()
top = sig.sort_values("score", ascending=False).head(12)
pbe = nap.IntervalSet(start=res["pbe"][0], end=res["pbe"][1])

fig = plt.figure(figsize=(15, 10.5))
gs = GridSpec(4, 6, figure=fig, hspace=.75, wspace=.35)
for j, (_, r) in enumerate(top.iterrows()):
    row, col = divmod(j, 3)
    P = res["posteriors"][int(r.event)]
    T = P.shape[0]
    tms = np.arange(T + 1) * pl.BIN * 1e3
    ax = fig.add_subplot(gs[row, 2 * col])
    ax.pcolormesh(tms, np.r_[centers - 0.016, centers[-1] + 0.016], P.T, cmap="magma", shading="auto")
    tt = np.arange(T) * pl.BIN
    com = P @ centers
    ax.plot(tt * 1e3 + 10, com, "o-", color="#4cc9f0", ms=2.5, lw=1)
    ax.set_title(f"r={r.wcorr:+.2f}  {abs(r.slope):.1f} m/s\n{int(r.ncells)} cells, {int(r.dur*1e3)} ms",
                 fontsize=8, pad=3)
    ax.set_ylim(0, 1.6)
    if col == 0:
        ax.set_ylabel("position (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    # spike raster ordered by place-field peak
    axr = fig.add_subplot(gs[row, 2 * col + 1])
    tcbest = res["tc"][r.template]
    order = np.argsort(np.argmax(tcbest, axis=0))
    for kk, ui in enumerate(order):
        sp = pc_units[res["pc_ids"][ui]].get(r.start, r.end)
        if len(sp):
            axr.plot((sp.t - r.start) * 1e3, np.full(len(sp), kk), "|", ms=3, mew=.7, color="k")
    axr.set_ylim(-1, len(order)); axr.set_xlim(0, r.dur * 1e3)
    axr.set_title("cells by field", fontsize=7, pad=3)
    axr.tick_params(labelsize=7)
    if row == 3:
        ax.set_xlabel("time (ms)", fontsize=8); axr.set_xlabel("time (ms)", fontsize=8)
fig.suptitle(f"{SESSION}: decoded position during sharp-wave ripples in POST-run sleep\n"
             "left = posterior P(position | spikes) in 20 ms bins, right = spikes ordered by place-field location",
             fontsize=12)
plt.savefig("figures/04_replay_examples.png", dpi=140, bbox_inches="tight")
print("saved figures/04_replay_examples.png")

# ---------------------------------------------------------------- FIG B: replay statistics
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(3, 3, figure=fig, hspace=.55, wspace=.32)

ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 40)
for df, lab, c in [(post, "POST sleep", "#2a9d8f"), (pre, "PRE sleep", "#8ecae6"),
                   (ctrl, "cell-ID shuffled template", "#adb5bd")]:
    ax.hist(df.wcorr.abs(), bins=bins, density=True, histtype="step", lw=1.8, color=c, label=lab)
ax.set(xlabel="|weighted correlation|", ylabel="density", title="Replay score distribution")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
labs, fr, lo, hi, ns = [], [], [], [], []
for df, lab in [(post, "POST"), (pre, "PRE"), (ctrl, "control")]:
    k, n = int(df.sig.sum()), len(df)
    ci = stats.binomtest(k, n).proportion_ci(0.95)
    labs.append(f"{lab}\nn={n}"); fr.append(100 * k / n); lo.append(100 * ci.low); hi.append(100 * ci.high); ns.append(n)
ax.bar(labs, fr, color=["#2a9d8f", "#8ecae6", "#adb5bd"])
ax.errorbar(labs, fr, yerr=[np.array(fr) - lo, np.array(hi) - np.array(fr)], fmt="none", ecolor="k", capsize=4)
ax.axhline(5, color="r", ls="--", lw=1, label="nominal 5%")
ax.set(ylabel="% events with significant replay", title="Significant replay events", ylim=(0, 11.5))
ax.legend(fontsize=7, loc="lower right")
p_bin = stats.binomtest(int(post.sig.sum()), len(post), ctrl.sig.mean(), alternative="greater").pvalue
ax.text(.03, .86, f"POST vs control\np = {p_bin:.1e}", transform=ax.transAxes, fontsize=7, va="top")

ax = fig.add_subplot(gs[0, 2])
sp = post[post.sig].speed
ax.hist(sp, bins=np.linspace(0, 25, 40), color="#2a9d8f")
ax.axvline(res["mean_run_speed"], color="r", lw=1.5)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.text(res["mean_run_speed"] + .6, ax.get_ylim()[1] * .80, f"running\n{res['mean_run_speed']:.2f} m/s",
        color="r", fontsize=7)
ax.set(xlabel="replay speed (m/s)", ylabel="events",
       title=f"Virtual speed of replayed trajectories\nmedian {sp.median():.1f} m/s "
             f"({sp.median()/res['mean_run_speed']:.0f}x running)")

ax = fig.add_subplot(gs[1, 0])
fwd = (post[post.sig].wcorr > 0).sum(); rev = (post[post.sig].wcorr < 0).sum()
ax.bar(["forward\n(r > 0)", "reverse\n(r < 0)"], [fwd, rev], color=["#e76f51", "#264653"])
pb = stats.binomtest(int(fwd), int(fwd + rev)).pvalue
ax.set(ylabel="significant events", title=f"Trajectory direction\nbinomial p = {pb:.2f}")

ax = fig.add_subplot(gs[1, 1])
s = post[post.sig]
ax.scatter(s.start_pos, s.end_pos, c=np.sign(s.wcorr), cmap="coolwarm", s=14, alpha=.8)
ax.plot([0, 1.6], [0, 1.6], "k--", lw=.8)
ax.set(xlabel="decoded start position (m)", ylabel="decoded end position (m)",
       title="Replayed trajectory endpoints")

ax = fig.add_subplot(gs[1, 2])
ax.hist(s.span * 100, bins=30, color="#2a9d8f")
ax.set(xlabel="path length covered (cm)", ylabel="events",
       title=f"Extent of replayed trajectories\nmedian {s.span.median()*100:.0f} cm of a 160 cm track")

ax = fig.add_subplot(gs[2, 0])
edges = [5, 10, 15, 20, 25, 60]
frac_p, frac_c, xs = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    mp = (post.ncells >= a) & (post.ncells < b)
    mc = (ctrl.ncells >= a) & (ctrl.ncells < b)
    if mp.sum() > 20:
        frac_p.append(100 * post.sig[mp].mean()); frac_c.append(100 * ctrl.sig[mc].mean())
        xs.append(f"{a}-{b}")
ax.plot(xs, frac_p, "o-", color="#2a9d8f", label="POST")
ax.plot(xs, frac_c, "o-", color="#adb5bd", label="control")
ax.set(xlabel="active cells in event", ylabel="% significant", title="Detection vs. event size")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[2, 1])
ax.plot(res["rip_trig"]["t"] * 1e3, res["rip_trig"]["raw"], "k")
ax.set(xlabel="ms from ripple peak", ylabel="uV", title="Ripple-triggered LFP average")

ax = fig.add_subplot(gs[2, 2])
e = res["example_ripple"]
ax.plot(e["t"], e["raw"], "k", lw=.8, label="raw")
ax.plot(e["t"], e["filt"] - 400, color="#e63946", lw=.8, label="140-230 Hz")
ax.axvspan(e["start"], e["end"], color="#ffb703", alpha=.3)
ax.set(xlabel="time from ripple peak (s)", title="Example sharp-wave ripple", yticks=[])
ax.legend(fontsize=7)

fig.suptitle(f"{SESSION}: replay statistics", fontsize=13)
plt.savefig("figures/05_replay_statistics.png", dpi=140, bbox_inches="tight")
print("saved figures/05_replay_statistics.png")
print(f"POST {post.sig.sum()}/{len(post)} ({100*post.sig.mean():.1f}%), "
      f"PRE {pre.sig.sum()}/{len(pre)} ({100*pre.sig.mean():.1f}%), "
      f"CTRL {ctrl.sig.sum()}/{len(ctrl)} ({100*ctrl.sig.mean():.1f}%), p={p_bin:.2e}")
