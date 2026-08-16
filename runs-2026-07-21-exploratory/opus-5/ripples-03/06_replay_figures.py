"""Stage 6: figures and statistics for the replay result, plus an independent
pairwise-reactivation (explained variance) analysis that does not use decoding.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

import swr_utils as su

SESSION = "Achilles-10252013"
BIN = 0.020

h5 = su.open_session(SESSION)
epochs = su.load_epochs(h5)
units = su.load_units(h5)
pyr = units.getby_category("cell_type")["excitatory"]

pf = np.load("place_fields.npz")
centers, is_place = pf["centers"], pf["is_place"]
ids = pf["unit_ids"][is_place]
spk = pyr[list(ids)]
tmpl = {k: pf[k][:, is_place] for k in ("right", "left")}
field_peak = {k: centers[np.argmax(tmpl[k], axis=0)] for k in tmpl}

df = pd.read_csv("replay_events.csv")
ctrl = pd.read_csv("replay_control.csv")
chk = np.load("decoder_check.npz")
post_npz = np.load("posteriors.npz")
time_npz = np.load("posterior_times.npz")
rip = np.load("ripples.npz")
fs = float(rip["fs"])
raw = np.load("lfp_raw.npy")
filt = np.load("lfp_filt.npy")
_, _, t0_lfp = su.lfp_meta(h5)

# ------------------------------------------------------------------ statistics
tab = [[int(df[(df.epoch == e) & df.significant].shape[0]),
        int(df[(df.epoch == e) & ~df.significant].shape[0])] for e in ("POST", "PRE")]
odds, p_fisher = stats.fisher_exact(tab)
print("POST vs PRE significant replay: OR=%.2f, Fisher p=%.2g" % (odds, p_fisher))
for e in ("PRE", "POST"):
    g = df[df.epoch == e]
    k, n = int(g.significant.sum()), len(g)
    p_bin = stats.binomtest(k, n, 0.05, alternative="greater").pvalue
    print("%s: %d/%d significant (%.1f%%), vs 5%% chance: p=%.2g" % (e, k, n, 100 * k / n, p_bin))
k, n = int(ctrl.significant.sum()), len(ctrl)
print("cell-ID shuffle control: %d/%d (%.1f%%)" % (k, n, 100 * k / n))


# ---------------------------------------------------- independent check: EV/REV
def corr_vector(group, ep, bin_size=0.100):
    c = np.asarray(group.count(bin_size, ep))
    r = np.corrcoef(c.T)
    iu = np.triu_indices(r.shape[0], 1)
    return r[iu]


ripple_ep = nap.IntervalSet(rip["start"], rip["end"])
run_ep = nap.IntervalSet(pf["run_start"], pf["run_end"])
v_run = corr_vector(spk, run_ep)
v_pre = corr_vector(spk, ripple_ep.intersect(epochs["PRE"]))
v_post = corr_vector(spk, ripple_ep.intersect(epochs["POST"]))
ok = np.isfinite(v_run) & np.isfinite(v_pre) & np.isfinite(v_post)
v_run, v_pre, v_post = v_run[ok], v_pre[ok], v_post[ok]


def partial(a, b, c):
    rab, rac, rbc = (np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1], np.corrcoef(b, c)[0, 1])
    return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))


ev = partial(v_run, v_post, v_pre) ** 2
rev = partial(v_run, v_pre, v_post) ** 2
print("explained variance EV=%.3f, reverse REV=%.3f (n=%d cell pairs)" % (ev, rev, len(v_run)))

# =============================================================== example events
sig = df[df.significant].copy()
sig["fwd"] = ((sig.template == "right") & (sig.slope > 0)) | ((sig.template == "left") & (sig.slope < 0))
best = pd.concat([
    sig[(sig.epoch == "POST") & sig.fwd].nlargest(3, "wcorr", keep="first"),
    sig[(sig.epoch == "POST") & ~sig.fwd].reindex(sig[(sig.epoch == "POST") & ~sig.fwd]
                                                  .wcorr.abs().sort_values(ascending=False).index)[:3],
])

fig, axes = plt.subplots(3, 6, figsize=(17, 8.5),
                         gridspec_kw={"height_ratios": [0.55, 1.5, 1.1], "hspace": 0.45, "wspace": 0.3})
for col, (_, ev_row) in enumerate(best.iterrows()):
    key = "%s_%d" % (ev_row.epoch, int(ev_row.event))
    post, tt = post_npz[key], time_npz[key]
    t_rel = (tt - tt[0]) * 1000
    k = ev_row.template
    pad = 0.05
    a, b = ev_row.t0 - pad, ev_row.t0 + ev_row.dur + pad
    sl = slice(int((a - t0_lfp) * fs), int((b - t0_lfp) * fs))
    lt = (np.arange(sl.start, sl.stop) / fs + t0_lfp - ev_row.t0) * 1000

    ax = axes[0, col]
    ax.plot(lt, raw[sl] - raw[sl].mean(), color="k", lw=0.6)
    ax.plot(lt, filt[sl] * 2 - 1500, color="C3", lw=0.6)
    ax.set(xticks=[], yticks=[], xlim=(lt[0], lt[-1]))
    ax.set_title("%s sleep, %s replay of %sward runs\n|r|=%.2f, p=%.3f, %.0f cm/s"
                 % (ev_row.epoch, "forward" if ev_row.fwd else "reverse", ev_row.template,
                    abs(ev_row.wcorr), ev_row.max_p, abs(ev_row.slope)), fontsize=8.5)

    ax = axes[1, col]
    ax.imshow(post, aspect="auto", origin="lower", cmap="magma",
              extent=[t_rel[0] - 10, t_rel[-1] + 10, 0, 160])
    mx = (post / post.sum(0) * centers[:, None]).sum(0)
    ax.plot(t_rel, mx, "o", color="w", ms=3, alpha=0.8)
    fit = np.polyfit(t_rel, mx, 1)
    ax.plot(t_rel, np.polyval(fit, t_rel), color="C0", lw=1.6)
    ax.set(xlim=(lt[0], lt[-1]), ylim=(0, 160))
    if col == 0:
        ax.set_ylabel("decoded position (cm)")

    ax = axes[2, col]
    iset = nap.IntervalSet(ev_row.t0 - pad, ev_row.t0 + ev_row.dur + pad)
    sub = spk.restrict(iset)
    order = np.argsort(field_peak[k])
    for row, u in enumerate(np.array(list(spk.keys()))[order]):
        s = (np.asarray(sub[u].index) - ev_row.t0) * 1000
        if len(s):
            ax.plot(s, np.full_like(s, row), "|", color="k", ms=3, mew=0.8)
    ax.axvspan(0, ev_row.dur * 1000, color="C3", alpha=0.10)
    ax.set(xlim=(lt[0], lt[-1]), ylim=(-1, len(spk)))
    ax.set_xlabel("time (ms)")
    if col == 0:
        ax.set_ylabel("cell (sorted by field position)")
fig.suptitle("Replay of the linear track inside sharp-wave ripples (post-task sleep)\n"
             "top: raw + ripple-filtered LFP;  middle: posterior P(position | spikes);  bottom: place-cell raster",
             fontsize=11)
fig.savefig("fig06_replay_examples.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ==================================================================== summary ==
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
err = np.concatenate([chk["err_right"], chk["err_left"]])
ax.hist(err, bins=np.arange(0, 160, 4), color="C0")
ax.axvline(np.median(err), color="k", ls="--")
ax.axvline(chk["chance"], color="C3", ls=":")
ax.text(np.median(err) + 4, ax.get_ylim()[1] * 0.9, "median %.1f cm" % np.median(err), fontsize=9)
ax.text(float(chk["chance"]) + 4, ax.get_ylim()[1] * 0.6, "chance", color="C3", fontsize=9)
ax.set(xlabel="decoding error while running (cm)", ylabel="count",
       title="The template decodes real position\n(250 ms bins, MAZE epoch)")

ax = fig.add_subplot(gs[0, 1])
labels, vals, ns = [], [], []
for e in ("PRE", "POST"):
    g = df[df.epoch == e]
    labels.append("%s sleep" % e)
    vals.append(100 * g.significant.mean())
    ns.append(len(g))
labels.append("cell-ID\nshuffle")
vals.append(100 * ctrl.significant.mean())
ns.append(len(ctrl))
bars = ax.bar(labels, vals, color=["0.6", "C0", "0.8"])
ax.axhline(5, color="C3", ls="--", label="chance (α=0.05)")
for b, v, n in zip(bars, vals, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, "%.1f%%\n(n=%d)" % (v, n), ha="center", fontsize=8)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="% of ripple events with significant replay",
       title="Replay is specific to sleep AFTER the run\nFisher p=%.1g" % p_fisher, ylim=(0, max(vals) * 1.35))

ax = fig.add_subplot(gs[0, 2])
bins = np.linspace(0, 1, 26)
for name, v, c in [("POST", df[df.epoch == "POST"].wcorr.abs(), "C0"),
                   ("PRE", df[df.epoch == "PRE"].wcorr.abs(), "0.5"),
                   ("cell-ID shuffle", ctrl.wcorr.abs(), "C3")]:
    ax.hist(v, bins=bins, histtype="step", density=True, lw=1.8, color=c, label=name)
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="|weighted correlation|", ylabel="density", title="Sequence scores by epoch")

ax = fig.add_subplot(gs[1, 0])
ax.hist(np.abs(sig.slope) / 100, bins=np.arange(0, 9, 0.4), color="C0")
ax.axvline(chk["run_speed"] / 100, color="C3", ls="--")
ax.annotate("running speed %.1f m/s" % (chk["run_speed"] / 100), xy=(chk["run_speed"] / 100, 0),
            xytext=(2.0, ax.get_ylim()[1] * 1.02), color="C3", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="C3", lw=1))
ax.set(xlabel="replay speed (m/s)", ylabel="count", xlim=(0, 9),
       title="Replay is time-compressed\nmedian %.1f m/s = %.0fx running"
             % (np.median(np.abs(sig.slope)) / 100, np.median(np.abs(sig.slope)) / float(chk["run_speed"])))

ax = fig.add_subplot(gs[1, 1])
cnt = [int((sig.epoch == e).sum() and ((sig.epoch == e) & sig.fwd).sum()) for e in ("PRE", "POST")]
rev_cnt = [int(((sig.epoch == e) & ~sig.fwd).sum()) for e in ("PRE", "POST")]
x = np.arange(2)
ax.bar(x - 0.18, cnt, 0.36, label="forward", color="C0")
ax.bar(x + 0.18, rev_cnt, 0.36, label="reverse", color="C1")
ax.set_xticks(x)
ax.set_xticklabels(["PRE", "POST"])
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="significant events", title="Forward and reverse replay\nboth occur (PRE counts are at chance)")

ax = fig.add_subplot(gs[1, 2])
ax.bar(["EV\n(POST | RUN, PRE)", "REV\n(PRE | RUN, POST)"], [ev, rev], color=["C0", "0.6"])
ax.set(ylabel="explained variance", title="Pairwise reactivation, no decoding\n%d cell pairs" % len(v_run))
for i, v in enumerate([ev, rev]):
    ax.text(i, v + 0.002, "%.3f" % v, ha="center", fontsize=9)

fig.suptitle("Hippocampal replay during sharp-wave ripples, %s (DANDI:000044)" % SESSION)
fig.savefig("fig07_replay_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig06, fig07")

with open("results_summary.txt", "w") as f:
    f.write("session: %s\n" % SESSION)
    f.write("ripples: %d\n" % len(rip["peak_t"]))
    f.write("place cells: %d\n" % len(ids))
    f.write("decoder median error: %.1f cm (chance %.1f cm)\n" % (np.median(err), chk["chance"]))
    for e in ("PRE", "POST"):
        g = df[df.epoch == e]
        f.write("%s: %d/%d significant (%.1f%%)\n" % (e, g.significant.sum(), len(g),
                                                      100 * g.significant.mean()))
    f.write("cell-ID shuffle: %.1f%% (n=%d)\n" % (100 * ctrl.significant.mean(), len(ctrl)))
    f.write("Fisher POST vs PRE: OR=%.2f p=%.3g\n" % (odds, p_fisher))
    f.write("replay speed median %.0f cm/s (run %.0f cm/s)\n" % (np.median(np.abs(sig.slope)), chk["run_speed"]))
    f.write("EV=%.3f REV=%.3f\n" % (ev, rev))
