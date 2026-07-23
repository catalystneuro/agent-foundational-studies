"""Figures for the auditory frequency-tuning analysis of DANDI:000986."""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats

from loader import list_assets, load_session
from analysis import bh_fdr, decode_frequency, RESP_WIN, BASE_WIN, WIN_DUR
from psthlib import psth, perievent_rel_times

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

R = pickle.load(open("results_all_sessions.pkl", "rb"))
FREQS = R[0]["freqs"]
FKHZ = [f"{f/1000:g}" for f in FREQS]
CMAP = cm.viridis(np.linspace(0, 0.92, len(FREQS)))
WIN, BS = (-0.2, 0.4), 0.005

# ---------------------------------------------------------------- pooled arrays
tc = np.concatenate([r["tc"] for r in R])
delta = np.concatenate([r["delta"] for r in R])
base = np.concatenate([r["base_rate"] for r in R])
p_resp = np.concatenate([r["p_resp"] for r in R])
p_tune = np.concatenate([r["p_tune"] for r in R])
bf_idx = np.concatenate([r["bf_idx"] for r in R])
depth = np.concatenate([r["depth"] for r in R])
sparse = np.concatenate([r["sparseness"] for r in R])
reliab = np.concatenate([r["reliability"] for r in R])
sess_id = np.concatenate([[i] * r["n_units"] for i, r in enumerate(R)])
q_resp, q_tune = bh_fdr(p_resp), bh_fdr(p_tune)
responsive = q_resp < 0.01
tuned = responsive & (q_tune < 0.01)
print(f"pooled: {len(tc)} units | responsive {responsive.mean():.1%} | "
      f"frequency-tuned {tuned.mean():.1%} ({tuned.sum()} units)")

# ============================================================ Fig 1: raw data
assets = list_assets()
nwb0, raw0 = load_session(assets[0])
units0 = nwb0["units"]
trials0 = raw0.trials.to_dataframe()
keys0 = list(units0.keys())
rates0 = np.array([len(units0[k]) for k in keys0])
order0 = np.argsort(-rates0)

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.7, 1.5], hspace=0.45)

t0 = trials0.start_time.values[100]
t1 = t0 + 8.0
ax = fig.add_subplot(gs[0])
show = [keys0[i] for i in order0[:70]]
for row, k in enumerate(show):
    st = units0[k].t
    st = st[(st >= t0) & (st <= t1)]
    ax.plot(st - t0, np.full_like(st, row), "|", color="0.25", ms=2.5, mew=0.5)
sel = trials0[(trials0.start_time >= t0) & (trials0.start_time <= t1)]
for _, tr in sel.iterrows():
    ci = int(np.flatnonzero(FREQS == tr.stim_frequency)[0])
    ax.axvspan(tr.start_time - t0, tr.start_time - t0 + tr.stim_duration,
               color=CMAP[ci], alpha=0.85, lw=0)
ax.set_ylabel("unit (sorted by rate)")
ax.set_xlim(0, t1 - t0)
ax.set_title(f"A  Raw spiking during pure-tone presentation "
             f"({raw0.subject.subject_id}, session {raw0.session_id}); "
             f"colored bars = 25 ms tones", loc="left")
handles = [plt.Line2D([], [], color=c, lw=6) for c in CMAP]
ax.legend(handles, [f"{s} kHz" for s in FKHZ], ncol=5, loc="upper right",
          fontsize=7, frameon=False, bbox_to_anchor=(1.0, 1.28))

ax = fig.add_subplot(gs[1])
allsp = np.concatenate([units0[k].t for k in keys0])
edges = np.arange(t0, t1 + 0.01, 0.01)
ax.plot(edges[:-1] - t0, np.histogram(allsp, edges)[0] / 0.01 / len(keys0),
        color="k", lw=0.8)
ax.set_xlim(0, t1 - t0)
ax.set_xlabel("time (s)")
ax.set_ylabel("pop. rate (Hz)")
ax.set_title("B  Population firing rate", loc="left")

ax = fig.add_subplot(gs[2])
pup = nwb0["pupil_diameter"]
run = nwb0["running_speed"]
ax.plot(np.asarray(pup.t) / 60, np.asarray(pup.d).squeeze(),
        color="#7b3294", lw=0.4)
ax2 = ax.twinx()
ax2.plot(np.asarray(run.t) / 60, np.asarray(run.d).squeeze(),
         color="#008837", lw=0.3, alpha=0.6)
ax2.set_ylabel("running speed (cm/s)", color="#008837")
ax2.spines["top"].set_visible(False)
spont = raw0.intervals["spontaneous_blocks"].to_dataframe()
for _, s in spont.iterrows():
    ax.axvspan(s.start_time / 60, s.stop_time / 60, color="0.85", lw=0, zorder=0)
ax.set_xlabel("time (min)")
ax.set_ylabel("pupil diameter (norm.)", color="#7b3294")
ax.set_title("C  Behavioural state across the session "
             "(grey = spontaneous blocks, white = tone blocks)", loc="left")
fig.savefig("fig01_raw_data.png", bbox_inches="tight")
plt.close(fig)

# ==================================================== Fig 2: population PSTHs
pf = np.concatenate([r["pf_psth"] for r in R], axis=1)   # (n_freq, n_units, n_bins)
tvec = R[0]["tvec"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
ax = axes[0]
m = pf.mean(0)
ax.plot(tvec, m.mean(0), "k")
sem = m.std(0, ddof=1) / np.sqrt(m.shape[0])
ax.fill_between(tvec, m.mean(0) - sem, m.mean(0) + sem, color="k", alpha=0.25, lw=0)
ax.axvspan(0, 0.025, color="orange", alpha=0.3, lw=0)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("A  Mean response, all tones", loc="left")

ax = axes[1]
for i in range(len(FREQS)):
    ax.plot(tvec, pf[i].mean(0), color=CMAP[i], label=f"{FKHZ[i]} kHz")
ax.axvspan(0, 0.025, color="orange", alpha=0.3, lw=0)
ax.legend(fontsize=7, frameon=False)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title("B  Split by tone frequency (all units)", loc="left")

ax = axes[2]
pf_t = pf[:, tuned, :]
for i in range(len(FREQS)):
    ax.plot(tvec, pf_t[i].mean(0), color=CMAP[i])
ax.axvspan(0, 0.025, color="orange", alpha=0.3, lw=0)
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("firing rate (Hz)")
ax.set_title(f"C  Frequency-tuned units only (n={tuned.sum()})", loc="left")
fig.tight_layout()
fig.savefig("fig02_population_psth.png", bbox_inches="tight")
plt.close(fig)

# ==================================================== Fig 3: example units
s0 = R[0]
cand = np.flatnonzero((bh_fdr(s0["p_tune"]) < 0.01) & (bh_fdr(s0["p_resp"]) < 0.01))
score = s0["depth"][cand] * np.clip(s0["reliability"][cand], 0, 1)
best = cand[np.argsort(-score)]
# pick four examples that span different best frequencies
chosen, seen = [], set()
for u in best:
    b = s0["bf_idx"][u]
    if b not in seen:
        chosen.append(u)
        seen.add(b)
    if len(chosen) == 4:
        break
onsets0 = trials0.start_time.values
freq0 = trials0.stim_frequency.values

fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.5),
                         gridspec_kw=dict(height_ratios=[1.5, 1, 1], hspace=0.5))
for col, u in enumerate(chosen):
    k = s0["unit_keys"][u]
    st = units0[k].t
    ax = axes[0, col]
    yoff = 0
    for i, f in enumerate(FREQS):
        on = onsets0[freq0 == f][:120]
        rel, tri = perievent_rel_times(st, on, WIN)
        ax.plot(rel, yoff + tri, "|", color=CMAP[i], ms=2, mew=0.5)
        yoff += len(on)
        ax.axhline(yoff, color="0.8", lw=0.5)
    ax.axvspan(0, 0.025, color="orange", alpha=0.25, lw=0)
    ax.set_xlim(*WIN)
    ax.set_ylim(0, yoff)
    ax.set_title(f"unit {k}  (BF = {FKHZ[s0['bf_idx'][u]]} kHz)", fontsize=9)
    if col == 0:
        ax.set_ylabel("trials, grouped by frequency")

    ax = axes[1, col]
    for i, f in enumerate(FREQS):
        on = onsets0[freq0 == f]
        tt, r = psth(st, on, WIN, 0.01)
        ax.plot(tt, r, color=CMAP[i], lw=1)
    ax.axvspan(0, 0.025, color="orange", alpha=0.25, lw=0)
    ax.set_xlim(*WIN)
    ax.set_xlabel("time from onset (s)")
    if col == 0:
        ax.set_ylabel("rate (Hz)")

    ax = axes[2, col]
    ax.errorbar(np.log2(FREQS / 1000), s0["tc"][u], yerr=s0["tc_sem"][u],
                marker="o", color="k", ms=4, lw=1.2, capsize=2)
    ax.axhline(s0["base_rate"][u], color="0.5", ls="--", lw=1)
    ax.set_xticks(np.log2(FREQS / 1000))
    ax.set_xticklabels(FKHZ)
    ax.set_xlabel("tone frequency (kHz)")
    if col == 0:
        ax.set_ylabel("evoked rate (Hz)\n(dashed = baseline)")
fig.suptitle("Example frequency-tuned auditory cortex units "
             f"({raw0.subject.subject_id}, session {raw0.session_id}): "
             "raster, PSTH and frequency tuning curve", y=0.98)
fig.savefig("fig03_example_units.png", bbox_inches="tight")
plt.close(fig)

# ============================================ Fig 4: population tuning structure
d = delta[tuned]
norm = d / np.abs(d).max(1, keepdims=True)
srt = np.lexsort((-norm.max(1), bf_idx[tuned]))
fig = plt.figure(figsize=(11, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.35)

ax = fig.add_subplot(gs[0])
im = ax.imshow(norm[srt], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               interpolation="nearest")
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("frequency-tuned units\n(sorted by best frequency)")
ax.set_title("A  Normalised tuning curves", loc="left")
plt.colorbar(im, ax=ax, label="evoked rate (norm.)")

ax = fig.add_subplot(gs[1])
counts = np.array([[np.sum(bf_idx[tuned & (sess_id == s)] == i)
                    for i in range(len(FREQS))] for s in range(len(R))])
ax.bar(range(len(FREQS)), counts.sum(0), color=CMAP)
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("number of units")
chi = stats.chisquare(counts.sum(0))
ax.set_title(f"B  Best-frequency distribution\n"
             f"$\\chi^2$ vs uniform p = {chi.pvalue:.1e}", loc="left")

ax = fig.add_subplot(gs[2])
n_f = len(FREQS)
aligned = np.full((tuned.sum(), 2 * n_f - 1), np.nan)
for j, (row, b) in enumerate(zip(norm, bf_idx[tuned])):
    aligned[j, (n_f - 1 - b):(2 * n_f - 1 - b)] = row
off = np.arange(-(n_f - 1), n_f)
mu = np.nanmean(aligned, 0)
se = np.nanstd(aligned, 0) / np.sqrt(np.sum(~np.isnan(aligned), 0))
ax.errorbar(off, mu, yerr=se, marker="o", color="crimson", ms=4, lw=1.4)
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("octaves from best frequency")
ax.set_ylabel("evoked rate (norm.)")
ax.set_title("C  BF-aligned mean tuning curve", loc="left")
fig.savefig("fig04_population_tuning.png", bbox_inches="tight")
plt.close(fig)

# ================================================= Fig 5: statistics & reliability
fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
ax = axes[0]
ax.hist(np.log10(np.clip(p_tune, 1e-30, 1)), bins=50, color="0.4")
ax.axvline(np.log10(0.01), color="crimson", ls="--")
ax.set_xlabel("log$_{10}$ p (Kruskal-Wallis\nacross frequencies)")
ax.set_ylabel("units")
ax.set_title("A  Frequency-tuning test", loc="left")

ax = axes[1]
frac_r = np.array([responsive[sess_id == s].mean() for s in range(len(R))])
frac_t = np.array([tuned[sess_id == s].mean() for s in range(len(R))])
x = np.arange(len(R))
ax.bar(x - 0.2, frac_r, 0.4, label="sound-responsive", color="0.6")
ax.bar(x + 0.2, frac_t, 0.4, label="frequency-tuned", color="crimson")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['subject']}-{r['session_id']}" for r in R],
                   rotation=90, fontsize=6)
ax.set_ylabel("fraction of units")
ax.legend(fontsize=7, frameon=False)
ax.set_title("B  Per session (FDR q < 0.01)", loc="left")

ax = axes[2]
ax.hist(reliab[tuned], bins=np.linspace(-1, 1, 41), color="crimson",
        alpha=0.8, label="tuned")
ax.hist(reliab[~responsive], bins=np.linspace(-1, 1, 41), color="0.6",
        alpha=0.7, label="not responsive")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("split-half tuning-curve correlation")
ax.set_ylabel("units")
ax.legend(fontsize=7, frameon=False)
ax.set_title(f"C  Reliability (tuned median\nr = {np.nanmedian(reliab[tuned]):.2f})",
             loc="left")

ax = axes[3]
ax.scatter(base[~tuned], depth[~tuned], s=4, color="0.7", label="other")
ax.scatter(base[tuned], depth[tuned], s=5, color="crimson", label="tuned")
ax.set_xscale("log")
ax.set_xlabel("baseline firing rate (Hz)")
ax.set_ylabel("tuning modulation depth")
ax.legend(fontsize=7, frameon=False)
ax.set_title("D  Depth vs baseline rate", loc="left")
fig.tight_layout()
fig.savefig("fig05_statistics.png", bbox_inches="tight")
plt.close(fig)

# ======================================================= Fig 6: decoding
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
cms = np.stack([r["decode"]["cm"] for r in R]).mean(0)
ax = axes[0]
im = ax.imshow(cms, cmap="magma", vmin=0, vmax=cms.max())
ax.set_xticks(range(len(FREQS)))
ax.set_xticklabels(FKHZ)
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels(FKHZ)
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("true frequency (kHz)")
for i in range(len(FREQS)):
    for j in range(len(FREQS)):
        ax.text(j, i, f"{cms[i, j]:.2f}", ha="center", va="center",
                color="w" if cms[i, j] < cms.max() * 0.6 else "k", fontsize=7)
ax.set_title("A  Mean confusion matrix\n(15 sessions)", loc="left")
plt.colorbar(im, ax=ax, label="P(decoded | true)")

ax = axes[1]
acc = np.array([r["decode"]["acc"] for r in R])
shuf = np.array([r["decode"]["shuffle_mean"] for r in R])
ax.bar(np.arange(len(R)), acc, color="steelblue")
ax.axhline(1 / len(FREQS), color="k", ls="--", lw=1, label="chance (0.20)")
ax.set_xticks(np.arange(len(R)))
ax.set_xticklabels([f"{r['subject']}-{r['session_id']}" for r in R],
                   rotation=90, fontsize=6)
ax.set_ylabel("decoding accuracy")
ax.legend(fontsize=7, frameon=False)
ax.set_title(f"B  Single-trial decoding\nmean = {acc.mean():.2f}", loc="left")

ax = axes[2]
rng = np.random.default_rng(0)
sizes = [1, 2, 5, 10, 20, 50, 100, 200]
curves = []
for r in R[:5]:
    n = r["ev_counts"].shape[1]
    row = []
    for s in sizes:
        if s > n:
            row.append(np.nan)
            continue
        a = [decode_frequency(r["ev_counts"], r["trial_freq"], n_splits=3,
                              subset=rng.choice(n, s, replace=False))["acc"]
             for _ in range(3)]
        row.append(np.mean(a))
    curves.append(row)
curves = np.array(curves)
ax.errorbar(sizes, np.nanmean(curves, 0),
            yerr=np.nanstd(curves, 0) / np.sqrt(np.sum(~np.isnan(curves), 0)),
            marker="o", color="steelblue", ms=4)
ax.axhline(1 / len(FREQS), color="k", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("number of units in decoder")
ax.set_ylabel("decoding accuracy")
ax.set_title("C  Accuracy vs population size\n(5 sessions, mean ± SEM)", loc="left")
fig.tight_layout()
fig.savefig("fig06_decoding.png", bbox_inches="tight")
plt.close(fig)

np.savez("summary_stats.npz", tc=tc, delta=delta, base=base, p_resp=p_resp,
         p_tune=p_tune, q_resp=q_resp, q_tune=q_tune, bf_idx=bf_idx, depth=depth,
         sparseness=sparse, reliability=reliab, sess_id=sess_id, freqs=FREQS,
         acc=np.array([r["decode"]["acc"] for r in R]))
print("figures written")
