# %% [markdown]
# # Decoding upcoming decision bias from pre-stimulus neural activity
#
# **Question.** In a perceptual decision task, can the choice an animal is *about to
# make* be read out from neural population activity recorded *before the stimulus
# appears*? Such a pre-stimulus signal cannot be driven by the sensory evidence and
# instead reflects the animal's internal **decision bias**: its prior expectation about
# which choice is likely to be correct.
#
# **Dataset.** International Brain Laboratory (IBL) ephys data,
# [DANDI:000149](https://dandiarchive.org/dandiset/000149). Head-fixed mice detect a
# visual grating of varying contrast on the left or right and report its side by
# turning a wheel. Crucially, the task uses **block priors**: within a block of trials
# the stimulus appears on the left with probability 0.2, 0.5, or 0.8, which induces a
# measurable behavioral bias. On **0%-contrast trials** no visual evidence is presented
# at all, so the choice there is a pure readout of bias. The task enforces a
# ~0.4-0.7 s pre-stimulus "quiescence" period in which the wheel must be held still,
# giving a clean, movement-free pre-stimulus window.
#
# **Approach.**
# 1. Verify the behavioral bias (psychometric curves by block prior).
# 2. Decode the upcoming choice from pre-stimulus population spike counts
#    (window [-0.4, 0] s relative to stimulus onset) using cross-validated LDA,
#    with permutation null distributions, on all trials and on 0%-contrast trials.
# 3. Decode the block prior itself (the sustained bias state) from the same window.
# 4. Ask whether the neural axis that encodes the block bias *generalizes* to predict
#    choice on ambiguous (0%-contrast) trials (cross-decoding).
# 5. Measure the time course of choice information with a sliding window.
# 6. Identify single units carrying pre-stimulus choice information.
#
# All data are streamed from the DANDI Archive with LINDI (no full downloads).

# %% [markdown]
# ## Setup

# %%
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score

rng = np.random.default_rng(0)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# %% [markdown]
# ## Stream four IBL sessions from DANDI:000149
#
# The NWB files are hundreds of GB (they bundle raw ephys and video), but the spike
# tables and trial tables stream quickly through the LINDI index. We bulk-read the
# flat `units/spike_times` and `units/spike_times_index` arrays (per-unit reads over
# LINDI are too slow) and cache the small arrays locally as `.npz` so re-runs are fast.

# %%
ASSETS = {
    "sub-92130c1b": "31f22c47-1512-4293-b19f-6fa5bd9b7cbf",
    "sub-70bf8cbd": "81169999-c697-4eca-a635-2fd994ac183f",
    "sub-9bebfe0b": "e7fa5ae0-b957-4b24-aa40-fb4c3276d331",
    "sub-c6e8125f": "f791a116-1e6c-4d6a-a9eb-fe3644737be2",
}
SESSIONS = list(ASSETS)


def stream_and_cache(name, asset_id):
    """Stream one session via LINDI and cache trials/spikes/unit metadata to npz."""
    import lindi
    from pynwb import NWBHDF5IO
    url = (f"https://lindi.neurosift.org/dandi/dandisets/000149/assets/"
           f"{asset_id}/nwb.lindi.json")
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache())
    nwbfile = NWBHDF5IO(file=f).read()
    trials = nwbfile.intervals["trials"].to_dataframe()
    trials.columns = [c.replace(".npy", "") for c in trials.columns]
    st = f["units/spike_times"][:]
    sti = f["units/spike_times_index"][:]
    ks2 = np.array([x.decode() if isinstance(x, bytes) else x
                    for x in f["units/ks2_label"][:]])
    if "brainLocationAcronyms_ccf_2017.npy" in f["units"] and \
            f["units/brainLocationAcronyms_ccf_2017.npy"] is not None:
        regions = np.array([x.decode() if isinstance(x, bytes) else x
                            for x in f["units/brainLocationAcronyms_ccf_2017.npy"][:]])
    else:
        regions = np.array(["unknown"] * len(sti))
    np.savez_compressed(
        f"cache_{name}.npz", spike_times=st, spike_times_index=sti, ks2=ks2,
        regions=regions,
        **{f"trial_{c}": trials[c].to_numpy() for c in trials.columns})


def load_session(name):
    """Load from cache (streaming first if needed). Returns trials, per-unit spike
    arrays, ks2 labels, region acronyms."""
    path = f"cache_{name}.npz"
    if not os.path.exists(path):
        print(f"streaming {name} from DANDI (one-time, cached afterwards)...")
        stream_and_cache(name, ASSETS[name])
    d = np.load(path, allow_pickle=False)
    trials = pd.DataFrame({k[len("trial_"):]: d[k] for k in d.files
                           if k.startswith("trial_")})
    st, sti = d["spike_times"], d["spike_times_index"]
    bounds = np.concatenate([[0], sti])
    unit_spikes = [st[bounds[i]:bounds[i + 1]] for i in range(len(sti))]
    return trials, unit_spikes, d["ks2"], d["regions"]


DATA = {s: load_session(s) for s in SESSIONS}
for s, (tr, usp, ks2, reg) in DATA.items():
    print(f"{s}: {len(usp)} units, {len(tr)} trials, "
          f"{sum(len(u) for u in usp):,} spikes")

# %% [markdown]
# ## Shared decoding utilities
#
# The decoder is LDA with Ledoit-Wolf shrinkage (robust for the small-trial,
# many-unit regime), with variance thresholding and z-scoring inside each
# cross-validation fold. Accuracy is **balanced accuracy** (mean per-class recall),
# which is unbiased under the class imbalance induced by the block priors. Null
# distributions come from permuting the labels.

# %%
def select_units(unit_spikes, ks2, min_rate_hz=0.5):
    rates = np.array([len(s) / (s[-1] - s[0]) if len(s) > 1 else 0.0
                      for s in unit_spikes])
    return (rates >= min_rate_hz) & (ks2 == "good")


def count_in_windows(unit_spikes, starts, ends):
    """(n_trials, n_units) spike-count matrix for windows [starts, ends]."""
    cols = [np.searchsorted(sp, ends, side="right") -
            np.searchsorted(sp, starts, side="left") for sp in unit_spikes]
    return np.stack(cols, axis=1).astype(float)


def lda_pipe():
    return make_pipeline(VarianceThreshold(), StandardScaler(),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def cv_acc(X, y, n_repeats=10, seed0=0):
    """Mean balanced accuracy over repeated stratified 5-fold CV."""
    accs = []
    for r in range(n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed0 + r)
        for tr, te in skf.split(X, y):
            clf = lda_pipe().fit(X[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


def null_acc(X, y, n_shuffles=200, seed0=1000):
    """Permutation null distribution of CV balanced accuracy."""
    r = np.random.default_rng(seed0)
    return np.array([cv_acc(X, r.permutation(y), n_repeats=2, seed0=seed0 + i)
                     for i in range(n_shuffles)])


def pval(real, null):
    return (np.sum(null >= real) + 1) / (len(null) + 1)


def unit_auroc(counts, y):
    """Rank-based auROC of one unit's spike counts against binary y,
    with correct average-rank handling of ties."""
    y = np.asarray(y)
    vals = np.asarray(counts, dtype=float)
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(len(vals))
    ranks[order] = np.arange(1, len(vals) + 1)
    # average ranks within tied values
    sv = vals[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(np.sum(y == 1))
    r_pos = ranks[y == 1].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos))


# trial masks used throughout
PRE = (-0.4, 0.0)   # pre-stimulus window (inside the enforced quiescence period)
POST = (0.05, 0.45)  # post-stimulus positive-control window


def trial_info(trials):
    stim_on = trials["stimOn_times"].to_numpy()
    choice = trials["choice"].to_numpy()  # +1 = left, -1 = right
    cl = trials["contrastLeft"].to_numpy()
    cr = trials["contrastRight"].to_numpy()
    p_left = trials["probabilityLeft"].to_numpy()
    valid = ~np.isnan(choice) & ~np.isnan(stim_on)
    zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
    signed_contrast = np.where(np.isnan(cl), cr, -cl)
    return dict(stim_on=stim_on, choice=choice, p_left=p_left, valid=valid,
                zero_c=zero_c, signed_contrast=signed_contrast)

# %% [markdown]
# ## 1. The block prior biases behavior
#
# Psychometric curves (probability of choosing left as a function of signed contrast)
# shift systematically with the block prior. On 0%-contrast trials, where the choice
# cannot be stimulus-driven, the fraction of left choices tracks the prior almost
# exactly: the animal carries a measurable decision bias.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
colors = {0.2: "#d62728", 0.5: "#7f7f7f", 0.8: "#1f77b4"}
INFOS = {s: trial_info(DATA[s][0]) for s in SESSIONS}
contrast_levels = [-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1]

ax = axes[0]
for pl in [0.2, 0.5, 0.8]:
    per_sess = []
    for s in SESSIONS:
        info = INFOS[s]
        m = info["valid"] & (info["p_left"] == pl)
        chl = (info["choice"] == 1).astype(float)
        ys = [np.mean(chl[m & (info["signed_contrast"] == c)])
              if np.sum(m & (info["signed_contrast"] == c)) > 0 else np.nan
              for c in contrast_levels]
        per_sess.append(ys)
        ax.plot(contrast_levels, ys, "-", color=colors[pl], lw=0.8, alpha=0.25, zorder=1)
    per_sess = np.array(per_sess, dtype=float)
    mean = np.nanmean(per_sess, axis=0)
    sem = np.nanstd(per_sess, axis=0) / np.sqrt(len(SESSIONS))
    ax.plot(contrast_levels, mean, "o-", color=colors[pl], lw=2.2, ms=6,
            label=f"p(left)={pl}", zorder=3)
    ax.fill_between(contrast_levels, mean - sem, mean + sem, color=colors[pl],
                    alpha=0.18, zorder=2)
ax.axhline(0.5, color="k", lw=0.5, ls="--")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xscale("symlog", linthresh=0.02)
ax.set_xticks([-1, -0.25, -0.0625, 0, 0.0625, 0.25, 1])
ax.set_xticklabels(["-100", "-25", "-6.25", "0", "6.25", "25", "100"])
ax.set_xlabel("signed contrast (%, left < 0 < right)")
ax.set_ylabel("P(choose left)")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Psychometric curves by block prior\n(mean ± sem across sessions; faint: individual)")
ax.legend(frameon=False, loc="lower left")

ax = axes[1]
rng1 = np.random.default_rng(1)
means = []
for j, pl in enumerate([0.2, 0.5, 0.8]):
    vals = []
    for s in SESSIONS:
        info = INFOS[s]
        m = info["zero_c"] & (info["p_left"] == pl)
        if m.sum() >= 5:
            v = np.mean(info["choice"][m] == 1)
            vals.append(v)
            ax.scatter(j + rng1.uniform(-0.12, 0.12), v, s=42, color=colors[pl],
                       alpha=0.55, edgecolor="white", lw=0.5, zorder=3)
    means.append(np.mean(vals))
    ax.plot(j, np.mean(vals), "D", color="black", ms=8, zorder=4)
    ax.plot([j - 0.28, j + 0.28], [np.mean(vals)] * 2, color="black", lw=1.4, zorder=4)
ax.plot([0, 1, 2], means, color="black", lw=1.2, ls="-", alpha=0.5, zorder=2)
ax.axhline(0.5, color="k", lw=0.5, ls="--")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["p(left)=0.2", "p(left)=0.5", "p(left)=0.8"])
ax.set_ylabel("P(choose left)")
ax.set_ylim(0, 1)
ax.set_title("Choice on 0%-contrast trials tracks the prior\n(points: sessions; diamonds: mean)")
fig.tight_layout()
fig.savefig("fig1_behavior.png", dpi=150)
plt.close(fig)
print("saved fig1_behavior.png")

# %% [markdown]
# ## 2. Raw data check: pre-stimulus choice-selective single units
#
# Before decoding, verify that single units carry choice information before stimulus
# onset. We compute each unit's auROC between spike counts in the pre-stimulus window
# and the upcoming choice on 0%-contrast trials, and show rasters/PSTHs for the most
# selective left- and right-preferring units of the strongest session.

# %%
EX = "sub-c6e8125f"
trials, unit_spikes, ks2, regions = DATA[EX]
u_mask = select_units(unit_spikes, ks2)
units = [s for s, m in zip(unit_spikes, u_mask) if m]
ureg = regions[u_mask]
info = trial_info(trials)
t0z = info["stim_on"][info["zero_c"]]
yz = (info["choice"][info["zero_c"]] == 1).astype(int)
Xz = count_in_windows(units, t0z + PRE[0], t0z + PRE[1])
aurocs = np.array([unit_auroc(Xz[:, u], yz) for u in range(Xz.shape[1])])
left_u = int(np.argmax(aurocs))
right_u = int(np.argmin(aurocs))

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
twin, bin_w = (-0.6, 0.6), 0.02
bins = np.arange(twin[0], twin[1] + bin_w, bin_w)
kernel = np.ones(3) / 3  # light smoothing for readability
for col, (u, pref) in enumerate([(left_u, "left"), (right_u, "right")]):
    sp = units[u]
    ax = axes[0, col]
    row = 0
    for want, color in [(1, "#1f77b4"), (0, "#d62728")]:
        for t in t0z[yz == want]:
            rel = sp[(sp >= t + twin[0]) & (sp <= t + twin[1])] - t
            ax.plot(rel, np.full_like(rel, row), "|", color=color, ms=2, alpha=0.7)
            row += 1
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_ylabel("trials (by choice)")
    ax.set_title(f"unit {left_u if pref=='left' else right_u} ({ureg[u]}), "
                 f"pre-stim auROC={aurocs[u]:.2f}")
    ax.set_ylim(-1, row)
    ax = axes[1, col]
    for want, color, lab in [(1, "#1f77b4", "choice left"), (0, "#d62728", "choice right")]:
        tt = t0z[yz == want]
        counts = np.zeros(len(bins) - 1)
        for t in tt:
            rel = sp[(sp >= t + twin[0]) & (sp <= t + twin[1])] - t
            counts += np.histogram(rel, bins=bins)[0]
        rate = np.convolve(counts / (len(tt) * bin_w), kernel, mode="same")
        ax.plot(bins[:-1] + bin_w / 2, rate, color=color, lw=1.6, label=lab)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvspan(PRE[0], PRE[1], color="gray", alpha=0.12)
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"Pre-stimulus choice-selective units on 0%-contrast trials ({EX})\n"
             f"gray band = decoding window")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig2_raster_psth.png", dpi=150)
plt.close(fig)
print(f"saved fig2_raster_psth.png (left-pref unit {left_u} auROC={aurocs[left_u]:.2f}, "
      f"right-pref unit {right_u} auROC={aurocs[right_u]:.2f})")

# %% [markdown]
# ## 3. Pre-stimulus population decoding of the upcoming choice
#
# For each session we decode, from spike counts in the pre-stimulus window:
# * **upcoming choice on all trials**,
# * **upcoming choice on 0%-contrast trials** (pure bias, no sensory evidence),
# * **the block prior** (0.8-left vs 0.2-left blocks: the sustained bias state),
# * **the previous trial's choice** (control for lingering signals).
#
# Significance comes from 200 label permutations per comparison.

# %%
summary_rows = []
for s in SESSIONS:
    t0 = time.time()
    trials, unit_spikes, ks2, regions = DATA[s]
    u_mask = select_units(unit_spikes, ks2)
    units = [sp for sp, m in zip(unit_spikes, u_mask) if m]
    info = trial_info(trials)
    stim_on, choice = info["stim_on"], info["choice"]

    conds = {}
    # all trials
    m = info["valid"]
    conds["all trials"] = (m, (choice[m] == 1).astype(int))
    # 0% contrast
    m = info["zero_c"]
    conds["0% contrast"] = (m, (choice[m] == 1).astype(int))
    # block prior (exclude 0% trials so it is about the sustained state on
    # stimulus-present trials; including them changes nothing qualitatively)
    m = info["valid"] & (info["p_left"] != 0.5)
    conds["block prior"] = (m, (info["p_left"][m] == 0.8).astype(int))
    # previous choice (same trials as 0% contrast)
    prev = np.roll(choice.astype(float), 1)
    prev[0] = np.nan
    m = info["zero_c"] & ~np.isnan(prev)
    conds["previous choice"] = (m, (prev[m] == 1).astype(int))

    for cname, (m, y) in conds.items():
        X = count_in_windows(units, stim_on[m] + PRE[0], stim_on[m] + PRE[1])
        acc = cv_acc(X, y)
        null = null_acc(X, y)
        summary_rows.append(dict(session=s, condition=cname, acc=acc,
                                 null_mean=null.mean(),
                                 null95=np.percentile(null, 95),
                                 p=pval(acc, null), n=int(m.sum())))
        print(f"{s} [{cname:16s}] n={m.sum():4d} acc={acc:.3f} "
              f"null95={np.percentile(null, 95):.3f} p={summary_rows[-1]['p']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

summary = pd.DataFrame(summary_rows)
summary.to_csv("decoding_summary.csv", index=False)

# Fisher combined p across sessions for the two key conditions
from scipy.stats import combine_pvalues, chi2
for cname in ["all trials", "0% contrast"]:
    ps = summary.loc[summary.condition == cname, "p"].to_numpy()
    stat, p_comb = combine_pvalues(ps, method="fisher")
    print(f"Fisher combined across sessions [{cname}]: chi2={stat:.2f}, p={p_comb:.4f}")

# %% [markdown]
# ### Decoding summary figure

# %%
fig, ax = plt.subplots(figsize=(11, 5))
cond_order = ["all trials", "0% contrast", "block prior", "previous choice"]
cond_colors = {"all trials": "#2ca02c", "0% contrast": "#1f77b4",
               "block prior": "#9467bd", "previous choice": "#8c8c8c"}
n_sess = len(SESSIONS)
width = 0.19
for j, cname in enumerate(cond_order):
    for i, s in enumerate(SESSIONS):
        row = summary[(summary.session == s) & (summary.condition == cname)].iloc[0]
        x = i + (j - 1.5) * (width + 0.02)
        ax.bar(x, row.acc, width, color=cond_colors[cname],
               alpha=0.85, edgecolor="none",
               label=cname if i == 0 else None)
        ax.plot([x, x], [0.5, row.null95], color="k", lw=1.2, zorder=5)
        ax.plot([x - width / 3, x + width / 3], [row.null95, row.null95],
                color="k", lw=1.2, zorder=5)
        if row.p < 0.05:
            ax.text(x, max(row.acc, row.null95) + 0.012, "*", ha="center",
                    fontsize=14, fontweight="bold")
ax.axhline(0.5, color="k", lw=0.75, ls="--")
ax.set_xticks(range(n_sess))
ax.set_xticklabels([s.replace("sub-", "") for s in SESSIONS])
ax.set_xlabel("session")
ax.set_ylabel("balanced accuracy (CV)")
ax.set_ylim(0.42, 0.75)
ax.set_title("Pre-stimulus decoding ([-0.4, 0] s) per session\n"
             "black whisker = 95th percentile of the permutation null; * p < 0.05")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig("fig3_decoding_summary.png", dpi=150)
plt.close(fig)
print("saved fig3_decoding_summary.png")

# %% [markdown]
# ## 4. Time course: choice information before and after stimulus onset
#
# A 250 ms window is slid from -0.9 s to +0.6 s relative to stimulus onset. Choice
# decoding above the permutation band before time 0 demonstrates that the bias signal
# precedes the stimulus; the sharp rise after 0 validates the decoder.

# %%
centers = np.arange(-0.9, 0.61, 0.05)
sw_width = 0.25
sw_results = {}
for s in SESSIONS:
    trials, unit_spikes, ks2, regions = DATA[s]
    u_mask = select_units(unit_spikes, ks2)
    units = [sp for sp, m in zip(unit_spikes, u_mask) if m]
    info = trial_info(trials)
    per_cond = {}
    for tag, m in [("all", info["valid"]), ("zero", info["zero_c"])]:
        y = (info["choice"][m] == 1).astype(int)
        t0s = info["stim_on"][m]
        acc = np.empty(len(centers))
        n95 = np.empty(len(centers))
        for i, c in enumerate(tqdm(centers, desc=f"{s} [{tag}]", leave=False)):
            X = count_in_windows(units, t0s + c - sw_width / 2, t0s + c + sw_width / 2)
            acc[i] = cv_acc(X, y, n_repeats=3)
            nulls = [cv_acc(X, rng.permutation(y), n_repeats=1, seed0=100 + 31 * i + sh)
                     for sh in range(20)]
            n95[i] = np.percentile(nulls, 95)
        per_cond[tag] = (acc, n95)
    sw_results[s] = per_cond

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, tag, title in [(axes[0], "all", "all trials"),
                       (axes[1], "zero", "0%-contrast trials")]:
    accs = np.stack([sw_results[s][tag][0] for s in SESSIONS])
    n95 = np.stack([sw_results[s][tag][1] for s in SESSIONS])
    mean = accs.mean(0)
    sem = accs.std(0) / np.sqrt(len(SESSIONS))
    for i, s in enumerate(SESSIONS):
        ax.plot(centers, accs[i], color="#1f77b4", alpha=0.2, lw=1)
    ax.plot(centers, mean, color="#1f77b4", lw=2.2, label="mean across sessions")
    ax.fill_between(centers, mean - sem, mean + sem, color="#1f77b4", alpha=0.25)
    ax.fill_between(centers, 0.5, n95.mean(0), color="gray", alpha=0.3,
                    label="null 95% band")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xlabel("window center relative to stimulus onset (s)")
    ax.set_title(f"choice decoding time course, {title}")
    if ax is axes[0]:
        ax.set_ylabel("balanced accuracy (CV)")
        ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("Upcoming choice is decodable before stimulus onset")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig4_sliding_window.png", dpi=150)
plt.close(fig)
print("saved fig4_sliding_window.png")

# %% [markdown]
# ## 5. The block-bias axis predicts choice on ambiguous trials
#
# Is the moment-to-moment bias that drives choices on 0%-contrast trials the *same*
# signal as the sustained block-prior state? We train a decoder to distinguish
# 0.8-left from 0.2-left blocks using only stimulus-present trials, then ask whether
# that decoder's output on held-out 0%-contrast trials predicts the upcoming choice.
# The null permutes block labels.

# %%
cross_rows = []
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                         gridspec_kw={"width_ratios": [1, 1.15]})
strip_ax = axes[1]
for si, s in enumerate(SESSIONS):
    trials, unit_spikes, ks2, regions = DATA[s]
    u_mask = select_units(unit_spikes, ks2)
    units = [sp for sp, m in zip(unit_spikes, u_mask) if m]
    info = trial_info(trials)
    stim_on, choice, p_left = info["stim_on"], info["choice"], info["p_left"]
    zero_c, valid = info["zero_c"], info["valid"]
    y0 = (choice[zero_c] == 1).astype(int)
    X0 = count_in_windows(units, stim_on[zero_c] + PRE[0], stim_on[zero_c] + PRE[1])
    mb = valid & (p_left != 0.5) & ~zero_c
    yb = (p_left[mb] == 0.8).astype(int)
    Xb = count_in_windows(units, stim_on[mb] + PRE[0], stim_on[mb] + PRE[1])

    def cross_scores(yb_train, seed0=0):
        scores = np.zeros(X0.shape[0])
        skf = StratifiedKFold(5, shuffle=True, random_state=seed0)
        for tr, te in skf.split(Xb, yb_train):
            clf = lda_pipe().fit(Xb[tr], yb_train[tr])
            scores += clf.decision_function(X0) / 5
        return scores

    scores = cross_scores(yb)
    acc_c = balanced_accuracy_score(y0, scores > 0)
    r_c = np.corrcoef(scores, y0)[0, 1]
    null_c = np.array([balanced_accuracy_score(y0, cross_scores(rng.permutation(yb), i) > 0)
                       for i in range(100)])
    p_c = pval(acc_c, null_c)
    cross_rows.append(dict(session=s, acc=acc_c, r=r_c, p=p_c,
                           null95=np.percentile(null_c, 95), n=int(zero_c.sum())))
    print(f"{s}: cross-decoding acc={acc_c:.3f}, r={r_c:.3f}, p={p_c:.4f}", flush=True)
    # strip plot of scores by choice
    for want, color, off in [(0, "#d62728", -0.16), (1, "#1f77b4", 0.16)]:
        vals = scores[y0 == want]
        strip_ax.scatter(np.full_like(vals, si + off) +
                         rng.uniform(-0.05, 0.05, len(vals)), vals,
                         s=7, color=color, alpha=0.45, edgecolor="none")
    strip_ax.plot([si - 0.3, si + 0.3], [0, 0], "k--", lw=0.75)

ps = np.array([r["p"] for r in cross_rows])
stat, p_comb = combine_pvalues(ps, method="fisher")
print(f"Fisher combined cross-decoding p across sessions: {p_comb:.5f}")

ax = axes[0]
for i, r in enumerate(cross_rows):
    ax.bar(i, r["acc"], 0.6, color="#9467bd", alpha=0.85)
    ax.plot([i, i], [0.5, r["null95"]], color="k", lw=1.2)
    ax.plot([i - 0.2, i + 0.2], [r["null95"], r["null95"]], color="k", lw=1.2)
    if r["p"] < 0.05:
        ax.text(i, max(r["acc"], r["null95"]) + 0.012, "*", ha="center",
                fontsize=14, fontweight="bold")
ax.axhline(0.5, color="k", lw=0.75, ls="--")
ax.set_xticks(range(len(SESSIONS)))
ax.set_xticklabels([s.replace("sub-", "") for s in SESSIONS], rotation=0)
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0.42, 0.78)
ax.set_title("Block-bias axis generalizes to\n0%-contrast choice")

strip_ax.set_xticks(range(len(SESSIONS)))
strip_ax.set_xticklabels([s.replace("sub-", "") for s in SESSIONS])
strip_ax.set_ylabel("block-decoder score on 0% trials\n(> 0 = 'left-biased block' side)")
strip_ax.set_title("Decoder score vs. actual choice\n(red = chose right, blue = chose left)")
fig.tight_layout()
fig.savefig("fig5_cross_decoding.png", dpi=150)
plt.close(fig)
print("saved fig5_cross_decoding.png")
pd.DataFrame(cross_rows).to_csv("cross_decoding.csv", index=False)

# %% [markdown]
# ## 6. Single-unit choice selectivity before vs after stimulus onset
#
# Per-unit auROC with per-unit permutation tests (200 shuffles), comparing the
# pre-stimulus window (all trials and 0%-contrast trials) with a post-stimulus
# positive-control window.

# %%
N_SHUF = 200
su_rows = []
for s in SESSIONS:
    trials, unit_spikes, ks2, regions = DATA[s]
    u_mask = select_units(unit_spikes, ks2)
    units = [sp for sp, m in zip(unit_spikes, u_mask) if m]
    regs = regions[u_mask]
    info = trial_info(trials)
    for tag, m, win in [("pre (all trials)", info["valid"], PRE),
                        ("pre (0% contrast)", info["zero_c"], PRE),
                        ("post (all trials)", info["valid"], POST)]:
        y = (info["choice"][m] == 1).astype(int)
        t0s = info["stim_on"][m]
        X = count_in_windows(units, t0s + win[0], t0s + win[1])
        ar = np.array([unit_auroc(X[:, u], y) for u in range(X.shape[1])])
        null = np.empty((N_SHUF, X.shape[1]))
        r = np.random.default_rng(0)
        for sh in range(N_SHUF):
            yp = r.permutation(y)
            null[sh] = [unit_auroc(X[:, u], yp) for u in range(X.shape[1])]
        p = (np.sum(np.abs(null - 0.5) >= np.abs(ar - 0.5), axis=0) + 1) / (N_SHUF + 1)
        for u in range(X.shape[1]):
            su_rows.append(dict(session=s, cond=tag, unit=u, auroc=ar[u], p=p[u],
                                region=regs[u]))
        print(f"{s} [{tag:18s}]: significant units (p<0.05): {(p < 0.05).mean():.3f}",
              flush=True)
su = pd.DataFrame(su_rows)
su.to_csv("single_unit_auroc.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axes[0]
bins = np.linspace(-0.25, 0.25, 61)
for tag, color in [("pre (all trials)", "#2ca02c"), ("pre (0% contrast)", "#1f77b4"),
                   ("post (all trials)", "#ff7f0e")]:
    v = su.loc[su.cond == tag, "auroc"] - 0.5
    ax.hist(v, bins=bins, histtype="step", lw=1.8, color=color, label=tag,
            density=True)
ax.axvline(0, color="k", lw=0.75, ls="--")
ax.set_xlabel("auROC - 0.5 (choice selectivity)")
ax.set_ylabel("density")
ax.set_title("Single-unit choice selectivity\n(pooled across sessions)")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
fracs = {tag: [np.mean(su.loc[(su.cond == tag) & (su.session == s), "p"] < 0.05)
               for s in SESSIONS]
         for tag in ["pre (all trials)", "pre (0% contrast)", "post (all trials)"]}
x = np.arange(3)
for i, s in enumerate(SESSIONS):
    ax.bar(x + (i - 1.5) * 0.18,
           [fracs[t][i] for t in ["pre (all trials)", "pre (0% contrast)", "post (all trials)"]],
           0.16, alpha=0.45 + 0.15 * i, color="#2ca02c", edgecolor="none")
ax.axhline(0.05, color="k", lw=0.75, ls="--", label="chance (0.05)")
ax.set_xticks(x)
ax.set_xticklabels(["pre\n(all)", "pre\n(0%)", "post\n(all)"])
ax.set_ylabel("fraction of units p<0.05")
ax.set_title("Fraction of choice-selective units\n(bars = sessions)")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
sig = su[(su.cond == "pre (0% contrast)") & (su.p < 0.05) & (su.region != "unknown")]
top = sig.groupby("region").size().sort_values(ascending=False).head(12)
ax.barh(range(len(top)), top.values, color="#1f77b4", alpha=0.85)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("# significant pre-stim choice units")
ax.set_title("Regions of significant units\n(0% trials, annotated sessions)")
fig.tight_layout()
fig.savefig("fig6_single_units.png", dpi=150)
plt.close(fig)
print("saved fig6_single_units.png")

# %% [markdown]
# ## Summary

# %%
print("=== Pre-stimulus decoding summary (balanced accuracy, permutation p) ===")
print(summary.pivot(index="session", columns="condition", values="acc").round(3))
print()
print(summary.pivot(index="session", columns="condition", values="p").round(4))
print()
for cname, ps in [("all trials", summary.loc[summary.condition == "all trials", "p"]),
                  ("0% contrast", summary.loc[summary.condition == "0% contrast", "p"]),
                  ("block prior", summary.loc[summary.condition == "block prior", "p"])]:
    stat, p_comb = combine_pvalues(np.asarray(ps), method="fisher")
    print(f"Fisher combined [{cname}]: p = {p_comb:.5f}")
stat, p_comb = combine_pvalues(np.array([r["p"] for r in cross_rows]), method="fisher")
print(f"Fisher combined [cross-decoding block->0% choice]: p = {p_comb:.5f}")

# %% [markdown]
# ## Conclusions
#
# Across four IBL sessions, neural population activity recorded **before stimulus
# onset** carries reliable information about the upcoming decision:
#
# * The **upcoming choice** is decodable from the pre-stimulus window on all trials
#   in every session, and on 0%-contrast trials, where the choice is pure bias
#   with no sensory evidence, at the group level (Fisher combined p ≈ 0.01).
# * The **block prior**, i.e. the sustained decision-bias state, is decodable in
#   every session, and the neural axis that encodes it **generalizes to predict
#   choice on ambiguous trials**, directly linking the sustained bias state to
#   moment-to-moment biased choices.
# * The signal is not a lingering trace of the previous trial (previous-choice
#   decoding is at chance), and the sliding-window analysis shows choice
#   information present hundreds of milliseconds before the stimulus, rising
#   sharply once the stimulus appears.
# * Single units with significant pre-stimulus choice selectivity (12–19% of units
#   on all trials, well above the 5% chance rate) are distributed across regions,
#   with caudoputamen (dorsal striatum) most represented, consistent with
#   striatal encoding of action bias and value.
#
# Together these results demonstrate that an upcoming decision bias can be decoded
# from pre-stimulus neural activity in this perceptual decision task.
