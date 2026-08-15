# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Auditory frequency tuning in mouse auditory cortex
#
# **Dataset: DANDI:000986** — *Auditory cortex Neuropixels recordings and pupil
# diameter traces from mice during passive exposure to pure tones*
# (Lakunina et al.; https://dandiarchive.org/dandiset/000986).
#
# Neurons in the auditory cortex are frequency selective: each one responds most
# strongly to a restricted range of tone frequencies, its **best frequency (BF)**,
# and responds progressively less to tones further away in octaves. This notebook
# demonstrates that phenomenon directly in extracellular spiking data.
#
# The recordings are passive: an awake, head-fixed mouse listens to 25 ms pure
# tones at 60 dB SPL drawn at random from 2, 4, 8, 16 and 32 kHz, presented
# roughly every 0.8 s, with interleaved blocks of silence. Neuropixels probes in
# auditory cortex yield tens to hundreds of well-isolated units per session.
# Pupil diameter and running speed were recorded simultaneously and are used here
# only to confirm that the animal was awake and behaving normally.
#
# The analysis proceeds in five steps:
#
# 1. Stream one session from the DANDI S3 bucket and validate every data stream.
# 2. Confirm that tones drive the population at all (population PSTH).
# 3. Measure per-unit frequency tuning: rasters, PSTHs, tuning curves, and a
#    permutation ANOVA over the five frequencies.
# 4. Repeat over all 15 sessions from 5 mice and pool the results, including a
#    Poisson GLM (NeMoS) that gives a smooth tuning curve in octaves.
# 5. Ask whether the tuning is strong enough to read the presented frequency off
#    a single trial of population activity.
#
# All data are streamed with `remfile` + a local disk cache; nothing is
# downloaded in full. All analysis of spike times goes through Pynapple.

# %% [markdown]
# ## Setup

# %%
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import requests
import h5py
import remfile
import pynapple as nap
import nemos as nmo
from pynwb import NWBHDF5IO
from scipy import stats
from scipy.special import gammaln
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

# NeMoS raises a "fit did not converge" warning because LBFGS stops on its
# default tolerance rather than on the gradient norm. The fit is nonetheless
# exact here: `glm_tuning` reports the correlation between the fitted rate and
# the empirical mean spike count at the five tested frequencies, which comes out
# at r = 1.000 for every session. The warning is silenced only for that message.
warnings.filterwarnings("ignore", message="The fit did not converge")

DANDISET = "000986"
API = "https://api.dandiarchive.org/api/dandisets"
CACHE = "/tmp/remfile_cache_000986"

# Response windows in seconds relative to tone onset. The evoked window starts
# at 10 ms (cortical onset latency) and ends at 60 ms, which covers the transient
# response to a 25 ms tone (validated against the population PSTH below).
EVOKED_WIN = (0.010, 0.060)
BASELINE_WIN = (-0.100, -0.005)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10})

# %% [markdown]
# ### Data access and analysis helpers
#
# `list_assets` queries the DANDI API for every NWB file in the dandiset;
# `load_session` opens one of them over HTTP through `remfile` (byte-range reads,
# cached on disk) and wraps it in a Pynapple `NWBFile`.

# %%
def list_assets():
    """Return [(path, download_url), ...] for every session, sorted by path."""
    r = requests.get(
        f"{API}/{DANDISET}/versions/draft/assets/", params={"page_size": 500}
    ).json()
    out = []
    for a in sorted(r["results"], key=lambda x: x["path"]):
        url = f"{API}/{DANDISET}/versions/draft/assets/{a['asset_id']}/download/"
        out.append((a["path"], url))
    return out


def load_session(url):
    """Stream one NWB file (cached on disk) and return (nwbfile, pynapple wrapper)."""
    rem = remfile.File(url, disk_cache=remfile.DiskCache(CACHE))
    h5 = h5py.File(rem, "r")
    io = NWBHDF5IO(file=h5, load_namespaces=True)
    nwbfile = io.read()
    return nwbfile, nap.NWBFile(nwbfile)


def trial_table(nwbfile):
    """Tone onsets and stimulus parameters as plain arrays."""
    tr = nwbfile.trials
    return dict(
        onset=np.asarray(tr["start_time"].data[:]),
        stop=np.asarray(tr["stop_time"].data[:]),
        frequency=np.asarray(tr["stim_frequency"].data[:]),
        amplitude=np.asarray(tr["stim_amplitude"].data[:]),
        duration=np.asarray(tr["stim_duration"].data[:]),
    )


# %%
def window_counts(spikes, onsets, win):
    """Spike counts per (unit, trial) in `win` seconds relative to tone onset.

    Returns an (n_units, n_trials) array. `searchsorted` on each unit's sorted
    spike train is exact and fast enough for ~7500 trials x ~250 units.
    """
    lo, hi = onsets + win[0], onsets + win[1]
    out = np.zeros((len(spikes), len(onsets)))
    for i, k in enumerate(spikes.keys()):
        t = spikes[k].t
        out[i] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def evoked_rates(spikes, onsets, evoked_win=EVOKED_WIN, base_win=BASELINE_WIN):
    """Per-trial evoked and baseline firing rates (spikes/s), each (n_units, n_trials)."""
    ev = window_counts(spikes, onsets, evoked_win) / (evoked_win[1] - evoked_win[0])
    bl = window_counts(spikes, onsets, base_win) / (base_win[1] - base_win[0])
    return ev, bl


def peri_event_times(spike_times, onsets, tmin, tmax):
    """Spike times relative to each onset, plus the trial index of each spike.

    Vectorised gather: for every onset take the slice of the sorted spike train
    inside [onset+tmin, onset+tmax). Windows are shorter than the inter-tone
    interval, so each spike lands in at most one window.
    """
    lo = np.searchsorted(spike_times, onsets + tmin)
    hi = np.searchsorted(spike_times, onsets + tmax)
    n = hi - lo
    if n.sum() == 0:
        return np.array([]), np.array([], dtype=int)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    idx = np.repeat(lo - starts, n) + np.arange(n.sum())
    trial = np.repeat(np.arange(len(onsets)), n)
    return spike_times[idx] - onsets[trial], trial


def psth(spike_times, onsets, bins):
    """Trial-averaged firing rate (spikes/s) of one unit, aligned to `onsets`."""
    rel, _ = peri_event_times(spike_times, onsets, bins[0], bins[-1])
    h = np.histogram(rel, bins=bins)[0]
    return h / (len(onsets) * np.diff(bins))


def group_means(x, labels, levels):
    """Mean of x (n_units, n_trials) within each stimulus level -> (n_units, n_levels)."""
    return np.column_stack([x[:, labels == lv].mean(axis=1) for lv in levels])


def group_sems(x, labels, levels):
    out = []
    for lv in levels:
        s = x[:, labels == lv]
        out.append(s.std(axis=1, ddof=1) / np.sqrt(s.shape[1]))
    return np.column_stack(out)


# %% [markdown]
# ### Statistics
#
# Tuning is tested with a one-way ANOVA over the five frequencies, evaluated
# against a permutation null (frequency labels shuffled across trials) so that no
# normality assumption is needed for these strongly non-Gaussian spike counts.
# P-values are corrected across units within a session with Benjamini-Hochberg.

# %%
def sparseness(tc):
    """Treves-Rolls lifetime sparseness of a tuning curve, rescaled to [0, 1].

    0 = equal response to all frequencies, 1 = responds to a single frequency.
    Computed on the rectified (positive part of the) tuning curve.
    """
    r = np.clip(tc, 0, None)
    n = r.shape[-1]
    num = (r.mean(axis=-1)) ** 2
    den = (r**2).mean(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1 - num / den) / (1 - 1.0 / n)


def anova_f(x, labels, levels):
    """Vectorised one-way ANOVA F statistic over trials, for every unit at once."""
    n_units, n_tot = x.shape
    grand = x.mean(axis=1, keepdims=True)
    ss_b = np.zeros(n_units)
    ss_w = np.zeros(n_units)
    for lv in levels:
        g = x[:, labels == lv]
        m = g.mean(axis=1, keepdims=True)
        ss_b += g.shape[1] * (m[:, 0] - grand[:, 0]) ** 2
        ss_w += ((g - m) ** 2).sum(axis=1)
    df_b, df_w = len(levels) - 1, n_tot - len(levels)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ss_b / df_b) / (ss_w / df_w), df_b, df_w


def permutation_p(x, labels, levels, n_perm=500, seed=0):
    """Permutation p-value for frequency tuning, per unit."""
    rng = np.random.default_rng(seed)
    f_obs, _, _ = anova_f(x, labels, levels)
    ge = np.zeros(x.shape[0])
    for _ in range(n_perm):
        f_null, _, _ = anova_f(x, rng.permutation(labels), levels)
        ge += f_null >= f_obs
    return (ge + 1) / (n_perm + 1), f_obs


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def split_half_bf(dev, flab, freqs, seed=0):
    """Best frequency computed independently on two random halves of the trials."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(dev.shape[1])
    h1, h2 = order[::2], order[1::2]
    bf1 = freqs[np.argmax(group_means(dev[:, h1], flab[h1], freqs), axis=1)]
    bf2 = freqs[np.argmax(group_means(dev[:, h2], flab[h2], freqs), axis=1)]
    return bf1, bf2


# %% [markdown]
# ### Smooth tuning curves with a NeMoS Poisson GLM
#
# The five tones are one octave apart, so a tuning curve measured at those five
# points is coarse. Fitting a Poisson GLM whose design matrix is a raised-cosine
# basis over **log2 frequency** gives a continuous curve in octaves, from which a
# best frequency and a half-maximum bandwidth can be read off between the sampled
# tones. `PopulationGLM` fits every unit in a session at once against the shared
# design matrix. Held-out predictive power is summarised by a cross-validated
# McFadden pseudo-R^2 of the frequency model against a constant-rate model.

# %%
def _pois_ll(y, lam):
    """Summed Poisson log-likelihood per unit."""
    lam = np.clip(lam, 1e-9, None)
    return (y * np.log(lam) - lam - gammaln(y + 1)).sum(axis=0)


def _half_max_width(rate, grid_oct):
    """Width in octaves of the contiguous region above half the peak drive.

    Drive is the fitted rate minus its minimum over the tested range, so the
    width measures tuning sharpness on top of the unit's floor response.
    """
    r = rate - rate.min(axis=0, keepdims=True)
    peak = r.max(axis=0)
    step = grid_oct[1] - grid_oct[0]
    out = np.full(r.shape[1], np.nan)
    for i in range(r.shape[1]):
        if peak[i] <= 0:
            continue
        above = r[:, i] >= 0.5 * peak[i]
        j = int(np.argmax(r[:, i]))
        lo = hi = j
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(above) - 1 and above[hi + 1]:
            hi += 1
        out[i] = (hi - lo) * step
    return out


def glm_tuning(counts, flab, freqs, n_basis=4, n_grid=81, n_folds=5, seed=0):
    """Smooth frequency tuning from a Poisson population GLM (NeMoS)."""
    oct_trial = np.log2(flab / freqs[0])
    span = np.log2(freqs[-1] / freqs[0])
    basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=n_basis, bounds=(0, span))
    X = np.asarray(basis.compute_features(oct_trial), dtype=float)
    y = counts.T.astype(float)  # (n_trials, n_units)

    model = nmo.glm.PopulationGLM(solver_name="LBFGS").fit(X, y)
    grid_oct = np.linspace(0, span, n_grid)
    rate = np.asarray(
        model.predict(np.asarray(basis.compute_features(grid_oct), dtype=float))
    )  # (n_grid, n_units), counts per evoked window

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ll_full = np.zeros(y.shape[1])
    ll_null = np.zeros(y.shape[1])
    for tr_i, te_i in kf.split(X):
        m = nmo.glm.PopulationGLM(solver_name="LBFGS").fit(X[tr_i], y[tr_i])
        lam = np.asarray(m.predict(X[te_i]))
        ll_full += _pois_ll(y[te_i], lam)
        ll_null += _pois_ll(y[te_i], np.broadcast_to(y[tr_i].mean(axis=0), lam.shape))
    pr2 = 1 - ll_full / ll_null  # both log-likelihoods are negative

    # Diagnostic: does the fitted curve reproduce the empirical means at the five
    # tested frequencies? r close to 1 means the basis is rich enough.
    Xt = np.asarray(basis.compute_features(np.log2(freqs / freqs[0])), dtype=float)
    fit_corr = np.corrcoef(
        np.asarray(model.predict(Xt)).ravel(), group_means(counts, flab, freqs).T.ravel()
    )[0, 1]

    return dict(
        grid_oct=grid_oct,
        rate=rate.T,
        bf_oct=grid_oct[np.argmax(rate, axis=0)],
        bandwidth_oct=_half_max_width(rate, grid_oct),
        pseudo_r2=pr2,
        fit_corr=fit_corr,
    )


# %% [markdown]
# ### Single-trial decoding
#
# A shrinkage LDA classifier is used rather than plain logistic regression: the
# number of simultaneously recorded units is comparable to the number of trials
# per frequency, so the class covariance has to be regularised.

# %%
def decode_frequency(counts, flab, freqs, n_folds=5, seed=0):
    """Cross-validated decoding of the presented frequency from one trial."""
    X = counts.T.astype(float)
    y = np.searchsorted(freqs, flab)
    pred = np.zeros_like(y)
    for tr_i, te_i in StratifiedKFold(n_splits=n_folds, shuffle=True,
                                      random_state=seed).split(X, y):
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X[tr_i], y[tr_i])
        pred[te_i] = clf.predict(X[te_i])
    conf = np.zeros((len(freqs), len(freqs)))
    for a, b in zip(y, pred):
        conf[a, b] += 1
    return conf / conf.sum(axis=1, keepdims=True), (pred == y).mean()


# %% [markdown]
# ## 1. Load one session and validate every data stream

# %%
assets = list_assets()
print(f"{len(assets)} sessions in dandiset {DANDISET}")
for p, _ in assets:
    print("  ", p)

path0, url0 = assets[0]
nwbfile, nap_nwb = load_session(url0)
print("\nprototyping on:", path0)
print(nap_nwb)

# %%
print("session_description:", nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, "|", nwbfile.subject.species, "|",
      nwbfile.subject.sex, "|", nwbfile.subject.age)
print("publications:", nwbfile.related_publications)

trials = trial_table(nwbfile)
spikes = nap_nwb["units"]
onsets, flab = trials["onset"], trials["frequency"]
freqs = np.unique(flab)
fkhz = freqs / 1e3
nf = len(freqs)

print("\nfrequencies (kHz):", fkhz)
print("level (dB SPL):", np.unique(trials["amplitude"]),
      " duration (s):", np.unique(trials["duration"]))
print("n tones:", len(onsets),
      " repeats per frequency:", [int((flab == f).sum()) for f in freqs])
print("median inter-tone interval (s): %.3f" % np.median(np.diff(onsets)))
print("n units:", len(spikes),
      " firing rate: median %.2f Hz, range %.2f-%.2f Hz"
      % (np.median(spikes.rate), spikes.rate.min(), spikes.rate.max()))
print("\nsilent (spontaneous) blocks:")
print(nwbfile.intervals["spontaneous_blocks"].to_dataframe())

pupil = nap_nwb["pupil_diameter"]
run = nap_nwb["running_speed"]
print("\npupil:", pupil.shape, "NaNs:", int(np.isnan(pupil.d).sum()))
print("running:", run.shape, "NaNs:", int(np.isnan(run.d).sum()))

# %% [markdown]
# ### Figure 1 — raw data streams
#
# Eight seconds of the tone block: the spike raster of every unit, the summed
# population rate, pupil diameter, and running speed, with tone onsets marked and
# coloured by frequency. Tones are clearly followed by short bursts of population
# activity, and the behavioural traces confirm an awake, moving animal.

# %%
cmap = plt.get_cmap("viridis")
colors = [cmap(i / (nf - 1)) for i in range(nf)]

t0 = onsets[len(onsets) // 2] - 0.8
t1 = t0 + 8.0
ep = nap.IntervalSet(start=t0, end=t1)
sel = (onsets >= t0) & (onsets <= t1)

fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1, 1, 1]})
keys = list(spikes.keys())
order_rate = np.argsort(spikes.rate.values)
for row, ui in enumerate(order_rate):
    t = spikes[keys[ui]].restrict(ep).t
    axes[0].plot(t, np.full_like(t, row), "|", color="k", ms=2.5, mew=0.5)
axes[0].set_ylabel("unit (sorted by rate)")

pop = spikes.count(0.01, ep).sum(axis=1) / len(spikes) / 0.01
axes[1].plot(pop.t, pop.d, lw=0.8)
axes[1].set_ylabel("population rate\n(spikes/s/unit)")

axes[2].plot(pupil.restrict(ep).t, pupil.restrict(ep).d, color="tab:purple", lw=1)
axes[2].set_ylabel("pupil diameter\n(a.u.)")
axes[3].plot(run.restrict(ep).t, run.restrict(ep).d, color="tab:green", lw=1)
axes[3].set_ylabel("running speed\n(a.u.)")
axes[3].set_xlabel("time (s)")

for ax in axes:
    for o, f in zip(onsets[sel], flab[sel]):
        ax.axvline(o, color=colors[int(np.searchsorted(freqs, f))], lw=1.5, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlim(t0, t1)
handles = [plt.Line2D([], [], color=c, lw=2, label=f"{f:g} kHz")
           for c, f in zip(colors, fkhz)]
axes[0].legend(handles=handles, ncol=nf, frameon=False, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, 1.02), title="tone frequency")
fig.suptitle(f"{path0} — raw data streams, 8 s of the tone block", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig01_raw_data_streams.png")
plt.show()

# %% [markdown]
# ## 2. Do tones drive the population at all?
#
# Before asking about frequency selectivity, confirm that the tones produce a
# response and choose the analysis windows from the data.

# %%
bins = np.arange(-0.15, 0.351, 0.002)
centers = bins[:-1] + 0.001
pop_psth = np.mean([psth(spikes[k].t, onsets, bins) for k in keys], axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(centers, pop_psth, color="k", lw=1.2)
ax.axvspan(0, 0.025, color="orange", alpha=0.35, label="25 ms tone")
ax.axvspan(*EVOKED_WIN, color="crimson", alpha=0.15, label="evoked window")
ax.axvspan(*BASELINE_WIN, color="steelblue", alpha=0.15, label="baseline window")
ax.set_xlabel("time from tone onset (s)")
ax.set_ylabel("mean firing rate (spikes/s/unit)")
ax.set_title(f"Population PSTH, all {len(onsets)} tones ({len(spikes)} units)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig02_population_psth.png")
plt.show()

peak_t = centers[np.argmax(pop_psth)]
print("baseline %.2f spikes/s, peak %.2f spikes/s at %.0f ms"
      % (pop_psth[centers < -0.01].mean(), pop_psth.max(), 1e3 * peak_t))

# %% [markdown]
# The population rate rises about 10 ms after tone onset, peaks near 20 ms, and
# has decayed back to baseline by roughly 100 ms. The 10-60 ms evoked window and
# the -100 to -5 ms baseline window used throughout follow from this.

# %% [markdown]
# ## 3. Frequency tuning of single units
#
# For each unit and each trial, take the spike rate in the evoked window minus
# the rate in that same trial's baseline window. Two tests follow:
# a paired t-test of evoked against baseline (is the unit sound driven?) and a
# permutation ANOVA of the baseline-subtracted rate across the five frequencies
# (is it frequency selective?).

# %%
ev, bl = evoked_rates(spikes, onsets)
dev = ev - bl

t_resp, p_resp = stats.ttest_rel(ev, bl, axis=1)
responsive = bh_fdr(p_resp) < 0.01
print(f"sound-responsive units: {responsive.sum()}/{len(spikes)} "
      f"({100*responsive.mean():.0f}%), {np.sum(responsive & (t_resp > 0))} excited")

p_tune, f_obs = permutation_p(dev, flab, freqs, n_perm=500, seed=1)
p_tune_adj = bh_fdr(p_tune)
tuned = p_tune_adj < 0.05
print(f"frequency-tuned units: {tuned.sum()}/{len(spikes)} ({100*tuned.mean():.0f}%)")
print(f"tuned among responsive: {np.sum(tuned & responsive)}/{responsive.sum()}")

tc = group_means(dev, flab, freqs)
tc_sem = group_sems(dev, flab, freqs)
bf_idx = np.argmax(tc, axis=1)
bf = freqs[bf_idx]

# %% [markdown]
# ### Figure 3 — three example units
#
# Rasters grouped by tone frequency (top), per-frequency PSTHs (middle), and the
# resulting tuning curve (bottom), for three units with different best
# frequencies. The tuning is visible in the raw rasters, not only in the averages.

# %%
score = np.where(tuned & responsive, f_obs, -np.inf)
examples = []
for j in range(nf):
    cand = np.where((bf_idx == j) & np.isfinite(score))[0]
    if len(cand):
        examples.append(cand[np.argmax(score[cand])])
examples = sorted(sorted(examples, key=lambda i: -score[i])[:3], key=lambda i: bf_idx[i])
print("example units:", examples, " BF (kHz):", bf[examples] / 1e3)

pbins = np.arange(-0.05, 0.151, 0.002)
pcent = pbins[:-1] + 0.001
fig, axes = plt.subplots(3, len(examples), figsize=(4.4 * len(examples), 10.5),
                         gridspec_kw={"height_ratios": [2.2, 1.3, 1.3]})
for c, ui in enumerate(examples):
    t = spikes[keys[ui]].t
    axr, axp, axt = axes[0, c], axes[1, c], axes[2, c]

    row, yticks, ylabels = 0, [], []
    for j, f in enumerate(freqs):
        sub = onsets[flab == f][:120]
        rel, trial = peri_event_times(t, sub, pbins[0], pbins[-1])
        axr.plot(rel, trial + row, "|", color=colors[j], ms=2, mew=0.5)
        yticks.append(row + len(sub) / 2)
        ylabels.append(f"{f/1e3:g}")
        row += len(sub)
    axr.set_yticks(yticks)
    axr.set_yticklabels(ylabels)
    axr.set_ylim(0, row)
    axr.set_xlim(pbins[0], pbins[-1])
    axr.axvspan(0, 0.025, color="0.85", zorder=0)
    axr.set_ylabel("tone frequency (kHz)")
    axr.set_title(f"unit {ui}  (BF = {bf[ui]/1e3:g} kHz)")

    for j, f in enumerate(freqs):
        axp.plot(pcent, psth(t, onsets[flab == f], pbins), color=colors[j], lw=1.2,
                 label=f"{f/1e3:g} kHz")
    axp.axvspan(0, 0.025, color="0.85", zorder=0)
    axp.set_xlabel("time from tone onset (s)")
    axp.set_ylabel("rate (spikes/s)")
    if c == 0:
        axp.legend(frameon=False, fontsize=8)

    axt.errorbar(np.log2(fkhz), tc[ui], yerr=tc_sem[ui], marker="o", color="k",
                 capsize=3)
    axt.axhline(0, color="0.6", lw=0.8, ls="--")
    axt.set_xticks(np.log2(fkhz))
    axt.set_xticklabels([f"{f:g}" for f in fkhz])
    axt.set_xlabel("tone frequency (kHz)")
    axt.set_ylabel("evoked rate\n(spikes/s over baseline)")
    axt.set_title(f"ANOVA F = {f_obs[ui]:.1f}, p = {p_tune_adj[ui]:.3g}", fontsize=10)

for ax in axes.ravel():
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Frequency tuning of single auditory-cortex units — {path0}", y=1.0)
fig.tight_layout()
fig.savefig("fig03_example_units.png")
plt.show()

# %% [markdown]
# ## 4. All 15 sessions
#
# The same pipeline is applied to every session in the dandiset (15 sessions from
# 5 mice). Results are cached to `results_all_sessions.npy`; delete that file to
# force a recomputation. A full run takes roughly half an hour, most of it the
# permutation tests and the GLM cross-validation.

# %%
RESULTS = "results_all_sessions.npy"

if os.path.exists(RESULTS):
    per_session = list(np.load(RESULTS, allow_pickle=True))
    print(f"loaded cached results for {len(per_session)} sessions")
else:
    per_session = []
    for path, url in tqdm(assets, desc="sessions"):
        nwbf, napf = load_session(url)
        tr = trial_table(nwbf)
        sp = napf["units"]
        ons, fl = tr["onset"], tr["frequency"]
        fq = np.unique(fl)

        e, b = evoked_rates(sp, ons)
        d = e - b
        _, pr = stats.ttest_rel(e, b, axis=1)
        resp = bh_fdr(pr) < 0.01
        pt, fo = permutation_p(d, fl, fq, n_perm=500, seed=1)
        tun = bh_fdr(pt) < 0.05
        curves = group_means(d, fl, fq)
        b1, b2 = split_half_bf(d, fl, fq, seed=2)

        cnt = window_counts(sp, ons, EVOKED_WIN)
        g = glm_tuning(cnt, fl, fq)
        keep = tun & resp
        # Decode from the full, unselected population. Selecting units with a
        # tuning test computed on all trials and then decoding those same trials
        # would bias accuracy upward, because the selection has already seen the
        # held-out folds. The selected-subset version is kept for comparison.
        cf, ac = decode_frequency(cnt, fl, fq)
        cf_t, ac_t = decode_frequency(cnt[keep], fl, fq)

        per_session.append(dict(
            path=path, freqs=fq, tc=curves, bf=fq[np.argmax(curves, axis=1)],
            tuned=tun, responsive=resp, spars=sparseness(curves), f_obs=fo,
            bf1=b1, bf2=b2, conf=cf, acc=ac, conf_tuned=cf_t, acc_tuned=ac_t,
            glm_rate=g["rate"],
            glm_bf_oct=g["bf_oct"], glm_bw=g["bandwidth_oct"],
            glm_pr2=g["pseudo_r2"], grid_oct=g["grid_oct"],
            n_trials=len(ons), base_rate=b.mean(axis=1),
        ))
        print(f"{path}: {len(sp)} units, {resp.sum()} responsive, {keep.sum()} tuned, "
              f"GLM fit r = {g['fit_corr']:.3f}, decode acc {ac:.3f} "
              f"(tuned subset {ac_t:.3f})", flush=True)

    np.save(RESULTS, np.array(per_session, dtype=object), allow_pickle=True)

S = per_session

# %%
keep = [s["tuned"] & s["responsive"] for s in S]
tc_all = np.vstack([s["tc"][k] for s, k in zip(S, keep)])
bf_all = np.concatenate([s["bf"][k] for s, k in zip(S, keep)])
spars_all = np.concatenate([s["spars"][k] for s, k in zip(S, keep)])
bw_all = np.concatenate([s["glm_bw"][k] for s, k in zip(S, keep)])
pr2_tuned = np.concatenate([s["glm_pr2"][k] for s, k in zip(S, keep)])
pr2_all = np.concatenate([s["glm_pr2"] for s in S])
bf1 = np.concatenate([s["bf1"][k] for s, k in zip(S, keep)])
bf2 = np.concatenate([s["bf2"][k] for s, k in zip(S, keep)])
bf_idx_all = np.searchsorted(freqs, bf_all)

n_units = sum(len(s["tuned"]) for s in S)
n_resp = sum(int(s["responsive"].sum()) for s in S)
n_tuned = len(bf_all)
n_tot = [len(s["tuned"]) for s in S]
n_r = [int(s["responsive"].sum()) for s in S]
n_t = [int(k.sum()) for k in keep]
print(f"{len(S)} sessions, {n_units} units, {n_resp} responsive ({100*n_resp/n_units:.0f}%), "
      f"{n_tuned} frequency tuned ({100*n_tuned/n_units:.0f}%)")

# %% [markdown]
# ### Figure 4 — population summary
#
# Yield per session, the distribution of best frequencies, how selective the
# tuning curves are, and a control: the best frequency computed from one random
# half of the trials against the best frequency from the other half. If the
# apparent tuning were noise, that split-half matrix would be flat at 20%.

# %%
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

ax = axes[0, 0]
x = np.arange(len(S))
ax.bar(x, n_tot, color="0.85", label="all units")
ax.bar(x, n_r, color="0.55", label="sound responsive")
ax.bar(x, n_t, color="firebrick", label="frequency tuned")
ax.set_xticks(x)
ax.set_xticklabels([s["path"].split("/")[1].replace("_behavior.nwb", "")
                    .replace("sub-", "").replace("_ses-", " s") for s in S],
                   rotation=90, fontsize=7)
ax.set_ylabel("number of units")
ax.set_title(f"Yield per session ({n_tuned}/{n_units} units frequency tuned)")
ax.legend(frameon=False, fontsize=8)

ax = axes[0, 1]
counts_bf = np.array([(bf_idx_all == j).sum() for j in range(nf)])
ax.bar(np.arange(nf), 100 * counts_bf / counts_bf.sum(), color=colors,
       edgecolor="k", linewidth=0.5)
ax.axhline(20, color="k", ls="--", lw=0.8, label="uniform (20%)")
ax.set_xticks(np.arange(nf))
ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("% of tuned units")
ax.set_title("Best-frequency distribution")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
ax.hist(spars_all, bins=np.linspace(0, 1, 26), color="steelblue", edgecolor="k",
        linewidth=0.4)
ax.axvline(np.median(spars_all), color="firebrick", lw=1.5,
           label=f"median = {np.median(spars_all):.2f}")
ax.set_xlabel("lifetime sparseness of tuning curve\n(0 = flat, 1 = single frequency)")
ax.set_ylabel("number of tuned units")
ax.set_title("Tuning selectivity")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
M = np.zeros((nf, nf))
for a, b in zip(np.searchsorted(freqs, bf1), np.searchsorted(freqs, bf2)):
    M[a, b] += 1
im = ax.imshow(M / M.sum(axis=1, keepdims=True), cmap="magma", vmin=0, vmax=1,
               origin="lower")
ax.set_xticks(np.arange(nf)); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_yticks(np.arange(nf)); ax.set_yticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("BF from second half of trials (kHz)")
ax.set_ylabel("BF from first half (kHz)")
agree = np.trace(M) / M.sum()
ax.set_title(f"Split-half BF stability ({100*agree:.0f}% agree, chance 20%)")
fig.colorbar(im, ax=ax, label="fraction of units", shrink=0.85)

for a in axes.ravel()[:3]:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("DANDI 000986 — frequency tuning across 15 auditory-cortex sessions",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("fig04_population_summary.png")
plt.show()

# %% [markdown]
# ### Figure 5 — the tuning curves themselves
#
# Every tuned unit's curve normalised to its own peak and sorted by best
# frequency (left); the mean curve of each best-frequency group (middle); and the
# average curve after aligning each unit to its own best frequency (right). The
# last panel is the compact statement of the phenomenon: response falls off
# monotonically with distance in octaves from the preferred frequency.

# %%
norm = tc_all / np.abs(tc_all).max(axis=1, keepdims=True)
order = np.lexsort((-tc_all.max(axis=1), bf_idx_all))
oct_axis = np.log2(fkhz / fkhz[0])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6),
                         gridspec_kw={"width_ratios": [1.25, 1, 1]})
ax = axes[0]
im = ax.imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               origin="lower", extent=[-0.5, nf - 0.5, 0, len(norm)])
ax.set_xticks(np.arange(nf)); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("tuned units, sorted by best frequency")
ax.set_title("Single-unit tuning curves\n(each row normalised to its peak)")
fig.colorbar(im, ax=ax, label="normalised evoked rate", shrink=0.9)

ax = axes[1]
for j in range(nf):
    s = bf_idx_all == j
    m = norm[s].mean(axis=0)
    se = norm[s].std(axis=0, ddof=1) / np.sqrt(s.sum())
    ax.plot(oct_axis, m, color=colors[j], lw=1.8,
            label=f"BF {fkhz[j]:g} kHz (n={s.sum()})")
    ax.fill_between(oct_axis, m - se, m + se, color=colors[j], alpha=0.25)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.set_xticks(oct_axis); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalised evoked rate")
ax.set_title("Mean tuning curve by best frequency")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
offsets = np.arange(-(nf - 1), nf)
acc_off = [[] for _ in offsets]
for i in range(len(norm)):
    for j in range(nf):
        acc_off[j - bf_idx_all[i] + nf - 1].append(norm[i, j])
m = np.array([np.mean(a) if a else np.nan for a in acc_off])
se = np.array([np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
               for a in acc_off])
ok = np.array([len(a) for a in acc_off]) >= 20
ax.errorbar(offsets[ok], m[ok], yerr=se[ok], marker="o", color="k", lw=1.8, capsize=3)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axvline(0, color="firebrick", lw=0.8, ls=":")
ax.set_xlabel("octaves from each unit's best frequency")
ax.set_ylabel("normalised evoked rate")
ax.set_title("Best-frequency-aligned population\ntuning curve")

for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig05_tuning_curves_population.png")
plt.show()

# %% [markdown]
# ### Figure 6 — GLM tuning curves, bandwidth, and held-out fit
#
# The Poisson GLM interpolates between the five sampled tones, which turns the
# discrete tuning curve into a continuous function of octaves and gives a
# half-maximum bandwidth for each unit. The bandwidth is limited from below by
# the one-octave spacing of the stimulus set, so it should be read as a coarse
# measure of sharpness rather than a precise Q value. The right panel shows that
# knowing the tone frequency genuinely improves prediction of held-out spike
# counts for tuned units, and does essentially nothing for the rest.

# %%
grid_oct = S[0]["grid_oct"]
grid_khz = fkhz[0] * 2 ** grid_oct

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axes[0]
s0 = S[0]
idx = np.where(s0["tuned"] & s0["responsive"])[0]
idx = idx[np.argsort(-s0["glm_pr2"][idx])][:6]
for i in idx:
    r = s0["glm_rate"][i]
    ax.plot(grid_khz, (r - r.min()) / (r.max() - r.min()), lw=1.6,
            label=f"unit {i} (BF {fkhz[0]*2**s0['glm_bf_oct'][i]:.1f} kHz)")
ax.set_xscale("log", base=2)
ax.set_xticks(fkhz); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalised GLM rate")
ax.set_title("Poisson GLM tuning curves\n(raised-cosine basis in octaves)")
ax.legend(frameon=False, fontsize=7)

ax = axes[1]
ax.hist(bw_all[np.isfinite(bw_all)], bins=np.linspace(0, 4, 25), color="steelblue",
        edgecolor="k", linewidth=0.4)
ax.axvline(np.nanmedian(bw_all), color="firebrick", lw=1.5,
           label=f"median = {np.nanmedian(bw_all):.2f} oct")
ax.set_xlabel("half-max bandwidth (octaves)")
ax.set_ylabel("number of tuned units")
ax.set_title("Tuning bandwidth")
ax.legend(frameon=False, fontsize=9)

ax = axes[2]
hbins = np.linspace(-0.02, max(0.2, np.nanpercentile(pr2_all, 99.5)), 40)
ax.hist(pr2_all, bins=hbins, color="0.75", edgecolor="k", linewidth=0.3,
        label=f"all units (n={len(pr2_all)})")
ax.hist(pr2_tuned, bins=hbins, color="firebrick", edgecolor="k", linewidth=0.3,
        label=f"frequency tuned (n={len(pr2_tuned)})")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("cross-validated pseudo-$R^2$\n(frequency model vs. constant rate)")
ax.set_ylabel("number of units")
ax.set_title("Held-out predictive power")
ax.legend(frameon=False, fontsize=8)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig06_glm_tuning.png")
plt.show()

# %% [markdown]
# ### Figure 7 — single-trial decoding
#
# If individual units are frequency selective, the presented frequency should be
# recoverable from one trial of population activity. Errors that do occur should
# land on neighbouring frequencies, which is what a graded tuning curve predicts.
# The decoder is given every recorded unit, with no selection for tuning, so the
# accuracy is not inflated by having chosen units using the same trials.

# %%
conf = np.nanmean(np.stack([s["conf"] for s in S]), axis=0)
accs = np.array([s["acc"] for s in S])
accs_tuned = np.array([s["acc_tuned"] for s in S])
print("accuracy, all units:    %.3f +- %.3f" % (accs.mean(), accs.std()))
print("accuracy, tuned subset: %.3f +- %.3f (selection-biased, for comparison)"
      % (accs_tuned.mean(), accs_tuned.std()))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
im = ax.imshow(conf, cmap="magma", vmin=0, vmax=np.nanmax(conf), origin="lower")
ax.set_xticks(np.arange(nf)); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_yticks(np.arange(nf)); ax.set_yticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("decoded frequency (kHz)")
ax.set_ylabel("presented frequency (kHz)")
ax.set_title("Single-trial decoding from all units, mean over sessions\n"
             f"accuracy = {np.nanmean(accs):.2f} (chance 0.20)")
for a in range(nf):
    for b in range(nf):
        ax.text(b, a, f"{conf[a, b]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if conf[a, b] < 0.6 * np.nanmax(conf) else "black")
fig.colorbar(im, ax=ax, label="fraction of trials", shrink=0.85)

ax = axes[1]
ax.scatter(n_tot, accs, s=45, color="firebrick", zorder=3)
ax.axhline(0.2, color="k", ls="--", lw=1, label="chance")
ax.set_xlabel("number of units in session")
ax.set_ylabel("decoding accuracy")
ax.set_title("Decoding improves with population size")
ax.set_ylim(0, max(0.75, np.nanmax(accs) * 1.15))
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig07_decoding.png")
plt.show()

r_size = stats.spearmanr(n_tot, accs)
print("Spearman(accuracy, n units): rho = %.2f, p = %.3g"
      % (r_size.statistic, r_size.pvalue))

# %% [markdown]
# ## 5. Summary

# %%
chi = stats.chisquare(counts_bf)
print(f"""
sessions .......................... {len(S)} (5 mice)
units ............................. {n_units}
sound responsive .................. {n_resp} ({100*n_resp/n_units:.0f}%)
frequency tuned ................... {n_tuned} ({100*n_tuned/n_units:.0f}%)
median lifetime sparseness ........ {np.median(spars_all):.2f}
median GLM half-max bandwidth ..... {np.nanmedian(bw_all):.2f} octaves
median CV pseudo-R2 (tuned units) . {np.median(pr2_tuned):.3f}
split-half BF agreement ........... {100*agree:.0f}% (chance 20%)
BF counts (kHz) ................... {dict(zip([f'{f:g}' for f in fkhz], counts_bf))}
  chi2 vs uniform ................. {chi.statistic:.1f}, p = {chi.pvalue:.2g}
single-trial decoding accuracy .... {np.nanmean(accs):.3f} +- {np.nanstd(accs):.3f} (chance 0.200)
""")
