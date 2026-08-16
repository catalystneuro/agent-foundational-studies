# %% [markdown]
# # Spectrotemporal Receptive Fields in Mouse Auditory Cortex
#
# This notebook demonstrates spectrotemporal receptive field (STRF) estimation in the
# auditory system using real data from the DANDI Archive. An STRF describes how a
# neuron's firing rate depends on the frequency content of the recent acoustic
# stimulus: it is a two-dimensional function of frequency and time lag, with
# excitatory subfields (sounds that drive the cell) and suppressive subfields
# (sounds that reduce its firing).
#
# We use DANDI dataset
# [000986](https://dandiarchive.org/dandiset/000986) (Jaramillo lab, version
# 0.251031.1939): Neuropixels recordings from mouse auditory cortex during passive
# presentation of short pure tones (25 ms, 60 dB SPL) at five octave-spaced
# frequencies (2, 4, 8, 16, 32 kHz), with roughly 1450 trials per frequency per
# session. Because each tone is spectrally narrow and temporally brief, the
# tone-evoked response measured as a function of tone frequency and post-onset
# latency directly traces out the neuron's spectrotemporal response field. This
# is a standard way to estimate STRFs when dense random stimuli (such as dynamic
# ripples) are not available.
#
# The analysis proceeds in four steps:
#
# 1. Stream one session with LINDI and validate the data streams (raster, PSTH).
# 2. Estimate a model-free "rate-map" STRF per unit: the baseline-subtracted
#    firing rate in each (frequency, latency) bin.
# 3. Estimate a regularized STRF per unit with a Poisson GLM (NeMoS) over a
#    two-dimensional B-spline basis, and validate it on held-out trials.
# 4. Pool six sessions (five mice) for population statistics: best frequency,
#    peak latency, suppressive subfields, and STRF separability (SVD).

# %% [markdown]
# ## 1. Setup
#
# The NWB files are streamed from the DANDI S3 bucket through the neurosift LINDI
# index with a local disk cache (no bulk downloads). `pynapple` provides the
# perievent cross-validation, `nemos` the GLM, and `scipy` the statistics.
# Figures are written to the working directory as PNG files.

# %%
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln
from tqdm import tqdm

import lindi
import pynapple as nap
from pynwb import NWBHDF5IO

import jax
jax.config.update("jax_enable_x64", True)  # needed for LBFGS convergence in nemos
import nemos as nmo

# --- analysis constants -------------------------------------------------------
FREQ_VALS = np.array([2000., 4000., 8000., 16000., 32000.])  # tone frequencies (Hz)
LOGF = np.log2(FREQ_VALS / 1000.)                            # log2 kHz: 1..5
BIN_EDGES = np.arange(-0.05, 0.155, 0.005)                   # -50..150 ms, 5 ms bins
BIN_C = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BASE_MASK = BIN_C < 0                                        # pre-onset baseline bins
EVOKED_MASK = (BIN_C >= 0.005) & (BIN_C < 0.060)             # 5-60 ms response window
BIN_S = 0.005                                                # bin width in seconds

# Six sessions, one or two per mouse (five mice total).
ASSETS = {
    "LA3_ses3":  "5e111970-9331-41d0-81b2-829e1c0f8040",
    "LA8_ses1":  "60303460-38be-44a0-951e-82c7957d1217",
    "LA8_ses2":  "ce06d820-e471-4413-a3a8-9c0b21da8680",
    "LA9_ses1":  "d19d0ca3-7c9a-41fe-bd97-b4ee8899a612",
    "LA11_ses2": "b8d3abca-0e78-4df1-9a51-d122a383be63",
    "LA12_ses1": "eb82c81a-87a0-40a4-b70e-535ac0909c86",
}

# %% [markdown]
# ## 2. Streaming Data Access
#
# Each session is one NWB file of 200-300 MB. We stream it through the LINDI
# index rather than downloading it. Two implementation details are worth noting.
# First, the spike times are read in bulk from the HDF5 handle (`units/spike_times`
# plus `units/spike_times_index`) and sliced locally, because the pynwb ragged-array
# read path is slow over remote files. Second, trial onsets are sorted before use,
# since the perievent binning below relies on `searchsorted`.

# %%
def load_session(asset_id):
    """Stream one session; return per-unit spike times, trial onsets, and frequencies."""
    url = f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/{asset_id}/nwb.lindi.json"
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=lindi.LocalCache())
    nwbfile = NWBHDF5IO(file=f).read()
    sp = f["units/spike_times"][:]
    idx = f["units/spike_times_index"][:]
    starts = np.concatenate([[0], idx[:-1]])
    unit_spikes = [np.asarray(sp[s:e]) for s, e in zip(starts, idx)]
    tr = nwbfile.trials.to_dataframe()
    onsets = tr["start_time"].values
    freqs = tr["stim_frequency"].values.astype(float)
    order = np.argsort(onsets)
    return unit_spikes, onsets[order], freqs[order], nwbfile


def trial_bin_counts(spikes, onsets):
    """Spike counts in (trial, time bin) relative to tone onset."""
    counts = np.zeros((len(onsets), len(BIN_C)), dtype=np.int16)
    for ti, t0 in enumerate(onsets):
        lo, hi = np.searchsorted(spikes, [t0 + BIN_EDGES[0], t0 + BIN_EDGES[-1]])
        if hi > lo:
            c, _ = np.histogram(spikes[lo:hi] - t0, bins=BIN_EDGES)
            counts[ti] = c
    return counts

# %% [markdown]
# ## 3. Data Validation: Tone-Evoked Responses
#
# Before estimating STRFs we inspect the raw data for one session. The figure
# below shows, for an example unit: the spike raster aligned to tone onset with
# trials grouped and colored by tone frequency, the per-frequency PSTH, the
# unit's firing rate over the full session (a stability check), and a
# cross-validation of the alignment code in which the pooled perievent histogram
# computed with pynapple's `compute_perievent` is overlaid on the direct
# `searchsorted` binning used throughout this notebook (the two agree exactly).
# The evoked response is clearly visible as a frequency-dependent increase in
# firing roughly 20-60 ms after tone onset.

# %%
EX_SESSION = "LA11_ses2"
EX_UNIT = 50  # a strongly tuned unit in this session (see Figure 2)

unit_spikes, onsets, freqs, nwbfile = load_session(ASSETS[EX_SESSION])
print(f"{EX_SESSION}: {len(unit_spikes)} units, {len(onsets)} trials")
print("frequencies (kHz):", sorted(np.unique(freqs) / 1000))
print("tone duration (s):", np.unique(nwbfile.trials.to_dataframe()["stim_duration"]))

counts_ex = trial_bin_counts(unit_spikes[EX_UNIT], onsets)

# Cross-validate the alignment with pynapple's perievent machinery: the pooled
# perievent spike histogram must equal the direct searchsorted binning above.
nwb = nap.NWBFile(nwbfile)
pe = nap.compute_perievent(nwb["units"][EX_UNIT], nap.Ts(onsets), window=(-0.05, 0.15))
all_rel = (np.concatenate([pe[k].t for k in pe.keys()]) if len(pe.keys())
           else np.array([]))
psth_pp = np.histogram(all_rel, bins=BIN_EDGES)[0] / len(onsets) / BIN_S
psth_direct = counts_ex.sum(axis=0) / len(onsets) / BIN_S
xcheck_corr = np.corrcoef(psth_pp, psth_direct)[0, 1]
print(f"pynapple cross-check: correlation with direct binning = {xcheck_corr:.4f}")

fig = plt.figure(figsize=(14, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
cmap_f = {fv: c for fv, c in zip(FREQ_VALS, plt.cm.viridis(np.linspace(0, 0.9, 5)))}

ax = fig.add_subplot(gs[0, :])
order = np.argsort(freqs, kind="stable")
spk = unit_spikes[EX_UNIT]
win = (-0.05, 0.15)
for row, ti in enumerate(order):
    t0 = onsets[ti]
    lo, hi = np.searchsorted(spk, [t0 + win[0], t0 + win[1]])
    ts = spk[lo:hi] - t0
    ax.scatter(ts * 1000, np.full_like(ts, row), s=1, color=cmap_f[freqs[ti]], lw=0)
ax.set_xlim(win[0] * 1000, win[1] * 1000)
ax.axvline(0, color="k", lw=0.5)
ax.axvline(25, color="k", lw=0.5, ls="--")
ax.set_xlabel("time from tone onset (ms)")
ax.set_ylabel("trial (sorted by frequency)")
ax.set_title(f"unit {EX_UNIT} spike raster (color = tone frequency)")

ax2 = fig.add_subplot(gs[1, 0])
for fv in FREQ_VALS:
    m = freqs == fv
    ax2.plot(BIN_C * 1000, counts_ex[m].mean(axis=0) / BIN_S,
             color=cmap_f[fv], label=f"{int(fv / 1000)} kHz")
ax2.axvline(0, color="k", lw=0.5)
ax2.axvline(25, color="k", lw=0.5, ls="--")
ax2.set_xlabel("time from onset (ms)")
ax2.set_ylabel("rate (Hz)")
ax2.legend(frameon=False, fontsize=8)
ax2.set_title("per-frequency PSTH")

ax3 = fig.add_subplot(gs[1, 1])
t_edges = np.arange(onsets[0], onsets[-1], 1.0)
ax3.plot(0.5 * (t_edges[:-1] + t_edges[1:]), np.histogram(spk, t_edges)[0], lw=0.5)
ax3.set_xlabel("session time (s)")
ax3.set_ylabel("spikes / s")
ax3.set_title("unit firing rate across session")

ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(BIN_C * 1000, psth_direct, "k", lw=1.5, label="direct binning")
ax4.plot(BIN_C * 1000, psth_pp, "r", ls="--", lw=1.2, label="pynapple perievent")
ax4.axvline(0, color="k", lw=0.5)
ax4.axvline(25, color="k", lw=0.5, ls="--")
ax4.set_xlabel("time from onset (ms)")
ax4.set_ylabel("rate (Hz)")
ax4.legend(frameon=False, fontsize=8)
ax4.set_title(f"alignment cross-check (r = {xcheck_corr:.4f})")

fig.savefig("fig1_data_validation.png", dpi=150)
print("saved fig1_data_validation.png")

# %% [markdown]
# ## 4. Rate-Map STRF Estimation
#
# The model-free STRF of a unit is the baseline-subtracted mean firing rate in
# each (frequency, latency) bin: for every tone frequency we average the
# per-trial spike counts over trials, convert to Hz, and subtract the pre-onset
# baseline rate. Positive (red) values are excitatory subfields, negative (blue)
# values are suppressive subfields.
#
# We also compute two statistics per unit: a Wilcoxon signed-rank test of evoked
# versus baseline spike counts (tone responsiveness) and a Kruskal-Wallis test
# across the five frequencies (frequency tuning). Units passing both at
# p < 0.01 are carried forward as "tuned" units.

# %%
def rate_map_strf(counts, freqs):
    """Baseline-subtracted mean rate per (frequency, time bin). Returns (5, n_bins)."""
    out = np.zeros((len(FREQ_VALS), len(BIN_C)))
    for fi, fv in enumerate(FREQ_VALS):
        m = freqs == fv
        out[fi] = counts[m].mean(axis=0) / BIN_S - counts[m][:, BASE_MASK].mean() / BIN_S
    return out


def responsiveness_stats(counts, freqs):
    """Wilcoxon evoked-vs-baseline (min over freqs) + Kruskal-Wallis across freqs."""
    ev = counts[:, EVOKED_MASK].sum(axis=1)
    ba = counts[:, BASE_MASK].sum(axis=1) * (EVOKED_MASK.sum() / BASE_MASK.sum())
    best_p, per_freq_ev = 1.0, []
    for fv in FREQ_VALS:
        m = freqs == fv
        per_freq_ev.append(ev[m])
        if m.sum() > 10:
            try:
                best_p = min(best_p, stats.wilcoxon(ev[m], ba[m]).pvalue)
            except ValueError:
                pass
    try:
        kw_p = stats.kruskal(*per_freq_ev).pvalue
    except ValueError:
        kw_p = 1.0
    return best_p, kw_p


def characterize_strf(strf):
    """BF, peak latency, peak/trough rates, and separability of one STRF."""
    resp = strf[:, ~BASE_MASK & (BIN_C < 0.100)]   # 0-100 ms response window
    t = BIN_C[~BASE_MASK & (BIN_C < 0.100)]
    pk = np.unravel_index(np.argmax(resp), resp.shape)
    M = resp - resp.mean()
    sv = np.linalg.svd(M, compute_uv=False)
    sep = sv[0] ** 2 / np.sum(sv ** 2) if sv.sum() > 0 else np.nan
    return dict(bf=FREQ_VALS[pk[0]], peak_lat=t[pk[1]], peak_hz=resp[pk],
                min_hz=resp.min(), separability=sep)


def analyze_session_ratemaps(asset_id, label):
    """Load one session and compute rate-map STRFs, stats, and characters per unit."""
    unit_spikes, onsets, freqs, _ = load_session(asset_id)
    n_units = len(unit_spikes)
    counts_all = np.stack([trial_bin_counts(s, onsets) for s in unit_spikes])
    strfs = np.stack([rate_map_strf(counts_all[u], freqs) for u in range(n_units)])
    p_resp = np.zeros(n_units)
    p_tune = np.zeros(n_units)
    for u in range(n_units):
        p_resp[u], p_tune[u] = responsiveness_stats(counts_all[u], freqs)
    tuned = (p_resp < 0.01) & (p_tune < 0.01)
    chars = [characterize_strf(strfs[u]) for u in range(n_units)]
    print(f"[{label}] {n_units} units, {tuned.sum()} tuned "
          f"(responsive p<0.01 and KW p<0.01)")
    return dict(unit_spikes=unit_spikes, onsets=onsets, freqs=freqs,
                counts_all=counts_all, strfs=strfs, p_resp=p_resp, p_tune=p_tune,
                tuned=tuned, chars=chars, label=label)


sess = analyze_session_ratemaps(ASSETS[EX_SESSION], EX_SESSION)
strfs, tuned = sess["strfs"], sess["tuned"]
bf = np.array([c["bf"] for c in sess["chars"]])
peak_lat = np.array([c["peak_lat"] for c in sess["chars"]])

# --- Figure 2: example rate-map STRFs -----------------------------------------
tu = np.where(tuned)[0]
strong = tu[np.argsort(-np.array([c["peak_hz"] for c in sess["chars"]])[tu])]
picks = []
for fv in FREQ_VALS:  # strongest unit per best frequency, for diversity
    cand = [u for u in strong if bf[u] == fv]
    if cand:
        picks.append(cand[0])
for u in strong:
    if len(picks) >= 9:
        break
    if u not in picks:
        picks.append(u)

fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
for ax, u in zip(axes.flat, picks):
    vm = np.abs(strfs[u]).max()  # per-panel scaling: STRF magnitudes vary widely
    im = ax.imshow(strfs[u], aspect="auto", origin="lower", cmap="RdBu_r",
                   extent=[BIN_C[0] * 1000, BIN_C[-1] * 1000, -0.5, 4.5],
                   vmin=-vm, vmax=vm)
    ax.set_yticks(range(5))
    ax.set_yticklabels([str(int(fv / 1000)) for fv in FREQ_VALS], fontsize=8)
    ax.axvline(0, color="k", lw=0.4)
    ax.axvline(25, color="k", lw=0.4, ls="--")
    ax.set_title(f"unit {u} | BF {int(bf[u] / 1000)} kHz | peak {peak_lat[u] * 1000:.0f} ms",
                 fontsize=9)
    ax.set_xlabel("time (ms)", fontsize=8)
    if ax in axes[:, 0]:
        ax.set_ylabel("freq (kHz)", fontsize=8)
fig.suptitle("Rate-map STRFs of example tuned units (per-panel color scale)")
fig.colorbar(im, ax=axes, label="evoked - baseline (Hz)", shrink=0.8)
fig.savefig("fig2_example_strfs.png", dpi=150)
print("saved fig2_example_strfs.png")

# %% [markdown]
# ## 5. Regularized STRF Estimation with a Poisson GLM
#
# The rate map is a raw average and is therefore noisy, especially in bins with
# few spikes. A classical alternative is to estimate the STRF by regression: we
# model the spike count in each (trial, 5 ms bin) as Poisson with log rate given
# by a linear combination of basis functions over (frequency, latency), and fit
# the coefficients by maximum likelihood with a small ridge penalty. We use
# NeMoS with a tensor-product basis: 4 B-splines over log2 frequency and 9
# B-splines over latency, giving 36 features plus an intercept that absorbs the
# baseline rate. Because every unit in a session shares the same stimulus
# sequence, a single design matrix serves all units and we fit them jointly with
# `PopulationGLM`.
#
# Model performance is quantified on 20% held-out trials as McFadden's
# pseudo-R^2 relative to an intercept-only null model. The Poisson
# log-likelihoods include the log(y!) term in both models (it cancels in
# likelihood-ratio differences but not in the ratio form of pseudo-R^2).

# %%
FREQ_BASIS = nmo.basis.BSplineEval(n_basis_funcs=4)
TIME_BASIS = nmo.basis.BSplineEval(n_basis_funcs=9)


def build_design(freqs, bin_c_ms):
    """Tensor-product design matrix over (frequency, latency)."""
    fb = FREQ_BASIS.compute_features(LOGF)                 # (5, 4)
    tb = TIME_BASIS.compute_features(bin_c_ms)             # (n_bins, 9)
    fidx = np.searchsorted(FREQ_VALS, freqs)
    Ft = fb[fidx]                                          # (n_trials, 4)
    X3 = np.einsum("ta,nb->tnab", Ft, tb).reshape(len(freqs), len(bin_c_ms), -1)
    return X3.reshape(-1, X3.shape[-1]), fb, tb


def fit_glm_strfs(sess):
    """Fit one Poisson GLM per tuned unit (shared design). Returns rate maps + pseudo-R2."""
    counts_all, freqs = sess["counts_all"], sess["freqs"]
    tu = np.where(sess["tuned"])[0]
    n_trials, n_bins = len(sess["onsets"]), len(BIN_C)
    X, fb, tb = build_design(freqs, BIN_C * 1000)
    rng = np.random.default_rng(0)
    tr_trials = rng.random(n_trials) < 0.8
    m = np.repeat(tr_trials, n_bins)
    Y = counts_all[tu].transpose(1, 2, 0).reshape(-1, len(tu)).astype(float)

    model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(),
                                  regularizer_strength=1e-5,
                                  solver_name="LBFGS",
                                  solver_kwargs=dict(maxiter=5000, tol=1e-8))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model.fit(X[m], Y[m])
    converged = not any("did not converge" in str(w.message) for w in wlist)

    null = nmo.glm.PopulationGLM(solver_name="LBFGS",
                                 solver_kwargs=dict(maxiter=2000))
    X0 = np.ones((X.shape[0], 1))
    null.fit(X0[m], Y[m])

    Yte = Y[~m]
    r_m = np.clip(np.asarray(model.predict(X[~m])), 1e-12, None)
    r_0 = np.clip(np.asarray(null.predict(X0[~m])), 1e-12, None)
    pll = lambda y, r: (y * np.log(r) - r - gammaln(y + 1)).mean(axis=0)
    pseudo_r2 = 1.0 - pll(Yte, r_m) / pll(Yte, r_0)

    Xp = np.einsum("fa,nb->fnab", fb, tb).reshape(len(fb) * len(tb), -1)
    rate = np.asarray(model.predict(Xp)) / BIN_S            # Hz
    glm_rate = np.full_like(sess["strfs"], np.nan)
    glm_rate[tu] = rate.reshape(len(fb), len(tb), len(tu)).transpose(2, 0, 1)

    out = np.full(len(sess["tuned"]), np.nan)
    out[tu] = pseudo_r2
    print(f"[{sess['label']}] GLM converged: {converged}; "
          f"median test pseudo-R2 = {np.nanmedian(pseudo_r2):.3f}")
    return glm_rate, out


glm_rate, pseudo_r2 = fit_glm_strfs(sess)

# --- Figure 3: GLM STRF vs data rate map --------------------------------------
tu2 = np.where(sess["tuned"] & np.isfinite(pseudo_r2))[0]
top = tu2[np.argsort(-pseudo_r2[tu2])[:3]]

fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
for r, u in enumerate(top):
    g = glm_rate[u] - glm_rate[u][:, BASE_MASK].mean()
    vm = np.abs(strfs[u]).max()
    for c, (M, ttl) in enumerate([(strfs[u], "data rate map"), (g, "GLM STRF")]):
        ax = axes[r, c]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r",
                       extent=[BIN_C[0] * 1000, BIN_C[-1] * 1000, -0.5, 4.5],
                       vmin=-vm, vmax=vm)
        ax.set_yticks(range(5))
        ax.set_yticklabels([str(int(fv / 1000)) for fv in FREQ_VALS], fontsize=8)
        ax.axvline(0, color="k", lw=0.4)
        ax.axvline(25, color="k", lw=0.4, ls="--")
        ax.set_title(f"unit {u} {ttl}", fontsize=9)
        if c == 0:
            ax.set_ylabel("freq (kHz)", fontsize=8)
    ax = axes[r, 2]
    bfi = int(np.argmin(np.abs(FREQ_VALS - bf[u])))
    ax.plot(BIN_C * 1000, strfs[u][bfi], "k", label="data")
    ax.plot(BIN_C * 1000, g[bfi], "r", label="GLM")
    ax.axvline(0, color="k", lw=0.4)
    ax.axvline(25, color="k", lw=0.4, ls="--")
    ax.set_title(f"BF={int(bf[u] / 1000)} kHz slice | pseudo-$R^2$={pseudo_r2[u]:.3f}",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("time (ms)", fontsize=8)
fig.suptitle("Poisson GLM STRF (tensor-product B-spline basis, ridge) vs raw rate map")
fig.savefig("fig3_glm_strf.png", dpi=150)
print("saved fig3_glm_strf.png")

# %% [markdown]
# ## 6. Population Analysis Across Six Sessions
#
# We now repeat the rate-map and GLM analysis for six sessions from five mice
# (LA3, LA8, LA9, LA11, LA12) and pool the tuned units. For each unit we record
# the best frequency (BF), the peak latency, the prominence of the suppressive
# subfield, the GLM pseudo-R^2, and the separability index. The separability
# index is the fraction of STRF variance explained by the best rank-1
# approximation (outer product of a spectral profile and a temporal profile);
# values near 1 mean the STRF is well described as "which frequencies, weighted
# the same way at every latency", while low values indicate spectrotemporal
# interactions such as frequency sweeps. This takes several minutes because of
# the per-session GLM fits.

# %%
results = {}
for label, aid in tqdm(ASSETS.items(), desc="sessions"):
    if label == EX_SESSION:
        s = sess  # already analyzed above
        s["glm_rate"], s["pseudo_r2"] = glm_rate, pseudo_r2
    else:
        s = analyze_session_ratemaps(aid, label)
        s["glm_rate"], s["pseudo_r2"] = fit_glm_strfs(s)
    results[label] = s
    np.savez(f"session_{label}.npz",
             strfs=s["strfs"], glm_rate=s["glm_rate"], pseudo_r2=s["pseudo_r2"],
             p_resp=s["p_resp"], p_tune=s["p_tune"], tuned=s["tuned"],
             bf=np.array([c["bf"] for c in s["chars"]]),
             peak_lat=np.array([c["peak_lat"] for c in s["chars"]]),
             peak_hz=np.array([c["peak_hz"] for c in s["chars"]]),
             min_hz=np.array([c["min_hz"] for c in s["chars"]]),
             separability=np.array([c["separability"] for c in s["chars"]]))

# --- pool tuned units ----------------------------------------------------------
pool = {}
for key in ["strfs", "glm_rate"]:
    pool[key] = np.concatenate([results[l][key][results[l]["tuned"]] for l in ASSETS])
for key in ["pseudo_r2"]:
    pool[key] = np.concatenate([results[l][key][results[l]["tuned"]] for l in ASSETS])
for key in ["bf", "peak_lat", "peak_hz", "min_hz", "separability"]:
    pool[key] = np.concatenate(
        [np.array([c[key] for c in results[l]["chars"]])[results[l]["tuned"]]
         for l in ASSETS]
    )
n_all = sum(len(results[l]["tuned"]) for l in ASSETS)
n_tuned = sum(int(results[l]["tuned"].sum()) for l in ASSETS)
print(f"pooled: {n_tuned} tuned units out of {n_all} "
      f"({100 * n_tuned / n_all:.0f}%)")

# %% [markdown]
# ### Population figures

# %%
bf_p = pool["bf"]
lat_p = pool["peak_lat"] * 1000
sep_p = pool["separability"]
pk_p = pool["peak_hz"]
mn_p = pool["min_hz"]
pr2_p = pool["pseudo_r2"]
supp_p = -mn_p / (pk_p - mn_p + 1e-9)

fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

ax = axes[0, 0]
cnt, _ = np.histogram(bf_p / 1000, bins=np.array([1, 3, 6, 12, 24, 40]))
ax.bar(range(5), cnt, tick_label=["2-3", "4-6", "8-12", "16-24", "32-40"])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("# tuned units")
ax.set_title(f"BF distribution (n={len(bf_p)} tuned)")

ax = axes[0, 1]
ax.hist(lat_p, bins=np.arange(5, 105, 5), color="C0")
ax.axvline(np.median(lat_p), color="r", ls="--", label=f"median {np.median(lat_p):.0f} ms")
ax.set_xlabel("peak latency (ms)")
ax.set_ylabel("# units")
ax.legend(frameon=False)
ax.set_title("STRF peak latency")

ax = axes[0, 2]
ax.hist(sep_p, bins=np.linspace(0.2, 1.0, 33), color="C2")
ax.axvline(np.median(sep_p), color="r", ls="--", label=f"median {np.median(sep_p):.2f}")
ax.set_xlabel("separability index (rank-1 variance fraction)")
ax.set_ylabel("# units")
ax.legend(frameon=False)
ax.set_title("STRF separability (SVD)")

ax = axes[1, 0]
ax.scatter(bf_p / 1000, lat_p, s=6, alpha=0.4)
ax.set_xscale("log")
ax.set_xticks([2, 4, 8, 16, 32])
ax.set_xticklabels([2, 4, 8, 16, 32])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("peak latency (ms)")
ax.set_title("latency vs best frequency")

ax = axes[1, 1]
ax.hist(supp_p, bins=np.linspace(0, 1, 33), color="C3")
ax.axvline(np.median(supp_p), color="r", ls="--", label=f"median {np.median(supp_p):.2f}")
ax.set_xlabel("suppression index |min| / (max + |min|)")
ax.set_ylabel("# units")
ax.legend(frameon=False)
ax.set_title("suppressive subfield prominence")

ax = axes[1, 2]
ax.hist(pr2_p[np.isfinite(pr2_p)], bins=np.linspace(0, 0.35, 35), color="C4")
ax.axvline(np.nanmedian(pr2_p), color="r", ls="--",
           label=f"median {np.nanmedian(pr2_p):.3f}")
ax.set_xlabel("GLM test pseudo-$R^2$")
ax.set_ylabel("# units")
ax.legend(frameon=False)
ax.set_title("GLM STRF predictive performance")

fig.savefig("fig4_population.png", dpi=150)
print("saved fig4_population.png")

# --- Figure 5: most separable vs least separable STRFs -------------------------
order = np.argsort(sep_p)
strong = pk_p > np.percentile(pk_p, 50)
idx_sep = [i for i in order[::-1] if strong[i]][:4]
idx_insep = [i for i in order if strong[i]][:4]

fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
for r, idxs, ttl in [(0, idx_sep, "separable"), (1, idx_insep, "inseparable")]:
    for c, i in enumerate(idxs):
        ax = axes[r, c]
        vm = np.abs(pool["strfs"][i]).max()
        ax.imshow(pool["strfs"][i], aspect="auto", origin="lower", cmap="RdBu_r",
                  extent=[BIN_C[0] * 1000, BIN_C[-1] * 1000, -0.5, 4.5],
                  vmin=-vm, vmax=vm)
        ax.set_yticks(range(5))
        ax.set_yticklabels([str(int(fv / 1000)) for fv in FREQ_VALS], fontsize=8)
        ax.axvline(0, color="k", lw=0.4)
        ax.axvline(25, color="k", lw=0.4, ls="--")
        ax.set_title(f"{ttl} | sep={sep_p[i]:.2f} | BF {int(bf_p[i] / 1000)} kHz",
                     fontsize=9)
        if c == 0:
            ax.set_ylabel("freq (kHz)")
        ax.set_xlabel("time (ms)", fontsize=8)
fig.suptitle("Most separable (top) vs least separable (bottom) STRFs among tuned units")
fig.savefig("fig5_separability.png", dpi=150)
print("saved fig5_separability.png")

# %% [markdown]
# ## 7. Summary of Findings

# %%
print("=" * 70)
print(f"Sessions: {len(ASSETS)} ({len(ASSETS)} files, 5 mice), units total: {n_all}")
print(f"Tuned units (responsive + frequency-selective, p<0.01): {n_tuned} "
      f"({100 * n_tuned / n_all:.0f}%)")
print(f"Best frequency distribution (kHz bins 2-3/4-6/8-12/16-24/32-40): {cnt}")
print(f"Peak latency: median {np.median(lat_p):.0f} ms "
      f"(IQR {np.percentile(lat_p, 25):.0f}-{np.percentile(lat_p, 75):.0f} ms)")
print(f"Separability: median {np.median(sep_p):.2f}; "
      f"fraction > 0.7: {np.mean(sep_p > 0.7):.2f}")
print(f"Suppression index: median {np.median(supp_p):.2f}; "
      f"fraction > 0.2: {np.mean(supp_p > 0.2):.2f}")
print(f"GLM test pseudo-R2: median {np.nanmedian(pr2_p):.3f}; "
      f"fraction > 0.05: {np.nanmean(pr2_p > 0.05):.2f}")
print("=" * 70)

# %% [markdown]
# The typical auditory cortical neuron in this dataset has a compact STRF: a
# single excitatory subfield at its best frequency appearing 15-40 ms after tone
# onset, frequently flanked by a suppressive subfield at neighboring frequencies
# or later times. Most STRFs are highly separable, meaning frequency tuning and
# temporal response profile are largely independent, while a minority show
# spectrotemporal interactions. The Poisson GLM recovers the same structure as
# the raw rate maps while denoising them, and its held-out pseudo-R^2 confirms
# that the STRF captures a substantial fraction of the stimulus-driven variance
# for well-tuned units.
