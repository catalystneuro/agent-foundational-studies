# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# This notebook demonstrates frequency tuning of neurons in mouse auditory cortex
# using data from the DANDI Archive, dandiset
# [000986](https://dandiarchive.org/dandiset/000986) ("Auditory cortex Neuropixels
# recordings and pupil diameter traces from mice during passive exposure to pure
# tones", Jaramillo lab). Mice passively heard 25 ms pure tones at 60 dB SPL,
# drawn from five frequencies spaced one octave apart (2, 4, 8, 16, 32 kHz) in
# random order, while Neuropixels probes recorded from auditory cortex.
#
# The analysis proceeds in four steps:
#
# 1. Stream one example session with LINDI and verify that tones evoke
#    time-locked spiking (rasters and peri-stimulus time histograms).
# 2. Quantify per-trial responses in a 5-60 ms window after tone onset and
#    build a tuning curve (evoked firing rate versus frequency) for every unit.
# 3. Fit a Poisson GLM with a B-spline basis over log2 frequency (NeMoS) to get
#    a smooth tuning estimate and a per-unit measure of explained variance.
# 4. Repeat across all 15 sessions (5 mice, 1564 units) and summarize the
#    population: best-frequency distribution, tuning sharpness, and
#    consistency across sessions.
#
# Everything is computed with Pynapple/NWB for data access, NumPy/SciPy for
# statistics, and NeMoS for the GLM. Figures are written to `figures/`.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import lindi
from pynwb import NWBHDF5IO
from scipy import stats as sstats
from scipy.special import gammaln
from tqdm import tqdm
import nemos as nmo
import os

os.makedirs("figures", exist_ok=True)

# %% [markdown]
# The dandiset contains 15 sessions from 5 mice. Each session is a single NWB
# file with a `trials` table (tone onset/offset, frequency, duration,
# amplitude), a `units` table (sorted spike times), and behavior streams
# (pupil diameter, running speed) that we do not use here.

# %%
ASSETS = {
    "sub-LA3_ses-3": "5e111970-9331-41d0-81b2-829e1c0f8040",
    "sub-LA8_ses-1": "60303460-38be-44a0-951e-82c7957d1217",
    "sub-LA8_ses-2": "ce06d820-e471-4413-a3a8-9c0b21da8680",
    "sub-LA9_ses-1": "d19d0ca3-7c9a-41fe-bd97-b4ee8899a612",
    "sub-LA9_ses-3": "3d1c45a5-a26a-4083-ab68-57f3ecca1357",
    "sub-LA9_ses-4": "ea13b270-0975-4d88-bdf4-17e71efb009b",
    "sub-LA9_ses-5": "eafb5f48-0ed7-414c-a3cc-3b662a4506a2",
    "sub-LA11_ses-1": "aacd1c8a-73f7-469e-bf08-0afd5c1052f9",
    "sub-LA11_ses-2": "b8d3abca-0e78-4df1-9a51-d122a383be63",
    "sub-LA11_ses-3": "a7c6cce3-442a-4dc4-aa86-300558ae1909",
    "sub-LA11_ses-4": "36bbc777-6708-45e5-85f2-48f56b84496d",
    "sub-LA12_ses-1": "eb82c81a-87a0-40a4-b70e-535ac0909c86",
    "sub-LA12_ses-2": "d0986739-6cc0-4bc7-9d2f-8363c233ed64",
    "sub-LA12_ses-3": "ffb5c0b9-0d5b-418a-ad52-1c786818a7e5",
    "sub-LA12_ses-4": "b35476db-13dc-4569-a4ed-4e839adde857",
}
EXAMPLE_SESS = "sub-LA11_ses-2"

RESP = (0.005, 0.060)   # response window, seconds after tone onset
BASE = (-0.050, 0.0)    # baseline window, seconds before onset
FREQS = np.array([2000., 4000., 8000., 16000., 32000.])
LOG2F = np.log2(FREQS / 1000.)
PSTH_EDGES = np.arange(-0.1, 0.3 + 0.005, 0.005)
PSTH_CENTERS = 0.5 * (PSTH_EDGES[:-1] + PSTH_EDGES[1:])
GRID = np.linspace(LOG2F[0], LOG2F[-1], 100)  # fine grid for GLM curves

FREQ_COLORS = {2000.: "#2166ac", 4000.: "#67a9cf", 8000.: "#1a9850",
               16000.: "#f46d43", 32000.: "#762a83"}
FREQ_LABELS = [f"{int(f/1000)} kHz" for f in FREQS]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False,
})

LOCAL_CACHE = lindi.LocalCache()


# %%
def load_session(asset_id):
    """Stream one NWB file via LINDI; return trials dataframe and spike times."""
    url = (f"https://lindi.neurosift.org/dandi/dandisets/000986/assets/"
           f"{asset_id}/nwb.lindi.json")
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=LOCAL_CACHE)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    trials = nwbfile.intervals["trials"].to_dataframe()
    # bulk-read the flat spike-time arrays once; slicing per unit over LINDI
    # is far too slow
    st_idx = f['units/spike_times_index'][:]
    st_all = f['units/spike_times'][:]
    bounds = np.concatenate([[0], st_idx])
    spk_list = [st_all[bounds[i]:bounds[i + 1]] for i in range(len(st_idx))]
    return trials, spk_list


def count_in_windows(spk, starts, w):
    lo = np.searchsorted(spk, starts + w[0])
    hi = np.searchsorted(spk, starts + w[1])
    return hi - lo


def rel_times(spk, ons, win):
    """Spike times relative to the most recent onset (vectorized)."""
    idx = np.searchsorted(ons, spk, side="right") - 1
    valid = idx >= 0
    rel = spk[valid] - ons[idx[valid]]
    tr = idx[valid]
    keep = (rel >= win[0]) & (rel < win[1])
    return rel[keep], tr[keep]


# %% [markdown]
# ## One Example Session: Stimulus Design and Evoked Responses
#
# We start with session `sub-LA11_ses-2` (132 units, 7448 trials). Tones arrive
# every ~0.8 s in random frequency order. The raster and PSTH below show that
# tone onsets evoke a brisk, time-locked response peaking 10-30 ms after onset,
# and that the strength of that response depends on frequency for individual
# units.

# %%
trials_ex, spk_ex = load_session(ASSETS[EXAMPLE_SESS])
onsets_ex = trials_ex["start_time"].values
freq_ex = trials_ex["stim_frequency"].values
print(f"{EXAMPLE_SESS}: {len(spk_ex)} units, {len(trials_ex)} trials")
print("trials per frequency:", np.bincount(
    np.searchsorted(FREQS, freq_ex), minlength=5).tolist())

# %% [markdown]
# We pick one example unit per best frequency for plotting. The selection is
# principled: among units whose firing is significantly modulated by frequency
# (Kruskal-Wallis p < 0.001) and that have a clear excitatory peak (more than
# 5 Hz above baseline), take the unit where tone frequency explains the most
# variance, measured by the McFadden pseudo-R² of a Poisson GLM on log2
# frequency. The Kruskal-Wallis test and the GLM are defined in the next
# section; here we only use them to choose clean examples.

# %%
# quick per-unit stats on the example session, used only to pick example units
n_ex = len(spk_ex)
resp_ex = np.zeros((len(onsets_ex), n_ex))
base_ex = np.zeros((len(onsets_ex), n_ex))
for i, spk in enumerate(spk_ex):
    resp_ex[:, i] = count_in_windows(spk, onsets_ex, RESP)
    base_ex[:, i] = count_in_windows(spk, onsets_ex, BASE)
resp_rate_ex = resp_ex / (RESP[1] - RESP[0])
baseline_ex = base_ex.mean(axis=0) / (BASE[1] - BASE[0])
tuning_ex = np.stack([resp_rate_ex[freq_ex == fr].mean(axis=0) for fr in FREQS], axis=1)
evoked_ex = tuning_ex - baseline_ex[:, None]
kw_ex = np.array([sstats.kruskal(*[resp_ex[freq_ex == fr, u] for fr in FREQS]).pvalue
                  for u in range(n_ex)])
bf_ex = np.argmax(evoked_ex, axis=1)

# quick GLM on the example session, same model as in the GLM section below,
# used here only to rank candidate example units
_basis_ex = nmo.basis.BSplineEval(n_basis_funcs=4)
_X_ex = _basis_ex.compute_features(np.log2(freq_ex / 1000.))
_glm_ex = nmo.glm.PopulationGLM(solver_name="LBFGS",
                                solver_kwargs={"tol": 1e-6, "maxiter": 2000})
_glm_ex.fit(_X_ex, resp_ex)
_mu_m = np.asarray(_glm_ex.predict(_X_ex))
_mu_0 = resp_ex.mean(axis=0, keepdims=True) * np.ones_like(resp_ex)
_lyf = gammaln(resp_ex + 1)
_ll_m = (resp_ex * np.log(_mu_m + 1e-12) - _mu_m - _lyf).sum(axis=0)
_ll_0 = (resp_ex * np.log(_mu_0 + 1e-12) - _mu_0 - _lyf).sum(axis=0)
pr2_ex = np.clip(np.where(_ll_0 < 0, 1 - _ll_m / _ll_0, 0.0), 0, 1)

example_units = []
for fi in range(5):
    cand = np.where((bf_ex == fi) & (kw_ex < 0.001) & (evoked_ex.max(axis=1) > 5))[0]
    if not len(cand):
        cand = np.where((bf_ex == fi) & (kw_ex < 0.001))[0]
    example_units.append(cand[np.argmax(pr2_ex[cand])])
print("example units (one per best frequency):", example_units)

# %%
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.42, wspace=0.35)

# A: stimulus sequence (first 80 trials)
ax = fig.add_subplot(gs[0, 0])
nshow = 80
t0 = onsets_ex[0]
for fi, fr in enumerate(FREQS):
    m = (freq_ex == fr)[:nshow]
    ax.scatter((onsets_ex[:nshow][m] - t0), np.zeros(m.sum()) + fi,
               marker="|", s=60, color=FREQ_COLORS[fr], lw=1.5)
ax.set_yticks(range(5))
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("Time in session (s)")
ax.set_title("A  Tone presentation sequence\n(first 80 trials, random order)")
ax.set_ylim(-0.7, 4.7)

# B: raster of the 8 kHz example unit, grouped by frequency
ax = fig.add_subplot(gs[0, 1:])
u_raster = example_units[2]
spk = spk_ex[u_raster]
N_PER_FREQ = 150
ytick = 0
yticks = []
rng = np.random.default_rng(0)
for fi, fr in enumerate(FREQS):
    tr_idx = np.where(freq_ex == fr)[0]
    show = rng.choice(tr_idx, size=N_PER_FREQ, replace=False)
    ons = np.sort(onsets_ex[show])  # must be sorted for searchsorted
    rel, tr = rel_times(spk, ons, win=(-0.05, 0.1))
    ax.scatter(rel * 1000, tr + ytick, s=3, color=FREQ_COLORS[fr], rasterized=True)
    yticks.append(ytick + N_PER_FREQ / 2)
    ytick += N_PER_FREQ
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvspan(0, 25, color="gray", alpha=0.15)
ax.set_yticks(yticks)
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlim(-50, 100)
ax.set_ylim(0, ytick)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Trials (grouped by frequency)")
ax.set_title(f"B  Example unit {u_raster}: spike raster by frequency\n"
             f"({N_PER_FREQ} random trials per frequency)")

# C: PSTH of the same unit per frequency
ax = fig.add_subplot(gs[1, 0])
edges = np.arange(-0.1, 0.15 + 0.005, 0.005)
centers = 0.5 * (edges[:-1] + edges[1:])
for fi, fr in enumerate(FREQS):
    ons = np.sort(onsets_ex[freq_ex == fr])
    rel, _ = rel_times(spk, ons, win=(edges[0], edges[-1]))
    h = np.histogram(rel, bins=edges)[0] / (len(ons) * 0.005)
    ax.plot(centers * 1000, h, color=FREQ_COLORS[fr], lw=1.5, label=FREQ_LABELS[fi])
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvspan(0, 25, color="gray", alpha=0.15)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title(f"C  Unit {u_raster}: PSTH by frequency")
ax.legend(fontsize=8, frameon=False)

# D: population PSTH, example session
ax = fig.add_subplot(gs[1, 1:])
for fi, fr in enumerate(FREQS):
    ons = np.sort(onsets_ex[freq_ex == fr])
    tot = np.zeros(len(PSTH_CENTERS))
    for spk_i in spk_ex:
        rel, _ = rel_times(spk_i, ons, win=(PSTH_EDGES[0], PSTH_EDGES[-1]))
        tot += np.histogram(rel, bins=PSTH_EDGES)[0]
    ax.plot(PSTH_CENTERS * 1000, tot / (len(ons) * n_ex * 0.005),
            color=FREQ_COLORS[fr], lw=1.8, label=FREQ_LABELS[fi])
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvspan(0, 25, color="gray", alpha=0.15)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population rate (Hz/unit)")
ax.set_title(f"D  Population PSTH, all {n_ex} units\n({EXAMPLE_SESS})")
ax.legend(title="Tone frequency", fontsize=8, frameon=False)

fig.savefig("figures/fig1_stimulus_and_responses.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# The example unit responds almost exclusively to the 8 kHz tone, and the
# population PSTH shows that response magnitude depends on frequency across
# the population as well. To quantify this per unit, we count spikes in a
# 5-60 ms response window on every trial and compare across frequencies.

# %% [markdown]
# ## All Sessions: Tuning Curves, Statistics, and GLM Fits
#
# For each of the 15 sessions we compute, for every unit:
#
# * **Tuning curve**: mean firing rate in the response window for each
#   frequency, minus the mean rate in a -50-0 ms baseline window.
# * **Best frequency (BF)**: the frequency with the largest evoked response.
# * **Sparseness**: Vinje-Gallant sparseness of the evoked tuning curve
#   (0 = equal response to all frequencies, 1 = response to exactly one).
# * **Frequency modulation**: Kruskal-Wallis test of response-window spike
#   counts across the five frequency groups.
# * **Responsiveness**: Wilcoxon signed-rank test of response-window versus
#   baseline-window counts across all trials.
# * **GLM fit**: a Poisson GLM (NeMoS `PopulationGLM`) with a 4-knot B-spline
#   basis over log2 frequency, fit to per-trial response counts. This yields a
#   smooth tuning curve and a per-unit McFadden pseudo-R² against an
#   intercept-only null model. The full Poisson log-likelihood (including the
#   -log(y!) term) is used, since the term cancels in likelihood-ratio
#   differences but not in McFadden's ratio.
# * **Population PSTH** per frequency.

# %%
def analyze_session(name, asset_id):
    trials, spk_list = load_session(asset_id)
    onsets = trials["start_time"].values
    freq_trial = trials["stim_frequency"].values
    n_trials = len(onsets)
    n_units = len(spk_list)

    resp_counts = np.zeros((n_trials, n_units))
    base_counts = np.zeros((n_trials, n_units))
    for i, spk in enumerate(spk_list):
        resp_counts[:, i] = count_in_windows(spk, onsets, RESP)
        base_counts[:, i] = count_in_windows(spk, onsets, BASE)

    resp_rate = resp_counts / (RESP[1] - RESP[0])
    baseline = base_counts.mean(axis=0) / (BASE[1] - BASE[0])

    tuning = np.zeros((n_units, len(FREQS)))
    tuning_sem = np.zeros((n_units, len(FREQS)))
    for fi, fr in enumerate(FREQS):
        m = freq_trial == fr
        tuning[:, fi] = resp_rate[m].mean(axis=0)
        tuning_sem[:, fi] = resp_rate[m].std(axis=0) / np.sqrt(m.sum())
    evoked = tuning - baseline[:, None]

    kw_p = np.array([
        sstats.kruskal(*[resp_counts[freq_trial == fr, u] for fr in FREQS]).pvalue
        for u in range(n_units)
    ])
    wilcox_p = np.array([
        sstats.wilcoxon(resp_counts[:, u], base_counts[:, u]).pvalue
        if (resp_counts[:, u] != base_counts[:, u]).any() else 1.0
        for u in range(n_units)
    ])

    evoked_clip = np.clip(evoked, 0, None)
    bf_idx = np.argmax(evoked, axis=1)
    s1 = evoked_clip.sum(axis=1)
    s2 = (evoked_clip ** 2).sum(axis=1)
    n_f = len(FREQS)
    sparseness = np.zeros(n_units)
    ok = s2 > 0
    sparseness[ok] = (1 - s1[ok] ** 2 / (n_f * s2[ok])) / (1 - 1 / n_f)

    psth = np.zeros((len(FREQS), len(PSTH_CENTERS)))
    for fi, fr in enumerate(FREQS):
        ons = np.sort(onsets[freq_trial == fr])
        tot = np.zeros(len(PSTH_CENTERS))
        for spk in spk_list:
            rel, _ = rel_times(spk, ons, win=(PSTH_EDGES[0], PSTH_EDGES[-1]))
            tot += np.histogram(rel, bins=PSTH_EDGES)[0]
        psth[fi] = tot / (len(ons) * n_units * 0.005)

    # Poisson GLM with B-spline basis over log2 frequency
    basis = nmo.basis.BSplineEval(n_basis_funcs=4)
    X = basis.compute_features(np.log2(freq_trial / 1000.))
    glm = nmo.glm.PopulationGLM(solver_name="LBFGS",
                                solver_kwargs={"tol": 1e-6, "maxiter": 2000})
    glm.fit(X, resp_counts)
    mu_model = np.asarray(glm.predict(X))
    y = resp_counts
    mu_null = y.mean(axis=0, keepdims=True) * np.ones_like(y)
    eps = 1e-12
    log_yfac = gammaln(y + 1)
    ll_model = (y * np.log(mu_model + eps) - mu_model - log_yfac).sum(axis=0)
    ll_null = (y * np.log(mu_null + eps) - mu_null - log_yfac).sum(axis=0)
    pseudo_r2 = np.where(ll_null < 0, 1 - ll_model / ll_null, 0.0)
    pseudo_r2 = np.clip(pseudo_r2, 0, 1)
    glm_curves = np.asarray(glm.predict(basis.compute_features(GRID))) / (RESP[1] - RESP[0])
    glm_bf_log2 = GRID[np.argmax(glm_curves, axis=0)]

    return dict(name=name, n_units=n_units, baseline=baseline, tuning=tuning,
                tuning_sem=tuning_sem, evoked=evoked, kw_p=kw_p, wilcox_p=wilcox_p,
                bf_idx=bf_idx, sparseness=sparseness, psth=psth,
                pseudo_r2=pseudo_r2, glm_curves=glm_curves, glm_bf_log2=glm_bf_log2)


results = {}
for name, aid in tqdm(ASSETS.items(), desc="sessions"):
    results[name] = analyze_session(name, aid)
    r = results[name]
    tqdm.write(f"{name}: {r['n_units']} units, "
               f"{np.mean(r['wilcox_p'] < 0.05)*100:.0f}% responsive, "
               f"{np.mean(r['kw_p'] < 0.05)*100:.0f}% freq-modulated")

# %% [markdown]
# ## Population Results
#
# Pooling all 1564 units, we first look at example tuning curves with their
# GLM fits, then at the whole population as a heatmap sorted by best
# frequency.

# %%
SESS = list(results.keys())
EX_IDX = SESS.index(EXAMPLE_SESS)
cat = lambda key: np.concatenate([results[s][key] for s in SESS])
session_idx = np.concatenate([[i] * results[SESS[i]]["n_units"] for i in range(len(SESS))])
baseline = cat("baseline")
tuning = cat("tuning")
tuning_sem = cat("tuning_sem")
evoked = cat("evoked")
kw_p = cat("kw_p")
wilcox_p = cat("wilcox_p")
bf_idx = cat("bf_idx")
sparseness = cat("sparseness")
pseudo_r2 = cat("pseudo_r2")
glm_bf_log2 = cat("glm_bf_log2")
glm_curves = np.concatenate([results[s]["glm_curves"].T for s in SESS])  # units x grid
psth = np.stack([results[s]["psth"] for s in SESS])                      # sess x freq x time

kw_sig = kw_p < 0.05
resp_sig = wilcox_p < 0.05
tuned = kw_sig & resp_sig
ex_mask = session_idx == EX_IDX
print(f"total units: {len(kw_sig)}")
print(f"responsive (Wilcoxon p<0.05):        {resp_sig.mean()*100:.1f}%")
print(f"freq-modulated (Kruskal-Wallis p<0.05): {kw_sig.mean()*100:.1f}%")
print(f"both:                                {tuned.mean()*100:.1f}%")

# %%
# Figure 2: example tuning curves with GLM overlay
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
axes = axes.flat
for k, u in enumerate(example_units[:5]):
    ax = axes[k]
    ev = evoked[ex_mask][u]
    ax.errorbar(LOG2F, ev, yerr=tuning_sem[ex_mask][u], marker="o", ms=5,
                capsize=3, color="k", lw=1.2, label="data (mean±SEM)")
    ax.plot(GRID, glm_curves[ex_mask][u] - baseline[ex_mask][u],
            color="#d01c8b", lw=2, label="Poisson GLM fit")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(LOG2F)
    ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
    bf = FREQS[np.argmax(ev)]
    ax.set_title(f"unit {u}, BF = {int(bf/1000)} kHz, "
                 f"GLM pseudo-$R^2$ = {pseudo_r2[ex_mask][u]:.3f}", fontsize=10)
    ax.set_xlabel("Tone frequency (kHz)")
    ax.set_ylabel("Evoked rate − baseline (Hz)")
axes[5].axis("off")
axes[5].plot([], [], marker="o", color="k", lw=1.2, label="data (mean±SEM)")
axes[5].plot([], [], color="#d01c8b", lw=2, label="Poisson GLM fit")
axes[5].legend(loc="upper left", frameon=False, fontsize=11)
axes[5].text(0.0, 0.45, "One example unit per best\nfrequency, from session\n"
                        f"{EXAMPLE_SESS}.\n\nResponse window: 5–60 ms\n"
                        "after tone onset.\nBaseline: −50–0 ms.", fontsize=11)
fig.suptitle("Frequency tuning curves: example units", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig2_example_tuning_curves.png")
plt.close(fig)

# %% [markdown]
# Each example unit responds most strongly to one frequency and less to
# frequencies further away, the classic signature of auditory frequency
# tuning. The GLM's smooth curves track the raw means closely.

# %%
# Figure 3: population heatmap sorted by best frequency
norm = np.clip(evoked, 0, None)
row_max = norm.max(axis=1, keepdims=True)
norm = np.divide(norm, row_max, out=np.zeros_like(norm), where=row_max > 0)
order = np.lexsort((-sparseness, bf_idx))
fig, ax = plt.subplots(figsize=(6.5, 7))
im = ax.imshow(norm[order], aspect="auto", cmap="viridis",
               extent=[LOG2F[0] - 0.5, LOG2F[-1] + 0.5, len(norm), 0])
ax.set_xticks(LOG2F)
ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel("Units (sorted by best frequency)")
cb = plt.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Normalized evoked response")
counts = np.bincount(bf_idx[order], minlength=5)
for b in np.cumsum(counts)[:-1]:
    ax.axhline(b, color="white", lw=0.6, alpha=0.7)
ax.set_title(f"Population frequency tuning (n = {len(norm)} units,\n"
             f"15 sessions, 5 mice; each row normalized to its max)")
fig.tight_layout()
fig.savefig("figures/fig3_population_heatmap.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# Sorted by best frequency, the population tiles the full 2-32 kHz range:
# every frequency is the best frequency for a substantial block of units, and
# responses fall off with distance from the best frequency.

# %%
# Figure 4: population statistics
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

ax = axes[0, 0]
x = np.arange(5)
ax.bar(x - 0.2, np.bincount(bf_idx, minlength=5), width=0.4, color="#999999",
       label=f"all units (n={len(kw_sig)})")
ax.bar(x + 0.2, np.bincount(bf_idx[tuned], minlength=5), width=0.4,
       color="#1a9850", label=f"tuned units (n={tuned.sum()})")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of units")
ax.set_title("A  Best-frequency distribution")
ax.legend(frameon=False, fontsize=9)

ax = axes[0, 1]
bins = np.linspace(0, 1, 21)
ax.hist(sparseness[~tuned], bins=bins, alpha=0.6, color="#999999",
        label="not tuned", density=True)
ax.hist(sparseness[tuned], bins=bins, alpha=0.6, color="#1a9850",
        label="tuned", density=True)
ax.set_xlabel("Tuning sparseness (0 = flat, 1 = single-frequency)")
ax.set_ylabel("Density")
ax.set_title("B  Tuning sharpness")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 0]
neglogp = -np.log10(np.clip(kw_p, 1e-300, 1))
ax.scatter(neglogp[~tuned], pseudo_r2[~tuned], s=4, alpha=0.3, color="#999999",
           rasterized=True, label="not tuned")
ax.scatter(neglogp[tuned], pseudo_r2[tuned], s=4, alpha=0.3, color="#1a9850",
           rasterized=True, label="tuned")
ax.axvline(-np.log10(0.05), color="k", ls="--", lw=0.8)
ax.set_xlabel("$-\\log_{10}$(Kruskal–Wallis p)")
ax.set_ylabel("GLM pseudo-$R^2$ (McFadden)")
ax.set_title("C  Two views of tuning strength")
ax.legend(frameon=False, fontsize=9, markerscale=3)

ax = axes[1, 1]
frac_resp = [resp_sig[session_idx == i].mean() for i in range(len(SESS))]
frac_kw = [kw_sig[session_idx == i].mean() for i in range(len(SESS))]
frac_both = [tuned[session_idx == i].mean() for i in range(len(SESS))]
xs = np.arange(len(SESS))
ax.bar(xs - 0.25, frac_resp, width=0.25, color="#67a9cf", label="responsive")
ax.bar(xs, frac_kw, width=0.25, color="#f46d43", label="freq-modulated")
ax.bar(xs + 0.25, frac_both, width=0.25, color="#1a9850", label="both")
ax.set_xticks(xs)
ax.set_xticklabels([s.replace("sub-", "").replace("_", " ") for s in SESS],
                   rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Fraction of units")
ax.set_ylim(0, 1.05)
ax.set_title("D  Consistency across sessions")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("figures/fig4_population_stats.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# Best frequencies cover the whole tested range with a mild over-representation
# of the middle frequencies, tuned units have much sparser tuning curves than
# untuned units, and the two independent measures of tuning strength
# (Kruskal-Wallis significance and GLM pseudo-R²) agree. The pattern is
# consistent across all 15 sessions.

# %%
# Figure 5: GLM summary
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
glm_bf_idx = np.clip(np.round(glm_bf_log2 - 1).astype(int), 0, 4)
conf = np.zeros((5, 5))
for a, b in zip(bf_idx[tuned], glm_bf_idx[tuned]):
    conf[a, b] += 1
confn = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
im = ax.imshow(confn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
ax.set_yticklabels([f"{int(f/1000)}" for f in FREQS])
ax.set_xlabel("GLM best frequency (kHz)")
ax.set_ylabel("Raw best frequency (kHz)")
ax.set_title("A  Raw vs GLM best frequency\n(tuned units; rows normalized)")
for i in range(5):
    for j in range(5):
        if conf[i, j] > 0:
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if confn[i, j] > 0.5 else "black")
plt.colorbar(im, ax=ax, pad=0.02, label="Fraction of row")

ax = axes[1]
bins = np.linspace(0, np.percentile(pseudo_r2, 99), 40)
ax.hist(pseudo_r2[~tuned], bins=bins, alpha=0.6, density=True,
        color="#999999", label="not tuned")
ax.hist(pseudo_r2[tuned], bins=bins, alpha=0.6, density=True,
        color="#1a9850", label="tuned")
ax.set_xlabel("GLM pseudo-$R^2$ (McFadden, per unit)")
ax.set_ylabel("Density")
ax.set_title("B  GLM explanatory power")
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("figures/fig5_glm_summary.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# The GLM's best frequency agrees with the raw best frequency for the large
# majority of tuned units (strong diagonal), and pseudo-R² is clearly higher
# for tuned than untuned units. The off-diagonal entries between 8 and 16 kHz
# reflect units with broad tuning whose smooth GLM peak lands one octave away
# from the raw peak.

# %%
# Figure 6: firing-rate context and cross-session PSTH
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
bins = np.linspace(0, 30, 50)
ax.hist(baseline, bins=bins, color="#2166ac", alpha=0.7, density=True,
        label="baseline (−50–0 ms)")
ax.hist(evoked.max(axis=1) + baseline, bins=bins, color="#d01c8b",
        alpha=0.6, density=True, label="peak evoked (5–60 ms)")
ax.set_xlabel("Firing rate (Hz)")
ax.set_ylabel("Density")
ax.set_title("A  Baseline vs peak evoked rates")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
t_ms = PSTH_CENTERS * 1000
mean_psth = psth.mean(axis=0)
sem_psth = psth.std(axis=0) / np.sqrt(len(SESS))
for fi, fr in enumerate(FREQS):
    ax.plot(t_ms, mean_psth[fi], color=FREQ_COLORS[fr], lw=1.8, label=FREQ_LABELS[fi])
    ax.fill_between(t_ms, mean_psth[fi] - sem_psth[fi], mean_psth[fi] + sem_psth[fi],
                    color=FREQ_COLORS[fr], alpha=0.2, lw=0)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvspan(0, 25, color="gray", alpha=0.15)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population rate (Hz/unit)")
ax.set_title("B  Mean population PSTH across\n15 sessions (mean ± SEM)")
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("figures/fig6_rates_and_psth.png", bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ## Summary
#
# Across 1564 units from 15 sessions in 5 mice, passively presented 25 ms pure
# tones evoked a brisk response peaking 10-30 ms after onset. About 80% of
# units changed their firing rate in response to tones (Wilcoxon p < 0.05),
# and about 87% were significantly modulated by tone frequency
# (Kruskal-Wallis p < 0.05); 74% passed both criteria. Single units show
# V-shaped or sharply peaked tuning curves with best frequencies spanning the
# full tested range of 2-32 kHz, and the population tiles that range when
# sorted by best frequency. A Poisson GLM with a B-spline basis over log2
# frequency reproduces the raw tuning curves and assigns best frequencies that
# match the raw estimates for most tuned units, providing a smooth,
# model-based confirmation of the frequency-tuning phenomenon.

# %%
print("done. figures written to ./figures/")
