# %% [markdown]
# # Auditory Frequency Tuning in Mouse Auditory Cortex
#
# **Dataset:** [DANDI 000986](https://dandiarchive.org/dandiset/000986) —
# "Auditory cortex Neuropixels recordings and pupil diameter traces from mice
# during passive exposure to pure tones" (Jaramillo lab).
#
# **Phenomenon:** Neurons in auditory cortex respond selectively to the
# frequency of pure tones. Each cell has a *best frequency* (BF) that evokes
# the strongest response, and the *frequency tuning curve* (firing rate vs.
# tone frequency) is a fundamental descriptor of auditory processing.
#
# **Approach:**
# 1. Stream NWB files directly from the DANDI Archive with `remfile` (no full
#    downloads) and access them with Pynapple.
# 2. For every unit, count spikes in a short window after each tone onset
#    (5–55 ms) and subtract a pre-tone baseline, per tone frequency.
# 3. Test tuning significance with a Kruskal–Wallis test across frequencies,
#    extract best frequency and a selectivity index.
# 4. Pool 15 sessions from 5 mice for population statistics.
# 5. Fit a smooth tuning curve per unit with a Poisson GLM over a B-spline
#    basis on log2 frequency (NeMoS) and compare GLM vs. empirical estimates.
#
# The script runs end-to-end. Per-session results are cached in `cache/` so
# re-runs are fast; on a fresh machine everything is recomputed by streaming
# from DANDI (total runtime ~25 min, dominated by streaming and GLM fits).

# %% [markdown]
# ## Setup

# %%
import os
import json
import glob
import urllib.request

import numpy as np
import h5py
import remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import nemos as nmo

DANDISET = "000986"
VERSION = "0.251031.1939"
CACHE_DIR = "cache"
FIG_DIR = "figures"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Analysis windows relative to tone onset (tones are 25 ms, 60 dB SPL)
RESP_WIN = (0.005, 0.055)   # response window: 5-55 ms after onset
BASE_WIN = (-0.050, 0.0)    # baseline window: 50 ms before onset
WIN_DUR = RESP_WIN[1] - RESP_WIN[0]
MIN_RATE = 0.5              # Hz; exclude near-silent units
P_THRESH = 0.01             # Kruskal-Wallis significance threshold
REMCACHE = "/tmp/remfile_cache_auditory"

# %% [markdown]
# ## Session inventory
#
# The dandiset contains 15 sessions from 5 mice (LA3, LA8, LA9, LA11, LA12).
# We query the DANDI API for the asset list and resolve each asset's S3 URL
# for streaming.

# %%
def list_assets():
    url = (f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/"
           f"versions/{VERSION}/assets/?page_size=100")
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    return sorted((a["path"], a["asset_id"]) for a in data["results"])


def asset_s3_url(asset_id):
    with urllib.request.urlopen(
            f"https://api.dandiarchive.org/api/assets/{asset_id}/") as r:
        meta = json.load(r)
    for u in meta["contentUrl"]:
        if "s3" in u:
            return u
    return meta["contentUrl"][0]


assets = list_assets()
print(f"{len(assets)} sessions:")
for path, _ in assets:
    print("  ", path)

# %% [markdown]
# ## Data loading and per-session tuning analysis
#
# Each NWB file contains a `units` table (spike times) and a `trials`
# interval table with `stim_frequency` (2000–32000 Hz, octave spacing),
# `stim_duration` (25 ms) and `stim_amplitude` (60 dB). Tones are presented
# every ~0.8 s in random order while the mouse is passive.
#
# For each unit we count spikes falling in the response window and in the
# baseline window for every trial (vectorized with `searchsorted`), then
# compute the baseline-subtracted evoked rate per frequency.

# %%
def load_nwb_pynapple(s3_url):
    disk_cache = remfile.DiskCache(REMCACHE)
    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile)


def count_in_window(spike_times, onsets, win):
    """Count spikes in [onset+win0, onset+win1) for each trial (vectorized)."""
    if len(spike_times) == 0:
        return np.zeros(len(onsets), dtype=np.int16)
    tr = np.searchsorted(onsets, spike_times, side="right") - 1
    valid = tr >= 0
    tr = tr[valid]
    dt = spike_times[valid] - onsets[tr]
    in_win = (dt >= win[0]) & (dt < win[1])
    return np.bincount(tr[in_win], minlength=len(onsets)).astype(np.int16)


def analyze_session(path, asset_id):
    """Compute per-unit tone responses; cache to cache/<session>.npz."""
    name = path.split("/")[-1].replace("_behavior.nwb", "")
    out = os.path.join(CACHE_DIR, f"{name}.npz")
    if os.path.exists(out):
        print(f"cached: {name}")
        return name

    nwb = load_nwb_pynapple(asset_s3_url(asset_id))
    spikes = nwb["units"]          # TsGroup
    trials = nwb["trials"]         # IntervalSet with stim_frequency metadata

    freqs = np.sort(np.unique(np.asarray(trials["stim_frequency"])))
    onsets = np.asarray(trials["start"])
    trial_freq = np.asarray(trials["stim_frequency"])
    n_trials = len(onsets)
    n_units = len(spikes)
    session_dur = float(np.asarray(trials["end"])[-1])

    resp_counts = np.zeros((n_units, n_trials), dtype=np.int16)
    base_counts = np.zeros((n_units, n_trials), dtype=np.int16)
    mean_rates = np.zeros(n_units)
    for iu in tqdm(range(n_units), desc=name, leave=False):
        st = spikes[iu].t
        mean_rates[iu] = len(st) / session_dur
        resp_counts[iu] = count_in_window(st, onsets, RESP_WIN)
        base_counts[iu] = count_in_window(st, onsets, BASE_WIN)

    # baseline-subtracted evoked firing rate (Hz) per trial
    evoked = (resp_counts - base_counts) / WIN_DUR

    mean_evoked = np.zeros((n_units, len(freqs)))
    sem_evoked = np.zeros((n_units, len(freqs)))
    pvals = np.ones(n_units)
    for iu in range(n_units):
        for fi, f in enumerate(freqs):
            m = trial_freq == f
            mean_evoked[iu, fi] = evoked[iu, m].mean()
            sem_evoked[iu, fi] = evoked[iu, m].std() / np.sqrt(m.sum())
        groups = [resp_counts[iu, trial_freq == f] for f in freqs]
        pvals[iu] = stats.kruskal(*groups)[1]

    np.savez(out, freqs=freqs, trial_freq=trial_freq, onsets=onsets,
             resp_counts=resp_counts, base_counts=base_counts,
             mean_evoked=mean_evoked, sem_evoked=sem_evoked,
             pvals=pvals, mean_rates=mean_rates, session_dur=session_dur)
    print(f"done: {name} ({n_units} units, {n_trials} trials)")
    return name


session_names = [analyze_session(path, aid) for path, aid in assets]

# %% [markdown]
# ## Raw data: tones and evoked spiking
#
# Before any analysis, look at the raw data: a 15 s excerpt of the session
# showing tone onsets (colored by frequency) and the spike raster of 40
# units. Bursts of spikes time-locked to tone onsets are visible for some
# units.

# %%
# Use the first session (sub-LA11 ses-1) for spike-time-level figures
EXAMPLE_SESSION = "sub-LA11_ses-1"
example_asset = dict((p.split("/")[-1].replace("_behavior.nwb", ""), aid)
                     for p, aid in assets)[EXAMPLE_SESSION]
nwb_ex = load_nwb_pynapple(asset_s3_url(example_asset))
spikes_ex = nwb_ex["units"]
trials_ex = nwb_ex["trials"]
freqs = np.sort(np.unique(np.asarray(trials_ex["stim_frequency"])))
onsets_ex = np.asarray(trials_ex["start"])
trial_freq_ex = np.asarray(trials_ex["stim_frequency"])
print(f"{EXAMPLE_SESSION}: {len(spikes_ex)} units, {len(onsets_ex)} trials, "
      f"frequencies {freqs.astype(int)} Hz")

# %%
T0, T1 = 1000.0, 1015.0
N_UNITS_SHOW = 40
cmap = plt.get_cmap("viridis")
colors = {f: cmap(i / (len(freqs) - 1)) for i, f in enumerate(freqs)}

fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 4], hspace=0.08))
ax = axes[0]
m = (onsets_ex >= T0) & (onsets_ex < T1)
for t, f in zip(onsets_ex[m], trial_freq_ex[m]):
    ax.axvline(t, color=colors[f], lw=2, alpha=0.9)
ax.set_xlim(T0, T1)
ax.set_yticks([])
ax.set_ylabel("tone", rotation=0, ha="right", va="center")
ax.spines[["top", "right", "left"]].set_visible(False)
handles = [plt.Line2D([0], [0], color=colors[f], lw=2) for f in freqs]
ax.legend(handles, [f"{int(f/1000)} kHz" for f in freqs], ncol=5, fontsize=8,
          loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
          columnspacing=1.2)
ax.set_title("Passive pure-tone exposure: stimulus times and evoked spiking "
             f"({EXAMPLE_SESSION.replace('_', ' ')})", pad=40)

ax = axes[1]
for iu in range(N_UNITS_SHOW):
    st = spikes_ex[iu].t
    st = st[(st >= T0) & (st < T1)]
    ax.plot(st, np.full_like(st, iu), "|", color="k", ms=3, rasterized=True)
ax.set_ylim(-1, N_UNITS_SHOW)
ax.set_ylabel("unit #")
ax.set_xlabel("time in session (s)")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(os.path.join(FIG_DIR, "fig1_raw_data.png"), dpi=150)
plt.close(fig)
print("saved fig1_raw_data.png")

# %% [markdown]
# ## Perievent responses: raster and PSTH by frequency
#
# Aligning the spikes of one sharply tuned unit to tone onset and grouping
# trials by frequency shows the tuning directly in single trials: this unit
# (BF = 4 kHz) responds with a sustained rate increase starting ~25 ms after
# 4 kHz tones, while the other frequencies suppress its firing below the
# pre-tone baseline. Excitation at the best frequency combined with
# suppression at flanking frequencies is a common pattern in auditory cortex
# and sharpens frequency selectivity.

# %%
d_ex = np.load(os.path.join(CACHE_DIR, f"{EXAMPLE_SESSION}.npz"))
# choose a sharply tuned unit: significant, BF = 4 kHz, strong response
cand = np.where((d_ex["mean_rates"] >= MIN_RATE) & (d_ex["pvals"] < P_THRESH))[0]
bf_ex = d_ex["freqs"][np.argmax(d_ex["mean_evoked"], axis=1)]
cand4 = [iu for iu in cand if bf_ex[iu] == 4000.0]
cand4.sort(key=lambda iu: d_ex["pvals"][iu])
UNIT = int(cand4[0])
print(f"example unit: {UNIT} (p={d_ex['pvals'][UNIT]:.1e})")

WIN = (-0.05, 0.15)
pev = nap.compute_perievent(spikes_ex[UNIT], nap.Ts(onsets_ex), window=WIN)

fig, axes = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True,
                         gridspec_kw=dict(height_ratios=[3, 2], hspace=0.32))
ax = axes[0]
y = 0
yticks, ylabels = [], []
for f in freqs:
    tr_idx = np.where(trial_freq_ex == f)[0]
    if len(tr_idx) > 120:
        tr_idx = tr_idx[np.linspace(0, len(tr_idx) - 1, 120).astype(int)]
    for it in tr_idx:
        t = pev[it].t * 1000.0
        ax.plot(t, np.full_like(t, y), "|", color=colors[f], ms=2, rasterized=True)
        y += 1
    yticks.append(y - len(tr_idx) / 2)
    ylabels.append(f"{int(f/1000)} kHz")
    ax.axhline(y, color="0.85", lw=0.5)
ax.axvline(0, color="r", lw=1, ls="--")
ax.axvspan(0, 25, color="r", alpha=0.08)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_ylabel("trials grouped by tone frequency")
ax.set_title(f"Unit {UNIT}: spike raster aligned to tone onset "
             f"({EXAMPLE_SESSION.replace('_', ' ')})", pad=8)
ax.set_xlim(WIN[0] * 1000, WIN[1] * 1000)

ax = axes[1]
bin_size = 0.005
bins = np.arange(WIN[0], WIN[1] + bin_size, bin_size)
centers = 0.5 * (bins[:-1] + bins[1:])
for f in freqs:
    tr_idx = np.where(trial_freq_ex == f)[0]
    all_t = (np.concatenate([pev[it].t for it in tr_idx])
             if len(tr_idx) else np.array([]))
    counts, _ = np.histogram(all_t, bins=bins)
    rate = counts / (len(tr_idx) * bin_size)
    ax.plot(centers * 1000, rate, color=colors[f], lw=1.8,
            label=f"{int(f/1000)} kHz")
ax.axvline(0, color="r", lw=1, ls="--")
ax.axvspan(0, 25, color="r", alpha=0.08)
ax.set_xlabel("time from tone onset (ms)")
ax.set_ylabel("firing rate (Hz)")
ax.set_ylim(top=ax.get_ylim()[1] * 1.5)  # headroom for the legend
ax.legend(title="tone freq", fontsize=8, title_fontsize=8, ncol=5,
          loc="upper center", bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
ax.set_title("Peri-stimulus time histogram by frequency", pad=26)
fig.savefig(os.path.join(FIG_DIR, "fig2_perievent_raster_psth.png"), dpi=150)
plt.close(fig)
print("saved fig2_perievent_raster_psth.png")

# %% [markdown]
# ## Frequency tuning curves
#
# The tuning curve of a unit is its baseline-subtracted evoked firing rate as
# a function of tone frequency. Here are the 8 most significantly tuned units
# of the example session (Kruskal–Wallis across frequencies). Tuning curves
# are typically bell-shaped on a log-frequency axis, with a clear best
# frequency.

# %%
me, se, pv, rates = (d_ex["mean_evoked"], d_ex["sem_evoked"],
                     d_ex["pvals"], d_ex["mean_rates"])
keep = rates >= MIN_RATE
idx_all = np.where(keep)[0]
order = idx_all[np.argsort(pv[keep])]
logf = np.log2(freqs)

fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
for ax, iu in zip(axes.flat, order[:8]):
    ax.errorbar(logf, me[iu], yerr=se[iu], marker="o", ms=4, lw=1.5,
                color="k", capsize=2)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title(f"unit {iu}  (p={pv[iu]:.0e})", fontsize=9)
    ax.set_xticks(logf)
    ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
for ax in axes[-1]:
    ax.set_xlabel("tone frequency (kHz)")
for ax in axes[:, 0]:
    ax.set_ylabel("evoked rate (Hz)")
fig.suptitle("Example frequency tuning curves, best-tuned units "
             f"({EXAMPLE_SESSION.replace('_', ' ')})")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_example_tuning_curves.png"), dpi=150)
plt.close(fig)
print("saved fig3_example_tuning_curves.png")

# %% [markdown]
# ## Population analysis across all 15 sessions
#
# Pooling all sessions: for every unit with mean rate ≥ 0.5 Hz we compute the
# Kruskal–Wallis p-value (tuning significance), the best frequency, and a
# selectivity index `1 - mean/max` over the positive part of the tuning curve
# (0 = flat, 1 = responds to a single frequency).

# %%
session_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
all_meta = []
per_session = []
for spath in session_files:
    dd = np.load(spath)
    sname = os.path.basename(spath).replace(".npz", "")
    me_s, pv_s, rates_s = dd["mean_evoked"], dd["pvals"], dd["mean_rates"]
    keep_s = rates_s >= MIN_RATE
    me_s, pv_s, rates_s = me_s[keep_s], pv_s[keep_s], rates_s[keep_s]
    bf_s = freqs[np.argmax(me_s, axis=1)]
    pos = np.clip(me_s, 0, None)
    maxr = pos.max(axis=1)
    sel_s = np.where(maxr > 0,
                     1 - pos.mean(axis=1) / np.where(maxr > 0, maxr, 1),
                     np.nan)
    sig_s = pv_s < P_THRESH
    per_session.append(dict(name=sname, n_units=int(len(pv_s)),
                            n_tuned=int(sig_s.sum()),
                            frac_tuned=float(sig_s.mean())))
    for iu in range(len(pv_s)):
        all_meta.append(dict(session=sname, pval=pv_s[iu], bf=bf_s[iu],
                             sel=sel_s[iu], rate=rates_s[iu], sig=sig_s[iu],
                             me=me_s[iu]))

n_total = len(all_meta)
n_sig = sum(m["sig"] for m in all_meta)
sig_meta = [m for m in all_meta if m["sig"]]
bfs = np.array([m["bf"] for m in sig_meta])
sels = np.array([m["sel"] for m in sig_meta])
sels = sels[~np.isnan(sels)]
pvals_all = np.array([m["pval"] for m in all_meta])
print(f"units with rate >= {MIN_RATE} Hz: {n_total}; "
      f"significantly tuned: {n_sig} ({100*n_sig/n_total:.1f}%)")
print("per-session tuned fractions:",
      {ps["name"]: round(ps["frac_tuned"], 2) for ps in per_session})

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

ax = axes[0]
names = [ps["name"].replace("sub-", "") for ps in per_session]
fracs = [100 * ps["frac_tuned"] for ps in per_session]
ax.bar(range(len(names)), fracs, color="0.4")
ax.axhline(100 * n_sig / n_total, color="r", ls="--", lw=1,
           label=f"pooled: {100*n_sig/n_total:.0f}%")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("% units tuned")
ax.set_title("Fraction tuned per session\n(Kruskal-Wallis p<0.01)")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(np.log10(pvals_all + 1e-300), bins=50, color="0.4")
ax.axvline(np.log10(P_THRESH), color="r", ls="--", label=f"p={P_THRESH}")
ax.set_xlabel("log10(Kruskal-Wallis p)")
ax.set_ylabel("# units")
ax.set_title("Tuning significance (all sessions)")
ax.legend(fontsize=8)

ax = axes[2]
counts = [np.sum(bfs == f) for f in freqs]
ax.bar(np.arange(len(freqs)), counts, color="0.4")
ax.set_xticks(np.arange(len(freqs)))
ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("# tuned units")
ax.set_title(f"Best-frequency distribution (n={n_sig} tuned)")

ax = axes[3]
ax.hist(sels, bins=30, color="0.4")
ax.axvline(np.median(sels), color="r", ls="--",
           label=f"median={np.median(sels):.2f}")
ax.set_xlabel("selectivity index (1 - mean/max response)")
ax.set_ylabel("# units")
ax.set_title("Tuning selectivity (tuned units)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_population_summary.png"), dpi=150)
plt.close(fig)
print("saved fig4_population_summary.png")

# %% [markdown]
# ## Population tuning heatmap
#
# Normalizing each tuned unit's tuning curve to its peak and sorting units by
# best frequency reveals the population structure: the peak response marches
# along the diagonal, showing that the recorded population tiles the full
# 2–32 kHz range.

# %%
curves = np.array([m["me"] for m in sig_meta])
bf_idx = np.argmax(curves, axis=1)
pos = np.clip(curves, 0, None)
norm = pos / np.maximum(pos.max(axis=1, keepdims=True), 1e-9)
order = np.lexsort((-norm.max(axis=1), bf_idx))
norm_sorted = norm[order]

fig, ax = plt.subplots(figsize=(5.2, 6.5))
im = ax.imshow(norm_sorted, aspect="auto", cmap="inferno",
               interpolation="nearest",
               extent=[-0.5, len(freqs) - 0.5, len(norm_sorted), 0])
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("tuned units (sorted by best frequency)")
ax.set_title(f"Normalized tuning curves, {len(norm_sorted)} tuned units\n"
             "(all sessions pooled)")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("response / max")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_tuning_heatmap.png"), dpi=150)
plt.close(fig)
print("saved fig5_tuning_heatmap.png")

# %% [markdown]
# ## Smooth tuning curves with a Poisson GLM (NeMoS)
#
# The windowed-count estimate is discrete (5 frequencies). A Poisson GLM with
# a B-spline basis over log2 frequency gives a smooth tuning curve from the
# same spike counts:
#
# $$\text{count}_{\text{trial}} \sim \text{Poisson}\left(\exp\left(\beta_0 + \sum_k \beta_k B_k(\log_2 f_{\text{trial}})\right)\right)$$
#
# We fit every tuned unit (NeMoS `GLM`, LBFGS) and compare the GLM-based best
# frequency (argmax of the smooth curve on a fine grid) with the empirical
# one. Results are cached in `glm_fits.npz`.

# %%
GLM_OUT = "glm_fits.npz"
N_BASIS = 5

if os.path.exists(GLM_OUT):
    print("loading cached GLM fits")
    g = np.load(GLM_OUT)
    glm_curves, glm_bf, emp_bf = g["glm_curves"], g["glm_bf"], g["emp_bf"]
    grid = g["grid"]
else:
    basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS)
    grid = np.linspace(np.log2(freqs).min(), np.log2(freqs).max(), 100)
    glm_curves, glm_bf, emp_bf = [], [], []
    for spath in tqdm(session_files, desc="GLM fits"):
        dd = np.load(spath)
        X = basis.compute_features(np.log2(dd["trial_freq"]))
        Xg = basis.compute_features(grid)
        resp = dd["resp_counts"].astype(float)
        pv_s, rates_s = dd["pvals"], dd["mean_rates"]
        keep_s = (rates_s >= MIN_RATE) & (pv_s < P_THRESH)
        me_s = dd["mean_evoked"]
        for iu in np.where(keep_s)[0]:
            model = nmo.glm.GLM(solver_name="LBFGS",
                                solver_kwargs=dict(tol=1e-6, maxiter=500))
            model.fit(X, resp[iu])
            pred = model.predict(Xg) / WIN_DUR
            glm_curves.append(pred)
            glm_bf.append(2 ** grid[np.argmax(pred)])
            emp_bf.append(freqs[np.argmax(me_s[iu])])
    glm_curves = np.array(glm_curves)
    glm_bf = np.array(glm_bf)
    emp_bf = np.array(emp_bf)
    np.savez(GLM_OUT, glm_curves=glm_curves, glm_bf=glm_bf, emp_bf=emp_bf,
             freqs=freqs, grid=grid)

agree = (glm_bf == emp_bf).mean()
oct_diff = np.abs(np.log2(glm_bf / emp_bf))
print(f"GLM fits: {len(glm_bf)} tuned units")
print(f"BF agreement: exact {100*agree:.0f}%, "
      f"within 1 octave {100*np.mean(oct_diff <= 1):.0f}%, "
      f"median |diff| {np.median(oct_diff):.2f} octaves")

# %%
# example fits: one unit per best frequency from the example session
basis = nmo.basis.BSplineEval(n_basis_funcs=N_BASIS)
X_ex = basis.compute_features(np.log2(trial_freq_ex))
Xg_ex = basis.compute_features(grid)
resp_ex = d_ex["resp_counts"].astype(float)
cand = np.where((d_ex["mean_rates"] >= MIN_RATE) & (d_ex["pvals"] < P_THRESH))[0]
examples = []
for target in [2000, 4000, 8000, 16000]:
    match = [iu for iu in cand if bf_ex[iu] == target]
    match.sort(key=lambda iu: d_ex["pvals"][iu])
    examples.append(match[0])

fig = plt.figure(figsize=(13, 7.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.45, wspace=0.35)
for col, iu in enumerate(examples):
    ax = fig.add_subplot(gs[0, col])
    model = nmo.glm.GLM(solver_name="LBFGS",
                        solver_kwargs=dict(tol=1e-6, maxiter=500))
    model.fit(X_ex, resp_ex[iu])
    pred = model.predict(Xg_ex) / WIN_DUR
    ax.errorbar(logf, me[iu], yerr=se[iu], marker="o", ms=4, lw=0,
                elinewidth=1.2, color="k", capsize=2,
                label="empirical (mean ± SEM)")
    ax.plot(grid, pred, color="crimson", lw=2, label="Poisson GLM (B-spline)")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xticks(logf)
    ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
    ax.set_title(f"unit {iu} ({EXAMPLE_SESSION.replace('_', ' ')})", fontsize=9)
    if col == 0:
        ax.set_ylabel("evoked rate (Hz)")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("tone frequency (kHz)")

ax = fig.add_subplot(gs[1, :])
rng = np.random.default_rng(0)
jitter = lambda a: a * 2 ** rng.uniform(-0.08, 0.08, len(a))
ax.scatter(jitter(emp_bf), jitter(glm_bf), s=6, alpha=0.35, color="0.25",
           rasterized=True)
ax.plot([1500, 45000], [1500, 45000], "r--", lw=1, label="identity")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(freqs)
ax.set_yticks(freqs)
ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_yticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_xlabel("empirical best frequency (kHz)")
ax.set_ylabel("GLM best frequency (kHz)")
ax.set_title(f"GLM vs empirical best frequency, {len(glm_bf)} tuned units "
             f"(exact agreement {100*agree:.0f}%, "
             f"within 1 octave {100*np.mean(oct_diff<=1):.0f}%)")
ax.legend(fontsize=8)
fig.suptitle("Smooth tuning curves from a Poisson GLM with B-spline basis "
             "over log2 frequency (NeMoS)")
fig.savefig(os.path.join(FIG_DIR, "fig6_nemos_glm.png"), dpi=150)
plt.close(fig)
print("saved fig6_nemos_glm.png")

# %% [markdown]
# ## Summary
#
# - **85% of units** (1221/1431 with mean rate ≥ 0.5 Hz, pooled over 15
#   sessions from 5 mice) show **significant frequency tuning** to passive
#   pure tones (Kruskal–Wallis across 5 frequencies, p < 0.01).
# - Best frequencies **tile the full 2–32 kHz range** tested, with the
#   largest share of units preferring 8–16 kHz, in the middle of the mouse
#   hearing range.
# - Tuning is visible in single trials (perievent rasters) and in smooth
#   Poisson-GLM fits: GLM and empirical best frequencies agree within one
#   octave for **96%** of tuned units.
#
# ### Reuse notes
# - Data: DANDI 000986, streamed with remfile + h5py, accessed via Pynapple.
# - Per-session spike counts are cached in `cache/*.npz`; GLM fits in
#   `glm_fits.npz`. Delete these to recompute from scratch.
# - Analysis windows and thresholds are set in the Setup cell.

# %%
# machine-readable summary for the README
summary = dict(
    dandiset=DANDISET, version=VERSION,
    n_sessions=len(session_files), n_units_total=int(n_total),
    n_tuned=int(n_sig), frac_tuned=float(n_sig / n_total),
    bf_counts={int(f): int(np.sum(bfs == f)) for f in freqs},
    median_selectivity=float(np.median(sels)),
    glm_bf_within_1_oct=float(np.mean(oct_diff <= 1)),
    per_session=[{k: (int(v) if isinstance(v, (np.integer,)) else
                      float(v) if isinstance(v, (np.floating,)) else v)
                  for k, v in ps.items()} for ps in per_session],
    freqs=[float(f) for f in freqs],
)
with open("population_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("saved population_summary.json")
print(json.dumps({k: v for k, v in summary.items() if k != "per_session"},
                 indent=2))
