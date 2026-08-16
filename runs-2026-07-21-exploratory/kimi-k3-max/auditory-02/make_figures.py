"""Generate all figures from results_all_sessions.npz + example session rasters."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import lindi
from pynwb import NWBHDF5IO

R = np.load("results_all_sessions.npz", allow_pickle=False)
FREQS = R["freqs"]
LOG2F = np.log2(FREQS / 1000.)
GRID = R["grid"]
SESS = [str(s) for s in R["session_names"]]
EXAMPLE_SESS = "sub-LA11_ses-2"
EX_IDX = SESS.index(EXAMPLE_SESS)

FREQ_COLORS = {2000.: "#2166ac", 4000.: "#67a9cf", 8000.: "#1a9850",
               16000.: "#f46d43", 32000.: "#762a83"}
FREQ_LABELS = [f"{int(f/1000)} kHz" for f in FREQS]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False,
})

# masks
kw_sig = R["kw_p"] < 0.05
resp_sig = R["wilcox_p"] < 0.05
tuned = kw_sig & resp_sig
ex_mask = R["session_idx"] == EX_IDX

print(f"total units: {len(kw_sig)}, responsive: {resp_sig.mean()*100:.1f}%, "
      f"freq-modulated: {kw_sig.mean()*100:.1f}%, both: {tuned.mean()*100:.1f}%")


# ---------------------------------------------------------------- example data
def load_example_session():
    import analysis_core as ac
    local_cache = lindi.LocalCache()
    url = ("https://lindi.neurosift.org/dandi/dandisets/000986/assets/"
           f"{ac.ASSETS[EXAMPLE_SESS]}/nwb.lindi.json")
    f = lindi.LindiH5pyFile.from_lindi_file(url, local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    trials = nwbfile.intervals["trials"].to_dataframe()
    st_idx = f['units/spike_times_index'][:]
    st_all = f['units/spike_times'][:]
    bounds = np.concatenate([[0], st_idx])
    spk_list = [st_all[bounds[i]:bounds[i + 1]] for i in range(len(st_idx))]
    return trials, spk_list


def rel_times(spk, ons, win=(-0.1, 0.15)):
    idx = np.searchsorted(ons, spk, side="right") - 1
    valid = idx >= 0
    rel = spk[valid] - ons[idx[valid]]
    tr = idx[valid]
    keep = (rel >= win[0]) & (rel < win[1])
    return rel[keep], tr[keep]


# pick example units: for each frequency, among strongly tuned units with a
# clear excitatory peak (>5 Hz above baseline), take the highest pseudo-R2
ex_evoked = R["evoked"][ex_mask]
ex_pr2 = R["pseudo_r2"][ex_mask]
ex_kw = R["kw_p"][ex_mask]
ex_bf = R["bf_idx"][ex_mask]
example_units = []
for fi in range(5):
    cand = np.where((ex_bf == fi) & (ex_kw < 0.001) & (ex_evoked.max(axis=1) > 5))[0]
    if not len(cand):
        cand = np.where((ex_bf == fi) & (ex_kw < 0.001))[0]
    if len(cand):
        example_units.append(cand[np.argmax(ex_pr2[cand])])
print("example units (per BF):", example_units)

trials, spk_list = load_example_session()
onsets_all = trials["start_time"].values
freq_trial = trials["stim_frequency"].values

# ================================================================ FIGURE 1
# Stimulus design + example unit responses + population PSTH
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.42, wspace=0.35)

# A: stimulus sequence (first 80 trials)
ax = fig.add_subplot(gs[0, 0])
nshow = 80
t0 = onsets_all[0]
for fi, fr in enumerate(FREQS):
    m = (freq_trial == fr)[:nshow]
    ax.scatter((onsets_all[:nshow][m] - t0), np.zeros(m.sum()) + fi,
               marker="|", s=60, color=FREQ_COLORS[fr], lw=1.5)
ax.set_yticks(range(5))
ax.set_yticklabels(FREQ_LABELS)
ax.set_xlabel("Time in session (s)")
ax.set_title("A  Tone presentation sequence\n(first 80 trials, random order)")
ax.set_ylim(-0.7, 4.7)

# B: raster of one example unit (BF=8kHz preferred if available), grouped by freq
ax = fig.add_subplot(gs[0, 1:])
u_raster = example_units[2] if len(example_units) > 2 else example_units[0]
spk = spk_list[u_raster]
N_PER_FREQ = 150
ytick = 0
yticks, ylabels = [], []
rng = np.random.default_rng(0)
for fi, fr in enumerate(FREQS):
    tr_idx = np.where(freq_trial == fr)[0]
    show = rng.choice(tr_idx, size=N_PER_FREQ, replace=False)
    ons = np.sort(onsets_all[show])  # must be sorted for searchsorted
    rel, tr = rel_times(spk, ons, win=(-0.05, 0.1))
    ax.scatter(rel * 1000, tr + ytick, s=3, color=FREQ_COLORS[fr], rasterized=True)
    yticks.append(ytick + N_PER_FREQ / 2)
    ylabels.append(FREQ_LABELS[fi])
    ytick += N_PER_FREQ
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvspan(0, 25, color="gray", alpha=0.15)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
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
    ons = np.sort(onsets_all[freq_trial == fr])
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
psth_ex = R["psth"][EX_IDX]
t_ms = R["psth_centers"] * 1000
for fi, fr in enumerate(FREQS):
    ax.plot(t_ms, psth_ex[fi], color=FREQ_COLORS[fr], lw=1.8, label=FREQ_LABELS[fi])
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvspan(0, 25, color="gray", alpha=0.15)
ax.set_xlabel("Time from tone onset (ms)")
ax.set_ylabel("Population rate (Hz/unit)")
ax.set_title(f"D  Population PSTH, all {int(R['session_unit_counts'][EX_IDX])} units\n"
             f"({EXAMPLE_SESS})")
ax.legend(title="Tone frequency", fontsize=8, frameon=False)

plt.savefig("figures/fig1_stimulus_and_responses.png", bbox_inches="tight")
plt.close()
print("saved fig1")

# ================================================================ FIGURE 2
# Example tuning curves: raw means +/- SEM with GLM smooth overlay
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
axes = axes.flat
ex_tuning_sem = R["tuning_sem"][ex_mask]
ex_glm = R["glm_curves"][ex_mask]  # units x grid
for k, u in enumerate(example_units[:5]):
    ax = axes[k]
    ev = R["evoked"][ex_mask][u]
    ax.errorbar(LOG2F, ev, yerr=ex_tuning_sem[u], marker="o", ms=5, capsize=3,
                color="k", lw=1.2, label="data (mean±SEM)")
    ax.plot(GRID, ex_glm[u] - R["baseline"][ex_mask][u], color="#d01c8b", lw=2,
            label="Poisson GLM fit")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(LOG2F)
    ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
    bf = FREQS[np.argmax(ev)]
    ax.set_title(f"unit {u}, BF = {int(bf/1000)} kHz, "
                 f"GLM pseudo-$R^2$ = {ex_pr2[u]:.3f}", fontsize=10)
    ax.set_xlabel("Tone frequency (kHz)")
    ax.set_ylabel("Evoked rate − baseline (Hz)")
# 6th panel: legend + notes
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
print("saved fig2")

# ================================================================ FIGURE 3
# Population heatmap of normalized tuning, sorted by BF then sparseness
evoked = R["evoked"]
norm = np.clip(evoked, 0, None)
row_max = norm.max(axis=1, keepdims=True)
norm = np.divide(norm, row_max, out=np.zeros_like(norm), where=row_max > 0)
order = np.lexsort((-R["sparseness"], R["bf_idx"]))
fig, ax = plt.subplots(figsize=(6.5, 7))
im = ax.imshow(norm[order], aspect="auto", cmap="viridis",
               extent=[LOG2F[0] - 0.5, LOG2F[-1] + 0.5, len(norm), 0])
ax.set_xticks(LOG2F)
ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
ax.set_xlabel("Tone frequency (kHz)")
ax.set_ylabel("Units (sorted by best frequency)")
cb = plt.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Normalized evoked response")
# BF block boundaries
counts = np.bincount(R["bf_idx"][order], minlength=5)
bounds = np.cumsum(counts)
for b in bounds[:-1]:
    ax.axhline(b, color="white", lw=0.6, alpha=0.7)
ax.set_title(f"Population frequency tuning (n = {len(norm)} units,\n"
             f"15 sessions, 5 mice; each row normalized to its max)")
plt.tight_layout()
plt.savefig("figures/fig3_population_heatmap.png", bbox_inches="tight")
plt.close()
print("saved fig3")

# ================================================================ FIGURE 4
# Population statistics
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

# A: BF distribution
ax = axes[0, 0]
x = np.arange(5)
c_all = np.bincount(R["bf_idx"], minlength=5)
c_tuned = np.bincount(R["bf_idx"][tuned], minlength=5)
ax.bar(x - 0.2, c_all, width=0.4, color="#999999", label=f"all units (n={len(kw_sig)})")
ax.bar(x + 0.2, c_tuned, width=0.4, color="#1a9850",
       label=f"tuned units (n={tuned.sum()})")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(f/1000)}" for f in FREQS])
ax.set_xlabel("Best frequency (kHz)")
ax.set_ylabel("Number of units")
ax.set_title("A  Best-frequency distribution")
ax.legend(frameon=False, fontsize=9)

# B: sparseness distribution
ax = axes[0, 1]
bins = np.linspace(0, 1, 21)
ax.hist(R["sparseness"][~tuned], bins=bins, alpha=0.6, color="#999999",
        label="not tuned", density=True)
ax.hist(R["sparseness"][tuned], bins=bins, alpha=0.6, color="#1a9850",
        label="tuned", density=True)
ax.set_xlabel("Tuning sparseness (0 = flat, 1 = single-frequency)")
ax.set_ylabel("Density")
ax.set_title("B  Tuning sharpness")
ax.legend(frameon=False, fontsize=9)

# C: GLM pseudo-R2 vs KW significance
ax = axes[1, 0]
neglogp = -np.log10(np.clip(R["kw_p"], 1e-300, 1))
ax.scatter(neglogp[~tuned], R["pseudo_r2"][~tuned], s=4, alpha=0.3, color="#999999",
           rasterized=True, label="not tuned")
ax.scatter(neglogp[tuned], R["pseudo_r2"][tuned], s=4, alpha=0.3, color="#1a9850",
           rasterized=True, label="tuned")
ax.axvline(-np.log10(0.05), color="k", ls="--", lw=0.8)
ax.set_xlabel("$-\\log_{10}$(Kruskal–Wallis p)")
ax.set_ylabel("GLM pseudo-$R^2$ (McFadden)")
ax.set_title("C  Two views of tuning strength")
ax.legend(frameon=False, fontsize=9, markerscale=3)

# D: per-session fractions
ax = axes[1, 1]
frac_resp = [resp_sig[R["session_idx"] == i].mean() for i in range(len(SESS))]
frac_kw = [kw_sig[R["session_idx"] == i].mean() for i in range(len(SESS))]
frac_both = [tuned[R["session_idx"] == i].mean() for i in range(len(SESS))]
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

plt.tight_layout()
plt.savefig("figures/fig4_population_stats.png", bbox_inches="tight")
plt.close()
print("saved fig4")

# ================================================================ FIGURE 5
# GLM summary: raw BF vs GLM BF agreement + pseudo-R2 distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
glm_bf_idx = np.round(R["glm_bf_log2"] - 1).astype(int)  # grid starts at log2(2k/1k)=1
glm_bf_idx = np.clip(glm_bf_idx, 0, 4)
conf = np.zeros((5, 5))
for a, b in zip(R["bf_idx"][tuned], glm_bf_idx[tuned]):
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
bins = np.linspace(0, np.percentile(R["pseudo_r2"], 99), 40)
ax.hist(R["pseudo_r2"][~tuned], bins=bins, alpha=0.6, density=True,
        color="#999999", label="not tuned")
ax.hist(R["pseudo_r2"][tuned], bins=bins, alpha=0.6, density=True,
        color="#1a9850", label="tuned")
ax.set_xlabel("GLM pseudo-$R^2$ (McFadden, per unit)")
ax.set_ylabel("Density")
ax.set_title("B  GLM explanatory power")
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig5_glm_summary.png", bbox_inches="tight")
plt.close()
print("saved fig5")

# ================================================================ FIGURE 6
# Firing rate context: baseline vs evoked, and PSTH across sessions
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
bins = np.linspace(0, 30, 50)
ax.hist(R["baseline"], bins=bins, color="#2166ac", alpha=0.7, density=True,
        label="baseline (−50–0 ms)")
ax.hist(R["evoked"].max(axis=1) + R["baseline"], bins=bins, color="#d01c8b",
        alpha=0.6, density=True, label="peak evoked (5–60 ms)")
ax.set_xlabel("Firing rate (Hz)")
ax.set_ylabel("Density")
ax.set_title("A  Baseline vs peak evoked rates")
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
t_ms = R["psth_centers"] * 1000
mean_psth = R["psth"].mean(axis=0)
sem_psth = R["psth"].std(axis=0) / np.sqrt(len(SESS))
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

plt.tight_layout()
plt.savefig("figures/fig6_rates_and_psth.png", bbox_inches="tight")
plt.close()
print("saved fig6")

print("all figures done")
