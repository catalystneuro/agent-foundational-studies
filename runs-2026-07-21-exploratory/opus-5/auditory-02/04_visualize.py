"""Pooled, cross-session figures for the DANDI 000986 frequency-tuning analysis."""

import numpy as np
import matplotlib.pyplot as plt

import audlib as A

S = list(np.load("results_all_sessions.npy", allow_pickle=True))
freqs = S[0]["freqs"]
fkhz = freqs / 1e3
nf = len(freqs)
cmap = plt.get_cmap("viridis")
colors = [cmap(i / (nf - 1)) for i in range(nf)]

keep = [s["tuned"] & s["responsive"] for s in S]
tc = np.vstack([s["tc"][k] for s, k in zip(S, keep)])
bf = np.concatenate([s["bf"][k] for s, k in zip(S, keep)])
spars = np.concatenate([s["spars"][k] for s, k in zip(S, keep)])
bw = np.concatenate([s["glm_bw"][k] for s, k in zip(S, keep)])
pr2 = np.concatenate([s["glm_pr2"][k] for s, k in zip(S, keep)])
pr2_all = np.concatenate([s["glm_pr2"] for s in S])
bf1 = np.concatenate([s["bf1"][k] for s, k in zip(S, keep)])
bf2 = np.concatenate([s["bf2"][k] for s, k in zip(S, keep)])
n_units = sum(len(s["tuned"]) for s in S)
n_resp = sum(int(s["responsive"].sum()) for s in S)
n_tuned = len(bf)
bf_idx = np.searchsorted(freqs, bf)

print(f"{len(S)} sessions, {n_units} units, {n_resp} responsive, {n_tuned} tuned")

# ---------------------------------------------------------------- figure 4 --
# population summary: yield per session, BF distribution, sparseness, BF stability
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

ax = axes[0, 0]
x = np.arange(len(S))
n_tot = [len(s["tuned"]) for s in S]
n_r = [int(s["responsive"].sum()) for s in S]
n_t = [int((s["tuned"] & s["responsive"]).sum()) for s in S]
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
counts = np.array([(bf_idx == j).sum() for j in range(nf)])
ax.bar(np.arange(nf), 100 * counts / counts.sum(), color=colors, edgecolor="k",
       linewidth=0.5)
ax.axhline(20, color="k", ls="--", lw=0.8, label="uniform (20%)")
ax.set_xticks(np.arange(nf))
ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("% of tuned units")
ax.set_title("Best-frequency distribution")
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 0]
ax.hist(spars, bins=np.linspace(0, 1, 26), color="steelblue", edgecolor="k",
        linewidth=0.4)
ax.axvline(np.median(spars), color="firebrick", lw=1.5,
           label=f"median = {np.median(spars):.2f}")
ax.set_xlabel("lifetime sparseness of tuning curve\n(0 = flat, 1 = single frequency)")
ax.set_ylabel("number of tuned units")
ax.set_title("Tuning selectivity")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
M = np.zeros((nf, nf))
for a, b in zip(np.searchsorted(freqs, bf1), np.searchsorted(freqs, bf2)):
    M[a, b] += 1
Mn = M / M.sum(axis=1, keepdims=True)
im = ax.imshow(Mn, cmap="magma", vmin=0, vmax=1, origin="lower")
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
fig.savefig("fig04_population_summary.png", dpi=150)
plt.close(fig)
print("saved fig04_population_summary.png")

# ---------------------------------------------------------------- figure 5 --
# tuning-curve heatmap sorted by BF, and the BF-aligned average tuning curve
norm = tc / np.abs(tc).max(axis=1, keepdims=True)
order = np.lexsort((-tc.max(axis=1), bf_idx))
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
    sel = bf_idx == j
    m = norm[sel].mean(axis=0)
    se = norm[sel].std(axis=0, ddof=1) / np.sqrt(sel.sum())
    ax.plot(oct_axis, m, color=colors[j], lw=1.8,
            label=f"BF {fkhz[j]:g} kHz (n={sel.sum()})")
    ax.fill_between(oct_axis, m - se, m + se, color=colors[j], alpha=0.25)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.set_xticks(oct_axis); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalised evoked rate")
ax.set_title("Mean tuning curve by best frequency")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
# BF-aligned average: offset in octaves from each unit's own BF
offsets = np.arange(-(nf - 1), nf)
acc = [[] for _ in offsets]
for i in range(len(norm)):
    for j in range(nf):
        acc[j - bf_idx[i] + nf - 1].append(norm[i, j])
m = np.array([np.mean(a) if len(a) else np.nan for a in acc])
se = np.array([np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
               for a in acc])
n_off = np.array([len(a) for a in acc])
ok = n_off >= 20
ax.errorbar(offsets[ok], m[ok], yerr=se[ok], marker="o", color="k", lw=1.8,
            capsize=3)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.axvline(0, color="firebrick", lw=0.8, ls=":")
ax.set_xlabel("octaves from each unit's best frequency")
ax.set_ylabel("normalised evoked rate")
ax.set_title("Best-frequency-aligned population\ntuning curve")

for a in axes[1:]:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig05_tuning_curves_population.png", dpi=150)
plt.close(fig)
print("saved fig05_tuning_curves_population.png")

# ---------------------------------------------------------------- figure 6 --
# NeMoS Poisson GLM: smooth tuning curves, bandwidth, cross-validated fit
grid_oct = S[0]["grid_oct"]
grid_khz = fkhz[0] * 2 ** grid_oct

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

ax = axes[0]
s0 = S[0]
k0 = s0["tuned"] & s0["responsive"]
idx = np.where(k0)[0]
idx = idx[np.argsort(-s0["glm_pr2"][idx])][:6]
for i in idx:
    r = s0["glm_rate"][i]
    r = (r - r.min()) / (r.max() - r.min())
    ax.plot(grid_khz, r, lw=1.6,
            label=f"unit {i} (BF {fkhz[0]*2**s0['glm_bf_oct'][i]:.1f} kHz)")
ax.set_xscale("log", base=2)
ax.set_xticks(fkhz); ax.set_xticklabels([f"{f:g}" for f in fkhz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalised GLM rate")
ax.set_title("Poisson GLM tuning curves\n(raised-cosine basis in octaves)")
ax.legend(frameon=False, fontsize=7)

ax = axes[1]
ax.hist(bw[np.isfinite(bw)], bins=np.linspace(0, 4, 25), color="steelblue",
        edgecolor="k", linewidth=0.4)
ax.axvline(np.nanmedian(bw), color="firebrick", lw=1.5,
           label=f"median = {np.nanmedian(bw):.2f} oct")
ax.set_xlabel("half-max bandwidth (octaves)")
ax.set_ylabel("number of tuned units")
ax.set_title("Tuning bandwidth")
ax.legend(frameon=False, fontsize=9)

ax = axes[2]
bins = np.linspace(-0.02, max(0.2, np.nanpercentile(pr2_all, 99.5)), 40)
ax.hist(pr2_all, bins=bins, color="0.75", edgecolor="k", linewidth=0.3,
        label=f"all units (n={len(pr2_all)})")
ax.hist(pr2, bins=bins, color="firebrick", edgecolor="k", linewidth=0.3,
        label=f"frequency tuned (n={len(pr2)})")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("cross-validated pseudo-$R^2$\n(frequency model vs. constant rate)")
ax.set_ylabel("number of units")
ax.set_title("Held-out predictive power")
ax.legend(frameon=False, fontsize=8)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig06_glm_tuning.png", dpi=150)
plt.close(fig)
print("saved fig06_glm_tuning.png")

# ---------------------------------------------------------------- figure 7 --
# single-trial population decoding of tone frequency
conf = np.nanmean(np.stack([s["conf"] for s in S]), axis=0)
accs = np.array([s["acc"] for s in S])
accs_tuned = np.array([s["acc_tuned"] for s in S])
n_tot_arr = np.array(n_tot)

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
ax.scatter(n_tot_arr, accs, s=45, color="firebrick", zorder=3)
ax.axhline(0.2, color="k", ls="--", lw=1, label="chance")
ax.set_xlabel("number of units in session")
ax.set_ylabel("decoding accuracy")
ax.set_title("Decoding improves with population size")
ax.set_ylim(0, max(0.75, np.nanmax(accs) * 1.15))
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig07_decoding.png", dpi=150)
plt.close(fig)
print("saved fig07_decoding.png")

# ------------------------------------------------------------------ report --
from scipy import stats as st
r_stab = agree
print(f"""
SUMMARY
  sessions .................. {len(S)} (5 mice)
  units ..................... {n_units}
  sound responsive .......... {n_resp} ({100*n_resp/n_units:.0f}%)
  frequency tuned ........... {n_tuned} ({100*n_tuned/n_units:.0f}%)
  median sparseness ......... {np.median(spars):.2f}
  median GLM bandwidth ...... {np.nanmedian(bw):.2f} octaves
  median CV pseudo-R2 (tuned) {np.nanmedian(pr2):.3f}
  split-half BF agreement ... {100*r_stab:.0f}% (chance 20%)
  BF distribution (kHz) ..... {dict(zip([f'{f:g}' for f in fkhz], counts))}
  chi2 vs uniform ........... {st.chisquare(counts).statistic:.1f}, p = {st.chisquare(counts).pvalue:.2g}
  decoding acc (all units) .. {np.nanmean(accs):.3f} +- {np.nanstd(accs):.3f}
  decoding acc (tuned only) . {np.nanmean(accs_tuned):.3f} +- {np.nanstd(accs_tuned):.3f}
""")
np.savez("summary_stats.npz", counts=counts, spars=spars, bw=bw, pr2=pr2,
         accs=accs, accs_tuned=accs_tuned, agree=agree, n_units=n_units, n_resp=n_resp, n_tuned=n_tuned)
