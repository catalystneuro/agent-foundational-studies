"""Pooled ring-structure figures from cached per-session npz files."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "cache"
FIG = "../figures"
os.makedirs(FIG, exist_ok=True)
SESSIONS = ["Mouse17-130128", "Mouse20-130514", "Mouse24-131213",
            "Mouse25-140123", "Mouse28-140310"]
rng = np.random.default_rng(0)

data = {s: np.load(os.path.join(CACHE, f"{s}.npz")) for s in SESSIONS}

def upper_pairs(C):
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]

def ang_dist(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))

# ---------- Fig A: sorted correlation matrices per state (best session + all) ----------
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad(color="0.85")
fig, axes = plt.subplots(len(SESSIONS), 3, figsize=(9.5, 2.1 * len(SESSIONS)),
                         constrained_layout=True)
for r, s in enumerate(SESSIONS):
    d = data[s]
    pref = d["pref"][d["hd_idx"]]
    order = np.argsort(pref)
    for c, (key, lab) in enumerate([("corr_wake", "Wake"), ("corr_rem", "REM"),
                                    ("corr_nrem", "NREM")]):
        C = d[key][np.ix_(order, order)]
        ax = axes[r, c]
        im = ax.imshow(np.ma.masked_invalid(C), cmap=cmap, vmin=-0.5, vmax=0.5,
                       origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(lab, fontsize=11)
        if c == 0:
            ax.set_ylabel(s.replace("Mouse", "M"), fontsize=9)
fig.suptitle("Pairwise correlation matrices, HD cells sorted by wake preferred direction")
fig.colorbar(im, ax=axes, shrink=0.6, label="Pearson r", location="right")
plt.savefig(f"{FIG}/fig_corrmatrices_sorted.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_corrmatrices_sorted.png")

# ---------- Fig B: corr vs angular distance per state (pooled) ----------
bins = np.linspace(0, np.pi, 13)
bc = (bins[:-1] + bins[1:]) / 2
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

ax = axes[0]
colors = {"corr_wake": "k", "corr_rem": "r", "corr_nrem": "b"}
labels = {"corr_wake": "Wake", "corr_rem": "REM", "corr_nrem": "NREM"}
for key in ["corr_wake", "corr_rem", "corr_nrem"]:
    all_d, all_c = [], []
    for s in SESSIONS:
        d = data[s]
        pref = d["pref"][d["hd_idx"]]
        iu = np.triu_indices(len(pref), k=1)
        dd = ang_dist(pref[iu[0]], pref[iu[1]])
        cc = upper_pairs(d[key])
        m = ~np.isnan(cc)
        all_d.append(dd[m]); all_c.append(cc[m])
    all_d = np.concatenate(all_d); all_c = np.concatenate(all_c)
    inds = np.digitize(all_d, bins) - 1
    means = [np.nanmean(all_c[inds == i]) for i in range(len(bc))]
    sems = [np.nanstd(all_c[inds == i]) / np.sqrt(np.sum(inds == i)) for i in range(len(bc))]
    ax.errorbar(np.degrees(bc), means, yerr=sems, color=colors[key],
                label=labels[key], capsize=2, lw=1.5)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("Angular distance between preferred directions (deg)")
ax.set_ylabel("Mean pairwise correlation")
ax.set_title("Ring profile: correlation vs. tuning distance")
ax.legend(frameon=False)

# ---------- Fig C: corr-of-corr scatter wake vs sleep ----------
ax = axes[1]
all_w, all_r, all_n = [], [], []
for s in SESSIONS:
    d = data[s]
    w, r, n = upper_pairs(d["corr_wake"]), upper_pairs(d["corr_rem"]), upper_pairs(d["corr_nrem"])
    m = ~np.isnan(w) & ~np.isnan(r) & ~np.isnan(n)
    all_w.append(w[m]); all_r.append(r[m]); all_n.append(n[m])
all_w = np.concatenate(all_w); all_r = np.concatenate(all_r); all_n = np.concatenate(all_n)
ax.scatter(all_w, all_r, s=6, c="r", alpha=0.4, label=f"REM (r={np.corrcoef(all_w, all_r)[0,1]:.2f})")
ax.scatter(all_w, all_n, s=6, c="b", alpha=0.4, label=f"NREM (r={np.corrcoef(all_w, all_n)[0,1]:.2f})")
lim = [-0.6, 0.8]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Wake pairwise correlation")
ax.set_ylabel("Sleep pairwise correlation")
ax.set_title(f"Structure preserved in sleep ({len(all_w)} pairs, {len(SESSIONS)} sessions)")
ax.legend(frameon=False, markerscale=2)
plt.tight_layout()
plt.savefig(f"{FIG}/fig_ring_profile_and_preservation.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_ring_profile_and_preservation.png")

# ---------- permutation null for corr-of-corr ----------
def corr_of_corr(d, k1, k2):
    a, b = upper_pairs(d[k1]), upper_pairs(d[k2])
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[m], b[m])[0, 1]

nperm = 1000
null_r, null_n = [], []
for s in SESSIONS:
    d = data[s]
    nunits = len(d["hd_idx"])
    for _ in range(nperm // len(SESSIONS)):
        perm = rng.permutation(nunits)
        # permute rows/cols of the sleep matrix = shuffle pair labels
        Cr = d["corr_rem"][np.ix_(perm, perm)]
        Cn = d["corr_nrem"][np.ix_(perm, perm)]
        a = upper_pairs(d["corr_wake"])
        b = upper_pairs(Cr); c = upper_pairs(Cn)
        m = ~np.isnan(a) & ~np.isnan(b)
        null_r.append(np.corrcoef(a[m], b[m])[0, 1])
        m = ~np.isnan(a) & ~np.isnan(c)
        null_n.append(np.corrcoef(a[m], c[m])[0, 1])
null_r = np.array(null_r); null_n = np.array(null_n)

obs_r = [corr_of_corr(data[s], "corr_wake", "corr_rem") for s in SESSIONS]
obs_n = [corr_of_corr(data[s], "corr_wake", "corr_nrem") for s in SESSIONS]
print("per-session corr-of-corr wake-REM:", np.round(obs_r, 3))
print("per-session corr-of-corr wake-NREM:", np.round(obs_n, 3))
print("null REM: mean %.3f, 99th pct %.3f" % (null_r.mean(), np.percentile(null_r, 99)))
print("null NREM: mean %.3f, 99th pct %.3f" % (null_n.mean(), np.percentile(null_n, 99)))

fig, ax = plt.subplots(figsize=(5.5, 3.6))
b = np.linspace(-0.3, 1.0, 60)
ax.hist(null_r, bins=b, color="r", alpha=0.45, density=True, label="REM null (label perm.)")
ax.hist(null_n, bins=b, color="b", alpha=0.45, density=True, label="NREM null (label perm.)")
for v, c, lab in [(np.mean(obs_r), "r", f"observed wake-REM = {np.mean(obs_r):.2f}"),
                  (np.mean(obs_n), "b", f"observed wake-NREM = {np.mean(obs_n):.2f}")]:
    ax.axvline(v, color=c, lw=2)
    ax.text(v, ax.get_ylim()[1] * 0.9, lab, rotation=90, va="top", ha="right", color=c, fontsize=9)
ax.set_xlabel("Correlation of pairwise-correlation structure (wake vs sleep)")
ax.set_ylabel("Density")
ax.set_title("Sleep preserves the wake ring structure")
plt.tight_layout()
plt.savefig(f"{FIG}/fig_corr_of_corr_null.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig_corr_of_corr_null.png")

np.savez("pooled_stats.npz", obs_r=obs_r, obs_n=obs_n, null_r=null_r, null_n=null_n,
         all_w=all_w, all_r=all_r, all_n=all_n)
print("done")
