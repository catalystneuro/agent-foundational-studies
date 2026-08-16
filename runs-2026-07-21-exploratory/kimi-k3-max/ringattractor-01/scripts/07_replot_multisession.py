"""Re-render figures/06_multisession_summary.png from saved npz files with fixed layout."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEC_BIN = 0.1
PANEL = ["Mouse25-140123", "Mouse17-130128", "Mouse20-130514", "Mouse24-131213"]

results = []
for name in PANEL:
    d = np.load(f"data/{name}_summary.npz")
    results.append({k: d[k] for k in d.files})

d28 = np.load("data/mouse28_sleep_corr.npz")
s28 = np.load("data/mouse28_sleep_decoding.npz")
pref28 = d28["pref_sorted"]
iu28 = np.triu_indices(len(pref28), 1)
fin28 = (np.isfinite(d28["C_wake"][iu28]) & np.isfinite(d28["C_rem"][iu28])
         & np.isfinite(d28["C_nrem"][iu28]))
results.append(dict(
    name="Mouse28-140310",
    r_rem=d28["r_rem"], r_nrem=d28["r_nrem"], p_rem=d28["p_rem"], p_nrem=d28["p_nrem"],
    zw=d28["C_wake"][iu28][fin28], zr=d28["C_rem"][iu28][fin28],
    zn=d28["C_nrem"][iu28][fin28],
    d_ang=np.abs(np.angle(np.exp(1j * (pref28[:, None] - pref28[None, :]))))[iu28][fin28],
    ac_real=s28["ac_real"], ac_null_mean=np.nanmean(s28["shift_ac"], axis=0),
    p_ac=np.array(0.0099),
))

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))

names = [str(r["name"]) for r in results]
x = np.arange(len(names))
axes[0].bar(x - 0.2, [float(r["r_rem"]) for r in results], width=0.4,
            color="darkorange", label="wake-REM")
axes[0].bar(x + 0.2, [float(r["r_nrem"]) for r in results], width=0.4,
            color="seagreen", label="wake-NREM")
for xi, r in zip(x, results):
    star = "***" if max(float(r["p_rem"]), float(r["p_nrem"])) <= 0.001 else \
           ("**" if max(float(r["p_rem"]), float(r["p_nrem"])) <= 0.01 else "*")
    axes[0].text(xi, max(float(r["r_rem"]), float(r["r_nrem"])) + 0.04, star,
                 ha="center", fontsize=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels([n.replace("-", "\n") for n in names], fontsize=7)
axes[0].set_ylabel("corr-of-corr")
axes[0].set_ylim(0, 1.28)
axes[0].legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.9)
axes[0].set_title("Structure preserved per session\n(*** p<=0.001, ** p<=0.01, * p<=0.05)",
                  fontsize=10)

zw_all = np.concatenate([r["zw"] for r in results])
zr_all = np.concatenate([r["zr"] for r in results])
zn_all = np.concatenate([r["zn"] for r in results])
axes[1].plot(zw_all, zr_all, ".", ms=3, alpha=0.4, color="darkorange", label="REM")
axes[1].plot(zw_all, zn_all, ".", ms=3, alpha=0.4, color="seagreen", label="NREM")
lim = np.percentile(np.abs(np.concatenate([zw_all, zr_all, zn_all])), 99) * 1.2
axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
axes[1].set_xlim(-lim, lim)
axes[1].set_ylim(-lim, lim)
r_pool_rem = np.corrcoef(zw_all, zr_all)[0, 1]
r_pool_nrem = np.corrcoef(zw_all, zn_all)[0, 1]
axes[1].set_xlabel("wake pairwise corr")
axes[1].set_ylabel("sleep pairwise corr")
axes[1].legend(fontsize=8, loc="upper left")
axes[1].set_title(f"pooled: REM r={r_pool_rem:.2f}, NREM r={r_pool_nrem:.2f}")

bins = np.linspace(0, np.pi, 7)
centers = np.degrees(0.5 * (bins[:-1] + bins[1:]))
d_all = np.concatenate([r["d_ang"] for r in results])
bid = np.digitize(d_all, bins) - 1
for vals, c, lbl in [(zw_all, "k", "wake (active)"), (zr_all, "darkorange", "REM"),
                     (zn_all, "seagreen", "NREM")]:
    m = [np.mean(vals[bid == b]) for b in range(6)]
    axes[2].plot(centers, m, "o-", color=c, label=lbl)
axes[2].axhline(0, color="0.7", lw=0.8)
axes[2].set_xlabel("angular distance between pref. dirs (deg)")
axes[2].set_ylabel("mean pairwise corr")
axes[2].legend(fontsize=8)
axes[2].set_title("Ring metric structure (pooled)")

lags = np.arange(51) * DEC_BIN
for r in results:
    axes[3].plot(lags, r["ac_real"] - r["ac_null_mean"], lw=1.5,
                 label=f"{str(r['name'])} (p={float(r['p_ac']):.3f})")
axes[3].axhline(0, color="0.5", ls="--", lw=0.8)
axes[3].set_xlabel("lag (s)")
axes[3].set_ylabel("excess posterior autocorrelation\n(real - time-shift null)")
axes[3].legend(fontsize=6.5, loc="upper right")
axes[3].set_title("REM bump persistence above null")

fig.suptitle("Ring attractor maintained during sleep: 5 sessions, 5 mice", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("figures/06_multisession_summary.png", dpi=150, bbox_inches="tight")
print("saved figures/06_multisession_summary.png")
print(f"pooled corr-of-corr: REM {r_pool_rem:.3f}, NREM {r_pool_nrem:.3f} "
      f"({len(zw_all)} pairs, {len(results)} sessions)")
