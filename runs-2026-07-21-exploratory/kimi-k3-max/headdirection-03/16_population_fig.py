"""Population summary figure from per-session npz results."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results_multisession"
FIGDIR = "figs_proto"

files = sorted(glob.glob(os.path.join(OUT, "results_*.npz")))
print(f"{len(files)} sessions")

rows = []
all_mvl_hd, all_mvl_not, all_pref_hd = [], [], []
pair_rows = []
for f in files:
    d = np.load(f)
    name = str(d["session"])
    is_hd = d["is_hd"]
    mvl = d["mvl"]
    all_mvl_hd.append(mvl[is_hd])
    all_mvl_not.append(mvl[~is_hd & ~np.isnan(mvl)])
    all_pref_hd.append(d["pref"][is_hd])
    rows.append(dict(session=name, n_units=len(is_hd),
                     n_hd=int(is_hd.sum()),
                     r_rem=float(d["r_rem"]), r_nrem=float(d["r_nrem"]),
                     p_rem=float(d["p_rem"]) if "p_rem" in d.files else np.nan,
                     p_nrem=float(d["p_nrem"]) if "p_nrem" in d.files else np.nan))
    if "pairs_wake_rem" in d.files:
        pair_rows.append(dict(pw_rem=d["pairs_wake_rem"],
                              ps_rem=d["pairs_rem"],
                              pw_nrem=d["pairs_wake_nrem"],
                              ps_nrem=d["pairs_nrem"]))

df = pd.DataFrame(rows)
all_mvl_hd = np.concatenate(all_mvl_hd)
all_mvl_not = np.concatenate(all_mvl_not)
all_pref_hd = np.concatenate(all_pref_hd)
pw_rem = np.concatenate([p["pw_rem"] for p in pair_rows])
ps_rem = np.concatenate([p["ps_rem"] for p in pair_rows])
pw_nrem = np.concatenate([p["pw_nrem"] for p in pair_rows])
ps_nrem = np.concatenate([p["ps_nrem"] for p in pair_rows])
print(df.to_string(index=False))
print(f"total HD: {df.n_hd.sum()}/{df.n_units.sum()}, "
      f"REM pairs: {len(pw_rem)}, NREM pairs: {len(pw_nrem)}")

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(df))
ax.bar(x, df["n_units"], color="0.75", label="all units")
ax.bar(x, df["n_hd"], color="crimson", label="HD cells")
ax.set_xticks(x)
ax.set_xticklabels(df["session"], fontsize=7, rotation=45,
                   ha="right")
ax.set_ylabel("unit count")
ax.set_title("HD cells per session")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 1])
bins = np.arange(0, 1.01, 0.05)
ax.hist(all_mvl_not, bins=bins, color="0.6",
        label=f"not HD (n={len(all_mvl_not)})")
ax.hist(all_mvl_hd, bins=bins, color="crimson",
        label=f"HD (n={len(all_mvl_hd)})")
ax.set_xlabel("mean vector length")
ax.set_ylabel("unit count")
ax.set_title("Tuning strength, pooled sessions")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2], projection="polar")
occ_p, edges_p = np.histogram(all_pref_hd, bins=24, range=(0, 2 * np.pi))
centers_p = (edges_p[:-1] + edges_p[1:]) / 2
ax.bar(centers_p, occ_p, width=2 * np.pi / 24, color="crimson", alpha=0.8)
ax.set_title("Preferred directions, all HD cells", pad=18)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
ax.set_xticklabels(["0", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"], fontsize=8)
ax.set_rlabel_position(90)
ax.tick_params(axis="y", labelsize=7)
z = len(all_pref_hd) * np.abs(np.exp(1j * all_pref_hd).mean()) ** 2
p_unif = np.exp(-z)
ax.text(0.5, -0.14, f"Rayleigh z={z:.2f}, p={p_unif:.2f}",
        transform=ax.transAxes, ha="center", fontsize=9)

ax = fig.add_subplot(gs[1, 0])
valid_r = df.dropna(subset=["r_rem"])
xx = np.arange(len(valid_r))
w = 0.38
ax.bar(xx - w / 2, valid_r["r_rem"], width=w, color="tab:red",
       label="wake-REM")
ax.bar(xx + w / 2, valid_r["r_nrem"], width=w, color="tab:blue",
       label="wake-NREM")
ax.set_xticks(xx)
ax.set_xticklabels(valid_r["session"], fontsize=7, rotation=45,
                   ha="right")
ax.set_ylabel("correlation of correlations")
ax.set_ylim(0, 1.18)
ax.set_title("HD structure preserved in sleep")
ax.legend(fontsize=8, loc="upper left")

ax = fig.add_subplot(gs[1, 1:])
ax.scatter(pw_rem, ps_rem, s=14, color="tab:red", alpha=0.5,
           label=f"wake-REM pairs (n={len(pw_rem)})")
ax.scatter(pw_nrem, ps_nrem, s=14, color="tab:blue", alpha=0.5,
           label=f"wake-NREM pairs (n={len(pw_nrem)})")
ax.plot([-0.5, 1], [-0.5, 1], "k--", lw=1)
r_all_rem = np.corrcoef(pw_rem, ps_rem)[0, 1]
r_all_nrem = np.corrcoef(pw_nrem, ps_nrem)[0, 1]
ax.set_xlabel("wake pairwise correlation")
ax.set_ylabel("sleep pairwise correlation")
ax.set_title(f"All HD-cell pairs, pooled: r_REM={r_all_rem:.2f}, "
             f"r_NREM={r_all_nrem:.2f}")
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Head-direction cells across sessions, DANDI 000056", y=0.98)
fig.savefig(f"{FIGDIR}/fig5_population_summary.png", dpi=200)
plt.close(fig)
print("saved fig5")
