"""Population figures from results/all_units.csv (mirrors final script cells)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG = "figures"
all_units = pd.read_csv("results/all_units.csv")
freqs = np.array([2000., 4000., 8000., 16000., 32000.])
cmap = plt.cm.viridis
fcols = {f: cmap(i / (len(freqs) - 1)) for i, f in enumerate(freqs)}
tuned = all_units[all_units.kw_p < 0.01]
print(f"{len(all_units)} units, {len(tuned)} tuned "
      f"({100 * len(tuned) / len(all_units):.0f}%)")

# ---- fig7: heatmap ----
tune_cols = [f"tune_{int(f)}" for f in freqs]
exc = tuned[tuned.peak_net > 0].copy()
T = exc[tune_cols].to_numpy()
# min-max normalize each row so both excitatory peaks and flanking
# suppression are visible (rows span [0, 1])
row_min = T.min(axis=1, keepdims=True)
row_max = T.max(axis=1, keepdims=True)
denom = np.where(row_max - row_min > 0, row_max - row_min, 1)
T = (T - row_min) / denom
# sort by best frequency, then by center of mass of the rectified curve
rect = np.clip(exc[tune_cols].to_numpy(), 0, None)
com = (rect * np.arange(len(freqs))).sum(axis=1) / rect.sum(axis=1)
order = np.lexsort((com, exc["bf"].to_numpy()))
T = T[order]
exc_sorted = exc.iloc[order]

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(T, aspect="auto", cmap="magma",
               extent=[0, len(freqs) - 1, len(T), 0])
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel(f"tuned units (n = {len(T)}), sorted by best frequency")
ax.set_title("Normalized tuning curves across the population")
bf_vals = exc_sorted["bf"].to_numpy()
for f in freqs[:-1]:
    boundary = np.sum(bf_vals <= f)
    ax.axhline(boundary, color="cyan", lw=0.7, alpha=0.8)
plt.colorbar(im, label="normalized response")
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_population_heatmap.png", dpi=150)
plt.close()
print("saved fig7")

# ---- fig8: population stats ----
fig, axes = plt.subplots(2, 3, figsize=(13, 7))

ax = axes[0, 0]
per_ses = all_units.groupby(["subject", "session"]).agg(
    n=("kw_p", "size"), tuned=("kw_p", lambda s: (s < 0.01).mean())).reset_index()
labels = [f"{s.replace('sub-', '')}\n{ss}" for s, ss in zip(per_ses.subject, per_ses.session)]
ax.bar(range(len(per_ses)), 100 * per_ses.tuned, color="steelblue")
ax.set_xticks(range(len(per_ses)))
ax.set_xticklabels(labels, fontsize=6, rotation=90)
ax.set_ylabel("% units tuned (KW p<0.01)")
ax.set_title(f"Fraction tuned per session (overall {100 * len(tuned) / len(all_units):.0f}%)")

ax = axes[0, 1]
bf_counts = [np.sum(tuned.bf == f) for f in freqs]
ax.bar(range(len(freqs)), bf_counts, color=[fcols[f] for f in freqs])
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels([f"{int(f/1000)}" for f in freqs])
ax.set_xlabel("best frequency (kHz)")
ax.set_ylabel("# tuned units")
ax.set_title("Best-frequency distribution")

ax = axes[0, 2]
w = tuned.width_oct.dropna()
ax.hist(w, bins=np.arange(-0.5, 5, 1), color="teal", rwidth=0.85)
ax.set_xlabel("tuning width at half max (octaves)")
ax.set_ylabel("# units")
ax.set_title(f"Tuning width (median {w.median():.0f} octave)")

ax = axes[1, 0]
lat = tuned.latency_ms.dropna()
lat = lat[lat > 0]
ax.hist(lat, bins=np.arange(0, 105, 5), color="darkorange", rwidth=0.9)
ax.axvline(lat.median(), color="k", ls="--", label=f"median {lat.median():.0f} ms")
ax.set_xlabel("response latency at best frequency (ms)")
ax.set_ylabel("# units")
ax.legend(frameon=False)
ax.set_title("Response latency")

ax = axes[1, 1]
s = tuned.selectivity.dropna()
ax.hist(s, bins=20, color="purple", rwidth=0.9)
ax.set_xlabel("selectivity index (0 = flat, 1 = single frequency)")
ax.set_ylabel("# units")
ax.set_title(f"Frequency selectivity (median {s.median():.2f})")

ax = axes[1, 2]
ax.scatter(all_units.rate, all_units.peak_net,
           c=["crimson" if p < 0.01 else "gray" for p in all_units.kw_p],
           s=4, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("peak evoked - base (Hz)")
ax.set_title("Response magnitude vs firing rate")

plt.tight_layout()
plt.savefig(f"{FIG}/fig8_population_stats.png", dpi=150)
plt.close()
print("saved fig8")

# numbers for README / summary
print("\n--- summary numbers ---")
print(f"units analyzed: {len(all_units)}, tuned: {len(tuned)} "
      f"({100 * len(tuned) / len(all_units):.1f}%)")
print("BF counts:", dict(zip(freqs.astype(int), bf_counts)))
print(f"median width: {w.median()} oct; median latency: {lat.median()} ms; "
      f"median selectivity: {s.median():.3f}")
print(f"per-session tuned fraction range: {100*per_ses.tuned.min():.0f}-{100*per_ses.tuned.max():.0f}%")
print(f"sessions: {len(per_ses)}, mice: {all_units.subject.nunique()}")
