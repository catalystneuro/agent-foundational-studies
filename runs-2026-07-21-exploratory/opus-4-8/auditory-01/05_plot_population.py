"""Population-level figures: tuning heatmap, best-frequency distribution, statistics."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open("pooled_results.pkl", "rb") as fh:
    res = pickle.load(fh)
pop, tun, freqs = res["pop"], res["tun"], res["freqs"]
fcols = [c for c in tun.columns if isinstance(c, float)]
T = tun[fcols].values
TC_A, TC_B = res["tc_a"], res["tc_b"]   # split-half tuning curves
khz = np.array(fcols) / 1000
colors = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(fcols)))

sel = pop.sig_freq.values & pop.sig_sound.values
print(f"{sel.sum()} sound-responsive and frequency-selective units of {len(pop)}")

fig = plt.figure(figsize=(14.5, 10))
gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)

# ---- (a) normalized tuning curves, sorted by best frequency ------------------
ax = fig.add_subplot(gs[0, 0])
Ts = T[sel]
bf_a = np.argmax(TC_A[sel], 1)           # preferred frequency from odd trials
held = TC_B[sel]                          # tuning read out on even trials
norm = held / np.abs(held).max(1, keepdims=True)
order = np.lexsort((-norm[np.arange(len(norm)), bf_a], bf_a))
im = ax.imshow(norm[order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
               extent=[-0.5, len(fcols) - 0.5, len(order), 0], interpolation="nearest")
ax.set_xticks(range(len(fcols)))
ax.set_xticklabels([f"{int(k)}" for k in khz])
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("unit (sorted by preferred frequency)")
ax.set_title("a  cross-validated tuning curves\n(colour: evoked rate / peak |evoked|)",
             loc="left", pad=10, fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

# ---- (b) mean tuning curve per best-frequency group --------------------------
ax = fig.add_subplot(gs[0, 1])
for j, f in enumerate(fcols):
    grp = held[bf_a == j]
    if len(grp) < 5:
        continue
    m = grp / np.abs(grp).max(1, keepdims=True)
    ax.errorbar(khz, m.mean(0), yerr=m.std(0) / np.sqrt(len(m)), marker="o",
                color=colors[j], lw=1.8, capsize=2,
                label=f"BF {int(f/1000)} kHz (n={len(grp)})")
ax.set_xscale("log", base=2)
ax.set_xticks(khz)
ax.set_xticklabels([f"{int(k)}" for k in khz])
ax.axhline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("tone frequency (kHz)")
ax.set_ylabel("normalized evoked rate")
ax.set_title("b  mean tuning by preferred frequency\n(preference from held-out trials)",
             loc="left", pad=10, fontsize=11)
ax.legend(fontsize=7.5, frameon=False)

# ---- (c) best-frequency distribution per mouse -------------------------------
ax = fig.add_subplot(gs[0, 2])
sub = pop[sel]
ct = pd.crosstab(sub.subject, sub.best_freq, normalize="index")
bottom = np.zeros(len(ct))
for j, f in enumerate(fcols):
    v = ct[f].values if f in ct else np.zeros(len(ct))
    ax.bar(ct.index, v, bottom=bottom, color=colors[j], label=f"{int(f/1000)} kHz")
    bottom += v
ax.set_ylabel("fraction of frequency-selective units")
ax.set_xlabel("mouse")
ax.set_title("c  preferred-frequency composition\n(per mouse)", loc="left", pad=10,
             fontsize=11)
ax.legend(fontsize=7.5, frameon=False, ncol=5, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), columnspacing=0.8, handlelength=1.2)

# ---- (d) fraction responsive / selective per session -------------------------
ax = fig.add_subplot(gs[1, 0])
g = pop.groupby("session").agg(n=("unit", "size"), resp=("sig_sound", "mean"),
                               freqsel=("sig_freq", "mean"))
g = g.sort_values("resp")
x = np.arange(len(g))
ax.bar(x - 0.2, g.resp, 0.4, color="0.4", label="sound-responsive")
ax.bar(x + 0.2, g.freqsel, 0.4, color="tab:red", label="frequency-selective")
ax.set_xticks(x)
ax.set_xticklabels([f"{s} (n={n})" for s, n in zip(g.index, g.n)], rotation=90,
                   fontsize=7)
ax.set_ylabel("fraction of units")
ax.set_title("d  responsive and selective fractions", loc="left", pad=12)
ax.legend(fontsize=8, frameon=False)

# ---- (e) observed vs shuffled selectivity ------------------------------------
ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, 1, 41)
ax.hist(pop.sparseness[sel], bins, color="tab:red", alpha=0.7, label="observed")
ax.hist(pop.null_sparseness[sel], bins, color="0.5", alpha=0.7,
        label="frequency labels shuffled")
ax.set_xlabel("lifetime sparseness of tuning curve")
ax.set_ylabel("units")
ax.set_title("e  selectivity vs. shuffled control", loc="left", pad=12)
ax.legend(fontsize=8, frameon=False)

# ---- (f) tuning range vs peak evoked rate ------------------------------------
ax = fig.add_subplot(gs[1, 2])
ax.scatter(pop.peak_evoked[~sel], pop.tuning_range[~sel], s=6, color="0.7",
           label="not selective")
ax.scatter(pop.peak_evoked[sel], pop.tuning_range[sel], s=6, color="tab:red",
           label="frequency-selective")
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("peak evoked rate (spikes/s)")
ax.set_ylabel("best - worst frequency response (spikes/s)")
ax.set_title("f  depth of frequency modulation", loc="left", pad=12)
ax.legend(fontsize=8, frameon=False, loc="upper left")

fig.suptitle("Pure-tone frequency tuning in mouse auditory cortex "
             f"(DANDI:000986, {len(pop)} units, {pop.subject.nunique()} mice, "
             f"{pop.session.nunique()} sessions)", y=0.97)
fig.savefig("fig03_population_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig03_population_tuning.png")
