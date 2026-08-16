# Polish fig1: pooled psychometric + 0%-contrast bias, clean layout
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import decoding_lib as dl

SESSIONS = ["sub-92130c1b", "sub-70bf8cbd", "sub-9bebfe0b", "sub-c6e8125f"]
colors = {0.2: "#d62728", 0.5: "#7f7f7f", 0.8: "#1f77b4"}

def get_info(s):
    trials = pd.DataFrame({k[6:]: d[k] for k in d.files if k.startswith("trial_")}) \
        if False else None
    return None

# load per-session trial info
infos = {}
for s in SESSIONS:
    d = np.load(f"cache_{s}.npz")
    trials = pd.DataFrame({k[len("trial_"):]: d[k] for k in d.files if k.startswith("trial_")})
    cl, cr = trials["contrastLeft"].to_numpy(), trials["contrastRight"].to_numpy()
    choice = trials["choice"].to_numpy()
    stim_on = trials["stimOn_times"].to_numpy()
    valid = ~np.isnan(choice) & ~np.isnan(stim_on)
    zero_c = valid & (np.nan_to_num(cl) == 0) & (np.nan_to_num(cr) == 0)
    infos[s] = dict(
        sc=np.where(np.isnan(cl), cr, -cl),
        chl=(choice == 1).astype(float),
        p_left=trials["probabilityLeft"].to_numpy(),
        valid=valid, zero_c=zero_c,
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# --- left: psychometric, mean across sessions +- sem, faint per-session ---
ax = axes[0]
contrast_levels = [-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 1]
for pl in [0.2, 0.5, 0.8]:
    per_sess = []
    for s in SESSIONS:
        info = infos[s]
        m = info["valid"] & (info["p_left"] == pl)
        ys = [np.mean(info["chl"][m & (info["sc"] == c)]) if np.sum(m & (info["sc"] == c)) > 0
              else np.nan for c in contrast_levels]
        per_sess.append(ys)
        ax.plot(contrast_levels, ys, "-", color=colors[pl], lw=0.8, alpha=0.25, zorder=1)
    per_sess = np.array(per_sess)
    mean = np.nanmean(per_sess, axis=0)
    sem = np.nanstd(per_sess, axis=0) / np.sqrt(len(SESSIONS))
    ax.plot(contrast_levels, mean, "o-", color=colors[pl], lw=2.2, ms=6,
            label=f"p(left)={pl}", zorder=3)
    ax.fill_between(contrast_levels, mean - sem, mean + sem, color=colors[pl],
                    alpha=0.18, zorder=2)
ax.axhline(0.5, color="k", lw=0.5, ls="--")
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xscale("symlog", linthresh=0.02)
ax.set_xticks([-1, -0.25, -0.0625, 0, 0.0625, 0.25, 1])
ax.set_xticklabels(["-100", "-25", "-6.25", "0", "6.25", "25", "100"])
ax.set_xlabel("signed contrast (%, left < 0 < right)")
ax.set_ylabel("P(choose left)")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Psychometric curves by block prior\n(mean ± sem across sessions; faint: individual)")
ax.legend(frameon=False, loc="lower left")

# --- right: 0% contrast, per-session points + mean ---
ax = axes[1]
rng = np.random.default_rng(1)
for j, pl in enumerate([0.2, 0.5, 0.8]):
    vals = []
    for i, s in enumerate(SESSIONS):
        info = infos[s]
        m = info["zero_c"] & (info["p_left"] == pl)
        if m.sum() >= 5:
            v = np.mean(info["chl"][m])
            vals.append(v)
            ax.scatter(j + rng.uniform(-0.12, 0.12), v, s=42, color=colors[pl],
                       alpha=0.55, edgecolor="white", lw=0.5, zorder=3)
    ax.plot(j, np.mean(vals), "D", color="black", ms=8, zorder=4)
    ax.plot([j - 0.28, j + 0.28], [np.mean(vals)] * 2, color="black", lw=1.4, zorder=4)
means = [np.mean([np.mean(infos[s]["chl"][infos[s]["zero_c"] & (infos[s]["p_left"] == pl)])
                  for s in SESSIONS
                  if np.sum(infos[s]["zero_c"] & (infos[s]["p_left"] == pl)) >= 5])
         for pl in [0.2, 0.5, 0.8]]
ax.plot([0, 1, 2], means, color="black", lw=1.2, ls="-", alpha=0.5, zorder=2)
ax.axhline(0.5, color="k", lw=0.5, ls="--")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["p(left)=0.2", "p(left)=0.5", "p(left)=0.8"])
ax.set_ylabel("P(choose left)")
ax.set_ylim(0, 1)
ax.set_title("Choice on 0%-contrast trials tracks the prior\n(points: sessions; diamonds: mean)")
fig.tight_layout()
fig.savefig("fig1_behavior_polished.png", dpi=150)
print("saved fig1_behavior_polished.png")
