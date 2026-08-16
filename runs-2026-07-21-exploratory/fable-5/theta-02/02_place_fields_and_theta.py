"""Place fields and theta phase locking for one session of DANDI:000044."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import theta_lib as tl

res = tl.analyze_session(tl.SESSIONS[0])
spikes, pyr, pos, laps = res["spikes"], res["pyr"], res["position"], res["laps"]
run_ep, lfp, theta_phase = res["run_ep"], res["lfp"], res["theta_phase"]
lock, maps = res["locking"], res["maps"]

# ---------------------------------------------------------------------------------
# Figure 3: place field maps
# ---------------------------------------------------------------------------------
directions = list(maps.keys())
fig, axes = plt.subplots(len(directions), 3, figsize=(14, 4 * len(directions)),
                         width_ratios=[1.2, 1.2, 1], squeeze=False)
for row, direction in enumerate(directions):
    m = maps[direction]
    sel = m["rates"][m["is_place"]]
    order = np.argsort(np.argmax(sel, axis=1))
    norm = sel[order] / sel[order].max(axis=1, keepdims=True)
    im = axes[row, 0].imshow(
        norm, aspect="auto", origin="lower", cmap="viridis",
        extent=[res["track_range"][0], res["track_range"][1], 0, norm.shape[0]],
    )
    axes[row, 0].set_ylabel("Place cell (sorted by peak)")
    axes[row, 0].set_xlabel("Position (cm)")
    axes[row, 0].set_title("%sward runs: %d place cells" % (direction.capitalize(), norm.shape[0]))
    plt.colorbar(im, ax=axes[row, 0], label="Normalized rate")

    step = max(1, len(order) // 10)
    for i in order[::step]:
        axes[row, 1].plot(m["centers"], sel[i], lw=1.4)
    axes[row, 1].set_xlabel("Position (cm)")
    axes[row, 1].set_ylabel("Firing rate (Hz)")
    axes[row, 1].set_title("Rate maps of 10 example place cells")

    axes[row, 2].scatter(m["si"], m["peak"], s=16, c=np.where(m["is_place"], "tab:red", "0.75"))
    axes[row, 2].axvline(0.5, ls="--", lw=1, color="k")
    axes[row, 2].axhline(1.0, ls="--", lw=1, color="k")
    axes[row, 2].set_xlabel("Spatial information (bits/spike)")
    axes[row, 2].set_ylabel("Peak rate (Hz)")
    axes[row, 2].set_yscale("log")
    axes[row, 2].set_title("Place-cell selection\n(red = passes all criteria)")

fig.suptitle(
    "Directional place fields in dorsal CA1, %s (%d pyramidal cells)" % (res["session"], len(pyr)),
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig03_place_fields.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------------
# Theta phase locking
# ---------------------------------------------------------------------------------
phases = spikes.restrict(run_ep).value_from(theta_phase)
pyr_ids = [u for u in lock if lock[u]["cell_type"] == "excitatory" and lock[u]["n"] >= 50]
inh_ids = [u for u in lock if lock[u]["cell_type"] == "inhibitory" and lock[u]["n"] >= 50]
print(
    "theta locking (Rayleigh p<0.01): %d/%d pyramidal, %d/%d interneurons"
    % (
        sum(lock[u]["p"] < 0.01 for u in pyr_ids), len(pyr_ids),
        sum(lock[u]["p"] < 0.01 for u in inh_ids), len(inh_ids),
    )
)
all_pyr_ph = np.concatenate([phases[u].values for u in pyr_ids])
all_inh_ph = np.concatenate([phases[u].values for u in inh_ids])
mu_p, r_p, p_p = tl.rayleigh(all_pyr_ph)
mu_i, r_i, p_i = tl.rayleigh(all_inh_ph)
print("pyramidal pooled: preferred phase %.0f deg, MRL %.3f, Rayleigh p=%.2g" % (np.degrees(mu_p), r_p, p_p))
print("interneuron pooled: preferred phase %.0f deg, MRL %.3f, Rayleigh p=%.2g" % (np.degrees(mu_i), r_i, p_i))

# ---------------------------------------------------------------------------------
# Figure 4: theta entrainment
# ---------------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)
bins = np.linspace(0, 2 * np.pi, 37)
ctr = np.degrees((bins[:-1] + bins[1:]) / 2)
tt = np.linspace(0, 720, 400)

for col, (ph_all, color, label, mu, r) in enumerate(
    (
        (all_pyr_ph, "tab:blue", "Pyramidal cells", mu_p, r_p),
        (all_inh_ph, "tab:purple", "Interneurons", mu_i, r_i),
    )
):
    ax = fig.add_subplot(gs[0, col])
    h, _ = np.histogram(ph_all, bins=bins)
    h = h / h.sum()
    ax.bar(np.r_[ctr, ctr + 360], np.r_[h, h], width=10, color=color)
    ax.set_xlabel("Theta phase (deg)")
    ax.set_ylabel("Fraction of spikes")
    ax.set_ylim(0, h.max() * 1.35)
    ax.set_xlim(0, 720)
    ax.set_xticks([0, 180, 360, 540, 720])
    ax.axvline(np.degrees(mu), color="tab:red", lw=2)
    ax.axvline(np.degrees(mu) + 360, color="tab:red", lw=2)
    ax2 = ax.twinx()
    ax2.plot(tt, np.cos(np.radians(tt)), color="0.6", lw=1.2)
    ax2.set_yticks([])
    ax2.set_ylim(-1, 6)
    ax.set_title(
        "%s: n=%d spikes\npreferred phase %.0f$\\degree$ (red), MRL=%.3f"
        % (label, len(ph_all), np.degrees(mu), r)
    )

ax = fig.add_subplot(gs[0, 2], projection="polar")
for ids, color, lab in ((pyr_ids, "tab:blue", "pyramidal"), (inh_ids, "tab:purple", "interneuron")):
    mus = np.array([lock[u]["mu"] for u in ids])
    rs = np.array([lock[u]["mrl"] for u in ids])
    s = np.array([lock[u]["p"] < 0.01 for u in ids])
    ax.scatter(mus[s], rs[s], s=24, color=color, alpha=0.75, label=lab)
ax.set_title("Preferred phase vs. locking strength\n(cells with Rayleigh p<0.01)", pad=18, fontsize=10)
ax.set_rlabel_position(135)
ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.12), fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.hist([lock[u]["mrl"] for u in pyr_ids], bins=20, alpha=0.7, color="tab:blue", label="pyramidal")
ax.hist([lock[u]["mrl"] for u in inh_ids], bins=20, alpha=0.7, color="tab:purple", label="interneuron")
ax.set_xlabel("Mean resultant length")
ax.set_ylabel("Number of cells")
ax.set_title("Strength of theta phase locking")
ax.legend(fontsize=9)

# spike-triggered LFP average, pyramidal spikes pooled
ax = fig.add_subplot(gs[1, 1])
win = int(0.25 * tl.LFP_FS)
lfp_r = lfp.restrict(run_ep)
sp_all = np.sort(np.concatenate([spikes.restrict(run_ep)[u].t for u in pyr_ids]))
idx = np.searchsorted(lfp_r.t, sp_all)
idx = idx[(idx > win) & (idx < len(lfp_r) - win)]
sub = np.random.default_rng(0).choice(idx, size=min(20000, len(idx)), replace=False)
seg = np.stack([lfp_r.values[i - win : i + win] for i in sub])
ax.plot(np.arange(-win, win) / tl.LFP_FS * 1000, seg.mean(axis=0) * 1e3, "k")
ax.axvline(0, ls="--", color="tab:red")
ax.set_xlabel("Time from spike (ms)")
ax.set_ylabel("LFP (mV)")
ax.set_title("Spike-triggered LFP average\n(%d pyramidal spikes)" % len(sub))

ax = fig.add_subplot(gs[1, 2])
ax.scatter([spikes.rate[u] for u in pyr_ids], [lock[u]["mrl"] for u in pyr_ids],
           s=16, color="tab:blue", label="pyramidal")
ax.scatter([spikes.rate[u] for u in inh_ids], [lock[u]["mrl"] for u in inh_ids],
           s=16, color="tab:purple", label="interneuron")
ax.set_xscale("log")
ax.set_xlabel("Session firing rate (Hz)")
ax.set_ylabel("Mean resultant length")
ax.set_title("Locking strength vs. firing rate")
ax.legend(fontsize=9)

fig.suptitle(
    "Theta phase entrainment of CA1 spiking during running, %s "
    "(phase from LFP channel %d; 0$\\degree$ = peak of the 6-10 Hz filtered LFP, grey trace)"
    % (res["session"], res["best_ch"]),
    fontsize=12, y=1.03,
)
fig.savefig("fig04_theta_entrainment.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("done")
