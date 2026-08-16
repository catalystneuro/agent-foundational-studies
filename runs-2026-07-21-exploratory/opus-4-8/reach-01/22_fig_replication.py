"""fig06: MC_RTT replication and the position/velocity confound, side by side with MC_Maze."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BIN = 0.02
CACHE = "./cache"

R = pickle.load(open(f"{CACHE}/rtt_results.pkl", "rb"))
Mr = pickle.load(open(f"{CACHE}/rtt_binned.pkl", "rb"))["M"]
maze_lag = pickle.load(open(f"{CACHE}/lagsweep.pkl", "rb"))
maze_vel = pickle.load(open(f"{CACHE}/veltuning.pkl", "rb"))
maze_glm = pickle.load(open(f"{CACHE}/glm_results.pkl", "rb"))["res"]

vel_fit, keep = R["vel_fit"], R["keep"]
i_n, i_k = R["i_n"], R["i_k"]


def tuning_2d(counts, vel, bin_size, nbins=11, vmax=400.0, min_occ=25):
    edges = np.linspace(-vmax, vmax, nbins + 1)
    ix = np.digitize(vel[:, 0], edges) - 1
    iy = np.digitize(vel[:, 1], edges) - 1
    ok = (ix >= 0) & (ix < nbins) & (iy >= 0) & (iy < nbins)
    occ = np.zeros((nbins, nbins))
    np.add.at(occ, (iy[ok], ix[ok]), 1)
    maps = np.full((counts.shape[1], nbins, nbins), np.nan)
    for u in range(counts.shape[1]):
        s = np.zeros((nbins, nbins))
        np.add.at(s, (iy[ok], ix[ok]), counts[ok, u])
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / occ / bin_size
        m[occ < min_occ] = np.nan
        maps[u] = m
    return edges, maps


vel_edges, vel_maps = tuning_2d(Mr["counts"][i_n], Mr["vel"][i_k], BIN)
order = np.argsort(-vel_fit["r2"].values)[:3]

fig = plt.figure(figsize=(16.5, 12.5))
gs = fig.add_gridspec(3, 4, hspace=.62, wspace=.46, top=0.90, bottom=0.055, left=0.06, right=0.975)

# --- row 0: example units (polar tuning + 2-D map) and PD distribution -------------
for j, u in enumerate(order):
    ax = fig.add_subplot(gs[0, j], projection="polar")
    v, se = R["tc"][u], R["tc_se"][u]
    ax.errorbar(np.r_[R["vctr"], R["vctr"][0]], np.r_[v, v[0]], yerr=np.r_[se, se[0]],
                color="k", lw=1.4, marker=".", ms=4)
    g = np.linspace(-np.pi, np.pi, 200)
    ax.plot(g, np.clip(vel_fit["b0"][u] + vel_fit["mod_depth"][u] * np.cos(g - vel_fit["pd"][u]),
                       0, None), "r", lw=2)
    ax.set_title(f"MC_RTT unit {keep[u]}\nPD={np.degrees(vel_fit['pd'][u]):.0f}$\\degree$, "
                 f"MD={vel_fit['mod_depth'][u]:.1f} Hz", pad=22, fontsize=10)
    ax.set_rlabel_position(np.degrees(vel_fit["pd"][u]) + 150)
    ax.tick_params(labelsize=7)

ax = fig.add_subplot(gs[0, 3], projection="polar")
sig = vel_fit["p"].values < 0.01
ax.hist(vel_fit["pd"].values[sig], bins=np.linspace(-np.pi, np.pi, 25), color="seagreen")
ax.set_title(f"Preferred velocity directions\n({sig.sum()}/{len(vel_fit)} units, p<0.01)",
             pad=22, fontsize=10)
ax.set_rlabel_position(255)
ax.tick_params(labelsize=8)

# --- row 1: 2-D maps, lag comparison, speed tuning --------------------------------
ext = [vel_edges[0], vel_edges[-1], vel_edges[0], vel_edges[-1]]
for j, u in enumerate(order[:2]):
    ax = fig.add_subplot(gs[1, j])
    im = ax.imshow(vel_maps[u], origin="lower", extent=ext, cmap="viridis")
    ax.arrow(0, 0, 320 * np.cos(vel_fit["pd"][u]), 320 * np.sin(vel_fit["pd"][u]), color="w",
             width=12, head_width=42, length_includes_head=True)
    ax.set(xlabel="$v_x$ (mm/s)", ylabel="$v_y$ (mm/s)",
           title=f"unit {keep[u]}: 2-D velocity map")
    plt.colorbar(im, ax=ax, label="rate (Hz)", fraction=.046)

ax = fig.add_subplot(gs[1, 2])
for lags, md, lbl, c in [(maze_lag["lags"], maze_lag["md"], "MC_Maze (Jenkins, 115 units)", "tab:red"),
                         (R["LAGS"], R["md_by_lag"], "MC_RTT (Indy, 94 units)", "seagreen")]:
    m = np.median(md, 1)
    ax.plot(np.asarray(lags) * 1000, m / m.max(), "-o", ms=3, color=c, label=lbl)
    ax.axvline(np.asarray(lags)[int(np.argmax(m))] * 1000, color=c, ls="--", lw=1)
ax.set(xlabel="neural lead time (ms)", ylabel="median modulation depth (normalized)",
       title="Neural activity leads the kinematics\nin both datasets")
ax.legend(fontsize=7.5, loc="lower center")

ax = fig.add_subplot(gs[1, 3])
norm_r = R["spd_tun"] / np.nanmax(R["spd_tun"], 1, keepdims=True)
norm_z = maze_vel["stun"] / np.nanmax(maze_vel["stun"], 1, keepdims=True)
ax.plot(maze_vel["sctr"], np.nanmean(norm_z, 0), "-o", color="tab:red", lw=2, ms=4, label="MC_Maze")
ax.plot(R["sctr"], np.nanmean(norm_r, 0), "-o", color="seagreen", lw=2, ms=4, label="MC_RTT")
ax.set(xlabel="hand speed (mm/s)", ylabel="normalized rate",
       title="Speed tuning within each unit's PD\n(population mean, $\\pm45\\degree$)")
ax.legend(fontsize=8)

# --- row 2: the confound, GLM comparison, decoding --------------------------------
ax = fig.add_subplot(gs[2, 0])
names = list(R["conf"])
w = 0.35
for i, comp in enumerate(["$v_x$", "$v_y$"]):
    ax.bar(np.arange(2) + (i - .5) * w, [R["conf"][n][i] for n in names], width=w, label=comp,
           color=["tab:blue", "tab:cyan"][i])
ax.axhline(0, color="k", lw=.8)
ax.set(xticks=range(2), ylabel="cross-validated $R^2$", ylim=(-0.08, 0.72),
       title="How much of hand velocity is\npredictable from hand position?")
ax.set_xticklabels(["MC_Maze\n(center-out)", "MC_RTT\n(random target)"], fontsize=8.5)
ax.legend(fontsize=8, loc="upper center", ncol=2)
for i, n in enumerate(names):
    for k, comp in enumerate([0, 1]):
        v = R["conf"][n][comp]
        ax.text(i + (k - .5) * w, v + .02, f"{v:.2f}", ha="center", fontsize=7.5)

ax = fig.add_subplot(gs[2, 1:3])
mods = ["speed only", "direction only", "direction x speed", "position", "dir x speed + position"]
mod_lbl = ["speed", "direction", "direction $\\times$ speed", "position", "dir $\\times$ speed + pos"]
y = np.arange(len(mods))
ax.barh(y + .19, [np.median(maze_glm[m]) for m in mods], height=.36, color="tab:red",
        label="MC_Maze (115 units)")
ax.barh(y - .19, [np.median(R["glm_scores"][m]) for m in mods], height=.36, color="seagreen",
        label="MC_RTT (94 units)")
for i, m in enumerate(mods):
    ax.text(np.median(maze_glm[m]) + .0008, i + .19, f"{np.median(maze_glm[m]):.3f}",
            va="center", fontsize=8)
    ax.text(np.median(R["glm_scores"][m]) + .0008, i - .19, f"{np.median(R['glm_scores'][m]):.3f}",
            va="center", fontsize=8)
ax.set(yticks=y, xlabel="median held-out Poisson deviance explained",
       title="Poisson GLM encoding models (NeMoS): the position model only rivals the velocity\n"
             "model in the task where position and velocity are confounded")
ax.set_yticklabels(mod_lbl, fontsize=9)
ax.set_xlim(0, 0.048)
ax.legend(fontsize=8.5, loc="lower right")

ax = fig.add_subplot(gs[2, 3])
Y, pred, g = R["Y_dec"], R["pred"], R["g_dec"]
sel = np.unique(g)[30:34]
mask = np.isin(g, sel)
tt = np.arange(mask.sum()) * BIN
ax.plot(tt, Y[mask, 0], "k", lw=1.5, label="actual $v_x$")
ax.plot(tt, pred[mask, 0], "seagreen", lw=1.3, label="decoded $v_x$")
ax.plot(tt, Y[mask, 1] - 900, "k", lw=1.5)
ax.plot(tt, pred[mask, 1] - 900, "tab:blue", lw=1.3, label="decoded $v_y$")
ax.text(0.3, 300, "$v_x$", fontsize=10)
ax.text(0.3, -650, "$v_y$", fontsize=10)
ax.set(xlabel="time (s)", ylabel="velocity (mm/s), $v_y$ offset",
       title="MC_RTT velocity decoding\n$R^2$=%.2f / %.2f, median dir. error %.0f$\\degree$"
             % (R["r2_vel"][0], R["r2_vel"][1], np.median(R["ang_err"][R["fast"]])))
ax.legend(fontsize=7.5, ncol=1, loc="upper right")

fig.suptitle("Replication on a second dataset: MC_RTT (DANDI:000129, monkey Indy, self-paced "
             "random-target reaching)", y=0.975, fontsize=13)
plt.savefig("fig06_replication_mc_rtt.png", dpi=135, bbox_inches="tight")
plt.close(fig)
print("wrote fig06_replication_mc_rtt.png")
