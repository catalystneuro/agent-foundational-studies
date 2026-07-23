"""Stage 2: trial-based reach-direction tuning during movement and during the delay."""

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import reachlib as rl

d = rl.load_mc_maze()
units, hand_pos, hand_vel, trials = d["units"], d["hand_pos"], d["hand_vel"], d["trials"]
areas = d["areas"]
spikes = [units[i].t for i in units.keys()]

# Barrier-free trials produce straight reaches, so hand displacement after
# movement onset is an unambiguous reach direction.
straight = trials[(trials.num_barriers == 0) & (trials.success == 1)].copy()
onset = straight.move_onset_time.values
go = straight.go_cue_time.values
target_on = straight.target_on_time.values

t_hp = hand_pos.t
p0 = np.column_stack([np.interp(onset, t_hp, hand_pos.values[:, k]) for k in range(2)])
p1 = np.column_stack([np.interp(onset + 0.4, t_hp, hand_pos.values[:, k]) for k in range(2)])
disp = p1 - p0
theta = np.arctan2(disp[:, 1], disp[:, 0])
amp = np.hypot(*disp.T)
print("n straight trials:", len(straight), " median reach amplitude %.0f mm" % np.median(amp))

# Peak speed per trial, used later as the movement-vigour regressor.
speed_t = np.hypot(*hand_vel.values.T)
offsets = np.arange(0.0, 0.401, 0.005)
sp_peri = rl.peri_event_matrix(hand_vel.t, speed_t, onset, offsets)
peak_speed = sp_peri.max(1)
print("peak speed: median %.0f mm/s, IQR %.0f-%.0f"
      % (np.median(peak_speed), *np.percentile(peak_speed, [25, 75])))

# --- firing rates in the movement and delay epochs ------------------------
mov_rate = rl.window_rates(spikes, onset - 0.05, onset + 0.35)
long_delay = (go - target_on) > 0.4
del_rate = rl.window_rates(spikes, go - 0.35, go - 0.05)
print("trials with >400 ms delay:", int(long_delay.sum()))

mov = rl.fit_cosine(theta, mov_rate)
dly = rl.fit_cosine(theta[long_delay], del_rate[long_delay])
mov_depth, mov_p = rl.cosine_permutation_test(theta, mov_rate, n_perm=1000)
dly_depth, dly_p = rl.cosine_permutation_test(theta[long_delay], del_rate[long_delay], n_perm=1000)

sig_mov = mov_p < 0.01
sig_dly = dly_p < 0.01

# A significant modulation depth does not guarantee a usable preferred
# direction: on low-rate units the PD estimate is dominated by noise. Split the
# trials in half and keep the units whose two PD estimates agree within 45 deg.
rng = np.random.default_rng(0)
half = rng.permutation(len(theta)) < len(theta) // 2
pd_a = rl.fit_cosine(theta[half], mov_rate[half])["pd"]
pd_b = rl.fit_cosine(theta[~half], mov_rate[~half])["pd"]
reliable_mov = sig_mov & (np.abs(rl.circ_diff(pd_a, pd_b)) < np.pi / 4)
idx_d = np.flatnonzero(long_delay)
half_d = np.isin(np.arange(len(theta)), idx_d[rng.permutation(idx_d.size) < idx_d.size // 2])
pd_da = rl.fit_cosine(theta[half_d], del_rate[half_d])["pd"]
pd_db = rl.fit_cosine(theta[long_delay & ~half_d], del_rate[long_delay & ~half_d])["pd"]
reliable_dly = sig_dly & (np.abs(rl.circ_diff(pd_da, pd_db)) < np.pi / 4)
print("directionally tuned during movement: %d/%d (%.0f%%)"
      % (sig_mov.sum(), len(sig_mov), 100 * sig_mov.mean()))
print("directionally tuned during delay:    %d/%d (%.0f%%)"
      % (sig_dly.sum(), len(sig_dly), 100 * sig_dly.mean()))
print("median cosine R2 (tuned units): %.2f" % np.median(mov["r2"][sig_mov]))
print("PD reliable by split-half: movement %d, delay %d"
      % (reliable_mov.sum(), reliable_dly.sum()))

np.savez("cache/direction_tuning.npz", theta=theta, onset=onset, amp=amp,
         peak_speed=peak_speed, mov_rate=mov_rate, del_rate=del_rate,
         long_delay=long_delay, pd_mov=mov["pd"], pd_dly=dly["pd"],
         depth_mov=mov["depth"], depth_dly=dly["depth"], r2_mov=mov["r2"],
         p_mov=mov_p, p_dly=dly_p, areas=areas,
         reliable_mov=reliable_mov, reliable_dly=reliable_dly)

# --- Figure 3: example units, rasters + polar tuning ----------------------
order = np.argsort(-mov["depth"] * sig_mov)
examples = order[[0, 1, 2, 4]]
edges = np.linspace(-np.pi, np.pi, 9)
bin_idx = np.digitize(theta, edges) - 1
bin_centres = (edges[:-1] + edges[1:]) / 2
n_per_bin = np.array([(bin_idx == b).sum() for b in range(8)])
print("trials per 45-deg direction bin:", n_per_bin)
# The targets are not spaced exactly 45 deg apart, so some bins are thinly
# populated; those are excluded from the per-direction averages.
use_bin = np.flatnonzero(n_per_bin >= 30)

fig, axes = plt.subplots(3, len(examples), figsize=(4.2 * len(examples), 10.5),
                         gridspec_kw=dict(height_ratios=[1.4, 1, 1.3]))
psth_edges = np.arange(-0.4, 0.601, 0.01)
psth_t = psth_edges[:-1] + 0.005
kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
kernel /= kernel.sum()
cmap = plt.get_cmap("hsv")


def dir_colour(b):
    return cmap((bin_centres[b] + np.pi) / (2 * np.pi))


for col, u in enumerate(examples):
    st = spikes[u]
    ax = axes[0, col]
    row = 0
    for b in use_bin:
        tr_idx = np.flatnonzero(bin_idx == b)[:20]
        for i in tr_idx:
            lo, hi = np.searchsorted(st, [onset[i] - 0.4, onset[i] + 0.6])
            s = st[lo:hi] - onset[i]
            ax.plot(s, np.full_like(s, row), "|", ms=3.5, mew=0.8, color=dir_colour(b))
            row += 1
        ax.axhline(row - 0.5, color="0.8", lw=0.5)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlim(-0.4, 0.6)
    ax.set_ylim(-1, row)
    ax.set_title("unit %d\ndepth %.1f Hz, p = %.3f" % (u, mov["depth"][u], mov_p[u]),
                 fontsize=10, pad=6)
    if col == 0:
        ax.set_ylabel("trials, blocked by reach direction")

    ax = axes[1, col]
    for b in use_bin:
        tr_idx = np.flatnonzero(bin_idx == b)
        c = np.zeros(psth_t.size)
        for i in tr_idx:
            lo, hi = np.searchsorted(st, [onset[i] + psth_edges[0], onset[i] + psth_edges[-1]])
            c += np.histogram(st[lo:hi] - onset[i], bins=psth_edges)[0]
        rate = np.convolve(c / (len(tr_idx) * 0.01), kernel, mode="same")
        ax.plot(psth_t, rate, lw=1.4, color=dir_colour(b))
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("time from movement onset (s)")
    if col == 0:
        ax.set_ylabel("firing rate (Hz)")

    ax = fig.add_subplot(3, len(examples), 2 * len(examples) + col + 1, projection="polar")
    axes[2, col].axis("off")
    m = np.array([mov_rate[bin_idx == b, u].mean() for b in use_bin])
    se = np.array([mov_rate[bin_idx == b, u].std() / np.sqrt(n_per_bin[b]) for b in use_bin])
    th = bin_centres[use_bin]
    # Break the connecting line where a thinly sampled direction bin was dropped.
    gap = np.diff(np.append(use_bin, use_bin[0] + 8)) > 1
    th_p, m_p, se_p = [], [], []
    for i in range(len(th)):
        th_p += [th[i]]
        m_p += [m[i]]
        se_p += [se[i]]
        if gap[i]:
            nxt = th[(i + 1) % len(th)] + (2 * np.pi if i == len(th) - 1 else 0)
            th_p += [(th[i] + nxt) / 2]
            m_p += [np.nan]
            se_p += [0.0]
    th_p += [th[0] + 2 * np.pi]
    m_p += [m[0] if not gap[-1] else np.nan]
    se_p += [se[0]]
    ax.errorbar(th_p, m_p, yerr=se_p, color="k", marker="o", ms=4, lw=1.2, zorder=3)
    fine = np.linspace(-np.pi, np.pi, 200)
    fit = mov["baseline"][u] + mov["depth"][u] * np.cos(rl.circ_diff(fine, mov["pd"][u]))
    ax.plot(fine, np.clip(fit, 0, None), color="crimson", lw=1.6)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0°", "", "180°", ""])
    rmax = float(np.nanmax(m + se))  # noqa: F841
    ax.set_rticks(np.round(np.linspace(rmax / 3, rmax, 3), 0))
    ax.set_rlabel_position(45)
    ax.tick_params(labelsize=7)
    ax.set_title("preferred direction %.0f°" % np.degrees(mov["pd"][u]), fontsize=10, pad=24)
    if col == 0:
        ax.text(-0.28, 0.5, "mean rate in the\nmovement window (Hz)", transform=ax.transAxes,
                rotation=90, va="center", ha="center", fontsize=9)

fig.suptitle("Reach-direction tuning of single motor-cortical units "
             "(MC_Maze, monkey Jenkins)\ncolour = reach direction; red curve = cosine fit",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig("fig03_example_direction_tuning.png", dpi=150)
plt.close(fig)

# --- Figure 4: population summary ----------------------------------------
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0], projection="polar")
h, e = np.histogram(mov["pd"][reliable_mov], bins=np.linspace(-np.pi, np.pi, 17))
ax.bar(e[:-1], h, width=np.diff(e), align="edge", alpha=0.8, color="tab:blue")
ax.set_title("preferred directions\n(n=%d units with reliable PD)" % reliable_mov.sum(),
             fontsize=10, pad=20)

ax = fig.add_subplot(gs[0, 1])
dbins = np.histogram_bin_edges(mov["depth"], bins=22)
ax.hist([mov["depth"][sig_mov], mov["depth"][~sig_mov]], bins=dbins, stacked=True,
        color=["tab:blue", "0.75"], label=["tuned (p<0.01)", "not tuned"])
ax.set_xlabel("cosine modulation depth (Hz)")
ax.set_ylabel("units")
ax.set_title("strength of directional tuning", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[0, 2])
ax.hist(mov["r2"], bins=20, color="0.4")
ax.axvline(np.median(mov["r2"]), color="crimson", ls="--")
ax.set_xlabel("cosine-fit $R^2$ (single trials)")
ax.set_ylabel("units")
ax.set_title("variance explained by a cosine", fontsize=10)

ax = fig.add_subplot(gs[1, 0])
both = reliable_mov & reliable_dly
ax.scatter(np.degrees(dly["pd"][both]), np.degrees(mov["pd"][both]), s=18, color="tab:blue")
ax.plot([-180, 180], [-180, 180], "k--", lw=0.8)
ax.set_xlabel("preferred direction, delay (deg)")
ax.set_ylabel("preferred direction, movement (deg)")
dpd_all = rl.circ_diff(mov["pd"][both], dly["pd"][both])
med_shift = np.degrees(np.median(np.abs(dpd_all)))
frac45 = np.mean(np.abs(dpd_all) < np.pi / 4)
ax.set_title("preparatory vs movement PD\n(n=%d reliable in both, median |ΔPD| = %.0f°)"
             % (both.sum(), med_shift), fontsize=10)

ax = fig.add_subplot(gs[1, 1])
dpd = np.degrees(rl.circ_diff(mov["pd"][both], dly["pd"][both]))
ax.hist(dpd, bins=np.arange(-180, 181, 20), color="0.4")
ax.set_xlabel("PD(movement) - PD(delay) (deg)")
ax.set_ylabel("units")
ax.axvline(0, color="crimson", ls="--")
ax.set_title("PD shift between epochs\n%.0f%% within 45° (chance 50%%)" % (100 * frac45),
             fontsize=10)

ax = fig.add_subplot(gs[1, 2])
ax.scatter(dly["depth"], mov["depth"], s=16, color="tab:blue")
ax.set_xscale("log")
ax.set_yscale("log")
lim = max(mov["depth"].max(), dly["depth"].max()) * 1.2
ax.plot([0.02, lim], [0.02, lim], "k--", lw=0.8)
ax.set_xlabel("depth during delay (Hz)")
ax.set_ylabel("depth during movement (Hz)")
ax.set_title("directional signal is present\nbefore movement begins", fontsize=10)

fig.suptitle("Population summary of reach-direction tuning (%d motor-cortical units)" % len(units),
             fontsize=13)
fig.savefig("fig04_population_direction_tuning.png", dpi=150, bbox_inches="tight")
plt.close(fig)

circ_r = np.abs(np.exp(1j * mov["pd"][reliable_mov]).mean())
print("PD circular concentration (tuned units): r = %.3f" % circ_r)
print("delay vs movement PD: median |shift| %.0f deg (chance 90), %.0f%% within 45 deg "
      "(chance 50%%), n=%d" % (med_shift, 100 * frac45, both.sum()))
