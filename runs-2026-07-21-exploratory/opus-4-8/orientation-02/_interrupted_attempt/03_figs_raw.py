"""Raw-data and example-unit figures for the prototype session."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import oslib

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

SES = "715093703"
DIR_CMAP = plt.get_cmap("hsv")
AREA_COLORS = {"VISp": "#1f4e9c", "LGd": "#d1671a"}


def dir_color(d):
    return DIR_CMAP((d % 360) / 360.0)


d = np.load(f"extracted/{SES}.npz", allow_pickle=True)
area = d["area"]
dg = oslib.analyze_gratings(d["dg_rates"], d["dg_ori"], d["dg_tf"], n_perm=500)
np.save(f"extracted/{SES}_dg_tuning.npy", dg, allow_pickle=True)

nwbfile = oslib.open_session(SES)
tsg, meta = oslib.good_units(nwbfile)
unit_ids = np.asarray(list(tsg.keys()))
assert np.array_equal(unit_ids, d["unit_ids"])

# ---------------------------------------------------------------- figure 1
# A stretch of drifting-grating trials clear of every invalid-data interval.
inv = oslib.invalid_trial_mask(
    nwbfile, meta["probe_id"].values, d["dg_start"], d["dg_stop"]
).any(axis=0)
ok = np.flatnonzero(~inv & (d["dg_start"] > d["dg_start"][0] + 400))
i0 = ok[np.flatnonzero(np.diff(ok) == 1)[0]]
t0, t1 = d["dg_start"][i0], d["dg_start"][i0 + 10]
win = nap.IntervalSet(start=t0 - 2, end=t1 + 2)

# Sort units within each area by preferred orientation so the stimulus-locked
# structure is visible in the raw raster.
order = np.concatenate(
    [
        np.flatnonzero(area == a)[np.argsort(dg["pref_ori"][area == a])]
        for a in ("VISp", "LGd")
    ]
)
fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 4, 1.2, 1.2], hspace=0.32)

ax_stim = fig.add_subplot(gs[0])
for s, e, o in zip(d["dg_start"], d["dg_stop"], d["dg_ori"]):
    if e < t0 - 2 or s > t1 + 2:
        continue
    if np.isnan(o):
        ax_stim.axvspan(s, e, color="0.8")
        ax_stim.text((s + e) / 2, 0.5, "blank", ha="center", va="center", fontsize=7)
    else:
        ax_stim.axvspan(s, e, color=dir_color(o))
        ax_stim.text((s + e) / 2, 0.5, f"{int(o)}", ha="center", va="center", fontsize=7)
ax_stim.set_xlim(t0 - 2, t1 + 2)
ax_stim.set_yticks([])
ax_stim.tick_params(bottom=False, labelbottom=False)
ax_stim.set_ylabel("drift\ndirection", rotation=0, ha="right", va="center")
ax_stim.set_title(
    f"Session {SES}: drifting gratings, quality-filtered single units (color = drift direction in degrees)",
    fontsize=11, pad=8,
)

ax_r = fig.add_subplot(gs[1], sharex=ax_stim)
for row, ui in enumerate(order):
    t = tsg[int(unit_ids[ui])].restrict(win).t
    ax_r.plot(t, np.full_like(t, row), "|", ms=3.4, color=AREA_COLORS[area[ui]], mew=0.6)
n_visp = (area == "VISp").sum()
ax_r.axhline(n_visp - 0.5, color="k", lw=0.8, ls="--")
for b in d["dg_start"]:
    if t0 - 2 <= b <= t1 + 2:
        ax_r.axvline(b, color="0.75", lw=0.5, zorder=0)
ax_r.set_ylim(-1, len(order))
ax_r.set_ylabel("unit (sorted by preferred orientation)")
for name, lo, hi in [("VISp", 0, n_visp), ("LGd", n_visp, len(order))]:
    ax_r.text(
        1.005, (lo + hi) / 2 / len(order), name, transform=ax_r.transAxes,
        color=AREA_COLORS[name], rotation=90, va="center", ha="left", fontweight="bold",
    )
ax_r.tick_params(labelbottom=False)

ax_p = fig.add_subplot(gs[2], sharex=ax_stim)
for name in ("VISp", "LGd"):
    sub = tsg[[int(u) for u in unit_ids[area == name]]]
    rate = sub.count(0.02, ep=win).sum(axis=1) / 0.02 / (area == name).sum()
    ax_p.plot(rate.t, nap.Tsd(t=rate.t, d=np.asarray(rate.values)).smooth(0.05).values,
              color=AREA_COLORS[name], lw=1.1, label=name)
for b in d["dg_start"]:
    if t0 - 2 <= b <= t1 + 2:
        ax_p.axvline(b, color="0.75", lw=0.5, zorder=0)
ax_p.set_ylabel("population\nrate (Hz)")
ax_p.legend(frameon=False, ncol=2, loc="upper right")
ax_p.tick_params(labelbottom=False)

ax_s = fig.add_subplot(gs[3], sharex=ax_stim)
run = nwbfile.processing["running"]["running_speed"]
speed = nap.Tsd(t=np.asarray(run.timestamps[:]), d=np.asarray(run.data[:])).restrict(win)
ax_s.plot(speed.t, speed.values, color="0.35", lw=0.9)
ax_s.set_ylabel("running\nspeed (cm/s)")
ax_s.set_xlabel("time in session (s)")
ax_s.set_xlim(t0 - 2, t1 + 2)
fig.savefig("fig01_raw_activity.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("fig01 done")

# ---------------------------------------------------------------- figure 2
visp = np.flatnonzero(area == "VISp")
score = np.where(dg["tuned"][visp] & (dg["tc"][visp].max(axis=1) > 5), dg["gosi"][visp], -1)
ex = visp[np.argmax(score)]
ex_id = int(unit_ids[ex])
tf = dg["pref_sub"][ex]
print(f"example unit {ex_id}  gOSI={dg['gosi'][ex]:.2f}  DSI={dg['dsi'][ex]:.2f}  TF={tf} Hz")

dirs = dg["levels"]
fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 0.35, 1.35],
                      hspace=0.45, wspace=0.28)
peri_axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
bins = np.arange(-0.5, 2.5, 0.05)

sel = (d["dg_tf"] == tf) & ~np.isnan(d["dg_ori"])
psth_max = 0.0
for dd in dirs:  # common PSTH scale across directions
    starts = d["dg_start"][sel & (d["dg_ori"] == dd)]
    pe = nap.compute_perievent(tsg[ex_id], nap.Ts(t=starts), window=(-0.5, 2.5))
    t = np.concatenate([pe[k].t for k in pe.keys()])
    psth_max = max(psth_max, np.histogram(t, bins=bins)[0].max() / len(starts) / 0.05)

for k, dd in enumerate(dirs):
    ax = peri_axes[k]
    starts = d["dg_start"][sel & (d["dg_ori"] == dd)]
    pe = nap.compute_perievent(tsg[ex_id], nap.Ts(t=starts), window=(-0.5, 2.5))
    all_t = []
    for j, key in enumerate(pe.keys()):
        t = pe[key].t
        all_t.append(t)
        ax.plot(t, np.full_like(t, j), "|", ms=4, color=dir_color(dd), mew=0.8)
    all_t = np.concatenate(all_t)
    h, _ = np.histogram(all_t, bins=bins)
    psth = h / len(starts) / 0.05
    ax2 = ax.twinx()
    ax2.plot(bins[:-1] + 0.025, psth, color="k", lw=1.0)
    ax2.set_ylim(0, psth_max * 1.1)
    ax2.spines["top"].set_visible(False)
    if k % 4 != 3:
        ax2.tick_params(right=False, labelright=False)
    ax.axvspan(0, 2, color=dir_color(dd), alpha=0.12, zorder=0)
    ax.set_ylim(-1, len(starts))
    ax.set_title(f"{int(dd)}$\\degree$", fontsize=10, pad=4)
    if k % 4 == 0:
        ax.set_ylabel("trial")
    if k >= 4:
        ax.set_xlabel("time from onset (s)")
    if k % 4 == 3:
        ax2.set_ylabel("firing rate (Hz)")

axp = fig.add_subplot(gs[:, 5], projection="polar")
th = np.deg2rad(np.append(dirs, dirs[0]))
r = np.append(dg["tc"][ex], dg["tc"][ex][0])
e = np.append(dg["sem"][ex], dg["sem"][ex][0])
axp.plot(th, r, "-o", color=AREA_COLORS["VISp"], ms=4)
axp.fill_between(th, r - e, r + e, color=AREA_COLORS["VISp"], alpha=0.25)
axp.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, dg["baseline"][ex]),
         ls="--", color="0.5", lw=1, label="blank sweep")
axp.set_theta_zero_location("E")
axp.set_title(
    f"unit {ex_id} (VISp)\ngOSI = {dg['gosi'][ex]:.2f}, DSI = {dg['dsi'][ex]:.2f}, TF = {tf:g} Hz",
    fontsize=10, pad=22,
)
axp.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30))
fig.suptitle(
    "Example V1 unit: trial rasters and PSTHs for the 8 drift directions, and the resulting polar tuning curve",
    fontsize=11, y=0.98,
)
fig.savefig("fig02_example_unit.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("fig02 done")
np.save("extracted/example_unit.npy", np.array([ex_id, tf]))
