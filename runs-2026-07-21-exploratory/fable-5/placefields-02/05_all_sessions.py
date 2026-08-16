"""Run the place-field pipeline over all eight sessions of DANDI:000044.

Five sessions use a straight track (1.6 m or 2 m) and three use a closed
circular track (~2.85 m). They are treated differently, because on a straight
track the animal shuttles back and forth and fields are direction-specific,
whereas on the circular track it runs essentially one way and position is a
circular variable. Every excitatory unit is tested against its own
circular-shift null, and results are pooled so the conclusion does not rest on
one animal or one recording.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pf_lib

TARGET_BIN = 0.04     # metres per spatial bin, kept constant across sessions
N_SHUFFLES = 500
SMOOTH = 1.0
MIN_PEAK_RATE = 1.0
MIN_SPIKES = 50
ALPHA = 0.01

per_session = []
for asset, label in tqdm(pf_lib.SESSIONS.items(), total=len(pf_lib.SESSIONS),
                         desc="sessions"):
    nwbfile = pf_lib.open_session(asset)
    pf_lib.check_timebase(nwbfile)
    pos2d, lin, fs, maze = pf_lib.load_behavior(nwbfile)
    kind = pf_lib.maze_kind(maze)
    epochs = pf_lib.load_epochs(nwbfile)
    lin = lin.restrict(epochs["MazeEpoch"])
    rng_track = pf_lib.track_range(lin)
    length = rng_track[1] - rng_track[0]

    lin_valid, vel, run_eps = pf_lib.make_run_epochs(
        lin, fs, period=length if kind == "circular" else None)
    units = pf_lib.load_units(nwbfile)
    exc = units[np.where(units.cell_type == "excitatory")[0]]
    n_bins = int(round(length / TARGET_BIN))

    if kind == "linear":
        conds = ["rightward", "leftward"]
    else:
        # the animal circles one way; use only the dominant direction
        conds = [max(("rightward", "leftward"),
                     key=lambda k: run_eps[k].tot_length())]

    sess = dict(label=label, maze=maze, kind=kind, length=length,
                n_exc=len(exc), n_units=len(units), fs=fs, n_bins=n_bins,
                conds=conds, per_cond={}, maps={})
    for d in conds:
        ep = run_eps[d]
        rm = pf_lib.DirectionalRateMaps(lin_valid, ep, fs, n_bins, rng_track,
                                        smooth_bins=SMOOTH,
                                        circular=(kind == "circular"))
        maps = rm.rate_maps(exc)
        si = rm.si(maps)[0]
        null = rm.shuffle_si(exc, n_shuffles=N_SHUFFLES, seed=1, progress=False)
        pval = (np.sum(null >= si[None, :], axis=0) + 1) / (N_SHUFFLES + 1)
        peak_rate, peak_pos, width = pf_lib.field_metrics(
            maps, rm.centers, rm.bin_size, circular=(kind == "circular"))
        n_spk = np.array([len(exc[u].restrict(ep)) for u in exc.keys()])
        odd, even = rm.lap_split_maps(exc)
        ok = rm.valid_bins
        stab = np.array([
            np.corrcoef(odd[i, ok], even[i, ok])[0, 1]
            if np.isfinite(odd[i, ok]).all() and np.isfinite(even[i, ok]).all()
            and odd[i, ok].std() > 0 and even[i, ok].std() > 0 else np.nan
            for i in range(len(exc))])
        is_place = (pval < ALPHA) & (peak_rate >= MIN_PEAK_RATE) & (n_spk >= MIN_SPIKES)

        sess["per_cond"][d] = dict(
            n_laps=len(ep), dur=float(ep.tot_length()),
            n_place=int(is_place.sum()), frac=float(is_place.mean()),
            si=si[is_place], width=width[is_place], peak_rate=peak_rate[is_place],
            peak_pos=peak_pos[is_place] / length, stab=stab[is_place],
            null_median=float(np.median(null)))
        sess["maps"][d] = (maps, is_place, rm.valid_bins)

    anyplace = np.zeros(len(exc), bool)
    for d in conds:
        anyplace |= sess["maps"][d][1]
    sess["n_either"] = int(anyplace.sum())

    if kind == "linear":  # rightward vs leftward map correlation
        mr, pr_, okr = sess["maps"]["rightward"]
        ml, pl_, okl = sess["maps"]["leftward"]
        okb = okr & okl
        cc = [np.corrcoef(mr[i, okb], ml[i, okb])[0, 1]
              for i in np.where(pr_ | pl_)[0]
              if mr[i, okb].std() > 0 and ml[i, okb].std() > 0]
        sess["dir_corr"] = np.array(cc)
    else:
        sess["dir_corr"] = np.array([])
    per_session.append(sess)

# ---------------------------------------------------------------------------
print(f"\n{'session':<20}{'maze':<16}{'exc':>5}{'laps':>18}{'place cells':>22}")
for s in per_session:
    laps = "/".join(str(s["per_cond"][d]["n_laps"]) for d in s["conds"])
    pcs = "/".join(str(s["per_cond"][d]["n_place"]) for d in s["conds"])
    either = s["n_either"]
    maze_short = s["maze"].replace("Position", "")
    print(f"{s['label']:<20}{maze_short:<16}{s['n_exc']:>5}{laps:>18}"
          f"{pcs + '  (either ' + str(either) + ')':>22}")


def pool(key, kinds=("linear", "circular")):
    return np.concatenate([s["per_cond"][d][key] for s in per_session
                           if s["kind"] in kinds for d in s["conds"]])


tot_exc = sum(s["n_exc"] for s in per_session)
tot_either = sum(s["n_either"] for s in per_session)
all_si, all_w = pool("si"), pool("width")
all_pk, all_st, all_pos = pool("peak_rate"), pool("stab"), pool("peak_pos")
all_dc = np.concatenate([s["dir_corr"] for s in per_session])
fracs = np.array([s["per_cond"][d]["frac"] for s in per_session for d in s["conds"]])

print(f"\npooled across {len(per_session)} sessions "
      f"({sum(s['kind']=='linear' for s in per_session)} linear, "
      f"{sum(s['kind']=='circular' for s in per_session)} circular):")
print(f"  excitatory units: {tot_exc}; place cell in >=1 condition: {tot_either} "
      f"({100*tot_either/tot_exc:.0f}%)")
print(f"  per-condition fraction: {100*fracs.mean():.0f}% +/- {100*fracs.std():.0f}% (SD)")
print(f"  spatial information: median {np.median(all_si):.2f} bits/spike")
print(f"  field width: median {np.median(all_w):.2f} m")
print(f"  peak rate: median {np.median(all_pk):.1f} Hz")
print(f"  odd/even stability: median r = {np.nanmedian(all_st):.2f}")
print(f"  linear tracks, rightward vs leftward map correlation: "
      f"median r = {np.nanmedian(all_dc):.2f} (n={np.isfinite(all_dc).sum()})")
for k in ("linear", "circular"):
    si_k = pool("si", (k,))
    w_k = pool("width", (k,))
    print(f"  [{k:8s}] SI median {np.median(si_k):.2f} bits/spike, "
          f"width median {np.median(w_k):.2f} m, n={len(si_k)}")

np.savez("results_all_sessions.npz", all_si=all_si, all_w=all_w, all_pk=all_pk,
         all_st=all_st, all_pos=all_pos, all_dc=all_dc, fracs=fracs,
         labels=np.array([s["label"] for s in per_session]),
         kinds=np.array([s["kind"] for s in per_session]),
         n_exc=np.array([s["n_exc"] for s in per_session]),
         n_either=np.array([s["n_either"] for s in per_session]))

# ---------------------------------------------------------------------------
# Figure 8
# ---------------------------------------------------------------------------
labels = [f"{s['label']}\n({s['kind']})" for s in per_session]
x = np.arange(len(labels))
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

ax = axes[0, 0]
lin_mask = np.array([s["kind"] == "linear" for s in per_session])
for off, d, col in [(-0.2, "rightward", "tab:red"), (0.2, "leftward", "tab:blue")]:
    vals = [100 * s["per_cond"][d]["frac"]
            for s in per_session if s["kind"] == "linear"]
    ax.bar(x[lin_mask] + off, vals, 0.4, color=col, label=d)
# circular sessions have a single running direction, so one centred bar
circ = ~lin_mask
ax.bar(x[circ], [100 * s["per_cond"][s["conds"][0]]["frac"]
                 for s in per_session if s["kind"] == "circular"],
       0.5, color="tab:olive", label="circular (one direction)")
ax.axhline(100 * fracs.mean(), color="k", ls="--", lw=1,
           label=f"mean {100*fracs.mean():.0f}%")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
ax.set(ylabel="% of excitatory units",
       title="Place cells per session\n(circular sessions have one running direction)")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist([pool("si", ("linear",)), pool("si", ("circular",))], bins=35,
        stacked=True, color=["tab:green", "tab:olive"],
        label=["linear track", "circular track"])
ax.axvline(np.median(all_si), color="k", ls="--",
           label=f"median {np.median(all_si):.2f}")
ax.set(xlabel="spatial information (bits/spike)", ylabel="place cells",
       title=f"Spatial information, pooled (n={len(all_si)})")
ax.legend(fontsize=8)

ax = axes[0, 2]
ax.hist([pool("width", ("linear",)), pool("width", ("circular",))],
        bins=np.arange(0, 1.5, 0.05), stacked=True,
        color=["tab:orange", "peachpuff"], label=["linear", "circular"])
ax.axvline(np.median(all_w), color="k", ls="--",
           label=f"median {np.median(all_w):.2f} m")
ax.set(xlabel="field width at half maximum (m)", ylabel="place cells",
       title="Place field width")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.hist(all_pk, bins=np.arange(0, 40, 1.5), color="tab:purple")
ax.axvline(np.median(all_pk), color="k", ls="--",
           label=f"median {np.median(all_pk):.1f} Hz")
ax.set(xlabel="in-field peak rate (Hz)", ylabel="place cells",
       title="Peak firing rate")
ax.legend(fontsize=9)

ax = axes[1, 1]
ax.hist(all_pos, bins=np.linspace(0, 1, 21), color="tab:brown")
ax.set(xlabel="field peak, as a fraction of track length", ylabel="place cells",
       title="Fields tile the track")

ax = axes[1, 2]
ax.hist(all_dc[np.isfinite(all_dc)], bins=np.linspace(-1, 1, 41), color="tab:cyan")
ax.axvline(np.nanmedian(all_dc), color="k", ls="--",
           label=f"median r = {np.nanmedian(all_dc):.2f}")
ax.set(xlabel="correlation of rightward and leftward rate maps",
       ylabel="place cells",
       title="Direction selectivity (five linear-track sessions)")
ax.legend(fontsize=9)

fig.suptitle("Place fields across all eight sessions of DANDI:000044 "
             f"({tot_exc} excitatory units, {tot_either} place cells)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/fig08_all_sessions.png", dpi=150)
plt.close(fig)
print("saved figures/fig08_all_sessions.png")
