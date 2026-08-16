"""Cross-validated Bayesian decoding of position from the CA1 population.

If the place fields are real, a naive Bayes decoder built from HALF the laps
should recover the animal's position on the other half. Tuning curves come from
odd laps; position is decoded on even laps and compared with the tracked
position. A cell-identity shuffle gives the chance level.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pynapple as nap
import pf_lib

N_BINS = 40
SMOOTH = 1.0
BIN_SIZE = 0.2   # decoding time bin, seconds
DIRS = ["rightward", "leftward"]

nwbfile = pf_lib.open_session(pf_lib.PROTOTYPE)
pos2d, lin, fs, maze = pf_lib.load_behavior(nwbfile)
epochs = pf_lib.load_epochs(nwbfile)
lin = lin.restrict(epochs["MazeEpoch"])
lin_valid, vel, run_eps = pf_lib.make_run_epochs(lin, fs)
units = pf_lib.load_units(nwbfile)
exc = units[np.where(units.cell_type == "excitatory")[0]]
rng_track = pf_lib.track_range(lin_valid)

saved = np.load("results_single_session.npz")


def build_tuning(group, ep):
    """Smoothed tuning curves from pynapple, ready for decode_bayes."""
    tc = nap.compute_tuning_curves(group, lin_valid, bins=N_BINS,
                                   range=rng_track, epochs=ep,
                                   feature_names=["position"])
    v = np.asarray(tc.values, dtype=float)
    v = gaussian_filter1d(np.nan_to_num(v, nan=0.0), SMOOTH, axis=-1,
                          mode="nearest")
    # a hard zero would veto a position bin outright, so floor the rates
    tc.values[...] = np.maximum(v, 1e-3)
    return tc


out = {}
for d in DIRS:
    ep = run_eps[d]
    starts, ends = np.asarray(ep.start), np.asarray(ep.end)
    odd_ep = nap.IntervalSet(start=starts[0::2], end=ends[0::2])
    even_ep = nap.IntervalSet(start=starts[1::2], end=ends[1::2])

    # decode with the cells that are place cells in this direction
    sel = np.where(saved[f"{d}_is_place"])[0]
    unit_ids = np.array(list(exc.keys()))[sel]
    group = exc[list(unit_ids)]

    tc = build_tuning(group, odd_ep)                    # fitted on odd laps
    decoded, proba = nap.decode_bayes(tc, group, even_ep, BIN_SIZE)

    true_pos = lin_valid.interpolate(decoded)
    err = np.abs(decoded.values - true_pos.values)
    finite = np.isfinite(err)

    # chance level: permute which tuning curve belongs to which cell
    rng = np.random.default_rng(0)
    chance = []
    for _ in range(20):
        tc_s = tc.copy()
        tc_s.values[...] = tc.values[rng.permutation(len(unit_ids))]
        dec_s, _ = nap.decode_bayes(tc_s, group, even_ep, BIN_SIZE)
        e = np.abs(dec_s.values - lin_valid.interpolate(dec_s).values)
        chance.append(np.nanmedian(e[np.isfinite(e)]))

    # which held-out lap each decoded bin belongs to, for the strip plot
    lap_of_bin = np.searchsorted(even_ep.start, decoded.index.values,
                                 side="right") - 1

    r2 = np.corrcoef(decoded.values[finite], true_pos.values[finite])[0, 1] ** 2
    out[d] = dict(decoded=decoded, proba=proba, true_pos=true_pos,
                  err=err, finite=finite, chance=np.array(chance),
                  n_cells=len(sel), lap_of_bin=lap_of_bin, r2=r2,
                  centers=np.asarray(tc.coords["position"].values))
    print(f"[{d}] decoded {finite.sum()} time bins with {len(sel)} place cells")
    print(f"  median error {np.median(err[finite]):.3f} m  "
          f"(chance {np.mean(chance):.3f} +/- {np.std(chance):.3f} m)")
    print(f"  R^2 = {r2:.3f}")

# ---------------------------------------------------------------------------
# Figure 7
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 1, 1], hspace=0.42, wspace=0.28)

N_LAPS_SHOW = 10
for row, d in enumerate(DIRS):
    O = out[d]
    cen = O["centers"]

    # --- posterior strip: consecutive decoded bins, lap boundaries marked ---
    ax = fig.add_subplot(gs[row, 0])
    m = O["lap_of_bin"] < N_LAPS_SHOW
    P = np.asarray(O["proba"].values)[m]
    ax.imshow(P.T, aspect="auto", origin="lower", cmap="Greys",
              extent=[0, m.sum(), cen[0], cen[-1]])
    xs = np.arange(m.sum()) + 0.5
    ax.plot(xs, O["true_pos"].values[m], "-", lw=2.5, color="tab:green",
            alpha=0.85, label="tracked position")
    ax.plot(xs, O["decoded"].values[m], ".", ms=7, color="tab:red",
            label="decoded position")
    for b in np.where(np.diff(O["lap_of_bin"][m]) > 0)[0]:
        ax.axvline(b + 1, color="tab:blue", lw=1, ls=":")
    ax.set(xlabel=f"consecutive {BIN_SIZE*1000:.0f} ms decoding bins "
                  "(dotted lines separate held-out laps)",
           ylabel="position on track (m)", xlim=(0, m.sum()))
    ax.set_title(f"{d}: posterior over position, {N_LAPS_SHOW} held-out laps "
                 f"({O['n_cells']} place cells)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

    # --- decoded vs true ---
    ax = fig.add_subplot(gs[row, 1])
    ax.scatter(O["true_pos"].values, O["decoded"].values, s=9, alpha=0.35,
               color="tab:red" if d == "rightward" else "tab:blue")
    ax.plot(rng_track, rng_track, "k--", lw=1)
    ax.set(xlabel="true position (m)", ylabel="decoded position (m)",
           xlim=rng_track, ylim=rng_track)
    ax.set_title(f"{d}: decoded vs true\n$R^2$ = {O['r2']:.2f}", fontsize=11)
    ax.set_aspect("equal")

    # --- error distribution ---
    ax = fig.add_subplot(gs[row, 2])
    e = O["err"][O["finite"]]
    ax.hist(e, bins=np.arange(0, 1.65, 0.05), color="0.45")
    ax.axvline(np.median(e), color="tab:red", lw=2,
               label=f"median {np.median(e):.2f} m")
    ax.axvline(O["chance"].mean(), color="k", ls="--", lw=2,
               label=f"cell-shuffled {O['chance'].mean():.2f} m")
    ax.set(xlabel="|decoded - true| (m)", ylabel="time bins")
    ax.set_title(f"{d}: error distribution", fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("Position is decodable from CA1 place cells on held-out laps "
             "(tuning curves fitted on odd laps, decoding on even laps)",
             fontsize=13)
fig.savefig("figures/fig07_decoding.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figures/fig07_decoding.png")

np.savez("results_decoding.npz",
         **{f"{d}_{k}": np.asarray(out[d][k])
            for d in DIRS for k in ("err", "chance", "r2")})
