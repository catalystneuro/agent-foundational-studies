"""Stage 4: direction-specific place fields on the linear track.

These rate maps are the template the replay decoder inverts in stage 5, so they
are built only from running periods (speed > 5 cm/s) in the MAZE epoch.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter1d

import swr_utils as su

SESSION = "Achilles-10252013"
NBINS = 50
SPEED_MIN = 5.0  # cm/s
SMOOTH_BINS = 1.5
MIN_PEAK_RATE = 1.0  # Hz
MIN_SPATIAL_INFO = 0.3  # bits/spike


def spatial_information(tc, occupancy):
    """Skaggs information rate in bits per spike."""
    p = occupancy / occupancy.sum()
    r = np.asarray(tc)
    rbar = (p[:, None] * r).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p[:, None] * (r / rbar) * np.log2(r / rbar)
    return np.nansum(terms, axis=0)


def running_epochs(pos):
    """Split the track traversals into rightward and leftward runs."""
    speed = su.compute_speed(pos)  # |v|
    t, d = np.asarray(pos.index), np.asarray(pos.values, float)
    good = np.isfinite(d)
    grid = np.asarray(speed.index)
    interp = np.interp(grid, t[good], d[good])
    from scipy.ndimage import gaussian_filter1d as g1
    dt = float(np.median(np.diff(grid)))
    vel = np.gradient(g1(interp, 0.25 / dt), dt)
    run = speed.threshold(SPEED_MIN, "above").time_support.drop_short_intervals(0.5)
    v = nap.Tsd(t=grid, d=vel)
    right = v.restrict(run).threshold(0, "above").time_support.drop_short_intervals(0.3)
    left = v.restrict(run).threshold(0, "below").time_support.drop_short_intervals(0.3)
    return run, right, left


def build_templates(pyr, pos, epochs_dict):
    """Direction-specific rate maps (bins x cells), smoothed."""
    tcs, occs = {}, {}
    for name, ep in epochs_dict.items():
        tc = nap.compute_1d_tuning_curves(pyr, pos, nb_bins=NBINS, ep=ep,
                                          minmax=(0, 160))
        occ, edges = np.histogram(pos.restrict(ep).values, bins=NBINS, range=(0, 160))
        vals = gaussian_filter1d(np.asarray(tc, dtype=float), SMOOTH_BINS, axis=0, mode="nearest")
        tcs[name] = np.clip(vals, 1e-3, None)
        occs[name] = occ.astype(float)
    return tcs, occs, edges


if __name__ == "__main__":
    h5 = su.open_session(SESSION)
    ep = su.load_epochs(h5)
    units = su.load_units(h5)
    pos = su.load_position(h5)

    pos = pos[np.isfinite(pos.values)].restrict(ep["MAZE"])
    pyr = units.getby_category("cell_type")["excitatory"]

    run, right, left = running_epochs(pos)
    print("run %.0f s  ->  rightward %.0f s (%d laps), leftward %.0f s (%d laps)"
          % (run.tot_length(), right.tot_length(), len(right), left.tot_length(), len(left)))

    tcs, occs, edges = build_templates(pyr, pos, {"right": right, "left": left})
    centers = 0.5 * (edges[1:] + edges[:-1])

    si = {k: spatial_information(tcs[k], occs[k]) for k in tcs}
    peak = {k: tcs[k].max(0) for k in tcs}
    is_place = np.zeros(len(pyr), dtype=bool)
    for k in tcs:
        is_place |= (peak[k] >= MIN_PEAK_RATE) & (si[k] >= MIN_SPATIAL_INFO)
    print("place cells: %d of %d pyramidal cells" % (is_place.sum(), len(pyr)))
    print("median spatial info (place cells): %.2f bits/spike"
          % np.median(np.maximum(si["right"], si["left"])[is_place]))

    np.savez("place_fields.npz", right=tcs["right"], left=tcs["left"],
             centers=centers, is_place=is_place, si_right=si["right"], si_left=si["left"],
             unit_ids=np.array(list(pyr.keys())),
             run_start=run.start, run_end=run.end,
             right_start=right.start, right_end=right.end,
             left_start=left.start, left_end=left.end)

    # ------------------------------------------------------------------ figures
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35, height_ratios=[1.25, 1])

    idx = np.where(is_place)[0]
    for j, k in enumerate(["right", "left"]):
        ax = fig.add_subplot(gs[0, j])
        order = idx[np.argsort(np.argmax(tcs[k][:, idx], axis=0))]
        m = tcs[k][:, order].T
        m = m / m.max(1, keepdims=True)
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, 160, 0, len(order)])
        ax.set(xlabel="position on track (cm)", ylabel="place cell (sorted by own peak)",
               title="%sward runs" % k.capitalize())
        fig.colorbar(im, ax=ax, label="normalized rate")

    ax = fig.add_subplot(gs[0, 2])
    pk_all = centers[np.argmax(tcs["right"], axis=0)]
    mid_field = idx[(pk_all[idx] > 20) & (pk_all[idx] < 140)]
    best = mid_field[np.argsort(-si["right"][mid_field])][:6]
    for c in best:
        ax.plot(centers, tcs["right"][:, c], lw=1.4)
    ax.set(xlabel="position (cm)", ylabel="firing rate (Hz)",
           title="Six example place fields\n(rightward runs)")

    ax = fig.add_subplot(gs[1, 0])
    p = pos.restrict(nap.IntervalSet(pos.index[0], pos.index[0] + 300))
    ax.plot(p.index - p.index[0], p.values, color="0.5", lw=0.8)
    for s, e in zip(right.start, right.end):
        seg = pos.restrict(nap.IntervalSet(s, e))
        ax.plot(seg.index - p.index[0], seg.values, color="C0", lw=1.6)
    for s, e in zip(left.start, left.end):
        seg = pos.restrict(nap.IntervalSet(s, e))
        ax.plot(seg.index - p.index[0], seg.values, color="C3", lw=1.6)
    ax.set(xlim=(0, 300), xlabel="time on maze (s)", ylabel="position (cm)",
           title="Laps: rightward (blue) / leftward (red)")

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(np.maximum(si["right"], si["left"]), bins=30, color="0.7", label="all pyramidal")
    ax.hist(np.maximum(si["right"], si["left"])[is_place], bins=30, color="C0", label="place cells")
    ax.axvline(MIN_SPATIAL_INFO, color="k", ls=":")
    ax.legend(fontsize=8, frameon=False)
    ax.set(xlabel="spatial information (bits/spike)", ylabel="count", title="Spatial tuning")

    ax = fig.add_subplot(gs[1, 2])
    pk_r = centers[np.argmax(tcs["right"][:, is_place], axis=0)]
    pk_l = centers[np.argmax(tcs["left"][:, is_place], axis=0)]
    ax.plot(pk_r, pk_l, "o", ms=4, alpha=0.6)
    ax.plot([0, 160], [0, 160], "k:", lw=1)
    ax.set(xlabel="peak position, rightward (cm)", ylabel="peak position, leftward (cm)",
           title="Fields are direction-specific\n(r=%.2f)" % np.corrcoef(pk_r, pk_l)[0, 1])

    fig.suptitle("Place fields on the 1.6 m linear track, %s" % SESSION)
    fig.savefig("fig05_place_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig05")
