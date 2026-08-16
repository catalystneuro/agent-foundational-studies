"""Compute and visualize theta phase precession for CA1 place cells."""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from analyze import (load, build, place_fields, select_place_cells,
                     circ_lin_fit, N_POS_BINS)

MIN_FIELD_SPIKES = 30


def phase_at(spike_t, ph_t, ph_v):
    """Nearest-sample theta phase (rad) at spike times (avoids circular interp)."""
    idx = np.searchsorted(ph_t, spike_t)
    idx = np.clip(idx, 1, len(ph_t) - 1)
    left = ph_t[idx - 1]
    right = ph_t[idx]
    idx = np.where(np.abs(spike_t - left) <= np.abs(spike_t - right), idx - 1, idx)
    return ph_v[idx]


def collect(spikes, B, direction):
    """Return per-cell records with within-field position + theta phase per spike."""
    ep = B["run_r"] if direction == "right" else B["run_l"]
    centers, occ, tc, si = place_fields(spikes, B["cpos"], ep)
    pc = select_place_cells(centers, tc, si)
    ph_t = B["ph"].index.values
    ph_v = B["ph"].values
    cpos = B["cpos"]
    recs = {}
    for u, info in pc.items():
        s = spikes[u].restrict(ep)
        sp_t = s.index.values
        sp_pos = np.interp(sp_t, cpos.index.values, cpos.values)
        lo, hi = info["lo"], info["hi"]
        in_field = (sp_pos >= lo) & (sp_pos <= hi)
        if in_field.sum() < MIN_FIELD_SPIKES:
            continue
        xt = sp_t[in_field]
        # normalize as distance TRAVELLED through the field (0 = entry, 1 = exit),
        # accounting for movement direction (leftward runs traverse hi -> lo)
        if direction == "right":
            xnorm = (sp_pos[in_field] - lo) / (hi - lo)
        else:
            xnorm = (hi - sp_pos[in_field]) / (hi - lo)
        phi = phase_at(xt, ph_t, ph_v)              # -pi..pi
        a, phase0, rho, p = circ_lin_fit(xnorm, phi)
        recs[u] = dict(info=info, x=xnorm, phi=phi, tc=tc[u], centers=centers,
                       slope=a, phase0=phase0, rho=rho, p=p, n=in_field.sum())
    return recs


def deg(phi):
    return np.mod(np.degrees(phi), 360)


def plot_examples(recs_r, recs_l, fname):
    # rank by strongest (most negative, significant) precession
    allrecs = [("R→", u, r) for u, r in recs_r.items()] + \
              [("L←", u, r) for u, r in recs_l.items()]
    sig = [x for x in allrecs if x[2]["p"] < 0.05 and x[2]["slope"] < 0]
    sig.sort(key=lambda x: x[2]["slope"])   # most negative first
    show = sig[:6]
    fig, axes = plt.subplots(2, 6, figsize=(19, 6.5),
                             gridspec_kw={"height_ratios": [1, 2]})
    for j, (dirlab, u, r) in enumerate(show):
        info = r["info"]
        # top: place field
        ax = axes[0, j]
        ax.plot(r["centers"], r["tc"], "k")
        ax.axvspan(info["lo"], info["hi"], color="tab:orange", alpha=0.2)
        ax.set_title(f"unit {u} {dirlab}\npeak {info['peak']:.1f} Hz", fontsize=9)
        ax.set_xlim(0, 1.6)
        if j == 0:
            ax.set_ylabel("rate (Hz)")
        ax.set_xlabel("pos (m)", fontsize=8)
        # bottom: phase vs within-field position, 2 cycles
        ax = axes[1, j]
        x = r["x"]
        d = deg(r["phi"])
        for off in (0, 360):
            ax.plot(x, d + off, ".", ms=3, color="tab:blue", alpha=0.5)
        xx = np.linspace(0, 1, 50)
        yy = np.degrees(2 * np.pi * r["slope"] * xx + r["phase0"])
        for off in (0, 360, 720):
            ax.plot(xx, np.mod(yy, 360) + off - 0, "-", color="crimson", lw=1.5,
                    alpha=0.9) if False else None
        # draw continuous regression line across 2 cycles
        yline = np.degrees(2 * np.pi * r["slope"] * xx + r["phase0"])
        ax.plot(xx, yline - (np.floor((yline.min())/360))*360, "crimson", lw=2)
        ax.plot(xx, yline - (np.floor((yline.min())/360))*360 + 360, "crimson", lw=2)
        ax.set_ylim(0, 720)
        ax.set_xlim(0, 1)
        ax.set_title(f"slope {r['slope']*360:.0f}°/field\nρ={r['rho']:.2f}, p={r['p']:.3f}",
                     fontsize=9)
        if j == 0:
            ax.set_ylabel("theta phase (deg)")
        ax.set_xlabel("norm. position in field")
    fig.suptitle("Theta phase precession in CA1 place cells (Achilles-10252013, DANDI:000044)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fname, dpi=130)
    print("saved", fname)


def plot_population(recs_r, recs_l, fname):
    recs = {**{("R", u): r for u, r in recs_r.items()},
            **{("L", u): r for u, r in recs_l.items()}}
    slopes = np.array([r["slope"] * 360 for r in recs.values()])  # deg/field
    rhos = np.array([r["rho"] for r in recs.values()])
    ps = np.array([r["p"] for r in recs.values()])
    n_sig_neg = np.sum((ps < 0.05) & (slopes < 0))
    n_sig_pos = np.sum((ps < 0.05) & (slopes > 0))
    print(f"population: {len(recs)} fields; sig precessing (neg) = {n_sig_neg}, "
          f"sig positive = {n_sig_pos}")
    print(f"median slope = {np.median(slopes):.0f} deg/field, median rho = {np.median(rhos):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # (a) slope distribution
    ax[0].hist(slopes, bins=np.arange(-720, 400, 60), color="tab:blue",
               edgecolor="k", alpha=0.8)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axvline(np.median(slopes), color="crimson", lw=2, ls="--",
                  label=f"median {np.median(slopes):.0f}°")
    ax[0].set_xlabel("precession slope (deg / field)")
    ax[0].set_ylabel("# place fields")
    ax[0].set_title(f"(a) Slope distribution (n={len(recs)} fields)")
    ax[0].legend()
    # (b) rho distribution, color by significance
    ax[1].hist(rhos[ps < 0.05], bins=np.arange(-1, 1.05, 0.1), color="crimson",
               alpha=0.8, edgecolor="k", label="p<0.05")
    ax[1].hist(rhos[ps >= 0.05], bins=np.arange(-1, 1.05, 0.1), color="0.7",
               alpha=0.6, edgecolor="k", label="n.s.")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("circular-linear correlation ρ")
    ax[1].set_ylabel("# place fields")
    ax[1].set_title("(b) Phase-position correlation")
    ax[1].legend()
    # (c) pooled phase vs normalized position (all significant precessing cells)
    xs, phis = [], []
    for r in recs.values():
        if r["p"] < 0.05 and r["slope"] < 0:
            xs.append(r["x"]); phis.append(deg(r["phi"]))
    xs = np.concatenate(xs); phis = np.concatenate(phis)
    for off in (0, 360):
        ax[2].plot(xs, phis + off, ".", ms=1.5, color="tab:blue", alpha=0.15)
    # binned circular mean
    bins = np.linspace(0, 1, 11)
    bc = 0.5 * (bins[:-1] + bins[1:])
    mean_ph = []
    for k in range(len(bins) - 1):
        m = (xs >= bins[k]) & (xs < bins[k + 1])
        ang = np.deg2rad(phis[m])
        mean_ph.append(np.mod(np.degrees(np.angle(np.mean(np.exp(1j * ang)))), 360))
    mean_ph = np.array(mean_ph)
    for off in (0, 360):
        ax[2].plot(bc, mean_ph + off, "o-", color="crimson", lw=2, ms=6)
    ax[2].set_ylim(0, 720)
    ax[2].set_xlabel("norm. position in field")
    ax[2].set_ylabel("theta phase (deg)")
    ax[2].set_title(f"(c) Pooled precession ({n_sig_neg} sig. cells)")
    plt.tight_layout()
    plt.savefig(fname, dpi=130)
    print("saved", fname)
    return dict(n_fields=len(recs), n_sig_neg=int(n_sig_neg), n_sig_pos=int(n_sig_pos),
                median_slope=float(np.median(slopes)), median_rho=float(np.median(rhos)))


if __name__ == "__main__":
    np.random.seed(0)
    z, maze, pos, lfp, spikes = load()
    B = build(z, maze, pos, lfp, spikes)
    recs_r = collect(spikes, B, "right")
    recs_l = collect(spikes, B, "left")
    print(f"fields with >= {MIN_FIELD_SPIKES} field spikes: R={len(recs_r)}, L={len(recs_l)}")
    plot_examples(recs_r, recs_l, "fig2_example_precession.png")
    summary = plot_population(recs_r, recs_l, "fig3_population_summary.png")
    np.savez("precession_results.npz", **summary)
