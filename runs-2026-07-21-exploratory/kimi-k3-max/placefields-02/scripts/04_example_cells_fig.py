"""Figure 2: example place cells - spikes on trajectory + per-direction tuning."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "scripts")
from pf_common import load_session, compute_bouts, bout_axis, spike_maps

d = np.load("cache/place_fields.npz")
keys = d["keys"]; cell_type = d["cell_type"]; rates = d["rates"]
rates_pos = d["rates_pos"]; rates_neg = d["rates_neg"]
si = d["si_real"]; is_place = d["is_place"]; centers = d["centers"]

nwb, _ = load_session()
units = nwb["units"]
lin = nwb["1.6mLinearMazeLinearizedTimeSeries"]
t = lin.t
x = np.asarray(lin.values).ravel()
bouts, dt = compute_bouts(t, x)
tau, offsets, T_total = bout_axis(t, x, bouts)

# pick 6 excitatory place cells: high SI, spread-out field locations
exc_place = np.where(is_place & (cell_type == "excitatory"))[0]
peak_bin = np.nanargmax(rates[exc_place], axis=1)
order = np.lexsort((-si[exc_place], peak_bin))  # by peak bin, SI desc within
# greedy diverse pick: sort by SI, take cells whose peak is >2 bins from chosen
cand = exc_place[np.argsort(-si[exc_place])]
chosen = []
for c in cand:
    pb = np.nanargmax(rates[c])
    if all(abs(pb - np.nanargmax(rates[k])) > 2 for k in chosen):
        chosen.append(c)
    if len(chosen) == 6:
        break
chosen = chosen[::-1]

fig, axes = plt.subplots(6, 2, figsize=(11, 13),
                         gridspec_kw=dict(hspace=0.55, wspace=0.25,
                                          left=0.07, right=0.97, top=0.95, bottom=0.05))
for row, ci in enumerate(chosen):
    k = keys[ci]
    st = units[k].t
    tau_s, x_s, sdir = spike_maps(st, t, x, bouts, offsets)

    # left: position vs time (concatenated bout axis), spikes overlaid
    ax = axes[row, 0]
    ok = ~np.isnan(tau)
    ax.plot(tau[ok], x[ok], lw=0.4, color="0.75", zorder=1)
    ax.scatter(tau_s, x_s, s=4, c="crimson", zorder=2, rasterized=True)
    ax.set_ylabel("track position (m)", fontsize=9)
    ax.set_ylim(-0.05, 1.65)
    ax.set_xlim(0, T_total)
    ax.set_title(f"unit {k} — spikes on trajectory (SI={si[ci]:.2f} bits/spike)",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=8)
    if row == 5:
        ax.set_xlabel("time in run bouts (s)", fontsize=9)

    # right: per-direction tuning curves
    ax = axes[row, 1]
    ax.plot(centers, rates_pos[ci], color="tab:blue", lw=1.5, label="pos. direction")
    ax.plot(centers, rates_neg[ci], color="tab:orange", lw=1.5, label="neg. direction")
    ax.set_ylabel("firing rate (Hz)", fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_title(f"unit {k} — tuning by direction (peak {np.nanmax(rates[ci]):.1f} Hz)",
                 fontsize=9, loc="left")
    ax.tick_params(labelsize=8)
    if row == 0:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    if row == 5:
        ax.set_xlabel("track position (m)", fontsize=9)

fig.suptitle("Example CA1 place cells — DANDI 000044, Achilles 10252013", fontsize=13)
fig.savefig("figures/fig2_example_place_cells.png", dpi=150)
print("saved figures/fig2_example_place_cells.png")
print("chosen units:", [int(keys[c]) for c in chosen])
