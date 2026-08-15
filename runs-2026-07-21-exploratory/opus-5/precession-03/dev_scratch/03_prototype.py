"""Single-session prototype: epochs, place fields, theta phase, precession."""
import numpy as np, matplotlib.pyplot as plt, time
import pynapple as nap
import precession_lib as pl

t_start = time.time()
assets = pl.resolve_assets()
h5 = pl.open_session(assets["Achilles-10252013"])

pos, dt = pl.load_position(h5)
units = pl.load_units(h5)
eps, vel = pl.run_epochs(pos, dt)
print("dt", dt, "n pos samples", len(pos))
for d, e in eps.items():
    print(f"dir {d:+d}: {len(e)} runs, {e.tot_length():.1f} s total, "
          f"median {np.median(e.end - e.start):.2f} s")

maze = nap.IntervalSet(start=pos.index.values[0] - 1, end=pos.index.values[-1] + 1)
ch, chans, ratios = pl.pick_theta_channel(h5, maze.start[0])
print("theta channel", ch, "ratio", ratios.max(), "elapsed", time.time() - t_start)

lfp = pl.load_lfp_channel(h5, ch, maze.start[0], maze.end[0])
filt, phase = pl.theta_phase(lfp)
print("lfp loaded", len(lfp), "elapsed", time.time() - t_start)

exc = units[(units.cell_type == "excitatory")]
print("n excitatory", len(exc))

for d in (1, -1):
    ep = eps[d]
    tc = pl.rate_maps(exc, pos, ep)
    occ, edges = pl.occupancy(pos, ep, dt)
    si = pl.spatial_info(tc, occ)
    peaks = tc.max()
    cand = [u for u in tc.columns if peaks[u] > 1.0 and si[u] > 0.5]
    print(f"dir {d:+d}: {len(cand)} place cells (peak>1Hz, SI>0.5)")
    if d == 1:
        tc1, ep1, cand1, edges1 = tc, ep, cand, edges

# --- inspect the best candidate ---
centers = 0.5 * (edges1[:-1] + edges1[1:])
best = max(cand1, key=lambda u: tc1[u].max())
print("best unit", best, "peak rate", tc1[best].max())

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(centers, tc1[best].values)
ax[0, 0].set(xlabel="position (m)", ylabel="rate (Hz)", title=f"unit {best} rate map (dir +1)")
lo, hi, pk = pl.detect_field(tc1[best].values, edges1)
ax[0, 0].axvspan(lo, hi, alpha=0.2, color="orange")

spk = exc[best].restrict(ep1)
sp_pos = spk.value_from(pos)
sp_ph = spk.value_from(phase)
inf = (sp_pos.values >= lo) & (sp_pos.values <= hi)
x = (sp_pos.values[inf] - lo) / (hi - lo)
ph = sp_ph.values[inf]
print("spikes in field", inf.sum())
res = pl.circlin_pvalue(x, ph, n_shuf=200)
print(res)

ax[0, 1].plot(np.r_[x, x], np.r_[ph, ph + 2 * np.pi] * 180 / np.pi, ".", ms=3)
xx = np.linspace(0, 1, 100)
for k in (0, 1, 2):
    ax[0, 1].plot(xx, np.degrees(res["phi0"] + 2 * np.pi * res["slope"] * xx) + 360 * k, "r-")
ax[0, 1].set(ylim=(-180, 540), xlabel="normalised position in field",
             ylabel="theta phase (deg)",
             title=f"slope={res['slope']:.2f} cyc, rho={res['rho']:.2f}, p={res['p']:.3f}")

t0 = ep1.start[3]
w = nap.IntervalSet(start=t0 - 0.5, end=t0 + 2.5)
l = lfp.restrict(w); f = filt.restrict(w)
ax[1, 0].plot(l.index.values, l.values * 1e3, lw=0.5, color="gray")
ax[1, 0].plot(f.index.values, f.values * 1e3, lw=1, color="k")
s = exc[best].restrict(w)
ax[1, 0].plot(s.index.values, np.zeros(len(s)) + 0.4, "r|", ms=10)
ax[1, 0].set(xlabel="time (s)", ylabel="LFP (mV)", title="raw + 6-10 Hz theta, spikes")

ax[1, 1].hist(np.degrees(phase.restrict(ep1).values), bins=36)
ax[1, 1].set(xlabel="theta phase (deg)", title="phase distribution during runs (uniformity check)")
plt.tight_layout(); plt.savefig("check_prototype.png", dpi=110)
print("total elapsed", time.time() - t_start)
