"""Step 4: figures demonstrating theta phase precession in the example session."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pynapple as nap

import precession_lib as pl
from precession_03 import analyse_session, SESSION

df, ctx = analyse_session(SESSION)
df.to_csv("precession_single_session.csv", index=False)
pc = df[df.is_place_cell].copy()
per_field = ctx["per_field"]

# ===========================================================================
# Figure 4: raw demonstration -- one lap, one cell
# ===========================================================================
# pick the field with the strongest (most negative) rho and plenty of spikes
best = pc.sort_values("rho").iloc[0]
fd = per_field[(best.unit, best.direction)]
ep = ctx["runs"][best.direction]
spk = ctx["exc"][best.unit]

# among laps with enough in-field spikes, show the single pass whose own
# circular-linear fit is the clearest (an "example pass", as in the literature)
lap_scores = []
for i, (s_, e_) in enumerate(zip(ep.start, ep.end)):
    one = nap.IntervalSet(start=s_, end=e_)
    xl, pl_, tl = pl.field_spike_phase_position(spk, one, ctx["pos"], ctx["phase"],
                                                fd["lo"], fd["hi"], best.direction)
    if len(xl) < 8:
        lap_scores.append(np.inf)
        continue
    al, _, _, _ = pl.circlin_regress(xl, pl_)
    lap_scores.append(pl.circlin_corr(xl, pl_, al))
lap_i = int(np.argmin(lap_scores))
lap = nap.IntervalSet(start=ep.start[lap_i] - 0.2, end=ep.end[lap_i] + 0.2)

fig = plt.figure(figsize=(14, 8.5))
# a dedicated narrow column for the colourbar keeps the three left panels the
# same width, so the shared time axis actually lines up
gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.1, 1.4], hspace=0.5, wspace=0.32,
              width_ratios=[1.55, 0.12, 1])

t0 = lap.start[0]
axp = fig.add_subplot(gs[0, 0])
p_lap = ctx["pos"].restrict(lap)
axp.plot(p_lap.times() - t0, p_lap.values, "k", lw=2)
axp.axhspan(fd["lo"], fd["hi"], color="tab:orange", alpha=0.25, lw=0)
axp.set_ylabel("position (m)")
axp.set_title(f"Example {best.direction}ward pass (lap {lap_i+1} of {len(ep)}, the single pass with the\n"
              f"clearest precession); orange band = place field of unit {best.unit}",
              fontsize=10)

axl = fig.add_subplot(gs[1, 0], sharex=axp)
l_lap = ctx["lfp"].restrict(lap)
f_lap = ctx["filt"].restrict(lap)
axl.plot(l_lap.times() - t0, l_lap.values * 1e3, color="0.7", lw=0.7, label="raw LFP")
axl.plot(f_lap.times() - t0, f_lap.values * 1e3, "k", lw=1.2, label="theta (6-12 Hz)")
st = spk.restrict(lap)
sp_phi = np.interp(st.times(), ctx["phase"].times(), np.unwrap(ctx["phase"].values))
sp_phi = np.degrees((sp_phi + np.pi) % (2 * np.pi) - np.pi)
# put each spike on the theta wave at the moment it occurred
sp_amp = np.interp(st.times(), f_lap.times(), f_lap.values) * 1e3
axl.plot(st.times() - t0, sp_amp, "o", mfc="tab:red", mec="k", mew=0.4, ms=6, zorder=5,
         label="spikes of unit %s" % best.unit)
axl.set_ylabel("LFP (mV)")
axl.legend(loc="lower left", fontsize=8, ncol=3)
axl.set_title("Spikes ride down the theta wave, cycle after cycle", fontsize=10)

axs = fig.add_subplot(gs[2, 0], sharex=axp)
axs.plot(st.times() - t0, sp_phi, "o", mfc="tab:red", mec="k", mew=0.4, ms=7)
axs.plot(st.times() - t0, sp_phi + 360, "o", mfc="tab:red", mec="k", mew=0.4, ms=7, alpha=0.4)
# circular-linear fit for this single pass (phase against time within the field)
one = nap.IntervalSet(start=ep.start[lap_i], end=ep.end[lap_i])
xl, pl_, tl = pl.field_spike_phase_position(spk, one, ctx["pos"], ctx["phase"],
                                            fd["lo"], fd["hi"], best.direction)
a1, p01, _, _ = pl.circlin_regress(xl, pl_)
rho1 = pl.circlin_corr(xl, pl_, a1)
order = np.argsort(xl)
for off in (-720, -360, 0, 360, 720):
    axs.plot(tl[order] - t0, np.degrees(2 * np.pi * a1 * xl[order] + p01) + off,
             "k--", lw=1.3)
axs.text(0.98, 0.04, f"this pass: {a1*360:.0f}$\\degree$/field, $\\rho$ = {rho1:.2f}, "
         f"n = {len(xl)} spikes", transform=axs.transAxes, fontsize=9, ha="right",
         bbox=dict(fc="w", ec="0.7", alpha=0.9, pad=2))
axs.set_ylim(-180, 540)
axs.set_yticks([-180, 0, 180, 360, 540])
axs.set_ylabel("spike theta phase (deg)")
axs.set_xlabel("time within lap (s)")
axs.set_title("Successive spikes occur at progressively earlier theta phases", fontsize=10)
in_field = (p_lap.values >= fd["lo"]) & (p_lap.values <= fd["hi"])
if in_field.any():
    tf = p_lap.times()[in_field]
    axp.set_xlim(tf[0] - t0 - 0.35, tf[-1] - t0 + 0.35)

# right column: all laps overlaid for this cell
axr = fig.add_subplot(gs[:, 2])
x, phi = fd["x"], np.degrees(fd["phi"])
for off in (0, 360):
    axr.scatter(x, phi + off, s=16, c="k", alpha=0.55, edgecolors="none")
xx = np.linspace(0, 1, 100)
yy = np.degrees(2 * np.pi * best.slope_cycles * xx + best.phi0)
for off in (-360, 0, 360, 720):
    axr.plot(xx, yy + off, "r", lw=2)
axr.set_ylim(-180, 540)
axr.set_yticks([-180, 0, 180, 360, 540])
axr.set_xlim(0, 1)
axr.set_xlabel("normalised position in field\n(0 = entry, 1 = exit)")
axr.set_ylabel("theta phase (deg)")
axr.set_title(f"unit {best.unit}, {best.direction}ward, all {len(ctx['runs'][best.direction])} laps\n"
              f"slope {best.slope_deg:.0f} deg/field, "
              f"$\\rho$ = {best.rho:.2f}, p = {best.pval:.3f}, n = {len(x)} spikes",
              fontsize=10)

fig.suptitle(f"{SESSION}: theta phase precession, raw signals and single-cell summary",
             fontsize=13)
fig.savefig("fig04_precession_example.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# Figure 5: a gallery of example fields
# ===========================================================================
gallery = pc[(pc.pval < 0.05) & (pc.rho < 0)].sort_values("n_field_spikes",
                                                          ascending=False).head(8)
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for ax, (_, r) in zip(axes.ravel(), gallery.iterrows()):
    fd = per_field[(r.unit, r.direction)]
    x, phi = fd["x"], np.degrees(fd["phi"])
    for off in (0, 360):
        ax.scatter(x, phi + off, s=8, c="0.25", alpha=0.5, edgecolors="none")
    xx = np.linspace(0, 1, 100)
    yy = np.degrees(2 * np.pi * r.slope_cycles * xx + r.phi0)
    for off in (-360, 0, 360, 720):
        ax.plot(xx, yy + off, "r", lw=1.8)
    ax.set_ylim(-180, 540)
    ax.set_xlim(0, 1)
    ax.set_yticks([-180, 0, 180, 360, 540])
    ax.set_title(f"unit {r.unit} {r.direction[0].upper()}  |  {r.slope_deg:.0f}$\\degree$/field\n"
                 f"$\\rho$={r.rho:.2f}, p={r.pval:.3f}, n={int(r.n_field_spikes)}", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("theta phase (deg)")
for ax in axes[-1, :]:
    ax.set_xlabel("normalised position in field")
fig.suptitle(f"{SESSION}: single place fields with significant phase precession", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig05_precession_gallery.png", dpi=150)
plt.close(fig)

# ===========================================================================
# Figure 6: population summary for this session
# ===========================================================================
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.hist(pc.slope_deg, bins=np.arange(-720, 760, 60), color="0.35", edgecolor="w")
ax.axvline(0, color="k", lw=1)
ax.axvline(pc.slope_deg.median(), color="tab:red", lw=2,
           label=f"median {pc.slope_deg.median():.0f}$\\degree$")
ax.set_xlabel("regression slope (deg per field traversal)")
ax.set_ylabel("# place fields")
ax.set_title("Slopes are overwhelmingly negative")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[0, 1])
sig = pc.pval < 0.05
ax.hist([pc.rho[sig], pc.rho[~sig]], bins=np.arange(-0.8, 0.85, 0.08), stacked=True,
        color=["tab:red", "0.75"], edgecolor="w", label=["p < 0.05", "n.s."])
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("circular-linear correlation $\\rho$")
ax.set_ylabel("# place fields")
ax.set_title(f"{sig.sum()}/{len(pc)} fields significant ({100*sig.mean():.0f}%)")
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[0, 2])
ax.scatter(pc.field_width * 100, pc.slope_deg, c=np.where(sig, "tab:red", "0.7"), s=28)
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("field width (cm)")
ax.set_ylabel("slope (deg/field)")
ax.set_title("Slope vs field size")

# pooled phase-position density over all significant fields
ax = fig.add_subplot(gs[1, :2])
X = np.concatenate([per_field[(r.unit, r.direction)]["x"] for _, r in pc[sig].iterrows()])
P = np.degrees(np.concatenate([per_field[(r.unit, r.direction)]["phi"]
                               for _, r in pc[sig].iterrows()]))
H, xe, ye = np.histogram2d(np.r_[X, X], np.r_[P, P + 360],
                           bins=[np.linspace(0, 1, 21), np.linspace(-180, 540, 41)])
H = H / H.sum(axis=1, keepdims=True)
im = ax.imshow(H.T, origin="lower", aspect="auto", cmap="magma",
               extent=[0, 1, -180, 540], interpolation="bilinear")
ax.set_xlabel("normalised position in field (0 = entry, 1 = exit)")
ax.set_ylabel("theta phase (deg)")
ax.set_yticks([-180, 0, 180, 360, 540])
ax.set_title(f"Pooled spike density, {sig.sum()} significant fields, {len(X)} spikes\n"
             "(phase axis repeated over two cycles)")
fig.colorbar(im, ax=ax, label="P(phase | position)")

ax = fig.add_subplot(gs[1, 2], projection="polar")
# mean phase at the first and last quarter of the field
for lo, hi, c, lab in ((0.0, 0.25, "tab:blue", "field entry\n(first 25%)"),
                       (0.75, 1.0, "tab:orange", "field exit\n(last 25%)")):
    m = (X >= lo) & (X < hi)
    ph = np.radians(P[m])
    ax.hist(ph % (2 * np.pi), bins=36, range=(0, 2 * np.pi), alpha=0.55, color=c, label=lab,
            density=True)
    R = np.exp(1j * ph).mean()
    ax.annotate("", xy=(np.angle(R), np.abs(R) * 1.2), xytext=(0, 0),
                arrowprops=dict(color=c, width=2.5, headwidth=8))
ax.set_theta_zero_location("E")
ax.set_yticklabels([])
ax.set_title("Phase distribution\nearly vs late in field", fontsize=10, pad=22)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.12), fontsize=8)

fig.suptitle(f"{SESSION}: population statistics of theta phase precession", fontsize=13)
fig.savefig("fig06_precession_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("wrote fig04, fig05, fig06")
print(f"{sig.sum()}/{len(pc)} significant; median slope {pc.slope_deg.median():.0f} deg/field; "
      f"median rho {pc.rho.median():.3f}")
