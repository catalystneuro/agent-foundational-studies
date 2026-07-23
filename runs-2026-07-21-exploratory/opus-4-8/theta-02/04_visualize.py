"""Figures for theta phase entrainment and precession in one hc-11 session."""
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

import hc11
from analysis_core import analyze_session, precession

SESSION = "Achilles_10252013"
res = analyze_session(SESSION)
prec = precession(res, n_shuffle=20)
pos, phase, units = res["pos"], res["phase"], res["units"]
centers = res["centers"]

# =============================================================== fig 2: fields
place_r = [r for r in res["place"] if r["direction"] == "right"
           and r["cell_type"] == "excitatory" and r["field"] is not None
           and r["field"]["peak_rate"] >= hc11.MIN_PEAK_RATE and r["si"] >= 0.5]
place_l = [r for r in res["place"] if r["direction"] == "left"
           and r["cell_type"] == "excitatory" and r["field"] is not None
           and r["field"]["peak_rate"] >= hc11.MIN_PEAK_RATE and r["si"] >= 0.5]

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
occ_r = res["place"][0]["occ"]
occ_l = [r for r in res["place"] if r["direction"] == "left"][0]["occ"]
ax.plot(centers, occ_r, label="rightward runs")
ax.plot(centers, occ_l, label="leftward runs")
ax.set_xlabel("position (m)"), ax.set_ylabel("occupancy (s)")
ax.set_title("Occupancy during running", fontsize=10)
ax.legend(fontsize=8)

for j, (recs, lab) in enumerate([(place_r, "rightward"), (place_l, "leftward")]):
    ax = fig.add_subplot(gs[0, j + 1])
    M = np.array([r["tc"] / np.nanmax(r["tc"]) for r in recs])
    order = np.argsort([np.nanargmax(r["tc"]) for r in recs])
    im = ax.imshow(M[order], aspect="auto", origin="lower", cmap="magma",
                   extent=[centers[0], centers[-1], 0, len(recs)])
    ax.set_xlabel("position (m)")
    ax.set_ylabel("place cell (sorted)")
    ax.set_title(f"Normalised place fields, {lab} (n={len(recs)})", fontsize=10)
    plt.colorbar(im, ax=ax, label="norm. rate")

ax = fig.add_subplot(gs[1, :2])
bypos = sorted(place_r, key=lambda r: r["field"]["peak_pos"])
best = [bypos[i] for i in np.linspace(0, len(bypos) - 1, 6).astype(int)]
for r in best:
    ax.plot(centers, r["tc"], lw=1.6,
            label=f"unit {r['unit']} (SI {r['si']:.1f} bits/spk)")
ax.set_xlabel("position (m)"), ax.set_ylabel("firing rate (Hz)")
ax.set_title("Example place fields spanning the track, rightward runs", fontsize=10)
ax.legend(fontsize=8, ncol=2)

ax = fig.add_subplot(gs[1, 2])
si_all = [r["si"] for r in res["place"] if r["cell_type"] == "excitatory"]
si_pc = [r["si"] for r in place_r + place_l]
ax.hist(si_all, bins=np.linspace(0, 5, 30), color="tab:gray", label="all pyramidal")
ax.hist(si_pc, bins=np.linspace(0, 5, 30), color="tab:blue", label="place cells")
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.set_xlabel("spatial information (bits/spike)"), ax.set_ylabel("count")
ax.set_title("Spatial information", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle(f"{SESSION}: CA1 place fields on the 1.6 m linear track", fontsize=12)
fig.savefig("fig02_place_fields.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig02_place_fields.png")

# ======================================================== fig 3: entrainment
lock = [l for l in res["locking"] if l["n"] >= 100]
exc = [l for l in lock if l["cell_type"] == "excitatory"]
inh = [l for l in lock if l["cell_type"] == "inhibitory"]

fig = plt.figure(figsize=(14, 12))
outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.25, 1.25], hspace=0.48)
gs_top = outer[0].subgridspec(2, 1, height_ratios=[0.55, 1.45], hspace=0.06)
gs_mid = outer[1].subgridspec(1, 3, wspace=0.32)
gs_bot = outer[2].subgridspec(1, 3, wspace=0.42)

# raw: LFP + spike raster over a few theta cycles during a run
t0 = res["run_right"].start[12]
win = nap.IntervalSet(t0, t0 + 1.5)
ax_lfp = fig.add_subplot(gs_top[0])
lf = res["filt"].restrict(win)
raw = res["lfp"].restrict(win)
ax_lfp.plot(raw.t - t0, raw.values * 1e3, color="tab:gray", lw=0.6)
ax_lfp.plot(lf.t - t0, lf.values * 1e3, color="tab:blue", lw=1.6)
ax_lfp.set_ylabel("LFP (mV)"), ax_lfp.set_xlim(0, 1.5)
ax_lfp.tick_params(labelbottom=False)
ax_lfp.set_title("Spikes of CA1 place cells align to the theta cycle "
                 "(1.5 s of running; grey = raw LFP, blue = 6-10 Hz)", fontsize=10)

ax_r = fig.add_subplot(gs_top[1], sharex=ax_lfp)
order = np.argsort([np.nanargmax(r["tc"]) for r in place_r])
for k, idx in enumerate(order):
    st = units[place_r[idx]["unit"]].restrict(win)
    if len(st):
        ax_r.plot(st.t - t0, np.full(len(st), k), "|", color="k", ms=4)
ph_win = res["phase"].restrict(win)
for c in ph_win.t[1:][np.diff(ph_win.values) < -np.pi]:      # theta cycle starts
    ax_r.axvline(c - t0, color="tab:blue", lw=0.6, alpha=0.4)
ax_r.set_xlabel("time (s)")
ax_r.set_ylabel("place cell\n(sorted by field position)")
ax_r.set_ylim(-1, len(order))

# example phase histograms
ex = sorted(exc, key=lambda l: -l["mrl"])[:2] + sorted(inh, key=lambda l: -l["mrl"])[:1]
bins = np.linspace(0, 2 * np.pi, 25)
ctr = 0.5 * (bins[:-1] + bins[1:])
for j, l in enumerate(ex):
    ax = fig.add_subplot(gs_mid[j])
    h, _ = np.histogram(l["phases"], bins=bins)
    h = h / h.sum()
    ax.bar(np.r_[ctr, ctr + 2 * np.pi], np.r_[h, h],
           width=bins[1] - bins[0],
           color="tab:blue" if l["cell_type"] == "excitatory" else "tab:red",
           edgecolor="none")
    xx = np.linspace(0, 4 * np.pi, 300)
    ax.plot(xx, h.mean() * (1 + 0.5 * np.cos(xx)), color="k", lw=1)
    ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    ax.set_xticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("theta phase (deg)"), ax.set_ylabel("spike fraction")
    ax.set_title(f"unit {l['unit']} ({l['cell_type'][:3]}), MRL={l['mrl']:.2f}, "
                 f"n={l['n']} spikes", fontsize=9)

# population polar distribution of preferred phases
ax = fig.add_subplot(gs_bot[0], projection="polar")
sig_e = [l for l in exc if l["p"] < 0.01]
sig_i = [l for l in inh if l["p"] < 0.01]
for group, color, lab in [(sig_e, "tab:blue", "pyramidal"),
                          (sig_i, "tab:red", "interneuron")]:
    ax.plot([l["pref_phase"] for l in group], [l["mrl"] for l in group],
            "o", color=color, ms=5, alpha=0.75, label=lab)
ax.set_rlabel_position(150)
ax.set_rticks([0.2, 0.4, 0.6])
ax.tick_params(labelsize=7)
ax.set_title("Preferred phase (angle) and\nlocking strength (radius)",
             fontsize=10, pad=26)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

ax = fig.add_subplot(gs_bot[1])
pv = np.array([max(l["p"], 1e-300) for l in lock])
ax.hist([l["mrl"] for l in exc], bins=np.linspace(0, 0.6, 25), alpha=0.75,
        color="tab:blue", label=f"pyramidal (n={len(exc)})")
ax.hist([l["mrl"] for l in inh], bins=np.linspace(0, 0.6, 25), alpha=0.75,
        color="tab:red", label=f"interneuron (n={len(inh)})")
ax.set_xlabel("mean resultant length"), ax.set_ylabel("count")
ax.set_title(f"Locking strength: {(pv < 0.01).sum()}/{len(pv)} units\n"
             f"significant (Rayleigh p<0.01)", fontsize=10)
ax.legend(fontsize=8)

ax = fig.add_subplot(gs_bot[2])
allph = np.concatenate([l["phases"] for l in exc])
h, _ = np.histogram(allph, bins=bins)
h = h / h.sum()
ax.bar(np.r_[ctr, ctr + 2 * np.pi], np.r_[h, h], width=bins[1] - bins[0],
       color="tab:blue")
xx = np.linspace(0, 4 * np.pi, 300)
ax.plot(xx, h.mean() * (1 + 0.5 * np.cos(xx)), color="k", lw=1)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_xticklabels(["0", "180", "360", "540", "720"])
ax.set_xlabel("theta phase (deg)"), ax.set_ylabel("spike fraction")
ax.set_ylim(0, max(h) * 1.25)
ax.set_title(f"All pyramidal spikes pooled\n(n={len(allph):,})", fontsize=10)

fig.suptitle(f"{SESSION}: theta phase entrainment of CA1 units during running",
             fontsize=12)
fig.savefig("fig03_theta_entrainment.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig03_theta_entrainment.png")

# ================================================= fig 4: precession examples
show = sorted([q for q in prec if q["p"] < 0.01 and q["slope"] < 0],
              key=lambda q: q["rho"])[:6]
fig, axes = plt.subplots(2, 6, figsize=(16, 6),
                         gridspec_kw=dict(height_ratios=[1, 2.4], hspace=0.62,
                                          wspace=0.35))
for j, q in enumerate(show):
    a = axes[0, j]
    a.plot(q["centers"], q["tc"], color="k", lw=1.3)
    a.axvspan(q["field"]["lo"], q["field"]["hi"], color="tab:orange", alpha=0.25)
    a.set_title(f"unit {q['unit']}, {q['direction']}", fontsize=9)
    a.set_xlabel("position (m)", fontsize=8)
    if j == 0:
        a.set_ylabel("rate (Hz)", fontsize=8)
    a.tick_params(labelsize=7)

    a = axes[1, j]
    ph = np.degrees(q["phi"])
    a.plot(np.r_[q["x"], q["x"]], np.r_[ph, ph + 360], ".", ms=3,
           color="tab:blue", alpha=0.6)
    xx = np.linspace(0, 1, 200)
    yy = np.degrees(np.mod(q["slope"] * xx + q["phi0"], 2 * np.pi))
    yy = np.where(np.r_[False, np.abs(np.diff(yy)) > 180], np.nan, yy)
    for off in (0, 360, 720):
        a.plot(xx, yy + off, color="tab:red", lw=1.6)
    a.set_ylim(0, 720), a.set_xlim(0, 1)
    a.set_yticks([0, 180, 360, 540, 720])
    a.set_xlabel("position in field", fontsize=8)
    if j == 0:
        a.set_ylabel("theta phase (deg)", fontsize=8)
    ptxt = "p<1e-16" if q["p"] < 1e-16 else f"p={q['p']:.1e}"
    a.set_title(f"$\\rho$={q['rho']:.2f}, slope={np.degrees(q['slope']):.0f}$\\degree$\n"
                f"{ptxt}, n={q['n']}", fontsize=8)
    a.tick_params(labelsize=7)
fig.suptitle(f"{SESSION}: theta phase precession in single CA1 place fields",
             fontsize=12)
fig.savefig("fig04_precession_examples.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig04_precession_examples.png")

# =============================================== fig 5: precession population
slopes = np.array([q["slope"] for q in prec])
rhos = np.array([q["rho"] for q in prec])
pvals = np.array([q["p"] for q in prec])
sigmask = (pvals < 0.05) & (slopes < 0)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

X = np.concatenate([q["x"] for q in prec])
P = np.concatenate([q["phi"] for q in prec])
H, xe, ye = np.histogram2d(X, P, bins=[20, 24], range=[[0, 1], [0, 2 * np.pi]])
H = H / H.sum(axis=1, keepdims=True)
im = axes[0].imshow(np.tile(H.T, (2, 1)), origin="lower", aspect="auto",
                    extent=[0, 1, 0, 720], cmap="magma")
axes[0].set_xlabel("normalised position in field")
axes[0].set_ylabel("theta phase (deg)")
axes[0].set_yticks([0, 180, 360, 540, 720])
axes[0].set_title(f"Pooled spikes, all fields (n={len(prec)})", fontsize=10)
plt.colorbar(im, ax=axes[0], label="P(phase | position)")

axes[1].hist(np.degrees(slopes), bins=np.linspace(-720, 720, 41), color="tab:gray")
axes[1].axvline(0, color="tab:red", lw=1.5)
axes[1].axvline(np.degrees(np.median(slopes)), color="tab:blue", ls="--",
                label=f"median {np.degrees(np.median(slopes)):.0f}$\\degree$")
axes[1].set_xlabel("precession slope (deg per field)")
axes[1].set_ylabel("count"), axes[1].legend(fontsize=8)
axes[1].set_title(f"{(slopes < 0).mean() * 100:.0f}% of fields have a negative slope",
                  fontsize=10)

bb = np.linspace(-1, 1, 41)
axes[2].hist(rhos, bins=bb, color="tab:gray", density=True, label="observed")
shufr = np.concatenate([q["rho_shuffled"] for q in prec])
axes[2].hist(shufr, bins=bb, histtype="step", color="tab:red", lw=1.8,
             density=True, label=f"phase-shuffled\n({len(shufr)} draws)")
axes[2].axvline(0, color="k", lw=1)
axes[2].set_xlabel("circular-linear correlation $\\rho$")
axes[2].set_ylabel("density"), axes[2].legend(fontsize=7)
axes[2].set_title(f"{sigmask.sum()}/{len(prec)} fields precess significantly\n"
                  f"(p<0.05, negative slope)", fontsize=10)

axes[3].scatter([q["field"]["width"] for q in prec], np.degrees(slopes),
                c=np.where(sigmask, "tab:blue", "tab:gray"), s=18)
axes[3].axhline(0, color="tab:red", lw=1)
axes[3].set_xlabel("field width (m)"), axes[3].set_ylabel("slope (deg per field)")
axes[3].set_title("Slope vs field width", fontsize=10)

fig.suptitle(f"{SESSION}: population summary of theta phase precession", fontsize=12)
fig.tight_layout()
fig.savefig("fig05_precession_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig05_precession_population.png")
