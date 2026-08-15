"""Final figures from the cached multi-session results."""
import numpy as np, pandas as pd, pickle, matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
import pynapple as nap
import precession_lib as pl

mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
EX = "Achilles-10252013"

df = pd.read_csv("all_fields.csv")
res = pickle.load(open("all_sessions.pkl", "rb"))
assets = pl.resolve_assets()

# reload the example session for raw traces
h5 = pl.open_session(assets[EX])
pos, dt, maze = pl.load_position(h5)
units = pl.load_units(h5)
exc = units[(units.cell_type == "excitatory")]
eps, vel = pl.run_epochs(pos, dt, maze)
m0, m1 = pos.index.values[0], pos.index.values[-1]
ch = res[EX]["theta_ch"]
lfp = pl.load_lfp_channel(h5, ch, m0, m1)
filt, phase = pl.theta_phase(lfp)

# ---------------------------------------------------------------- figure 1 --
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(4, 2, figure=fig, hspace=0.65, wspace=0.25,
              height_ratios=[1, 1, 1.2, 1.2])
ax = fig.add_subplot(gs[0, :])
ax.plot(pos.index.values - m0, pos.values, ".", ms=0.6, color="C0")
ax.set(xlabel="time in maze epoch (s)", ylabel="position (m)",
       title=f"{EX}: linearised position, 1.6 m linear track ({len(pos)} tracked samples)")

t0 = eps[1].start[10] - 2
w = nap.IntervalSet(start=t0, end=t0 + 20)
ax = fig.add_subplot(gs[1, 0])
p = pos.restrict(w)
ax.plot(p.index.values - t0, p.values, ".-", ms=2, lw=0.5)
for s, e in zip(eps[1].start, eps[1].end):
    if t0 <= s <= t0 + 20:
        ax.axvspan(s - t0, e - t0, color="C1", alpha=0.25)
for s, e in zip(eps[-1].start, eps[-1].end):
    if t0 <= s <= t0 + 20:
        ax.axvspan(s - t0, e - t0, color="C2", alpha=0.25)
ax.set(xlabel="time (s)", ylabel="position (m)", title="20 s zoom; shading = rightward / leftward runs")

ax = fig.add_subplot(gs[1, 1])
fr, P = welch(lfp.values, fs=pl.LFP_FS, nperseg=4096)
run_lfp = lfp.restrict(eps[1].union(eps[-1]))
frr, Pr = welch(run_lfp.values, fs=pl.LFP_FS, nperseg=2048)
ax.semilogy(fr[fr < 30], P[fr < 30], label="whole maze epoch")
ax.semilogy(frr[frr < 30], Pr[frr < 30], label="running only")
ax.axvspan(*pl.THETA_BAND, color="C1", alpha=0.2)
ax.set(xlabel="frequency (Hz)", ylabel="PSD (V²/Hz)", title=f"LFP spectrum, channel {ch}")
ax.legend(fontsize=7)

ts = eps[1].start[10]
w2 = nap.IntervalSet(start=ts, end=ts + 2.0)
ax = fig.add_subplot(gs[2, :])
l, f = lfp.restrict(w2), filt.restrict(w2)
ax.plot(l.index.values - ts, l.values * 1e3, lw=0.6, color="0.6", label="raw LFP")
ax.plot(f.index.values - ts, f.values * 1e3, lw=1.4, color="k", label="6–10 Hz")
ax.set(ylabel="LFP (mV)", title="One rightward run: LFP and extracted theta")
ax.legend(fontsize=7, loc="upper right")
ax.set_xticklabels([])

ax = fig.add_subplot(gs[3, :])
fields = df[(df.session == EX) & (df.direction == 1)].sort_values("lo")
for i, (_, r) in enumerate(fields.iterrows()):
    s = exc[int(r.unit)].restrict(w2)
    ax.plot(s.index.values - ts, np.full(len(s), i), "|", ms=5, color="C3")
ax.set(xlabel="time (s)", ylabel="place cell #", xlim=(0, 2.0),
       title=f"spikes of the {len(fields)} rightward place cells during the same run")
fig.savefig("fig01_session_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2 --
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5),
                         gridspec_kw={"width_ratios": [1, 1, 0.8], "hspace": 0.45, "wspace": 0.35})
for j, d in enumerate((1, -1)):
    tc, edges, occ, si = res[EX]["tcs"][d]
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = [u for u in tc.columns if tc[u].max() > 1.0 and si[u] > 0.5]
    M = np.array([tc[u].values / tc[u].max() for u in keep])
    order = np.argsort(np.argmax(M, axis=1))
    im = axes[j, 0].imshow(M[order], aspect="auto", origin="lower", cmap="magma",
                           extent=[0, maze['length'], 0, len(keep)])
    axes[j, 0].set(xlabel="position (m)", ylabel="cell (sorted)",
                   title=f"{'rightward' if d>0 else 'leftward'} runs: {len(keep)} spatially tuned cells")
    plt.colorbar(im, ax=axes[j, 0], label="norm. rate")

    sel = df[(df.session == EX) & (df.direction == d)].sort_values("lo")
    for _, r in sel.iterrows():
        axes[j, 1].plot(centers, tc[int(r.unit)].values, lw=1)
    axes[j, 1].set(xlabel="position (m)", ylabel="rate (Hz)",
                   title=f"{len(sel)} place fields entering the precession analysis")
    axes[j, 2].bar(centers, occ, width=pl.POS_BIN * 0.9, color="0.5")
    axes[j, 2].set(xlabel="position (m)", ylabel="occupancy (s)", title="occupancy")
fig.savefig("fig02_place_fields.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 3 --
sel = df[(df.session == EX) & (df.p < 0.05)].sort_values("rho").head(6)
fig, axes = plt.subplots(2, 6, figsize=(16, 6),
                         gridspec_kw={"height_ratios": [0.5, 1], "hspace": 0.4, "wspace": 0.35})
for k, (_, r) in enumerate(sel.iterrows()):
    d = int(r.direction)
    tc, edges, occ, si = res[EX]["tcs"][d]
    centers = 0.5 * (edges[:-1] + edges[1:])
    axes[0, k].plot(centers, tc[int(r.unit)].values, color="k", lw=1)
    axes[0, k].axvspan(r.lo, r.lo + r.width, color="C1", alpha=0.25)
    axes[0, k].set(title=f"unit {int(r.unit)}, {'R' if d>0 else 'L'}", xlabel="pos (m)")
    if k == 0:
        axes[0, k].set_ylabel("rate (Hz)")
    x, ph, st = res[EX]["store"][(int(r.unit), d)]
    axes[1, k].plot(np.r_[x, x], np.degrees(np.r_[ph, ph + 2 * np.pi]), ".", ms=2.5, alpha=0.6)
    xx = np.linspace(0, 1, 50)
    for c in (-1, 0, 1, 2):
        axes[1, k].plot(xx, np.degrees(r.phi0 + 2 * np.pi * r.slope * xx) + 360 * c, "r-", lw=1.5)
    axes[1, k].set(ylim=(-180, 540), xlim=(0, 1), yticks=np.arange(-180, 541, 180),
                   xlabel="position in field",
                   title=f"{r.slope:.2f} cyc, ρ={r.rho:.2f}\np={r.p:.3f}, n={int(r.n)}")
    if k == 0:
        axes[1, k].set_ylabel("theta phase (deg)")
fig.suptitle(f"Theta phase precession in six {EX} place fields (spikes plotted over two theta cycles)", y=1.0)
fig.savefig("fig03_example_precession.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 4 --
r = sel.iloc[0]
d, u = int(r.direction), int(r.unit)
ep = eps[d]
spk_all = exc[u].restrict(ep)
st_all = spk_all.index.values
laps = []
for s, e in zip(ep.start, ep.end):
    m = (st_all >= s) & (st_all <= e)
    if m.sum() < 4:
        continue
    stl = st_all[m]
    xl = pl.field_fraction(nap.Ts(t=stl).value_from(pos).values, r.lo, r.width, d, maze)
    inf = (xl >= 0) & (xl <= 1)
    if inf.sum() >= 4:
        laps.append((s, e, stl, xl, inf))

fig = plt.figure(figsize=(12, 7.5))
gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.25, height_ratios=[1, 1, 1.4])

# two example traversals: theta with spikes drawn on the waveform
for k, (s, e, stl, xl, inf) in enumerate(laps[:2]):
    ax = fig.add_subplot(gs[k, :])
    w = nap.IntervalSet(start=s - 0.1, end=e + 0.1)
    f = filt.restrict(w)
    ax.plot(f.index.values - s, f.values * 1e3, color="0.55", lw=1.1)
    y = np.interp(stl[inf], f.index.values, f.values) * 1e3
    ax.scatter(stl[inf] - s, y, c=xl[inf], cmap="viridis", vmin=0, vmax=1,
               s=45, zorder=3, edgecolor="k", linewidth=0.4)
    ax.set(ylabel="theta (mV)", xlim=(-0.1, e - s + 0.1),
           title=f"traversal {k + 1}: spikes ride progressively earlier phases "
                 f"(colour = position in field)")
    if k == 1:
        ax.set_xlabel("time from run onset (s)")

ax = fig.add_subplot(gs[2, 0])
cmap = plt.get_cmap("tab10")
lap_slopes, xx = [], np.linspace(0, 1, 30)
for i, (s, e, stl, xl, inf) in enumerate(laps):
    if inf.sum() < 6:
        continue
    ph_l = nap.Ts(t=stl[inf]).value_from(phase).values
    fit = pl.circlin_fit(xl[inf], ph_l)
    lap_slopes.append(fit["slope"])
    if len(lap_slopes) <= 8:
        c = cmap(len(lap_slopes) - 1)
        ax.plot(np.r_[xl[inf], xl[inf]], np.degrees(np.r_[ph_l, ph_l + 2 * np.pi]),
                ".", ms=6, color=c, alpha=0.9)
        for k in (0, 1):
            ax.plot(xx, np.degrees(fit["phi0"] + 2 * np.pi * fit["slope"] * xx) + 360 * k,
                    "-", lw=1, color=c, alpha=0.8)
lap_slopes = np.array(lap_slopes)
ax.set(xlabel="normalised position in field", ylabel="theta phase (deg)",
       ylim=(-180, 540), yticks=np.arange(-180, 541, 180), xlim=(0, 1),
       title=f"single passes fitted individually: {len(lap_slopes)} passes,\n"
             f"median slope {np.median(lap_slopes):.2f} cyc, "
             f"{(lap_slopes < 0).mean() * 100:.0f}% negative")

ax = fig.add_subplot(gs[2, 1])
x, ph, _ = res[EX]["store"][(u, d)]
ax.plot(np.r_[x, x], np.degrees(np.r_[ph, ph + 2 * np.pi]), ".", ms=3, alpha=0.5, color="0.4")
edges_x = np.linspace(0, 1, 11)
mu = np.array([np.angle(np.exp(1j * ph[(x >= a) & (x < b)]).mean())
               for a, b in zip(edges_x[:-1], edges_x[1:])])
xc = 0.5 * (edges_x[:-1] + edges_x[1:])
mu_d = np.degrees(np.unwrap(mu))
ax.plot(xc, mu_d, "C3.-", lw=2, label="circular mean")
ax.plot(xc, mu_d + 360, "C3.-", lw=2)
ax.set(ylim=(-180, 540), yticks=np.arange(-180, 541, 180), xlabel="normalised position in field",
       ylabel="theta phase (deg)", title=f"all passes pooled: {r.slope:.2f} cycles, ρ={r.rho:.2f}")
ax.legend(fontsize=7, loc="lower left")
fig.suptitle(f"Single-traversal view of precession, {EX} unit {u} "
             f"({'rightward' if d > 0 else 'leftward'} runs)", y=0.97)
fig.savefig("fig04_single_pass.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 5 --
sig = df[df.p < 0.05]
allX = np.concatenate([v[0] for s in res for v in res[s]["store"].values()])
allP = np.concatenate([v[1] for s in res for v in res[s]["store"].values()])
sigkeys = {(row.session, int(row.unit), int(row.direction)) for row in sig.itertuples()}
sX = np.concatenate([v[0] for s in res for k, v in res[s]["store"].items() if (s, k[0], k[1]) in sigkeys])
sP = np.concatenate([v[1] for s in res for k, v in res[s]["store"].items() if (s, k[0], k[1]) in sigkeys])

fig = plt.figure(figsize=(13, 7.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
ax = fig.add_subplot(gs[0, 0])
bins = np.arange(-2.0, 2.01, 0.1)
ax.hist(df.slope, bins=bins, color="0.8", label=f"all fields (n={len(df)})")
ax.hist(sig.slope, bins=bins, color="C3", label=f"p<0.05 (n={len(sig)})")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(df.slope.median(), color="C0", ls="--", label=f"median {df.slope.median():.2f}")
ax.set(xlabel="regression slope (theta cycles per field)", ylabel="number of fields",
       title="Precession slopes are negative")
ax.legend(fontsize=7)

ax = fig.add_subplot(gs[0, 1])
ax.hist(df.rho, bins=np.arange(-0.8, 0.81, 0.05), color="0.8")
ax.hist(sig.rho, bins=np.arange(-0.8, 0.81, 0.05), color="C3")
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="circular–linear correlation ρ", ylabel="number of fields",
       title="Phase–position correlation")

ax = fig.add_subplot(gs[0, 2])
H, xe, ye = np.histogram2d(np.r_[sX, sX], np.degrees(np.r_[sP, sP + 2 * np.pi]),
                           bins=[np.linspace(0, 1, 26), np.linspace(-180, 540, 49)])
H = H / H.sum(axis=1, keepdims=True)
ax.pcolormesh(xe, ye, H.T, cmap="magma")
xc = 0.5 * (xe[:-1] + xe[1:])
mu = np.array([np.angle(np.exp(1j * sP[(sX >= a) & (sX < b)]).mean())
               for a, b in zip(xe[:-1], xe[1:])])
mu = np.degrees(np.unwrap(mu))
ax.plot(xc, mu, "w.-", lw=2, label="circular mean")
ax.plot(xc, mu + 360, "w.-", lw=2)
ax.set(xlabel="normalised position in field", ylabel="theta phase (deg)",
       ylim=(-180, 540), yticks=np.arange(-180, 541, 180),
       title=f"All spikes from significant fields\n(n={len(sX)} spikes, {len(sig)} fields)")
ax.legend(fontsize=7, loc="lower left")

ax = fig.add_subplot(gs[1, 0])
order = df.groupby("session").slope.median().sort_values().index
data = [df[df.session == s].slope.values for s in order]
bp = ax.boxplot(data, vert=True, tick_labels=[s.replace("-", "\n") for s in order],
                showfliers=False, patch_artist=True)
for p_ in bp["boxes"]:
    p_.set_facecolor("0.85")
for i, s in enumerate(order):
    v = df[df.session == s].slope.values
    ax.plot(np.random.default_rng(i).normal(i + 1, 0.06, len(v)), v, ".", ms=3, color="C3")
ax.axhline(0, color="k", lw=0.8)
ax.set(ylabel="slope (cycles / field)", title="Every session shows negative slopes")
ax.tick_params(axis="x", labelsize=6, rotation=60)

ax = fig.add_subplot(gs[1, 1])
ns = df.p >= .05
ax.scatter(df.width[ns], (-df.slope * 360 / df.width)[ns], c="0.75", s=14, label="n.s.")
ax.scatter(df.width[~ns], (-df.slope * 360 / df.width)[~ns], c="C3", s=14, label="p<0.05")
ax.axhline(0, color="k", lw=0.8)
ax.legend(fontsize=7)
ax.set(xlabel="field width (m)", ylabel="phase advance (deg / m)",
       title="Phase advance per metre vs field size")

ax = fig.add_subplot(gs[1, 2])
ax.hist(np.degrees(np.mod(df.phi0 + np.pi, 2 * np.pi) - np.pi), bins=18, color="0.6")
ax.set(xlabel="phase at field entry (deg)", ylabel="fields",
       title="Entry phase (Hilbert peak = 0°)")
fig.suptitle("Population summary: 8 sessions, 4 rats (DANDI:000044, Grosmark & Buzsáki 2016)", y=0.97)
fig.savefig("fig05_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 6 --
rng = np.random.default_rng(1)
null_rho, null_slope = [], []
for s in res:
    for k, v in res[s]["store"].items():
        x, ph, _ = v
        f = pl.circlin_fit(x, rng.permutation(ph))
        null_rho.append(f["rho"]); null_slope.append(f["slope"])
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
axes[0].hist(null_rho, bins=np.arange(-0.8, 0.81, 0.05), color="0.7", label="phase-shuffled")
axes[0].hist(df.rho, bins=np.arange(-0.8, 0.81, 0.05), histtype="step", color="C3", lw=2, label="observed")
axes[0].axvline(0, color="k", lw=0.8); axes[0].legend(fontsize=7)
axes[0].set(xlabel="ρ", ylabel="fields", title="Observed vs shuffled correlation")
axes[1].hist(null_slope, bins=np.arange(-2, 2.01, 0.1), color="0.7", label="phase-shuffled")
axes[1].hist(df.slope, bins=np.arange(-2, 2.01, 0.1), histtype="step", color="C3", lw=2, label="observed")
axes[1].axvline(0, color="k", lw=0.8); axes[1].legend(fontsize=7)
axes[1].set(xlabel="slope (cycles / field)", title="Observed vs shuffled slope")
ph_run = phase.restrict(eps[1].union(eps[-1])).values
axes[2].hist(np.degrees(ph_run), bins=36, color="0.6")
axes[2].set(xlabel="theta phase (deg)", ylabel="LFP samples",
            title="Phase occupancy during running\n(flat => no sampling bias)")
axes[2].set_ylim(0, axes[2].get_ylim()[1] * 1.25)
for d, c, lab in ((1, "C0", "rightward"), (-1, "C1", "leftward")):
    v = df[df.direction == d].slope
    axes[3].hist(v, bins=np.arange(-2, 2.01, 0.2), histtype="step", lw=2, color=c,
                 label=f"{lab} (n={len(v)}, median {v.median():.2f})")
axes[3].axvline(0, color="k", lw=0.8)
axes[3].legend(fontsize=7)
axes[3].set(xlabel="slope (cycles / field)", ylabel="fields",
            title="Both running directions precess\n(control for the coordinate flip)")
fig.tight_layout()
fig.savefig("fig06_controls.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("figures written")
