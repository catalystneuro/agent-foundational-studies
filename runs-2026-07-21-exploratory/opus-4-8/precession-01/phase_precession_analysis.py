# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theta Phase Precession in Hippocampal CA1 Place Cells
#
# **Dataset:** DANDI:000044 — *Diversity in neural firing dynamics supports both
# rigid and learned hippocampal sequences* (Grosmark, Long & Buzsáki).
# Session `sub-Achilles/ses-Achilles-10252013`.
#
# A rat runs back and forth on a 1.6 m linear track (the MAZE epoch, flanked by
# pre- and post-run sleep). The session provides sorted CA1 units, linearized
# position, and a 128-channel LFP at 1250 Hz.
#
# **Phenomenon.** As an animal crosses a place field, the spikes of the
# corresponding place cell fire at progressively *earlier* phases of the ongoing
# ~8 Hz theta oscillation. This "phase precession" (O'Keefe & Recce, 1993) means a
# cell's theta phase carries information about position within the field beyond
# what the firing rate alone conveys.
#
# **Approach.**
# 1. Stream the session and cache the MAZE-epoch data (spikes, position, one
#    strong-theta LFP channel).
# 2. Compute running speed and split the track traversals into rightward /
#    leftward runs.
# 3. Build direction-specific 1-D place fields and select place cells.
# 4. For place-field spikes, pair the within-field position (distance travelled)
#    with the instantaneous theta phase.
# 5. Quantify precession with a circular-linear regression (Kempter et al., 2012).

# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert, welch

np.random.seed(0)

CACHE = "achilles_maze_cache.npz"
ASSET = "Achilles-10252013"

# Analysis parameters
SPEED_TH = 0.10        # m/s running threshold
MIN_PEAK_RATE = 3.0    # Hz minimum place-field peak
MIN_SI = 0.3           # bits/spike minimum spatial information
N_POS_BINS = 60        # spatial bins over the 1.6 m track
FIELD_FRAC = 0.20      # field = contiguous bins above 20% of peak rate
MIN_FIELD_SPIKES = 30  # minimum in-field spikes to fit precession


# %% [markdown]
# ## 1. Stream the session and cache the MAZE-epoch data
#
# The NWB file is ~9 GB; we stream it with `remfile` and pull only what we need.
# The LFP is gzip-chunked per channel, so a single channel reads cheaply. We pick
# the LFP channel with the strongest theta/delta ratio during running. The result
# is cached to a local `.npz` so re-runs are fast.

# %%
def bandpass(x, lo, hi, fs, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def build_cache():
    from dandi.dandiapi import DandiAPIClient
    import remfile
    import h5py

    client = DandiAPIClient()
    d = client.get_dandiset("000044", "draft")
    asset = [x for x in d.get_assets() if ASSET in x.path][0]
    print("streaming", asset.path)
    rf = remfile.File(asset.download_url)
    h = h5py.File(rf, "r")

    # MAZE epoch = middle epoch (pre-sleep / MAZE / post-sleep)
    ep = h["intervals/epochs"]
    maze_start, maze_stop = float(ep["start_time"][1]), float(ep["stop_time"][1])
    print(f"MAZE epoch: {maze_start:.1f}-{maze_stop:.1f} s")

    # linearized position (0-1.6 m)
    lin = h["processing/behavior/1.6mLinearMazeLinearizedPosition/"
            "1.6mLinearMazeLinearizedTimeSeries"]
    pos_rate = float(lin["starting_time"].attrs["rate"])
    pos_t0 = float(lin["starting_time"][()])
    pos_data = lin["data"][:, 0].astype(float)
    pos_t = pos_t0 + np.arange(pos_data.shape[0]) / pos_rate

    # excitatory units, restricted to the maze epoch
    u = h["units"]
    st_index = u["spike_times_index"][:]
    st_all = u["spike_times"]
    cell_type = u["cell_type"][:]
    location = u["location"][:]
    starts = np.concatenate([[0], st_index[:-1]])
    trains, meta = [], []
    for i in range(len(st_index)):
        s = st_all[starts[i]:st_index[i]]
        ct = cell_type[i].decode() if isinstance(cell_type[i], bytes) else cell_type[i]
        loc = location[i].decode() if isinstance(location[i], bytes) else location[i]
        s = s[(s >= maze_start) & (s <= maze_stop)]
        if ct == "excitatory" and s.size > 20:
            trains.append(s)
            meta.append((i, ct, loc, s.size))
    print(f"kept {len(trains)} excitatory units")

    # choose LFP channel with strongest theta during a 200 s running window
    es = h["processing/ecephys/LFP/LFP/data"]
    lfp_rate = float(h["processing/ecephys/LFP/LFP/starting_time"].attrs["rate"])
    conv = float(es.attrs["conversion"])
    i0 = int(round(maze_start * lfp_rate))
    i1 = int(round(maze_stop * lfp_rate))
    w1 = min(i0 + int(200 * lfp_rate), i1)
    best_chan, best_ratio = 0, -1
    for ch in np.arange(4, 128, 5):
        x = es[i0:w1, ch].astype(float) * conv
        f, p = welch(x, fs=lfp_rate, nperseg=int(4 * lfp_rate))
        ratio = p[(f >= 6) & (f <= 12)].mean() / p[(f >= 1) & (f <= 4)].mean()
        if ratio > best_ratio:
            best_ratio, best_chan = ratio, int(ch)
    print(f"selected LFP channel {best_chan} (theta/delta={best_ratio:.2f})")
    lfp = es[i0:i1, best_chan].astype(float) * conv
    lfp_t = i0 / lfp_rate + np.arange(lfp.size) / lfp_rate

    np.savez_compressed(
        CACHE, maze_start=maze_start, maze_stop=maze_stop,
        pos_t=pos_t, pos=pos_data, pos_rate=pos_rate,
        lfp=lfp.astype(np.float32), lfp_t=lfp_t.astype(np.float64),
        lfp_rate=lfp_rate, best_chan=best_chan,
        spike_trains=np.array(trains, dtype=object),
        unit_meta=np.array(meta, dtype=object))
    print("wrote", CACHE)


if not os.path.exists(CACHE):
    build_cache()
else:
    print("using existing cache", CACHE)


# %% [markdown]
# ## 2. Load into pynapple objects; compute speed, direction, and theta phase

# %%
z = np.load(CACHE, allow_pickle=True)
maze = nap.IntervalSet(start=float(z["maze_start"]), end=float(z["maze_stop"]))
pos = nap.Tsd(t=z["pos_t"], d=z["pos"], time_support=maze)
lfp = nap.Tsd(t=z["lfp_t"], d=z["lfp"].astype(float), time_support=maze)
trains = z["spike_trains"]
meta = z["unit_meta"]
spikes = nap.TsGroup(
    {i: nap.Ts(t=np.asarray(trains[i], dtype=float)) for i in range(len(trains))},
    time_support=maze)
spikes.set_info(n_spk=np.array([m[3] for m in meta]),
                location=np.array([m[2] for m in meta]))
print(f"MAZE {maze.tot_length():.0f} s, {len(spikes)} excitatory units, "
      f"LFP channel {int(z['best_chan'])}")

# clean interpolated position (fill occasional tracking gaps)
_t = pos.index.values
_x = pos.values.copy()
_good = ~np.isnan(_x)
cpos = nap.Tsd(t=_t, d=np.interp(_t, _t[_good], _x[_good]), time_support=maze)

# running speed and signed velocity (smoothed)
_xs = cpos.smooth(0.25).values
_v = np.gradient(_xs, _t)
vel = nap.Tsd(t=_t, d=_v, time_support=maze)
speed = nap.Tsd(t=_t, d=np.abs(_v), time_support=maze)

# theta phase from the Hilbert transform of the 6-12 Hz band (0 rad = theta peak)
fs = float(z["lfp_rate"])
_filt = bandpass(lfp.values, 6, 12, fs)
_analytic = hilbert(_filt)
theta = nap.Tsd(t=lfp.index.values, d=_filt, time_support=maze)
theta_phase = nap.Tsd(t=lfp.index.values, d=np.angle(_analytic), time_support=maze)

# directional run epochs
run_r = vel.threshold(SPEED_TH).time_support.drop_short_intervals(0.3)       # rightward
run_l = (vel * -1).threshold(SPEED_TH).time_support.drop_short_intervals(0.3)  # leftward
print(f"rightward run {run_r.tot_length():.0f} s, leftward run {run_l.tot_length():.0f} s")


# %% [markdown]
# ## Validation: raw position, LFP + theta band, and spike raster

# %%
fig, ax = plt.subplots(3, 1, figsize=(12, 9))
t0 = maze.start[0]
w = (pos.index.values >= t0 + 100) & (pos.index.values <= t0 + 160)
ax[0].plot(pos.index.values[w] - t0, pos.values[w], "k", lw=1)
ax[0].set_ylabel("linear pos (m)")
ax[0].set_title("(a) Linearized position on the 1.6 m track (60 s window)")
axb = ax[0].twinx()
axb.plot(speed.index.values[w] - t0, speed.values[w], "tab:red", lw=0.8, alpha=0.6)
axb.set_ylabel("speed (m/s)", color="tab:red")

wl = (lfp.index.values >= t0 + 120) & (lfp.index.values <= t0 + 122)
ax[1].plot(lfp.index.values[wl] - t0, lfp.values[wl] * 1e3, color="0.6", lw=0.8, label="raw LFP")
ax[1].plot(theta.index.values[wl] - t0, theta.values[wl] * 1e3, "tab:blue", lw=1.5, label="6-12 Hz")
ax[1].set_ylabel("LFP (mV)")
ax[1].set_title(f"(b) CA1 LFP and theta band (channel {int(z['best_chan'])})")
ax[1].legend(loc="upper right", fontsize=8)

order = np.argsort(spikes.get_info("n_spk"))[::-1][:30]
for row, ui in enumerate(order):
    st = spikes[int(ui)].index.values
    st = st[(st >= t0 + 100) & (st <= t0 + 160)]
    ax[2].plot(st - t0, np.full_like(st, row), "|", color="k", ms=3, mew=0.5)
ax[2].set_ylabel("unit (by rate)")
ax[2].set_xlabel("time from window start (s)")
ax[2].set_title("(c) Spike raster, 30 most active excitatory units")
plt.tight_layout()
plt.savefig("fig1_raw_data_validation.png", dpi=130)
print("saved fig1_raw_data_validation.png")


# %% [markdown]
# ## 3. Direction-specific place fields and place-cell selection
#
# Firing-rate-vs-position tuning curves are computed separately for rightward and
# leftward runs (place fields on a linear track are direction-selective). We keep
# units with a peak rate > 3 Hz, Skaggs spatial information > 0.3 bits/spike, and a
# single contiguous field of plausible width.

# %%
def spatial_information(rate, occ):
    occ = occ / occ.sum()
    mean_rate = np.sum(rate * occ)
    if mean_rate <= 0:
        return 0.0
    r = rate / mean_rate
    nz = rate > 0
    return float(np.sum(occ[nz] * r[nz] * np.log2(r[nz])))


def place_fields(run_ep):
    bins = np.linspace(0, 1.6, N_POS_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    p = cpos.restrict(run_ep)
    dt = np.median(np.diff(cpos.index.values))
    occ = np.histogram(p.values, bins=bins)[0] * dt
    tc, si = {}, {}
    for uu in spikes.keys():
        s = spikes[uu].restrict(run_ep)
        if len(s) < 20:
            continue
        sp_pos = np.interp(s.index.values, cpos.index.values, cpos.values)
        cnt = np.histogram(sp_pos, bins=bins)[0]
        rate = np.where(occ > 0.05, cnt / np.where(occ > 0, occ, np.nan), 0.0)
        rate = np.nan_to_num(np.convolve(rate, np.ones(3) / 3, mode="same"))
        tc[uu] = rate
        si[uu] = spatial_information(rate, occ + 1e-9)
    return centers, occ, tc, si


def select_place_cells(centers, tc, si):
    keep = {}
    for uu, rate in tc.items():
        peak = rate.max()
        if peak < MIN_PEAK_RATE or si[uu] < MIN_SI:
            continue
        thr = FIELD_FRAC * peak
        above = rate >= thr
        pk = int(np.argmax(rate))
        lo = pk
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = pk
        while hi < len(rate) - 1 and above[hi + 1]:
            hi += 1
        width = centers[hi] - centers[lo]
        if width < 0.10 or width > 0.9:
            continue
        keep[uu] = dict(peak=peak, si=si[uu], pk_pos=centers[pk],
                        lo=centers[lo], hi=centers[hi], width=width)
    return keep


pf = {}
for name, ep in [("right", run_r), ("left", run_l)]:
    centers, occ, tc, si = place_fields(ep)
    pc = select_place_cells(centers, tc, si)
    pf[name] = dict(centers=centers, tc=tc, pc=pc, ep=ep)
    print(f"{name}: {len(pc)} place cells (of {len(tc)} active units)")


# %% [markdown]
# ## 4-5. Pair theta phase with within-field position and fit circular-linear regression
#
# For each place cell we take the spikes that fall inside its field during runs in
# the cell's direction, express position as the fraction of the field travelled
# (0 = entry, 1 = exit; flipped for leftward runs), read the nearest-sample theta
# phase at each spike, and fit the Kempter et al. (2012) circular-linear model. A
# negative slope / negative correlation is the signature of phase precession;
# significance comes from a phase-shuffle permutation test.

# %%
def circ_lin_fit(x, phi, slope_range=(-2.0, 2.0), n=2001, nperm=500):
    x = np.asarray(x); phi = np.asarray(phi)
    slopes = np.linspace(*slope_range, n)
    R = np.array([np.abs(np.mean(np.exp(1j * (phi - 2 * np.pi * s * x)))) for s in slopes])
    a = slopes[np.argmax(R)]
    phase0 = np.angle(np.mean(np.exp(1j * (phi - 2 * np.pi * a * x))))
    theta_c = 2 * np.pi * np.abs(a) * x
    phi_bar = np.angle(np.sum(np.exp(1j * phi)))
    th_bar = np.angle(np.sum(np.exp(1j * theta_c)))
    num = np.sum(np.sin(phi - phi_bar) * np.sin(theta_c - th_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2) * np.sum(np.sin(theta_c - th_bar) ** 2))
    rho = num / den if den > 0 else 0.0
    cnt = 0
    for _ in range(nperm):
        pp = np.random.permutation(phi)
        pb = np.angle(np.sum(np.exp(1j * pp)))
        rp = np.sum(np.sin(pp - pb) * np.sin(theta_c - th_bar)) / den if den > 0 else 0.0
        if abs(rp) >= abs(rho):
            cnt += 1
    return a, phase0, rho, (cnt + 1) / (nperm + 1)


def phase_at(spike_t, ph_t, ph_v):
    idx = np.clip(np.searchsorted(ph_t, spike_t), 1, len(ph_t) - 1)
    idx = np.where(np.abs(spike_t - ph_t[idx - 1]) <= np.abs(spike_t - ph_t[idx]),
                   idx - 1, idx)
    return ph_v[idx]


def collect(direction):
    d = pf[direction]
    ep, pc = d["ep"], d["pc"]
    ph_t, ph_v = theta_phase.index.values, theta_phase.values
    recs = {}
    for uu, info in pc.items():
        s = spikes[uu].restrict(ep)
        sp_t = s.index.values
        sp_pos = np.interp(sp_t, cpos.index.values, cpos.values)
        lo, hi = info["lo"], info["hi"]
        m = (sp_pos >= lo) & (sp_pos <= hi)
        if m.sum() < MIN_FIELD_SPIKES:
            continue
        if direction == "right":
            xnorm = (sp_pos[m] - lo) / (hi - lo)
        else:                                    # leftward runs traverse hi -> lo
            xnorm = (hi - sp_pos[m]) / (hi - lo)
        phi = phase_at(sp_t[m], ph_t, ph_v)
        a, phase0, rho, p = circ_lin_fit(xnorm, phi)
        recs[uu] = dict(info=info, x=xnorm, phi=phi, tc=d["tc"][uu],
                        centers=d["centers"], slope=a, phase0=phase0,
                        rho=rho, p=p, n=int(m.sum()))
    return recs


t = time.time()
recs_r = collect("right")
recs_l = collect("left")
print(f"fit {len(recs_r)} rightward + {len(recs_l)} leftward fields "
      f"in {time.time()-t:.0f}s")


# %% [markdown]
# ## Example place cells with clear phase precession

# %%
def deg(phi):
    return np.mod(np.degrees(phi), 360)


allrecs = [("R→", u, r) for u, r in recs_r.items()] + \
          [("L←", u, r) for u, r in recs_l.items()]
# canonical examples: significant precession, field well inside the track, and a
# physiological slope (< 1.5 theta cycles); rank by correlation strength
sig = sorted(
    [x for x in allrecs
     if x[2]["p"] < 0.05 and x[2]["slope"] < 0
     and x[2]["info"]["lo"] > 0.05 and x[2]["info"]["hi"] < 1.55
     and x[2]["slope"] * 360 > -540],
    key=lambda x: x[2]["rho"])[:6]

fig, axes = plt.subplots(2, 6, figsize=(19, 6.5),
                         gridspec_kw={"height_ratios": [1, 2]})
for j, (dirlab, u, r) in enumerate(sig):
    info = r["info"]
    ax = axes[0, j]
    ax.plot(r["centers"], r["tc"], "k")
    ax.axvspan(info["lo"], info["hi"], color="tab:orange", alpha=0.2)
    ax.set_title(f"unit {u} {dirlab}\npeak {info['peak']:.1f} Hz", fontsize=9)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("pos (m)", fontsize=8)
    if j == 0:
        ax.set_ylabel("rate (Hz)")
    ax = axes[1, j]
    for off in (0, 360):
        ax.plot(r["x"], deg(r["phi"]) + off, ".", ms=3, color="tab:blue", alpha=0.5)
    xx = np.linspace(0, 1, 50)
    yline = np.degrees(2 * np.pi * r["slope"] * xx + r["phase0"])
    base = np.floor(yline.min() / 360) * 360
    ax.plot(xx, yline - base, "crimson", lw=2)
    ax.plot(xx, yline - base + 360, "crimson", lw=2)
    ax.set_ylim(0, 720)
    ax.set_xlim(0, 1)
    ax.set_title(f"slope {r['slope']*360:.0f}°/field\nρ={r['rho']:.2f}, "
                 f"p={r['p']:.3f}", fontsize=9)
    ax.set_xlabel("norm. position in field")
    if j == 0:
        ax.set_ylabel("theta phase (deg)")
fig.suptitle("Theta phase precession in CA1 place cells "
             "(Achilles-10252013, DANDI:000044)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("fig2_example_precession.png", dpi=130)
print("saved fig2_example_precession.png")


# %% [markdown]
# ## Population summary

# %%
recs = {**{("R", u): r for u, r in recs_r.items()},
        **{("L", u): r for u, r in recs_l.items()}}
slopes = np.array([r["slope"] * 360 for r in recs.values()])
rhos = np.array([r["rho"] for r in recs.values()])
ps = np.array([r["p"] for r in recs.values()])
n_sig_neg = int(np.sum((ps < 0.05) & (slopes < 0)))
n_sig_pos = int(np.sum((ps < 0.05) & (slopes > 0)))
print(f"{len(recs)} fields; significant precessing (neg slope) = {n_sig_neg}, "
      f"significant positive = {n_sig_pos}")
print(f"median slope = {np.median(slopes):.0f} deg/field, "
      f"median rho = {np.median(rhos):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].hist(slopes, bins=np.arange(-720, 400, 60), color="tab:blue",
           edgecolor="k", alpha=0.8)
ax[0].axvline(0, color="k", lw=1)
ax[0].axvline(np.median(slopes), color="crimson", lw=2, ls="--",
              label=f"median {np.median(slopes):.0f}°")
ax[0].set_xlabel("precession slope (deg / field)")
ax[0].set_ylabel("# place fields")
ax[0].set_title(f"(a) Slope distribution (n={len(recs)} fields)")
ax[0].legend()

ax[1].hist(rhos[ps < 0.05], bins=np.arange(-1, 1.05, 0.1), color="crimson",
           alpha=0.8, edgecolor="k", label="p<0.05")
ax[1].hist(rhos[ps >= 0.05], bins=np.arange(-1, 1.05, 0.1), color="0.7",
           alpha=0.6, edgecolor="k", label="n.s.")
ax[1].axvline(0, color="k", lw=1)
ax[1].set_xlabel("circular-linear correlation ρ")
ax[1].set_ylabel("# place fields")
ax[1].set_title("(b) Phase-position correlation")
ax[1].legend()

xs = np.concatenate([r["x"] for r in recs.values() if r["p"] < 0.05 and r["slope"] < 0])
phis = np.concatenate([deg(r["phi"]) for r in recs.values()
                       if r["p"] < 0.05 and r["slope"] < 0])
for off in (0, 360):
    ax[2].plot(xs, phis + off, ".", ms=1.5, color="tab:blue", alpha=0.15)
bins = np.linspace(0, 1, 11)
bc = 0.5 * (bins[:-1] + bins[1:])
mean_ph = np.array([
    np.mod(np.degrees(np.angle(np.mean(
        np.exp(1j * np.deg2rad(phis[(xs >= bins[k]) & (xs < bins[k + 1])]))))), 360)
    for k in range(len(bins) - 1)])
for off in (0, 360):
    ax[2].plot(bc, mean_ph + off, "o-", color="crimson", lw=2, ms=6)
ax[2].set_ylim(0, 720)
ax[2].set_xlabel("norm. position in field")
ax[2].set_ylabel("theta phase (deg)")
ax[2].set_title(f"(c) Pooled precession ({n_sig_neg} sig. cells)")
plt.tight_layout()
plt.savefig("fig3_population_summary.png", dpi=130)
print("saved fig3_population_summary.png")


# %% [markdown]
# ## Conclusion
#
# Across both running directions, CA1 place cells show robust theta phase
# precession. The slope distribution is strongly biased negative (median
# ≈ −180°/field, i.e. roughly half a theta cycle of phase advance across a field),
# the circular-linear correlation is negative for essentially every significant
# cell, and the pooled phase-vs-position plot descends monotonically from field
# entry to exit. This reproduces the classic O'Keefe & Recce (1993) result on real
# CA1 recordings streamed directly from the DANDI Archive.
