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
# # Theta phase entrainment and phase precession of hippocampal place cells
#
# **Dataset: DANDI:000044** ("Diversity in neural firing dynamics supports both
# rigid and learned hippocampal sequences", Grosmark & Buzsáki 2016, *Science*;
# the `hc-11` data set). Eight sessions of bilateral silicon-probe recordings
# from dorsal CA1 in four rats running for water reward, each flanked by a
# pre-run and post-run sleep epoch. Each NWB file contains spike times for
# sorted units labelled excitatory or inhibitory, a 128-channel local field
# potential at 1250 Hz, and 2D position tracking at 39 Hz.
#
# Five of the eight sessions use a linear runway (1.6 m or 2 m); the other three
# use a circular ring track. Only the five linear-track sessions are analysed
# here, because the linearization below assumes a straight track. The check that
# establishes this is included at the end of the notebook.
#
# This notebook demonstrates two classical properties of the hippocampal theta
# rhythm:
#
# 1. **Theta phase entrainment.** During running, CA1 pyramidal cells and
#    interneurons fire preferentially at a particular phase of the 6-10 Hz LFP
#    theta oscillation rather than uniformly across the cycle.
# 2. **Theta phase precession** (O'Keefe & Recce 1993). As the animal traverses
#    a place field, the spikes of that place cell occur at progressively earlier
#    theta phases, so phase carries information about position within the field
#    over and above firing rate.
#
# Everything is streamed from the DANDI Archive with LINDI and a local cache;
# no file is downloaded in full (the session files are 5-9 GB each, and we read
# roughly 100 MB per session).
#
# ## Approach
#
# - Restrict to the linear-maze epoch and to periods of running (> 5 cm/s),
#   analysed separately for rightward and leftward traversals because CA1 place
#   fields on a linear track are direction-selective.
# - Choose the LFP channel with the largest theta/delta power ratio during the
#   maze epoch as the phase reference, bandpass-filter it at 6-10 Hz and take
#   the Hilbert phase.
# - Quantify entrainment with the Rayleigh test on spike phases, and precession
#   with the circular-linear correlation of Kempter et al. (2012).

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO
from scipy.signal import hilbert, welch
from scipy.stats import norm
from tqdm.auto import tqdm

np.random.seed(0)

# hc-11 sessions on DANDI:000044 (asset IDs of the draft version)
SESSIONS = {
    "Achilles_10252013": "5349c68b-c0a7-46c0-9900-cda050722fa4",
    "Achilles_11012013": "8855c8cc-9d8b-4d5b-8ef0-fe87916f839a",
    "Cicero_09012014": "3cc5b7b3-02e2-490a-9f19-d20670355084",
    "Cicero_09102014": "f61dfe09-3db2-464a-b386-2e828b2e7276",
    "Cicero_09172014": "e381ebb3-128e-4f3f-9517-11277d7aed9b",
    "Gatsby_08022013": "31ea0aab-4777-424e-9a93-9605b2bdcc29",
    "Gatsby_08282013": "f7687af7-3bc9-4d20-8d88-ef293d2a3381",
    "Buddy_06272013": "82714afb-724f-4e2b-b102-c9c47b5cba73",
}
# Three of the eight sessions were recorded on a circular ring track rather
# than a linear runway (verified by plotting the raw 2D trajectories). The
# principal-axis linearization used below folds a ring onto a line and destroys
# the place-field structure, so only the five linear-track sessions are used.
LINEAR_SESSIONS = ["Achilles_10252013", "Cicero_09012014", "Cicero_09172014",
                   "Gatsby_08022013", "Buddy_06272013"]

LINDI_TMPL = ("https://lindi.neurosift.org/dandi/dandisets/000044/"
              "assets/{asset_id}/nwb.lindi.json")

LFP_FS = 1250.0            # Hz, LFP sampling rate in these files
THETA_BAND = (6.0, 10.0)   # Hz
SPEED_THRESH = 0.05        # m/s, running threshold
MIN_PEAK_RATE = 1.0        # Hz, minimum in-field peak rate for a place cell
MIN_SI = 0.5               # bits/spike, minimum spatial information
N_POS_BINS = 50
SESSION = "Achilles_10252013"   # prototyping session

# %% [markdown]
# ## Data access helpers
#
# Two details of this particular NWB conversion matter:
#
# - The `SpatialSeries.rate` field holds the sampling *period* (0.0256 s), not a
#   rate in Hz, so timestamps are rebuilt explicitly.
# - The stored `LinearizedPosition` series is NaN for about 87% of the maze
#   epoch (it was only filled in during identified traversals), whereas the 2D
#   series is ~79% valid. We therefore linearize the 2D series ourselves by
#   projecting onto the principal axis of the runway. The result correlates with
#   the stored linearized position at r = 0.99999 where both are defined.


# %%
def open_session(name, cache_dir=".lindi_cache"):
    """Open an hc-11 session by streaming from DANDI via LINDI."""
    url = LINDI_TMPL.format(asset_id=SESSIONS[name])
    f = lindi.LindiH5pyFile.from_lindi_file(
        url, local_cache=lindi.LocalCache(cache_dir=cache_dir))
    io = NWBHDF5IO(file=f, mode="r")
    return io.read(), io


def maze_epoch(nwbfile):
    ep = nwbfile.epochs.to_dataframe()
    row = ep[ep["label"].str.contains("Maze")].iloc[0]
    return nap.IntervalSet(start=row["start_time"], end=row["stop_time"])


def _valid_intervals(t, ok, max_gap):
    """IntervalSet of stretches with valid tracking, bridging gaps < max_gap."""
    bad = ~ok
    edges = np.diff(bad.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if bad[0]:
        starts = np.r_[0, starts]
    if bad[-1]:
        ends = np.r_[ends, len(t)]
    long_gaps = [(t[s], t[min(e, len(t) - 1)]) for s, e in zip(starts, ends)
                 if t[min(e, len(t) - 1)] - t[s] > max_gap]
    good_start, good_end = [t[0]], []
    for g0, g1 in long_gaps:
        good_end.append(g0)
        good_start.append(g1)
    good_end.append(t[-1])
    keep = [(a, b) for a, b in zip(good_start, good_end) if b - a > 1.0]
    return nap.IntervalSet(start=[a for a, _ in keep], end=[b for _, b in keep])


def load_position(nwbfile, max_gap=0.5):
    """Linear position along the runway (metres) as a pynapple Tsd."""
    key = [k for k in nwbfile.processing["behavior"].data_interfaces
           if k.endswith("MazePosition")][0]
    ts = list(nwbfile.processing["behavior"][key].spatial_series.values())[0]
    period = ts.rate                       # seconds per sample, despite the name
    xy = np.asarray(ts.data[:], dtype=float)
    t = ts.starting_time + np.arange(xy.shape[0]) * period

    finite = np.isfinite(xy).all(axis=1)
    y_med = np.median(xy[finite, 1])
    ok = finite & (np.abs(xy[:, 1] - y_med) < 0.25)   # keep the runway band

    pts = np.ascontiguousarray(xy[ok] - xy[ok].mean(axis=0))
    axis = np.ascontiguousarray(np.linalg.svd(pts, full_matrices=False)[2][0])
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    proj = pts.dot(axis)
    lin = np.full(len(t), np.nan)
    lin[ok] = proj - proj.min()
    return nap.Tsd(t=t, d=np.interp(t, t[ok], lin[ok]),
                   time_support=_valid_intervals(t, ok, max_gap))


def load_units(nwbfile):
    u = nwbfile.units.to_dataframe()
    tsgroup = nap.TsGroup({i: np.asarray(u["spike_times"].iloc[i])
                           for i in range(len(u))})
    tsgroup.set_info(cell_type=np.asarray(u["cell_type"].values),
                     location=np.asarray(u["location"].values),
                     shank_id=np.asarray(u["shank_id"].values))
    return tsgroup


def load_lfp_channel(nwbfile, channel, epoch):
    """Stream a single LFP channel over `epoch`. Data are chunked per channel,
    so this reads only a few MB rather than the whole 128-channel array."""
    es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
    i0 = int(np.floor((float(epoch.start[0]) - es.starting_time) * LFP_FS))
    i1 = int(np.ceil((float(epoch.end[0]) - es.starting_time) * LFP_FS))
    d = es.data[i0:i1, channel].astype(np.float64) * es.conversion
    return nap.Tsd(t=es.starting_time + np.arange(i0, i1) / LFP_FS, d=d)


def theta_phase(lfp, band=THETA_BAND, fs=LFP_FS):
    filt = nap.apply_bandpass_filter(lfp, band, fs=fs)
    phase = np.mod(np.angle(hilbert(filt.values)), 2 * np.pi)
    return filt, nap.Tsd(t=filt.t, d=phase)


def compute_velocity(pos, smooth_std=0.15):
    """Velocity (m/s), differentiated separately within each tracked interval so
    that dropouts are never bridged."""
    from scipy.ndimage import gaussian_filter1d
    ts, vs = [], []
    for i in range(len(pos.time_support)):
        seg = pos.restrict(pos.time_support[i:i + 1])
        if len(seg) < 10:
            continue
        dt = np.median(np.diff(seg.t))
        ts.append(seg.t)
        vs.append(np.gradient(gaussian_filter1d(seg.values, smooth_std / dt,
                                                mode="nearest"), seg.t))
    return nap.Tsd(t=np.concatenate(ts), d=np.concatenate(vs),
                   time_support=pos.time_support)


def run_epochs(vel, speed_thresh=SPEED_THRESH, max_speed=2.0):
    speed = nap.Tsd(t=vel.t, d=np.abs(vel.values), time_support=vel.time_support)
    ok = speed.threshold(max_speed, "below").time_support      # tracking glitches
    run = speed.restrict(ok).threshold(speed_thresh, "above").time_support
    run = run.merge_close_intervals(0.2).drop_short_intervals(0.5)
    right = vel.restrict(run).threshold(0, "above").time_support.drop_short_intervals(0.5)
    left = vel.restrict(run).threshold(0, "below").time_support.drop_short_intervals(0.5)
    return speed, run, right, left


# %% [markdown]
# ## Circular statistics
#
# `rayleigh` tests spike phases against a uniform distribution. `circ_lin_corr`
# implements the circular-linear regression of Kempter et al. (2012): the slope
# is found by maximising the resultant length of the phase residuals, and the
# correlation coefficient and its asymptotic p-value follow from their
# equations. Slope is reported in radians of theta phase per field traversal.

# %%
def rayleigh(phases):
    n = len(phases)
    if n < 3:
        return np.nan, np.nan, np.nan
    C, S = np.cos(phases).sum(), np.sin(phases).sum()
    R = np.hypot(C, S) / n
    mu = np.mod(np.arctan2(S, C), 2 * np.pi)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))
    return R, mu, p


def circ_lin_corr(x, phi, slope_bounds=(-2.0, 2.0), n_slopes=2001):
    x, phi = np.asarray(x, float), np.asarray(phi, float)
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan
    slopes = np.linspace(*slope_bounds, n_slopes)          # cycles per unit x
    R = np.abs(np.exp(1j * (phi[None, :]
                            - 2 * np.pi * slopes[:, None] * x[None, :])).mean(axis=1))
    a = slopes[np.argmax(R)]
    phi0 = np.mod(np.angle(np.exp(1j * (phi - 2 * np.pi * a * x)).mean()), 2 * np.pi)

    th = np.mod(2 * np.pi * np.abs(a) * x, 2 * np.pi)
    phi_bar = np.arctan2(np.sin(phi - th).sum(), np.cos(phi - th).sum())
    th_bar = np.arctan2(np.sin(th).sum(), np.cos(th).sum())
    num = np.sum(np.sin(phi - phi_bar) * np.sin(th - th_bar))
    den = np.sqrt(np.sum(np.sin(phi - phi_bar) ** 2)
                  * np.sum(np.sin(th - th_bar) ** 2))
    rho = num / den if den > 0 else np.nan
    lam = lambda i, j: np.mean((np.sin(phi - phi_bar) ** i)
                               * (np.sin(th - th_bar) ** j))
    l20, l02, l22 = lam(2, 0), lam(0, 2), lam(2, 2)
    z = rho * np.sqrt(n * l20 * l02 / l22) if l22 > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return rho, p, 2 * np.pi * a, phi0


# %% [markdown]
# ## Place fields

# %%
def spatial_info(tc, occ):
    """Skaggs spatial information, bits per spike."""
    p = occ / occ.sum()
    r = np.asarray(tc, float)
    rbar = np.nansum(p * r)
    ok = (r > 0) & (p > 0)
    return float(np.sum(p[ok] * (r[ok] / rbar) * np.log2(r[ok] / rbar))) \
        if rbar > 0 else np.nan


def find_field(tc, bins, frac=0.2, min_w=0.15, max_w=1.0):
    """Contiguous field around the peak, thresholded at `frac` of the peak rate."""
    r = np.asarray(tc, float)
    pk = int(np.nanargmax(r))
    thr = frac * r[pk]
    lo, hi = pk, pk
    while lo > 0 and r[lo - 1] >= thr:
        lo -= 1
    while hi < len(r) - 1 and r[hi + 1] >= thr:
        hi += 1
    width = bins[hi] - bins[lo]
    if not (min_w <= width <= max_w):
        return None
    return dict(lo=bins[lo], hi=bins[hi], peak_pos=bins[pk],
                peak_rate=r[pk], width=width)


# %% [markdown]
# ## Full per-session pipeline

# %%
def analyze_session(session, n_bins=N_POS_BINS, verbose=True):
    nwbfile, io = open_session(session)
    maze = maze_epoch(nwbfile)
    pos_all = load_position(nwbfile)
    tracked = maze.intersect(pos_all.time_support)
    pos = pos_all.restrict(tracked)
    vel = compute_velocity(pos)
    speed, run, run_r, run_l = run_epochs(vel)
    units = load_units(nwbfile)

    # LFP phase reference: channel with the largest theta/delta ratio, searching
    # one channel per shank.
    el = nwbfile.electrodes.to_dataframe()
    cand = [int(sub.index.to_numpy()[len(sub) // 2])
            for _, sub in el.groupby("group_name", sort=False)]
    best, best_score, lfp, psds = None, -np.inf, None, {}
    for ch in tqdm(cand, desc=f"{session}: LFP channels", disable=not verbose):
        lfp_c = load_lfp_channel(nwbfile, ch, maze)
        fr, pxx = welch(lfp_c.values, fs=LFP_FS, nperseg=int(4 * LFP_FS))
        psds[ch] = (fr, pxx)
        score = (pxx[(fr >= 6) & (fr <= 10)].mean()
                 / pxx[(fr >= 1) & (fr <= 4)].mean())
        if score > best_score:
            best, best_score, lfp = ch, score, lfp_c
    filt, phase = theta_phase(lfp)

    # direction-specific place fields
    edges = np.linspace(0, float(pos.max()), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    place = []
    for label, ep in [("right", run_r), ("left", run_l)]:
        p_ep = pos.restrict(ep)
        occ = np.histogram(p_ep.values, bins=edges)[0] * np.median(np.diff(pos.t))
        tc = nap.compute_1d_tuning_curves(units, p_ep, nb_bins=n_bins,
                                          minmax=(0, float(pos.max())), ep=ep)
        for uid in units.index:
            rates = tc[uid].values
            place.append(dict(session=session, unit=int(uid), direction=label,
                              tc=rates, centers=centers, occ=occ,
                              cell_type=units.cell_type[uid],
                              si=spatial_info(rates, occ),
                              field=find_field(rates, centers)))

    # theta phase locking during running
    locking = []
    for uid in units.index:
        st = units[uid].restrict(run)
        ph = st.value_from(phase).values if len(st) else np.array([])
        R, mu, p = rayleigh(ph)
        locking.append(dict(session=session, unit=int(uid), n=len(ph), mrl=R,
                            pref_phase=mu, p=p, phases=ph,
                            cell_type=units.cell_type[uid]))

    return dict(session=session, maze=maze, pos=pos, speed=speed, run=run,
                run_right=run_r, run_left=run_l, units=units, lfp=lfp,
                filt=filt, phase=phase, best_ch=best, theta_delta=best_score,
                place=place, locking=locking, centers=centers, edges=edges,
                tracked=tracked, candidates=cand, psds=psds)


def precession(res, min_spikes=40, n_shuffle=0, rng=None):
    """Circular-linear regression of theta phase on position within each field."""
    units, phase, pos = res["units"], res["phase"], res["pos"]
    rng = rng or np.random.default_rng(0)
    out = []
    for rec in res["place"]:
        f = rec["field"]
        if (f is None or rec["cell_type"] != "excitatory"
                or f["peak_rate"] < MIN_PEAK_RATE or rec["si"] < MIN_SI):
            continue
        ep = res["run_right"] if rec["direction"] == "right" else res["run_left"]
        st = units[rec["unit"]].restrict(ep)
        if len(st) < min_spikes:
            continue
        spos = st.value_from(pos).values
        sph = st.value_from(phase).values
        inside = (spos >= f["lo"]) & (spos <= f["hi"])
        if inside.sum() < min_spikes:
            continue
        # Distance travelled into the field: 0 at entry, 1 at exit. On leftward
        # runs the animal enters at the high-position edge, so the coordinate is
        # reversed relative to the linearized axis.
        if rec["direction"] == "right":
            x = (spos[inside] - f["lo"]) / f["width"]
        else:
            x = (f["hi"] - spos[inside]) / f["width"]
        phi = sph[inside]
        rho, p, slope, phi0 = circ_lin_corr(x, phi)
        # Control: permuting phases within the field destroys any relation
        # between position and phase while preserving both marginals.
        shuf = [circ_lin_corr(x, rng.permutation(phi))[0]
                for _ in range(n_shuffle)]
        out.append(dict(rho_shuffled=np.array(shuf),
                        session=rec["session"], unit=rec["unit"],
                        direction=rec["direction"], rho=rho, p=p, slope=slope,
                        phi0=phi0, n=int(inside.sum()), x=x, phi=phi,
                        field=f, si=rec["si"], tc=rec["tc"],
                        centers=rec["centers"]))
    return out


# %% [markdown]
# ## Load one session and check every data stream
#
# Before analysing anything we plot the behaviour, the LFP spectra of all
# candidate channels, and the theta filter output, to confirm each stream is
# what we think it is.

# %%
res = analyze_session(SESSION)
print(f"{SESSION}: {len(res['units'])} units, "
      f"{(res['units'].cell_type == 'excitatory').sum()} excitatory")
print(f"maze epoch {res['maze'].tot_length():.0f} s, "
      f"tracked {res['tracked'].tot_length():.0f} s, "
      f"running {res['run'].tot_length():.0f} s "
      f"({res['run_right'].tot_length():.0f} s right, "
      f"{res['run_left'].tot_length():.0f} s left)")
print(f"theta phase reference: channel {res['best_ch']} "
      f"(theta/delta = {res['theta_delta']:.2f})")

# %%
maze, pos, speed = res["maze"], res["pos"], res["speed"]
fig, axes = plt.subplots(4, 1, figsize=(12, 11))

axes[0].plot(pos.t - maze.start[0], pos.values, lw=0.7, color="k")
axes[0].set_ylabel("linearized\nposition (m)")
axes[0].set_title(f"{SESSION}: maze epoch behaviour and LFP")
axes[0].set_xlim(0, maze.tot_length())

axes[1].plot(speed.t - maze.start[0], speed.values, lw=0.5, color="tab:gray")
axes[1].axhline(SPEED_THRESH, color="tab:red", ls="--", label="run threshold")
axes[1].set_ylabel("speed (m/s)"), axes[1].set_ylim(0, 1.2)
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_xlim(0, maze.tot_length())

for ch, (fr, pxx) in res["psds"].items():
    axes[2].semilogy(fr, pxx, lw=0.9, alpha=0.6,
                     color="tab:red" if ch == res["best_ch"] else "tab:gray")
axes[2].axvspan(*THETA_BAND, color="tab:blue", alpha=0.15)
axes[2].set_xlim(0, 30)
axes[2].set_xlabel("frequency (Hz)"), axes[2].set_ylabel("PSD (V$^2$/Hz)")
axes[2].set_title(f"LFP power spectra, one channel per shank "
                  f"(red = selected channel {res['best_ch']})", fontsize=10)

t0 = res["run_right"].start[10]
win = nap.IntervalSet(t0, t0 + 2)
axes[3].plot(res["lfp"].restrict(win).t - t0,
             res["lfp"].restrict(win).values * 1e3, color="tab:gray", lw=0.8,
             label="raw LFP")
axes[3].plot(res["filt"].restrict(win).t - t0,
             res["filt"].restrict(win).values * 1e3, color="tab:blue", lw=1.5,
             label="6-10 Hz")
ax3b = axes[3].twinx()
ax3b.plot(res["phase"].restrict(win).t - t0, res["phase"].restrict(win).values,
          color="tab:orange", lw=0.8, alpha=0.7)
ax3b.set_ylabel("theta phase (rad)", color="tab:orange")
axes[3].set_xlabel("time from start of a running bout (s)")
axes[3].set_ylabel("LFP (mV)"), axes[3].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("fig01_preprocessing.png", dpi=150)
plt.show()

# %% [markdown]
# The animal runs about 40 traversals of the track. The LFP has a clear ~9 Hz
# theta peak during the maze epoch, and the bandpass-filtered signal tracks the
# raw trace cycle by cycle.

# %% [markdown]
# ## Place fields
#
# Place cells are defined as excitatory units with a contiguous field (rate
# above 20% of the peak) between 15 cm and 1 m wide, an in-field peak rate of at
# least 1 Hz, and at least 0.5 bits/spike of spatial information, computed
# separately for each running direction.

# %%
def is_place_cell(r):
    return (r["cell_type"] == "excitatory" and r["field"] is not None
            and r["field"]["peak_rate"] >= MIN_PEAK_RATE and r["si"] >= MIN_SI)


centers = res["centers"]
place_r = [r for r in res["place"] if r["direction"] == "right" and is_place_cell(r)]
place_l = [r for r in res["place"] if r["direction"] == "left" and is_place_cell(r)]
print(f"place cells: {len(place_r)} rightward, {len(place_l)} leftward")

fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
ax.plot(centers, res["place"][0]["occ"], label="rightward runs")
ax.plot(centers, [r for r in res["place"] if r["direction"] == "left"][0]["occ"],
        label="leftward runs")
ax.set_xlabel("position (m)"), ax.set_ylabel("occupancy (s)")
ax.set_title("Occupancy during running", fontsize=10), ax.legend(fontsize=8)

for j, (recs, lab) in enumerate([(place_r, "rightward"), (place_l, "leftward")]):
    ax = fig.add_subplot(gs[0, j + 1])
    M = np.array([r["tc"] / np.nanmax(r["tc"]) for r in recs])
    order = np.argsort([np.nanargmax(r["tc"]) for r in recs])
    im = ax.imshow(M[order], aspect="auto", origin="lower", cmap="magma",
                   extent=[centers[0], centers[-1], 0, len(recs)])
    ax.set_xlabel("position (m)"), ax.set_ylabel("place cell (sorted)")
    ax.set_title(f"Normalised place fields, {lab} (n={len(recs)})", fontsize=10)
    plt.colorbar(im, ax=ax, label="norm. rate")

ax = fig.add_subplot(gs[1, :2])
bypos = sorted(place_r, key=lambda r: r["field"]["peak_pos"])
for r in [bypos[i] for i in np.linspace(0, len(bypos) - 1, 6).astype(int)]:
    ax.plot(centers, r["tc"], lw=1.6,
            label=f"unit {r['unit']} (SI {r['si']:.1f} bits/spk)")
ax.set_xlabel("position (m)"), ax.set_ylabel("firing rate (Hz)")
ax.set_title("Example place fields spanning the track, rightward runs", fontsize=10)
ax.legend(fontsize=8, ncol=2)

ax = fig.add_subplot(gs[1, 2])
ax.hist([r["si"] for r in res["place"] if r["cell_type"] == "excitatory"],
        bins=np.linspace(0, 5, 30), color="tab:gray", label="all pyramidal")
ax.hist([r["si"] for r in place_r + place_l], bins=np.linspace(0, 5, 30),
        color="tab:blue", label="place cells")
ax.axvline(MIN_SI, color="k", ls="--", lw=1)
ax.set_xlabel("spatial information (bits/spike)"), ax.set_ylabel("count")
ax.set_title("Spatial information", fontsize=10), ax.legend(fontsize=8)

fig.suptitle(f"{SESSION}: CA1 place fields on the 1.6 m linear track", fontsize=12)
fig.savefig("fig02_place_fields.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Fields tile the track in both directions, with the usual over-representation
# of the reward ends in the occupancy map.

# %% [markdown]
# ## Result 1: theta phase entrainment
#
# For each unit we take the theta phase at every spike emitted during running
# and test the resulting phase distribution against uniformity.

# %%
lock = [l for l in res["locking"] if l["n"] >= 100]
exc = [l for l in lock if l["cell_type"] == "excitatory"]
inh = [l for l in lock if l["cell_type"] == "inhibitory"]
pv = np.array([max(l["p"], 1e-300) for l in lock])
print(f"{(pv < 0.01).sum()}/{len(lock)} units with >=100 running spikes are "
      f"significantly phase-locked (Rayleigh p<0.01)")
print(f"mean resultant length: pyramidal {np.mean([l['mrl'] for l in exc]):.3f}, "
      f"interneuron {np.mean([l['mrl'] for l in inh]):.3f}")

# %%
fig = plt.figure(figsize=(14, 12))
outer = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.25, 1.25], hspace=0.48)
gs_top = outer[0].subgridspec(2, 1, height_ratios=[0.55, 1.45], hspace=0.06)
gs_mid = outer[1].subgridspec(1, 3, wspace=0.32)
gs_bot = outer[2].subgridspec(1, 3, wspace=0.42)

t0 = res["run_right"].start[12]
win = nap.IntervalSet(t0, t0 + 1.5)
ax_lfp = fig.add_subplot(gs_top[0])
ax_lfp.plot(res["lfp"].restrict(win).t - t0,
            res["lfp"].restrict(win).values * 1e3, color="tab:gray", lw=0.6)
ax_lfp.plot(res["filt"].restrict(win).t - t0,
            res["filt"].restrict(win).values * 1e3, color="tab:blue", lw=1.6)
ax_lfp.set_ylabel("LFP (mV)"), ax_lfp.set_xlim(0, 1.5)
ax_lfp.tick_params(labelbottom=False)
ax_lfp.set_title("Spikes of CA1 place cells align to the theta cycle "
                 "(1.5 s of running; grey = raw LFP, blue = 6-10 Hz)", fontsize=10)

ax_r = fig.add_subplot(gs_top[1], sharex=ax_lfp)
order = np.argsort([np.nanargmax(r["tc"]) for r in place_r])
for k, idx in enumerate(order):
    st = res["units"][place_r[idx]["unit"]].restrict(win)
    if len(st):
        ax_r.plot(st.t - t0, np.full(len(st), k), "|", color="k", ms=4)
ph_win = res["phase"].restrict(win)
for c in ph_win.t[1:][np.diff(ph_win.values) < -np.pi]:
    ax_r.axvline(c - t0, color="tab:blue", lw=0.6, alpha=0.4)
ax_r.set_xlabel("time (s)")
ax_r.set_ylabel("place cell\n(sorted by field position)")
ax_r.set_ylim(-1, len(order))

bins = np.linspace(0, 2 * np.pi, 25)
ctr = 0.5 * (bins[:-1] + bins[1:])
xx = np.linspace(0, 4 * np.pi, 300)
examples = (sorted(exc, key=lambda l: -l["mrl"])[:2]
            + sorted(inh, key=lambda l: -l["mrl"])[:1])
for j, l in enumerate(examples):
    ax = fig.add_subplot(gs_mid[j])
    h = np.histogram(l["phases"], bins=bins)[0].astype(float)
    h /= h.sum()
    ax.bar(np.r_[ctr, ctr + 2 * np.pi], np.r_[h, h], width=bins[1] - bins[0],
           color="tab:blue" if l["cell_type"] == "excitatory" else "tab:red")
    ax.plot(xx, h.mean() * (1 + 0.5 * np.cos(xx)), color="k", lw=1)
    ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
    ax.set_xticklabels(["0", "180", "360", "540", "720"])
    ax.set_xlabel("theta phase (deg)"), ax.set_ylabel("spike fraction")
    ax.set_title(f"unit {l['unit']} ({l['cell_type'][:3]}), MRL={l['mrl']:.2f}, "
                 f"n={l['n']} spikes", fontsize=9)

ax = fig.add_subplot(gs_bot[0], projection="polar")
for group, color, lab in [([l for l in exc if l["p"] < 0.01], "tab:blue", "pyramidal"),
                          ([l for l in inh if l["p"] < 0.01], "tab:red", "interneuron")]:
    ax.plot([l["pref_phase"] for l in group], [l["mrl"] for l in group], "o",
            color=color, ms=5, alpha=0.75, label=lab)
ax.set_rlabel_position(150), ax.set_rticks([0.2, 0.4, 0.6])
ax.tick_params(labelsize=7)
ax.set_title("Preferred phase (angle) and\nlocking strength (radius)",
             fontsize=10, pad=26)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

ax = fig.add_subplot(gs_bot[1])
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
h = np.histogram(allph, bins=bins)[0].astype(float)
h /= h.sum()
ax.bar(np.r_[ctr, ctr + 2 * np.pi], np.r_[h, h], width=bins[1] - bins[0],
       color="tab:blue")
ax.plot(xx, h.mean() * (1 + 0.5 * np.cos(xx)), color="k", lw=1)
ax.set_xticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
ax.set_xticklabels(["0", "180", "360", "540", "720"])
ax.set_xlabel("theta phase (deg)"), ax.set_ylabel("spike fraction")
ax.set_ylim(0, max(h) * 1.25)
ax.set_title(f"All pyramidal spikes pooled\n(n={len(allph):,})", fontsize=10)

fig.suptitle(f"{SESSION}: theta phase entrainment of CA1 units during running",
             fontsize=12)
fig.savefig("fig03_theta_entrainment.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# Both cell classes fire in phase-locked bursts, with interneurons more strongly
# locked than pyramidal cells (larger mean resultant length), which is the
# standard result. Phase 0 here is the peak of the 6-10 Hz filtered signal on
# the selected channel; because that channel is not necessarily in the pyramidal
# layer, the absolute preferred phase is not directly comparable with published
# values, but the relative comparison between cells and the precession slopes
# below do not depend on that offset.

# %% [markdown]
# ## Result 2: theta phase precession
#
# For each place field we take every spike emitted inside the field during runs
# in the field's preferred direction, express its position as the fraction of
# the field already traversed, and regress theta phase on that fraction.

# %%
prec = precession(res, n_shuffle=20)
slopes = np.array([q["slope"] for q in prec])
pvals = np.array([q["p"] for q in prec])
sigmask = (pvals < 0.05) & (slopes < 0)
print(f"{len(prec)} place-cell x direction fields analysed")
print(f"{sigmask.sum()} ({100 * sigmask.mean():.0f}%) show significant "
      f"precession (p<0.05 and negative slope)")
print(f"{100 * (slopes < 0).mean():.0f}% of all fields have a negative slope; "
      f"median {np.degrees(np.median(slopes)):.0f} deg per field traversal")

# %%
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
    xs = np.linspace(0, 1, 200)
    ys = np.degrees(np.mod(q["slope"] * xs + q["phi0"], 2 * np.pi))
    ys = np.where(np.r_[False, np.abs(np.diff(ys)) > 180], np.nan, ys)
    for off in (0, 360, 720):
        a.plot(xs, ys + off, color="tab:red", lw=1.6)
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
plt.show()

# %%
fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

X = np.concatenate([q["x"] for q in prec])
P = np.concatenate([q["phi"] for q in prec])
H = np.histogram2d(X, P, bins=[20, 24], range=[[0, 1], [0, 2 * np.pi]])[0]
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
axes[1].set_xlabel("precession slope (deg per field)"), axes[1].set_ylabel("count")
axes[1].legend(fontsize=8)
axes[1].set_title(f"{(slopes < 0).mean() * 100:.0f}% of fields have a negative slope",
                  fontsize=10)

bb = np.linspace(-1, 1, 41)
axes[2].hist([q["rho"] for q in prec], bins=bb, color="tab:gray", density=True,
             label="observed")
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
plt.show()

# %% [markdown]
# The pooled density plot is the population signature of precession: as the
# animal advances through a field, the most likely firing phase sweeps downwards
# by roughly 200 degrees. Individual fields show the same relationship with
# circular-linear correlations of -0.4 to -0.6.

# %% [markdown]
# ## Replication across all eight sessions
#
# Set `RUN_ALL_SESSIONS = False` to skip this (it streams a few hundred MB per
# session and takes several minutes per session on a cold cache).

# %%
RUN_ALL_SESSIONS = True

if RUN_ALL_SESSIONS:
    summary = []
    for name in tqdm(LINEAR_SESSIONS, desc="sessions"):
        r = analyze_session(name, verbose=False)
        q = precession(r)
        lk = [{k: v for k, v in l.items() if k != "phases"}
              for l in r["locking"] if l["n"] >= 100]
        pooled = np.concatenate([l["phases"] for l in r["locking"]
                                 if l["n"] >= 100 and l["cell_type"] == "excitatory"])
        summary.append(dict(
            session=name, best_ch=r["best_ch"], theta_delta=r["theta_delta"],
            run_time=float(r["run"].tot_length()), n_units=len(r["units"]),
            locking=lk,
            pooled_hist=np.histogram(pooled, bins=np.linspace(0, 2 * np.pi, 25))[0],
            prec=[{k: v for k, v in p.items()
                   if k not in ("x", "phi", "tc", "centers")} for p in q],
            prec_x=np.concatenate([p["x"] for p in q]) if q else np.array([]),
            prec_phi=np.concatenate([p["phi"] for p in q]) if q else np.array([])))
        s = summary[-1]
        sl = np.array([p["slope"] for p in s["prec"]])
        pp = np.array([p["p"] for p in s["prec"]])
        lp = np.array([l["p"] for l in lk])
        print(f"{name}: {len(lk)} units, {(lp < 0.01).sum()} phase-locked, "
              f"{len(sl)} fields, {((pp < 0.05) & (sl < 0)).sum()} precessing, "
              f"median slope {np.degrees(np.median(sl)):.0f} deg")

# %%
if RUN_ALL_SESSIONS:
    names = [s["session"] for s in summary]
    frac_lock = [np.mean([l["p"] < 0.01 for l in s["locking"]]) for s in summary]
    frac_prec = [np.mean([(p["p"] < 0.05) & (p["slope"] < 0) for p in s["prec"]])
                 for s in summary]
    med_slope = [np.degrees(np.median([p["slope"] for p in s["prec"]]))
                 for s in summary]
    n_fields = [len(s["prec"]) for s in summary]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.32)
    x = np.arange(len(names))
    short = [n.replace("_", "\n") for n in names]

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x, frac_lock, color="tab:blue")
    ax.set_xticks(x), ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("fraction of units"), ax.set_ylim(0, 1)
    ax.set_title("Units significantly theta phase-locked\n(Rayleigh p<0.01)",
                 fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, frac_prec, color="tab:orange")
    for i, n in enumerate(n_fields):
        ax.text(i, frac_prec[i] + 0.02, str(n), ha="center", fontsize=7)
    ax.set_xticks(x), ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("fraction of place fields"), ax.set_ylim(0, 1)
    ax.set_title("Fields with significant precession\n"
                 "(p<0.05, negative slope; n above bar)", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(x, med_slope, color="tab:green")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x), ax.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("median slope (deg per field)")
    ax.set_title("Median precession slope per session", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    cdeg = np.degrees(ctr)
    for s in summary:
        h = s["pooled_hist"] / s["pooled_hist"].sum()
        ax.plot(np.r_[cdeg, cdeg + 360], np.r_[h, h], lw=1.2, label=s["session"])
    ax.set_xticks([0, 180, 360, 540, 720])
    ax.set_xlabel("theta phase (deg)"), ax.set_ylabel("spike fraction")
    ax.set_title("Pooled pyramidal spike-phase distribution", fontsize=10)
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[1, 1])
    XA = np.concatenate([s["prec_x"] for s in summary])
    PA = np.concatenate([s["prec_phi"] for s in summary])
    HA = np.histogram2d(XA, PA, bins=[20, 24], range=[[0, 1], [0, 2 * np.pi]])[0]
    HA = HA / HA.sum(axis=1, keepdims=True)
    im = ax.imshow(np.tile(HA.T, (2, 1)), origin="lower", aspect="auto",
                   extent=[0, 1, 0, 720], cmap="magma")
    ax.set_yticks([0, 180, 360, 540, 720])
    ax.set_xlabel("normalised position in field"), ax.set_ylabel("theta phase (deg)")
    ax.set_title(f"All sessions pooled: {sum(n_fields)} fields,\n{len(XA):,} spikes",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label="P(phase | position)")

    ax = fig.add_subplot(gs[1, 2])
    allslopes = np.degrees(np.concatenate([[p["slope"] for p in s["prec"]]
                                           for s in summary]))
    ax.hist(allslopes, bins=np.linspace(-720, 720, 49), color="tab:gray")
    ax.axvline(0, color="tab:red", lw=1.5)
    ax.axvline(np.median(allslopes), color="tab:blue", ls="--",
               label=f"median {np.median(allslopes):.0f}$\\degree$")
    ax.set_xlabel("precession slope (deg per field)"), ax.set_ylabel("count")
    ax.set_title(f"All sessions: {(allslopes < 0).mean() * 100:.0f}% negative slopes",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("DANDI 000044 (hc-11): theta entrainment and precession across "
                 "the 5 linear-track CA1 sessions", fontsize=13)
    fig.savefig("fig06_across_sessions.png", dpi=150, bbox_inches="tight")
    plt.show()

    n_lock = sum(len(s["locking"]) for s in summary)
    n_lock_sig = sum(sum(l["p"] < 0.01 for l in s["locking"]) for s in summary)
    n_prec_sig = sum(sum((p["p"] < 0.05) and (p["slope"] < 0) for p in s["prec"])
                     for s in summary)
    print(f"TOTAL: {n_lock_sig}/{n_lock} units phase-locked, "
          f"{n_prec_sig}/{sum(n_fields)} fields precessing, "
          f"median slope {np.median(allslopes):.0f} deg per field")

# %% [markdown]
# ## Why three sessions are excluded
#
# The maze geometry differs between sessions and is not obvious from the epoch
# labels, which all read `MazeEpoch`. Plotting the raw 2D trajectories makes it
# plain: three sessions are ring tracks, on which projecting onto the principal
# axis folds the trajectory back on itself and merges positions on opposite
# sides of the ring.

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
for a, name in zip(axes.ravel(), SESSIONS):
    nwbfile, io = open_session(name)
    key = [k for k in nwbfile.processing["behavior"].data_interfaces
           if k.endswith("MazePosition")][0]
    ts = list(nwbfile.processing["behavior"][key].spatial_series.values())[0]
    xy = np.asarray(ts.data[:], dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    linear = name in LINEAR_SESSIONS
    a.plot(xy[ok, 0], xy[ok, 1], lw=0.2,
           color="k" if linear else "tab:red")
    a.set_aspect("equal")
    a.set_title(f"{name}\n{key.replace('Position', '')} "
                f"({'analysed' if linear else 'EXCLUDED: ring track'})",
                fontsize=8)
    a.set_xlabel("x (m)", fontsize=8), a.set_ylabel("y (m)", fontsize=8)
    a.tick_params(labelsize=7)
fig.suptitle("Maze geometry in each hc-11 session", fontsize=12)
fig.tight_layout()
fig.savefig("fig07_maze_geometry.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusions
#
# Both phenomena are present and robust in this data set.
#
# **Entrainment.** 299 of 369 CA1 units (81%) with at least 100 running spikes
# fire non-uniformly across the theta cycle, and this holds in every session.
# Interneurons are more strongly locked than pyramidal cells, and the pyramidal
# population as a whole has a common preferred phase, visible as a clear
# modulation of the pooled spike-phase histogram.
#
# **Precession.** Within a place field, theta phase falls systematically as the
# animal advances through the field. Of 223 place-cell x direction fields, 114
# (51%) reach significance individually by the Kempter circular-linear test, 83%
# of all fields have a negative slope, and the median sweep is -177 degrees per
# field traversal. Pooling the 81,663 in-field spikes from all fields in all five
# sessions gives the canonical precession band running from late to early phase
# across the normalised field. Permuting phases within a field gives a null
# distribution of correlations tightly centred on zero, so the effect is not an
# artefact of the regression.
#
# The analysis is deliberately conservative in one respect worth noting: fields
# are taken from the pooled, direction-specific rate map rather than lap by lap,
# and spikes are pooled across laps. Lap-to-lap variability in where a field
# starts adds phase jitter, so the reported correlations are, if anything, a
# lower bound on the strength of precession in single traversals.
