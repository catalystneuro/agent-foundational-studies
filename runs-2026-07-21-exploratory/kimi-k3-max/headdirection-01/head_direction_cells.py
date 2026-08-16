# %% [markdown]
# # Head Direction Cells in the Mouse Anterodorsal Thalamus and Postsubiculum
#
# **Data**: DANDI Archive dataset [000056](https://dandiarchive.org/dandiset/000056),
# Peyrache, Lacroix, Petersen & Buzsáki, *Nature Neuroscience* 2015,
# "Internally organized mechanisms of the head direction sense".
#
# Head direction (HD) cells are neurons that fire selectively when an animal's head
# points in a particular direction in the horizontal plane, independent of where the
# animal is. First described by Ranck (1984) in the rat postsubiculum, they are the
# core of the brain's "neural compass" and are found along the classic HD circuit:
# dorsal tegmental nucleus → lateral mammillary nucleus → anterodorsal thalamus (ADn)
# → postsubiculum → entorhinal cortex.
#
# This notebook demonstrates the phenomenon from raw data:
#
# 1. **Stream one recording session** (Mouse32-140820) from DANDI with LINDI.
# 2. **Reconstruct the head-direction angle** from two head-mounted LEDs.
# 3. **Identify the open-field exploration epoch** automatically from movement.
# 4. **Compute HD tuning curves** and classify HD cells with a shuffle test.
# 5. **Show that preferred directions are stable** across the session (split halves).
# 6. **Decode head direction** from the population with Bayesian decoding.
# 7. **Replicate across 5 sessions** from 5 different mice.
#
# Everything is computed with [Pynapple](https://pynapple.org) on streamed NWB data;
# no files are downloaded in full.

# %% [markdown]
# ## Setup

# %%
import lindi
from pynwb import NWBHDF5IO
import pynapple as nap
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng(0)

DANDISET = "000056"
# Sessions used below: one for the deep dive, five for the population summary.
MAIN_SESSION = "Mouse32-140820"
ASSET_IDS = {
    "Mouse32-140820": "43be9315-c73f-4e83-adea-20e73ce75466",
    "Mouse28-140310": "656704ea-a4cd-40f4-8158-a6533ebf2eee",
    "Mouse25-140124": "43c07029-2ac8-4900-9e51-8c44c3264080",
    "Mouse24-131213": "ada02790-6eb6-48ee-902d-9ba017303586",
    "Mouse20-130514": "748aa5de-c0de-4aa7-a7ef-2aad2f87a7eb",
}
LINDI_TMPL = "https://lindi.neurosift.org/dandi/dandisets/{}/assets/{}/nwb.lindi.json"

NB_BINS = 60
EDGES = np.linspace(-np.pi, np.pi, NB_BINS + 1)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


# %% [markdown]
# ## Helper functions
#
# The same core pipeline is used for the single-session deep dive and the
# multi-session replication, so it is factored into functions here.

# %%
def load_session(asset_id):
    """Stream an NWB session from DANDI via LINDI and wrap it in Pynapple."""
    local_cache = lindi.LocalCache()
    f = lindi.LindiH5pyFile.from_lindi_file(
        LINDI_TMPL.format(DANDISET, asset_id), local_cache=local_cache)
    io = NWBHDF5IO(file=f)
    nwbfile = io.read()
    return nap.NWBFile(nwbfile)


def compute_behavior(nwb):
    """Head direction and speed from the two head-mounted LEDs.

    The NWB file stores the (x, y) position of a red and a blue LED on the
    headstage at 39.0625 Hz; -1 marks failed detection. The head-direction
    angle is the angle of the blue->red LED vector. Speed is computed from
    the smoothed midpoint trajectory.
    """
    red, blue = nwb["RedLED"], nwb["BlueLED"]
    t = red.t
    red_xy = np.asarray(red.values)
    blue_xy = np.asarray(blue.values)
    valid = ((red_xy > 0).all(axis=1) & (blue_xy > 0).all(axis=1))

    hd = np.arctan2(red_xy[:, 1] - blue_xy[:, 1],
                    red_xy[:, 0] - blue_xy[:, 0])
    hd[~valid] = np.nan

    cx = (red_xy[:, 0] + blue_xy[:, 0]) / 2
    cy = (red_xy[:, 1] + blue_xy[:, 1]) / 2
    cx[~valid] = np.nan
    cy[~valid] = np.nan

    dt = np.median(np.diff(t))

    def interp_nan(x):
        isn = np.isnan(x)
        return np.interp(t, t[~isn], x[~isn])

    # 1 s boxcar smoothing before differentiating, to suppress pixel jitter
    win = int(round(1.0 / dt))
    kernel = np.ones(win) / win
    sx = np.convolve(interp_nan(cx), kernel, mode="same")
    sy = np.convolve(interp_nan(cy), kernel, mode="same")
    speed = np.sqrt(np.gradient(sx, dt) ** 2 + np.gradient(sy, dt) ** 2)
    return t, hd, cx, cy, speed, valid


def detect_openfield_epoch(t, cx, cy, speed, valid,
                           speed_thresh=3.0, gap_s=60, min_dur_s=300,
                           min_extent_px=150):
    """Find long active periods that span the open field.

    Active = smoothed speed above threshold; runs separated by < gap_s are
    merged; runs must last >= min_dur_s and cover >= min_extent_px in both
    x and y (this rejects movement inside the small rest box).
    """
    active = speed > speed_thresh
    d = np.diff(active.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if active[0]:
        starts = [0] + starts
    if active[-1]:
        ends = ends + [len(active)]
    merged = []
    for s, e in zip(starts, ends):
        if merged and t[s] - t[merged[-1][1]] < gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        a, b = t[s], t[e - 1]
        if b - a < min_dur_s:
            continue
        m = (t >= a) & (t < b) & valid
        if m.sum() < 1000:
            continue
        ext = min(np.nanmax(cx[m]) - np.nanmin(cx[m]),
                  np.nanmax(cy[m]) - np.nanmin(cy[m]))
        if ext > min_extent_px:
            out.append((a, b))
    return out


def mean_vector_length(tc_vals):
    """Mean resultant length of a tuning curve over CENTERS."""
    tot = np.nansum(tc_vals)
    if tot <= 0:
        return np.nan
    w = tc_vals / tot
    return np.abs(np.nansum(w * np.exp(1j * CENTERS)))


def preferred_direction(tc_vals):
    if np.all(np.isnan(tc_vals)) or np.nansum(tc_vals) <= 0:
        return np.nan
    return CENTERS[np.nanargmax(tc_vals)]


def circ_corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return np.corrcoef(a[m], b[m])[0, 1]


def shuffle_threshold(spikes, hd_t, hd_v, wake_ep, n_shuf=200, seed=0):
    """Per-unit 99th-percentile MVL under circular time shifts of spikes."""
    rng = np.random.default_rng(seed)
    occ, _ = np.histogram(hd_v, bins=EDGES)
    occ_t = occ * np.median(np.diff(hd_t))
    cols = np.array(list(spikes.keys()))
    mvl_shuf = np.full((len(cols), n_shuf), np.nan)
    bounds = list(zip(wake_ep.start, wake_ep.end))
    for j, c in enumerate(cols):
        spk = spikes[c].restrict(wake_ep)
        if len(spk) < 100:
            continue
        spk_t = spk.t
        for s_i in range(n_shuf):
            sh_ang = np.empty(0)
            for (a, b) in bounds:
                m = (spk_t >= a) & (spk_t < b)
                st = spk_t[m] - a
                L = b - a
                shift = rng.uniform(20, L - 20)
                st2 = (st + shift) % L + a
                sh_ang = np.concatenate([sh_ang, np.interp(st2, hd_t, hd_v)])
            counts, _ = np.histogram(sh_ang, bins=EDGES)
            mvl_shuf[j, s_i] = mean_vector_length(counts / np.maximum(occ_t, 1e-9))
    return np.nanpercentile(mvl_shuf, 99, axis=1)


# %% [markdown]
# ## Load the main session and reconstruct head direction
#
# Mouse32-140820 is a ~2.7 h recording: the mouse rests in a small box, then
# forages in a 53 x 46 cm open field, then rests again. The file contains 65
# sorted units (42 with spikes), the two LED position streams, LFP, and ripple
# annotations.

# %%
nwb = load_session(ASSET_IDS[MAIN_SESSION])
print(nwb)

t, hd, cx, cy, speed, valid = compute_behavior(nwb)
print(f"tracking: {len(t)} samples at {1/np.median(np.diff(t)):.2f} Hz, "
      f"{100*valid.mean():.1f}% valid")

hd_tsd = nap.Tsd(t=t, d=hd, time_support=nap.IntervalSet(0, t[-1]))

units = nwb["units"]
keep = [i for i in range(len(units)) if len(units[i]) > 0]
spikes = nap.TsGroup({i: units[i] for i in keep},
                     time_support=nap.IntervalSet(0, t[-1]))
print(f"{len(spikes)} units with spikes")

# %% [markdown]
# ## Identify the open-field exploration epoch
#
# The session mixes sleep/rest (animal nearly still in a corner or a small box)
# with open-field foraging. We detect long active periods and keep only those
# spanning the full arena.

# %%
openfield = detect_openfield_epoch(t, cx, cy, speed, valid)
print("open-field epoch(s):", [(round(a, 1), round(b, 1)) for a, b in openfield])
a0, b0 = openfield[0]
wake_ep = nap.IntervalSet(start=a0, end=b0)
print(f"using {a0:.0f}-{b0:.0f} s ({(b0-a0)/60:.1f} min)")

# %% [markdown]
# ### Figure 1: session overview
#
# Speed and head-direction angle over the whole session with the detected
# epoch highlighted, plus position coverage and HD occupancy within it.

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.35, wspace=0.25)

ax = fig.add_subplot(gs[0, 0])
ax.plot(t, speed, lw=0.2, color="0.4")
ax.axvspan(a0, b0, color="tab:green", alpha=0.15, label="detected open-field epoch")
ax.set_yscale("log")
ax.set_ylim(1e-2, 3e3)
ax.set_ylabel("head speed (px/s, log)")
ax.set_xlabel("time (s)")
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Full recording session: movement speed")

ax = fig.add_subplot(gs[1, 0])
ax.plot(t, np.degrees(hd), ",", color="0.6", rasterized=True)
m = (t >= a0) & (t <= b0)
ax.plot(t[m], np.degrees(hd[m]), ",", color="tab:blue", rasterized=True)
ax.set_ylabel("head direction (deg)")
ax.set_xlabel("time (s)")
ax.set_ylim(-180, 180)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.set_title("Head-direction angle (blue = open-field epoch)")

ax = fig.add_subplot(gs[0, 1])
ax.plot(cx[m], cy[m], ",", color="tab:blue", alpha=0.5, rasterized=True)
ax.set_aspect("equal")
ax.set_title("Position, open-field epoch")
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
ax.invert_yaxis()

ax = fig.add_subplot(gs[1, 1])
occ, edges_occ = np.histogram(hd[m & valid], bins=36, range=(-np.pi, np.pi))
theta = np.degrees((edges_occ[:-1] + edges_occ[1:]) / 2)
ax.bar(theta, occ / 39.0625, width=9, color="tab:blue", edgecolor="none")
ax.set_xlabel("head direction (deg)")
ax.set_ylabel("occupancy (s)")
ax.set_title("HD occupancy")
ax.set_xlim(-180, 180)

fig.suptitle(f"{MAIN_SESSION} (Peyrache et al. 2015, DANDI 000056): session overview",
             fontsize=13)
plt.savefig("fig1_session_overview.png", dpi=150)
plt.close()
print("saved fig1_session_overview.png")

# %% [markdown]
# ## Head-direction tuning curves
#
# For each unit we compute the firing rate as a function of head direction
# (60 bins of 6 deg) over the exploration epoch, and quantify tuning strength
# with the mean vector length (MVL) of the tuning curve. A unit is called an
# HD cell if its MVL exceeds the 99th percentile of MVLs obtained after
# circularly shifting the spike train in time (500 shuffles), which destroys
# the spike-angle relationship while preserving firing rates and occupancy.

# %%
tc = nap.compute_tuning_curves(spikes, hd_tsd, bins=[EDGES], epochs=wake_ep,
                               return_pandas=True)
cols = np.array(tc.columns)
mvl = np.array([mean_vector_length(tc[c].values) for c in cols])

hd_r = hd_tsd.restrict(wake_ep)
good = ~np.isnan(hd_r.values)
hd_t, hd_v = hd_r.t[good], hd_r.values[good]

thresh = shuffle_threshold(spikes, hd_t, hd_v, wake_ep, n_shuf=500, seed=0)
is_hd = mvl > thresh
print(f"HD cells: {is_hd.sum()} / {len(cols)} "
      f"(MVL > 99th pct of 500 time-shift shuffles)")

hd_ids = cols[is_hd]
order = np.argsort(-mvl[is_hd])
top_ids = hd_ids[order]

# %% [markdown]
# ### Figure 2: spikes of the strongest HD cells on the head-direction trace
#
# The raw phenomenon: as the animal's head sweeps around, each HD cell fires
# only when the head passes through its preferred direction.

# %%
w0, w1 = a0 + 700, a0 + 790
fig, ax = plt.subplots(figsize=(14, 6))
mt = (hd_tsd.t >= w0) & (hd_tsd.t < w1)
ax.plot(hd_tsd.t[mt], np.degrees(hd_tsd.values[mt]), color="0.75", lw=1.2, zorder=1)
cmap = plt.cm.hsv
show = top_ids[:8]
for k, u in enumerate(show):
    spk = spikes[u].restrict(nap.IntervalSet(w0, w1))
    if len(spk) == 0:
        continue
    ang = np.interp(spk.t, hd_tsd.t[mt], hd_tsd.values[mt])
    ax.scatter(spk.t, np.degrees(ang), s=14, color=cmap(k / len(show)),
               label=f"unit {u}", zorder=3, edgecolors="none")
ax.set_ylabel("head direction (deg)")
ax.set_xlabel("time (s)")
ax.set_ylim(-180, 180)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, markerscale=1.5)
ax.set_title("Spikes of the 8 strongest HD cells on the head-direction trace "
             "(90 s of exploration)")
plt.tight_layout()
plt.savefig("fig2_hd_spike_overlay.png", dpi=150)
plt.close()
print("saved fig2_hd_spike_overlay.png")

# %% [markdown]
# ### Figure 3: polar tuning curves of the 12 strongest HD cells

# %%
fig, axes = plt.subplots(3, 4, figsize=(13, 9.5),
                         subplot_kw={"projection": "polar"})
for k, ax in enumerate(axes.flat):
    rank = order[k]
    u = hd_ids[rank]
    v = tc[u].values
    th = np.r_[CENTERS, CENTERS[0] + 2 * np.pi]
    r = np.r_[v, v[0]]
    ax.plot(th, r, color="darkred", lw=2)
    ax.fill(th, r, color="darkred", alpha=0.25)
    ax.set_title(f"unit {u}\nMVL={mvl[is_hd][rank]:.2f}, "
                 f"peak={np.nanmax(v):.0f} Hz, "
                 f"pref={np.degrees(preferred_direction(v)):.0f}°", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_rlim(0, np.nanmax(v) * 1.1)
plt.suptitle("Head-direction tuning curves: 12 strongest HD cells "
             "(open-field epoch)", fontsize=13)
plt.tight_layout()
plt.savefig("fig3_tuning_curves.png", dpi=150)
plt.close()
print("saved fig3_tuning_curves.png")

# %% [markdown]
# ### Figure 4: population summary
#
# The MVL distribution is bimodal: a tuned population well above the shuffle
# threshold and an untuned one near zero. Preferred directions of the HD cells
# tile the whole circle, so the population represents every heading.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.hist(mvl[~is_hd], bins=20, range=(0, 1), color="0.6",
        label=f"non-HD ({(~is_hd).sum()})")
ax.hist(mvl[is_hd], bins=20, range=(0, 1), color="darkred",
        label=f"HD ({is_hd.sum()})")
ax.axvline(np.nanmedian(thresh), color="black", ls="--", lw=1.5,
           label="median 99% shuffle threshold")
ax.set_xlabel("mean vector length")
ax.set_ylabel("# units")
ax.legend(fontsize=9)
ax.set_title("Directional tuning strength")

ax = axes[1]
prefs = np.array([preferred_direction(tc[c].values) for c in cols])
ax.hist(np.degrees(prefs[is_hd]), bins=18, range=(-180, 180), color="darkred")
ax.set_xlabel("preferred direction (deg)")
ax.set_ylabel("# HD cells")
ax.set_title("Preferred directions tile the circle")
ax.set_xlim(-180, 180)

ax = axes[2]
peaks = np.array([np.nanmax(tc[c].values) for c in cols])
ax.hist(peaks[is_hd], bins=15, color="darkred")
ax.set_xlabel("peak firing rate (Hz)")
ax.set_ylabel("# HD cells")
ax.set_title("Peak rates of HD cells")

plt.suptitle(f"Population summary, {MAIN_SESSION}", fontsize=13)
plt.tight_layout()
plt.savefig("fig4_population_stats.png", dpi=150)
plt.close()
print("saved fig4_population_stats.png")

# %% [markdown]
# ## Stability of the preferred direction
#
# A defining property of HD cells is that the preferred direction is anchored
# to the environment and stable over time. Splitting the exploration epoch in
# half, HD cells keep nearly the same preferred direction (and their tuning
# curves correlate strongly), while untuned units drift arbitrarily.

# %%
mid = (a0 + b0) / 2
half1 = nap.IntervalSet(start=a0, end=mid)
half2 = nap.IntervalSet(start=mid, end=b0)
tc_h1 = nap.compute_tuning_curves(spikes, hd_tsd, bins=[EDGES], epochs=half1,
                                  return_pandas=True)
tc_h2 = nap.compute_tuning_curves(spikes, hd_tsd, bins=[EDGES], epochs=half2,
                                  return_pandas=True)
pref_h1 = np.array([preferred_direction(tc_h1[c].values) for c in cols])
pref_h2 = np.array([preferred_direction(tc_h2[c].values) for c in cols])
dpref = (pref_h2 - pref_h1 + np.pi) % (2 * np.pi) - np.pi
corr_h = np.array([circ_corr(tc_h1[c].values, tc_h2[c].values) for c in cols])

print("HD cells:     median |d pref| = %.1f deg, median split-half r = %.2f"
      % (np.degrees(np.nanmedian(np.abs(dpref[is_hd]))),
         np.nanmedian(corr_h[is_hd])))
print("non-HD units: median |d pref| = %.1f deg, median split-half r = %.2f"
      % (np.degrees(np.nanmedian(np.abs(dpref[~is_hd]))),
         np.nanmedian(corr_h[~is_hd])))

# %% [markdown]
# ### Figure 5: split-half stability

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.hist(np.degrees(dpref[is_hd]), bins=18, range=(-180, 180), color="darkred",
        label="HD", alpha=0.9)
ax.hist(np.degrees(dpref[~is_hd]), bins=18, range=(-180, 180), color="0.6",
        label="non-HD", alpha=0.7)
ax.set_xlabel("Δ preferred direction, 1st vs 2nd half (deg)")
ax.set_ylabel("# units")
ax.legend(fontsize=9)
ax.set_title("Preferred-direction stability")
ax.set_xlim(-180, 180)

ax = axes[1]
ax.hist(corr_h[is_hd], bins=15, range=(-1, 1), color="darkred", label="HD",
        alpha=0.9)
ax.hist(corr_h[~is_hd], bins=15, range=(-1, 1), color="0.6", label="non-HD",
        alpha=0.7)
ax.set_xlabel("split-half tuning-curve correlation")
ax.set_ylabel("# units")
ax.legend(fontsize=9)
ax.set_title("Tuning-curve stability")

ax = axes[2]
ax.scatter(np.abs(np.degrees(dpref[is_hd])), corr_h[is_hd], color="darkred",
           label="HD")
ax.scatter(np.abs(np.degrees(dpref[~is_hd])), corr_h[~is_hd], color="0.6",
           label="non-HD")
ax.set_xlabel("|Δ preferred direction| (deg)")
ax.set_ylabel("split-half correlation")
ax.legend(fontsize=9)
ax.set_title("Stability metrics agree")

plt.suptitle("HD cells are stable across the first and second half of exploration",
             fontsize=13)
plt.tight_layout()
plt.savefig("fig5_stability.png", dpi=150)
plt.close()
print("saved fig5_stability.png")

# %% [markdown]
# ## Bayesian decoding of head direction
#
# If the population genuinely encodes heading, the angle should be readable
# from spikes alone. We train a Bayesian decoder (Zhang et al. 1998) on the
# tuning curves from the first half of exploration and decode the second half
# in 100 ms bins, using only the 21 HD cells.

# %%
spikes_hd = spikes[hd_ids.tolist()]
tc_train = nap.compute_tuning_curves(spikes_hd, hd_tsd, bins=[EDGES],
                                     epochs=half1)
decoded, proba = nap.decode_bayes(tc_train, spikes_hd, half2, bin_size=0.1,
                                  time_units="s", uniform_prior=True)
actual = np.interp(decoded.t, hd_t, hd_v)
dec = decoded.values
err = (dec - actual + np.pi) % (2 * np.pi) - np.pi
ok = ~np.isnan(err)

err_chance = (rng.uniform(-np.pi, np.pi, ok.sum()) - actual[ok]
              + np.pi) % (2 * np.pi) - np.pi
print("decoded:  median |err| = %.1f deg, P(within 30 deg) = %.2f"
      % (np.degrees(np.nanmedian(np.abs(err))),
         np.mean(np.abs(err[ok]) < np.radians(30))))
print("chance:   median |err| = %.1f deg, P(within 30 deg) = %.2f"
      % (np.degrees(np.median(np.abs(err_chance))),
         np.mean(np.abs(err_chance) < np.radians(30))))

# %% [markdown]
# ### Figure 6: decoding performance

# %%
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
w0d, w1d = mid + 500, mid + 560
md = (decoded.t >= w0d) & (decoded.t < w1d)
mt = (hd_tsd.t >= w0d) & (hd_tsd.t < w1d)
ax.plot(hd_tsd.t[mt], np.degrees(hd_tsd.values[mt]), ".", ms=3, color="0.6",
        label="actual")
ax.plot(decoded.t[md], np.degrees(dec[md]), ".", ms=3, color="crimson",
        label="decoded (100 ms bins)")
ax.set_ylabel("head direction (deg)")
ax.set_xlabel("time (s)")
ax.set_ylim(-180, 180)
ax.set_yticks([-180, -90, 0, 90, 180])
ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"Bayesian decoding from {len(hd_ids)} HD cells "
             "(trained on first half of exploration)")

ax = fig.add_subplot(gs[1, 0])
ax.hist(np.degrees(err[ok]), bins=72, range=(-180, 180), color="crimson",
        density=True, label="decoded", alpha=0.85)
ax.hist(np.degrees(err_chance), bins=72, range=(-180, 180), color="0.6",
        density=True, histtype="step", lw=1.5, label="chance")
ax.set_xlabel("decoding error (deg)")
ax.set_ylabel("density")
ax.legend(fontsize=9)
ax.set_title(f"Error distribution (median |err| = "
             f"{np.degrees(np.nanmedian(np.abs(err))):.1f}°)")
ax.set_xlim(-180, 180)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(np.degrees(actual[ok]), np.degrees(dec[ok]), s=1, color="crimson",
           alpha=0.15, rasterized=True)
ax.plot([-180, 180], [-180, 180], "k--", lw=1)
ax.set_xlabel("actual HD (deg)")
ax.set_ylabel("decoded HD (deg)")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_aspect("equal")
ax.set_title("Decoded vs actual")

plt.suptitle("Population decoding of head direction", fontsize=13)
plt.savefig("fig6_decoding.png", dpi=150)
plt.close()
print("saved fig6_decoding.png")

# %% [markdown]
# ## Replication across sessions
#
# The same pipeline (epoch detection, tuning curves, shuffle classification,
# split-half stability) is run on four more sessions from four other mice.
# This takes a few minutes because each session is streamed and shuffled.

# %%
def process_session(name, asset_id, n_shuf=200, seed=0):
    nwb = load_session(asset_id)
    t, hd, cx, cy, speed, valid = compute_behavior(nwb)
    openfield = detect_openfield_epoch(t, cx, cy, speed, valid)
    if not openfield:
        print(f"  {name}: no open-field epoch found, skipped")
        return None
    a0, b0 = openfield[0]
    wake_ep = nap.IntervalSet(start=a0, end=b0)
    hd_tsd = nap.Tsd(t=t, d=hd, time_support=nap.IntervalSet(0, t[-1]))
    units = nwb["units"]
    keep = [i for i in range(len(units)) if len(units[i]) > 0]
    spikes = nap.TsGroup({i: units[i] for i in keep},
                         time_support=nap.IntervalSet(0, t[-1]))

    tc = nap.compute_tuning_curves(spikes, hd_tsd, bins=[EDGES], epochs=wake_ep,
                                   return_pandas=True)
    cols = np.array(tc.columns)
    mvl = np.array([mean_vector_length(tc[c].values) for c in cols])

    hd_r = hd_tsd.restrict(wake_ep)
    good = ~np.isnan(hd_r.values)
    thresh = shuffle_threshold(spikes, hd_r.t[good], hd_r.values[good],
                               wake_ep, n_shuf=n_shuf, seed=seed)
    is_hd = mvl > thresh

    mid = (a0 + b0) / 2
    tc_h1 = nap.compute_tuning_curves(spikes, hd_tsd, bins=[EDGES],
                                      epochs=nap.IntervalSet(a0, mid),
                                      return_pandas=True)
    tc_h2 = nap.compute_tuning_curves(spikes, hd_tsd, bins=[EDGES],
                                      epochs=nap.IntervalSet(mid, b0),
                                      return_pandas=True)
    pref_h1 = np.array([preferred_direction(tc_h1[c].values) for c in cols])
    pref_h2 = np.array([preferred_direction(tc_h2[c].values) for c in cols])
    dpref = (pref_h2 - pref_h1 + np.pi) % (2 * np.pi) - np.pi
    corr_h = np.array([circ_corr(tc_h1[c].values, tc_h2[c].values) for c in cols])
    pref = np.array([preferred_direction(tc[c].values) for c in cols])

    print(f"  {name}: epoch {a0:.0f}-{b0:.0f} s, {len(cols)} units, "
          f"{is_hd.sum()} HD cells ({100*is_hd.mean():.0f}%)")
    return dict(name=name, mvl=mvl, is_hd=is_hd, dpref=dpref,
                corr_h=corr_h, pref=pref)


session_results = []
for name, aid in tqdm(ASSET_IDS.items(), desc="sessions"):
    print(f"Processing {name} ...")
    r = process_session(name, aid)
    if r is not None:
        session_results.append(r)

# %% [markdown]
# ### Figure 7: pooled population statistics across 5 mice

# %%
all_mvl = np.concatenate([r["mvl"] for r in session_results])
all_hd = np.concatenate([r["is_hd"] for r in session_results])
hd_dpref = np.concatenate([r["dpref"][r["is_hd"]] for r in session_results])
non_dpref = np.concatenate([r["dpref"][~r["is_hd"]] for r in session_results])
hd_corr = np.concatenate([r["corr_h"][r["is_hd"]] for r in session_results])
non_corr = np.concatenate([r["corr_h"][~r["is_hd"]] for r in session_results])
hd_pref = np.concatenate([r["pref"][r["is_hd"]] for r in session_results])
frac_per_session = [r["is_hd"].mean() for r in session_results]
names = [r["name"] for r in session_results]

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

ax = axes[0, 0]
ax.bar(range(len(names)), [100 * f for f in frac_per_session], color="darkred")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("-", "\n") for n in names], fontsize=8)
ax.set_ylabel("% HD cells")
ax.set_title("HD cell fraction per session")
ax.set_ylim(0, 100)

ax = axes[0, 1]
ax.hist(all_mvl[~all_hd], bins=25, range=(0, 1), color="0.6",
        label=f"non-HD ({(~all_hd).sum()})")
ax.hist(all_mvl[all_hd], bins=25, range=(0, 1), color="darkred",
        label=f"HD ({all_hd.sum()})")
ax.set_xlabel("mean vector length")
ax.set_ylabel("# units")
ax.legend(fontsize=9)
ax.set_title("Tuning strength, all sessions pooled")

ax = axes[0, 2]
ax.hist(np.degrees(hd_pref), bins=18, range=(-180, 180), color="darkred")
ax.set_xlabel("preferred direction (deg)")
ax.set_ylabel("# HD cells")
ax.set_title("Preferred directions, pooled")
ax.set_xlim(-180, 180)

ax = axes[1, 0]
ax.hist(np.degrees(hd_dpref), bins=18, range=(-180, 180), color="darkred",
        label="HD", alpha=0.9)
ax.hist(np.degrees(non_dpref), bins=18, range=(-180, 180), color="0.6",
        label="non-HD", alpha=0.7)
ax.set_xlabel("Δ preferred direction, split-half (deg)")
ax.set_ylabel("# units")
ax.legend(fontsize=9)
ax.set_title("Preferred-direction stability, pooled")
ax.set_xlim(-180, 180)

ax = axes[1, 1]
parts = ax.violinplot([hd_corr[~np.isnan(hd_corr)], non_corr[~np.isnan(non_corr)]],
                      showmedians=True)
for pc, c in zip(parts["bodies"], ["darkred", "0.6"]):
    pc.set_facecolor(c)
ax.set_xticks([1, 2])
ax.set_xticklabels(["HD", "non-HD"])
ax.set_ylabel("split-half tuning-curve correlation")
ax.set_title("Tuning stability, pooled")

ax = axes[1, 2]
ax.scatter(np.abs(np.degrees(hd_dpref)), hd_corr, color="darkred", s=12,
           alpha=0.6, label="HD")
ax.scatter(np.abs(np.degrees(non_dpref)), non_corr, color="0.6", s=12,
           alpha=0.6, label="non-HD")
ax.set_xlabel("|Δ preferred direction| (deg)")
ax.set_ylabel("split-half correlation")
ax.legend(fontsize=9)
ax.set_title("Stability metrics, pooled")

plt.suptitle(f"Head direction cells across {len(names)} sessions "
             f"({all_hd.sum()} HD cells of {len(all_mvl)} units)", fontsize=13)
plt.tight_layout()
plt.savefig("fig7_multisession.png", dpi=150)
plt.close()
print("saved fig7_multisession.png")

# %% [markdown]
# ## Summary
#
# - Streaming the Peyrache et al. (2015) dataset from DANDI (000056), we
#   reconstructed head direction from the two head-mounted LEDs and isolated
#   the open-field foraging epoch automatically.
# - In Mouse32-140820, 21 of 42 units are significantly direction-tuned
#   (mean vector length above the 99th percentile of 500 circular time-shift
#   shuffles). Their tuning curves are unimodal with preferred directions
#   tiling the full circle: the classic head direction cell signature.
# - Preferred directions are stable within the environment (split-half
#   correlation ~0.9 for HD cells vs ~0.1 for untuned units).
# - A Bayesian decoder trained on the first half of exploration reads out head
#   direction from 100 ms population vectors with a median error of ~11 deg,
#   far better than chance (~89 deg).
# - The result replicates across 5 sessions from 5 mice: a large fraction of
#   recorded units in the ADn/PoSub region are HD cells with stable,
#   circle-tiling preferred directions.
