# %% [markdown]
# # Orientation Selectivity in Mouse Visual Cortex (DANDI:000021)
#
# This notebook demonstrates **orientation selectivity**, the signature response
# property of neurons in the primary visual cortex (V1), first described by
# Hubel and Wiesel (1962): individual cortical neurons respond strongly to
# elongated stimuli of a particular orientation and weakly or not at all to
# orthogonal orientations.
#
# **Dataset**: [DANDI:000021](https://dandiarchive.org/dandiset/000021), the
# Allen Institute *Visual Coding - Neuropixels* dataset (Brain Observatory 1.1).
# Mice viewed full-field drifting gratings (8 drift directions x 5 temporal
# frequencies, 2 s presentations, contrast 0.8) while spiking activity was
# recorded with up to 6 Neuropixels probes spanning visual cortical areas
# (VISp, VISl, VISpm, VISam, VISrl) and other regions.
#
# **Approach**:
# 1. Stream three sessions with LINDI (no full downloads).
# 2. Keep well-isolated ("good") units in visual cortical areas, and drop
#    grating sweeps that overlap the session's `invalid_times` intervals.
# 3. Build a per-sweep spike-count matrix and compute each unit's direction
#    tuning curve at its preferred temporal frequency.
# 4. Quantify orientation selectivity with the global orientation selectivity
#    index (gOSI, 1 - circular variance at twice the angle) and direction
#    selectivity with gDSI.
# 5. Assess significance with a permutation test (shuffle direction labels
#    within temporal frequency; the null recomputes the full statistic,
#    including preferred-TF selection).
#
# Everything runs end-to-end; figures are saved as PNG files.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt
import lindi
import pynapple as nap
from pynwb import NWBHDF5IO
from tqdm import tqdm

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# Three sessions from DANDI:000021 (session-level NWB files, LINDI references on neurosift)
SESSIONS = [
    ("58703c97-c0a9-4736-b684-73c85c1a444a", "ses-715093703"),
    ("02291b99-e583-498b-9929-b68bba2c50e2", "ses-719161530"),
    ("224b57e5-c9a3-46ef-85db-966713f3ccbe", "ses-721123822"),
]
LINDI_TEMPLATE = "https://lindi.neurosift.org/dandi/dandisets/000021/assets/{}/nwb.lindi.json"

VISUAL_AREAS = ["VISp", "VISl", "VISpm", "VISam", "VISrl"]
DIRECTIONS = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
TEMP_FREQS = np.array([1, 2, 4, 8, 15], dtype=float)
DIR_COLORS = plt.cm.hsv(np.linspace(0, 1, 9))[:8]
N_PERM = 500
ALPHA = 0.01
RNG = np.random.default_rng(42)

local_cache = lindi.LocalCache()

# %% [markdown]
# ## Analysis Functions
#
# The pipeline is written as small functions so each stage can be checked
# independently. `analyze_session` streams one session and returns per-unit
# tuning curves, selectivity indices, and permutation-test p-values.

# %%
def load_session(asset_id):
    """Open a session NWB file via LINDI streaming (with local caching)."""
    f = lindi.LindiH5pyFile.from_lindi_file(LINDI_TEMPLATE.format(asset_id), local_cache=local_cache)
    nwbfile = NWBHDF5IO(file=f).read()
    return nwbfile, nap.NWBFile(nwbfile)


def get_grating_table(nwbfile):
    """Read the drifting_gratings_presentations interval table.

    The table has a compound `timeseries` column that breaks `to_dataframe()`
    under LINDI, so columns are read directly. Sweeps overlapping the
    `invalid_times` table (flagged by the Allen SDK, e.g. stimulus display
    problems) are excluded.
    """
    tab = nwbfile.intervals["drifting_gratings_presentations"]
    starts = tab["start_time"].data[:]
    stops = tab["stop_time"].data[:]
    direction = tab["orientation"].data[:]      # drift direction (deg); NaN = blank sweep
    temp_freq = tab["temporal_frequency"].data[:]

    inv = nwbfile.intervals["invalid_times"]
    inv_starts, inv_stops = inv["start_time"].data[:], inv["stop_time"].data[:]
    invalid = np.zeros(len(starts), bool)
    for s, e in zip(inv_starts, inv_stops):
        invalid |= (starts < e) & (stops > s)
    keep = ~invalid
    print(f"  excluded {invalid.sum()}/{len(starts)} sweeps overlapping invalid_times")
    return {
        "starts": starts[keep], "stops": stops[keep],
        "direction": direction[keep], "temp_freq": temp_freq[keep],
    }


def select_visual_units(nwbfile):
    """Boolean mask over the units table: quality == 'good' and peak channel in a visual area."""
    units_tab = nwbfile.units
    quality = units_tab["quality"].data[:]
    peak_ch = units_tab["peak_channel_id"].data[:]
    unit_ids = units_tab["id"].data[:] if hasattr(units_tab["id"], "data") else units_tab.id[:]

    elec_ids = nwbfile.electrodes["id"].data[:] if hasattr(nwbfile.electrodes["id"], "data") else nwbfile.electrodes.id[:]
    elec_loc = nwbfile.electrodes["location"].data[:]
    elec_loc = [x.decode() if isinstance(x, bytes) else str(x) for x in elec_loc]
    id2loc = dict(zip(elec_ids.tolist(), elec_loc))
    unit_loc = np.array([id2loc[int(p)] for p in peak_ch])

    keep = (quality == "good") & np.isin(unit_loc, VISUAL_AREAS)
    return keep, unit_ids, unit_loc


def sweep_count_matrix(ts_group, kept_unit_ids, starts, stops):
    """(n_sweeps, n_units) spike counts, vectorized with searchsorted + np.add.at."""
    n_sweeps = len(starts)
    key_to_col = {k: i for i, k in enumerate(kept_unit_ids)}
    spike_times, spike_unit = [], []
    for k in kept_unit_ids:
        t = ts_group[k].t
        spike_times.append(t)
        spike_unit.append(np.full(len(t), key_to_col[k]))
    spike_times = np.concatenate(spike_times)
    spike_unit = np.concatenate(spike_unit)

    sweep_idx = np.searchsorted(starts, spike_times, side="right") - 1
    valid = (sweep_idx >= 0) & (sweep_idx < n_sweeps)
    valid[valid] &= spike_times[valid] < stops[sweep_idx[valid]]

    counts = np.zeros((n_sweeps, len(kept_unit_ids)))
    np.add.at(counts, (sweep_idx[valid], spike_unit[valid]), 1.0)
    return counts


def direction_tuning(rates, direction, temp_freq):
    """Mean rate per (direction, temporal frequency); tuning at each unit's preferred TF."""
    nonblank = ~np.isnan(direction)
    n_units = rates.shape[1]
    resp = np.zeros((n_units, len(DIRECTIONS), len(TEMP_FREQS)))
    for i, d in enumerate(DIRECTIONS):
        for j, tf in enumerate(TEMP_FREQS):
            m = nonblank & (direction == d) & (temp_freq == tf)
            resp[:, i, j] = rates[m].mean(axis=0)
    pref_tf = resp.mean(axis=1).argmax(axis=1)  # TF with the strongest mean response
    tuning = resp[np.arange(n_units), :, pref_tf]
    return tuning


def selectivity_indices(tuning):
    """gOSI (orientation, 2*theta) and gDSI (direction, theta) via normalized vector sums."""
    theta = np.deg2rad(DIRECTIONS)
    sum_r = tuning.sum(axis=1).astype(float)
    sum_r[sum_r == 0] = np.nan
    gosi = np.abs((tuning * np.exp(2j * theta)).sum(axis=1)) / sum_r
    gdsi = np.abs((tuning * np.exp(1j * theta)).sum(axis=1)) / sum_r
    pref_ori = (np.angle((tuning * np.exp(2j * theta)).sum(axis=1), deg=True) / 2) % 180
    pref_dir = np.angle((tuning * np.exp(1j * theta)).sum(axis=1), deg=True) % 360
    return gosi, gdsi, pref_ori, pref_dir


def permutation_test(rates, direction, temp_freq, gosi, n_perm=N_PERM, rng=RNG):
    """Shuffle direction labels within each temporal frequency.

    The null recomputes the full statistic (responses per direction x TF,
    preferred-TF selection, gOSI) so that selection bias is captured.
    Units with zero evoked response (NaN gOSI) are excluded (p = 1).
    """
    nonblank = ~np.isnan(direction)
    n_units = rates.shape[1]
    valid = ~np.isnan(gosi)
    gosi_null = np.full((n_perm, n_units), np.nan)
    theta = np.deg2rad(DIRECTIONS)
    for p in tqdm(range(n_perm), desc="permutations", leave=False):
        shuffled = direction.copy()
        for tf in TEMP_FREQS:
            m = nonblank & (temp_freq == tf)
            shuffled[m] = rng.permutation(shuffled[m])
        perm_resp = np.zeros((n_units, len(DIRECTIONS), len(TEMP_FREQS)))
        for i, d in enumerate(DIRECTIONS):
            for j, tf in enumerate(TEMP_FREQS):
                m = (shuffled == d) & (temp_freq == tf)
                if m.sum() > 0:
                    perm_resp[:, i, j] = rates[m].mean(axis=0)
        pref_tf_p = perm_resp.mean(axis=1).argmax(axis=1)
        tuning_p = perm_resp[np.arange(n_units), :, pref_tf_p]
        s = tuning_p.sum(axis=1)
        s[s == 0] = np.nan
        gosi_null[p] = np.abs((tuning_p * np.exp(2j * theta)).sum(axis=1)) / s

    p_values = np.ones(n_units)
    p_values[valid] = (np.sum(gosi_null[:, valid] >= gosi[None, valid], axis=0) + 1) / (n_perm + 1)
    return p_values


def analyze_session(asset_id, session_name):
    """Full pipeline for one session. Returns a dict of per-unit results."""
    nwbfile, nwb = load_session(asset_id)
    grat = get_grating_table(nwbfile)
    keep, unit_ids, unit_loc = select_visual_units(nwbfile)
    kept_unit_ids = unit_ids[keep]
    ts_group = nwb["units"]

    counts = sweep_count_matrix(ts_group, kept_unit_ids, grat["starts"], grat["stops"])
    rates = counts / (grat["stops"] - grat["starts"])[:, None]
    blank_rate = rates[np.isnan(grat["direction"])].mean(axis=0)

    tuning = direction_tuning(rates, grat["direction"], grat["temp_freq"])
    gosi, gdsi, pref_ori, pref_dir = selectivity_indices(tuning)
    p_values = permutation_test(rates, grat["direction"], grat["temp_freq"], gosi)
    sig = (p_values < ALPHA) & ~np.isnan(gosi)

    print(f"{session_name}: {keep.sum()} good visual units, "
          f"{sig.sum()} orientation-selective ({100 * sig.mean():.1f}%), "
          f"median gOSI {np.nanmedian(gosi):.3f}")
    return {
        "session": session_name, "tuning": tuning, "gosi": gosi, "gdsi": gdsi,
        "pref_ori": pref_ori, "pref_dir": pref_dir, "p_values": p_values, "sig": sig,
        "blank_rate": blank_rate, "unit_loc": unit_loc[keep], "kept_unit_ids": kept_unit_ids,
        "grat": grat, "ts_group": ts_group,
    }

# %% [markdown]
# ## Run the Pipeline on Three Sessions
#
# Each session is streamed, its good visual-cortex units are extracted, and the
# tuning + permutation analysis is run. The first session is kept for
# single-unit example plots.

# %%
results = []
for asset_id, session_name in SESSIONS:
    print(f"--- {session_name} ---")
    results.append(analyze_session(asset_id, session_name))

# pool units across sessions
tuning_all = np.concatenate([r["tuning"] for r in results])
gosi_all = np.concatenate([r["gosi"] for r in results])
gdsi_all = np.concatenate([r["gdsi"] for r in results])
pref_ori_all = np.concatenate([r["pref_ori"] for r in results])
sig_all = np.concatenate([r["sig"] for r in results])
blank_all = np.concatenate([r["blank_rate"] for r in results])
loc_all = np.concatenate([r["unit_loc"] for r in results])
ses_all = np.concatenate([[r["session"]] * len(r["gosi"]) for r in results])
n_units_all = len(gosi_all)
print(f"\nPooled: {n_units_all} units, {sig_all.sum()} orientation-selective "
      f"({100 * sig_all.mean():.1f}%), median gOSI {np.nanmedian(gosi_all):.3f}")

# %% [markdown]
# ## Figure 1: Raw Data, Example Unit Raster and PSTH
#
# A highly selective VISp unit from the first session. Top: spike raster for
# every drifting-grating sweep, grouped and colored by drift direction (gray
# band = 2 s stimulus). Bottom left: per-direction PSTH. Bottom right: the
# direction tuning curve with the spontaneous (blank-sweep) rate.

# %%
r0 = results[0]
visp_sig = np.where((r0["unit_loc"] == "VISp") & r0["sig"])[0]
ex_local = visp_sig[np.argmax(r0["gosi"][visp_sig])]
ex_uid = r0["kept_unit_ids"][ex_local]
ex_spikes = r0["ts_group"][ex_uid].t
grat = r0["grat"]
starts, stops, direction = grat["starts"], grat["stops"], grat["direction"]

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.3)

ax = fig.add_subplot(gs[0, :])
window = (-0.5, 2.5)
row, yticks, ylabels = 0, [], []
for di, d in enumerate(DIRECTIONS):
    sweeps = np.where(direction == d)[0]
    for s in sweeps:
        t0 = starts[s]
        spk = ex_spikes[(ex_spikes >= t0 + window[0]) & (ex_spikes <= t0 + window[1])] - t0
        ax.plot(spk, np.full(len(spk), row), "|", color=DIR_COLORS[di], ms=3, mew=0.6)
        row += 1
    yticks.append(row - len(sweeps) / 2)
    ylabels.append(f"{int(d)}°")
    ax.axhline(row - 0.5, color="k", lw=0.3, alpha=0.3)
ax.axvspan(0, 2, color="gray", alpha=0.15)
ax.set_yticks(yticks, ylabels)
ax.set_xlim(window)
ax.set_xlabel("Time from grating onset (s)")
ax.set_ylabel("Drift direction")
ax.set_title(f"Example VISp unit {ex_uid}: spike raster across drifting-grating sweeps (gray = 2 s stimulus)")

ax = fig.add_subplot(gs[1, 0])
bin_size = 0.05
edges = np.arange(window[0], window[1] + bin_size, bin_size)
centers = edges[:-1] + bin_size / 2
for di, d in enumerate(DIRECTIONS):
    sweeps = np.where(direction == d)[0]
    all_spk = np.concatenate([
        ex_spikes[(ex_spikes >= starts[s] + window[0]) & (ex_spikes <= starts[s] + window[1])] - starts[s]
        for s in sweeps
    ])
    h, _ = np.histogram(all_spk, bins=edges)
    ax.plot(centers, h / (len(sweeps) * bin_size), color=DIR_COLORS[di], lw=1.2, label=f"{int(d)}°")
ax.axvspan(0, 2, color="gray", alpha=0.15)
ax.set_xlabel("Time from grating onset (s)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("PSTH per drift direction")
ax.legend(fontsize=7, ncol=2, frameon=False)

ax = fig.add_subplot(gs[1, 1])
tc = r0["tuning"][ex_local]
ax.plot(DIRECTIONS, tc, "o-", color="k")
ax.axhline(r0["blank_rate"][ex_local], color="gray", ls="--", lw=1, label="blank (spont.)")
ax.set_xlabel("Drift direction (°)")
ax.set_ylabel("Mean rate (Hz)")
ax.set_xticks(DIRECTIONS)
ax.set_title(f"Direction tuning (gOSI={r0['gosi'][ex_local]:.2f}, gDSI={r0['gdsi'][ex_local]:.2f})")
ax.legend(fontsize=8, frameon=False)
fig.savefig("fig1_example_raster_psth.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig1_example_raster_psth.png")

# %% [markdown]
# ## Figure 2: Example Tuning Curves Across Visual Areas
#
# The most selective significant unit (peak evoked rate >= 2 Hz) from each of
# the five visual cortical areas, pooled across sessions. Top: cartesian
# direction tuning (dashed line = spontaneous rate). Bottom: the same curves
# in polar coordinates. Units tuned to two directions 180° apart are
# orientation-selective but not direction-selective, the classic V1 simple/complex
# cell pattern.

# %%
examples, seen_areas = [], set()
order = np.argsort(np.where(sig_all, gosi_all, -np.inf))[::-1]
peak_rate = tuning_all.max(axis=1)
for i in order:
    a = loc_all[i]
    if a not in seen_areas and peak_rate[i] >= 2.0:
        examples.append(i)
        seen_areas.add(a)
    if len(examples) == 5:
        break

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 5, wspace=0.55, hspace=0.5)
for k, i in enumerate(examples):
    tc = tuning_all[i]
    ax = fig.add_subplot(gs[0, k])
    ax.plot(DIRECTIONS, tc, "o-", color="C0", ms=4)
    ax.axhline(blank_all[i], color="gray", ls="--", lw=0.8)
    ax.set_xticks([0, 180])
    ax.set_xlabel("Direction (°)", fontsize=8)
    if k == 0:
        ax.set_ylabel("Rate (Hz)")
    ax.set_title(f"{loc_all[i]} unit\ngOSI={gosi_all[i]:.2f}", fontsize=9)
    axp = fig.add_subplot(gs[1, k], projection="polar")
    th = np.deg2rad(np.append(DIRECTIONS, 0))
    axp.plot(th, np.append(tc, tc[0]), color="C0")
    axp.fill(th, np.append(tc, tc[0]), color="C0", alpha=0.25)
    axp.set_theta_zero_location("N")
    axp.set_rticks([])
    axp.set_thetagrids([0, 90, 180, 270], labels=["0°", "90°", "180°", "270°"], fontsize=7)
    axp.tick_params(pad=2)
fig.suptitle("Example orientation/direction tuning curves (top: cartesian; bottom: polar)", y=1.0)
fig.savefig("fig2_example_tuning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig2_example_tuning_curves.png")

# %% [markdown]
# ## Figure 3: Population Statistics (Pooled Across Sessions)
#
# - **gOSI distribution**: significant units (permutation test, p < 0.01) in blue.
# - **Preferred orientation**: broad coverage of all orientations.
# - **gOSI vs gDSI**: most selective units fall below the diagonal, i.e. they
#   are orientation- but not direction-selective.
# - **Fraction significant by area**: selectivity is common in all five areas.

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
ax = axes[0]
bins = np.linspace(0, 1, 41)
ax.hist(gosi_all[~sig_all], bins=bins, color="gray", alpha=0.8, label=f"not sig. ({(~sig_all).sum()})")
ax.hist(gosi_all[sig_all], bins=bins, color="C0", alpha=0.8, label=f"sig. p<0.01 ({sig_all.sum()})")
ax.set_xlabel("gOSI")
ax.set_ylabel("Units")
ax.set_title("Orientation selectivity (gOSI)")
ax.legend(fontsize=8, frameon=False)

ax = axes[1]
ax.hist(pref_ori_all[sig_all], bins=np.linspace(0, 180, 19), color="C0")
ax.set_xlabel("Preferred orientation (°)")
ax.set_ylabel("Units")
ax.set_title(f"Preferred orientation (n={sig_all.sum()} sig.)")

ax = axes[2]
ax.scatter(gosi_all[~sig_all], gdsi_all[~sig_all], s=6, color="gray", alpha=0.5, label="not sig.")
ax.scatter(gosi_all[sig_all], gdsi_all[sig_all], s=6, color="C0", alpha=0.6, label="sig. OS")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("gOSI")
ax.set_ylabel("gDSI")
ax.set_title("Orientation vs direction selectivity")
ax.legend(fontsize=8, frameon=False)

ax = axes[3]
fracs = [sig_all[loc_all == a].mean() for a in VISUAL_AREAS]
counts_a = [(loc_all == a).sum() for a in VISUAL_AREAS]
ax.bar(VISUAL_AREAS, fracs, color="C0")
for x, (fr, c) in enumerate(zip(fracs, counts_a)):
    ax.text(x, fr + 0.02, f"n={c}", ha="center", fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("Fraction significant")
ax.set_title("Orientation selectivity by area")
fig.tight_layout()
fig.savefig("fig3_population_stats.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig3_population_stats.png")

# %% [markdown]
# ## Figure 4: Population Tuning Heatmap
#
# Normalized direction tuning of every significantly orientation-selective
# unit, sorted by preferred orientation. The two bright diagonal bands, 180°
# apart, show that most units respond to both drift directions of their
# preferred orientation axis, and that the population tiles all orientations.

# %%
sig_tuning = tuning_all[sig_all]
norm = sig_tuning / np.maximum(sig_tuning.max(axis=1, keepdims=True), 1e-9)
sort_order = np.argsort(pref_ori_all[sig_all])
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(norm[sort_order], aspect="auto", cmap="viridis", extent=[0, 360, sig_all.sum(), 0])
ax.set_xticks(DIRECTIONS)
ax.set_xlabel("Drift direction (°)")
ax.set_ylabel("Units (sorted by preferred orientation)")
ax.set_title(f"Normalized direction tuning, {sig_all.sum()} significantly orientation-selective units")
fig.colorbar(im, ax=ax, label="Normalized rate", shrink=0.8)
fig.savefig("fig4_population_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig4_population_heatmap.png")

# %% [markdown]
# ## Figure 5: Consistency Across Sessions
#
# Per-session summary: number of good visual units, fraction significantly
# orientation-selective, and median gOSI. The effect is stable across
# sessions, so it is not an idiosyncrasy of one animal or probe placement.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
ses_names = [r["session"].replace("ses-", "") for r in results]
n_per_ses = [len(r["gosi"]) for r in results]
frac_per_ses = [r["sig"].mean() for r in results]
med_gosi_per_ses = [np.nanmedian(r["gosi"]) for r in results]
med_gosi_sig = [np.nanmedian(r["gosi"][r["sig"]]) for r in results]

axes[0].bar(ses_names, n_per_ses, color="C0")
axes[0].set_ylabel("Good visual units")
axes[0].set_title("Units per session")
axes[0].set_xlabel("Session")

axes[1].bar(ses_names, frac_per_ses, color="C0")
for x, fr in enumerate(frac_per_ses):
    axes[1].text(x, fr + 0.01, f"{100 * fr:.0f}%", ha="center", fontsize=9)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Fraction significant (p<0.01)")
axes[1].set_title("Orientation-selective fraction")
axes[1].set_xlabel("Session")

x = np.arange(len(results))
axes[2].bar(x - 0.2, med_gosi_per_ses, width=0.4, color="gray", label="all units")
axes[2].bar(x + 0.2, med_gosi_sig, width=0.4, color="C0", label="sig. units")
axes[2].set_xticks(x, ses_names)
axes[2].set_ylim(0, 0.42)
axes[2].set_ylabel("Median gOSI")
axes[2].set_title("Selectivity strength")
axes[2].set_xlabel("Session")
axes[2].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("fig5_cross_session.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved fig5_cross_session.png")

# %% [markdown]
# ## Results
#
# Across three sessions of the Allen Visual Coding dataset we find that a large
# fraction of well-isolated visual-cortex units are robustly tuned to grating
# orientation: their firing rate during the 2 s stimulus depends strongly on
# the grating's drift direction, with tuning curves that peak at one
# orientation axis and fall to near the spontaneous rate at the orthogonal
# orientation. A permutation test that shuffles direction labels (and
# recomputes the full statistic, including selection of each unit's preferred
# temporal frequency) shows this tuning is far stronger than expected by
# chance. Most selective units respond to both drift directions of the same
# orientation axis (gOSI high, gDSI low), the classic orientation-selective,
# direction-insensitive signature of V1 neurons, and preferred orientations
# tile the full 0-180° range. The effect is consistent across sessions and
# across all five recorded visual areas.

# %%
# save aggregated results for reuse
np.savez(
    "orientation_selectivity_results.npz",
    tuning=tuning_all, gosi=gosi_all, gdsi=gdsi_all, pref_ori=pref_ori_all,
    sig=sig_all, blank_rate=blank_all, unit_loc=loc_all, session=ses_all,
)
print("saved orientation_selectivity_results.npz")
