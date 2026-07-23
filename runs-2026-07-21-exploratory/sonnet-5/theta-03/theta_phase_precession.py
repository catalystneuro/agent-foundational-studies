# %% [markdown]
# # Theta Phase Entrainment and Precession in Hippocampal Place Cells
#
# This notebook demonstrates two classic properties of the rodent hippocampal theta
# rhythm using real extracellular recordings from the DANDI Archive:
#
# 1. **Theta phase entrainment**: hippocampal pyramidal cells fire preferentially at a
#    particular phase of the ongoing 6-10 Hz theta oscillation in the local field
#    potential (LFP).
# 2. **Theta phase precession**: as a rat runs through the place field of a CA1 place
#    cell, the theta phase at which the cell fires systematically shifts from late to
#    early phase (O'Keefe & Recce, 1993).
#
# ## Dataset
#
# We use **DANDI:000044** ("Diversity in neural firing dynamics supports both rigid
# and learned hippocampal sequences", Grosmark & Buzsaki), session
# `sub-Achilles_ses-Achilles-10252013`. This session contains simultaneous:
# - 128-channel LFP recorded at 1250 Hz from CA1 silicon probes,
# - 137 spike-sorted units (120 excitatory, 17 inhibitory) with cell-type labels,
# - linearized position on a 1.6 m linear track (`MazeEpoch`, ~34.5 minutes).
#
# Data is streamed directly from the DANDI S3 bucket with `remfile` (no full download).

# %%
import h5py
import remfile
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import pynapple as nap

nap.nap_config.suppress_conversion_warnings = True
np.random.seed(0)

DANDISET_ID = "000044"
ASSET_PATH = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
ASSET_URL = (
    "https://api.dandiarchive.org/api/dandisets/000044/versions/draft/assets/"
    "5349c68b-c0a7-46c0-9900-cda050722fa4/download/"
)

# %% [markdown]
# ## 1. Load the NWB File (streaming, cached to disk)

# %%
disk_cache = remfile.DiskCache("/tmp/remfile_cache")
rem_file = remfile.File(ASSET_URL, disk_cache=disk_cache)
h5f = h5py.File(rem_file, "r")
io = NWBHDF5IO(file=h5f, load_namespaces=True)
nwbfile = io.read()
print(nwbfile)

# %% [markdown]
# ## 2. Extract Epochs, Units, and Position
#
# The session has three experimental epochs: `PREEpoch` (pre-task sleep/rest),
# `MazeEpoch` (linear-track running), and `POSTEpoch` (post-task sleep/rest). We
# restrict all analysis to `MazeEpoch`.
#
# The linearized position `SpatialSeries` has a metadata bug: its `rate` field
# actually stores the sampling **period** (in seconds), not the sampling rate. We
# recover the true rate as `1 / rate` and verify it against the epoch duration.

# %%
epochs_df = nwbfile.intervals["epochs"].to_dataframe()
print(epochs_df)

maze_start = float(epochs_df.loc[epochs_df.label == "MazeEpoch", "start_time"].iloc[0])
maze_stop = float(epochs_df.loc[epochs_df.label == "MazeEpoch", "stop_time"].iloc[0])
maze_ep = nap.IntervalSet(start=maze_start, end=maze_stop)
print(f"MazeEpoch: {maze_start:.1f} - {maze_stop:.1f} s ({maze_stop - maze_start:.1f} s)")

units_df = nwbfile.units.to_dataframe()
tsgroup = nap.TsGroup({i: units_df.loc[i, "spike_times"] for i in units_df.index})
tsgroup.set_info(
    cell_type=units_df["cell_type"].values,
    location=units_df["location"].values,
    shank_id=units_df["shank_id"].values,
)
print(tsgroup)
print(units_df["cell_type"].value_counts())

# %%
lin_ss = nwbfile.processing["behavior"]["1.6mLinearMazeLinearizedPosition"].spatial_series[
    "1.6mLinearMazeLinearizedTimeSeries"
]
real_rate = 1.0 / lin_ss.rate  # metadata bug: `rate` field stores the sampling period
n_pos = lin_ss.data.shape[0]
pos_t = lin_ss.starting_time + np.arange(n_pos) / real_rate
linpos_vals = lin_ss.data[:, 0]
print(f"Recovered position sampling rate: {real_rate:.3f} Hz")
print(f"Position track duration from timestamps: {n_pos / real_rate:.1f} s "
      f"(MazeEpoch duration: {maze_stop - maze_start:.1f} s)")

valid = ~np.isnan(linpos_vals)
print(f"Valid (tracked) position samples: {valid.sum()} / {len(valid)} "
      f"({100 * valid.mean():.1f}%) -- gaps correspond to time at the untracked reward wells")

lin_pos = nap.Tsd(t=pos_t[valid], d=linpos_vals[valid], time_support=maze_ep)

# %% [markdown]
# ### Segmenting Track Traversals
#
# The animal is tracked only while actively running the 1.6 m track; position is
# missing while it sits at the reward wells. Each contiguous block of valid samples
# is therefore one track traversal ("trial"). We split trials into **outbound**
# (increasing position) and **inbound** (decreasing position) runs to keep place
# fields direction-specific, as is standard for linear-track place cell analyses.

# %%
d = np.diff(valid.astype(int))
seg_starts = np.where(d == 1)[0] + 1
seg_ends = np.where(d == -1)[0] + 1
if valid[0]:
    seg_starts = np.r_[0, seg_starts]
if valid[-1]:
    seg_ends = np.r_[seg_ends, len(valid)]
seg_lens = seg_ends - seg_starts
keep = seg_lens >= 10  # at least ~0.25 s of continuous tracking
seg_starts, seg_ends = seg_starts[keep], seg_ends[keep]

trial_start_t = pos_t[seg_starts]
trial_end_t = pos_t[seg_ends - 1]
directions = np.array([
    1 if linpos_vals[s:e][-1] > linpos_vals[s:e][0] else -1
    for s, e in zip(seg_starts, seg_ends)
])
trials_ep = nap.IntervalSet(start=trial_start_t, end=trial_end_t)
out_ep = trials_ep[directions == 1]
in_ep = trials_ep[directions == -1]
print(f"{len(trials_ep)} track traversals: {len(out_ep)} outbound, {len(in_ep)} inbound")

# %% [markdown]
# ## 3. Raw Data Overview

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

axes[0].plot(lin_pos.t, lin_pos.values, ".", color="0.3", markersize=2)
for k, (s, e) in enumerate(zip(out_ep.start, out_ep.end)):
    axes[0].axvspan(s, e, color="tab:blue", alpha=0.15, label="outbound" if k == 0 else None)
for k, (s, e) in enumerate(zip(in_ep.start, in_ep.end)):
    axes[0].axvspan(s, e, color="tab:orange", alpha=0.15, label="inbound" if k == 0 else None)
axes[0].set_ylabel("Linearized\nposition (m)")
axes[0].set_title("Linear-track running epoch: tracked position and place-cell raster")
axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

pyr_all = tsgroup.getby_category("cell_type")["excitatory"]
example_ids = list(pyr_all.index[:25])
for i, uid in enumerate(example_ids):
    st = pyr_all[uid].restrict(maze_ep)
    axes[1].plot(st.t, np.full(len(st), i), "|", color="k", markersize=3)
axes[1].set_ylabel("Unit #")
axes[1].set_xlabel("Time (s)")
axes[1].set_xlim(maze_start, maze_start + 300)

plt.tight_layout()
plt.savefig("fig1_raw_data_overview.png", dpi=150)
plt.close()
print("saved fig1_raw_data_overview.png")

# %% [markdown]
# ## 4. Extract Theta-Band LFP Phase
#
# We first pick the LFP channel with the strongest theta rhythm by comparing
# theta-band (6-10 Hz) to delta-band (2-4 Hz) power across all 128 channels over a
# sample of running time, following the standard convention for choosing a "theta
# reference channel".

# %%
lfp_es = nwbfile.processing["ecephys"]["LFP"]["LFP"]
fs = lfp_es.rate
conv = lfp_es.conversion
electrodes_df = nwbfile.electrodes.to_dataframe()
bad = electrodes_df["bad_electrode"].values

scan_start_idx = int(round(maze_start * fs))
scan_n = int(200 * fs)
scan_data = lfp_es.data[scan_start_idx:scan_start_idx + scan_n, :].astype(np.float64) * conv

theta_power = np.full(scan_data.shape[1], np.nan)
delta_power = np.full(scan_data.shape[1], np.nan)
for ch in range(scan_data.shape[1]):
    if bad[ch]:
        continue
    f, pxx = sp.signal.welch(scan_data[:, ch], fs=fs, nperseg=int(fs * 4))
    theta_power[ch] = pxx[(f >= 6) & (f <= 10)].mean()
    delta_power[ch] = pxx[(f >= 2) & (f <= 4)].mean()

theta_delta_ratio = theta_power / delta_power
theta_channel = int(np.nanargmax(theta_delta_ratio))
print(f"Selected theta-reference channel: {theta_channel} "
      f"(group={electrodes_df.iloc[theta_channel]['group_name']}, "
      f"theta/delta ratio={theta_delta_ratio[theta_channel]:.2f})")

# %% [markdown]
# We then load that single channel's LFP continuously across the full `MazeEpoch`
# (filtering over the whole continuous recording, not the short disjoint trial
# segments, avoids edge artifacts from the bandpass filter), bandpass-filter 6-10 Hz,
# and take the Hilbert transform to get the instantaneous theta phase.

# %%
start_idx = int(round(maze_start * fs))
stop_idx = int(round(maze_stop * fs))
lfp_vals = lfp_es.data[start_idx:stop_idx, theta_channel].astype(np.float64) * conv
lfp_t = maze_start + np.arange(len(lfp_vals)) / fs
lfp = nap.Tsd(t=lfp_t, d=lfp_vals, time_support=maze_ep)

theta_band = nap.apply_bandpass_filter(lfp, cutoff=(6, 10), fs=fs)
analytic_phase = np.mod(np.angle(sp.signal.hilbert(theta_band.values)), 2 * np.pi)
theta_phase = nap.Tsd(t=theta_band.t, d=analytic_phase, time_support=maze_ep)

freqs, pxx = sp.signal.welch(lfp.values, fs=fs, nperseg=int(fs * 4))

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 7))

window = (maze_start + 100, maze_start + 104)
raw_win = lfp.restrict(nap.IntervalSet(*window))
theta_win = theta_band.restrict(nap.IntervalSet(*window))
axes[0].plot(raw_win.t, raw_win.values * 1e3, color="0.4", lw=0.8, label="raw LFP")
axes[0].plot(theta_win.t, theta_win.values * 1e3, color="tab:red", lw=1.5, label="6-10 Hz filtered")
axes[0].set_ylabel("LFP (mV)")
axes[0].set_xlabel("Time (s)")
axes[0].set_title(f"Example raw LFP and theta-band filtered trace (channel {theta_channel})")
axes[0].legend(loc="upper right", fontsize=8)

phase_win = theta_phase.restrict(nap.IntervalSet(*window))
ax2 = axes[1]
ax2.plot(theta_win.t, theta_win.values * 1e3, color="tab:red", lw=1.2)
ax2b = ax2.twinx()
ax2b.plot(phase_win.t, phase_win.values, color="tab:blue", lw=1, alpha=0.7)
ax2.set_ylabel("Theta LFP (mV)", color="tab:red")
ax2b.set_ylabel("Theta phase (rad)", color="tab:blue")
ax2.set_xlabel("Time (s)")
ax2.set_title("Theta phase (Hilbert transform) tracks oscillation cycles")

axes[2].semilogy(freqs, pxx)
axes[2].axvspan(6, 10, color="tab:red", alpha=0.2, label="theta band (6-10 Hz)")
axes[2].set_xlim(0, 30)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Power (V²/Hz)")
axes[2].set_title("LFP power spectrum during running shows a clear theta peak")
axes[2].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("fig2_theta_extraction.png", dpi=150)
plt.close()
print("saved fig2_theta_extraction.png")

# %% [markdown]
# ## 5. Identify Place Cells (Outbound Direction)
#
# We compute position tuning curves for all excitatory (pyramidal) units during
# outbound runs and rank cells by spatial information (bits/spike, Skaggs et al.,
# 1993).

# %%
pyr = tsgroup.getby_category("cell_type")["excitatory"]

tc_out = nap.compute_tuning_curves(
    pyr, lin_pos, bins=40, range=[(0.0, 1.6)], epochs=out_ep, feature_names=["position"]
)
occ = tc_out.attrs["occupancy"]
occ_p = occ / occ.sum()
mean_rate = (tc_out.values * occ_p[None, :]).sum(axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = tc_out.values / mean_rate[:, None]
    logterm = np.where(ratio > 0, np.log2(ratio), 0.0)
    spatial_info = np.nansum(occ_p[None, :] * tc_out.values * logterm, axis=1) / mean_rate
spatial_info = np.where(mean_rate > 0, spatial_info, 0.0)

summary = pd.DataFrame({
    "unit": tc_out.coords["unit"].values,
    "mean_rate": mean_rate,
    "peak_rate": tc_out.values.max(axis=1),
    "spatial_info": spatial_info,
})
summary = summary[summary["mean_rate"] > 0.2].sort_values("spatial_info", ascending=False)
print(summary.head(10).to_string(index=False))

# %%
bin_centers = tc_out.coords["position"].values
place_cell_ids = summary["unit"].values
place_maps = tc_out.sel(unit=place_cell_ids).values
norm_maps = place_maps / place_maps.max(axis=1, keepdims=True)
peak_bin = norm_maps.argmax(axis=1)
order = np.argsort(peak_bin)

fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.4, 1]})

im = axes[0].imshow(norm_maps[order], aspect="auto", cmap="viridis",
                     extent=[bin_centers[0], bin_centers[-1], len(order), 0])
axes[0].set_xlabel("Linearized position (m)")
axes[0].set_ylabel("Place cell # (sorted by field location)")
axes[0].set_title(f"Population place fields tile the track (n={len(order)} cells, outbound runs)")
plt.colorbar(im, ax=axes[0], label="normalized firing rate")

top_ids = place_cell_ids[:6]
for uid in top_ids:
    axes[1].plot(bin_centers, tc_out.sel(unit=uid).values, label=f"unit {uid}")
axes[1].set_xlabel("Linearized position (m)")
axes[1].set_ylabel("Firing rate (Hz)")
axes[1].set_title("Example place fields (top 6 by spatial information)")
axes[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig("fig3_place_fields.png", dpi=150)
plt.close()
print("saved fig3_place_fields.png")

# %% [markdown]
# ## 6. Theta Phase Entrainment
#
# For each place cell we take spikes fired during any track traversal (both
# directions) and compute the theta phase at each spike time. We then test for
# non-uniformity of the phase distribution with a Rayleigh test (Zar, 1999,
# asymptotic approximation), and quantify entrainment strength with the mean
# resultant vector length.

# %%
def rayleigh_test(phases):
    n = len(phases)
    C, S = np.sum(np.cos(phases)), np.sum(np.sin(phases))
    R = np.hypot(C, S)
    Rbar = R / n
    Z = n * Rbar ** 2
    p = np.exp(-Z) * (
        1 + (2 * Z - Z ** 2) / (4 * n)
        - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2)
    )
    mean_phase = np.arctan2(S, C) % (2 * np.pi)
    return Rbar, p, mean_phase


entrain_rows = []
for uid in place_cell_ids:
    sp_ts = pyr[uid].restrict(trials_ep)
    if len(sp_ts) < 20:
        continue
    ph = sp_ts.value_from(theta_phase).values
    Rbar, p, mean_phase = rayleigh_test(ph)
    entrain_rows.append(dict(unit=uid, n_spikes=len(sp_ts), Rbar=Rbar, p=p, mean_phase=mean_phase, phases=ph))
entrain_df = pd.DataFrame(entrain_rows).sort_values("Rbar", ascending=False)
n_sig = (entrain_df["p"] < 0.05).sum()
print(f"{n_sig} / {len(entrain_df)} place cells show significant theta phase locking (Rayleigh p<0.05)")
print(entrain_df[["unit", "n_spikes", "Rbar", "p", "mean_phase"]].head(10).to_string(index=False))

# %%
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(2, 3)

example_units = entrain_df["unit"].values[:3]
for i, uid in enumerate(example_units):
    row = entrain_df[entrain_df.unit == uid].iloc[0]
    ax = fig.add_subplot(gs[0, i], projection="polar")
    counts, bin_edges = np.histogram(row["phases"], bins=18, range=(0, 2 * np.pi))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(centers, counts, width=2 * np.pi / 18, color="tab:blue", alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.set_title(f"unit {uid}\nR={row['Rbar']:.2f}, p={row['p']:.1e}", fontsize=9, pad=25)
    ax.set_yticklabels([])

ax_scatter = fig.add_subplot(gs[1, :])
sig = entrain_df["p"] < 0.05
ax_scatter.scatter(entrain_df.loc[~sig, "n_spikes"], entrain_df.loc[~sig, "Rbar"],
                    color="0.6", label=f"not significant (n={(~sig).sum()})", s=25)
ax_scatter.scatter(entrain_df.loc[sig, "n_spikes"], entrain_df.loc[sig, "Rbar"],
                    color="tab:red", label=f"p<0.05 (n={sig.sum()})", s=25)
ax_scatter.set_xscale("log")
ax_scatter.set_xlabel("Number of spikes during running")
ax_scatter.set_ylabel("Resultant vector length (R)")
ax_scatter.set_title("Population theta phase-locking strength across place cells")
ax_scatter.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig4_theta_entrainment.png", dpi=150)
plt.close()
print("saved fig4_theta_entrainment.png")

# %% [markdown]
# ## 7. Theta Phase Precession
#
# For each place cell we take spikes fired inside the place field (defined as the
# contiguous region of the tuning curve above 20% of peak rate) during outbound
# runs, and relate spike phase to normalized in-field position with a
# **circular-linear regression** (Kempter et al., 2012): for a grid of candidate
# slopes we find the slope that maximizes the resultant vector length of
# `phase - 2*pi*slope*position`, then compute the circular-linear correlation
# coefficient `rho`. Significance is assessed by a permutation test that shuffles
# spike-position assignments.

# %%
def field_bounds(rate_curve, bin_centers, thresh_frac=0.2):
    peak_idx = np.argmax(rate_curve)
    thresh = thresh_frac * rate_curve[peak_idx]
    lo = peak_idx
    while lo > 0 and rate_curve[lo - 1] > thresh:
        lo -= 1
    hi = peak_idx
    while hi < len(rate_curve) - 1 and rate_curve[hi + 1] > thresh:
        hi += 1
    return bin_centers[lo], bin_centers[hi]


def circular_linear_corr(x, theta, slopes):
    best_R, best_a, best_phi0 = -1, None, None
    for a in slopes:
        u = theta - 2 * np.pi * a * x
        C, S = np.mean(np.cos(u)), np.mean(np.sin(u))
        R = np.hypot(C, S)
        if R > best_R:
            best_R, best_a, best_phi0 = R, a, np.arctan2(S, C)
    phi_x = np.mod(2 * np.pi * best_a * x + best_phi0, 2 * np.pi)
    theta_bar = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
    phi_bar = np.arctan2(np.mean(np.sin(phi_x)), np.mean(np.cos(phi_x)))
    num = np.sum(np.sin(theta - theta_bar) * np.sin(phi_x - phi_bar))
    den = np.sqrt(np.sum(np.sin(theta - theta_bar) ** 2) * np.sum(np.sin(phi_x - phi_bar) ** 2))
    rho = num / den
    return best_a, rho, best_phi0


slopes = np.linspace(-3, 3, 601)  # candidate slopes, cycles per field-length
rng = np.random.default_rng(0)
n_perm = 300

precession_rows = []
candidates = summary[(summary["mean_rate"] > 0.2) & (summary["peak_rate"] > 1.0)]["unit"].values
for uid in candidates:
    rate_curve = tc_out.sel(unit=uid).values
    x0, x1 = field_bounds(rate_curve, bin_centers, thresh_frac=0.2)
    field_width = x1 - x0
    if field_width < 0.15:
        continue
    sp_ts = pyr[uid].restrict(out_ep)
    sp_pos = sp_ts.value_from(lin_pos).values
    in_field = (sp_pos >= x0) & (sp_pos <= x1)
    if in_field.sum() < 30:
        continue
    sp_t_field = sp_ts.t[in_field]
    sp_pos_field = sp_pos[in_field]
    sp_phase_field = sp_ts.value_from(theta_phase).values[in_field]
    x_norm = (sp_pos_field - x0) / field_width

    a_hat, rho, phi0 = circular_linear_corr(x_norm, sp_phase_field, slopes)
    perm_rhos = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(x_norm)
        _, r_p, _ = circular_linear_corr(shuffled, sp_phase_field, slopes)
        perm_rhos[i] = r_p
    p_perm = (np.sum(np.abs(perm_rhos) >= np.abs(rho)) + 1) / (n_perm + 1)

    precession_rows.append(dict(
        unit=uid, x0=x0, x1=x1, field_width=field_width, n_spikes=int(in_field.sum()),
        slope=a_hat, rho=rho, p_perm=p_perm, x_norm=x_norm, phase=sp_phase_field,
    ))

precession_df = pd.DataFrame(precession_rows)
print(precession_df[["unit", "field_width", "n_spikes", "slope", "rho", "p_perm"]]
      .sort_values("rho", ascending=False).to_string(index=False))

# %%
example_cells = (
    precession_df[(precession_df["slope"] < 0) & (precession_df["p_perm"] < 0.05)]
    .sort_values("rho", ascending=False)["unit"].values[:4]
)

fig, axes = plt.subplots(1, len(example_cells), figsize=(4 * len(example_cells), 4), sharey=True)
for ax, uid in zip(axes, example_cells):
    row = precession_df[precession_df.unit == uid].iloc[0]
    x_norm, phase = row["x_norm"], row["phase"]
    ax.scatter(x_norm, phase, s=8, color="tab:blue", alpha=0.6)
    ax.scatter(x_norm, phase + 2 * np.pi, s=8, color="tab:blue", alpha=0.6)
    xs = np.linspace(0, 1, 50)
    _, _, phi0 = circular_linear_corr(x_norm, phase, [row["slope"]])
    fit = np.mod(2 * np.pi * row["slope"] * xs + phi0, 2 * np.pi)
    order_xs = np.argsort(xs)
    ax.plot(xs[order_xs], fit[order_xs], color="tab:red", lw=2)
    ax.plot(xs[order_xs], fit[order_xs] + 2 * np.pi, color="tab:red", lw=2)
    ax.set_xlabel("Normalized in-field position")
    ax.set_title(f"unit {uid}\nslope={row['slope']:.2f} cyc/field, ρ={row['rho']:.2f}\n"
                 f"perm p={row['p_perm']:.3f}", fontsize=9)
    ax.set_ylim(0, 4 * np.pi)
axes[0].set_ylabel("Theta phase (rad)")

plt.suptitle("Theta phase precession: spike phase decreases as the rat crosses the place field", y=1.05)
plt.tight_layout()
plt.savefig("fig5_phase_precession_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig5_phase_precession_examples.png")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

sig = precession_df["p_perm"] < 0.05
axes[0].scatter(precession_df.loc[~sig, "slope"], precession_df.loc[~sig, "rho"],
                color="0.6", s=35, label=f"not significant (n={(~sig).sum()})")
axes[0].scatter(precession_df.loc[sig, "slope"], precession_df.loc[sig, "rho"],
                color="tab:red", s=35, label=f"perm p<0.05 (n={sig.sum()})")
axes[0].axvline(0, color="k", lw=0.7, ls="--")
axes[0].set_xlabel("Circular-linear slope (cycles / field)")
axes[0].set_ylabel("Circular-linear correlation ρ")
axes[0].set_title("Phase precession across candidate place cells")
axes[0].legend(fontsize=8)

n_neg_sig = ((precession_df["slope"] < 0) & sig).sum()
n_pos_sig = ((precession_df["slope"] > 0) & sig).sum()
axes[1].hist(precession_df["slope"], bins=15, color="tab:blue", alpha=0.7, edgecolor="k")
axes[1].axvline(0, color="k", lw=0.7, ls="--")
axes[1].set_xlabel("Circular-linear slope (cycles / field)")
axes[1].set_ylabel("Number of cells")
axes[1].set_title(f"Slope distribution\n({n_neg_sig} sig. negative, {n_pos_sig} sig. positive of {len(precession_df)})")

plt.tight_layout()
plt.savefig("fig6_precession_summary.png", dpi=150)
plt.close()
print("saved fig6_precession_summary.png")

# %% [markdown]
# ## Summary
#
# - A single LFP channel selected for maximal theta/delta power ratio shows a clear
#   spectral peak at ~9 Hz during track running (Figure 2).
# - The majority of CA1 place cells fire at a preferred, non-uniform theta phase
#   (Figure 4): the Rayleigh test rejects circular uniformity (p<0.05) for a large
#   fraction of place cells, i.e. they are **theta phase entrained**.
# - When restricting spikes to those fired inside each cell's place field on
#   outbound runs, spike phase decreases systematically with normalized in-field
#   position for many cells, giving a **negative circular-linear slope** with a
#   significant circular-linear correlation by permutation test (Figures 5-6). This
#   is the classic **theta phase precession** signature described by O'Keefe and
#   Recce (1993).
