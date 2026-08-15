# %% [markdown]
# # Theta Phase Precession in Hippocampal Place Cells
#
# This notebook demonstrates **theta phase precession**: as a rat runs through a
# place cell's firing field, the cell fires at progressively earlier phases of the
# ongoing 6-12 Hz hippocampal theta rhythm. First described by O'Keefe & Recce
# (1993), phase precession converts the animal's position within a field into a
# temporal (phase) code and is a cornerstone of hippocampal sequence coding.
#
# **Dataset.** DANDI:000044, *"Diversity in neural firing dynamics supports both
# rigid and learned hippocampal sequences"* (Grosmark, Long & Buzsaki). We use
# session `sub-Buddy` (bilateral CA1 silicon-probe recordings) while the rat runs
# back and forth on a 1.6 m linear maze. The file provides linearized position
# (~39 Hz), 128-channel LFP at 1250 Hz, and spike-sorted CA1 units.
#
# **Approach.** We restrict to the maze-running epoch, extract theta phase from the
# LFP channel with the strongest theta, keep only samples where the rat is actually
# running (speed > 0.15 m/s), identify directional traversals, build 1-D place
# fields per direction, and then, for each place cell, regress spike theta-phase
# against normalized position within the field using the circular-linear method of
# Kempter et al. (2012).

# %%
import numpy as np
import pynapple as nap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pp_lib as L

np.random.seed(0)
ASSET_ID = '185b8a36-d671-4688-ba05-9e89a902c486'  # sub-Buddy
SPEED_THR = 0.15      # m/s, running threshold
MIN_TRAVERSAL = 0.4   # m net displacement to count a directional run
NB_BINS = 80          # spatial bins across 1.6 m track
PEAK_RATE_MIN = 3.0   # Hz, place-cell peak-rate criterion
FIELD_FRAC = 0.35     # in-field = bins above this fraction of peak rate

# %% [markdown]
# ## 1. Load data and restrict to the maze epoch

# %%
nwb = L.open_nwb(ASSET_ID)
maze = L.maze_interval(nwb)
print(f"Maze epoch: {maze.start[0]:.0f}-{maze.end[0]:.0f} s "
      f"({(maze.end[0]-maze.start[0])/60:.1f} min)")

pos, pos_rate = L.load_position(nwb, maze)
run_ep = L.running_epochs(pos, pos_rate, speed_thr=SPEED_THR)
print(f"Track length: {pos.min():.2f}-{pos.max():.2f} m; "
      f"{len(pos)} valid position samples; "
      f"{run_ep.tot_length():.0f} s of running (speed > {SPEED_THR} m/s)")

# Units: keep putative excitatory (pyramidal) CA1 cells
napnwb = nap.NWBFile(nwb)
units = napnwb['units']
udf = nwb.units.to_dataframe()
exc_ids = udf.index[udf['cell_type'] == 'excitatory'].tolist()
units_exc = units[exc_ids]
print(f"{len(units)} total units; {len(units_exc)} putative excitatory (pyramidal)")

# %% [markdown]
# ## 2. Theta phase from the best LFP channel
#
# We pick the channel with the largest 6-12 Hz power, band-pass filter it, and take
# the Hilbert transform for an instantaneous theta phase. Spike phase is later
# interpolated on the unit circle (via cos/sin) so it never jumps at the +/-pi wrap.

# %%
cand = list(range(0, 128, 8))
pw = []
for ch in tqdm(cand, desc="coarse theta sweep"):
    lfp_c, rate = L.load_lfp_channel(nwb, maze, ch)
    pw.append(np.mean(L.bandpass(lfp_c.values, rate) ** 2))
coarse = cand[int(np.argmax(pw))]
best_ch, best_p = coarse, -1
for ch in tqdm(range(max(0, coarse - 6), min(128, coarse + 7)), desc="fine theta sweep"):
    lfp_c, rate = L.load_lfp_channel(nwb, maze, ch)
    p = np.mean(L.bandpass(lfp_c.values, rate) ** 2)
    if p > best_p:
        best_ch, best_p = ch, p
print(f"Theta reference channel: {best_ch}")

lfp, rate = L.load_lfp_channel(nwb, maze, best_ch)
phase, amp, filt, cos_t, sin_t = L.theta_phase(lfp, rate)

# %% [markdown]
# ## 0. Validation: theta rhythm, phase, and behaviour
#
# Before analysing, confirm the raw LFP contains a clean ~8 Hz theta rhythm, that the
# extracted phase advances monotonically (sawtooth), and that the rat runs back and
# forth on the track.

# %%
t0 = maze.start[0] + 720
sl = nap.IntervalSet(start=t0, end=t0 + 2)
lfp_s = lfp.restrict(sl); filt_s = filt.restrict(sl); ph_s = phase.restrict(sl)
posw = nap.IntervalSet(start=maze.start[0] + 600, end=maze.start[0] + 720)
pos_s = pos.restrict(posw)

fig, ax = plt.subplots(3, 1, figsize=(11, 6.5))
ax[0].plot(lfp_s.index - t0, lfp_s.values, 'k', lw=0.7, label='raw LFP')
ax[0].plot(filt_s.index - t0, filt_s.values, 'C3', lw=1.8, label='6-12 Hz')
ax[0].set_title(f'Channel {best_ch}: raw vs theta-filtered LFP (2 s)')
ax[0].set_ylabel('LFP'); ax[0].legend(loc='upper right', fontsize=8)
ax[1].plot(ph_s.index - t0, ph_s.values, 'C0', lw=1)
ax[1].set_ylabel('theta phase (rad)'); ax[1].set_xlabel('time (s)')
ax[2].plot(pos_s.index - maze.start[0], pos_s.values, 'C2', lw=1)
ax[2].set_title('Linearized position (2 min): back-and-forth runs')
ax[2].set_ylabel('position (m)'); ax[2].set_xlabel('time in maze (s)')
fig.tight_layout()
fig.savefig('fig0_validation.png', dpi=120)
plt.close(fig)

# %% [markdown]
# ## 3. Identify directional running traversals
#
# The rat alternates between the two ends of the track. Because the linearized
# position is defined only during track running, each contiguous run of valid
# samples is one traversal; we classify each by net displacement as *rightward*
# (increasing position) or *leftward*, and intersect with the running epochs.

# %%
right_ep, left_ep = L.traversal_epochs(pos, pos_rate, run_ep, min_disp=MIN_TRAVERSAL)
print(f"{len(right_ep)} rightward and {len(left_ep)} leftward running traversals")

# %% [markdown]
# ## 4. Place fields per running direction
#
# Place fields are direction-dependent on linear tracks, so we compute 1-D tuning
# curves separately for each direction, using only running periods, and lightly
# smooth them before detecting the field.

# %%
tc_right = L.smooth_tuning(nap.compute_1d_tuning_curves(
    units_exc, pos, NB_BINS, ep=right_ep, minmax=(0, 1.6)))
tc_left = L.smooth_tuning(nap.compute_1d_tuning_curves(
    units_exc, pos, NB_BINS, ep=left_ep, minmax=(0, 1.6)))
bin_ctr = tc_right.index.values


def place_cells(tc):
    """Return {unit: (peak_pos, lo, hi)} for units passing the place-cell criterion."""
    out = {}
    for u in tc.columns:
        r = tc[u].values
        if np.nanmax(r) < PEAK_RATE_MIN:
            continue
        pk = int(np.nanargmax(r))
        thr = FIELD_FRAC * r[pk]
        lo = pk
        while lo > 0 and r[lo - 1] >= thr:
            lo -= 1
        hi = pk
        while hi < len(r) - 1 and r[hi + 1] >= thr:
            hi += 1
        # require a reasonably compact single field (< 60% of track)
        if (hi - lo) < 0.6 * NB_BINS and (hi - lo) >= 3:
            out[u] = (bin_ctr[pk], bin_ctr[lo], bin_ctr[hi])
    return out


pc_right = place_cells(tc_right)
pc_left = place_cells(tc_left)
print(f"Place cells: {len(pc_right)} rightward, {len(pc_left)} leftward")

# %% [markdown]
# ## 5. Phase precession per place cell
#
# For each place cell we collect the spikes fired inside its field during running
# traversals in the preferred direction. Each spike gets (i) the animal's position
# normalized to [0,1] along the direction of travel and (ii) the theta phase
# interpolated on the unit circle. We fit the circular-linear model of Kempter et
# al. (2012), yielding a slope (radians of phase per field), a phase offset, and a
# signed circular-linear correlation with a p-value.

# %%
def precession_for_cell(unit, field, ep, direction, min_spikes=25):
    pk, lo, hi = field
    lo = max(lo, 0.0); hi = min(hi, 1.6)
    st = units_exc[unit].restrict(ep)
    if len(st) == 0:
        return None
    sp_pos = pos.interpolate(st).values         # position at each spike
    sp_ph = L.interp_phase(cos_t, sin_t, st)    # theta phase at each spike
    infield = (sp_pos >= lo) & (sp_pos <= hi) & np.isfinite(sp_pos) & np.isfinite(sp_ph)
    x = sp_pos[infield]; ph = sp_ph[infield]
    if len(x) < min_spikes:
        return None
    # normalize position along the DIRECTION OF TRAVEL: 0 = field entry, 1 = exit.
    if direction == 'right':
        xn = (x - lo) / (hi - lo)
    else:
        xn = (hi - x) / (hi - lo)
    slope, phi0, rho, pval = L.circ_lin_regression(ph, xn)
    return dict(unit=unit, n=len(x), xn=xn, ph=ph, slope=slope,
                phi0=phi0, rho=rho, pval=pval, field=(lo, pk, hi))


results = []
for direction, pcs, ep in [('right', pc_right, right_ep), ('left', pc_left, left_ep)]:
    for u, field in pcs.items():
        res = precession_for_cell(u, field, ep, direction)
        if res is not None:
            res['direction'] = direction
            results.append(res)
print(f"{len(results)} place-cell x direction fits with enough in-field spikes")

n_sig = sum(r['pval'] < 0.05 for r in results)
n_neg = sum((r['pval'] < 0.05) and (r['slope'] < 0) for r in results)
print(f"Significant circular-linear correlation (p<0.05): {n_sig}/{len(results)}")
print(f"  of which negative slope (precessing): {n_neg}/{n_sig}")
med_slope = np.median([r['slope'] for r in results]) / (2 * np.pi)
print(f"Median slope: {med_slope:.2f} theta cycles per field pass")

# %% [markdown]
# ## 6. Figure: place fields of the example cells

# %%
# rank significant precessing cells by correlation strength for display
prec = sorted([r for r in results if r['slope'] < 0 and r['pval'] < 0.05],
              key=lambda r: r['rho'])
examples = prec[:6]

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, r in zip(axes.ravel(), examples):
    tc = tc_right if r['direction'] == 'right' else tc_left
    ax.plot(bin_ctr, tc[r['unit']].values, 'C0')
    lo, pk, hi = r['field']
    ax.axvspan(lo, hi, color='C1', alpha=0.2)
    ax.set_title(f"unit {r['unit']} ({r['direction']}), peak "
                 f"{tc[r['unit']].max():.1f} Hz", fontsize=9)
    ax.set_xlabel('position (m)'); ax.set_ylabel('rate (Hz)')
fig.suptitle('Example place fields (shaded = field used for precession)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('fig1_place_fields.png', dpi=120)
plt.close(fig)

# %% [markdown]
# ## 7. Figure: phase precession for example cells
#
# Spike phase (plotted twice over 0-720 deg for visual clarity) versus normalized
# position in the field. The red line is the circular-linear fit; negative slopes are
# the signature of phase precession.

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, r in zip(axes.ravel(), examples):
    ph_deg = np.rad2deg(r['ph'] % (2 * np.pi))
    for off in (0, 360):
        ax.plot(r['xn'], ph_deg + off, '.', ms=3, color='0.3', alpha=0.5)
    xx = np.linspace(0, 1, 100)
    yy = np.rad2deg((r['slope'] * xx + r['phi0']))
    for off in (0, 360, 720):
        ax.plot(xx, (yy % 360) + off - 360, 'C3', lw=1.5)
    ax.set_ylim(0, 720); ax.set_xlim(0, 1)
    ax.set_title(f"unit {r['unit']} ({r['direction']})\n"
                 f"slope {r['slope']/(2*np.pi):.2f} cyc, "
                 f"rho={r['rho']:.2f}, p={r['pval']:.1e}", fontsize=9)
    ax.set_xlabel('normalized position in field')
    ax.set_ylabel('theta phase (deg)')
fig.suptitle('Theta phase precession: spike phase vs. position in field', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('fig2_precession_examples.png', dpi=120)
plt.close(fig)

# %% [markdown]
# ## 8. Figure: population summary
#
# Across all place cells we summarize (a) the distribution of fit slopes, (b) the
# signed circular-linear correlations, and (c) spikes pooled across the significant
# precessing cells after aligning each cell's phase offset, showing the
# population-level precessing band.

# %%
slopes = np.array([r['slope'] / (2 * np.pi) for r in results])
rhos = np.array([r['rho'] for r in results])
sig = np.array([r['pval'] < 0.05 for r in results], dtype=bool)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].hist(slopes, bins=np.arange(-2.0, 1.1, 0.2), color='C0', edgecolor='k')
ax[0].axvline(0, color='k', ls='--')
ax[0].axvline(np.median(slopes), color='C3', label=f'median {np.median(slopes):.2f}')
ax[0].set_xlabel('slope (theta cycles / field)'); ax[0].set_ylabel('# cells')
ax[0].set_title('Precession slopes'); ax[0].legend(fontsize=8)

ax[1].hist(rhos[~sig], bins=np.arange(-1, 1.05, 0.1), color='0.7',
           edgecolor='k', label='n.s.')
ax[1].hist(rhos[sig], bins=np.arange(-1, 1.05, 0.1), color='C1',
           edgecolor='k', label='p<0.05')
ax[1].axvline(0, color='k', ls='--')
ax[1].set_xlabel('signed circular-linear correlation'); ax[1].set_ylabel('# cells')
ax[1].set_title('Correlation coefficients'); ax[1].legend(fontsize=8)

# pooled scatter: align by phase offset so precessing bands overlay
allx, allp = [], []
for r in results:
    if r['slope'] < 0 and r['pval'] < 0.05:
        aligned = (r['ph'] - r['phi0']) % (2 * np.pi)
        allx.append(r['xn']); allp.append(aligned)
n_pool = len(allx)
allx = np.concatenate(allx) if allx else np.array([0.5])
allp = np.rad2deg(np.concatenate(allp)) if n_pool else np.array([180.0])
for off in (0, 360):
    ax[2].plot(allx, allp + off, '.', ms=2, color='C0', alpha=0.25)
bins = np.linspace(0, 1, 11)
bc = 0.5 * (bins[:-1] + bins[1:])
mean_ph = [np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(
    allp[(allx >= bins[i]) & (allx < bins[i + 1])]))))) % 360 for i in range(10)]
ax[2].plot(bc, mean_ph, 'C3o-', lw=2, label='circular mean')
ax[2].plot(bc, np.array(mean_ph) + 360, 'C3o-', lw=2)
ax[2].set_ylim(0, 720); ax[2].set_xlabel('normalized position in field')
ax[2].set_ylabel('aligned theta phase (deg)')
ax[2].set_title(f'Pooled precessing cells (n={n_pool})')
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig('fig3_population_summary.png', dpi=120)
plt.close(fig)

# %% [markdown]
# ## 9. Summary
#
# The example cells show the hallmark negative slope: spikes occur at late theta
# phases when the rat enters the field and at progressively earlier phases as it
# crosses, spanning roughly one theta cycle. Across the population, precession slopes
# cluster below zero and the significant circular-linear correlations are
# predominantly negative, reproducing the classic O'Keefe & Recce phenomenon in
# freely-moving CA1 place cells.

# %%
print("Figures written:")
for f in ['fig0_validation.png', 'fig1_place_fields.png',
          'fig2_precession_examples.png', 'fig3_population_summary.png']:
    print("  ", f)
