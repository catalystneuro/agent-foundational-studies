# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Decoding hippocampal replay during sharp-wave ripples
#
# **Dataset:** [DANDI:000044](https://dandiarchive.org/dandiset/000044) — Grosmark & Buzsáki,
# *"Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences"*
# (Science, 2016). Session `sub-Achilles_ses-Achilles-10252013`.
#
# A rat runs back and forth on a 1.6 m linear track (the **Maze** epoch) while ~120 CA1
# pyramidal cells and the local field potential (LFP) are recorded with silicon probes.
# The track run is flanked by **PRE** and **POST** sleep/rest epochs.
#
# **The phenomenon.** During sharp-wave ripples (SWRs, ~150–250 Hz LFP oscillations that occur
# during immobility and slow-wave sleep), the hippocampus spontaneously reactivates place-cell
# sequences that recapitulate spatial trajectories the animal has run, compressed ~20× in time.
# This is *hippocampal replay*.
#
# **The demonstration.** We (1) build place fields from track running, (2) detect SWRs from the
# CA1 LFP, (3) validate a Bayesian position decoder on running data, and (4) apply that decoder
# inside each ripple. We show that the decoded position sweeps smoothly across the track — a
# replayed trajectory — far more often than expected by chance, more so in POST than PRE sleep,
# and at a "virtual" speed ~20× the animal's real running speed.
#
# All data are streamed from the DANDI S3 bucket with `remfile` (no full download).

# %%
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import h5py, remfile, pynapple as nap
from pynwb import NWBHDF5IO
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

np.random.seed(0)
S3 = "https://dandiarchive.s3.amazonaws.com/blobs/4f5/a84/4f5a84aa-a6e4-496a-9b23-535fa6fbd3ae"
FS_LFP = 1250.0     # LFP sampling rate (Hz)
RIPPLE_CH = 1       # CA1 pyramidal-layer channel chosen empirically (highest ripple power)

# %% [markdown]
# ## 1. Stream the NWB file and load core data
#
# We load spike times (as a Pynapple `TsGroup`), the linearized 1-D track position, the
# experiment epochs (PRE / Maze / POST), and the scored brain states (Awake / Non-REM / REM).

# %%
rf = remfile.File(S3, disk_cache=remfile.DiskCache('/tmp/remfile_cache'))
nwb = NWBHDF5IO(file=h5py.File(rf, "r"), load_namespaces=True).read()

# --- units (spike times + cell-type / location metadata) ---
ut = nwb.units
cell_type = ut["cell_type"].data[:]
location = ut["location"].data[:]
spikes = {i: nap.Ts(np.asarray(ut["spike_times"][i])) for i in range(len(ut.id[:]))}
units = nap.TsGroup(spikes)
units.set_info(cell_type=np.array([cell_type[i] for i in units.index]),
               location=np.array([location[i] for i in units.index]))

# --- linearized position: the NWB 'rate' field stores the sample interval (dt) ---
sp = list(nwb.processing["behavior"].data_interfaces["1.6mLinearMazeLinearizedPosition"]
          .spatial_series.values())[0]
t_pos = sp.starting_time + np.arange(sp.data.shape[0]) * sp.rate
position = nap.Tsd(t=t_pos, d=np.asarray(sp.data[:]).squeeze())

# --- epochs ---
ep = nwb.intervals["epochs"]; ep_lab = ep["label"].data[:]
epochs = {ep_lab[i]: nap.IntervalSet(start=ep["start_time"][i], end=ep["stop_time"][i])
          for i in range(len(ep.id[:]))}

# --- brain states ---
stt = nwb.processing["behavior"].data_interfaces["states"]
slab = stt["label"].data[:]; s0 = stt["start_time"].data[:]; s1 = stt["stop_time"].data[:]
states = {name: nap.IntervalSet(start=np.asarray(s0)[slab == name],
                                end=np.asarray(s1)[slab == name])
          for name in np.unique(slab)}

print(f"{len(units)} units  |  "
      f"{np.sum(units.get_info('cell_type')=='excitatory')} excitatory (pyramidal), "
      f"{np.sum(units.get_info('cell_type')=='inhibitory')} inhibitory")
print("epochs:", {k: (round(float(v.start[0]), 1), round(float(v.end[0]), 1)) for k, v in epochs.items()})
pyr = units[units.get_info('cell_type') == 'excitatory']

# %% [markdown]
# ## 2. Running epochs and place fields
#
# The linearized position is only defined while the animal is on the track (~87% NaN otherwise).
# We build a "valid tracking" support that breaks at sampling gaps > 100 ms, then keep periods
# where running speed exceeds 0.1 m/s. Place fields (1-D tuning curves) are computed separately
# for rightward and leftward runs, since place fields on a linear track are direction-specific.

# %%
def valid_support(p, max_gap=0.1):
    """IntervalSet covering contiguous position samples (breaks at gaps > max_gap)."""
    t = p.t; brk = np.where(np.diff(t) > max_gap)[0]
    s = [t[0]]; e = []
    for b in brk:
        e.append(t[b]); s.append(t[b + 1])
    e.append(t[-1])
    return nap.IntervalSet(start=np.array(s), end=np.array(e))

p = position[~np.isnan(position.values)]                       # valid position samples
speed = nap.Tsd(t=p.t, d=np.abs(np.gradient(p.values, p.t)))
vel = nap.Tsd(t=p.t, d=np.gradient(p.values, p.t))
run = speed.threshold(0.1, 'above').time_support.intersect(valid_support(p))
rightward = vel.threshold(0, 'above').time_support.intersect(run)
leftward = vel.threshold(0, 'below').time_support.intersect(run)
print(f"valid tracking: {valid_support(p).tot_length():.0f} s | "
      f"running: {run.tot_length():.0f} s "
      f"(rightward {rightward.tot_length():.0f} s, leftward {leftward.tot_length():.0f} s)")

NB = 50  # position bins across the 1.6 m track
tc_R = nap.compute_1d_tuning_curves(pyr, p, nb_bins=NB, ep=rightward, minmax=(0, 1.6))
tc_L = nap.compute_1d_tuning_curves(pyr, p, nb_bins=NB, ep=leftward, minmax=(0, 1.6))

# %%
order = np.argsort(np.argmax(tc_R.values, axis=0))  # sort cells by rightward place-field peak
fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, tc, title in [(axes[0], tc_R, 'Rightward runs'), (axes[1], tc_L, 'Leftward runs')]:
    M = tc.values[:, order].T
    M = M / (M.max(axis=1, keepdims=True) + 1e-9)
    ax.imshow(M, aspect='auto', origin='lower', extent=[0, 1.6, 0, M.shape[0]], cmap='viridis')
    ax.set_xlabel('Position on track (m)'); ax.set_title(title)
axes[0].set_ylabel('Cell (sorted by rightward place-field peak)')
plt.tight_layout(); plt.savefig('fig_placefields.png', dpi=130)
print("Place fields tile the track along the diagonal; note direction-specificity "
      "(same sort looks scrambled for leftward runs).")

# %% [markdown]
# ## 3. Detect sharp-wave ripples from the CA1 LFP
#
# We stream one CA1 channel (chosen empirically for the strongest ripple power), band-pass at
# 150–250 Hz, take the Hilbert envelope, z-score it over immobility, and detect events crossing
# 3 SD (with a 5 SD peak requirement), 15–250 ms long. Detection is restricted to immobility
# (everything except track running), where SWRs occur.

# %%
es = list(nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series.values())[0]
lfp = nap.Tsd(t=np.arange(es.data.shape[0]) / FS_LFP,
              d=np.asarray(es.data[:, RIPPLE_CH]).astype(np.float32))   # streams ~87 MB, ~60-90 s

b, a = butter(4, [150 / (FS_LFP / 2), 250 / (FS_LFP / 2)], btype='band')
rip_filt = filtfilt(b, a, lfp.values.astype(float))
env = gaussian_filter1d(np.abs(hilbert(rip_filt)), sigma=0.008 * FS_LFP)  # ~8 ms smoothing
env = nap.Tsd(t=lfp.t, d=env)

whole = nap.IntervalSet(start=lfp.t[0], end=lfp.t[-1])
immobile = whole.set_diff(run)                       # detect ripples outside running
e_imm = env.restrict(immobile)
z = nap.Tsd(t=env.t, d=(env.values - e_imm.values.mean()) / e_imm.values.std())

cand = z.threshold(3.0, 'above').time_support.intersect(immobile)
cand = cand.drop_short_intervals(0.015).merge_close_intervals(0.02)
cand = cand.drop_short_intervals(0.015).drop_long_intervals(0.250)
starts, ends, pk_t, pk_z = [], [], [], []
for s, e in zip(cand.start, cand.end):
    seg = z.get(s, e)
    if seg.values.max() >= 5.0:                      # require a strong peak
        i = np.argmax(seg.values)
        starts.append(s); ends.append(e); pk_t.append(seg.t[i]); pk_z.append(float(seg.values[i]))
ripples = nap.IntervalSet(start=np.array(starts), end=np.array(ends))
pk_t = np.array(pk_t)
print(f"{len(ripples)} ripples | median duration "
      f"{np.median(ripples.end - ripples.start) * 1000:.0f} ms")
for name in ['PREEpoch', 'POSTEpoch']:
    n = len(ripples.intersect(epochs[name]))
    print(f"  {name}: {n} ripples ({n / (epochs[name].tot_length() / 60):.1f}/min)")

# %% [markdown]
# ### 3a. Validate the detector
# A single strong POST ripple (raw LFP, ripple-band, and pyramidal raster) plus the
# ripple-triggered population firing rate averaged over all POST ripples.

# %%
post = epochs['POSTEpoch']
in_post = (pk_t > post.start[0]) & (pk_t < post.end[0])
ci = np.where(in_post)[0][np.argmax(np.array(pk_z)[in_post])]
tc0 = pk_t[ci]; w = 0.15; seg = nap.IntervalSet(tc0 - w, tc0 + w)
filt_tsd = nap.Tsd(t=lfp.t, d=rip_filt)

fig, axs = plt.subplots(3, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1.2, 1.2, 2]})
axs[0].plot((lfp.restrict(seg).t - tc0) * 1000, lfp.restrict(seg).values, 'k', lw=0.7)
axs[0].set_ylabel('LFP (a.u.)'); axs[0].set_title('Raw CA1 LFP (POST-sleep ripple)')
axs[1].plot((filt_tsd.restrict(seg).t - tc0) * 1000, filt_tsd.restrict(seg).values, 'C0', lw=0.7)
axs[1].set_ylabel('150-250 Hz'); axs[1].set_title('Ripple-band filtered')
cols = list(pyr.keys())
for row, idx in enumerate(order):
    stt2 = pyr[cols[idx]].restrict(seg)
    axs[2].plot((stt2.t - tc0) * 1000, np.full(len(stt2.t), row), '|', color='k', ms=4, mew=0.7)
axs[2].set_ylabel('Cell (by place field)'); axs[2].set_xlabel('Time from ripple peak (ms)')
axs[2].set_title('Pyramidal spikes')
for ax in axs:
    for s, e in zip(ripples.intersect(seg).start, ripples.intersect(seg).end):
        ax.axvspan((s - tc0) * 1000, (e - tc0) * 1000, color='orange', alpha=0.2)
plt.tight_layout(); plt.savefig('fig_ripple_example.png', dpi=130)

# %%
mua = nap.Ts(np.sort(np.concatenate([pyr[u].t for u in pyr.keys()])))
peth = nap.compute_perievent(mua, nap.Ts(pk_t[in_post]), window=(-0.4, 0.4))
allt = np.concatenate([peth[k].t for k in peth.keys()])
bins = np.arange(-0.4, 0.401, 0.01)
rate = np.histogram(allt, bins)[0] / (in_post.sum() * 0.01)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(bins[:-1] * 1000, rate, width=10, align='edge', color='C3', alpha=0.85)
ax.axvline(0, color='k', ls='--')
ax.set_xlabel('Time from ripple peak (ms)'); ax.set_ylabel('Population rate (spk/s)')
ax.set_title(f'Ripple-triggered pyramidal population firing (N={in_post.sum()} POST ripples)')
plt.tight_layout(); plt.savefig('fig_ripple_mua.png', dpi=130)
print(f"Population rate: baseline {rate[:10].mean():.0f} -> peak {rate.max():.0f} spk/s "
      f"({rate.max()/rate[:10].mean():.1f}x)")

# %% [markdown]
# ## 4. Bayesian decoder — validation on running
#
# We use Pynapple's memoryless Bayesian decoder (`decode_1d`). The template is a set of place
# fields built from all running periods, restricted to cells with a peak rate > 1 Hz. Before
# trusting it inside ripples, we confirm it recovers the animal's *actual* position during
# running at coarse (250 ms) bins.

# %%
tc_all = nap.compute_1d_tuning_curves(pyr, p, nb_bins=NB, ep=run, minmax=(0, 1.6))
active = tc_all.columns[tc_all.max() > 1.0]
tc_all = tc_all[list(active)]
grp = pyr[list(active)]
tc_dec = tc_all.copy()
tc_dec.loc[:, :] = gaussian_filter1d(tc_all.values, 1.0, axis=0)   # light smoothing
print(f"decoding template: {len(active)} place cells")

dec, prob = nap.decode_1d(tc_dec, grp, run, 0.25)
act = np.interp(dec.t, p.t, p.values)
err = np.abs(dec.values - act)
print(f"median decoding error {np.median(err):.3f} m | corr(decoded, actual) = "
      f"{np.corrcoef(dec.values, act)[0, 1]:.2f}")

segv = nap.IntervalSet(run.start[5], run.start[5] + 40)
plt.figure(figsize=(10, 4))
plt.plot(p.restrict(segv).t, p.restrict(segv).values, 'k.', ms=3, label='actual')
plt.plot(dec.restrict(segv).t, dec.restrict(segv).values, 'C1.-', ms=5, lw=0.8, label='decoded')
plt.xlabel('Time (s)'); plt.ylabel('Position (m)'); plt.legend()
plt.title(f'Decoder validation on running (median err {np.median(err):.2f} m, '
          f'r={np.corrcoef(dec.values, act)[0,1]:.2f})')
plt.tight_layout(); plt.savefig('fig_decoder_validation.png', dpi=130)

# %% [markdown]
# ## 5. Decode replay inside ripples
#
# For each ripple we decode position in 20 ms bins. Short ripples are padded to a minimum
# 100 ms window centered on the burst. A *replayed trajectory* is a decoded posterior that
# sweeps monotonically across the track, quantified by the **weighted correlation** between
# decoded position and elapsed time (posterior probability as weights). Significance is
# assessed per event against 250 per-time-bin position shuffles.

# %%
def weighted_corr(P, xbins, tbins):
    P = P / (P.sum() + 1e-12)
    X = xbins[None, :] * np.ones((P.shape[0], 1))
    T = tbins[:, None] * np.ones((1, P.shape[1]))
    mx = (P * X).sum(); mt = (P * T).sum()
    cov = (P * (X - mx) * (T - mt)).sum()
    vx = (P * (X - mx) ** 2).sum(); vt = (P * (T - mt) ** 2).sum()
    return cov / np.sqrt(vx * vt) if vx > 0 and vt > 0 else np.nan

def shuffle_null(P, xbins, tbins, n=250, seed=0):
    rng = np.random.RandomState(seed); npos = P.shape[1]; out = np.empty(n)
    for i in range(n):
        Ps = np.stack([np.roll(P[r], rng.randint(npos)) for r in range(P.shape[0])])
        out[i] = abs(weighted_corr(Ps, xbins, tbins))
    return out

def pad_ripples(rip, mindur=0.1):
    s = np.array(rip.start); e = np.array(rip.end); c = (s + e) / 2; short = (e - s) < mindur
    return nap.IntervalSet(start=np.where(short, c - mindur / 2, s),
                           end=np.where(short, c + mindur / 2, e))

def decode_events(events, bin_size=0.02, min_bins=5, min_active=5):
    xbins = np.asarray(tc_dec.index); out = []
    for s, e in zip(events.start, events.end):
        if (e - s) < min_bins * bin_size:
            continue
        ev = nap.IntervalSet(s, e)
        _, prob = nap.decode_1d(tc_dec, grp, ev, bin_size)
        P = prob.values
        if P.shape[0] < min_bins:
            continue
        n_active = int((grp.count(bin_size, ev).values.sum(0) > 0).sum())
        if n_active < min_active:
            continue
        tb = prob.index.values - prob.index.values[0]
        out.append(dict(start=s, end=e, P=P, tbins=tb, xbins=xbins,
                        wc=weighted_corr(P, xbins, tb), n_active=n_active))
    return out

def replay_speed(e):
    pk = e['xbins'][np.argmax(e['P'], axis=1)]
    return abs(np.polyfit(e['tbins'], pk, 1)[0])   # m/s

events = {}
for name in ['PREEpoch', 'POSTEpoch']:
    evs = decode_events(pad_ripples(ripples.intersect(epochs[name]), 0.1))
    for k, e in enumerate(evs):
        null = shuffle_null(e['P'], e['xbins'], e['tbins'], n=250, seed=k)
        e['p'] = (np.sum(null >= abs(e['wc'])) + 1) / (len(null) + 1)
        e['sig'] = e['p'] < 0.05
    events[name] = evs
    frac = np.mean([e['sig'] for e in evs])
    print(f"{name}: {len(evs)} candidate events, "
          f"{np.sum([e['sig'] for e in evs])} significant ({100*frac:.0f}%)")

# %% [markdown]
# ### 5a. Example replay trajectories (POST sleep)
# Each panel is the Bayesian posterior over track position (color) through a single ripple;
# cyan dots mark the maximum-a-posteriori decoded position. These sweep smoothly across the
# track — both forward (increasing) and reverse (decreasing) — at high weighted correlation.

# %%
sig_post = [e for e in events['POSTEpoch'] if e['sig'] and e['P'].shape[0] >= 5]
def span(e):
    pk = e['xbins'][np.argmax(e['P'], axis=1)]; return pk.max() - pk.min()
examples = sorted([e for e in sig_post if span(e) > 0.5], key=lambda e: -abs(e['wc']))[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, e in zip(axes.flat, examples):
    xb = e['xbins']; tb = e['tbins'] * 1000
    ax.imshow(e['P'].T, origin='lower', aspect='auto',
              extent=[tb[0] - 10, tb[-1] + 10, xb[0], xb[-1]], cmap='hot')
    pk = xb[np.argmax(e['P'], axis=1)]
    ax.plot(tb, pk, 'c.', ms=6)
    ax.plot(tb, np.polyval(np.polyfit(tb, pk, 1), tb), 'c-', lw=1, alpha=0.7)
    ax.set_title(f"wc={e['wc']:.2f}, {replay_speed(e):.0f} m/s, {e['n_active']} cells", fontsize=9)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Decoded pos (m)')
fig.suptitle('Example hippocampal replay events during POST-sleep sharp-wave ripples\n'
             '(Bayesian posterior; cyan = MAP decoded position)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig('fig_replay_examples.png', dpi=130)

# %% [markdown]
# ### 5b. Population summary
# (left) Observed trajectory scores have a heavy tail beyond the position-shuffle null.
# (center) Significant replay far exceeds the 5% chance level and is more frequent in POST
# than PRE sleep (experience-dependent enhancement). (right) Replay unfolds at a "virtual"
# speed ~20× the animal's ~0.5 m/s running speed — the temporal-compression hallmark of replay.

# %%
pre, pos_ev = events['PREEpoch'], events['POSTEpoch']
obs = np.abs([e['wc'] for e in pos_ev])
shuf = np.array([shuffle_null(e['P'], e['xbins'], e['tbins'], n=1, seed=1000 + k)[0]
                 for k, e in enumerate(pos_ev)])
fr_pre, fr_post = np.mean([e['sig'] for e in pre]), np.mean([e['sig'] for e in pos_ev])
n_pre, n_post = len(pre), len(pos_ev)
se_pre = np.sqrt(fr_pre * (1 - fr_pre) / n_pre); se_post = np.sqrt(fr_post * (1 - fr_post) / n_post)
speeds = np.array([replay_speed(e) for e in pre + pos_ev if e['sig']])

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
bb = np.linspace(0, 1, 26)
axs[0].hist(obs, bb, alpha=0.6, density=True, color='C3', label='observed')
axs[0].hist(shuf, bb, alpha=0.6, density=True, color='0.5', label='shuffle')
axs[0].set_xlabel('|weighted correlation|'); axs[0].set_ylabel('density')
axs[0].set_title('POST ripple trajectory scores\nvs position shuffle'); axs[0].legend()
axs[1].bar([0, 1], [fr_pre * 100, fr_post * 100], yerr=[se_pre * 100, se_post * 100],
           color=['C0', 'C3'], capsize=5, width=0.6)
axs[1].axhline(5, ls='--', color='k', label='chance (5%)')
axs[1].set_xticks([0, 1]); axs[1].set_xticklabels([f'PRE\n(n={n_pre})', f'POST\n(n={n_post})'])
axs[1].set_ylabel('% significant replay (p<0.05)')
axs[1].set_title('Significant replay events\nPRE vs POST sleep'); axs[1].legend()
axs[2].hist(speeds, bins=np.linspace(0, 25, 26), color='C4', alpha=0.8)
axs[2].axvline(0.5, color='k', ls='--', label='running speed ~0.5 m/s')
axs[2].set_xlabel('Replay speed (m/s)'); axs[2].set_ylabel('# events')
axs[2].set_title(f'Virtual replay speed\n(median {np.median(speeds):.0f} m/s, '
                 f'{np.median(speeds)/0.5:.0f}x running)'); axs[2].legend()
plt.tight_layout(); plt.savefig('fig_replay_summary.png', dpi=130)

# two-proportion z-test: POST vs PRE significant fraction
p_pool = (np.sum([e['sig'] for e in pre]) + np.sum([e['sig'] for e in pos_ev])) / (n_pre + n_post)
zscore = (fr_post - fr_pre) / np.sqrt(p_pool * (1 - p_pool) * (1 / n_pre + 1 / n_post))
print(f"POST replay ({100*fr_post:.0f}%) > PRE ({100*fr_pre:.0f}%): "
      f"z={zscore:.1f}, p={2*(1-norm.cdf(abs(zscore))):.1e}")
print(f"median virtual replay speed {np.median(speeds):.0f} m/s (~{np.median(speeds)/0.5:.0f}x running)")

# %% [markdown]
# ## Conclusion
#
# Applying a place-field Bayesian decoder inside CA1 sharp-wave ripples recovers compressed,
# smoothly-swept spatial trajectories — hippocampal replay of the 1.6 m linear track. These
# trajectories are far more structured than position-shuffled controls, occur significantly
# more often during POST-experience sleep than during PRE sleep, and unfold ~20× faster than
# the animal ever ran. Together these are the defining signatures of hippocampal replay.
