# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hippocampal replay: decoding spatial trajectories during sharp-wave ripples
#
# During awake exploration, hippocampal CA1 place cells fire at specific locations
# ("place fields") and collectively tile the environment. During subsequent rest and
# sleep, these same cells reactivate in compressed sequences that recapitulate the
# spatial trajectories the animal experienced. These reactivations are time-locked to
# **sharp-wave ripples (SWRs)**, brief (~50 ms) 150–250 Hz oscillations in the CA1 LFP,
# and are the neural signature of **memory replay**.
#
# This notebook demonstrates replay end-to-end from real DANDI data:
#
# 1. Build direction-specific place fields from running on a 1.6 m linear track.
# 2. Validate a Bayesian position decoder against the animal's true trajectory.
# 3. Detect sharp-wave ripples in the CA1 LFP during post-run sleep.
# 4. Decode the spatial trajectory represented within each ripple.
# 5. Test each event for sequential ("replay") structure against a place-field
#    circular-shift shuffle, and quantify the population of replay events.
#
# **Dataset.** DANDI:000044, *"Diversity in neural firing dynamics supports both rigid
# and learned hippocampal sequences"* (Grosmark, Long & Buzsáki 2016) — bilateral CA1
# silicon-probe recordings in rats running a linear track, flanked by pre- and post-run
# sleep. We use session `sub-Achilles/ses-Achilles-10252013`.

# %%
import numpy as np
import pandas as pd
import h5py, remfile
from pynwb import NWBHDF5IO
import pynapple as nap
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Stream the session from DANDI
#
# The NWB file is ~9 GB (raw + LFP), so we stream it with `remfile` and a local disk
# cache, reading only the arrays we need. The LFP is chunked one channel at a time, so
# reading a single ripple-detection channel pulls only ~90 MB.

# %%
from dandi.dandiapi import DandiAPIClient

DANDISET, SESSION = "000044", (
    "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
)
with DandiAPIClient() as client:
    asset = client.get_dandiset(DANDISET, "draft").get_asset_by_path(SESSION)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
print("Streaming:", s3_url)

rem = remfile.File(s3_url, disk_cache=remfile.DiskCache("/tmp/remfile_cache"))
h5 = h5py.File(rem, "r")
nwb = NWBHDF5IO(file=h5, load_namespaces=True).read()

# Behavioral epochs: PRE sleep / Maze (linear track) / POST sleep
epochs = nwb.epochs.to_dataframe()
epd = {str(r.label): (float(r.start_time), float(r.stop_time)) for r in epochs.itertuples()}
maze = nap.IntervalSet(*epd["MazeEpoch"])
post = nap.IntervalSet(*epd["POSTEpoch"])
pre = nap.IntervalSet(*epd["PREEpoch"])
print(epochs[["start_time", "stop_time", "label"]].to_string())

# %% [markdown]
# ## 2. Load spikes, position, and LFP
#
# We keep excitatory (putative pyramidal) CA1 units, the linearized track position, and
# one CA1 LFP channel with strong ripple-band power for SWR detection.

# %%
# Excitatory units -> pynapple TsGroup
ct = nwb.units["cell_type"][:]
st = nwb.units["spike_times"]
spikes = {i: np.asarray(st[i]) for i in range(len(nwb.units)) if str(ct[i]) == "excitatory"}
units = nap.TsGroup({i: nap.Ts(t=v) for i, v in spikes.items()})
print(f"{len(units)} excitatory CA1 units")

# Linearized position on the 1.6 m track (defined only during the Maze epoch)
lin = list(nwb.processing["behavior"].data_interfaces[
    "1.6mLinearMazeLinearizedPosition"].spatial_series.values())[0]
xl = np.asarray(lin.data[:]).ravel()
tl = lin.starting_time + np.arange(xl.size) / lin.rate
ok = ~np.isnan(xl)
pos = nap.Tsd(t=tl[ok], d=xl[ok]).restrict(maze)

# LFP: pick the channel with the strongest ripple-band power from a POST sample
es = list(nwb.processing["ecephys"].data_interfaces["LFP"].electrical_series.values())[0]
fs = es.rate
_b, _a = butter(4, [150 / (fs / 2), 250 / (fs / 2)], btype="band")
_seg = es.data[int(25000 * fs):int(25020 * fs), :].astype(float)
ripple_ch = int(np.argmax(np.sqrt((filtfilt(_b, _a, _seg, axis=0) ** 2).mean(0))))
print(f"LFP fs={fs} Hz; ripple detection channel = {ripple_ch}")
lfp = es.data[:, ripple_ch].astype(float)          # streams one channel (~90 MB)
t_lfp = np.arange(lfp.size) / fs

# %% [markdown]
# ### Raw-data sanity check
# The animal shuttles back and forth on the linear track, and the CA1 LFP shows clear
# high-frequency ripple events during sleep.

# %%
fig, ax = plt.subplots(2, 1, figsize=(13, 5))
ax[0].plot(pos.index - maze.start[0], pos.values, "k", lw=0.5)
ax[0].set(xlabel="Time in maze (s)", ylabel="Position (m)",
          title="Behavior: running on the 1.6 m linear track")
seg = slice(int(25000 * fs), int(25002 * fs))       # 2 s of POST sleep LFP
ax[1].plot((t_lfp[seg] - t_lfp[seg][0]) * 1000, lfp[seg], "0.3", lw=0.6)
ax[1].set(xlabel="Time (ms)", ylabel="LFP (a.u.)",
          title=f"Raw CA1 LFP during POST sleep (channel {ripple_ch})")
plt.tight_layout(); plt.savefig("fig0_raw_data.png", dpi=130); plt.close()

# %% [markdown]
# ## 3. Direction-specific place fields
#
# We compute tuning curves of firing rate vs. linearized position during running
# (>10 cm/s), separately for leftward and rightward runs. We restrict analysis to the
# **track interior [0.1, 1.5] m**: the extreme boundary bins are contaminated by
# reward-consumption / turnaround firing that otherwise swamps the interior place fields
# and pins the decoder to the track ends.

# %%
LO, HI, NB = 0.10, 1.50, 40
xbins = np.linspace(LO, HI, NB)
dt = np.median(np.diff(pos.index))
xs = uniform_filter1d(pos.values, int(0.2 / dt))
vel = np.gradient(xs, pos.index)                    # signed m/s
speed = np.abs(vel)

def to_ep(mask):
    e = nap.Tsd(t=pos.index, d=mask.astype(float)).threshold(0.5, "above").time_support
    return e.drop_short_intervals(0.3)

run_R = to_ep((speed > 0.10) & (vel > 0))
run_L = to_ep((speed > 0.10) & (vel < 0))
tc_R = nap.compute_1d_tuning_curves(units, pos, nb_bins=NB, ep=run_R, minmax=(LO, HI))
tc_L = nap.compute_1d_tuning_curves(units, pos, nb_bins=NB, ep=run_L, minmax=(LO, HI))
for tc in (tc_R, tc_L):
    tc.iloc[:, :] = uniform_filter1d(tc.values, 2, axis=0)

def place_cells(tc):                                # peak >2 Hz, single localized field
    keep = []
    for j in range(tc.shape[1]):
        f = tc.values[:, j]; pk = f.max()
        if pk < 2.0:
            continue
        if (f > 0.5 * pk).sum() / NB * (HI - LO) > 1.0:
            continue
        keep.append(j)
    return np.array(keep, int)

pcR, pcL = place_cells(tc_R), place_cells(tc_L)
pc = np.union1d(pcR, pcL)
uids = np.array(tc_R.columns)[pc].astype(int)
TR = tc_R.iloc[:, pc].values + 1e-3                 # (position-bins, cells)
TL = tc_L.iloc[:, pc].values + 1e-3
print(f"Place cells: {len(pcR)} rightward, {len(pcL)} leftward, {len(pc)} union (decoding set)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, tc, pcs, name in [(axes[0], tc_R, pcR, "Rightward"), (axes[1], tc_L, pcL, "Leftward")]:
    m = tc.values[:, pcs]; order = np.argsort(np.argmax(m, 0))
    ax.imshow((m[:, order] / (m[:, order].max(0) + 1e-9)).T, aspect="auto", origin="lower",
              extent=[LO, HI, 0, len(order)], cmap="viridis")
    ax.set(xlabel="Position (m)", ylabel="Place cell (sorted by peak)",
           title=f"{name} place fields (n={len(pcs)})")
axes[2].plot(pos.index - maze.start[0], pos.values, "k", lw=0.5)
axes[2].set(xlabel="Time in maze (s)", ylabel="Position (m)", title="Behavioral trajectory")
plt.tight_layout(); plt.savefig("fig1_place_fields.png", dpi=130); plt.close()

# %% [markdown]
# The sorted place fields form a clean diagonal band tiling the track — each cell has a
# single, well-localized field — which is exactly the population code a decoder needs.

# %% [markdown]
# ## 4. Bayesian decoder and validation
#
# We use a standard memoryless Bayesian decoder (Poisson likelihood, uniform prior):
# $P(x \mid \mathbf{n}) \propto \prod_i f_i(x)^{n_i}\, e^{-\tau f_i(x)}$, where $f_i(x)$
# is cell $i$'s place field and $n_i$ its spike count in a time bin of width $\tau$.
# Before trusting it on ripples, we confirm it reconstructs the animal's **true** position
# during running.

# %%
def _posterior(N, logT, meanrate, tau):
    with np.errstate(over="ignore", invalid="ignore"):
        lp = N @ logT.T - tau * meanrate
    lp = lp - lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)

logTR, logTL, sumTR, sumTL = np.log(TR), np.log(TL), TR.sum(1), TL.sum(1)

def bayes_decode(N, tau):
    """N: (time-bins, cells) spike counts. Returns posteriors under the R and L templates."""
    return (_posterior(N, logTR, sumTR, tau), _posterior(N, logTL, sumTL, tau))

grp = units[[int(u) for u in uids]]
true_all, dec_all = [], []
example = None
for tc, rep, name in [(tc_R.iloc[:, pc], run_R, "R"), (tc_L.iloc[:, pc], run_L, "L")]:
    dec, proba = nap.decode_1d(tc, grp, rep, bin_size=0.25)
    com = (proba.values * xbins[None, :]).sum(1) / proba.values.sum(1)
    ti = np.interp(dec.index, pos.restrict(rep).index, pos.restrict(rep).values)
    m = (ti >= 0.15) & (ti <= 1.45)                 # decoder covers the interior only
    true_all.append(ti[m]); dec_all.append(dec.values[m])
    if name == "L":
        example = (dec.index - dec.index[0], ti, com)
true = np.concatenate(true_all); decm = np.concatenate(dec_all)
err = np.abs(decm - true)
print(f"RUN decoding error (interior): median {np.median(err)*100:.0f} cm on a 140 cm interior")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
Hc, _, _ = np.histogram2d(true, decm, bins=[np.linspace(LO, HI, 29)] * 2)
Hc = Hc / (Hc.sum(1, keepdims=True) + 1e-9)
im = axes[0].imshow(Hc.T, origin="lower", extent=[LO, HI, LO, HI], aspect="auto", cmap="magma")
axes[0].plot([LO, HI], [LO, HI], "w--", lw=1, alpha=0.7)
axes[0].set(xlabel="True position (m)", ylabel="Decoded position (m)",
            title=f"Decoder confusion (running)\nmedian err = {np.median(err)*100:.0f} cm")
plt.colorbar(im, ax=axes[0], label="P(decoded | true)")
et, etrue, ecom = example
sl = slice(0, 120)
axes[1].plot(et[sl], etrue[sl], "k", lw=2, label="true position")
axes[1].plot(et[sl], ecom[sl], "C3.-", ms=5, lw=0.8, label="decoded (posterior mean)")
axes[1].set(xlabel="Time (s)", ylabel="Position (m)", ylim=(0, 1.6),
            title="Example continuous decoding (leftward runs)")
axes[1].legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig("fig3_decode_validation.png", dpi=130); plt.close()

# %% [markdown]
# ## 5. Detect sharp-wave ripples during sleep
#
# We band-pass the CA1 LFP at 150–250 Hz, take the Hilbert envelope, z-score it against
# the sleep periods, and detect events crossing 5 SD (peak) with 2 SD boundaries and a
# 20–250 ms duration. We restrict detection to PRE and POST sleep.

# %%
sleep = nap.IntervalSet(start=[pre.start[0], post.start[0]], end=[pre.end[0], post.end[0]])
filt = filtfilt(_b, _a, lfp)
env = gaussian_filter1d(np.abs(hilbert(filt)), int(0.008 * fs))
env_tsd = nap.Tsd(t=t_lfp, d=env)
z = (env - env_tsd.restrict(sleep).values.mean()) / env_tsd.restrict(sleep).values.std()
z_tsd = nap.Tsd(t=t_lfp, d=z)

cand = z_tsd.restrict(sleep).threshold(2.0, "above").time_support
rs, re, rpk, rpt = [], [], [], []
for s, e in zip(cand.start, cand.end):
    seg = z_tsd.get(s, e)
    if len(seg) and seg.values.max() >= 5.0 and 0.02 <= (e - s) <= 0.25:
        rs.append(s); re.append(e); rpk.append(float(seg.values.max()))
        rpt.append(float(seg.index[np.argmax(seg.values)]))
rs, re, rpk, rpt = map(np.asarray, (rs, re, rpk, rpt))
in_post = (rpt >= post.start[0]) & (rpt <= post.end[0])
print(f"{len(rs)} ripples: {int((~in_post).sum())} PRE, {int(in_post.sum())} POST "
      f"({in_post.sum()/(post.end[0]-post.start[0]):.2f} Hz in POST)")

# figure: example ripple + summary distributions
fig = plt.figure(figsize=(14, 8)); gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
b = np.where(in_post)[0][np.argsort(rpk[in_post])[-6]]
c0 = rpt[b]; w = 0.15; sl = slice(int((c0 - w) * fs), int((c0 + w) * fs))
tt = (t_lfp[sl] - c0) * 1000
axr = fig.add_subplot(gs[0, :])
axr.plot(tt, lfp[sl], "0.3", lw=0.7, label="raw LFP")
axr.plot(tt, filt[sl] - lfp[sl].std() * 3, "C3", lw=0.7, label="150–250 Hz")
axr.axvspan((rs[b] - c0) * 1000, (re[b] - c0) * 1000, color="gold", alpha=0.3, label="detected ripple")
axr.set(xlabel="Time from ripple peak (ms)", ylabel="LFP (a.u.)",
        title=f"Example POST sharp-wave ripple (peak z = {rpk[b]:.1f})")
axr.legend(loc="upper right", fontsize=8)
for k, (data, xl_, ttl) in enumerate([
        ((re - rs)[in_post] * 1000, "Duration (ms)", "Ripple durations"),
        (rpk[in_post], "Peak envelope (SD)", "Ripple amplitudes"),
        ((rpt[in_post] - post.start[0]) / 60, "Time in POST (min)", "Occurrence in POST sleep")]):
    a = fig.add_subplot(gs[1, k]); a.hist(data, bins=30, color=f"C{k}")
    a.set(xlabel=xl_, ylabel="# ripples", title=ttl)
plt.tight_layout(); plt.savefig("fig2_ripples.png", dpi=130); plt.close()

# %% [markdown]
# ## 6. Decode trajectories within ripples and test for replay
#
# For each POST ripple we bin spikes at ~15 ms, decode a posterior over position for each
# bin, and summarize the sequence with a **weighted correlation** between decoded position
# and time (a straight sweep across the track gives $|r|\to1$). We take whichever
# directional template gives the stronger sweep. Significance comes from a **place-field
# circular-shift shuffle**: randomly rotating each cell's place field along the track
# destroys the spatial sequence while preserving firing statistics; an event is "replay"
# if its $|r|$ exceeds the 95th percentile of 300 shuffles.

# %%
def wcorr(P):
    T = P.shape[0]; tv = np.arange(T); sw = P.sum()
    mt = (P * tv[:, None]).sum() / sw; mx = (P * xbins[None, :]).sum() / sw
    ct = (P * (tv[:, None] - mt) * (xbins[None, :] - mx)).sum() / sw
    vt = (P * (tv[:, None] - mt) ** 2).sum() / sw; vx = (P * (xbins[None, :] - mx) ** 2).sum() / sw
    return 0.0 if vt <= 0 or vx <= 0 else ct / np.sqrt(vt * vx)

def event_counts(s, e):
    nb = max(5, int(round((e - s) / 0.015)))
    edges = np.linspace(s, e, nb + 1); N = np.zeros((nb, len(uids))); nact = 0
    for j, u in enumerate(uids):
        sp = spikes[u]; sp = sp[(sp >= s) & (sp < e)]
        if sp.size:
            nact += 1; N[:, j] = np.histogram(sp, bins=edges)[0]
    return N, nact

NSH = 300
recs = []
base = np.arange(NB)[:, None]
post_idx = np.where(in_post)[0]
for i in tqdm(post_idx, desc="replay", mininterval=5):
    N, nact = event_counts(rs[i], re[i])
    if nact < 5 or N.sum() < 10:                     # require a well-populated event
        continue
    T = N.shape[0]; tau = (re[i] - rs[i]) / T
    PR, PL = bayes_decode(N, tau)
    rR, rL = wcorr(PR), wcorr(PL)
    P, robs, d = (PR, rR, "R") if abs(rR) >= abs(rL) else (PL, rL, "L")
    shuf = np.empty(NSH)
    for k in range(NSH):
        off = rng.integers(0, NB, len(uids)); ii = (base - off[None, :]) % NB
        TRs = np.take_along_axis(TR, ii, 0); TLs = np.take_along_axis(TL, ii, 0)
        pr = _posterior(N, np.log(TRs), TRs.sum(1), tau)
        pl = _posterior(N, np.log(TLs), TLs.sum(1), tau)
        shuf[k] = max(abs(wcorr(pr)), abs(wcorr(pl)))
    pval = (1 + np.sum(shuf >= abs(robs))) / (NSH + 1)
    mapx = xbins[np.argmax(P, 1)]
    recs.append(dict(idx=i, start=rs[i], end=re[i], peak=rpk[i], ncells=nact,
                     nspikes=int(N.sum()), T=T, r=robs, direction=d, pval=pval,
                     span=float(mapx.max() - mapx.min())))
res = pd.DataFrame(recs)
n_sig = int((res.pval < 0.05).sum())
print(f"\n{n_sig}/{len(res)} POST ripples show significant replay "
      f"({100*n_sig/len(res):.1f}%; chance 5%); median trajectory span "
      f"{res[res.pval<0.05].span.median()*100:.0f} cm")

# %% [markdown]
# ## 7. Example replay events
#
# Each panel shows a decoded posterior sweeping across the track (top, with the fitted
# line) and the corresponding CA1 spikes ordered by place-field location (bottom). The
# ordered, diagonal firing during a ~100 ms ripple is a compressed replay of a track
# trajectory. Both **forward** (same order as running) and **reverse** replays appear.

# %%
pfR = xbins[np.argmax(TR, 0)]; pfL = xbins[np.argmax(TL, 0)]
sig = res[(res.pval < 0.05) & (res.ncells >= 8)].copy()
sig["q"] = sig.r.abs() * np.log(sig.nspikes)
sig = sig.sort_values("q", ascending=False)
picks = pd.concat([sig[sig.direction == "R"].head(2),
                   sig[sig.direction == "L"].head(2)]).sort_values("start")

fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.45, wspace=0.35)
for k, ev in enumerate(picks.itertuples()):
    s, e, d = ev.start, ev.end, ev.direction; dur = e - s
    nb = max(5, int(round(dur / 0.02))); N, _ = event_counts(s, e)
    N = np.zeros((nb, len(uids)))
    edges = np.linspace(s, e, nb + 1)
    for j, u in enumerate(uids):
        spk = spikes[u]; spk = spk[(spk >= s) & (spk < e)]
        if spk.size: N[:, j] = np.histogram(spk, bins=edges)[0]
    tau = dur / nb; PR, PL = bayes_decode(N, tau)
    P = PR if d == "R" else PL; pf = pfR if d == "R" else pfL
    axP = fig.add_subplot(gs[0, k])
    axP.imshow(P.T, aspect="auto", origin="lower", extent=[0, dur * 1000, LO, HI], cmap="hot")
    tv = np.arange(nb); sw = P.sum()
    mt = (P * tv[:, None]).sum() / sw; mx = (P * xbins[None, :]).sum() / sw
    b_ = (P * (tv[:, None] - mt) * (xbins[None, :] - mx)).sum() / (P * (tv[:, None] - mt) ** 2).sum()
    axP.plot(np.linspace(0, dur * 1000, nb), mx + b_ * (tv - mt), "c-", lw=2, label=f"r={ev.r:+.2f}")
    kind = "Forward" if (ev.r > 0) == (d == "R") else "Reverse"
    axP.set(xlabel="Time in ripple (ms)", ylabel="Decoded pos (m)",
            title=f"{kind} replay ({d}), p={ev.pval:.3f}")
    axP.legend(fontsize=8, loc="upper right")
    axR = fig.add_subplot(gs[1, k])
    for row, j in enumerate(np.argsort(pf)):
        spk = spikes[uids[j]]; spk = spk[(spk >= s) & (spk < e)]
        if spk.size: axR.plot((spk - s) * 1000, np.full(spk.size, row), "|", color="k", ms=5, mew=1)
    axR.set(xlim=(0, dur * 1000), ylim=(-1, len(pf)), xlabel="Time in ripple (ms)",
            ylabel="Cell (by field pos)", title="Spike raster (place-field order)")
fig.suptitle("Hippocampal replay: decoded trajectories during POST-sleep sharp-wave ripples", fontsize=13)
plt.savefig("fig4_replay_examples.png", dpi=130); plt.close()

# %% [markdown]
# ## 8. Population summary

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
frac = 100 * (res.pval < 0.05).mean()
ax[0].bar(["observed", "chance"], [frac, 5], color=["C3", "0.6"])
ax[0].axhline(5, ls="--", color="k", lw=0.8)
ax[0].set(ylabel="% ripples with significant replay",
          title=f"Replay prevalence\n{n_sig}/{len(res)} = {frac:.0f}% (chance 5%)")
ax[1].hist(res[res.pval >= 0.05].r.abs(), bins=25, density=True, alpha=0.6, color="0.6", label="n.s.")
ax[1].hist(res[res.pval < 0.05].r.abs(), bins=25, density=True, alpha=0.7, color="C3", label="p<0.05")
ax[1].set(xlabel="|weighted correlation|", ylabel="density", title="Sequence score distribution")
ax[1].legend()
sd = res[res.pval < 0.05]
ax[2].hist(sd.span * 100, bins=25, color="C0")
ax[2].axvline(sd.span.median() * 100, color="k", ls="--", lw=1, label=f"median {sd.span.median()*100:.0f} cm")
ax[2].set(xlabel="Decoded trajectory span (cm)", ylabel="# replay events", title="Distance represented per replay")
ax[2].legend()
plt.tight_layout(); plt.savefig("fig5_replay_summary.png", dpi=130); plt.close()

# %% [markdown]
# ## Conclusion
#
# Working entirely from real DANDI CA1 recordings, we reconstructed place fields on a
# linear track, validated a Bayesian decoder (~20 cm median error on a 140 cm interior),
# detected sharp-wave ripples in post-run sleep at a physiological ~0.3 Hz rate, and
# decoded the spatial trajectory represented inside each ripple. Roughly **one in five
# POST-sleep ripples contained a statistically significant, sequential trajectory** —
# a fourfold enrichment over the 5% chance level from place-field shuffles — with
# individual events sweeping a median of ~1.1 m of the track in ~100 ms, in both forward
# and reverse order. This is hippocampal replay: the compressed reactivation of learned
# spatial trajectories during sharp-wave ripples, the proposed neural substrate of memory
# consolidation.
