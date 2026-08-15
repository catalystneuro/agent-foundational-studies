import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pynapple as nap
from load_data import load_cache
from replay import bayesian_decode, weighted_corr, score_event_shuffle

d = load_cache()
post = nap.IntervalSet(*d["epochs"]["POSTEpoch"])

# ripples + place fields from previous steps
R = np.load("ripples_post.npz")
rip = nap.IntervalSet(start=R["start"], end=R["end"])
pk_t = R["peak_t"]
P = np.load("placefields.npz", allow_pickle=True)
tc = P["tc"]                      # (bins, all pyr units in tc.columns order)
cols = P["columns"]
centers = P["centers"]
place_units = P["place_units"]
# subset tuning to place cells, ordered by peak
col_index = {int(c): i for i, c in enumerate(cols)}
pu = [int(u) for u in place_units]
tuning = tc[:, [col_index[u] for u in pu]]   # (bins, n_place)
peak_loc = centers[np.argmax(tuning, axis=0)]
order = np.argsort(peak_loc)
pu_ord = [pu[i] for i in order]
tuning_ord = tuning[:, order]

# spike trains of place cells
spikes = {int(u): nap.Ts(d["spikes"][u]) for u in pu_ord}
grp = nap.TsGroup(spikes)

# ---- 1) Population reactivation around ripples (PSTH / perievent raster) ----
pk = nap.Ts(pk_t)
# population MUA rate
allspk = np.sort(np.concatenate([d["spikes"][u] for u in pu_ord]))
mua = nap.Ts(allspk)
peth = nap.compute_perievent(mua, pk, minmax=(-0.5, 0.5))
# build PSTH histogram
binsz = 0.01
edges = np.arange(-0.5, 0.5 + binsz, binsz)
allrel = np.concatenate([np.asarray(peth[i].index.values) for i in peth.index])
psth, _ = np.histogram(allrel, bins=edges)
psth = psth / (len(pk_t) * binsz)   # Hz per ripple
print("baseline MUA ~%.0f Hz, peak ~%.0f Hz" % (
    np.median(psth[:20]), psth.max()))

# ---- 2) Bayesian replay decoding within each ripple ----
bin_s = 0.02
def decode_event(ep):
    ct = grp.count(bin_s, ep=ep)
    if ct.shape[0] < 3:
        return None, None
    post_p = bayesian_decode(ct.values, tuning_ord, bin_s)
    return post_p, ct.index.values

# score all ripples with enough spikes
rng = np.random.default_rng(1)
scores, pcts, nspk = [], [], []
posts = []
for i in range(len(rip)):
    ep = nap.IntervalSet(rip.start[i], rip.end[i])
    pp, tt = decode_event(ep)
    if pp is None:
        scores.append(np.nan); pcts.append(np.nan); nspk.append(0); posts.append(None)
        continue
    total_spikes = int(grp.restrict(ep).count().values.sum())
    nspk.append(total_spikes)
    if total_spikes < 5:
        scores.append(np.nan); pcts.append(np.nan); posts.append(None); continue
    obs, pct, _ = score_event_shuffle(pp, centers, n_shuffle=250, rng=rng)
    scores.append(obs); pcts.append(pct); posts.append(pp)

scores = np.array(scores); pcts = np.array(pcts); nspk = np.array(nspk)
valid = ~np.isnan(scores)
sig = valid & (pcts >= 0.95)
print("scored ripples:", valid.sum(), " significant replay (p<.05):", sig.sum(),
      "(%.0f%%)" % (100 * sig.sum() / max(1, valid.sum())))

# ---- Figure ----
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

axp = fig.add_subplot(gs[0, :])
axp.bar(edges[:-1] * 1e3, psth, width=binsz * 1e3, color="C0", align="edge")
axp.axvline(0, color="C3", ls="--")
axp.set_xlabel("time from ripple peak (ms)"); axp.set_ylabel("population rate (Hz)")
axp.set_title("CA1 pyramidal population reactivation around ripples (n=%d)" % len(pk_t))

# example replay events: pick top-scoring significant
best = np.where(sig)[0]
best = best[np.argsort(scores[best])[::-1]][:3]
for j, ev in enumerate(best):
    ax = fig.add_subplot(gs[1, j])
    pp = posts[ev]
    ax.imshow(pp.T, aspect="auto", origin="lower",
              extent=[0, pp.shape[0] * bin_s * 1e3, centers[0], centers[-1]],
              cmap="hot")
    ax.set_xlabel("time in ripple (ms)"); ax.set_ylabel("decoded pos (m)")
    ax.set_title("Replay r=%.2f (p=%.3f)" % (scores[ev], 1 - pcts[ev]), fontsize=9)

# raster of one example event (ordered by place-field location)
ax = fig.add_subplot(gs[2, 0])
ev = best[0]
ep = nap.IntervalSet(rip.start[ev] - 0.05, rip.end[ev] + 0.05)
for row, u in enumerate(pu_ord):
    st = grp[u].restrict(ep).index.values
    ax.plot((st - rip.start[ev]) * 1e3, np.full_like(st, row), "|k", ms=4)
ax.set_xlabel("time in ripple (ms)"); ax.set_ylabel("cell (by place order)")
ax.set_title("Ordered spike raster (replay event)", fontsize=9)

# score distribution vs shuffle expectation
ax = fig.add_subplot(gs[2, 1])
ax.hist(scores[valid], bins=25, color="C0", alpha=0.8, label="observed")
ax.set_xlabel("weighted corr |r|"); ax.set_ylabel("count")
ax.set_title("Replay scores (%d/%d sig.)" % (sig.sum(), valid.sum()), fontsize=9)

ax = fig.add_subplot(gs[2, 2])
ax.hist(pcts[valid], bins=20, color="C2")
ax.axvline(0.95, color="C3", ls="--", label="p=0.05")
ax.set_xlabel("shuffle percentile"); ax.set_ylabel("count")
ax.set_title("Significance vs shuffle", fontsize=9); ax.legend(fontsize=7)
plt.savefig("fig_replay.png", dpi=130)
print("saved fig_replay.png")

np.savez("replay_scores.npz", scores=scores, pcts=pcts, nspk=nspk, psth=psth,
         edges=edges)
