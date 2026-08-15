"""Stage 7: cross-session summary."""
import pickle, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

files = sorted(glob.glob("cache/session_*.pkl"))
R = [pickle.load(open(f, "rb")) for f in files]
names = [r["session"] for r in R]
print("sessions:", names)

rows = []
for r in R:
    for k in ["POST", "PRE", "CONTROL"]:
        df = r["events"][k]
        ci = stats.binomtest(int(df.sig.sum()), len(df)).proportion_ci(0.95)
        rows.append(dict(session=r["session"], epoch=k, n=len(df), nsig=int(df.sig.sum()),
                         frac=100 * df.sig.mean(), lo=100 * ci.low, hi=100 * ci.high,
                         med_speed=float(df[df.sig].speed.median()) if df.sig.sum() else np.nan,
                         fwd=int((df[df.sig].wcorr > 0).sum()), rev=int((df[df.sig].wcorr < 0).sum())))
S = pd.DataFrame(rows)
S.to_csv("cache/07_summary.csv", index=False)
print(S.to_string(index=False))

pooled = {k: pd.concat([r["events"][k] for r in R], ignore_index=True) for k in ["POST", "PRE", "CONTROL"]}
p_post = stats.binomtest(int(pooled["POST"].sig.sum()), len(pooled["POST"]),
                         pooled["CONTROL"].sig.mean(), alternative="greater").pvalue
p_pre = stats.binomtest(int(pooled["PRE"].sig.sum()), len(pooled["PRE"]),
                        pooled["CONTROL"].sig.mean(), alternative="greater").pvalue
tbl = np.array([[pooled["POST"].sig.sum(), len(pooled["POST"]) - pooled["POST"].sig.sum()],
                [pooled["PRE"].sig.sum(), len(pooled["PRE"]) - pooled["PRE"].sig.sum()]])
chi2, p_pp = stats.chi2_contingency(tbl)[:2]
print(f"\nPOOLED  POST {pooled['POST'].sig.sum()}/{len(pooled['POST'])} "
      f"({100*pooled['POST'].sig.mean():.1f}%)  PRE {100*pooled['PRE'].sig.mean():.1f}%  "
      f"CONTROL {100*pooled['CONTROL'].sig.mean():.1f}%")
print(f"POST vs control p={p_post:.2e} | PRE vs control p={p_pre:.3f} | POST vs PRE chi2 p={p_pp:.2e}")

fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
cols = {"POST": "#2a9d8f", "PRE": "#8ecae6", "CONTROL": "#adb5bd"}

ax = axs[0, 0]
w = 0.26
for i, k in enumerate(["POST", "PRE", "CONTROL"]):
    sub = S[S.epoch == k]
    x = np.arange(len(R)) + (i - 1) * w
    ax.bar(x, sub.frac, w, color=cols[k], label=k.lower())
    ax.errorbar(x, sub.frac, yerr=[sub.frac - sub.lo, sub.hi - sub.frac], fmt="none", ecolor="k",
                capsize=2, lw=.8)
ax.axhline(5, color="r", ls="--", lw=1)
ax.set_xticks(range(len(R)))
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="% events with significant replay", title="Replay detection per session")
ax.legend(fontsize=8)

ax = axs[0, 1]
fr, lo, hi = [], [], []
for k in ["POST", "PRE", "CONTROL"]:
    d = pooled[k]; ci = stats.binomtest(int(d.sig.sum()), len(d)).proportion_ci(0.95)
    fr.append(100 * d.sig.mean()); lo.append(100 * ci.low); hi.append(100 * ci.high)
labels = [f"{k}\nn={len(pooled[k])}" for k in ["POST", "PRE", "CONTROL"]]
ax.bar(labels, fr, color=[cols[k] for k in ["POST", "PRE", "CONTROL"]])
ax.errorbar(labels, fr, yerr=[np.array(fr) - lo, np.array(hi) - np.array(fr)], fmt="none",
            ecolor="k", capsize=4)
ax.axhline(5, color="r", ls="--", lw=1)
ax.set(ylabel="% significant", title="Pooled across sessions",
       ylim=(0, max(hi) * 1.35))
ax.text(.03, .95, f"POST vs control  p = {p_post:.1e}\nPOST vs PRE      p = {p_pp:.1e}",
        transform=ax.transAxes, fontsize=8, va="top")

ax = axs[0, 2]
for r in R:
    d = r["events"]["POST"]
    ax.hist(d[d.sig].speed, bins=np.linspace(0, 25, 30), histtype="step", lw=1.5, label=r["session"][:12])
ax.set(xlabel="replay speed (m/s)", ylabel="events", title="Virtual speed of replayed trajectories")
ax.legend(fontsize=7)

ax = axs[1, 0]
ax.bar(range(len(R)), [100 * r["med_err"] for r in R], color="#457b9d")
ax.set_xticks(range(len(R))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="median |error| (cm)", title="Cross-validated position decoding on running laps")

ax = axs[1, 1]
ax.bar(np.arange(len(R)) - .2, [len(r["pc_ids"]) for r in R], .4, label="place cells", color="#e76f51")
ax.bar(np.arange(len(R)) + .2, [r["n_pyr"] for r in R], .4, label="pyramidal cells", color="#264653")
ax.set_xticks(range(len(R))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="units", title="Recorded population"); ax.legend(fontsize=8)

ax = axs[1, 2]
sub = S[S.epoch == "POST"]
ax.bar(np.arange(len(R)) - .2, sub.fwd, .4, label="forward", color="#e76f51")
ax.bar(np.arange(len(R)) + .2, sub.rev, .4, label="reverse", color="#264653")
ax.set_xticks(range(len(R))); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set(ylabel="significant events", title="Forward vs reverse replay (POST)"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/06_cross_session_summary.png", dpi=140)
print("saved figures/06_cross_session_summary.png")
