"""Stage 7: run the whole pipeline on four sessions from three rats."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import pipeline as pl

SESSIONS = ["Achilles-10252013", "Gatsby-08022013", "Cicero-09172014", "Buddy-06272013"]

rows, dfs = [], {}
for s in SESSIONS:
    print("\n===", s, "===")
    summary, df, ctrl, extras = pl.run_session(s)
    summary["frac_ctrl_n"] = len(ctrl)
    rows.append(summary)
    df["session"] = s
    dfs[s] = df
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})

res = pd.DataFrame(rows)
res.to_csv("multi_session_summary.csv", index=False)
pd.concat(dfs.values()).to_csv("multi_session_events.csv", index=False)
print("\n", res[["session", "n_place", "n_ripples", "rate_nrem", "rate_rem",
                 "decoder_err_cm", "frac_pre", "frac_post", "frac_ctrl", "ev", "rev"]])

# pooled test across sessions
all_ev = pd.concat(dfs.values())
tab = [[int(all_ev[(all_ev.epoch == e) & all_ev.significant].shape[0]),
        int(all_ev[(all_ev.epoch == e) & ~all_ev.significant].shape[0])] for e in ("POST", "PRE")]
odds, p = stats.fisher_exact(tab)
print("pooled POST vs PRE: %s  OR=%.2f p=%.2g" % (tab, odds, p))

# ------------------------------------------------------------------- figure
labels = [s.split("-")[0] for s in SESSIONS]
x = np.arange(len(SESSIONS))
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

ax = axes[0, 0]
for i, (k, c) in enumerate([("rate_nrem", "C0"), ("rate_awake", "C2"), ("rate_rem", "C1")]):
    ax.bar(x + (i - 1) * 0.27, res[k], 0.27, label=k.replace("rate_", ""), color=c)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="ripple rate (Hz)", title="Ripple rate by brain state")

ax = axes[0, 1]
ax.bar(x - 0.2, res.ripple_dur_ms, 0.4, color="C0", label="duration (ms)")
ax.bar(x + 0.2, res.ripple_freq_hz, 0.4, color="C3", label="frequency (Hz)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(title="Ripple duration and frequency", ylabel="ms  /  Hz")

ax = axes[0, 2]
ax.bar(x - 0.2, res.n_place, 0.4, label="place cells", color="C0")
ax.bar(x + 0.2, res.decoder_err_cm, 0.4, label="decoder error (cm)", color="C1")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(title="Template quality", ylabel="count  /  cm")

ax = axes[1, 0]
ax.bar(x - 0.27, 100 * res.frac_pre, 0.27, label="PRE sleep", color="0.6")
ax.bar(x, 100 * res.frac_post, 0.27, label="POST sleep", color="C0")
ax.bar(x + 0.27, 100 * res.frac_ctrl, 0.27, label="cell-ID shuffle", color="0.85")
ax.axhline(5, color="C3", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="% events with significant replay",
       title="Replay is enriched in POST sleep\nin every session (pooled p=%.1g)" % p)

ax = axes[1, 1]
ax.bar(x - 0.2, res.ev, 0.4, label="EV", color="C0")
ax.bar(x + 0.2, res.rev, 0.4, label="REV", color="0.6")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend(fontsize=8, frameon=False)
ax.set(ylabel="explained variance", title="Pairwise reactivation (EV > REV)")

ax = axes[1, 2]
for s in SESSIONS:
    sig = dfs[s][dfs[s].significant]
    ax.hist(np.abs(sig.slope) / 100, bins=np.arange(0, 12, 0.75), histtype="step", lw=1.6,
            label=s.split("-")[0], density=True)
ax.legend(fontsize=8, frameon=False)
ax.set(xlabel="replay speed (m/s)", ylabel="density", title="Replay speed")

fig.suptitle("Sharp-wave ripples and replay across four sessions of DANDI:000044")
fig.savefig("fig08_multi_session.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig08")
