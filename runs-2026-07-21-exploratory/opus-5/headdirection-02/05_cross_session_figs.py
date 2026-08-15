"""Summary figure over all 31 sessions of DANDI:000939."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

units_df = pd.read_csv("all_units.csv")
sess = pd.read_csv("session_summary.csv")

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32)

# --- yield per session
ax = fig.add_subplot(gs[0, 0])
o = sess.sort_values("frac_hd")
y = np.arange(len(o))
ax.barh(y, 100 * o.frac_hd, color="crimson", height=0.75, label="direction-modulated")
ax.plot(100 * o.n_author_hd / o.n_units, y, "o", ms=3.5, color="k",
        label="author HD label")
ax.set(yticks=y, yticklabels=o.session, ylabel="session",
       xlabel="% of recorded units", title="Head-direction cells per session")
ax.tick_params(axis="y", labelsize=6)
ax.legend(fontsize=7, loc="lower right")

# --- pooled tuning strength
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, 1, 41)
ax.hist(units_df.mvl[~units_df.hd_cell], bins=bins, color="0.6", label="not significant")
ax.hist(units_df.mvl[units_df.hd_cell], bins=bins, color="crimson", alpha=0.85,
        label="direction-modulated")
ax.axvline(0.3, ls="--", color="k", lw=0.8)
ax.text(0.31, ax.get_ylim()[1] * 0.55, "MVL = 0.3", fontsize=7)
ax.set(xlabel="mean vector length", ylabel="units",
       title=f"All {len(units_df)} units, {len(sess)} sessions")
ax.legend(fontsize=8)

# --- pooled preferred directions
ax = fig.add_subplot(gs[0, 2], projection="polar")
sel = units_df[units_df.hd_cell & (units_df.mvl > 0.3)]
ax.hist(sel.pref_dir, bins=np.linspace(0, 2 * np.pi, 37), color="#1b6ca8")
ax.set_title(f"Preferred directions\n({len(sel)} cells, all sessions)", pad=20,
             fontsize=10)
ax.tick_params(labelsize=7)

# --- decoding
ax = fig.add_subplot(gs[1, 0])
ax.scatter(sess.n_decode_cells, sess.decode_err_deg, s=26, color="crimson")
ax.axhline(90, ls="--", color="0.5", lw=1)
ax.text(sess.n_decode_cells.max(), 86, "chance", ha="right", va="top", fontsize=8,
        color="0.4")
ax.set(xlabel="cells used for decoding", ylabel="median decoding error (deg)",
       ylim=(0, 100),
       title=f"Held-out decoding\n(median {sess.decode_err_deg.median():.1f}$\\degree$ "
             "across sessions)")

# --- cross-environment coherence
ax = fig.add_subplot(gs[1, 1])
cs = sess.dropna(subset=["rot_coherence"])
ax.scatter(cs.rot_coherence, cs.rot_resid_deg, s=26, color="#1b6ca8")
ax.set(xlabel="rotation coherence R", ylabel="median residual shift (deg)",
       xlim=(0.5, 1.02), ylim=(0, None),
       title=f"Square vs triangle arena\n({len(cs)} sessions with both)")

# --- agreement with the published labels
ax = fig.add_subplot(gs[1, 2])
ax.hist(100 * sess.agree_exc, bins=np.arange(80, 102, 2), color="0.5")
ax.axvline(100 * sess.agree_exc.median(), color="crimson", lw=2)
ax.set(xlabel="agreement with author labels (%)", ylabel="sessions",
       title=f"Median agreement {100*sess.agree_exc.median():.0f}%\n(excitatory cells only)")

fig.suptitle("Head-direction coding across all 31 sessions of DANDI:000939", y=0.97)
fig.savefig("fig07_cross_session.png", dpi=150, bbox_inches="tight")
print("wrote fig07")

print(f"units: {len(units_df)}, significant: {int(units_df.hd_cell.sum())} "
      f"({100*units_df.hd_cell.mean():.0f}%), of which MVL>0.3: "
      f"{int((units_df.hd_cell & (units_df.mvl>0.3)).sum())}")
print(f"decoding error: median {sess.decode_err_deg.median():.1f} deg, "
      f"range {sess.decode_err_deg.min():.1f}-{sess.decode_err_deg.max():.1f}")
print(f"rotation coherence: median {cs.rot_coherence.median():.2f}, "
      f"residual {cs.rot_resid_deg.median():.1f} deg")
