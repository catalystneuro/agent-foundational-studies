"""Inject the cross-session numbers into README.md at the <!--MULTI--> marker."""

import numpy as np
import pandas as pd

import multisummary as M

S, pooled, st = M.summarize()
post, pre, ctrl = pooled["POSTEpoch"], pooled["PREEpoch"], pooled["control"]
sig = post[post.significant]

rows = "\n".join(
    f"| {r.session} | {int(r.n_place)} | {int(r.n_events_POST)} | "
    f"{100 * r.frac_sig_POST:.1f}% | {100 * r.frac_sig_PREE:.1f}% | "
    f"{100 * r.frac_sig_ctrl:.1f}% |"
    for _, r in S.iterrows())

excess = 100 * (S.frac_sig_POST - S.frac_sig_ctrl)
rho = S.n_place.corr(excess, method="spearman")

block = f"""## Across sessions

The identical pipeline run on all five straight-track sessions gives the same ordering in every
one of them: POST sleep above the cell-identity control, and PRE sleep at or near it.

| session | place cells | POST events | POST significant | PRE significant | control |
| --- | --- | --- | --- | --- | --- |
{rows}

Pooled over {len(S)} sessions and {len(post)} POST candidate events, {100 * post.significant.mean():.1f}% are
significant against {100 * ctrl.significant.mean():.1f}% for the cell-identity control
(χ² p = {st['p_post_ctrl']:.2g}) and {100 * pre.significant.mean():.1f}% for PRE sleep. The per-session
comparison is consistent in sign (Wilcoxon signed-rank p = {st['p_w_post_ctrl']:.3f}, n = {len(S)}), and the
pooled |weighted correlation| is higher in POST than in the control
(Mann-Whitney p = {st['p_u_post_ctrl']:.2g}) while PRE is not (p = {st['p_u_pre_ctrl']:.2f}). Pooled replay speed is
{sig.slope.abs().median():.1f} m/s with {100 * (sig.slope > 0).mean():.0f}% forward and {100 * (sig.slope < 0).mean():.0f}% reverse events.

The size of the effect tracks the size of the recorded ensemble (Spearman rho = {rho:.2f} between
the number of place cells and the POST-minus-control excess). The Achilles session contributes
100 cells and a 2:1 excess over control; the sessions with 29 to 40 cells show the same direction
at a smaller margin. That is the expected behaviour of a population decoder rather than a
separate finding, but it does mean the single-session claim rests mainly on the largest ensemble.
"""

# Refresh the prototype-session numbers as well.
pst = np.load("cache/replay_stats.npz")
p_ = pd.read_csv("cache/replay_events_POSTEpoch.csv")
r_ = pd.read_csv("cache/replay_events_PREEpoch.csv")
c_ = pd.read_csv("cache/replay_events_POSTEpoch_idshuffle.csv")
proto = f"""The effect is well above the pipeline's own false-positive rate. In the prototype session
{100 * p_.significant.mean():.1f}% of {len(p_)} POST candidate events pass both shuffle tests, against
{100 * c_.significant.mean():.1f}% for the cell-identity shuffled control (χ² p = {float(pst['p_ctrl']):.2g};
|weighted correlation| Mann-Whitney p = {float(pst['p_u_ctrl']):.2g}). PRE-sleep ripples, recorded before the
animal had ever run the track, are statistically indistinguishable from that control on both
measures ({100 * r_.significant.mean():.1f}%, χ² p = {float(pst['p_prectrl']):.2f}; |r| p = {float(pst['p_u_prectrl']):.2f}), and POST exceeds
PRE on both (χ² p = {float(pst['p_chi']):.2g}; |r| p = {float(pst['p_u']):.2g}). That contrast is what ties the POST-sleep
sequences to the experience rather than to any standing structure in the ensemble.
"""

readme = open("README.md").read()
a = readme.index("The effect is well above")
b = readme.index("Two caveats.")
readme = readme[:a] + proto + "\n" + readme[b:]
start = readme.index("<!--MULTI-->")
end = readme.index("## Files")
open("README.md", "w").write(readme[:start] + block + "\n" + readme[end:])
print("README.md updated")
