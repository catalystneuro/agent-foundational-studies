# 04_precession.py — theta phase precession: circular-linear correlation of spike phase vs position
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from achilles_loader import open_nwb

N_SHUFFLE = 1000
MIN_FIELD_SPIKES = 20
MIN_TRAV_SPIKES = 4
RNG = np.random.default_rng(7)
TWO_PI = 2 * np.pi

C = np.load("cache_preproc.npz")
F = np.load("cache_fields.npz")
pos_t, lin_v = C["pos_t"], C["lin"]
bouts = np.column_stack([C["bout_starts"], C["bout_ends"]])
bout_dir = C["bout_dir"]
maze_start = float(C["maze_start"])
lfp_t, theta_phase = C["lfp_t"], C["theta_phase"]
LFP_FS = 1.0 / np.median(np.diff(lfp_t))
cos_ph, sin_ph = np.cos(theta_phase), np.sin(theta_phase)

nwbfile, nwb, h5 = open_nwb()
units = nwb["units"]

place_keys = F["place_keys"]
centers = F["centers"]
print(f"{len(place_keys)} place cell-directions")

def theta_phase_at(t):
    """theta phase (rad, 0 = LFP peak) at times t, by linear interp of the analytic signal"""
    x = (t - maze_start) * LFP_FS
    i0 = np.floor(x).astype(int)
    frac = x - i0
    i0 = np.clip(i0, 0, len(theta_phase) - 2)
    c = cos_ph[i0] * (1 - frac) + cos_ph[i0 + 1] * frac
    s = sin_ph[i0] * (1 - frac) + sin_ph[i0 + 1] * frac
    return np.mod(np.arctan2(s, c), TWO_PI)

def spikes_lin_pos(spk_t):
    idx = np.clip(np.searchsorted(pos_t, spk_t) - 1, 0, len(pos_t) - 1)
    return lin_v[idx]

def in_bouts(t, d_bouts):
    starts, ends = d_bouts[:, 0], d_bouts[:, 1]
    idx = np.searchsorted(starts, t, side="right") - 1
    inb = np.zeros(len(t), dtype=bool)
    ok = idx >= 0
    inb[ok] = t[ok] <= ends[idx[ok]]
    return inb

def circ_lin_r(pos, ph):
    """Kempter et al. 2012 circular-linear correlation magnitude"""
    xc = np.corrcoef(pos, np.cos(ph))[0, 1]
    xs = np.corrcoef(pos, np.sin(ph))[0, 1]
    cs = np.corrcoef(np.cos(ph), np.sin(ph))[0, 1]
    denom = 1 - cs**2
    if denom <= 0 or not np.isfinite(xc + xs + cs):
        return np.nan
    return np.sqrt((xc**2 + xs**2 - 2 * xc * xs * cs) / denom)

# slope grid for sign + slope estimation (circular-linear regression, Mardia-style)
SLOPES = np.linspace(-6 * np.pi, 6 * np.pi, 4321)  # rad per normalized field length

def fit_slope(xn, ph):
    """xn in [0,1]; returns (best slope rad/unit-xn, resultant length at best slope)"""
    z = np.exp(1j * (ph[:, None] - xn[:, None] * SLOPES[None, :]))
    R = np.abs(z.mean(axis=0))
    i = int(np.argmax(R))
    return SLOPES[i], float(R[i])

records = []
trav_records = []
spike_store = {}  # (uid, d) -> (pos, phase) for plotting/pooling
for uid, d in tqdm(place_keys, desc="precession"):
    uid = int(uid)
    d = int(d)
    lo, hi, width, peak_pos = F[f"field_{uid}_{d}"]
    spk_t = np.asarray(units[uid].t)
    d_bouts = bouts[bout_dir == d]
    mask = in_bouts(spk_t, d_bouts)
    st = spk_t[mask]
    sp = spikes_lin_pos(st)
    ok = np.isfinite(sp) & (sp >= lo) & (sp <= hi)
    st, sp = st[ok], sp[ok]
    if len(st) < MIN_FIELD_SPIKES:
        continue
    ph = theta_phase_at(st)
    # normalize position to travel progress through the field (0 at entry, 1 at exit),
    # oriented by the direction of motion so precession has the same sign for both directions
    if d == +1:
        xn = (sp - lo) / (hi - lo)
    else:
        xn = (hi - sp) / (hi - lo)

    r_cl = circ_lin_r(xn, ph)
    slope, R_best = fit_slope(xn, ph)
    signed_r = np.sign(slope) * r_cl

    # phase-permutation shuffle (breaks phase-position pairing, keeps marginals)
    null = np.empty(N_SHUFFLE)
    for i in range(N_SHUFFLE):
        null[i] = circ_lin_r(xn, RNG.permutation(ph))
    p_val = (np.sum(null >= r_cl) + 1) / (N_SHUFFLE + 1)

    # per-traversal correlations
    trav_r, trav_n = [], []
    for s, e in d_bouts:
        m = (st >= s) & (st <= e)
        if m.sum() < MIN_TRAV_SPIKES:
            continue
        xn_t = xn[m]
        if xn_t.max() - xn_t.min() < 0.25:  # must span >= 1/4 of the field
            continue
        r_t = circ_lin_r(xn_t, ph[m])
        sl_t, _ = fit_slope(xn_t, ph[m])
        if np.isfinite(r_t):
            trav_r.append(np.sign(sl_t) * r_t)
            trav_n.append(int(m.sum()))
    trav_r = np.array(trav_r)

    records.append(dict(uid=uid, d=d, n_spikes=len(st), r_cl=r_cl, signed_r=signed_r,
                        slope=slope, slope_cycles=slope / TWO_PI, p_val=p_val,
                        lo=lo, hi=hi, peak_pos=peak_pos,
                        trav_mean=np.mean(trav_r) if len(trav_r) else np.nan,
                        trav_n=len(trav_r),
                        trav_frac_neg=np.mean(np.array(trav_r) < 0) if len(trav_r) else np.nan))
    spike_store[(uid, d)] = (xn, ph)
    for tr, tn in zip(trav_r, trav_n):
        trav_records.append(dict(uid=uid, d=d, r=tr, n=tn))

import json
with open("precession_results.json", "w") as f:
    json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items()} for r in records], f, indent=1)
np.savez_compressed("cache_precession.npz",
                    **{f"spk_{u}_{d}": np.column_stack([xn, ph])
                       for (u, d), (xn, ph) in spike_store.items()})

rec = records
sig_neg = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] < 0]
sig_pos = [r for r in rec if r["p_val"] < 0.05 and r["signed_r"] > 0]
print(f"\nanalyzed cell-directions: {len(rec)}")
print(f"significant (p<0.05): {len(sig_neg) + len(sig_pos)} "
      f"({len(sig_neg)} negative / precessing, {len(sig_pos)} positive)")
print(f"median signed r: {np.median([r['signed_r'] for r in rec]):.3f}")
print(f"median slope: {np.median([r['slope_cycles'] for r in rec]):.2f} cycles/field")
tr = [r["trav_mean"] for r in rec if np.isfinite(r["trav_mean"])]
print(f"per-traversal mean r: median {np.median(tr):.3f} over {len(tr)} cell-directions, "
      f"frac negative {np.mean(np.array(tr) < 0):.2f}")
