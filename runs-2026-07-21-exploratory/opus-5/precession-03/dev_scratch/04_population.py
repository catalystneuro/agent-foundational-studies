"""Population precession across place cells; prototype on one session."""
import numpy as np, matplotlib.pyplot as plt, pandas as pd, time
import pynapple as nap
from tqdm import tqdm
import precession_lib as pl

MIN_SPIKES = 50
MIN_PEAK = 1.0
MIN_SI = 0.5
EDGE_MARGIN = 0.12      # m, exclude fields touching the track ends
FIELD_W = (0.15, 0.90)  # m

t_start = time.time()
assets = pl.resolve_assets()
h5 = pl.open_session(assets["Achilles-10252013"])
pos, dt = pl.load_position(h5)
units = pl.load_units(h5)
eps, vel = pl.run_epochs(pos, dt)
maze0, maze1 = pos.index.values[0], pos.index.values[-1]
ch, _, _ = pl.pick_theta_channel(h5, maze0)
lfp = pl.load_lfp_channel(h5, ch, maze0, maze1)
filt, phase = pl.theta_phase(lfp)
exc = units[(units.cell_type == "excitatory")]
print("setup", time.time() - t_start)

rows, spike_store = [], {}
for d in (1, -1):
    ep = eps[d]
    tc = pl.rate_maps(exc, pos, ep)
    occ, edges = pl.occupancy(pos, ep, dt)
    si = pl.spatial_info(tc, occ)
    for u in tqdm(tc.columns, desc=f"dir {d}"):
        r = tc[u].values
        if r.max() < MIN_PEAK or si[u] < MIN_SI:
            continue
        fld = pl.detect_field(r, edges)
        if fld is None:
            continue
        lo, hi, pk = fld
        if lo < EDGE_MARGIN or hi > pl.TRACK_LEN - EDGE_MARGIN:
            continue
        if not (FIELD_W[0] <= hi - lo <= FIELD_W[1]):
            continue
        spk = exc[u].restrict(ep)
        sp_pos = spk.value_from(pos)
        sp_ph = spk.value_from(phase)
        m = (sp_pos.values >= lo) & (sp_pos.values <= hi)
        if m.sum() < MIN_SPIKES:
            continue
        x = pl.field_fraction(sp_pos.values[m], lo, hi, d)
        ph = sp_ph.values[m]
        res = pl.circlin_pvalue(x, ph, n_shuf=200,
                                rng=np.random.default_rng(int(u) * 10 + d))
        res.update(unit=int(u), direction=d, peak=float(r.max()), si=si[u],
                   width=hi - lo, com=0.5 * (lo + hi))
        rows.append(res)
        spike_store[(int(u), d)] = (x, ph)

df = pd.DataFrame(rows)
print(df.describe())
sig = df[df.p < 0.05]
print(f"\n{len(df)} place fields, {len(sig)} significant (p<0.05), "
      f"{(sig.slope < 0).mean() * 100:.0f}% of significant have negative slope")
print("median slope (sig):", sig.slope.median(), "median rho:", sig.rho.median())
df.to_csv("proto_fields.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].hist(df.slope, bins=np.arange(-2.5, 2.6, 0.15), color="lightgray", label="all fields")
ax[0].hist(sig.slope, bins=np.arange(-2.5, 2.6, 0.15), color="C3", label="p<0.05")
ax[0].axvline(0, color="k", lw=0.8); ax[0].legend()
ax[0].set(xlabel="slope (theta cycles / field)", ylabel="count")
ax[1].hist(df.rho, bins=20, color="lightgray"); ax[1].hist(sig.rho, bins=20, color="C3")
ax[1].axvline(0, color="k", lw=0.8); ax[1].set(xlabel="circular-linear rho")
X = np.concatenate([v[0] for v in spike_store.values()])
P = np.concatenate([v[1] for v in spike_store.values()])
ax[2].hist2d(np.r_[X, X], np.degrees(np.r_[P, P + 2 * np.pi]), bins=[25, 40], cmap="magma")
ax[2].set(xlabel="normalised position in field", ylabel="theta phase (deg)",
          title="all place-field spikes pooled")
plt.tight_layout(); plt.savefig("check_population.png", dpi=110)
print("elapsed", time.time() - t_start)
