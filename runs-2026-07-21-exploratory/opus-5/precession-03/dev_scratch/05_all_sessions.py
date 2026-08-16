"""Run the precession pipeline over every session in DANDI:000044."""
import numpy as np, pandas as pd, pickle, time, zlib
import pynapple as nap
from tqdm import tqdm
import precession_lib as pl

MIN_SPIKES, MIN_PEAK, MIN_SI, MIN_PASSES = 50, 1.0, 0.5, 10
EDGE_FRAC = 0.075          # of track length; linear tracks only
FIELD_FRAC = (0.10, 0.55)  # field width as a fraction of track length


def analyze_session(name, asset):
    h5 = pl.open_session(asset)
    pos, dt, maze = pl.load_position(h5)
    units = pl.load_units(h5)
    eps, vel = pl.run_epochs(pos, dt, maze)
    m0, m1 = pos.index.values[0], pos.index.values[-1]
    ch, chans, ratios = pl.pick_theta_channel(h5, m0)
    lfp = pl.load_lfp_channel(h5, ch, m0, m1)
    filt, phase = pl.theta_phase(lfp)
    exc = units[(units.cell_type == "excitatory")]
    L = maze["length"]

    rows, store, tcs = [], {}, {}
    for d in (1, -1):
        ep = eps[d]
        if len(ep) == 0 or ep.tot_length() < 20:
            continue
        tc = pl.rate_maps(exc, pos, ep, maze)
        occ, edges = pl.occupancy(pos, ep, dt, maze)
        si = pl.spatial_info(tc, occ)
        tcs[d] = (tc, edges, occ, si)
        for u in tc.columns:
            r = tc[u].values
            if r.max() < MIN_PEAK or si[u] < MIN_SI:
                continue
            fld = pl.detect_field(r, edges, maze)
            if fld is None:
                continue
            lo, width, pk = fld
            if not (FIELD_FRAC[0] * L <= width <= FIELD_FRAC[1] * L):
                continue
            if not maze["circular"] and (lo < EDGE_FRAC * L or
                                         lo + width > L - EDGE_FRAC * L):
                continue
            spk = exc[u].restrict(ep)
            sp_pos, sp_ph = spk.value_from(pos), spk.value_from(phase)
            m, _ = pl.in_field(sp_pos.values, lo, width, maze)
            if m.sum() < MIN_SPIKES:
                continue
            # number of separate traversals that fired at least one in-field spike
            st_in = spk.index.values[m]
            n_passes = int(np.sum([np.any((st_in >= s) & (st_in <= e))
                                   for s, e in zip(ep.start, ep.end)]))
            if n_passes < MIN_PASSES:
                continue
            x = pl.field_fraction(sp_pos.values[m], lo, width, d, maze)
            ph = sp_ph.values[m]
            res = pl.circlin_pvalue(x, ph, n_shuf=200,
                                    rng=np.random.default_rng(zlib.crc32(f"{name}-{u}-{d}".encode())))
            res.update(session=name, unit=int(u), direction=d, peak=float(r.max()),
                       si=si[u], lo=lo, width=width, n_runs=len(ep), n_passes=n_passes,
                       maze=maze["name"], track_len=L, circular=maze["circular"])
            rows.append(res)
            store[(int(u), d)] = (x, ph, spk.index.values[m])
    return dict(name=name, rows=rows, store=store, tcs=tcs, eps=eps, dt=dt,
                maze=maze, theta_ch=ch, ch_ratios=(chans, ratios),
                n_exc=len(exc), n_units=len(units))


if __name__ == "__main__":
    assets = pl.resolve_assets()
    out, t0 = {}, time.time()
    for name, aid in tqdm(sorted(assets.items()), desc="sessions"):
        r = analyze_session(name, aid)
        sig = sum(1 for x in r["rows"] if x["p"] < 0.05)
        print(f"{name}: {r['maze']['name']} L={r['maze']['length']}m, "
              f"{r['n_exc']} exc units, {len(r['rows'])} fields, {sig} sig, "
              f"theta ch {r['theta_ch']}, {time.time()-t0:.0f}s", flush=True)
        out[name] = r
    df = pd.concat([pd.DataFrame(r["rows"]) for r in out.values()], ignore_index=True)
    df.to_csv("all_fields.csv", index=False)
    with open("all_sessions.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(df.groupby("session")[["slope", "rho"]].median())
    sig = df[df.p < 0.05]
    print(f"\nTOTAL {len(df)} fields, {len(sig)} significant, "
          f"{(sig.slope < 0).mean()*100:.0f}% negative, median slope {sig.slope.median():.2f}")
    print(df.groupby("direction").slope.describe())
