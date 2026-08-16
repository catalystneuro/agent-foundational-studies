import numpy as np, warnings; warnings.filterwarnings("ignore")
import pynapple as nap, place_lib as pl

def load_behavior(nwbfile, max_gap_s=0.5, min_coverage=0.6):
    lin_ss = pl._spatial_series(nwbfile, "Linearized")
    xy_ss = pl._spatial_series(nwbfile, "Spatial")
    period = float(lin_ss.rate); t0 = float(lin_ss.starting_time)
    lin = np.asarray(lin_ss.data[:]).squeeze().astype(float)
    xy = np.asarray(xy_ss.data[:]); t = t0 + np.arange(lin.size) * period
    track_len = float(np.nanmax(lin))

    ok = np.isfinite(lin)
    # bridge tracking dropouts shorter than max_gap_s so one traversal stays one block
    idx = np.flatnonzero(ok)
    filled = ok.copy()
    max_gap = int(round(max_gap_s / period))
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            lin[a+1:b] = np.interp(np.arange(a+1, b), [a, b], [lin[a], lin[b]])
            filled[a+1:b] = True

    e = np.flatnonzero(np.diff(filled.astype(int)))
    starts = np.r_[0 if filled[0] else [], e[filled[e+1]] + 1].astype(int)
    stops = np.r_[e[~filled[e+1]] + 1, lin.size if filled[-1] else []].astype(int)
    runs = []
    for a, b in zip(np.atleast_1d(starts), np.atleast_1d(stops)):
        d = lin[b-1] - lin[a]
        if abs(d) >= min_coverage * track_len:
            runs.append((t[a], t[b-1], np.sign(d), lin[a], lin[b-1], xy[a,0], xy[b-1,0]))
    return np.array(runs), lin, t, filled, track_len

nwbfile = pl.open_nwb("Achilles_10252013")
runs, lin, t, filled, L = load_behavior(nwbfile)
print("kept traversals:", len(runs), " right:", (runs[:,2]>0).sum(), " left:", (runs[:,2]<0).sum())
print("median duration %.2f s" % np.median(runs[:,1]-runs[:,0]))
for r in runs[:6]:
    print(f"t {r[0]:.1f}-{r[1]:.1f} dir {int(r[2]):+d} lin {r[3]:.2f}->{r[4]:.2f} x {r[5]:.2f}->{r[6]:.2f}")
