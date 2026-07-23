import numpy as np, warnings; warnings.filterwarnings("ignore")
import place_lib as pl, analysis_lib as al
import pynapple as nap

nwbfile = pl.open_nwb("Achilles_10252013")
beh = pl.load_behavior(nwbfile)
units = pl.load_units(nwbfile)
eps = pl.epochs_of(nwbfile)
maze = eps["MazeEpoch"]

pyr = units[np.flatnonzero(units.cell_type == "excitatory")]
print("pyramidal units:", len(pyr))

for d in ("right", "left"):
    ep = al.running_epochs(beh, d)
    print(d, "n_ep", len(ep), "tot", round(ep.tot_length(), 1))
    tc = al.rate_maps(pyr, beh, ep)
    print("  tc shape", tc.shape, "occupancy sum", float(np.sum(tc.attrs["occupancy"])))
    si = al.spatial_information(tc)
    sp = al.sparsity(tc)
    fs = al.field_stats(tc, beh["track_len"])
    r = al.split_half_stability(pyr, beh, ep)
    print("  SI  median %.2f  max %.2f" % (np.median(si), si.max()))
    print("  sparsity median %.2f" % np.nanmedian(sp))
    print("  peak rate median %.2f max %.2f" % (np.median(fs["peak_rate"]), fs["peak_rate"].max()))
    print("  width median %.2f" % np.median(fs["width"]))
    print("  stability median %.2f  frac>0.5 %.2f" % (np.nanmedian(r), np.nanmean(r > 0.5)))
