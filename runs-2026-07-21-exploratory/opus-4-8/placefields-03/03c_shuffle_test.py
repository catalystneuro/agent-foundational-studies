import numpy as np, time, warnings; warnings.filterwarnings("ignore")
import place_lib as pl, analysis_lib as al
nwbfile = pl.open_nwb("Achilles_10252013")
beh = pl.load_behavior(nwbfile); units = pl.load_units(nwbfile)
eps = pl.epochs_of(nwbfile); maze = eps["MazeEpoch"]
pyr = units[np.flatnonzero(units.cell_type == "excitatory")]
ep = al.running_epochs(beh, "right")
t0 = time.time()
null = al.si_null(pyr, beh, ep, maze, n_shuffles=20, seed=1, progress=False)
print("20 shuffles in %.1fs" % (time.time() - t0), null.shape)
si = al.spatial_information(al.rate_maps(pyr, beh, ep))
print("null median %.3f  p95 %.3f" % (np.median(null), np.percentile(null, 95)))
print("frac obs > null p95 (per unit):", np.mean(si > np.percentile(null, 95, axis=0)))
