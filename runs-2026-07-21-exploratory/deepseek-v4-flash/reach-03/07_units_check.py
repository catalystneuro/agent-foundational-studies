"""Check electrode indexing and unit-id iteration."""
import numpy as np
import mcmaze_data as md

nwbfile, nwb = md.load_nwb()
elec_locs = np.asarray(nwbfile.electrodes["location"][:])
unit_elec = md.unit_electrode_indices()
print("unit_elec len:", len(unit_elec), "range:", unit_elec.min(), unit_elec.max())
print("elec_locs[:10]:", elec_locs[:10])
print("elec_locs[90:100]:", elec_locs[90:100])
print("elec_locs[185:195]:", elec_locs[185:195])
regions = elec_locs[unit_elec]
print("unit regions:", np.unique(regions, return_counts=True))

units = nwb["units"]
ids = list(units.index)
print("n unit ids:", len(ids), ids[:5], ids[-3:])
ts = units[ids[0]]
print("first unit spikes:", len(np.asarray(ts.t)))
print("obs_intervals of unit 0 (n intervals):", len(ts.time_support))
print("total session t span: %.1f .. %.1f" % (ts.t[0], ts.t[-1]))