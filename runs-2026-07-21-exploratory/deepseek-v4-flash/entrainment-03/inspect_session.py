# %% [markdown]
# # Inspect IBL session structure
# %%
import sys
sys.path.insert(0, ".")
import session_config as sc

asset_id = "f791a116-1e6c-4d6a-a9eb-fe3644737be2"
nwbfile, io = sc.open_nwb(asset_id)
print(nwbfile)

# %%
print("=== INTERVALS ===")
for name, iv in nwbfile.intervals.items():
    print(name, len(iv))
if len(nwbfile.intervals) and "interval_trials" in nwbfile.intervals:
    tr = nwbfile.intervals["interval_trials"]
    print("trials columns:", list(tr.colnames))
    print("n trials:", len(tr))

print("=== UNITS ===")
u = nwbfile.units
print("n units:", len(u))
print("columns:", list(u.colnames))

# brain region related columns
for c in u.colnames:
    if "brain" in c.lower() or "location" in c.lower() or "acronym" in c.lower():
        try:
            vals = u[c][:]
            print(c, "dtype", vals.dtype, "n unique:", len(set(vals.tolist()[:])),
                  list(set(vals.tolist()[:]))[:12])
        except Exception as e:
            print(c, "ERR", e)

print("=== ELECTRODES ===")
el = nwbfile.electrodes
print("n electrodes:", len(el))
print("columns:", list(el.colnames))
for c in el.colnames:
    if any(k in c.lower() for k in ["brain", "region", "location", "depth", "x", "y", "z"]):
        try:
            vals = el[c][:]
            print(c, "sample:", vals[:5])
        except Exception as e:
            print(c, "ERR", e)

# %%
# processing modules
print("=== PROCESSING ===")
proc_names = list(nwbfile.processing.keys()) if hasattr(nwbfile, "processing") else []
print(proc_names)
for pn in proc_names:
    mod = nwbfile.processing[pn]
    for dn, d in mod.data_interfaces.items():
        print(" ", pn, "->", dn, type(d))