"""Print the structure of one DANDI:000021 session so the extraction can be written against it."""
import sys

import numpy as np
import pynapple as nap

import oslib

ses = sys.argv[1] if len(sys.argv) > 1 else "715093703"
nwbfile = oslib.open_session(ses)

print("session_id:", nwbfile.session_id)
print("description:", nwbfile.session_description)
print("subject:", nwbfile.subject.subject_id, nwbfile.subject.genotype, nwbfile.subject.age)

print("\n=== interval tables ===")
for name, iv in nwbfile.intervals.items():
    print(f"  {name:48s} n={len(iv):6d}")

print("\n=== units ===")
u = nwbfile.units.to_dataframe()
print("  n units:", len(u))
print("  columns:", sorted(u.columns.tolist()))

tsg, meta = oslib.good_units(nwbfile)
print("  QC-passing units in visual areas:", len(tsg))
print(meta["area"].value_counts().to_string())

print("\n=== drifting gratings ===")
dg = oslib.stim_table(nwbfile, "drifting_gratings_presentations")
print("  n trials:", len(dg), "| columns:", dg.columns.tolist())
print("  orientations:", sorted(dg["orientation"].unique()))
print("  temporal freqs:", sorted(dg["temporal_frequency"].unique()))
print("  contrast:", sorted(dg["contrast"].unique()))
print("  duration: mean %.3f s  min %.3f  max %.3f"
      % ((dg.stop_time - dg.start_time).mean(), (dg.stop_time - dg.start_time).min(),
         (dg.stop_time - dg.start_time).max()))
print("  reps per (dir, tf):", dg.groupby(["orientation", "temporal_frequency"]).size().unique())
print("  blank sweeps:", len(oslib.blank_table(nwbfile, "drifting_gratings_presentations")))
print("  block spans %.1f - %.1f s" % (dg.start_time.min(), dg.stop_time.max()))

print("\n=== static gratings ===")
sg = oslib.stim_table(nwbfile, "static_gratings_presentations")
print("  n trials:", len(sg))
print("  orientations:", sorted(sg["orientation"].unique()))
print("  spatial freqs:", sorted(sg["spatial_frequency"].unique()))
print("  phases:", sorted(sg["phase"].unique()))
print("  duration: mean %.3f s" % (sg.stop_time - sg.start_time).mean())
print("  block spans %.1f - %.1f s" % (sg.start_time.min(), sg.stop_time.max()))

print("\n=== invalid times ===")
print(nwbfile.invalid_times.to_dataframe() if nwbfile.invalid_times is not None else "  none")

print("\n=== processing ===")
for name, mod in nwbfile.processing.items():
    print(" ", name, list(mod.data_interfaces))

print("\n=== pynapple view ===")
print(nap.NWBFile(nwbfile))
