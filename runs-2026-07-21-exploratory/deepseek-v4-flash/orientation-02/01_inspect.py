"""Quick structural inspection of the session NWB file."""
import pynapple as nap

from load_data import load_session

io, nwbfile = load_session()
nwb = nap.NWBFile(nwbfile)
print("=" * 80)
print("NWB FILE OVERVIEW")
print("=" * 80)
print(nwbfile)
print()
print("Pynapple wrapper:")
print(nwb)

print("=" * 80)
print("UNITS")
print("=" * 80)
units = nwb["units"]
print(f"Number of units reported by pynapple: {len(units)}")
print("Unit metadata columns (raw units table):")
print(list(nwbfile.units.colnames))

print("=" * 80)
print("INTERVALS")
print("=" * 80)
for name in nwbfile.intervals.keys():
    table = nwbfile.intervals[name]
    print(f"\n--- {name} ---")
    print("columns:", list(table.colnames))
    print("n_rows:", len(table))
    if name == 'drifting_gratings_presentations':
        df = table.to_dataframe()
        print(df.head(12))
        print("...")
        print("orientation values:", sorted(df['orientation'].dropna().unique()))
        print("temporal_frequency values:", sorted(df['temporal_frequency'].dropna().unique()))

if 'invalid_times' in nwbfile.intervals:
    it = nwbfile.intervals['invalid_times'].to_dataframe()
    print("\ninvalid_times rows:", len(it))
    print(it.head(20))

print("=" * 80)
print("ELECTRODES")
print("=" * 80)
print("electrode columns:", list(nwbfile.electrodes.colnames))
print("location values:", sorted(nwbfile.electrodes['location'][:].tolist()))

io.close()