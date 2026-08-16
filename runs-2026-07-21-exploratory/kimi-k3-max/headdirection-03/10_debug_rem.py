"""Debug wake vs REM preferred directions for HD cells."""
import numpy as np
import pynapple as nap
from hd_analysis import (load_session, compute_head_direction, get_epochs,
                         analyze_session, valid_hd_samples)

ASSET_ID = "4cc64fe0-7b1e-404c-8b86-fb5659292830"
nwb, io = load_session(ASSET_ID)
hd = compute_head_direction(nwb)
wake, rem, nrem = get_epochs(nwb)
res = analyze_session(nwb, n_shuf=100, seed=0, verbose=False)

hd_cells = np.where(res["is_hd"])[0]
print(f"{'key':>4} {'mvl':>6} {'pref_w':>7} {'pref_rem':>8} {'mvl_rem':>7} "
      f"{'cnt_rem':>8} {'pref_nrem':>9} {'mvl_nrem':>8} {'cnt_nrem':>8}")
for i in hd_cells:
    print(f"{res['keys'][i]:>4} {res['mvl'][i]:>6.2f} "
          f"{np.degrees(res['pref_wake'][i]):>7.0f} "
          f"{np.degrees(res['pref_rem'][i]):>8.0f} {res['mvl_tc_rem'][i]:>7.2f} "
          f"{res['cnt_rem'][i]:>8.0f} "
          f"{np.degrees(res['pref_nrem'][i]):>9.0f} {res['mvl_tc_nrem'][i]:>8.2f} "
          f"{res['cnt_nrem'][i]:>8.0f}")

# REM HD occupancy coverage
hd_t_r, hd_d_r = valid_hd_samples(hd, rem)
print("\nREM valid HD samples:", len(hd_d_r),
      f"over {rem.tot_length():.0f}s REM")
occ, _ = np.histogram(hd_d_r, bins=60, range=(0, 2 * np.pi))
print("REM occupancy: min", occ.min(), "max", occ.max(),
      "| frac bins with <10 samples:", np.mean(occ < 10))
print("REM occupancy resultant:", np.abs(np.exp(1j*hd_d_r).mean()))

# NaN fraction in REM tracking
hd_rem = hd.restrict(rem)
print("REM NaN frac:", np.isnan(hd_rem.values).mean())

# also: spike-angle based pref in REM (bypass tuning curve)
units = nwb["units"]
rates = units.metadata["rate"].values
units_f = units[rates > 0.1]
keys = list(units_f.keys())
print("\nspike-angle pref in REM (direct):")
for i in hd_cells:
    u = keys[i]
    sp = units_f[u].restrict(rem)
    if len(sp) < 50:
        continue
    ang = np.interp(sp.t, hd_t_r, hd_d_r)
    z = np.exp(1j * ang).mean()
    print(f"unit {u}: n={len(sp)} pref_rem_direct={np.degrees(np.angle(z))%360:.0f} "
          f"mvl={np.abs(z):.2f} | pref_wake={np.degrees(res['pref_wake'][i]):.0f}")
