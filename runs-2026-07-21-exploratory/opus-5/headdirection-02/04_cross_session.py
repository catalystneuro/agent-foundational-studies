"""Run the head-direction pipeline over every session in DANDI:000939."""

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

import hd_analysis as ha
import hd_io

NB_BINS = 120
P_THRESH = 0.01
STAB_THRESH = 0.5
BIN = 0.2       # decoding bin (s)

assets = hd_io.list_assets("000939")
unit_rows, session_rows = [], []

for path, aid, size in tqdm(assets, desc="sessions"):
    subject = path.split("/")[0].replace("sub-", "")
    nwb, nwbfile, io = hd_io.open_session(aid, dandiset="000939")
    eps = ha.get_epochs(nwbfile)
    units = nwb["units"]
    md = units.metadata

    ep = eps["wake_square"]
    hd = ha.clean_head_direction(nwb["head-direction"], ep)
    tc, stats, _ = ha.session_stats(units, hd, ep, nb_bins=NB_BINS, n_draws=500)
    stats["hd_cell"] = (stats.p_shift < P_THRESH) & (stats.stability > STAB_THRESH)
    stats["author_hd"] = md["is_head_direction"].values.astype(bool)
    stats["is_fs"] = md["is_fast_spiking"].values.astype(bool)
    stats["is_exc"] = md["is_excitatory"].values.astype(bool)
    stats["session"] = subject
    unit_rows.append(stats.reset_index().rename(columns={"index": "unit"}))

    # --- decoding on a held-out half, with cells selected on the training half
    mid = ep.start[0] + ep.tot_length() / 2
    ep_tr = nap.IntervalSet(start=ep.start[0], end=mid)
    ep_te = nap.IntervalSet(start=mid, end=ep.end[-1])
    hd_tr, hd_te = hd.restrict(ep_tr), hd.restrict(ep_te)
    _, st_tr, _ = ha.session_stats(units, hd_tr, ep_tr, nb_bins=60, n_draws=300)
    sel = list(st_tr.index[(st_tr.p_shift < P_THRESH) & (st_tr.stability > 0)])

    med_err = np.nan
    if len(sel) >= 2:
        tcx = nap.compute_tuning_curves(units[sel], hd_tr, bins=60,
                                        range=[(0, 2 * np.pi)], epochs=ep_tr)
        dec, _ = nap.decode_bayes(tcx, units[sel], ep_te, BIN)
        true = hd_te.bin_average(BIN, ep_te)
        ok = ~np.isnan(true.values)
        med_err = float(np.degrees(np.median(
            np.abs(ha.circ_diff(dec.values[ok], true.values[ok])))))

    # --- second environment, when the session has one without optogenetics
    R = offset = resid = np.nan
    n_both = 0
    if "wake_triangle" in eps:
        ep2 = eps["wake_triangle"]
        hd2 = ha.clean_head_direction(nwb["head-direction"], ep2)
        tc2 = ha.tuning_curves(units, hd2, ep2, nb_bins=NB_BINS)
        mvl2, pref2 = ha._mvl_and_pref(tc2)
        mvl2 = pd.Series(mvl2, index=tc2.columns)
        pref2 = pd.Series(pref2, index=tc2.columns)
        both = [i for i in stats.index[stats.hd_cell]
                if mvl2[i] > 0.3 and stats.mvl[i] > 0.3]
        n_both = len(both)
        if n_both >= 5:
            R, offset = ha.rotation_coherence(stats.pref_dir[both].values,
                                              pref2[both].values)
            d = ha.circ_diff(pref2[both].values, stats.pref_dir[both].values)
            resid = float(np.degrees(np.median(np.abs(ha.circ_diff(d, offset)))))

    exc = stats[stats.is_exc]
    session_rows.append(dict(
        session=subject, file=path.split("/")[-1],
        duration_s=float(ep.tot_length()), n_units=len(stats),
        n_hd=int(stats.hd_cell.sum()), frac_hd=float(stats.hd_cell.mean()),
        n_author_hd=int(stats.author_hd.sum()),
        agree_exc=float((exc.hd_cell == exc.author_hd).mean()) if len(exc) else np.nan,
        n_exc=int(stats.is_exc.sum()), n_fs=int(stats.is_fs.sum()),
        n_fs_hd=int((stats.is_fs & stats.hd_cell).sum()),
        median_mvl_hd=float(stats.mvl[stats.hd_cell].median()),
        decode_err_deg=med_err, n_decode_cells=len(sel),
        rot_coherence=R, rot_offset_deg=float(np.degrees(offset)) if np.isfinite(offset) else np.nan,
        rot_resid_deg=resid, n_cross_env=n_both,
    ))
    io.close()

units_df = pd.concat(unit_rows, ignore_index=True)
sess_df = pd.DataFrame(session_rows)
units_df.to_csv("all_units.csv", index=False)
sess_df.to_csv("session_summary.csv", index=False)

print(sess_df[["session", "n_units", "n_hd", "frac_hd", "agree_exc",
               "decode_err_deg", "rot_coherence"]].to_string(index=False))
print(f"\npooled: {len(units_df)} units, {int(units_df.hd_cell.sum())} HD cells "
      f"({100*units_df.hd_cell.mean():.0f}%) in {len(sess_df)} sessions")
print(f"median decoding error across sessions: "
      f"{sess_df.decode_err_deg.median():.1f} deg")
