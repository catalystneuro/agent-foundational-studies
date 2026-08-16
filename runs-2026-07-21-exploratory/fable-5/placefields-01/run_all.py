"""Run the place-cell pipeline on the primary session and on all 8 sessions."""

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import place_cell_lib as pcl
import figures as F

PRIMARY = "sub-Achilles/sub-Achilles_ses-Achilles-10252013_behavior+ecephys.nwb"
COMMON_BINS = 40


def analyse(S, n_shuffles=pcl.N_SHUFFLES, progress=None):
    stats, per_dir = pcl.classify_place_cells(
        S["units"], S["position"], S["run_ep"], S["lap_dir"], S["speed"], S["dt"],
        n_bins=S["n_bins"], track_cm=S["track_cm"],
        circular=S["maze_type"] == "circular",
        n_shuffles=n_shuffles, progress=progress)
    fields = pcl.field_table(S["units"], stats, per_dir)
    return stats, per_dir, fields


def decode_session(S, stats, per_dir, rng):
    pc_ids = stats.index[stats["is_place_cell"]].values
    ens = S["units"][list(pc_ids)]
    errs, trues, decs = [], [], []
    for name in per_dir:
        _, dcd, tru = pcl.crossvalidated_decoding(
            ens, S["position"], per_dir[name]["laps"], S["speed"], S["dt"],
            n_bins=S["n_bins"], track_cm=S["track_cm"],
            circular=S["maze_type"] == "circular")
        errs.append(np.abs(dcd - tru))
        trues.append(tru)
        decs.append(dcd)
    err = np.concatenate(errs)
    tru, dcd = np.concatenate(trues), np.concatenate(decs)
    chance = np.concatenate([np.abs(rng.permutation(dcd) - tru) for _ in range(20)])
    return float(np.median(err)), float(np.median(chance))


def resample_map(rates, n_out=COMMON_BINS):
    """Interpolate rate maps onto a common fraction-of-track axis."""
    src = np.linspace(0, 1, rates.shape[1])
    dst = np.linspace(0, 1, n_out)
    return np.stack([np.interp(dst, src, r) for r in rates])


def main():
    rng = np.random.default_rng(0)
    urls = pcl.list_session_urls()

    # ---- primary session, in detail --------------------------------------
    print(f"=== primary session: {PRIMARY.split('/')[-1]}")
    S = pcl.load_session(urls[PRIMARY])
    stats, per_dir, fields = analyse(S, progress=tqdm)
    print(f"  {len(S['units'])} units, {stats.is_place_cell.sum()} place cells, "
          f"{len(fields)} fields")
    ch, theta_ratio = pcl.pick_lfp_channel(S["nwbfile"], float(S["run_ep"].start[0]))

    F.fig_session_overview(S, stats, per_dir, ch, "fig01_session_overview.png")
    F.fig_example_cells(S, stats, per_dir, fields, "fig02_example_place_cells.png")
    F.fig_population_maps(S, stats, per_dir, "fig03_population_maps.png")
    F.fig_spatial_information(S, stats, per_dir, "fig04_spatial_information.png")
    F.fig_field_properties(S, stats, per_dir, fields, "fig05_field_properties.png")
    dec = F.decoding_results(S, stats, per_dir)
    F.fig_decoding(S, stats, per_dir, dec, "fig06_decoding.png")
    print(f"  decoding: median {np.median(dec['error']):.1f} cm "
          f"(chance {np.median(dec['chance']):.0f} cm)")

    ref = max(per_dir, key=lambda d: len(per_dir[d]["laps"]))
    glm = pcl.glm_position_vs_speed(
        S["units"], S["position"], S["speed"], per_dir[ref]["laps"], S["dt"],
        n_bins=S["n_bins"], track_cm=S["track_cm"])
    F.fig_glm(S, stats, per_dir, fields, glm, ref, "fig07_glm_position_vs_speed.png")
    pcm = stats.is_place_cell.values
    print(f"  GLM ({ref}): position {np.nanmedian(glm['gains']['position'][pcm]):.3f} "
          f"vs speed {np.nanmedian(glm['gains']['speed'][pcm]):.3f} bits/spike")
    stats.to_csv("place_cell_stats_primary.csv")
    fields.to_csv("place_fields_primary.csv", index=False)

    # ---- all sessions ------------------------------------------------------
    rows, all_fields, all_maps, glm_rows = [], [], [], []
    for path, url in tqdm(list(urls.items()), desc="sessions"):
        t0 = time.time()
        Si = S if path == PRIMARY else pcl.load_session(url)
        st, pd_, fl = (stats, per_dir, fields) if path == PRIMARY else analyse(Si)
        med_err, chance_err = decode_session(Si, st, pd_, rng)

        refi = max(pd_, key=lambda d: len(pd_[d]["laps"]))
        gi = pcl.glm_position_vs_speed(
            Si["units"], Si["position"], Si["speed"], pd_[refi]["laps"], Si["dt"],
            n_bins=Si["n_bins"], track_cm=Si["track_cm"])
        m = st["is_place_cell"].values & np.isfinite(gi["gains"]["position"])
        glm_rows.append(pd.DataFrame({
            "session": Si["session_id"], "maze_type": Si["maze_type"],
            "position": gi["gains"]["position"][m],
            "speed": gi["gains"]["speed"][m],
            "position+speed": gi["gains"]["position+speed"][m]}))

        n_exc = int((st["cell_type"] == "excitatory").sum())
        rows.append(dict(
            session=Si["session_id"], subject=Si["subject_id"],
            maze_type=Si["maze_type"], track_cm=Si["track_cm"],
            n_units=len(Si["units"]), n_excitatory=n_exc,
            n_laps=len(Si["run_ep"]),
            run_time_s=float((Si["run_ep"].end - Si["run_ep"].start).sum()),
            n_place_cells=int(st["is_place_cell"].sum()),
            place_cell_fraction=float(st["is_place_cell"].sum()) / n_exc,
            n_fields=len(fl),
            median_si=float(fl["si"].median()),
            median_width_cm=float(fl["width"].median()),
            median_peak_rate=float(fl["peak_rate"].median()),
            median_participation=float(fl["participation"].median()),
            decode_error_cm=med_err, chance_error_cm=chance_err,
            seconds=time.time() - t0))

        fl = fl.assign(session=Si["session_id"], track_cm=Si["track_cm"],
                       maze_type=Si["maze_type"])
        all_fields.append(fl)
        for name, d in pd_.items():
            sel = fl.loc[fl["direction"] == name, "k"].astype(int).values
            if sel.size:
                all_maps.append(resample_map(d["rates"][sel]))
        if path != PRIMARY:
            Si["io"].close()

    summary = pd.DataFrame(rows)
    pooled_fields = pd.concat(all_fields, ignore_index=True)
    pooled_maps = np.concatenate(all_maps)
    pooled_glm = pd.concat(glm_rows, ignore_index=True)
    summary.to_csv("session_summary.csv", index=False)
    pooled_fields.to_csv("place_fields_all_sessions.csv", index=False)

    F.fig_multisession(summary, pooled_fields, pooled_maps, pooled_glm,
                       "fig08_across_sessions.png")

    print("\n=== session summary ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary[["session", "maze_type", "track_cm", "n_excitatory",
                       "n_place_cells", "place_cell_fraction", "n_fields",
                       "median_si", "median_width_cm", "decode_error_cm",
                       "chance_error_cm"]].to_string(index=False))
    print(f"\npooled: {len(pooled_fields)} fields, "
          f"{pooled_fields['unit'].groupby(pooled_fields['session']).nunique().sum()} "
          f"place cells across {len(summary)} sessions")
    print(f"position > speed for "
          f"{(pooled_glm['position'] > pooled_glm['speed']).mean():.1%} of place cells")


if __name__ == "__main__":
    main()
