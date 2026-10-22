"""Compute spatial info, gridness and head-direction tuning for every unit in DANDI:000582,
with time-shift shuffles for significance thresholds. Saves results/units.csv and results/maps.npz."""
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_lib import (autocorr2d, gridness, hd_curves, list_sessions, load_session, mean_vector, rate_maps,
                         shuffled_groups, spatial_info)


def analyze(args):
    path, url = args
    rng = np.random.default_rng(zlib.crc32(path.encode()))  # stable per-session seed
    units, position, head_dir, moving, meta = load_session(url)
    maps, occ = rate_maps(units, position, moving, meta["arena"])
    si = spatial_info(maps, occ)
    acs = [autocorr2d(m) for m in maps]
    gs = np.array([gridness(a) for a in acs])
    if head_dir is not None:
        curves, centers = hd_curves(units, head_dir, moving)
        mvl, pref = mean_vector(curves, centers)
    else:
        curves = np.full((len(units), 60), np.nan)
        mvl = pref = np.full(len(units), np.nan)

    start, end = position.time_support.start[0], position.time_support.end[-1]
    rows, shuf = [], []
    for k, u in enumerate(units.index):
        sh = shuffled_groups(units[u], start, end, rng)
        smaps, socc = rate_maps(sh, position, moving, meta["arena"])
        s_si = spatial_info(smaps, socc)
        s_g = np.array([gridness(autocorr2d(m)) for m in smaps])
        if head_dir is not None:
            sc, scen = hd_curves(sh, head_dir, moving)
            s_mvl = mean_vector(sc, scen)[0]
        else:
            s_mvl = np.full(len(sh), np.nan)
        shuf.append(np.stack([s_si, s_g, s_mvl]))
        rows.append(dict(
            path=path, subject=meta["subject_id"], session=meta["session_id"], box_cm=meta["box_side"],
            unit=units.metadata["unit_name"].iloc[k], layer=units.metadata["histology"].iloc[k],
            n_spikes=len(units[u]), rate=units[u].restrict(moving).rate,
            spatial_info=si[k], gridness=gs[k], mvl=mvl[k], pref_dir=pref[k], has_hd=head_dir is not None,
        ))
    return rows, maps, acs, curves, shuf


if __name__ == "__main__":
    sessions = list_sessions()
    print(f"{len(sessions)} sessions")
    rows, maps, acs, curves, shuf = [], [], [], [], []
    with Pool(os.cpu_count()) as pool:
        for r, m, a, c, s in tqdm(pool.imap(analyze, sessions), total=len(sessions)):
            rows += r
            maps += list(m)
            acs += list(a)
            curves += list(c)
            shuf += s
    df = pd.DataFrame(rows)
    shuf = np.array(shuf)  # (unit, metric[si, grid, mvl], shuffle)
    df.to_csv("results/units.csv", index=False)
    # maps differ in size between box sizes, so store them as object arrays
    obj = lambda x: np.array(x + [None], dtype=object)[:-1]
    np.savez_compressed("results/maps.npz", maps=obj(maps), acs=obj(acs), curves=np.array(curves), shuffles=shuf)
    print(df.groupby("layer").size())
    print(df.describe())
