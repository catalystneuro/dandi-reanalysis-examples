"""Population decoding: can position and head direction be read out from the recorded MEC cells?

Bayesian decoding (pynapple.decode_bayes) with interleaved cross-validation: tuning curves are
fit on even minutes and the animal's position and heading are decoded on odd minutes, and vice
versa. Chance is estimated by decoding spike trains that were circularly shifted in time
(same tuning curves, broken relation to behaviour). Saves results/decoding.csv and an example trace.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

from spatial_lib import list_sessions, load_session

MIN_UNITS = 5
POS_BIN_CM = 10.0
HD_BINS = 30  # 12 deg bins
BIN = 0.25  # s
WIN = 4  # sliding window of 4 bins = 1 s
N_CHANCE = 10


def folds(position, moving):
    t0 = position.time_support.start[0]
    edges = np.arange(t0, position.time_support.end[-1] + 60, 60)
    minutes = nap.IntervalSet(edges[:-1], edges[1:])
    # keep running periods of at least 1 s so each decoding window has data
    even = minutes[::2].intersect(moving).drop_short_intervals(WIN * BIN)
    odd = minutes[1::2].intersect(moving).drop_short_intervals(WIN * BIN)
    return [(even, odd), (odd, even)]


def decode(units, feature, train, test, bins, rng_, circular):
    tc = nap.compute_tuning_curves(units, feature, bins=bins, range=rng_, epochs=train, fs=50.0)
    dec, _ = nap.decode_bayes(tc, units, epochs=test, bin_size=BIN, sliding_window_size=WIN, uniform_prior=False)
    # behaviour at each decoded bin centre (decoder output uses a 1 s sliding window)
    f = feature.restrict(test)
    if circular:  # average heading as unit vectors to avoid wrap-around artefacts
        uv = nap.TsdFrame(t=f.t, d=np.c_[np.cos(f.values), np.sin(f.values)], time_support=test)
        uv = uv.bin_average(BIN, test)
        true = nap.Tsd(t=uv.t, d=np.mod(np.arctan2(uv.values[:, 1], uv.values[:, 0]), 2 * np.pi))
    else:
        true = f.bin_average(BIN, test)
    tr = dec.value_from(true)
    fin = lambda a: np.isfinite(np.asarray(a, float)).reshape(len(a), -1).all(1)
    keep = fin(dec.values) & fin(tr.values)  # drop bins without decoded or tracked values
    t, d, trv = dec.t[keep], dec.values[keep], tr.values[keep]
    if circular:
        err = np.degrees(np.abs(np.angle(np.exp(1j * (d - trv)))))
    else:
        err = np.linalg.norm(d - trv, axis=1)
    return err, t, d, trv


def shift_group(units, start, end, rng):
    T = end - start
    return nap.TsGroup({u: nap.Ts(np.sort(np.mod(units[u].t - start + s, T)) + start)
                        for u, s in zip(units.index, rng.uniform(20, T - 20, len(units)))})


def run(args):
    path, url = args
    rng = np.random.default_rng(0)
    units, position, head_dir, moving, meta = load_session(url)
    if len(units) < MIN_UNITS:
        return None
    start, end = position.time_support.start[0], position.time_support.end[-1]
    out = dict(path=path, subject=meta["subject_id"], session=meta["session_id"], n_units=len(units))
    out["box_cm"] = meta["box_side"]
    nb = int(meta["box_side"] / POS_BIN_CM)
    specs = [("pos", position, [nb, nb], meta["arena"], False)]
    if head_dir is not None:
        specs.append(("hd", head_dir, HD_BINS, [(0, 2 * np.pi)], True))
    traces = {}
    for name, feat, bins, rng_, circ in specs:
        errs, chance = [], []
        for train, test in folds(position, moving):
            e, t, d, tr = decode(units, feat, train, test, bins, rng_, circ)
            errs.append(e)
            traces.setdefault(name, []).append((t, d, tr))
            for _ in range(N_CHANCE):
                ce, *_ = decode(shift_group(units, start, end, rng), feat, train, test, bins, rng_, circ)
                chance.append(np.median(ce))
        out[f"{name}_err"] = np.median(np.concatenate(errs))
        out[f"{name}_chance"] = np.median(chance)
        out[f"{name}_chance_p05"] = np.percentile(chance, 5)
    return out, traces


if __name__ == "__main__":
    sessions = list_sessions()
    with Pool(os.cpu_count()) as pool:
        res = [r for r in tqdm(pool.imap(run, sessions), total=len(sessions)) if r is not None]
    df = pd.DataFrame([r[0] for r in res])
    df.to_csv("results/decoding.csv", index=False)
    print(df.describe())
    # save the traces of the session with the most units that has head direction
    best = df[df.hd_err.notna()].sort_values("n_units", ascending=False).index[0]
    tr = res[best][1]
    np.savez(f"results/decoding_example.npz", path=df.loc[best, "path"],
             **{f"{k}_{f}_{i}": v for k, folds_ in tr.items() for f, fold in enumerate(folds_)
                for i, v in zip(("t", "dec", "true"), fold)})
    print("example:", df.loc[best].to_dict())
