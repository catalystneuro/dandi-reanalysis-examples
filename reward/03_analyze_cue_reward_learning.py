"""Per-trial learning measures and the trial at which each mouse learns the cue.

For every CS+ presentation (cumulative trial index across sessions):
- anticipatory licking: lick rate in the 1.25 s cue-to-sucrose delay minus the rate
  in the 1.25 s before the cue (Hz)
- dopamine cue response: mean dF/F 0-0.5 s after cue onset minus mean dF/F in the
  1 s before the cue
- dopamine reward response: mean dF/F 0-0.5 s after sucrose delivery minus the same
  pre-cue baseline

Learning trial per mouse: the elbow of the cumulative curve of a measure, i.e. the
trial at which the cumulative sum falls furthest below the straight line joining its
start and end. A flat-then-rising cumulative curve has its elbow where the measure
switches on. This follows the cumulative-sum logic used in the source paper; the
exact procedure there may differ. As a check, a second criterion is also reported:
the first trial at which a 5-trial running mean reaches half of the mouse's
asymptote (mean of its last 20% of trials, at least 3 trials). This one cannot
resolve learning faster than ~2 trials.

Output: data/trials.csv, data/learning.csv, data/peth.npz (dopamine PETHs and lick
counts per trial, rows in the same order as trials.csv)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

HERE = Path(__file__).parent
DATA = HERE / "data"

DELAY = 1.25  # CS onset to sucrose (s), constant in all included sessions
PETH_WINDOW = (-2.0, 6.0)
BASELINE = (-1.0, 0.0)
CUE_WIN = (0.0, 0.5)
REWARD_WIN = (DELAY, DELAY + 0.5)


def window_mean(peth, win):
    """Mean of each column (trial) of a perievent TsdFrame inside a window."""
    return np.nanmean(peth.get(win[0], win[1]).values, axis=0)


def analyze_session(row):
    s = np.load(DATA / row["file"])
    # compute_perievent needs uniform sampling; one session has a single dropped
    # sample, so every trace is linearly resampled onto a uniform grid at its median rate.
    t_raw = s["dff_t"]
    t_uniform = np.arange(t_raw[0], t_raw[-1], np.median(np.diff(t_raw)))
    dff = nap.Tsd(t=t_uniform, d=np.interp(t_uniform, t_raw, s["dff"].astype(float)))
    # Events share the recording's time support; otherwise pynapple restricts licks to
    # [first lick, last lick] and silently drops cues outside that range.
    support = dff.time_support
    cs = nap.Ts(s["cs"], time_support=support)
    lick = nap.Ts(s["lick"], time_support=support)

    # Some sessions end within the PETH window after the last cue; keep trials whose
    # full window lies inside the recording.
    keep = (cs.t + PETH_WINDOW[0] > dff.t[0]) & (cs.t + PETH_WINDOW[1] < dff.t[-1])
    cs = nap.Ts(cs.t[keep], time_support=support)

    peth = nap.compute_perievent(dff, cs, window=PETH_WINDOW)  # TsdFrame: time x trial
    base = window_mean(peth, BASELINE)
    cue = window_mean(peth, CUE_WIN) - base
    rew = window_mean(peth, REWARD_WIN) - base

    lick_aligned = nap.compute_perievent(lick, cs, window=(-DELAY, DELAY))  # TsGroup per trial
    pre = np.array([np.sum(lick_aligned[i].t < 0) for i in lick_aligned], dtype=float) / DELAY
    ant = np.array([np.sum(lick_aligned[i].t >= 0) for i in lick_aligned], dtype=float) / DELAY
    if len(lick) == 0:
        # 60sD-F8 day02a ("part 1 before software crash") logged no licks at all while
        # the photometry is normal: treat licking as missing, not as zero.
        pre[:] = np.nan
        ant[:] = np.nan

    trials = pd.DataFrame(
        dict(
            subject=row["subject"],
            group=row["group"],
            mean_iti_s=row["mean_iti_s"],
            session=row["session"],
            session_idx=row["session_idx"],
            trial_in_session=np.flatnonzero(keep),
            trial=row["first_trial"] + np.flatnonzero(keep),
            cs_time=cs.t,
            lick_pre_hz=pre,
            lick_delay_hz=ant,
            anticipatory_lick_hz=ant - pre,
            da_cue=cue,
            da_reward=rew,
        )
    )
    # Resample PETHs onto a common 50 Hz grid so sessions can be stacked.
    grid = np.arange(PETH_WINDOW[0], PETH_WINDOW[1] + 1e-9, 0.02)
    peth_grid = np.stack([np.interp(grid, peth.t, peth.values[:, j]) - base[j] for j in range(peth.shape[1])])
    # Lick raster as counts in 100 ms bins, same window as the dopamine PETH.
    lick_peri = nap.compute_perievent(lick, cs, window=PETH_WINDOW)
    edges = np.arange(PETH_WINDOW[0], PETH_WINDOW[1] + 1e-9, 0.1)
    lick_hist = np.stack([np.histogram(lick_peri[i].t, edges)[0] for i in lick_peri])
    return trials, grid, peth_grid, lick_hist


def elbow(values):
    """Index where the cumulative sum lies furthest below the start-to-end chord."""
    c = np.concatenate([[0], np.cumsum(values)])
    chord = np.linspace(0, c[-1], len(c))
    return int(np.argmax(chord - c))


def half_asymptote(values, k=5):
    """First trial (0-based) where a k-trial running mean reaches half the asymptote."""
    asymptote = values[-max(int(len(values) * 0.2), 3):].mean()
    running = np.convolve(values, np.ones(k) / k, "valid")
    return int(np.argmax(running >= 0.5 * asymptote))


def main():
    sessions = pd.read_csv(DATA / "sessions.csv")
    all_trials, peths, lick_hists = [], [], []
    for _, row in tqdm(sessions.iterrows(), total=len(sessions), desc="Sessions"):
        trials, grid, peth, lick_hist = analyze_session(row)
        all_trials.append(trials)
        peths.append(peth)
        lick_hists.append(lick_hist)
    trials = pd.concat(all_trials, ignore_index=True)
    trials.to_csv(DATA / "trials.csv", index=False)
    np.savez_compressed(
        DATA / "peth.npz",
        t=grid,
        peth=np.concatenate(peths).astype(np.float32),
        lick_edges=np.arange(PETH_WINDOW[0], PETH_WINDOW[1] + 1e-9, 0.1),
        lick_counts=np.concatenate(lick_hists).astype(np.int16),
    )
    print(trials.groupby("group")[["anticipatory_lick_hz", "da_cue", "da_reward"]].describe().T)

    rows = []
    for subject, tr in trials.groupby("subject", sort=False):
        tr = tr.sort_values("trial")
        row = dict(subject=subject, group=tr["group"].iloc[0], mean_iti_s=tr["mean_iti_s"].iloc[0], n_trials=len(tr))
        for name, col in [("lick", "anticipatory_lick_hz"), ("da", "da_cue")]:
            valid = tr[col].notna().values
            v = tr[col].values[valid]
            trial_number = tr["trial"].values[valid] + 1  # 1-based cumulative trial number
            i = elbow(v)
            row[f"learned_trial_{name}"] = trial_number[i]  # first trial after the elbow
            # Means before vs after the elbow, to confirm the elbow marks an increase.
            row[f"{name}_before"] = v[:i].mean() if i else np.nan
            row[f"{name}_after"] = v[i:].mean()
            row[f"learned_trial_{name}_half"] = trial_number[half_asymptote(v)]
        row["da_cue_trial1"] = tr["da_cue"].values[0]
        rows.append(row)
    learning = pd.DataFrame(rows)
    learning["learned_time_lick_h"] = learning["learned_trial_lick"] * learning["mean_iti_s"] / 3600
    learning["learned_time_da_h"] = learning["learned_trial_da"] * learning["mean_iti_s"] / 3600
    learning.to_csv(DATA / "learning.csv", index=False)
    pd.set_option("display.width", 200)
    print(learning)
    pd.set_option("display.max_columns", 30)
    cols = ["learned_trial_lick", "learned_trial_da", "learned_time_lick_h", "learned_time_da_h",
            "learned_trial_lick_half", "learned_trial_da_half"]
    print(learning.groupby("group")[cols].agg(["median", "min", "max"]).T)


if __name__ == "__main__":
    main()
