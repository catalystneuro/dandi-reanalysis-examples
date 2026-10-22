"""Step 3, features: cortex responds to edges at particular orientations.

Full-field drifting gratings were shown for 2 s at 8 directions x 5 temporal
frequencies (about 15 repeats each), interleaved with blank screens. For each
unit we take the mean firing rate during each 2 s presentation, find its
preferred temporal frequency, and compute at that frequency:

* global orientation selectivity index, OSI = |sum r(theta) exp(2i theta)| / sum r(theta)
  (0 = same response to all orientations, 1 = responds to a single orientation)
* direction selectivity index, DSI = |sum r(theta) exp(i theta)| / sum r(theta)

where r(theta) is the mean firing rate (as in the Allen Brain Observatory
"global OSI"; subtracting the blank rate and clipping at 0 inflates OSI for weak,
noisy units). Units are included if their responses differ across the 40 grating conditions (one-way
ANOVA, p < 0.01) and their preferred condition drives at least 1 Hz above blank.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from tqdm import tqdm

from common import AREA_LABELS, AREAS, SESSIONS, counts_matrix, load_session
from plotting import AREA_COLORS, FIG_DIR, RESULTS_DIR, style

WINDOW = (0.0, 2.0)


def tuning(units, stims, invalid):
    dg = stims["drifting_gratings"]
    ori = dg.metadata["orientation"].values
    tf = dg.metadata["temporal_frequency"].values
    rate = counts_matrix(units, invalid, dg.start, WINDOW) / (WINDOW[1] - WINDOW[0])
    blank = np.isnan(ori)
    dirs = np.unique(ori[~blank])
    tfs = np.unique(tf[~blank])
    curves = np.full((len(units), len(tfs), len(dirs)), np.nan)
    for a, f in enumerate(tfs):
        for b, d in enumerate(dirs):
            curves[:, a, b] = np.nanmean(rate[(tf == f) & (ori == d)], 0)
    blank_rate = np.nanmean(rate[blank], 0)
    groups = [(tf == f) & (ori == d) for f in tfs for d in dirs]
    pvals = np.array([
        f_oneway(*[rate[g, j][~np.isnan(rate[g, j])] for g in groups]).pvalue
        if np.isnan(rate[:, j]).mean() < 0.1 else np.nan
        for j in range(len(units))
    ])
    return curves, blank_rate, pvals, dirs, tfs


def selectivity(curves, blank_rate, dirs):
    pref_tf = np.nanargmax(np.nanmax(curves, 2), 1)
    r = curves[np.arange(len(curves)), pref_tf]
    th = np.deg2rad(dirs)
    s = r.sum(1)
    osi = np.abs((r * np.exp(2j * th)).sum(1)) / s
    dsi = np.abs((r * np.exp(1j * th)).sum(1)) / s
    peak = r.max(1) - blank_rate
    return osi, dsi, peak, pref_tf, r


def analyze_session(sid):
    units, stims, invalid = load_session(sid)
    curves, blank_rate, p, dirs, tfs = tuning(units, stims, invalid)
    osi, dsi, peak, pref_tf, r = selectivity(curves, blank_rate, dirs)
    df = pd.DataFrame(
        {
            "session": sid,
            "unit_id": list(units.keys()),
            "area": units.metadata["area"].values,
            "anova_p": p,
            "responsive": (p < 0.01) & (peak >= 1.0),
            "osi": osi,
            "dsi": dsi,
            "peak_above_blank_hz": peak,
            "pref_tf_hz": tfs[pref_tf],
            "pref_direction": dirs[np.nanargmax(r, 1)],
        }
    )
    return df, r, dirs


def main():
    dfs, curves = [], []
    for sid in tqdm(SESSIONS, desc="orientation tuning"):
        df, r, dirs = analyze_session(sid)
        dfs.append(df)
        curves.append(r)
    df = pd.concat(dfs, ignore_index=True)
    curves = np.concatenate(curves)
    df.to_csv(RESULTS_DIR / "orientation_tuning_units.csv", index=False)
    resp = df[df.responsive]
    summary = (
        df.groupby("area")
        .agg(n_units=("unit_id", "size"), frac_responsive=("responsive", "mean"))
        .join(resp.groupby("area").agg(median_osi=("osi", "median"), median_dsi=("dsi", "median"),
                                       frac_osi_above_0_3=("osi", lambda x: np.mean(x > 0.3))))
        .reindex(AREAS)
    )
    print(summary.round(3))
    summary.to_csv(RESULTS_DIR / "orientation_tuning_summary.csv")
    plot(df, curves, dirs)


def plot(df, curves, dirs):
    resp = df[df.responsive]
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.35, wspace=0.35)
    # Examples: median-OSI LGN unit vs the most selective V1 units
    examples = []
    lgn = resp[resp.area == "LGd"].sort_values("osi")
    examples.append(("LGd", lgn.index[len(lgn) // 2], "typical LGN unit"))
    v1 = resp[resp.area == "VISp"].sort_values("peak_above_blank_hz", ascending=False)
    v1 = v1[v1.osi > 0.4]
    for k in v1.index[:3]:
        examples.append(("VISp", k, "selective V1 unit"))
    th = np.deg2rad(np.append(dirs, dirs[0]))
    for i, (a, k, lab) in enumerate(examples):
        ax = fig.add_subplot(gs[0, i], projection="polar")
        r = curves[k]
        ax.plot(th, np.append(r, r[0]), color=AREA_COLORS[a], lw=2)
        ax.fill(th, np.append(r, r[0]), color=AREA_COLORS[a], alpha=0.2)
        ax.set_xticks(np.deg2rad(dirs))
        ax.set_xticklabels([f"{int(d)}°" for d in dirs], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"{lab}\nOSI {df.osi[k]:.2f}; Hz per grating direction", fontsize=9, pad=14)

    ax = fig.add_subplot(gs[1, 0:2])
    bins = np.linspace(0, 1, 26)
    for a in ["LGd", "VISp"]:
        v = resp.loc[resp.area == a, "osi"]
        ax.hist(v, bins, density=True, histtype="stepfilled", alpha=0.35, color=AREA_COLORS[a])
        ax.hist(v, bins, density=True, histtype="step", lw=2, color=AREA_COLORS[a],
                label=f"{AREA_LABELS[a]}, n={len(v)}, median {v.median():.2f}")
    ax.set_xlabel("orientation selectivity index (OSI)")
    ax.set_ylabel("density")
    ax.set_title("Orientation selectivity emerges between LGN and V1", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = fig.add_subplot(gs[1, 2:4])
    data = [resp.loc[resp.area == a, "osi"].values for a in AREAS]
    bp = ax.boxplot(data, positions=range(len(AREAS)), widths=0.6, showfliers=False, patch_artist=True)
    for patch, a in zip(bp["boxes"], AREAS):
        patch.set_facecolor(AREA_COLORS[a])
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_xticks(range(len(AREAS)), [f"{AREA_LABELS[a].split(' ')[0]}\nn={len(d)}" for a, d in zip(AREAS, data)],
                  fontsize=8)
    ax.set_ylabel("OSI")
    ax.set_title("OSI of grating-responsive units by area", fontsize=10)
    style(ax)
    fig.savefig(FIG_DIR / "04_orientation_tuning.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
