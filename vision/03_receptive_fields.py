"""Step 2, space: each neuron watches only a small patch of the visual field.

Small Gabor patches (20 deg diameter) were flashed for 250 ms at 81 positions on a
9 x 9 grid spaced 10 deg apart (azimuth and elevation, -40 to +40 deg), about
45 times per position. For each unit we count spikes 30-280 ms after onset (to
allow for response latency) and average per position to get a receptive-field
(RF) map.

A unit has a significant RF if the variance of its 81 position means exceeds
the 99th percentile of 500 label shuffles. RF size is the number of grid squares
above half of the map's peak (after subtracting its minimum) times 100 deg^2;
RF centre is the rate-weighted centroid of those squares.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import AREA_LABELS, AREAS, SESSIONS, counts_matrix, load_session
from plotting import AREA_COLORS, FIG_DIR, RESULTS_DIR, style

WINDOW = (0.03, 0.28)
GRID = np.arange(-40, 41, 10)
N_SHUFFLE = 500


def rf_maps(units, stims, invalid):
    g = stims["gabors"]
    x = g.metadata["x_position"].values
    y = g.metadata["y_position"].values
    counts = counts_matrix(units, invalid, g.start, WINDOW)
    xi = np.searchsorted(GRID, x)
    yi = np.searchsorted(GRID, y)
    pos = yi * len(GRID) + xi
    maps = np.full((len(units), len(GRID), len(GRID)), np.nan)
    pvals = np.full(len(units), np.nan)
    split_r = np.full(len(units), np.nan)
    rng = np.random.default_rng(0)
    onehot = np.eye(len(GRID) ** 2)[pos]  # trials x positions
    for j in range(len(units)):
        ok = ~np.isnan(counts[:, j])
        if ok.sum() < 0.9 * len(ok):
            continue
        c = counts[ok, j]
        oh = onehot[ok]
        n = oh.sum(0)
        means = (c @ oh) / n
        maps[j] = means.reshape(len(GRID), len(GRID)) / (WINDOW[1] - WINDOW[0])
        stat = means.var()
        perm = np.stack([rng.permutation(c) for _ in range(N_SHUFFLE)])
        null = ((perm @ oh) / n).var(1)
        pvals[j] = (np.sum(null >= stat) + 1) / (N_SHUFFLE + 1)
        half = np.arange(len(c)) % 2 == 0
        m1 = (c[half] @ oh[half]) / oh[half].sum(0)
        m2 = (c[~half] @ oh[~half]) / oh[~half].sum(0)
        split_r[j] = np.corrcoef(m1, m2)[0, 1]
    return maps, pvals, split_r


def rf_shape(m):
    z = m - m.min()
    if z.max() <= 0:
        return np.nan, np.nan, np.nan
    above = z >= 0.5 * z.max()
    w = z * above
    yy, xx = np.meshgrid(GRID, GRID, indexing="ij")
    return above.sum() * 100.0, (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()


def analyze_session(sid):
    units, stims, invalid = load_session(sid)
    maps, p, split_r = rf_maps(units, stims, invalid)
    shape = np.array([rf_shape(m) if not np.isnan(m).all() else (np.nan,) * 3 for m in maps])
    df = pd.DataFrame(
        {
            "session": sid,
            "unit_id": list(units.keys()),
            "area": units.metadata["area"].values,
            "p_value": p,
            "has_rf": p < 0.01,
            "split_half_r": split_r,
            "rf_area_deg2": shape[:, 0],
            "rf_azimuth": shape[:, 1],
            "rf_elevation": shape[:, 2],
        }
    )
    return df, maps


def main():
    dfs, all_maps = [], []
    for sid in tqdm(SESSIONS, desc="receptive fields"):
        df, maps = analyze_session(sid)
        dfs.append(df)
        all_maps.append(maps)
    df = pd.concat(dfs, ignore_index=True)
    maps = np.concatenate(all_maps)
    df.to_csv(RESULTS_DIR / "receptive_fields_units.csv", index=False)
    np.save(RESULTS_DIR / "receptive_field_maps.npy", maps)
    tested = df[~df["p_value"].isna()]
    summary = (
        tested.groupby("area")
        .agg(
            n_units=("unit_id", "size"),
            frac_with_rf=("has_rf", "mean"),
            median_rf_area_deg2=("rf_area_deg2", lambda s: s[tested.loc[s.index, "has_rf"]].median()),
        )
        .reindex(AREAS)
    )
    print(summary.round(2))
    summary.to_csv(RESULTS_DIR / "receptive_fields_summary.csv")
    plot(df, maps)


def plot(df, maps):
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(3, 7, height_ratios=[1, 1, 1.5], hspace=0.6, wspace=0.5)
    ext = [-45, 45, -45, 45]
    # Row 1-2: the strongest RF in each area (two examples per area)
    for i, a in enumerate(AREAS):
        sel = df[(df["area"] == a) & df["has_rf"]].copy()
        if sel.empty:
            continue
        # Most reliable maps (odd vs even trials), centred away from the screen
        # edge so the whole RF is visible
        sel = sel[(sel.rf_azimuth.abs() <= 30) & (sel.rf_elevation.abs() <= 30)]
        best = sel.sort_values("split_half_r", ascending=False).index[:2]
        for r, k in enumerate(best):
            ax = fig.add_subplot(gs[r, i])
            ax.imshow(maps[k], origin="lower", extent=ext, cmap="magma")
            ax.set_xticks([-40, 0, 40])
            ax.set_yticks([-40, 0, 40])
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(AREA_LABELS[a].split(" ")[0], color=AREA_COLORS[a], fontsize=11, fontweight="bold")
            if i == 0:
                ax.set_ylabel("elevation (deg)", fontsize=8)
            if r == 1:
                ax.set_xlabel("azimuth (deg)", fontsize=8)
    fig.text(0.5, 0.94, "Example receptive fields: firing rate for a Gabor patch at each screen position",
             ha="center", fontsize=12)

    sig = df[df["has_rf"]]
    tested = df[~df["p_value"].isna()]
    ax = fig.add_subplot(gs[2, 0:2])
    frac = [tested.loc[tested.area == a, "has_rf"].mean() for a in AREAS]
    ax.bar(range(len(AREAS)), frac, color=[AREA_COLORS[a] for a in AREAS])
    ax.set_xticks(range(len(AREAS)), [AREA_LABELS[a].split(" ")[0] for a in AREAS], fontsize=8)
    ax.set_ylabel("fraction of units")
    ax.set_title("Units with a significant RF", fontsize=10)
    style(ax)

    ax = fig.add_subplot(gs[2, 2:4])
    data = [sig.loc[sig.area == a, "rf_area_deg2"].values for a in AREAS]
    bp = ax.boxplot(data, positions=range(len(AREAS)), widths=0.6, showfliers=False, patch_artist=True)
    for patch, a in zip(bp["boxes"], AREAS):
        patch.set_facecolor(AREA_COLORS[a])
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_xticks(range(len(AREAS)), [AREA_LABELS[a].split(" ")[0] for a in AREAS], fontsize=8)
    ax.set_ylabel("RF area (deg$^2$)")
    ax.set_title("RF size (units with a significant RF)", fontsize=10)
    style(ax)

    ax = fig.add_subplot(gs[2, 4:7])
    v1 = sig[sig.area == "VISp"]
    sessions = v1.session.unique()
    cmap = plt.get_cmap("tab20")
    for i, s in enumerate(sessions):
        d = v1[v1.session == s]
        ax.scatter(d.rf_azimuth, d.rf_elevation, s=10, color=cmap(i % 20), alpha=0.7, lw=0)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(f"V1 RF centres, {len(v1)} units\n(colour = mouse; each probe samples one patch)", fontsize=10)
    style(ax)
    fig.savefig(FIG_DIR / "03_receptive_fields.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
