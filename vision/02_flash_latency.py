"""Step 1, timing: when does each area first respond to a full-field flash?

For each session and area, pool all quality-filtered units into one population
PSTH (1 ms bins, Gaussian smoothing sigma = 2 ms) around the 150 full-field
flashes (white and black, 250 ms each). The area's latency is the first time the
population rate reaches half of its peak above baseline within 0-150 ms. An area
is included in a session only if it has at least 10 units with valid data and
its peak exceeds the baseline mean by more than 5 baseline SDs.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from common import AREA_LABELS, AREAS, SESSIONS, load_session, psth
from plotting import AREA_COLORS, FIG_DIR, RESULTS_DIR, style

WINDOW = (-0.1, 0.35)
MIN_UNITS = 10


def analyze_session(sid):
    units, stims, invalid = load_session(sid)
    t, rate = psth(units, invalid, stims["flashes"].start, WINDOW, 0.001)
    area = units.metadata["area"].values
    rows, curves = [], {}
    for a in AREAS:
        m = (area == a) & ~np.isnan(rate).all(0)
        if m.sum() < MIN_UNITS:
            continue
        p = gaussian_filter1d(rate[:, m].mean(1), 2)
        mu, sd = p[t < 0].mean(), p[t < 0].std()
        w = (t > 0) & (t < 0.15)
        pk = p[w].max()
        if pk < mu + 5 * sd:
            continue
        lat = t[w][np.argmax(p[w] >= mu + 0.5 * (pk - mu))]
        rows.append({"session": sid, "area": a, "n_units": int(m.sum()), "latency_ms": lat * 1000})
        curves[a] = (p - mu) / (pk - mu)
    return pd.DataFrame(rows), t, curves


def main():
    dfs, curves = [], {a: [] for a in AREAS}
    for sid in tqdm(SESSIONS, desc="flash latency"):
        df, t, c = analyze_session(sid)
        dfs.append(df)
        for a, v in c.items():
            curves[a].append(v)
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(RESULTS_DIR / "flash_latency_per_session.csv", index=False)
    summary = (
        df.groupby("area")
        .agg(
            n_sessions=("session", "size"),
            total_units=("n_units", "sum"),
            median_latency_ms=("latency_ms", "median"),
            q25=("latency_ms", lambda x: np.percentile(x, 25)),
            q75=("latency_ms", lambda x: np.percentile(x, 75)),
        )
        .reindex(AREAS)
    )
    print(summary.round(1))
    summary.to_csv(RESULTS_DIR / "flash_latency_summary.csv")

    # Paired comparison: in sessions recording both, how often is LGN earlier than V1?
    wide = df.pivot(index="session", columns="area", values="latency_ms")
    pair = wide[["LGd", "VISp"]].dropna()
    print(f"LGN earlier than V1 in {np.sum(pair.LGd < pair.VISp)}/{len(pair)} sessions; "
          f"median difference {np.median(pair.VISp - pair.LGd):.1f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for a in AREAS:
        if not curves[a]:
            continue
        ax.plot(t * 1000, np.mean(curves[a], 0), color=AREA_COLORS[a], lw=2,
                label=f"{AREA_LABELS[a]} ({len(curves[a])} mice)")
    ax.axvspan(0, 250, color="0.94", zorder=-1, label="flash on")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_xlim(-30, 150)
    ax.set_ylim(-0.25, 1.05)
    ax.set_xlabel("time from flash onset (ms)")
    ax.set_ylabel("population rate (fraction of peak above baseline)")
    ax.set_title("Population response to a full-field flash")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)

    ax = axes[1]
    for sid, row in wide.iterrows():
        xs = [i for i, a in enumerate(AREAS) if a in row and not np.isnan(row[a])]
        ax.plot(xs, [row[AREAS[i]] for i in xs], color="0.8", lw=0.7, zorder=1)
    for i, a in enumerate(AREAS):
        v = df.loc[df["area"] == a, "latency_ms"].values
        if len(v) == 0:
            continue
        ax.scatter(np.full(len(v), i), v, s=22, color=AREA_COLORS[a], zorder=2, edgecolor="white", lw=0.5)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="k", lw=2, zorder=3)
        ax.text(i, 26, f"{np.median(v):.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(AREAS)), [AREA_LABELS[a].split(" ")[0] for a in AREAS])
    ax.set_ylim(23, 100)
    ax.set_ylabel("latency to half-peak (ms)")
    ax.set_title("Latency per area; dots = mice, bar and number = median", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_flash_latency.png", dpi=150)


if __name__ == "__main__":
    main()
