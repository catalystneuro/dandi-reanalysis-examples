"""Load one session, summarize it, and plot raw spiking around visual stimuli."""

import matplotlib.pyplot as plt
import numpy as np

from common import AREA_LABELS, AREAS, load_session
from plotting import AREA_COLORS, FIG_DIR, style

SESSION = "715093703"


def summarize(units, stims, invalid):
    print(units)
    for name, ep in stims.items():
        print(f"{name}: {len(ep)} presentations, {ep.start[0]:.0f}-{ep.end[-1]:.0f} s")
    for p, inv in invalid.items():
        if len(inv):
            print(f"{p}: invalid periods {[(round(a), round(b)) for a, b in zip(inv.start, inv.end)]}")
    area = units.metadata["area"].values
    for a in AREAS:
        print(f"{a:6s} {np.sum(area == a):4d} units")


def plot_raw(units, stims):
    """Raster of every unit for 1.5 s of natural-scene presentations, sorted by area."""
    ns = stims["natural_scenes"]
    t0 = ns.start[200]
    t1 = t0 + 1.5
    area = units.metadata["area"].values
    fig, ax = plt.subplots(figsize=(9, 6.5))
    row = 0
    yticks = []
    for a in AREAS:
        ids = [k for k, ar in zip(units.keys(), area) if ar == a]
        if not ids:
            continue
        for k in ids:
            t = units[k].t
            t = t[(t >= t0) & (t < t1)] - t0
            ax.vlines(t, row, row + 0.9, color=AREA_COLORS[a], lw=0.6)
            row += 1
        yticks.append((row - len(ids) / 2, f"{AREA_LABELS[a]}\n({len(ids)})"))
        ax.axhline(row, color="0.85", lw=0.5)
    on = ns.start[(ns.start >= t0) & (ns.start < t1)] - t0
    frames = ns.metadata["frame"].values[(ns.start >= t0) & (ns.start < t1)]
    for x, fr in zip(on, frames):
        ax.axvline(x, color="0.3", lw=0.8, ls="--")
        ax.text(x + 0.005, -7, f"image {int(fr)}", fontsize=8, va="center")
    ax.set_yticks([y for y, _ in yticks], [lab for _, lab in yticks], fontsize=8)
    ax.set_ylim(row, -14)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("time from first image onset (s)")
    ax.set_title(f"Raw spikes from {row} units, session {SESSION}: natural images flashed every 250 ms")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_raw_raster.png", dpi=150)


if __name__ == "__main__":
    units, stims, invalid = load_session(SESSION)
    summarize(units, stims, invalid)
    plot_raw(units, stims)
