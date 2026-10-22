"""Shared plotting style."""

from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Thalamus in a dark tone, V1 distinct, higher areas along a sequential ramp
AREA_COLORS = {
    "LGd": "#3b3b3b",
    "VISp": "#1f6fb4",
    "VISl": "#2a9d8f",
    "VISrl": "#8ab17d",
    "VISal": "#e9c46a",
    "VISpm": "#f4a261",
    "VISam": "#e76f51",
}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
