"""
common.py — shared configuration and data loading for the analysis scripts.

Every script imports from here so that the data filter, category definitions,
random seeds and output paths are identical across the whole pipeline.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import btl

DATA_PATH = Path("arena_research_data.csv")
LANGUAGE = "en"            # None -> all languages
MIN_COMPARISONS = 50       # flag models with fewer decisive comparisons
SEED = 20251108            # master seed for every random operation
N_BOOT = 200               # session-cluster bootstrap replicates for ranks

CATEGORIES = {             # column in the cleaned CSV -> display name
    "math": "Math",
    "coding": "Coding",
    "creative_writing": "Creative writing",
}

OUT = Path("outputs")
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Validated colour-blind-safe categorical palette (fixed order; never cycled)
COLORS = {"Overall": "#2a78d6", "Math": "#eb6834", "Coding": "#1baf7a",
          "Creative writing": "#eda100", "Shrunk": "#6250d6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for c in CATEGORIES:
        df[c] = df[c].astype(bool)
    return df


def load_decisive(language=LANGUAGE) -> pd.DataFrame:
    """English (by default) decisive comparisons with outcome column y."""
    raw = load_raw()
    df = raw if language is None else raw[raw["language"] == language]
    return btl.prepare_decisive(df)
