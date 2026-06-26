"""Shared paper-ready matplotlib style for thesis figures.

Import this from any notebook in ``data_analysis/`` to get a consistent,
publication-quality look and a one-call figure saver that writes vector PDF
(for ``\\includegraphics`` in LaTeX) plus a PNG preview.

    from plot_style import apply_paper_style, save_fig, figsize, PALETTE
    apply_paper_style()                       # set once per notebook
    fig, ax = plt.subplots(figsize=figsize(width="text", ratio=0.6))
    ax.bar(...)
    save_fig(fig, "bsd10k_class_counts")      # -> figures/bsd10k_class_counts.{pdf,png}

LaTeX side, once:
    \\includegraphics[width=\\linewidth]{figures/bsd10k_class_counts.pdf}

Set ``apply_paper_style(usetex=True)`` to render text with your LaTeX install
so figure fonts match the document exactly (needs a working latex + dvipng).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# --- LaTeX layout widths (inches) -----------------------------------------
# Measure yours with \the\textwidth / \the\columnwidth in the document and
# divide by 72.27. These are sensible defaults.
TEXT_WIDTH = 6.3     # full \textwidth of a single-column article
COLUMN_WIDTH = 3.39  # \columnwidth of a two-column (IEEE-style) layout

# Colourblind-friendly qualitative palette (Wong, 2011).
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]

# Figures are written here (next to this module) regardless of notebook cwd.
FIG_DIR = Path(__file__).resolve().parent / "figures"


def figsize(width: str | float = "text", ratio: float = 0.62) -> tuple[float, float]:
    """Figure size in inches. ``width`` = 'text', 'column', or a number of inches.
    ``ratio`` is height/width (default ~golden)."""
    w = {"text": TEXT_WIDTH, "column": COLUMN_WIDTH}.get(width, width)
    return (float(w), float(w) * ratio)


def apply_paper_style(usetex: bool = False, font_size: int = 9) -> None:
    """Apply the paper style globally. Call once near the top of a notebook."""
    mpl.rcParams.update({
        # sizing / output
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": True,
        # fonts
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # embed real fonts, not Type 3 (most publishers reject Type 3)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # spines / grid
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        # lines / legend
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=PALETTE),
    })
    if usetex:
        mpl.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })


def save_fig(fig, name: str, formats=("pdf", "png"), fig_dir: Path | None = None) -> Path:
    """Save ``fig`` as ``name.<fmt>`` for each format into the figures dir.
    Returns the directory written to."""
    d = Path(fig_dir) if fig_dir is not None else FIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(d / f"{name}.{fmt}")
    return d
