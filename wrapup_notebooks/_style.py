"""Shared look-and-feel for the wrap-up notebooks.

This module holds the *aesthetics* and nothing else: the palette, the rcParams, and
figure saving. It deliberately contains **no plotting functions** -- every panel is
drawn explicitly in the notebook cell that owns it, so it stays editable.

One line in a notebook's setup cell wires everything up::

    import _style as st
    P = st.init(notebook="01_behaviour", theme=THEME, scale=FONT_SCALE)

Change a colour or a font size here, re-run that one cell, and every figure in every
wrap-up notebook restyles.

Colour discipline
-----------------
Keys are **meanings**, not colours (``P["correct"]``, not ``P["green"]``), so a figure
never hard-codes a hex value. The base palette is Okabe-Ito: colourblind-safe and still
legible once a projector flattens it. The rule that keeps a figure honest is
**one colour family per figure** -- if a figure codes outcome, then phase and axis in
that same figure are greyscale. Anything not encoding a variable is grey or black.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["init", "palette", "save_fig", "day_colors", "day_cmap", "swatch",
           "repo_root", "figdir", "THEMES", "SEQ_CMAP"]

THEMES = ("light", "dark")
SEQ_CMAP = "viridis"          # the one ordinal/day ramp, same in both themes

# How much of the ramp to use. On dark slides viridis's near-black end disappears into
# the background, so the dark theme starts further up. Measured, not guessed: #440154
# against #111111 is a contrast ratio of 1.2.
_RAMP_SPAN = {"light": (0.05, 0.95), "dark": (0.28, 0.97)}


# --------------------------------------------------------------------------- palettes
# Okabe-Ito on white; hand-lightened counterparts on dark (programmatic lightening
# goes muddy, so the dark values are chosen, not computed).
_PALETTES = {
    "light": {
        # --- outcome -------------------------------------------------------
        "correct":      "#009E73",   # bluish green
        "incorrect":    "#D55E00",   # vermillion
        # --- decoder axis / remapping block --------------------------------
        "intuitive":    "#0072B2",   # blue
        "new_axis":     "#CC79A7",   # reddish purple
        "washout":      "#56B4E9",   # sky blue -- reads as "part way back to intuitive"
        # --- training phase (aliases of the axis family, on purpose) -------
        "stable":       "#0072B2",
        "remap":        "#CC79A7",
        # --- task condition ------------------------------------------------
        "baseline":     "#E69F00",   # orange -- task-free
        "task":         "#0072B2",
        # --- signals that are not a category -------------------------------
        "cursor":       "#1A1A1A",   # the cursor trace itself: near-black
        "reward":       "#0072B2",   # reward markers
        "reward_tone":  "#BBBBBB",   # the 13.1-octave clamp: present but receding
        "single":       "#C8C8C8",   # individual trials behind a mean
        "null":         "#8A8A8A",   # shuffle / chance / extrapolated trend
        "accent":       "#F0E442",   # yellow, for a single call-out only
        # --- furniture -----------------------------------------------------
        "fg":           "#1A1A1A",
        "bg":           "#FFFFFF",
        "muted":        "#6E6E6E",
        "grid":         "#D9D9D9",
    },
    "dark": {
        "correct":      "#34C796",
        "incorrect":    "#FF8A4C",
        "intuitive":    "#56B4E9",
        "new_axis":     "#E29BC4",
        "washout":      "#8FD3F4",
        "stable":       "#56B4E9",
        "remap":        "#E29BC4",
        "baseline":     "#F0B553",
        "task":         "#56B4E9",
        "cursor":       "#EDEDED",
        "reward":       "#8FD3F4",
        "reward_tone":  "#6A6A6A",
        "single":       "#4E4E4E",
        "null":         "#9A9A9A",
        "accent":       "#F5EE7A",
        "fg":           "#E8E8E8",
        "bg":           "#111111",
        "muted":        "#A0A0A0",
        "grid":         "#3A3A3A",
    },
}

# Roles grouped for the swatch, and as documentation of the one-family-per-figure rule.
_FAMILIES = {
    "outcome":    ["correct", "incorrect"],
    "axis/block": ["intuitive", "new_axis", "washout"],
    "phase":      ["stable", "remap"],
    "condition":  ["baseline", "task"],
    "signals":    ["cursor", "reward", "reward_tone", "single", "null", "accent"],
    "furniture":  ["fg", "muted", "grid", "bg"],
}

_STATE = {"theme": "light", "scale": 1.0, "figdir": None}


# ------------------------------------------------------------------------ paths
def repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file) until the repo root is found."""
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "utilities").is_dir() and (p / "builders").is_dir():
            return p
    return here.parent.parent


def figdir() -> Path:
    """Where the active notebook's figures are being written."""
    if _STATE["figdir"] is None:
        raise RuntimeError("call st.init(notebook=...) first")
    return _STATE["figdir"]


# ------------------------------------------------------------------------ palette
def palette(theme: str | None = None) -> dict:
    """A copy of the palette for ``theme`` (default: the active one)."""
    t = theme or _STATE["theme"]
    if t not in _PALETTES:
        raise ValueError(f"theme must be one of {THEMES}, got {t!r}")
    return dict(_PALETTES[t])


def day_colors(n: int, *, cmap: str = SEQ_CMAP, theme: str | None = None,
               lo: float | None = None, hi: float | None = None):
    """``n`` colours along the ordinal ramp -- for day, session or any ordered level.

    Never use a categorical palette for something ordered; the ramp is the whole point.
    Pair with :func:`day_cmap` for the colourbar, so the bar shows the same span the
    lines were drawn from."""
    t = theme or _STATE["theme"]
    span_lo, span_hi = _RAMP_SPAN[t]
    lo = span_lo if lo is None else lo
    hi = span_hi if hi is None else hi
    return plt.get_cmap(cmap)(np.linspace(lo, hi, max(int(n), 1)))


def day_cmap(*, cmap: str = SEQ_CMAP, theme: str | None = None, n: int = 256):
    """The ordinal ramp as a colormap, truncated exactly like :func:`day_colors`.

    Use for the colourbar: ``plt.cm.ScalarMappable(cmap=st.day_cmap(), norm=...)``.
    Passing the untruncated ``SEQ_CMAP`` instead makes the bar span a wider range than
    any line actually uses."""
    t = theme or _STATE["theme"]
    lo, hi = _RAMP_SPAN[t]
    base = plt.get_cmap(cmap)
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{cmap}_{t}", base(np.linspace(lo, hi, n)))


# ------------------------------------------------------------------------ style
def init(notebook: str, *, theme: str = "light", scale: float = 1.0,
         out_root: Path | None = None, verbose: bool = True) -> dict:
    """Apply the wrap-up style and return the palette.

    Parameters
    ----------
    notebook : str
        Short name of the calling notebook. Figures go to
        ``output/wrapup_notebooks/<notebook>/``.
    theme : {"light", "dark"}
        ``"dark"`` lightens the palette and inverts the furniture for dark slides.
    scale : float
        Multiplies every font size at once. 1.0 = 14 pt body, sized for a figure
        filling roughly half a 16:9 slide.
    """
    if theme not in _PALETTES:
        raise ValueError(f"theme must be one of {THEMES}, got {theme!r}")
    P = _PALETTES[theme]
    root = Path(out_root) if out_root is not None else repo_root() / "output" / "wrapup_notebooks"
    fd = root / notebook
    fd.mkdir(parents=True, exist_ok=True)
    _STATE.update(theme=theme, scale=float(scale), figdir=fd)

    s = float(scale)
    mpl.rcParams.update({
        # --- type: presentation sizes, not print sizes ---------------------
        "font.family": "DejaVu Sans",
        "font.size": 14 * s,
        "axes.titlesize": 16 * s,
        "axes.labelsize": 14 * s,
        "xtick.labelsize": 12 * s,
        "ytick.labelsize": 12 * s,
        "legend.fontsize": 12 * s,
        "figure.titlesize": 17 * s,
        # --- weight: thicker than print, survives a projector --------------
        "lines.linewidth": 1.8,
        "lines.markersize": 6.0,
        "patch.linewidth": 1.0,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.3, "ytick.major.width": 1.3,
        "xtick.major.size": 5.0, "ytick.major.size": 5.0,
        "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
        # --- furniture: no chartjunk ---------------------------------------
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": P["grid"], "grid.linewidth": 0.9, "grid.alpha": 0.6,
        "legend.frameon": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,
        # --- theme ----------------------------------------------------------
        "figure.facecolor": P["bg"], "axes.facecolor": P["bg"],
        "savefig.facecolor": P["bg"], "savefig.edgecolor": "none",
        "text.color": P["fg"], "axes.labelcolor": P["fg"], "axes.titlecolor": P["fg"],
        "xtick.color": P["fg"], "ytick.color": P["fg"], "axes.edgecolor": P["fg"],
        # --- output ---------------------------------------------------------
        "figure.dpi": 130, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "image.cmap": SEQ_CMAP,
    })
    if verbose:
        print(f"theme={theme}  scale={scale:g}  figures -> {fd.relative_to(repo_root())}")
    return dict(P)


def save_fig(fig, name: str, *, formats=("pdf", "png"), dpi: int = 400,
             transparent: bool = False, verbose: bool = True) -> list[Path]:
    """Save ``fig`` into the active notebook's figure directory as PDF + PNG.

    PDF keeps text as text (``pdf.fonttype=42``) so labels stay editable in Illustrator.
    ``transparent=True`` drops the background, for dropping a panel onto a slide whose
    colour does not match the theme exactly."""
    out = []
    for f in formats:
        p = figdir() / f"{name}.{f}"
        fig.savefig(p, format=f, dpi=dpi, transparent=transparent,
                    bbox_inches="tight",
                    facecolor=("none" if transparent else fig.get_facecolor()))
        out.append(p)
    if verbose:
        print("saved " + ", ".join(p.name for p in out))
    return out


# ------------------------------------------------------------------------ swatch
def swatch(theme: str | None = None, *, figsize=(11.0, 4.6)):
    """Draw the palette, grouped by family, with hex codes.

    Useful in a setup cell, and for matching slide furniture (titles, arrows drawn in
    Illustrator or PowerPoint) to the figures."""
    t = theme or _STATE["theme"]
    P = _PALETTES[t]
    ncol = max(len(v) for v in _FAMILIES.values())
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(P["bg"])
    ax.set_facecolor(P["bg"])
    for row, (fam, keys) in enumerate(_FAMILIES.items()):
        y = len(_FAMILIES) - row - 1
        ax.text(-0.15, y + 0.36, fam, ha="right", va="center",
                fontsize=12, color=P["muted"], style="italic")
        for col, k in enumerate(keys):
            ax.add_patch(plt.Rectangle((col, y), 0.9, 0.72, facecolor=P[k],
                                       edgecolor=P["fg"], lw=0.8))
            ax.text(col + 0.45, y + 0.44, k, ha="center", va="center", fontsize=10,
                    color=_readable_on(P[k]), weight="bold")
            ax.text(col + 0.45, y + 0.18, P[k], ha="center", va="center",
                    fontsize=8.5, color=_readable_on(P[k]), family="monospace")
    ax.set_xlim(-2.7, ncol)
    ax.set_ylim(-0.3, len(_FAMILIES) + 0.15)
    ax.set_axis_off()
    ax.set_title(f"wrap-up palette - {t}", color=P["fg"], fontsize=14, loc="left")
    fig.tight_layout()
    return fig, ax


def _readable_on(hex_color: str) -> str:
    """Black or white, whichever is readable on ``hex_color`` (relative luminance)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#000000" if L > 0.45 else "#FFFFFF"
