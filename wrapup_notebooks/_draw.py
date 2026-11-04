"""Annotation helpers shared by the wrap-up notebooks.

The split, deliberately:

* ``_style.py``  -- aesthetics (palette, rcParams, saving). No drawing.
* ``_draw.py``   -- **one layer added to an axes you already made.** Every function
  here takes ``ax`` first, takes plain numpy arrays (never a project data object),
  and never creates a figure, chooses a layout, or sets limits.
* the notebook cell -- the figure itself: ``plt.subplots``, every ``ax.plot``,
  every label and limit.

So a reader still sees what the figure *is* in the cell. What lives here is only the
boilerplate that carries no information -- "loop over 200 trials and draw an axvspan
for each" -- which would otherwise be pasted into every notebook.

Time arguments are uniform: arrays are in **frames**, and ``sr`` / ``t0`` convert to the
axis units actually plotted, as ``(frame - t0) / sr``. Pass ``sr=1, t0=0`` to stay in
frames.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

__all__ = ["runs", "span_bands", "shade_trials", "mark_events",
           "legend_from", "label_panel", "hz_axis"]


# --------------------------------------------------------------------------- runs
def runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """``(start, stop)`` of every contiguous True run in ``mask``, stop-exclusive.

    The one primitive everything else here is built on, and useful on its own for
    turning a per-frame task flag into intervals."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


# --------------------------------------------------------------------- shading
def span_bands(ax, mask, *, color, alpha: float = 0.18, sr: float = 30.0,
               t0: float = 0.0, zorder: float = 0.0, lw: float = 0.0,
               min_frames: int = 1, label: str | None = None):
    """Shade every contiguous True run of ``mask`` as a vertical band.

    ``min_frames`` drops single-frame flicker, which several of the rig's state flags
    have. Only the first band carries ``label``, so the legend gets one entry."""
    out = []
    for i, (a, b) in enumerate(runs(mask)):
        if b - a < min_frames:
            continue
        out.append(ax.axvspan((a - t0) / sr, (b - t0) / sr, color=color, alpha=alpha,
                              lw=lw, zorder=zorder,
                              label=label if (label and not out) else None))
    return out


def shade_trials(ax, starts, ends, outcomes, *, colors: dict, alpha: float = 0.16,
                 sr: float = 30.0, t0: float = 0.0, zorder: float = 0.0,
                 clip: tuple[float, float] | None = None):
    """Shade each trial by outcome.

    ``colors`` maps the outcome value to a colour, e.g.
    ``{1: P["correct"], 0: P["incorrect"]}``. Trials are skipped where start or end is
    not finite, or where they fall entirely outside ``clip`` (a frame window)."""
    out = []
    lo, hi = (clip if clip is not None else (-np.inf, np.inf))
    for s, e, oc in zip(np.asarray(starts, float), np.asarray(ends, float),
                        np.asarray(outcomes, float)):
        if not (np.isfinite(s) and np.isfinite(e)) or e <= lo or s >= hi:
            continue
        c = colors.get(oc, colors.get(int(oc) if np.isfinite(oc) else None))
        if c is None:
            continue
        out.append(ax.axvspan((max(s, lo) - t0) / sr, (min(e, hi) - t0) / sr,
                              color=c, alpha=alpha, lw=0, zorder=zorder))
    return out


def mark_events(ax, frames, y, *, sr: float = 30.0, t0: float = 0.0,
                clip: tuple[float, float] | None = None, **kw):
    """Scatter events (reward frames, threshold crossings, ...) at height ``y``.

    ``y`` is a scalar, or an array sampled at ``frames`` after clipping. Remaining
    keyword arguments go straight to ``ax.scatter``."""
    f = np.asarray(frames, float)
    f = f[np.isfinite(f)]
    if clip is not None:
        f = f[(f >= clip[0]) & (f < clip[1])]
    if f.size == 0:
        return None
    yy = np.full(f.size, float(y)) if np.isscalar(y) else np.asarray(y, float)
    return ax.scatter((f - t0) / sr, yy, **kw)


# ---------------------------------------------------------------------- legend
def legend_from(ax, entries: Sequence[tuple], **kw):
    """Build a legend from ``(label, kind, colour)`` tuples, no proxy-handle boilerplate.

    ``kind`` is ``"line"``, ``"patch"`` or ``"marker"``. A 4th element, if present, is a
    dict of extra artist kwargs (``lw``, ``ls``, ``alpha``, ``marker``, ...)::

        dw.legend_from(ax, [("cursor", "line", P["cursor"]),
                            ("correct", "patch", P["correct"], dict(alpha=0.16)),
                            ("reward", "marker", P["reward"], dict(marker="o"))])
    """
    handles = []
    for e in entries:
        label, kind, color = e[0], e[1], e[2]
        extra = dict(e[3]) if len(e) > 3 else {}
        if kind == "patch":
            handles.append(Patch(facecolor=color, label=label, **extra))
        elif kind == "marker":
            extra.setdefault("marker", "o")
            extra.setdefault("ls", "none")
            handles.append(Line2D([], [], color=color, label=label, **extra))
        elif kind == "line":
            handles.append(Line2D([], [], color=color, label=label, **extra))
        else:
            raise ValueError(f"kind must be line/patch/marker, got {kind!r}")
    return ax.legend(handles=handles, **kw)


def label_panel(ax, letter: str, *, x: float = -0.02, y: float = 1.06,
                size: float = 16.0, weight: str = "bold", **kw):
    """Put a panel letter in axes coordinates, above and left of the axes."""
    return ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
                   fontsize=size, fontweight=weight, **kw)


# ------------------------------------------------------------------------ axes
def hz_axis(ax, *, side: str = "right", hz_ticks=(1000, 2000, 4000, 8000, 16000),
            label: str = "tone (Hz)", color=None, ref_hz: float = 1.0):
    """Add a Hz-labelled twin to an axis already plotted in **octaves** (log2 Hz).

    The octave scale is the perceptual one and is what the analysis uses, but an
    audience reads Hz, so both belong on the figure. ``ref_hz`` is the frequency the
    octave scale is measured from: leave it at 1 for an absolute ``log2(f)`` axis, or
    pass 1000 for an axis plotted as ``log2(f / 1 kHz)``. Returns the twin axes."""
    tw = ax.twinx() if side == "right" else ax.twiny()
    setter = (tw.set_ylim, tw.set_yticks, tw.set_yticklabels, tw.set_ylabel)
    if side != "right":
        setter = (tw.set_xlim, tw.set_xticks, tw.set_xticklabels, tw.set_xlabel)
    lim = ax.get_ylim() if side == "right" else ax.get_xlim()
    setter[0](lim)
    t = np.asarray([h for h in hz_ticks if lim[0] <= np.log2(h / ref_hz) <= lim[1]], float)
    setter[1](np.log2(t / ref_hz))
    setter[2]([f"{int(v):,}" if v < 10000 else f"{v/1000:g}k" for v in t])
    setter[3](label)
    tw.spines["right" if side == "right" else "top"].set_visible(True)
    if color is not None:
        tw.tick_params(colors=color)
        (tw.yaxis if side == "right" else tw.xaxis).label.set_color(color)
    return tw
