#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared publication style for the MME bar charts: recessive frame, white bar
gaps, selective value labels, and ONE colour per entity across every figure."""
from __future__ import annotations

INK   = "#333333"   # primary text ink
INK2  = "#555555"   # secondary ink (value labels)
GRID  = "#E4E2DD"   # recessive horizontal grid
SPINE = "#999999"

# manuscript-wide member colours (identical in every figure)
MCOL = {"GR4J": "#1f77b4", "HECHMS": "#ff7f0e", "LISFLOOD": "#9467bd"}
# method colours (fixed across all method charts; grey = neutral baseline)
METH = {"best": "#8A8A8A", "mean": "#4C9BD6", "OPTW": "#D1495B",
        "BMA": "#E0A458", "sqrt": "#7E63C5", "high": "#E0A458",
        "QM": "#2ca02c", "FINAL": "#2ca02c"}

RC = {"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
      "xtick.labelsize": 14, "ytick.labelsize": 13.5, "legend.fontsize": 13.5,
      "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
      "axes.edgecolor": SPINE, "axes.linewidth": 1.0,
      "xtick.color": INK, "ytick.color": INK, "text.color": INK,
      "axes.labelcolor": INK}

# white edges create a clean gap between adjacent bars / stacked segments
BAR = dict(edgecolor="white", linewidth=1.3, zorder=3)

def beautify(ax):
    """No top/right spines; light horizontal grid drawn behind the bars."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

def label_bars(ax, bars, fmt="{:.2f}", dy=0.006, fs=11):
    """Value labels above ONE selected series (secondary ink, never series colour)."""
    for b in bars:
        h = b.get_height()
        va, off = ("bottom", dy) if h >= 0 else ("top", -dy)
        ax.text(b.get_x() + b.get_width() / 2, h + off, fmt.format(h),
                ha="center", va=va, fontsize=fs, color=INK2, zorder=5)
