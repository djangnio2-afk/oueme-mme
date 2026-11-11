#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily hydrographs over the test period, every model on the same axes.

Three gauges, not seven: over 4,018 days seven panels would be unreadable, and
these three carry the argument. Bétérou upstream, where the peaks are missed by
every member; Domè on the Zou, the only gauge with regime-conditional weights;
Bonou at the outlet, where the guarded correction acts.

Five curves on one axis. No range band and no inset: all five series are drawn,
and what makes them readable is the graphical hierarchy rather than a reduction
of their number. The three members are thin dashed lines, each with its own
dash pattern, recognizable where the colours overlap and in black and white;
the observations and the final product are solid and thicker, laid on top. Tall
panels separate the peaks better than any colour setting.

Two scales. On the linear one the peaks dominate and the recessions flatten
onto zero; on the logarithmic one the low flows and the recessions become
visible, which is where the members diverge most.

Reads  results/series/ensemble_test.csv, results/tables/ensemble_summary.csv
       data/member_simulations_daily.csv
Writes results/figures/hydrographs_test.png, hydrographs_test_log.png
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plot_style  # titres Times New Roman, gras, centres
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mme_lib import TAB, FIG, MEMBERS, SERIES, load_long

COL_OBS, COL_FIN = "#111111", "#1E8C3A"
# Epaisseurs. Tout est fin — la figure est posee sur 17,7 cm, soit une
# reduction de 0.62 : 0.45 pt devient 0.28 pt sur la page, ce que les 400 ppp
# rendent encore. La HIERARCHIE compte plus que l'epaisseur absolue : les
# deux series pleines restent presque deux fois plus epaisses que les
# membres, sinon on ne sait plus ou regarder.
LW_M, LW_FIN, LW_OBS = 0.42, 0.78, 0.72
# un motif de tirets par membre : la figure reste lisible en noir et blanc
TRAIT = {"GR4J":     ("#1F6FB4", (0, (4.0, 1.4))),
         "HECHMS":   ("#E07B13", (0, (1.3, 1.3))),
         "LISFLOOD": ("#8E52B8", (0, (5.0, 1.2, 1.0, 1.2)))}
LAB_M = {"GR4J": "GR4J", "HECHMS": "HEC-HMS", "LISFLOOD": "LISFLOOD"}
PANNEAUX = [("BETEROU", "Bétérou", "upstream"),
            ("DOME", "Domè", "Zou tributary"),
            ("BONOU", "Bonou", "basin outlet")]
CRUE = (pd.Timestamp("2010-06-15"), pd.Timestamp("2010-12-31"))
VARIANTES = {"lin": "hydrographs_test.png",
             "log": "hydrographs_test_log.png"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 400,
                     "savefig.bbox": "tight"})

d = pd.read_csv(SERIES / "ensemble_test.csv", parse_dates=["date"])
s = pd.read_csv(TAB / "ensemble_summary.csv").set_index("station")
mem = load_long()[["date", "station"] + MEMBERS]
d = d.merge(mem, on=["date", "station"], how="left")


def dessiner(variante):
    log = variante == "log"
    fig, axs = plt.subplots(3, 1, figsize=(11.2, 8.6), sharex=True)
    fig.subplots_adjust(hspace=0.30)

    for ax, (cle, nom, role) in zip(axs, PANNEAUX):
        g = d[d.station == cle].sort_values("date")
        ax.axvspan(*CRUE, color="#F2C94C", alpha=0.18, lw=0, zorder=0)

        # les membres d'abord, dessous, fins et pointilles
        for m in MEMBERS:
            col, tir = TRAIT[m]
            ax.plot(g.date, g[m], color=col, lw=LW_M, ls=tir, alpha=0.92,
                    zorder=2, solid_capstyle="butt")
        # puis les deux series qui portent la lecture, pleines et par-dessus
        ax.plot(g.date, g.FINAL, color=COL_FIN, lw=LW_FIN, zorder=4)
        ax.plot(g.date, g.obs, color=COL_OBS, lw=LW_OBS, zorder=5)

        r = s.loc[cle]
        ax.set_title("%s — %s" % (nom, role), loc="left", fontsize=11,
                     fontweight="bold", pad=3)
        ax.text(0.997, 0.945,
                "final MME:  KGE %.3f   NSE %.3f   FHV %+.1f%%"
                % (r.KGE_FINAL, r.NSE_FINAL, r.FHV_FINAL),
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#333333",
                bbox=dict(fc="white", ec="#CCCCCC", lw=0.6, pad=2.6,
                          alpha=0.94))
        ax.set_ylabel("Q (m³/s)", fontsize=10)
        if log:
            ax.set_yscale("log")
            haut = 1.9 * max(g.obs.max(), g[MEMBERS].max().max())
            ax.set_ylim(1.0, haut)
        else:
            ax.set_ylim(0, 1.08 * max(g.obs.max(), g.FINAL.max(),
                                      g[MEMBERS].max().max()))
        ax.margins(x=0.004)
        ax.grid(axis="y", alpha=0.22, lw=0.6, which="major")
        for c in ("top", "right"):
            ax.spines[c].set_visible(False)

    axs[-1].xaxis.set_major_locator(mdates.YearLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # les cles de legende restent un peu plus epaisses que le trace : a
    # 0.42 pt un tiret de legende ne se voit plus du tout
    poignees = ([Line2D([0], [0], color=COL_OBS, lw=1.2, label="Observed"),
                 Line2D([0], [0], color=COL_FIN, lw=1.3, label="Final MME")] +
                [Line2D([0], [0], color=TRAIT[m][0], lw=0.9, ls=TRAIT[m][1],
                        label=LAB_M[m]) for m in MEMBERS] +
                [Patch(fc="#F2C94C", alpha=0.18, ec="none",
                       label="2010 flood season")])
    fig.legend(handles=poignees, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=len(poignees), frameon=False,
               fontsize=10, columnspacing=1.8, handlelength=2.6,
               handletextpad=0.6)
    fig.suptitle("Final ensemble and its three members against observations, "
                 "independent 2006–2016 period", y=1.052, fontsize=13,
                 fontweight="bold")
    fig.savefig(FIG / VARIANTES[variante], bbox_inches="tight")
    plt.close(fig)
    print("Saved ->", VARIANTES[variante])


for v in (sys.argv[1:] or list(VARIANTES)):
    dessiner(v)
