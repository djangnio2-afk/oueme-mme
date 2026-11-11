#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The figures of the paper that depend on the set of combination methods.

The visual grammar is the one of the manuscript, held in chart_style: recessive
frame, discreet horizontal grid, white gaps between bars, one labelled series,
fixed member colours.

Reads  results/tables/method_benchmark.csv, retained_weights.csv,
       performance_by_gauge.csv, weights_by_scheme.csv,
       significance_bootstrap.csv, transferability_loso.csv
       results/series/ensemble_daily.csv
Writes results/figures/
  method_heatmap.png             leading method of each family and the final
                                 product, at the seven gauges, both criteria
  performance_by_gauge.png       bars for the retained series, the whole
                                 benchmark scattered behind
  retained_weights.png           the convex weights of the delivered product
  weights_by_scheme.png          the four weighting schemes side by side
  high_flow_bias.png             high-flow volume bias, with the benchmark range
  postprocessing_comparison.png  what quantile mapping costs and returns
  gain_over_best_member.png      best single member, convex blend, final
  bonou_2010_three_states.png    the outlet before, during and after correction
  bonou_2010_convex_blend.png    the same, without the final product
  transferability.png            weights fitted elsewhere, applied here
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_style  # titres Times New Roman, gras, centrés
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mme_lib import ROOT, DATA, TAB, SERIES, FIG
from chart_style import RC, INK, INK2, GRID, MCOL, BAR, beautify, label_bars
from mme_lib import kge_val    # le mélange convexe retenu est recalculé ici

plt.rcParams.update(RC)
plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "legend.fontsize": 14})

ORD = ["BONOU", "AHLAN", "SAVE", "BETEROU", "KABOUA", "DOME", "ATCHERIGBE"]
SLABEL = {"BONOU": "Bonou", "AHLAN": "Ahlan", "SAVE": "Savè",
          "BETEROU": "Bétérou", "KABOUA": "Kaboua", "DOME": "Domè",
          "ATCHERIGBE": "Atchérigbé"}
MEM = ["GR4J", "HECHMS", "LISFLOOD"]
MEMLAB = {"GR4J": "GR4J", "HECHMS": "HEC-HMS", "LISFLOOD": "LISFLOOD"}
GREEN = "#2ca02c"

# ---------------------------------------------------------------- données ----
B = pd.read_csv(TAB / "method_benchmark.csv")
P = pd.read_csv(SERIES / "ensemble_daily.csv", parse_dates=["date"])
W = pd.read_csv(TAB / "retained_weights.csv").set_index("station")
PERF = pd.read_csv(TAB / "performance_by_gauge.csv")

# ordre de lecture des familles : du plus simple au plus appris. Il ne sert qu'à
# ranger les méthodes ; aucun nom de famille n'apparaît sur les figures.
FAM_ORDRE = ["reference", "moyenne_non_ponderee", "ponderation_statique",
             "ponderation_convexe", "convexe_stabilisee", "probabiliste",
             "regression_libre", "regression_contrainte",
             "regression_regularisee", "poids_dynamiques",
             "apprentissage_statistique", "post_traitement"]

# nom anglais de chaque méthode, identique à celui de la figure de classement
MLAB = {"B0_meilleur_membre": "Best single member",
        "B1_moyenne_simple": "Simple mean", "Mediane_membres": "Member median",
        "Moyenne_geometrique": "Geometric mean", "B2_pondere_KGE": "KGE-proportional",
        "Bates_Granger": "Bates-Granger", "Poids_Akaike": "Akaike weights",
        "Poids_bayesiens_BIC": "BIC weights", "OPTW": "Convex weights",
        "GR_convexe_NNLS": "Convex non-negative least squares", "OPTW_espace_log": "Convex weights, log space",
        "OPTW_espace_racine": "Convex weights, square-root space", "BAG_blocs": "Block bagging",
        "SHRINK_contraction": "Shrinkage", "BMA_moyenne": "Bayesian mixture mean",
        "GR_C_regression": "Least-squares regression",
        "Superensemble_Krishnamurti": "Superensemble",
        "Superensemble_modifie": "Modified superensemble", "Ridge_L2": "Ridge",
        "Lasso_L1": "Lasso", "Elastic_net": "Elastic net",
        "Commutation_mensuelle": "Monthly switching",
        "Kalman_poids_recursifs": "Recursive Kalman weights",
        "Regime_haut_bas": "Two-regime weighting", "Foret_aleatoire": "Random forest",
        "Gradient_boosting": "Gradient boosting", "Reseau_neurones": "Neural network",
        "SVR": "Support-vector regression", "OPTW_QM_partout": "Quantile mapping",
        "EMOS_NGR": "Non-negative least squares", "QRA_mediane": "Median regression",
        "MCP_Todini": "Normal-quantile regression"}


def lab(m):
    return MLAB.get(m, m.replace("_", " "))


def nm(s):
    return SLABEL.get(str(s).upper(), str(s).title())


# ordre des méthodes : par famille, puis par KGE moyen décroissant dans la famille
moy = B.groupby(["famille", "methode"]).KGE_val.mean().reset_index()
moy["rang_fam"] = moy.famille.map({f: i for i, f in enumerate(FAM_ORDRE)})
moy = moy.sort_values(["rang_fam", "KGE_val"], ascending=[True, False])
METHODES = moy.methode.tolist()
FAM = dict(zip(moy.methode, moy.famille))

# métriques du produit FINAL, par station
FINAL = {}
for st in ORD:
    g = PERF[(PERF.station == st) & (PERF.periode == "val")].set_index("serie")
    d = P[(P.station == st) & (P.periode == "val")]
    o, s = d.obs.to_numpy(), d.FINAL.to_numpy()
    FINAL[st] = dict(KGE=g.loc["FINAL", "KGE"], NSE=g.loc["FINAL", "NSE"],
                     FHV=g.loc["FINAL", "FHV"], PBIAS=g.loc["FINAL", "PBIAS"],
                     r=float(np.corrcoef(s, o)[0, 1]),
                     beta=float(s.mean() / o.mean()),
                     gamma=float((s.std() / s.mean()) / (o.std() / o.mean())))
FINAL = pd.DataFrame(FINAL).T


# Le mélange convexe RETENU, calculé une fois depuis les poids du Tableau 9.
# Ni la colonne OPTW du banc ni celle des prédictions ne le donnent : la première
# est l'optimum non contraint, qui diffère à Ahlan où le schéma retenu est la
# contraction ; la seconde porte à Bonou la série déjà propagée depuis Ahlan.
_BOOT = pd.read_csv(TAB / "significance_bootstrap.csv")
_BOOT = _BOOT[_BOOT.comparaison == "FINAL - best"].set_index(["critere", "station"])
_CONV = {}
for _s in ORD:
    _d = P[P.station == _s].set_index("date").sort_index()
    _m = (_d.index >= "2006-01-01") & (_d.index <= "2016-12-31")
    _w = W.loc[_s, ["w_GR4J", "w_HECHMS", "w_LISFLOOD"]].to_numpy(float)
    _nu = _d[MEM].to_numpy() @ (_w / _w.sum())
    _o, _p = _d["obs"].to_numpy()[_m], _nu[_m]
    _ok = np.isfinite(_o) & np.isfinite(_p)
    _o, _p = _o[_ok], _p[_ok]
    _h = _o >= np.percentile(_o, 98)
    _CONV[_s] = {"KGE": kge_val(_p, _o),
                 "NSE": 1 - np.sum((_p - _o) ** 2) / np.sum((_o - _o.mean()) ** 2),
                 "FHV": 100 * (_p[_h].sum() - _o[_h].sum()) / _o[_h].sum()}


def pivot(col):
    """stations (lignes, ordre du manuscrit) × méthodes (colonnes, par famille)"""
    p = B.pivot_table(index="station", columns="methode", values=col)
    return p.reindex(index=ORD, columns=METHODES)


# ============================================================ 1. HEATMAP =====
# Une colonne par famille du banc : la méthode qui y obtient le meilleur KGE
# moyen. La famille est portée AU-DESSUS de la grille, le nom de la méthode sous
# l'axe — les deux niveaux se lisent séparément, sans se confondre.
FAM_LAB = {"reference": "Best single member",
           "moyenne_non_ponderee": "Unweighted means",
           "ponderation_statique": "Skill-based static weights",
           "ponderation_convexe": "Optimized convex",
           "convexe_stabilisee": "Stabilized convex",
           "probabiliste": "Bayesian model averaging",
           "regression_libre": "Unconstrained regr.",
           "regression_contrainte": "Constrained regr.",
           "regression_regularisee": "Regularized regr.",
           "poids_dynamiques": "Dynamic and conditional weights",
           "apprentissage_statistique": "Statistical learning",
           "post_traitement": "Post-processing"}
FAM_COL = {"reference": "#7F7F7F", "moyenne_non_ponderee": "#1F77B4",
           "ponderation_statique": "#17BECF", "ponderation_convexe": "#D62728",
           "convexe_stabilisee": "#E377C2", "poids_dynamiques": "#FF7F0E",
           "regression_libre": "#8C564B", "regression_contrainte": "#6B4C9A",
           "regression_regularisee": "#C49C94", "post_traitement": "#2CA02C",
           "probabiliste": "#9467BD", "apprentissage_statistique": "#BCBD22"}
chefs = (B.groupby(["famille", "methode"]).KGE_val.mean().reset_index()
         .sort_values("KGE_val", ascending=False).drop_duplicates("famille")
         .sort_values("KGE_val", ascending=False))
LEAD = list(chefs.methode)
LEADFAM = dict(zip(chefs.methode, chefs.famille))
GREEN_F = "#1B7F3B"


def heatmap_panel(ax, col, final_col, ttl, vmin, vmax, seuil, bandeau=True):
    piv = B.pivot_table(index="station", columns="methode", values=col)
    piv = piv.reindex(index=ORD, columns=LEAD)
    piv["FINAL"] = FINAL[final_col].reindex(ORD).astype(float)
    A = piv.to_numpy(dtype=float)
    cols = list(piv.columns)
    im = ax.imshow(A, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab(c) if c != "FINAL" else "FINAL" for c in cols],
                       rotation=32, ha="right", fontsize=14)
    ax.set_yticks(range(len(ORD)))
    ax.set_yticklabels([nm(s) for s in ORD], fontsize=15)
    for r in range(len(ORD)):
        for c in range(len(cols)):
            v = A[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=13,
                        color="black" if v > seuil else "white")
    # bandeau de famille au-dessus de la grille, chacun relié à sa colonne
    for c, m in enumerate(cols if bandeau else []):
        f = LEADFAM.get(m)
        txt = FAM_LAB[f] if f else "Final product"
        colr = FAM_COL.get(f, GREEN_F)
        ax.text(c, -0.72, txt, rotation=36, rotation_mode="anchor",
                ha="left", va="bottom", fontsize=13, color=colr, style="italic")
        ax.plot([c - 0.44, c + 0.44], [-0.60, -0.60], color=colr, lw=3.0,
                solid_capstyle="butt", clip_on=False)
    ax.set_ylim(len(ORD) - 0.5, -0.62 if bandeau else -0.5)
    ax.axvline(len(cols) - 1.5, color="white", lw=3.4)
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(ttl, pad=168 if bandeau else 12, loc="left", fontsize=18)
    return im


# Les deux panneaux sont EMPILÉS : côte à côte, chaque grille de treize
# colonnes se retrouvait comprimée sur moins de la moitié de la largeur de la
# page et ses valeurs se chevauchaient. Empilés, ils disposent chacun de toute
# la largeur. Seul le panneau du bas porte les noms de méthodes.
fig, (h1, h2) = plt.subplots(2, 1, figsize=(13.5, 12.6), constrained_layout=True)
i1 = heatmap_panel(h1, "KGE_val", "KGE", "(a) KGE", 0.50, 0.95, 0.62)
i2 = heatmap_panel(h2, "NSE_val", "NSE", "(b) NSE", 0.35, 0.90, 0.52,
                   bandeau=False)
h1.set_xticklabels([])          # les noms de méthodes ne sont écrits qu'une fois
cb1 = fig.colorbar(i1, ax=h1, shrink=0.86, pad=0.012)
cb2 = fig.colorbar(i2, ax=h2, shrink=0.86, pad=0.012)
cb1.set_label("KGE", fontsize=15)
cb2.set_label("NSE", fontsize=15)
for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=13)
fig.suptitle("Leading method of each family and the final product, "
             "at the seven gauges", fontsize=22)
fig.savefig(FIG / "method_heatmap.png")
plt.close(fig)
print("  - method_heatmap.png")

# =================================== 2. performance par station ==============
CANON = [("B0_meilleur_membre", "Best single member", "#8A8A8A"),
         ("B1_moyenne_simple", "Simple mean", "#4C9BD6"),
         ("OPTW", "OPTW", "#D1495B"),
         ("BMA_moyenne", "BMA", "#E0A458")]


def panneau_barres(ax, col, final_key, ylab, ttl, ylim):
    piv = B.pivot_table(index="station", columns="methode", values=col)
    x = np.arange(len(ORD))
    w = 0.17
    for j, (m, nomm, col_) in enumerate(CANON):
        ax.bar(x + (j - 2) * w, [piv.loc[s, m] for s in ORD], w * 0.92,
               label=nomm, color=col_, **BAR)
    bf = ax.bar(x + 2 * w, [FINAL.loc[s, final_key] for s in ORD], w * 0.92,
                label="Final product", color=GREEN, **BAR)
    label_bars(ax, bf, "{:.2f}", dy=0.006, fs=11)
    ax.set_xticks(x)
    ax.set_xticklabels([nm(s) for s in ORD])
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    beautify(ax)
    ax.set_title(ttl, pad=12, loc="left", fontsize=17)


fig, (b1, b2) = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)
panneau_barres(b1, "KGE_val", "KGE", "KGE (test 2006-2016)", "(a) KGE", (0.40, 1.0))
panneau_barres(b2, "NSE_val", "NSE", "NSE (test 2006-2016)", "(b) NSE", (0.30, 0.95))
b1.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.19), frameon=False,
          fontsize=14)
fig.suptitle("Ensemble performance at each gauge, test period 2006-2016",
             fontsize=20)
fig.savefig(FIG / "performance_by_gauge.png")
plt.close(fig)
print("  - performance_by_gauge.png (KGE + NSE)")

# ======================================================= 3. poids retenus ====
def arrondi_somme_un(w):
    """Arrondi au centieme dont la somme fait exactement 1, plus grands restes
    d'abord — la meme regle que le Tableau 9, sans quoi 0.675 devient 0.68 dans
    la figure et 0.67 dans le tableau."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    r = np.floor(w * 100).astype(int)
    for _ in range(100 - int(r.sum())):
        r[int(np.argmax(w * 100 - r))] += 1
    return r / 100.0


# les poids publies de la figure 9 sont ceux du Tableau 9, arrondis pareillement
_W9 = {s: arrondi_somme_un([W.loc[s, f"w_{m}"] for m in MEM]) for s in ORD}
fig, ax = plt.subplots(figsize=(13.5, 6.6))
bottom = np.zeros(len(ORD))
for m in MEM:
    vals = np.array([_W9[s][MEM.index(m)] for s in ORD], dtype=float)
    b = ax.bar([nm(s) for s in ORD], vals, 0.62, bottom=bottom,
               label=MEMLAB[m], color=MCOL[m], **BAR)
    for r, v, bt in zip(b, vals, bottom):
        if v >= 0.12:
            ax.text(r.get_x() + r.get_width() / 2, bt + v / 2, f"{v:.2f}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    zorder=5, color="#333333" if m == "HECHMS" else "white")
    bottom += vals
# Plus de badge de schéma : la fenêtre de calage étant prise d'un seul tenant,
# ce sont les poids convexes optimisés aux sept stations et une étiquette
# constante, tronquée de surcroît, n'apprend rien.
ax.set_ylabel("Convex weight", fontsize=17)
ax.tick_params(labelsize=15)
ax.set_ylim(0, 1.10)
beautify(ax)
ax.set_title("Retained convex weights", pad=52, fontsize=20)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15),
          frameon=False, fontsize=15)
fig.savefig(FIG / "retained_weights.png")
plt.close(fig)
print("  - retained_weights.png")

# =========================== 4. quatre schémas de pondération (au lieu de 2) ==
# Les quatre vecteurs de la campagne de production, recalcules et verifies
# contre retained_weights.csv par src/poids_quatre_schemas.py. L'ancienne source
# ajustait les poids sur le KGE seul et donnait des
# vecteurs sans rapport avec ceux du manuscrit.
PW = pd.read_csv(TAB / "weights_by_scheme.csv")
PW = PW.set_index(["station", "schema"])


def poids_schema(sch):
    return {s: [float(PW.loc[(s, sch), f"w_{m}"]) for m in MEM] for s in ORD}


PANNEAUX = [("OPTW", "(a) Optimized convex"), ("BAG", "(b) Block bagging"),
            ("SHRINK", "(c) Shrinkage"), ("BMA", "(d) Bayesian model averaging")]
# disposition 2x2 : en une seule rangee les quatre panneaux etaient trop
# etroits, les noms de stations et les valeurs illisibles
fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True,
                        sharey=True)
axs = axs.ravel()
for ax, (sch, ttl) in zip(axs, PANNEAUX):
    src = poids_schema(sch)
    bottom = np.zeros(len(ORD))
    for k, m in enumerate(MEM):
        vals = np.array([arrondi_somme_un(src[s])[k] for s in ORD])
        b = ax.bar([nm(s) for s in ORD], vals, 0.62, bottom=bottom,
                   label=MEMLAB[m], color=MCOL[m], **BAR)
        for r, v, bt in zip(b, vals, bottom):
            if v >= 0.12:
                ax.text(r.get_x() + r.get_width() / 2, bt + v / 2, f"{v:.2f}",
                        ha="center", va="center", fontsize=15, fontweight="bold",
                        zorder=5, color="#333333" if m == "HECHMS" else "white")
        bottom += vals
    ax.set_title(ttl, fontsize=19, pad=12)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="x", rotation=28, labelsize=15)
    ax.tick_params(axis="y", labelsize=14)
    beautify(ax)
for a in (axs[0], axs[2]):
    a.set_ylabel("Weight", fontsize=17)
axs[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.10),
              frameon=False, fontsize=16)
fig.suptitle("Member weights under the four weighting schemes", fontsize=22)
fig.savefig(FIG / "weights_by_scheme.png")
plt.close(fig)
print("  - weights_by_scheme.png")

# ============================================ 5. biais de pointe (FHV) =======
piv = pivot("FHV_val")
x = np.arange(len(ORD))
w = 0.26
fig, ax = plt.subplots(figsize=(14.5, 6.8))
# Le produit final n'a pas sa place ici : la figure appuie un argument qui
# precede la correction, celui de la borne d'une combinaison convexe. Elle ne
# porte donc que la moyenne simple et le melange convexe.
for j, (m, nomm, col_) in enumerate([("B1_moyenne_simple", "Simple mean", "#4C9BD6"),
                                     ("OPTW", "Convex blend", "#D1495B")]):
    v = ([piv.loc[s, m] for s in ORD] if m != "OPTW"
         else [_CONV[s]["FHV"] for s in ORD])          # le mélange RETENU
    b = ax.bar(x + (j - 0.5) * w, v, w * 0.92, label=nomm, color=col_, **BAR)
    if m == "OPTW":
        label_bars(ax, b, "{:.0f}", dy=0.9, fs=11)
ax.axhline(0, color=INK, lw=1.0, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([nm(s) for s in ORD])
ax.set_ylabel("FHV — bias on the top 2 % of flows (%)")
beautify(ax)
ax.set_title("High-flow bias at each gauge, test period 2006-2016", pad=48,
             loc="left")
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False,
          fontsize=13)
fig.savefig(FIG / "high_flow_bias.png")
plt.close(fig)
print("  - high_flow_bias.png")

# ================================= 6. famille post-traitement (ex-axe B) =====
POST = ["OPTW", "OPTW_QM_partout", "EMOS_NGR", "QRA_mediane", "MCP_Todini"]
sub = B[B.methode.isin(POST)].copy()
sub["absFHV"] = sub.FHV_val.abs()          # moyenne des VALEURS ABSOLUES, sinon
g = sub.groupby("methode")[["KGE_val", "NSE_val", "absFHV"]].mean().reindex(POST)
gb = sub[sub.station == "BONOU"].set_index("methode").reindex(POST)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6.8), constrained_layout=True)
xx = np.arange(len(POST))
cols = ["#D1495B"] + ["#2CA02C"] * 4
w = 0.36
for ax, moy7, bon, ylab, ttl, fmt, dy in (
        (a1, g.KGE_val, gb.KGE_val, "KGE (test 2006–2016)",
         "(a) KGE", "{:.3f}", 0.002),
        (a2, g.absFHV, gb.absFHV, "Absolute high-flow bias (%)",
         "(b) |FHV|", "{:.1f}", 0.4)):
    b1 = ax.bar(xx - w / 2, moy7, w * 0.94, color=cols, **BAR)
    ax.bar(xx + w / 2, bon, w * 0.94, color=cols, hatch="////",
           edgecolor="white", linewidth=1.3, zorder=3, alpha=0.85)
    label_bars(ax, b1, fmt, dy=dy, fs=11.5)
    ax.set_xticks(xx)
    ax.set_xticklabels([lab(m) for m in POST], rotation=22, ha="right")
    ax.set_ylabel(ylab)
    ax.set_title(ttl, pad=14)
    beautify(ax)
a1.set_ylim(0.70, 0.97)
a1.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, fc="#9A9A9A", ec="white"),
    plt.Rectangle((0, 0), 1, 1, fc="#9A9A9A", ec="white", hatch="////",
                  alpha=0.85)],
    labels=["mean of the 7 gauges", "Bonou"],
    ncol=2, loc="upper left", bbox_to_anchor=(0.02, 0.99), frameon=False,
    fontsize=13)
# sans y explicite, constrained_layout réserve lui-même la place du titre :
# le forcer au-dessus du cadre le faisait chevaucher les titres de panneaux
fig.suptitle("The convex blend against quantile mapping and the three regressions usually presented as post-processors",
             fontsize=20)
fig.savefig(FIG / "postprocessing_comparison.png")
plt.close(fig)
print("  - postprocessing_comparison.png")

# ============================== 7. progression membre -> OPTW -> FINAL =======
# Les deux moitiés du critère du projet, l'une sous l'autre : on veut voir si le
# gain du MME sur le meilleur membre tient quand on passe du KGE au NSE.
def progression(ax, col, final_key, ylab, ttl, ylim, dy):
    piv = B.pivot_table(index="station", columns="methode", values=col)
    x = np.arange(len(ORD))
    w = 0.26
    for j, (v, nomm, col_) in enumerate([
            ([piv.loc[s, "B0_meilleur_membre"] for s in ORD],
             "Best single member", "#8A8A8A"),
            ([_CONV[s][final_key] for s in ORD], "Convex blend", "#D1495B"),
            ([FINAL.loc[s, final_key] for s in ORD], "Final product", GREEN)]):
        b = ax.bar(x + (j - 1) * w, v, w * 0.92, label=nomm, color=col_, **BAR)
        if j == 2:
            label_bars(ax, b, "{:.3f}", dy=0.006, fs=11)
    for i, s in enumerate(ORD):
        d = float(_BOOT.loc[(final_key, s), "observe"])
        ax.text(i, max(FINAL.loc[s, final_key],
                       piv.loc[s, "B0_meilleur_membre"]) + dy,
                f"{d:+.3f}", ha="center", fontsize=12,
                color=GREEN if d >= 0 else "#B03A2E", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([nm(s) for s in ORD])
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    beautify(ax)
    ax.set_title(ttl, pad=12, loc="left", fontsize=17)


fig, (p1, p2) = plt.subplots(2, 1, figsize=(14.5, 12), constrained_layout=True)
progression(p1, "KGE_val", "KGE", "KGE (test 2006-2016)", "(a) KGE",
            (0.40, 1.03), 0.045)
progression(p2, "NSE_val", "NSE", "NSE (test 2006-2016)", "(b) NSE",
            (0.30, 1.00), 0.045)
p1.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.19), frameon=False)
fig.suptitle("From the best single member to the final product", fontsize=20)
fig.savefig(FIG / "gain_over_best_member.png")
plt.close(fig)
print("  - gain_over_best_member.png")

# Le diagramme de Taylor et la decomposition du KGE du manuscrit ne portent
# que les trois membres et le produit FINAL : aucun schema ancien n'y figure.
# Ils restent donc exactement tels quels (produits par figures_all.py).

# =========================== 8. Bonou, crue de 2010, produit V3 =============
# Ces deux figures venaient des scripts de chaîne, qui recalculent le MME avec
# l'ancienne règle. Elles sont reconstruites ici sur les prédictions V3.
bo = P[P.station == "BONOU"].set_index("date").sort_index()
# La fenêtre couvre 2009 et 2010. Bonou n'a aucune observation en 2009 : la
# ligne observée y est simplement absente, ce qui montre du même coup que la
# chaîne tourne en continu, y compris sur les années non jaugées.
bo = bo.loc["2009-01-01":"2011-06-01"]
# Le membre de reference est choisi sur le calage, comme partout ailleurs
# dans l'article ; a Bonou les deux fenetres designent le meme, GR4J.
_g = PERF[(PERF.station == "BONOU") & (PERF.periode == "cal")].set_index("serie")
best_bo = max(MEM, key=lambda m: _g.loc[m, "KGE"])
# La colonne OPTW des prédictions porte, À BONOU SEULEMENT, la série déjà
# propagée depuis Ahlan : son pic de 2010 culmine à 1 487 m³/s, soit +40 %, et
# non les 1 761 m³/s (+66 %) du mélange convexe nu. Étiquetée « convex blend »,
# elle rendait illisible le chiffre central du paragraphe. On reconstruit donc
# le mélange nu à partir des poids retenus et l'on trace les trois états.
_wb = W.loc["BONOU", ["w_GR4J", "w_HECHMS", "w_LISFLOOD"]].to_numpy(float)
bo = bo.assign(CONVEXE=bo[MEM].to_numpy() @ (_wb / _wb.sum()))

for nom, avec_final, ttl in (
        ("bonou_2010_three_states.png", True, "Bonou outlet - 2009 and 2010 flood seasons"),
        ("bonou_2010_convex_blend.png", False,
         "Bonou outlet - 2009 and 2010, weighted combination")):
    fig, ax = plt.subplots(figsize=(15, 7.3))
    ax.fill_between(bo.index, bo.BMA_lo, bo.BMA_hi, color="#2E8B57", alpha=0.16,
                    label="BMA 90 % interval")
    ax.plot(bo.index, bo.obs, color="k", lw=2.2, label="Observed")
    ax.plot(bo.index, bo[best_bo], color="#888888", lw=1.3, ls="--",
            label="Best single member (" + MEMLAB[best_bo] + ")")
    ax.plot(bo.index, bo.CONVEXE, color="#D1495B", lw=1.5, ls=":",
            label="Convex blend")
    if avec_final:
        ax.plot(bo.index, bo.OPTW, color="#E8A33D", lw=1.5,
                label="After propagation from Ahlan")
        ax.plot(bo.index, bo.FINAL, color="#2E8B57", lw=2.2,
                label="Final product")
    ax.set_ylabel("Discharge (m3/s)".replace("m3", "m\u00b3"), fontsize=18)
    ax.set_title(ttl, fontsize=20, pad=14)
    ax.legend(loc="upper left", fontsize=15)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=16)
    fig.savefig(FIG / nom, bbox_inches="tight")
    plt.close(fig)
    print("  -", nom)

# ============================= 9. transferabilite spatiale (LOSO) ==========
# Reconstruite ici : ordre de stations commun a tout le jeu, valeurs sur les
# DEUX series (la comparaison est le sujet de la figure) et ecritures agrandies.
LO = pd.read_csv(TAB / "transferability_loso.csv").set_index("station")
x = np.arange(len(ORD))
w = 0.38
fig, ax = plt.subplots(figsize=(14, 6.6))
# Les deux valeurs d'une même station sont proches : posées à la même hauteur
# elles se touchaient. On décale verticalement celle de la barre de gauche.
for dx, col_, nomm, coul, dy_ in ((-w / 2, "KGE_OPTW_stationspecific",
                                   "Station-specific weights", "#8A8A8A", 0.048),
                                  (w / 2, "KGE_OPTW_LOSO",
                                   "Transferred (leave-one-station-out)", GREEN, 0.008)):
    v = [float(LO.loc[s, col_]) for s in ORD]
    b = ax.bar(x + dx, v, w * 0.94, label=nomm, color=coul, **BAR)
    label_bars(ax, b, "{:.3f}", dy=dy_, fs=12)
ax.set_xticks(x)
ax.set_xticklabels([nm(s) for s in ORD], fontsize=16)
ax.tick_params(axis="y", labelsize=14)
ax.set_ylabel("KGE", fontsize=17)
ax.set_ylim(0.40, 1.0)
beautify(ax)
ax.set_title("Spatial transferability of the convex weights", fontsize=20,
             pad=52)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False,
          fontsize=14)
fig.savefig(FIG / "transferability.png", bbox_inches="tight")
plt.close(fig)
print("  - transferability.png")


print(f"\n{len(METHODES)} méthodes du banc intégrées ; figures écrites dans {FIG}")
