#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flood-frequency analysis of the annual maxima, observed and simulated.

Fitting. Ordinary maximum likelihood diverges on samples of five to eleven
values: the generalized extreme-value likelihood is unbounded, and the
optimizer returns shapes from -1.27 to +2.01 depending on the sample, which
puts the hundred-year flood at Savè at 42,000 m3/s. The fit is therefore the
generalized maximum likelihood of Martins and Stedinger (2000), that is,
maximum likelihood with a Beta(6, 9) prior confining the shape to ]-0.5, 0.5[,
the standard remedy in hydrology for short records. L-moments only provide the
starting point of the optimization.

Year selection. An annual maximum means something only if the flood season is
covered. A filter on the total number of days is misleading: it accepted Savè
2012 (82 season days out of 153) and rejected Savè 2009 (152 out of 153), the
dry season being complete in both cases, and it let through Kaboua 2011, whose
annual maximum was 17 m3/s over 101 days. The requirement here is 90 % of the
1 July to 31 October window, where every observed maximum falls.

Reads  results/series/ensemble_test.csv
Writes results/tables/flood_quantiles.csv, flood_distribution_parameters.csv
       results/figures/annual_maxima.png, flood_frequency.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_style  # titres Times New Roman, gras, centrés
from scipy.special import gamma as _gamma, betaln
from scipy.optimize import minimize
from mme_lib import TAB, SERIES, FIG
from chart_style import beautify, BAR

plt.rcParams.update({"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

SLABEL = {"BONOU": "Bonou", "AHLAN": "Ahlan", "SAVE": "Savè", "BETEROU": "Bétérou",
          "KABOUA": "Kaboua", "DOME": "Domè", "ATCHERIGBE": "Atchérigbé"}
ORD = ["BONOU", "AHLAN", "SAVE", "BETEROU", "KABOUA", "DOME", "ATCHERIGBE"]
GREEN, BLACK = "#2ca02c", "k"
TR = np.array([2, 5, 10, 20, 50])
SEASON = (7, 10)          # juillet a octobre : tous les maxima annuels y tombent
COVER = 0.90


# ------------------------------------------------------------- point de depart
def _lmom3(x):
    x = np.sort(np.asarray(x, float)); n = len(x); j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    l1, l2 = b0, 2 * b1 - b0
    return l1, l2, (6 * b2 - 6 * b1 + b0) / l2


def _start(x):
    """Estimateur par L-moments (Hosking), utilise seulement pour initialiser."""
    l1, l2, t3 = _lmom3(x)
    c = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
    k = 7.8590 * c + 2.9554 * c * c
    if abs(k) < 1e-6:
        a = l2 / np.log(2)
        return l1 - 0.5772156649 * a, a, 0.0
    gk = _gamma(1 + k)
    a = l2 * k / ((1 - 2.0 ** -k) * gk)
    return l1 - a * (1 - gk) / k, a, k


# --------------------------------- vraisemblance maximale generalisee (GML)
PB, QB = 6.0, 9.0          # a priori Beta de Martins & Stedinger (2000) sur la forme


def _neg_post(th, x):
    """-(log-vraisemblance + log a priori). Parametrage de Hosking : la GEV vaut
    F(x) = exp(-z^(1/k)) avec z = 1 - k (x - u)/s."""
    u, ls, k = th
    if not (-0.499 < k < 0.499):
        return 1e12
    s = np.exp(ls)
    if abs(k) < 1e-8:
        # Limite de Gumbel, traitee analytiquement. Sans ce garde-fou, le terme
        # k(x-u)/s disparait sous l'epsilon machine, z devient exactement 1, la
        # somme des log s'annule et l'objectif chute artificiellement : le
        # minimiseur retient alors ce faux minimum (cas d'Atcherigbe).
        y = (x - u) / s
        ll = -len(x) * ls - np.sum(y) - np.sum(np.exp(-y))
    else:
        z = 1.0 - k * (x - u) / s
        if np.any(z <= 1e-9):
            return 1e12
        ll = -len(x) * ls + (1.0 / k - 1.0) * np.sum(np.log(z)) - np.sum(z ** (1.0 / k))
    if not np.isfinite(ll):
        return 1e12
    lpr = (PB - 1) * np.log(0.5 + k) + (QB - 1) * np.log(0.5 - k) - betaln(PB, QB)
    return -(ll + lpr)


def gev_gml(x):
    """(loc, scale, k) par vraisemblance maximale generalisee."""
    u0, s0, k0 = _start(x)
    k0 = float(np.clip(k0, -0.4, 0.4))
    best = None
    for kk in (k0, -0.1, 0.0, 0.1):
        r = minimize(_neg_post, [u0, np.log(max(s0, 1e-3)), kk], args=(x,),
                     method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-8, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    u, ls, k = best.x
    return float(u), float(np.exp(ls)), float(k)


def qgev(T, u, s, k):
    y = -np.log(1.0 - 1.0 / np.asarray(T, float))
    return u - s * np.log(y) if abs(k) < 1e-9 else u + s * (1 - y ** k) / k


# ------------------------------------------------------- maxima annuels valides
def annual_maxima(d):
    m = d.date.dt.month
    ns = d[(m >= SEASON[0]) & (m <= SEASON[1])].groupby(d.date.dt.year).size()
    full = pd.date_range("2001-07-01", "2001-10-31").size          # 123 jours
    keep = ns[ns >= COVER * full].index
    return d[d.date.dt.year.isin(keep)].groupby(d.date.dt.year)[["obs", "FINAL"]].max().sort_index()


fin = pd.read_csv(SERIES / "ensemble_test.csv", parse_dates=["date"]).dropna(subset=["obs", "FINAL"])
AM = {st: annual_maxima(fin[fin.station == st]) for st in ORD}

FIT, rows = {}, []
for st in ORD:
    g = AM[st]; o, m = g.obs.to_numpy(), g.FINAL.to_numpy()
    fo, fm = gev_gml(o), gev_gml(m)
    FIT[st] = dict(n=len(g), obs=fo, mme=fm)
    for T in TR:
        rows.append(dict(station=SLABEL[st], T=int(T),
                         obs=qgev(T, *fo), mme=qgev(T, *fm)))
Q = pd.DataFrame(rows)
Q["dev_pct"] = 100 * (Q.mme - Q.obs) / Q.obs

print("=== annees retenues (saison de crue couverte a 90 %) ===")
for st in ORD:
    print(f"  {SLABEL[st]:11s} n = {len(AM[st]):2d}   {', '.join(str(y) for y in AM[st].index)}")

print("\n=== ajustement GEV (vraisemblance maximale generalisee) ===")
print("station      n    k obs   k MME    accord annee par annee (r)")
for st in ORD:
    g = AM[st]; r = np.corrcoef(g.obs, g.FINAL)[0, 1]
    print(f"{SLABEL[st]:11s} {FIT[st]['n']:2d}  {FIT[st]['obs'][2]:+6.3f}  {FIT[st]['mme'][2]:+6.3f}      {r:+.2f}")
ko = np.array([FIT[s]["obs"][2] for s in ORD]); km = np.array([FIT[s]["mme"][2] for s in ORD])
print(f"\nforme observee : de {ko.min():+.3f} a {ko.max():+.3f} (moyenne {ko.mean():+.3f})")
print(f"forme simulee  : de {km.min():+.3f} a {km.max():+.3f} (moyenne {km.mean():+.3f})")

print("\n=== quantiles GEV (m3/s) ===")
piv = Q.pivot(index="station", columns="T", values=["obs", "mme"]).round(0)
print(piv.loc[[SLABEL[s] for s in ORD]].to_string())
print("\n=== ecart MME - observe (%) ===")
d = Q.pivot(index="station", columns="T", values="dev_pct").round(1)
d.loc["MOYENNE"] = d.mean(); d.loc["MEDIANE"] = d.iloc[:-1].median()
print(d.loc[[SLABEL[s] for s in ORD] + ["MOYENNE", "MEDIANE"]].to_string())

Q.round(1).to_csv(TAB / "flood_quantiles.csv", index=False)
pd.DataFrame([dict(station=SLABEL[s], n=FIT[s]["n"],
                   loc_obs=FIT[s]["obs"][0], scale_obs=FIT[s]["obs"][1], k_obs=FIT[s]["obs"][2],
                   loc_mme=FIT[s]["mme"][0], scale_mme=FIT[s]["mme"][1], k_mme=FIT[s]["mme"][2])
              for s in ORD]).round(3).to_csv(TAB / "flood_distribution_parameters.csv", index=False)


def grid():
    fig, ax = plt.subplots(2, 4, figsize=(23, 12), constrained_layout=True)
    ax = ax.ravel(); ax[7].axis("off"); return fig, ax


# ------------------------------------------------------------ maxima annuels
fig, ax = grid()
fig.suptitle("Annual maximum daily streamflow", fontsize=22)
for i, st in enumerate(ORD):
    g = AM[st]; yrs = g.index.to_numpy()
    ax[i].bar(yrs - 0.2, g.obs, 0.4, label="Observed", color=BLACK, **BAR)
    ax[i].bar(yrs + 0.2, g.FINAL, 0.4, label="MME", color=GREEN, **BAR)
    ax[i].set_title(f"{SLABEL[st]}  (n = {len(g)})")
    ax[i].set_ylabel("Annual max Q (m³/s)")
    ax[i].set_xticks(yrs); ax[i].set_xticklabels(yrs, rotation=90)
    beautify(ax[i])
hb = [plt.Rectangle((0, 0), 1, 1, color=BLACK, label="Observed"),
      plt.Rectangle((0, 0), 1, 1, color=GREEN, label="MME")]
fig.legend(handles=hb, ncol=2, loc="outside lower center", frameon=False, fontsize=16)
fig.savefig(FIG / "annual_maxima.png"); plt.close(fig)

# --------------------------------------------------------- courbes GEV
YT = -np.log(-np.log(1 - 1.0 / TR))
fig, ax = grid()
fig.suptitle("Flood-frequency curves", fontsize=22)
for i, st in enumerate(ORD):
    a = ax[i]; g = AM[st]; n = len(g)
    yy = np.linspace(YT[0] - 0.55, YT[-1] + 0.15, 240)
    Tg = 1.0 / (1.0 - np.exp(-np.exp(-yy)))
    pp = (np.arange(1, n + 1) - 0.4) / (n + 0.2)          # positions de Cunnane
    for arr, col, mk, f in [(g.obs.to_numpy(), BLACK, "o", FIT[st]["obs"]),
                            (g.FINAL.to_numpy(), GREEN, "s", FIT[st]["mme"])]:
        a.scatter(-np.log(-np.log(pp)), np.sort(arr), s=48, color=col, marker=mk, zorder=4)
        a.plot(yy, qgev(Tg, *f), color=col, lw=2.4, zorder=3)
    a.set_xticks(YT); a.set_xticklabels([str(t) for t in TR]); a.set_xlim(yy[0], yy[-1])
    a.set_title(f"{SLABEL[st]}  (n = {n})")
    a.set_xlabel("Return period (yr)"); a.set_ylabel("Q (m³/s)")
    beautify(a); a.grid(axis="both", alpha=0.3)
h = [plt.Line2D([], [], color=BLACK, marker="o", ls="-", lw=2.4, ms=9, label="Observed"),
     plt.Line2D([], [], color=GREEN, marker="s", ls="-", lw=2.4, ms=9, label="MME")]
fig.legend(handles=h, ncol=2, loc="outside lower center", frameon=False, fontsize=16)
fig.savefig(FIG / "flood_frequency.png"); plt.close(fig)
print(f"\nfigures ecrites dans {FIG}")
