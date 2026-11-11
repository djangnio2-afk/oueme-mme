# -*- coding: utf-8 -*-
"""Paired year-block bootstrap, on both efficiencies.

Both criteria are tested rather than one. The members are well calibrated, so
the gain of the combination on the mean Kling-Gupta efficiency turns slightly
negative while it stays clear on the Nash-Sutcliffe efficiency, and a verdict
reported on a single criterion would be the verdict on whichever criterion
suits.

Resampling: the calendar years of the test period are drawn with replacement,
jointly for all stations, which preserves the seasonal autocorrelation and the
dependence between stations.

Reads  results/series/ensemble_daily.csv
       results/tables/performance_by_gauge.csv
Writes results/tables/significance_bootstrap.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from mme_lib import ROOT, DATA, TAB, SERIES, FIG
from mme_lib import STATIONS, kge_val

B = 2000
RNG = np.random.default_rng(7)


def nse(s, o):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


P = pd.read_csv(SERIES / "ensemble_daily.csv", parse_dates=["date"])
PERF = pd.read_csv(TAB / "performance_by_gauge.csv")
MEM = ["GR4J", "HECHMS", "LISFLOOD"]

d = P[(P.periode == "val")].dropna(subset=["obs"]).copy()
d["an"] = d.date.dt.year
ANS = np.sort(d.an.unique())

# Meilleur membre choisi SUR LE CALAGE, comme toute autre décision de ce
# travail. Le choisir sur la validation reviendrait à donner à la référence une
# information que l'ensemble n'a pas : la sélection y gagne 0.003 de KGE moyen
# (0.837 au lieu de 0.834), et ce gain est une fuite, pas une performance.
best = {}
for st in STATIONS:
    g = PERF[(PERF.station == st) & (PERF.periode == "cal")].set_index("serie")
    best[st] = max(MEM, key=lambda m: g.loc[m, "KGE"])

# séries alignées par station
SER = {}
for st in STATIONS:
    s = d[d.station == st]
    SER[st] = dict(an=s.an.to_numpy(), obs=s.obs.to_numpy(),
                   best=s[best[st]].to_numpy(), OPTW=s.OPTW.to_numpy(),
                   FINAL=s.FINAL.to_numpy())

PAIRES = [("OPTW", "best"), ("FINAL", "best"), ("FINAL", "OPTW")]
CRIT = {"KGE": kge_val, "NSE": nse}
lignes = []

for nom_c, f in CRIT.items():
    # valeurs observées
    obsv = {st: {k: f(SER[st][k], SER[st]["obs"]) for k in ("best", "OPTW", "FINAL")}
            for st in STATIONS}
    # tirages
    diff = {st: {p: np.empty(B) for p in PAIRES} for st in STATIONS}
    diff["MEAN"] = {p: np.empty(B) for p in PAIRES}
    for b in range(B):
        ans = RNG.choice(ANS, size=len(ANS), replace=True)
        acc = {p: [] for p in PAIRES}
        for st in STATIONS:
            m = np.concatenate([np.flatnonzero(SER[st]["an"] == a) for a in ans])
            o = SER[st]["obs"][m]
            v = {k: f(SER[st][k][m], o) for k in ("best", "OPTW", "FINAL")}
            for p in PAIRES:
                diff[st][p][b] = v[p[0]] - v[p[1]]
                acc[p].append(v[p[0]] - v[p[1]])
        for p in PAIRES:
            diff["MEAN"][p][b] = float(np.nanmean(acc[p]))
    for cle in list(STATIONS) + ["MEAN"]:
        for p in PAIRES:
            # un tirage peut tomber sur un jeu d'années où le KGE d'une station
            # est indéfini (variance observée nulle) : on l'écarte au lieu de
            # propager un NaN dans tout l'intervalle
            e = diff[cle][p]
            e = e[np.isfinite(e)]
            lignes.append(dict(
                n_valide=len(e),
                critere=nom_c, station=cle, comparaison=f"{p[0]} - {p[1]}",
                observe=(round(obsv[cle][p[0]] - obsv[cle][p[1]], 4)
                         if cle != "MEAN" else
                         round(float(np.mean([obsv[s][p[0]] - obsv[s][p[1]]
                                              for s in STATIONS])), 4)),
                moyenne=round(float(e.mean()), 4),
                ic_bas=round(float(np.percentile(e, 2.5)), 4),
                ic_haut=round(float(np.percentile(e, 97.5)), 4),
                P_positif=round(float(np.mean(e > 0)), 3)))
    print(f"critère {nom_c} : fait", flush=True)

R = pd.DataFrame(lignes)
R.to_csv(TAB / "significance_bootstrap.csv", index=False)

print("\n=== moyenne des 7 stations, bootstrap par blocs d'années (B=2000) ===")
print(f"{'critère':8s} {'comparaison':16s} {'observé':>9s} {'IC 95 %':>22s} {'P(diff>0)':>8s}")
for _, r in R[R.station == "MEAN"].iterrows():
    print(f"{r.critere:8s} {r.comparaison:16s} {r.observe:+9.4f} "
          f"[{r.ic_bas:+.4f}, {r.ic_haut:+.4f}] {r.P_positif:8.2f}")

print("\n=== station par station, FINAL - meilleur membre ===")
print(f"{'station':11s} {'KGE':>9s} {'IC 95 %':>22s} {'P':>5s}   "
      f"{'NSE':>9s} {'IC 95 %':>22s} {'P':>5s}")
for st in STATIONS:
    k = R[(R.critere == "KGE") & (R.station == st) &
          (R.comparaison == "FINAL - best")].iloc[0]
    n = R[(R.critere == "NSE") & (R.station == st) &
          (R.comparaison == "FINAL - best")].iloc[0]
    print(f"{st:11s} {k.observe:+9.4f} [{k.ic_bas:+.4f}, {k.ic_haut:+.4f}] "
          f"{k.P_positif:5.2f}   {n.observe:+9.4f} "
          f"[{n.ic_bas:+.4f}, {n.ic_haut:+.4f}] {n.P_positif:5.2f}")
print("\nécrit : tables/significance_bootstrap.csv")
