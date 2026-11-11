# -*- coding: utf-8 -*-
"""The weight vectors of the four weighting schemes, on the production basis.

Equal weights, skill-proportional weights, the optimized convex vector and the
Bayesian mixture, recomputed with the exact logic of the delivered fit: same
functions, same composite criterion, same calibration window, same blocks. The
run checks that the vector of the retained scheme reproduces
retained_weights.csv to within 0.001; if it does, the other three belong to the
same campaign.

Reads  data/member_simulations_daily.csv
       results/tables/retained_weights.csv
Writes results/tables/weights_by_scheme.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from mme_lib import ROOT, DATA, TAB, SERIES, FIG
from mme_lib import STATIONS, kge_val, fit_optw, fit_bma  # noqa: E402

MEM = ["GR4J", "HECHMS", "LISFLOOD"]
CAL = ("1997-01-01", "2005-12-31")
BLOCS = [("1997", "1999"), ("2000", "2002"), ("2003", "2005")]
LAMB = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
EGAL = np.ones(3) / 3


def nse(s, o):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def S(p, o):                                   # critère composite du projet
    return 0.5 * kge_val(p, o) + 0.5 * nse(p, o)


def msk(i, a, b):
    return (i >= a) & (i <= b + "-12-31")


def poids(X, y, idx, sch):
    if sch == "OPTW":
        return fit_optw(X, y, score=S)
    if sch == "BAG":
        ws = [fit_optw(X[msk(idx, a, b)], y[msk(idx, a, b)], score=S)
              for a, b in BLOCS if msk(idx, a, b).sum() > 100]
        w = np.mean(ws, axis=0)
        return w / w.sum()
    # La contraction se regle comme dans le banc : son intensite est estimee
    # HORS ECHANTILLON, par re-echantillonnage en blocs d'annees. Estimee sur les
    # jours qui ont ajuste les poids elle sortirait a zero, et le panneau de la
    # figure 10 reproduirait a l'identique celui des poids optimises alors que le
    # banc lui attribue un score distinct.
    best, bl = -9, 0.0
    for lam in LAMB:
        ks = []
        for a_, b_ in BLOCS:
            m = msk(idx, a_, b_)
            if m.sum() < 100 or (~m).sum() < 300:
                continue
            wl = (1 - lam) * fit_optw(X[~m], y[~m], score=S) + lam * EGAL
            ks.append(S(X[m] @ wl, y[m]))
        v = float(np.nanmean(ks)) if ks else -9
        if v > best:
            best, bl = v, lam
    return (1 - bl) * fit_optw(X, y, score=S) + bl * EGAL


V = pd.read_csv(DATA / "member_simulations_daily.csv", parse_dates=["date"])
REF = pd.read_csv(TAB / "retained_weights.csv").set_index("station")

lignes, ecart_max = [], 0.0
for st in STATIONS:
    d = V[V.station == st].set_index("date").sort_index().dropna(subset=["obs"])
    tr = d.loc[CAL[0]:CAL[1]]
    X, y, idx = tr[MEM].to_numpy(), tr["obs"].to_numpy(), tr.index
    w = {sch: poids(X, y, idx, sch) for sch in ("OPTW", "BAG", "SHRINK")}
    w["BMA"], _ = fit_bma(X, y)

    retenu = REF.loc[st, "schema"]
    # Le vecteur RETENU porte la regle de parcimonie du fit de production : une
    # part inferieure a TAU est mise a zero et les membres gardes sont reajustes.
    # Les trois autres schemas restent bruts : ils sont ici pour comparaison, et
    # leur appliquer la regle a l'un seul fausserait la comparaison.
    TAU = 0.10
    wr = w[retenu]
    if 0 < min(wr[wr > 0]) < TAU:
        garde = wr >= TAU
        if garde.sum() == 1:
            wr = garde.astype(float)
        else:
            wg = poids(X[:, garde], y, idx, retenu)
            wr = np.zeros(3)
            wr[garde] = wg
        w[retenu] = wr

    attendu = REF.loc[st, ["w_GR4J", "w_HECHMS", "w_LISFLOOD"]].to_numpy(float)
    e = float(np.max(np.abs(w[retenu] - attendu)))
    ecart_max = max(ecart_max, e)
    print(f"{st:12s} schéma retenu {retenu:7s} écart max au T2 : {e:.4f}")

    for sch in ("OPTW", "BAG", "SHRINK", "BMA"):
        lignes.append(dict(station=st, schema=sch, retenu=(sch == retenu),
                           **{f"w_{m}": round(float(v), 4)
                              for m, v in zip(MEM, w[sch])}))

out = TAB / "weights_by_scheme.csv"
pd.DataFrame(lignes).to_csv(out, index=False)
print(f"\nécart maximal au vecteur publié : {ecart_max:.4f}")
if ecart_max > 0.001:
    print("VÉRIFICATION ÉCHOUÉE")
    raise SystemExit(1)          # ne pas laisser passer un T30 desaccorde du T2
print("VÉRIFICATION OK")
print(f"écrit : {out}")
