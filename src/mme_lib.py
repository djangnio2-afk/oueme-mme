#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared library: metrics, convex weighting, quantile mapping, data access.

Imported by every script. It also resolves the folders of the package from its
own location, so the scripts run from wherever they are called. No scipy
dependency.
"""
from __future__ import annotations
from pathlib import Path
import os
import sys
import itertools
import numpy as np
import pandas as pd

# une console qui n'est pas en UTF-8 ne doit pas arreter un calcul
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # racine du paquet
DATA = ROOT / "data"                                # les trois membres, le reseau
TAB  = ROOT / "results" / "tables"                  # les tableaux de l'article
SERIES = ROOT / "results" / "series"                # les series journalieres
# le dossier de sortie des figures se detourne par MME_FIG_DIR
FIG = (Path(os.environ["MME_FIG_DIR"]) if os.environ.get("MME_FIG_DIR")
       else ROOT / "results" / "figures")
for _d in (TAB, SERIES, FIG):
    _d.mkdir(parents=True, exist_ok=True)

MEMBERS  = ["GR4J", "HECHMS", "LISFLOOD"]
STATIONS = ["BETEROU", "SAVE", "KABOUA", "ATCHERIGBE", "DOME", "AHLAN", "BONOU"]
# TRAIN = fenêtre NOMINALE : HMS/LISFLOOD y sont valides dès 1995 (QM, BMA,
# diagnostics), mais GR4J ne démarre qu'en 1997 -> le fit des poids à 3 membres
# est DE FACTO 1997-2005 (masquage NaN de kge_val). Expérience 05/08/2026
# (GR4J prolongé à 1995, poids re-figés fenêtre pleine) : KGE test moyen
# 0.833 -> 0.831 — aucun gain, config conservée..
TRAIN = ("1995-01-01", "2005-12-31")
TEST  = ("2006-01-01", "2016-12-31")
RNG = np.random.default_rng(7)

# ---------------- metrics ----------------
def kge_val(sim, obs):
    m = ~(np.isnan(sim) | np.isnan(obs)); s, o = sim[m], obs[m]
    if s.size < 30 or o.std() == 0 or o.mean() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r-1)**2 + (s.mean()/o.mean()-1)**2 +
                       ((s.std()/s.mean())/(o.std()/o.mean())-1)**2)

def metrics(sim, obs, dates):
    sim = np.asarray(sim, float); obs = np.asarray(obs, float)
    r = np.corrcoef(sim, obs)[0, 1]
    beta = sim.mean()/obs.mean(); gamma = (sim.std()/sim.mean())/(obs.std()/obs.mean())
    K = 1 - np.sqrt((r-1)**2+(beta-1)**2+(gamma-1)**2)
    NSE = 1 - np.sum((sim-obs)**2)/np.sum((obs-obs.mean())**2)
    PB = 100*np.sum(sim-obs)/np.sum(obs)
    thr = np.percentile(obs, 98); h = obs >= thr
    FHV = 100*(sim[h].sum()-obs[h].sum())/obs[h].sum()
    df = pd.DataFrame({"s": sim, "o": obs}, index=pd.to_datetime(dates))
    g = df.groupby(df.index.year).max(); PKB = 100*((g.s-g.o)/g.o).mean()
    return dict(KGE=K, r=r, beta=beta, gamma=gamma, NSE=NSE, PBIAS=PB, FHV=FHV, peakbias=PKB)

# ---------------- convex combiners ----------------
def _simplex_grid(K, N):
    pts = []
    for c in itertools.combinations(range(N+K-1), K-1):
        prev, parts = -1, []
        for x in c:
            parts.append(x-prev-1); prev = x
        parts.append(N+K-1-prev-1)
        pts.append([p/N for p in parts])
    return np.array(pts)

def fit_optw(X, y, N=50, refine=True, score=None):
    """convex weights (w>=0, sum=1) maximising `score(pred,y)` (default KGE)."""
    score = score or kge_val
    G = _simplex_grid(X.shape[1], N)
    sc = np.array([score(X@w, y) for w in G])
    w = G[np.nanargmax(sc)]
    if refine:
        F = _simplex_grid(X.shape[1], 4*N)
        sel = F[np.all((F >= w-1.0/N-1e-9) & (F <= w+1.0/N+1e-9), axis=1)]
        if len(sel):
            sc2 = np.array([score(X@ww, y) for ww in sel])
            if np.nanmax(sc2) >= np.nanmax(sc):
                w = sel[np.nanargmax(sc2)]
    return w

def fit_bma(X, y, iters=300, tol=1e-7):
    n, K = X.shape
    w = np.full(K, 1/K)
    s2 = np.full(K, np.var(y - X.mean(axis=1)) + 1e-6)
    ll_old = -np.inf
    for _ in range(iters):
        d = (y[:, None] - X)
        g = np.exp(-0.5*d**2/s2) / np.sqrt(2*np.pi*s2)
        num = w*g; den = num.sum(axis=1, keepdims=True) + 1e-300
        z = num/den
        w = z.mean(axis=0)
        s2 = np.maximum((z*d**2).sum(axis=0)/(z.sum(axis=0)+1e-12), 1e-6)
        ll = np.log(den).sum()
        if abs(ll-ll_old) < tol*abs(ll_old+1e-9):
            break
        ll_old = ll
    return w/w.sum(), s2

def qm_fit(sim_tr, obs_tr):
    """empirical quantile-mapping transfer (sorted sim -> sorted obs)."""
    return np.sort(sim_tr), np.sort(obs_tr)

def qm_apply(sim, ss, oo):
    """apply QM; multiplicative extrapolation above training range (restores extremes)."""
    out = np.interp(sim, ss, oo)
    hi = sim > ss[-1]
    if hi.any() and ss[-1] > 0:
        out[hi] = oo[-1] * (sim[hi] / ss[-1])
    return out

def bma_quantile(X, w, s2, q, ndraw=600):
    """q-th percentile of the BMA predictive mixture per row (discharge >= 0)."""
    n, K = X.shape
    comp = RNG.choice(K, size=(n, ndraw), p=w)
    means = X[np.arange(n)[:, None], comp]
    draws = means + RNG.normal(size=(n, ndraw)) * np.sqrt(s2[comp])
    return np.maximum(np.percentile(draws, q, axis=1), 0.0)

def bma_interval(X, w, s2, lo=5, hi=95, ndraw=2000):   # 400 tirages sous-estimaient les quantiles 5/95 (bande trop etroite)
    n, K = X.shape
    comp = RNG.choice(K, size=(n, ndraw), p=w)
    means = X[np.arange(n)[:, None], comp]
    draws = means + RNG.normal(size=(n, ndraw))*np.sqrt(s2[comp])
    qlo = np.maximum(np.percentile(draws, lo, axis=1), 0.0)   # discharge >= 0 (physical)
    qhi = np.maximum(np.percentile(draws, hi, axis=1), 0.0)
    return qlo, qhi

# ---------------- configuration V3 : critère, post-traitement, cascade -------
# Partagées par finalize_v3.py, qui les AJUSTE sur le calage, et par
# mme_apply_extended.py, qui les APPLIQUE au continu. Une seule implémentation :
# dupliquées, les deux divergeraient silencieusement et le produit continu
# cesserait de reproduire le produit publié sans que rien ne le signale.
def nse_val(sim, obs):
    m = ~(np.isnan(sim) | np.isnan(obs)); s, o = sim[m], obs[m]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)

def score_composite(sim, obs):
    """S = 1/2 KGE + 1/2 NSE — le critère qui décide de tout dans le projet."""
    return 0.5 * kge_val(sim, obs) + 0.5 * nse_val(sim, obs)

def qm_full(sim_tr, obs_tr):
    """Transfert quantile a quantile complet, sur tous les jours de calage."""
    return np.sort(sim_tr), np.sort(obs_tr), None

def qm_lisse(sim_tr, obs_tr, n=20):
    """Meme transfert, resume a n quantiles : moins ajuste au bruit du calage."""
    p = np.linspace(0, 100, n)
    return np.percentile(sim_tr, p), np.percentile(obs_tr, p), None

QM_VARIANTES = {"QM_FULL": qm_full, "QM_LISSE": qm_lisse}

def qm_apply2(sim, ss, oo, seuil=None):
    """Applique le transfert. Au-dela du domaine appris, prolongement ADDITIF :
    une interpolation simple ecreterait les extremes, qui sont precisement ce
    qui commande l'hydraulique."""
    sim = np.asarray(sim, float)
    out = np.interp(sim, ss, oo, left=oo[0], right=oo[-1])
    hors = sim > ss[-1]
    out[hors] = oo[-1] + (sim[hors] - ss[-1])
    if seuil is None:
        return out
    res = sim.copy()
    res[sim > seuil] = out[sim > seuil]
    return res

def transfert(delta, L, k, alpha):
    """Transport vers l'aval de la masse retiree en amont : retard de L jours,
    laminage par reservoir lineaire de constante k, gain alpha."""
    delta = np.asarray(delta, float)
    d = np.zeros_like(delta)
    if L > 0:
        d[L:] = delta[:-L]
    else:
        d = delta.copy()
    out, sto = np.zeros_like(d), 0.0
    for i in range(len(d)):
        rel = sto / k
        out[i] = rel
        sto += d[i] - rel
    return alpha * out

def melange_regime(X, w, regime=None):
    """Melange convexe, commute entre deux vecteurs si la station porte un
    regime. `regime` vaut None ou (seuil, w_haut, w_bas)."""
    if regime is None:
        return X @ w
    seuil, wh, wl = regime
    return np.where(X.mean(axis=1) >= seuil, X @ np.asarray(wh, float),
                    X @ np.asarray(wl, float))

# ---------------- data access ----------------
def load_long():
    return pd.read_csv(DATA / "member_simulations_daily.csv", parse_dates=["date"])

# Où vivent les trois membres de production. UNE seule déclaration : harmonize_v3
# les assemble, mme_apply_extended les applique, et les deux doivent lire
# exactement les mêmes fichiers — sinon le produit continu et le produit publié
# décrivent deux ensembles différents sans que rien ne le signale.
GR_COL = {"BETEROU": "Beterou", "SAVE": "Save", "KABOUA": "Kaboua",
          "ATCHERIGBE": "Atcherigbe", "DOME": "Dome", "AHLAN": "Ahlan",
          "BONOU": "Bonou"}
LIS_ORDER = ["BETEROU", "KABOUA", "SAVE", "ATCHERIGBE", "AHLAN", "DOME", "BONOU"]

def load_members(dropna=True):
    """NOT AVAILABLE IN THIS DEPOSIT — reads the raw model outputs.

    This helper rebuilds the harmonized dataset from the native GR4J,
    HEC-HMS and LISFLOOD outputs, which are not deposited. Use
    load_long(), which reads data/member_simulations_daily.csv.

    Les trois membres étendus, par station, sur leur calendrier commun.

    GR4J     : qsim_extended.csv, sortie d'airGRiwrm + Domè corrigé (config C1)
    HEC-HMS  : qsim_extended_hms_v7.csv, membre natif v7 (multiplicateurs hybrides)
    LISFLOOD : fin3_extended_dis.tss, calage fin3 rejoué depuis 1992
    """
    gr = pd.read_csv(ROOT / "02_hydrologie/GR4J/outputs/chirps/qsim_extended.csv",
                     parse_dates=["Date"]).set_index("Date")
    hms = pd.read_csv(ROOT / "02_hydrologie/HEC-HMS/Modelling_V7/results"
                      / "qsim_extended_hms_v7.csv", parse_dates=["Date"]).set_index("Date")
    tss = ROOT / "01_data" / "lisflood" / "fin3_extended_dis.tss"
    lines = tss.read_text().splitlines()
    n_cols = int(lines[1].split()[0])
    arr = np.array([list(map(float, ln.split()))
                    for ln in lines[2 + n_cols:] if ln.strip()])
    idx = pd.date_range("1992-01-01", periods=len(arr), freq="D")   # fin3 démarre en 1992
    lis = pd.DataFrame(arr[:, 1:], index=idx, columns=LIS_ORDER)
    out = {}
    for st in STATIONS:
        d = pd.DataFrame({"GR4J": gr[GR_COL[st]], "HECHMS": hms[st], "LISFLOOD": lis[st]})
        out[st] = d.dropna() if dropna else d
    return out

def station_split(long, st):
    d = long[long.station == st].dropna(subset=["obs"]+MEMBERS).set_index("date").sort_index()
    tr = d.loc[TRAIN[0]:TRAIN[1]]; te = d.loc[TEST[0]:TEST[1]]
    return (tr[MEMBERS].to_numpy(), tr["obs"].to_numpy(),
            te[MEMBERS].to_numpy(), te["obs"].to_numpy(), te.index, tr, te)

def best_member(tr_X, tr_y):
    ks = [kge_val(tr_X[:, i], tr_y) for i in range(tr_X.shape[1])]
    return int(np.nanargmax(ks)), ks
