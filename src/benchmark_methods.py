# -*- coding: utf-8 -*-
"""The thirty-two combination configurations, under one protocol.

The graduated series of the paper (best single member, simple mean, skill
weighting, constrained regression, optimized convex weights, Bayesian model
averaging, convex weights plus quantile mapping), extended to the families of
the literature that series leaves out:

  static weightings     Bates-Granger (inverse variance), Akaike weights,
                        Bayesian information-criterion weights, convex
                        non-negative least squares, median, geometric mean
  regularized regr.     ridge (L2), lasso (L1), elastic net
  super-ensembles       Krishnamurti (regression on anomalies), modified
                        super-ensemble (debiased members, Ajami)
  transformed spaces    convex weights in square-root and in log space
  stabilization         block bagging, shrinkage, Kalman (recursive weights,
                        final state frozen)
  conditional           high and low regime, monthly switching
  probabilistic         Bayesian model averaging, EMOS/NGR, QRA, MCP (Todini)
  statistical learning  random forest, gradient boosting, neural network, SVR

The protocol is identical for all of them: learning on the 1997-2005
calibration window only, no test observation used at any point, including by
the dynamic methods, whose weights are frozen at the end of calibration; then
a single pass over the 2006-2016 test period.

Reads  data/member_simulations_daily.csv
Writes results/tables/method_benchmark.csv, method_benchmark_summary.csv
       results/series/benchmark_test.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from mme_lib import ROOT, DATA, TAB, SERIES, FIG
from mme_lib import STATIONS, kge_val, fit_optw, fit_bma

MEM = ["GR4J", "HECHMS", "LISFLOOD"]
CAL, VAL = ("1997-01-01", "2005-12-31"), ("2006-01-01", "2016-12-30")
BLOCS = [("1997", "1999"), ("2000", "2002"), ("2003", "2005")]
EGAL = np.ones(3) / 3
LAMB = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
FAMILLE = {}


def nse(s, o):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def S(p, o):
    return 0.5 * kge_val(p, o) + 0.5 * nse(p, o)


def pbias(s, o):
    return 100 * (s.sum() - o.sum()) / o.sum()


def fhv(s, o, q=98):
    h = o >= np.percentile(o, q)
    return 100 * (s[h].sum() - o[h].sum()) / o[h].sum()


def msk(i, a, b):
    return (i >= a) & (i <= b + "-12-31")


def qm_pair(s, o, n=20):
    p = np.linspace(0, 100, n)
    return np.percentile(s, p), np.percentile(o, p)


def qm_app(sim, ss, oo):
    out = np.interp(sim, ss, oo, left=oo[0], right=oo[-1])
    h = sim > ss[-1]
    out[h] = oo[-1] + (sim[h] - ss[-1])
    return out


def methodes(Xtr, ytr, itr, Xte, ite):
    R, F = {}, {}

    def add(nom, fam, ptr, pte):
        R[nom] = (np.asarray(ptr, float), np.asarray(pte, float))
        F[nom] = fam

    # ---------- série graduée du manuscrit ----------
    kg = np.array([kge_val(Xtr[:, i], ytr) for i in range(3)])
    # le membre de référence est choisi sur le critère composite S, comme toute
    # autre décision du travail (à Atchérigbé le KGE seul renverse le choix)
    sc = np.array([S(Xtr[:, i], ytr) for i in range(3)])
    add("B0_meilleur_membre", "reference", Xtr[:, sc.argmax()], Xte[:, sc.argmax()])
    add("B1_moyenne_simple", "moyenne_non_ponderee", Xtr @ EGAL, Xte @ EGAL)
    w = np.clip(kg, 0, None)
    w /= w.sum()
    add("B2_pondere_KGE", "ponderation_statique", Xtr @ w, Xte @ w)
    A = np.column_stack([Xtr, np.ones(len(Xtr))])
    B = np.column_stack([Xte, np.ones(len(Xte))])
    b = np.linalg.lstsq(A, ytr, rcond=None)[0]
    add("GR_C_regression", "regression_libre", A @ b, B @ b)
    w_opt = fit_optw(Xtr, ytr, score=S)
    add("OPTW", "ponderation_convexe", Xtr @ w_opt, Xte @ w_opt)
    wb, s2 = fit_bma(Xtr, ytr)
    add("BMA_moyenne", "probabiliste", Xtr @ wb, Xte @ wb)
    ss, oo = qm_pair(Xtr @ w_opt, ytr)
    add("OPTW_QM_partout", "post_traitement", qm_app(Xtr @ w_opt, ss, oo),
        qm_app(Xte @ w_opt, ss, oo))

    # ---------- pondérations statiques classiques ----------
    var = np.array([np.mean((Xtr[:, i] - ytr) ** 2) for i in range(3)])
    w = (1 / var) / np.sum(1 / var)
    add("Bates_Granger", "ponderation_statique", Xtr @ w, Xte @ w)
    n = len(ytr)
    for nom, pen in (("Poids_Akaike", 2 * 4), ("Poids_bayesiens_BIC", np.log(n) * 4)):
        ic = n * np.log(var) + pen
        dl = ic - ic.min()
        w = np.exp(-dl / 2) / np.exp(-dl / 2).sum()
        add(nom, "ponderation_statique", Xtr @ w, Xte @ w)
    try:
        from scipy.optimize import nnls
        wn, _ = nnls(Xtr, ytr)
        wn = wn / wn.sum() if wn.sum() > 0 else EGAL
    except Exception:
        wn = EGAL
    add("GR_convexe_NNLS", "ponderation_convexe", Xtr @ wn, Xte @ wn)
    add("Mediane_membres", "moyenne_non_ponderee", np.median(Xtr, axis=1), np.median(Xte, axis=1))
    gtr = np.exp(np.mean(np.log(np.clip(Xtr, 1e-3, None)), axis=1))
    gte = np.exp(np.mean(np.log(np.clip(Xte, 1e-3, None)), axis=1))
    add("Moyenne_geometrique", "moyenne_non_ponderee", gtr, gte)

    # ---------- régressions régularisées ----------
    # variables centrées-réduites + pénalité choisie par CV interne au calage,
    # sinon la pénalité est négligeable devant l'échelle des débits et les trois
    # méthodes retombent exactement sur GR-C.
    try:
        from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        alphas = np.logspace(-2, 4, 25)
        for nom, mod in (("Ridge_L2", RidgeCV(alphas=alphas)),
                         ("Lasso_L1", LassoCV(alphas=alphas, cv=3, max_iter=20000,
                                              random_state=7)),
                         ("Elastic_net", ElasticNetCV(alphas=alphas, cv=3,
                                                      l1_ratio=[.2, .5, .8],
                                                      max_iter=20000, random_state=7))):
            pipe = make_pipeline(StandardScaler(), mod)
            pipe.fit(Xtr, ytr)
            add(nom, "regression_regularisee", pipe.predict(Xtr), pipe.predict(Xte))
    except Exception:
        pass

    # ---------- super-ensembles ----------
    mo, mx = ytr.mean(), Xtr.mean(axis=0)
    Atr, Ate = Xtr - mx, Xte - mx
    bk = np.linalg.lstsq(Atr, ytr - mo, rcond=None)[0]
    add("Superensemble_Krishnamurti", "regression_libre", mo + Atr @ bk, mo + Ate @ bk)
    Xb_tr = Xtr - (mx - mo)          # membres débiaisés (Ajami 2006)
    Xb_te = Xte - (mx - mo)
    bm = np.linalg.lstsq(np.column_stack([Xb_tr, np.ones(n)]), ytr, rcond=None)[0]
    add("Superensemble_modifie", "regression_libre",
        np.column_stack([Xb_tr, np.ones(n)]) @ bm,
        np.column_stack([Xb_te, np.ones(len(Xte))]) @ bm)

    # ---------- espaces transformés ----------
    Xs, ys = np.sqrt(np.clip(Xtr, 0, None)), np.sqrt(np.clip(ytr, 0, None))
    ws = fit_optw(Xs, ys, score=S)
    add("OPTW_espace_racine", "ponderation_convexe", (Xs @ ws) ** 2,
        (np.sqrt(np.clip(Xte, 0, None)) @ ws) ** 2)
    Xl, yl = np.log1p(np.clip(Xtr, 0, None)), np.log1p(np.clip(ytr, 0, None))
    wl = fit_optw(Xl, yl, score=S)
    add("OPTW_espace_log", "ponderation_convexe", np.expm1(Xl @ wl),
        np.expm1(np.log1p(np.clip(Xte, 0, None)) @ wl))

    # ---------- stabilisation ----------
    ws_ = [fit_optw(Xtr[msk(itr, a, b)], ytr[msk(itr, a, b)], score=S)
           for a, b in BLOCS if msk(itr, a, b).sum() > 100]
    wbag = np.mean(ws_, axis=0)
    wbag /= wbag.sum()
    add("BAG_blocs", "convexe_stabilisee", Xtr @ wbag, Xte @ wbag)
    # La contraction se règle comme le veut sa propre littérature : l'intensité
    # est estimée hors échantillon, par ré-échantillonnage en blocs d'années à
    # l'intérieur du calage. C'est constitutif de la méthode et non une étape de
    # sélection ajoutée : une intensité estimée sur les jours qui ont servi à
    # ajuster les poids sort toujours à zéro, ce qui reviendrait à supprimer la
    # méthode du champ comparé.
    best, bl = -9, 0.0
    for lam in LAMB:
        ks = []
        for a, b in BLOCS:
            m = msk(itr, a, b)
            if m.sum() < 100 or (~m).sum() < 300:
                continue
            wlam = (1 - lam) * fit_optw(Xtr[~m], ytr[~m], score=S) + lam * EGAL
            ks.append(S(Xtr[m] @ wlam, ytr[m]))
        v = float(np.nanmean(ks)) if ks else -9
        if v > best:
            best, bl = v, lam
    wsh = (1 - bl) * w_opt + bl * EGAL
    add("SHRINK_contraction", "convexe_stabilisee", Xtr @ wsh, Xte @ wsh)
    # Kalman : poids récursifs sur le calage, ÉTAT FINAL FIGÉ pour la validation
    wk = EGAL.copy()
    P = np.eye(3) * 0.1
    Q, Rk = np.eye(3) * 1e-6, float(np.var(ytr)) * 0.5
    pk = np.empty(n)
    for t in range(n):
        h = Xtr[t]
        pk[t] = h @ wk
        P = P + Q
        K = P @ h / (h @ P @ h + Rk)
        wk = wk + K * (ytr[t] - pk[t])
        P = P - np.outer(K, h) @ P
    add("Kalman_poids_recursifs", "poids_dynamiques", pk, Xte @ wk)

    # ---------- conditionnelles ----------
    cons_tr, cons_te = Xtr.mean(axis=1), Xte.mean(axis=1)
    seuil = np.percentile(cons_tr, 75)
    hi_tr, hi_te = cons_tr > seuil, cons_te > seuil
    w_hi = fit_optw(Xtr[hi_tr], ytr[hi_tr], score=S)
    w_lo = fit_optw(Xtr[~hi_tr], ytr[~hi_tr], score=S)
    add("Regime_haut_bas", "poids_dynamiques", np.where(hi_tr, Xtr @ w_hi, Xtr @ w_lo),
        np.where(hi_te, Xte @ w_hi, Xte @ w_lo))
    ptr, pte = np.empty(n), np.empty(len(Xte))
    for mois in range(1, 13):
        a = itr.month == mois
        bsel = ite.month == mois
        if a.sum() < 30:
            ptr[a] = Xtr[a] @ EGAL
            pte[bsel] = Xte[bsel] @ EGAL
            continue
        j = int(np.argmax([kge_val(Xtr[a, i], ytr[a]) for i in range(3)]))
        ptr[a], pte[bsel] = Xtr[a, j], Xte[bsel, j]
    add("Commutation_mensuelle", "poids_dynamiques", ptr, pte)

    # ---------- probabilistes ----------
    # regression de la MEDIANE (perte L1) sur les membres : c'est une
    # regression contrainte par la perte, pas un post-traitement. Aucune
    # exception n'est avalee ici : une methode absente fausserait le compte
    # annonce dans l'article.
    import statsmodels.api as sm
    q = sm.QuantReg(ytr, sm.add_constant(Xtr)).fit(q=0.5).params
    add("QRA_mediane", "regression_contrainte", sm.add_constant(Xtr) @ q,
        sm.add_constant(Xte, has_constant="add") @ q)
    # EMOS / NGR : moyenne = régression à coefficients positifs (moments 1 et 2)
    try:
        from scipy.optimize import nnls
        wg, _ = nnls(np.column_stack([Xtr, np.ones(n)]), ytr)
        # moindres carres a coefficients POSITIFS avec ordonnee a l'origine :
        # une regression contrainte, et non une correction de distribution
        add("EMOS_NGR", "regression_contrainte",
            np.column_stack([Xtr, np.ones(n)]) @ wg,
            np.column_stack([Xte, np.ones(len(Xte))]) @ wg)
    except Exception:
        pass
    try:
        from scipy.stats import norm

        def nqt(v):
            vs = np.sort(v)
            eta = norm.ppf((np.arange(1, len(vs) + 1) - 0.5) / len(vs))
            return (lambda x: np.interp(x, vs, eta)), (lambda e: np.interp(e, eta, vs))
        fw = [nqt(Xtr[:, i])[0] for i in range(3)]
        fo_, io_ = nqt(ytr)
        Etr = np.column_stack([fw[i](Xtr[:, i]) for i in range(3)] + [np.ones(n)])
        Ete = np.column_stack([fw[i](Xte[:, i]) for i in range(3)] + [np.ones(len(Xte))])
        bb = np.linalg.lstsq(Etr, fo_(ytr), rcond=None)[0]
        # regression en espace normal-quantile puis retour : contrainte par la
        # transformation, elle ne corrige aucune distribution a posteriori
        add("MCP_Todini", "regression_contrainte", io_(Etr @ bb), io_(Ete @ bb))
    except Exception:
        pass

    # ---------- apprentissage statistique ----------
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        for nom, mod in (("Foret_aleatoire", RandomForestRegressor(
                              n_estimators=300, min_samples_leaf=5,
                              random_state=7, n_jobs=-1)),
                         ("Gradient_boosting", GradientBoostingRegressor(
                              n_estimators=300, max_depth=3, learning_rate=0.05,
                              subsample=0.8, random_state=7))):
            mod.fit(Xtr, ytr)
            add(nom, "apprentissage_statistique", mod.predict(Xtr), mod.predict(Xte))
        sc = StandardScaler().fit(Xtr)
        mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=1500, random_state=7)
        mlp.fit(sc.transform(Xtr), ytr)
        add("Reseau_neurones", "apprentissage_statistique", mlp.predict(sc.transform(Xtr)),
            mlp.predict(sc.transform(Xte)))
        sy = ytr.std()
        sv = SVR(C=10.0, epsilon=0.1, gamma="scale")
        sv.fit(sc.transform(Xtr), ytr / sy)
        add("SVR", "apprentissage_statistique", sv.predict(sc.transform(Xtr)) * sy,
            sv.predict(sc.transform(Xte)) * sy)
    except Exception as e:
        print("  (sklearn :", str(e)[:60], ")")
    FAMILLE.update(F)
    return R


def peakbias(s, o, idx):
    """biais moyen sur les maxima annuels de saison de crue (juin-décembre)"""
    g = pd.DataFrame({"s": s, "o": o}, index=idx)
    g = g[(g.index.month >= 6) & (g.index.month <= 12)].groupby(
        lambda t: t.year).max()
    g = g[g.o > 0]
    return float(np.mean((g.s - g.o) / g.o)) * 100 if len(g) else np.nan


V = pd.read_csv(DATA / "member_simulations_daily.csv", parse_dates=["date"])
lignes, series = [], []
for st in STATIONS:
    d = V[V.station == st].set_index("date").sort_index().dropna(subset=["obs"])
    tr, te = d.loc[CAL[0]:CAL[1]], d.loc[VAL[0]:VAL[1]]
    Xtr, ytr, itr = tr[MEM].to_numpy(), tr["obs"].to_numpy(), tr.index
    Xte, yte, ite = te[MEM].to_numpy(), te["obs"].to_numpy(), te.index
    for nom, (ptr, pte) in methodes(Xtr, ytr, itr, Xte, ite).items():
        lignes.append(dict(station=st, methode=nom, famille=FAMILLE.get(nom, ""),
                           KGE_cal=round(kge_val(ptr, ytr), 3),
                           NSE_cal=round(nse(ptr, ytr), 3),
                           KGE_val=round(kge_val(pte, yte), 3),
                           NSE_val=round(nse(pte, yte), 3),
                           PBIAS_val=round(pbias(pte, yte), 1),
                           FHV_val=round(fhv(pte, yte), 1),
                           # décomposition du KGE : indispensable au diagramme de
                           # Taylor et à la figure de décomposition, qui doivent
                           # désormais couvrir tout le banc et non les seuls
                           # OPTW/BMA du jeu de figures d'origine
                           r_val=round(float(np.corrcoef(pte, yte)[0, 1]), 3),
                           beta_val=round(float(pte.mean() / yte.mean()), 3),
                           gamma_val=round(float((pte.std() / pte.mean()) /
                                                 (yte.std() / yte.mean())), 3),
                           peakbias_val=round(peakbias(pte, yte, ite), 1)))
        series.append(pd.DataFrame({"date": ite, "station": st, "methode": nom,
                                    "sim": np.round(pte, 3)}))
    print(f"{st} fait", flush=True)

B = pd.DataFrame(lignes)
B.to_csv(TAB / "method_benchmark.csv", index=False)
pd.concat(series, ignore_index=True).to_csv(
    SERIES / "benchmark_test.csv", index=False)
G = B.groupby(["famille", "methode"])[["KGE_cal", "NSE_cal", "KGE_val", "NSE_val",
                                       "PBIAS_val", "FHV_val"]].mean().round(3)
G["surajust"] = (G.KGE_cal - G.KGE_val).round(3)
G = G.reset_index().sort_values("KGE_val", ascending=False)
G.to_csv(TAB / "method_benchmark_summary.csv", index=False)
print(f"\n=== BANC COMPARATIF — {B.methode.nunique()} méthodes, moyenne 7 stations ===")
print(f"{'méthode':28s} {'famille':15s} {'KGEcal':>7s} {'KGEval':>7s} "
      f"{'NSEval':>7s} {'PBIAS':>6s} {'FHV':>7s} {'surajust':>8s}")
for _, r in G.iterrows():
    print(f"{r.methode:28s} {r.famille:15s} {r.KGE_cal:7.3f} {r.KGE_val:7.3f} "
          f"{r.NSE_val:7.3f} {r.PBIAS_val:6.1f} {r.FHV_val:7.1f} {r.surajust:8.3f}")
print("\nécrits : tables/method_benchmark.csv + method_benchmark_summary.csv")
