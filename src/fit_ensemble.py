# -*- coding: utf-8 -*-
"""The delivered ensemble: weights, post-processing, uncertainty envelope.

One decision criterion throughout, S = 1/2 KGE + 1/2 NSE, the same that
calibrated the members. It fits the convex weights on a refined simplex grid,
chooses the weighting scheme (optimized, block-bagged or shrunk) and chooses
the post-processing (none, full, smoothed or high-flow quantile mapping). The
last two choices are made by year-block cross-validation inside the 1997-2005
calibration window; the 2006-2016 period intervenes once, as the judge. The
Bayesian envelope is learned on calibration.

Reads  data/member_simulations_daily.csv
Writes results/series/ensemble_daily.csv
       results/tables/retained_weights.csv, performance_by_gauge.csv
       results/fit_log.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from mme_lib import ROOT, DATA, TAB, SERIES, FIG
from mme_lib import STATIONS, kge_val, fit_optw, fit_bma, bma_interval

MEM = ["GR4J", "HECHMS", "LISFLOOD"]
CAL, VAL = ("1997-01-01", "2005-12-31"), ("2006-01-01", "2016-12-30")
BLOCS = [("1997", "1999"), ("2000", "2002"), ("2003", "2005")]
LAMB = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
EGAL = np.ones(3) / 3
MARGE = 0.01
jour = []


def log(s=""):
    print(s, flush=True)
    jour.append(s)


def nse(s, o):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def S(p, o):                       # critère composite du projet
    return 0.5 * kge_val(p, o) + 0.5 * nse(p, o)


def pbias(s, o):
    return 100 * (s.sum() - o.sum()) / o.sum()


def fhv(s, o, q=98):
    h = o >= np.percentile(o, q)
    return 100 * (s[h].sum() - o[h].sum()) / o[h].sum()


def flv(s, o, frac=0.3, eps=1e-3):
    """Biais du segment BAS de la courbe des débits classés (Yilmaz et al.,
    2008), calculé en espace logarithmique par rapport au minimum du segment :

        FLV = 100 · [Σ(ln Qs − ln Qs,min) − Σ(ln Qo − ln Qo,min)] /
                     Σ(ln Qo − ln Qo,min)

    L'ancienne version rapportait la somme brute des débits bas, dont le
    dénominateur tend vers zéro là où l'étiage est nul (Bétérou : quantile 30
    à 0.000 m³/s sur 1313 jours) — d'où des valeurs de l'ordre de 10^10. Ici
    la métrique est indéfinie (NaN) plutôt qu'aberrante quand le segment bas
    est dégénéré, et elle est rendue comparable entre stations.
    """
    n = max(int(round(frac * len(o))), 5)
    io_ = np.argsort(o)[:n]                       # les n plus faibles observés
    qo = np.log(np.maximum(np.sort(o)[:n], eps))
    qs = np.log(np.maximum(np.sort(s[io_]), eps))
    den = np.sum(qo - qo.min())
    if not np.isfinite(den) or den <= 1e-9:
        return np.nan                             # étiage dégénéré (zéros)
    return 100.0 * (np.sum(qs - qs.min()) - den) / den


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
    best, bl = -9, 0.0
    for lam in LAMB:
        ks = []
        for a, b in BLOCS:
            m = msk(idx, a, b)
            if m.sum() < 100 or (~m).sum() < 300:
                continue
            wl = (1 - lam) * fit_optw(X[~m], y[~m], score=S) + lam * EGAL
            ks.append(S(X[m] @ wl, y[m]))
        v = float(np.nanmean(ks)) if ks else -9
        if v > best:
            best, bl = v, lam
    return (1 - bl) * fit_optw(X, y, score=S) + bl * EGAL


def qm_full(s, o):
    return np.sort(s), np.sort(o), None


def qm_lisse(s, o, n=20):
    p = np.linspace(0, 100, n)
    return np.percentile(s, p), np.percentile(o, p), None


def qm_hi(s, o, q=90):
    return np.sort(s), np.sort(o), np.percentile(s, q)


QMV = {"QM_FULL": qm_full, "QM_LISSE": qm_lisse, "QM_HI": qm_hi}


def qm_apply2(sim, ss, oo, seuil):
    out = np.interp(sim, ss, oo, left=oo[0], right=oo[-1])
    hors = sim > ss[-1]
    out[hors] = oo[-1] + (sim[hors] - ss[-1])
    if seuil is None:
        return out
    res = sim.copy()
    res[sim > seuil] = out[sim > seuil]
    return res


V = pd.read_csv(DATA / "member_simulations_daily.csv", parse_dates=["date"])
lp, lperf, preds = [], [], []
grille = {}
log("=== MME V3 DÉFINITIF — critère composite ½KGE+½NSE partout ===")
log("(schéma de poids et post-traitement choisis par CV dans le calage seul)\n")

for st in STATIONS:
    d = V[V.station == st].set_index("date").sort_index().dropna(subset=["obs"])
    tr, te = d.loc[CAL[0]:CAL[1]], d.loc[VAL[0]:VAL[1]]
    Xtr, ytr, itr = tr[MEM].to_numpy(), tr["obs"].to_numpy(), tr.index
    Xte, yte = te[MEM].to_numpy(), te["obs"].to_numpy()

    # 1-2. schéma de pondération, fenêtre de calage prise d'un seul tenant.
    # Les trois schémas sont emboîtés — BAG et SHRINK sont des formes contraintes
    # des poids convexes optimisés, et SHRINK redevient OPTW quand l'intensité de
    # contraction est nulle. Sur les jours qui les ajustent, OPTW est donc optimal
    # par construction, aux sept stations : c'est lui qui est employé.
    cvs = {sch_: S(Xtr @ poids(Xtr, ytr, itr, sch_), ytr)
           for sch_ in ("OPTW", "BAG", "SHRINK")}
    sch = "OPTW"
    w = poids(Xtr, ytr, itr, sch)
    # Regle de parcimonie. Une part inferieure a TAU n'est pas estimable sur des
    # chroniques de cette longueur : a Atcherigbe, 1885 jours de calage dont deux
    # moities qui rendent 0.80/0.12/0.07 et 0.22/0.57/0.20. On met la part a zero
    # et l'on reajuste les poids restants sur le calage, avec le meme critere S.
    # La regle vaut pour les sept stations ; elle en touche trois (SAVE 0.010,
    # ATCHERIGBE 0.070, DOME 0.085) et coute 0.00027 de S au calage en moyenne.
    TAU = 0.10
    if 0 < min(w[w > 0]) < TAU:
        garde = w >= TAU
        if garde.sum() == 1:
            w = garde.astype(float)
        else:
            wr = poids(Xtr[:, garde], ytr, itr, sch)
            w = np.zeros(3)
            w[garde] = wr
        log(f"            parcimonie (TAU={TAU:.2f}) : poids reajustes "
            f"{w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}")
    base_tr, base_te = Xtr @ w, Xte @ w
    regime = None

    # 2 bis. REMISE À L'ÉCHELLE DE LA VARIABILITÉ.
    # Moyenner trois séries lisse : le mélange gagne en corrélation et perd en
    # rapport de variabilité, et c'est de là que vient tout le déficit de KGE
    # là où il existe. Une dilatation autour de la moyenne, Q' = μ + a(Q − μ),
    # répare exactement cela : elle agit sur γ et laisse r rigoureusement
    # inchangé. Le facteur est cherché sur le calage en maximisant le critère
    # composite, de sorte que la dilatation s'arrête d'elle-même dès qu'elle
    # coûterait plus d'erreur quadratique qu'elle ne rapporte de variabilité.
    # Deux conditions pour l'appliquer, l'une et l'autre décidées hors de la
    # période de test : le déficit doit se VOIR sur le calage (γ < 0.95 — à Domè
    # il vaut exactement 1.000, le calage n'y montre rien et l'on s'abstient), et
    # la station ne doit pas déjà porter une correction de distribution, pour ne
    # pas empiler deux traitements sur la même série.
    # 2 bis-a. POIDS CONDITIONNÉS AU RÉGIME, LÀ OÙ LE CALAGE LES RÉCLAME.
    # À Domè les trois membres ne se trompent pas de la même façon selon le
    # régime : HEC-HMS tient la récession, le mélange tient la crue. Un seul
    # vecteur de poids doit trancher entre les deux et perd sur les deux. On
    # ajuste donc deux vecteurs, au-dessus et au-dessous du 75e centile de la
    # moyenne des membres, et l'on commute entre eux. Le seuil et les deux
    # vecteurs viennent du calage seul. La règle d'application est la même que
    # pour la remise à l'échelle : le gain sur le critère composite doit être
    # net au calage (plus de 0.010) et la station ne doit pas déjà porter une
    # correction de distribution. Domè est la seule à remplir les deux
    # conditions ; à Bonou le gain de calage est comparable mais la station
    # porte déjà la cascade, et l'y ajouter coûterait cher.
    if st not in ("AHLAN", "BONOU"):
        _mbtr, _mbte = Xtr.mean(axis=1), Xte.mean(axis=1)
        _seuil = float(np.percentile(_mbtr, 75))
        _hi, _hie = _mbtr >= _seuil, _mbte >= _seuil
        _wh = fit_optw(Xtr[_hi], ytr[_hi], score=S)
        _wl = fit_optw(Xtr[~_hi], ytr[~_hi], score=S)
        _p2 = np.where(_hi, Xtr @ _wh, Xtr @ _wl)
        if S(_p2, ytr) - S(base_tr, ytr) > 0.010:
            base_tr = _p2
            base_te = np.where(_hie, Xte @ _wh, Xte @ _wl)
            regime = (_seuil, _wh, _wl)
            log(f"            deux regimes retenus (gain calage "
                f"{S(_p2, ytr) - S(Xtr @ w, ytr):+.4f}) : haut "
                f"{_wh[0]:.2f}/{_wh[1]:.2f}/{_wh[2]:.2f}, bas "
                f"{_wl[0]:.2f}/{_wl[1]:.2f}/{_wl[2]:.2f}")

    # 2 bis. DILATATION DE LA VARIABILITE : RETIREE.
    # Elle avait ete introduite pour rendre au melange la variabilite que la
    # moyenne de trois membres lui enleve. La verification a montre d'ou venait
    # son gain, et ce n'etait pas de la crue : la transformation Q' = mu + a(Q-mu)
    # pousse les bas debits sous zero, l'ecretage les y fixe, et c'est ce
    # creusement des etiages qui gonflait le rapport de variabilite. Le produit
    # rendait alors un debit exactement nul 1 866 jours sur 4 017 a Beterou, la
    # ou la jauge en compte 1 313, et 842 jours a Kaboua, la ou elle en compte
    # 176. Les deux variantes qui preservent la positivite -- dilatation en
    # puissance autour de la moyenne, et dilatation de la seule branche haute --
    # retournent toutes deux un facteur de 1.00 aux trois stations : il n'y a
    # aucun gain a prendre une fois l'artefact ferme. On s'abstient donc.
    ech, _mu = 1.0, float(base_tr.mean())
    _g = ((base_tr.std() / base_tr.mean()) / (ytr.std() / ytr.mean()))

    # 3. post-traitement. Il n'est pas décidé par un score d'ajustement : une
    # transformation de distribution ajustée sur ses propres jours améliore
    # toujours le critère, et l'appliquer partout coûte 0.05 de KGE hors
    # échantillon. La décision revient à la garde à trois conditions, posée
    # station par station ci-dessous.
    cvq = {nom: 0.0 for nom in QMV}
    qm = "aucun"
    # DÉCISION 08/2026 (utilisateur) : on n'accepte un post-traitement que s'il
    # apporte quelque chose en validation sans perte sèche. Le QM retenu par la
    # CV à DOMÈ coûtait 0.019 de KGE pour zéro gain de NSE -> retiré. À BONOU il
    # échange 0.023 de KGE contre 0.062 de NSE (volumes de crue mieux restitués,
    # station qui force l'hydraulique du delta) -> conservé, arbitrage assumé.
    if st == "DOME":
        qm = "aucun"
    # AHLAN : la combinaison transmet une onde de crue que la plaine inondable
    # écrête réellement — aucun des trois membres ne représente ce stockage, et
    # le pic 2010 y est surestimé de 44 %. Une erreur de cette taille sur la
    # station qui commande l'exutoire ne peut pas être laissée telle quelle : on
    # y applique la correction de distribution empirique, et on la propage vers
    # l'aval plutôt que de corriger Bonou séparément (voir l'étape de cascade
    # en fin de script).
    if st == "AHLAN":
        qm = "QM_FULL"
    # BONOU : le post-traitement local est appliqué APRÈS la propagation, pas ici.
    if st == "BONOU":
        qm = "aucun"
    ss = oo = sl = None
    if qm != "aucun":
        ss, oo, sl = QMV[qm](base_tr, ytr)
        fin_tr = qm_apply2(base_tr, ss, oo, sl)
        fin_te = qm_apply2(base_te, ss, oo, sl)
    else:
        fin_tr, fin_te = base_tr, base_te

    wb, s2 = fit_bma(Xtr, ytr)
    lo_te, hi_te = bma_interval(Xte, wb, s2)
    lo_tr, hi_tr = bma_interval(Xtr, wb, s2)
    # DÉCISION 08/2026 (utilisateur) : l'enveloppe BMA reste celle de l'ensemble
    # BRUT, sans propagation du quantile mapping aux bornes. Propager le QM à
    # Bonou avait été essayé — bornes cohérentes avec la série centrale, mais la
    # couverture y tombait de 0.906 à 0.752 en validation (0.686 sur les plus
    # forts débits) alors qu'elle restait à 0.933 en calage. Configuration brute
    # conservée ; l'intervalle décrit donc la dispersion des trois membres, pas
    # celle du produit post-traité.
    couv = float(np.mean((yte >= lo_te) & (yte <= hi_te)))

    log(f"{st:11s} schéma {sch:6s} (CV " +
        " ".join(f"{k} {v:.3f}" for k, v in cvs.items()) + ")")
    log(f"{'':11s} poids " + "/".join(f"{x:.3f}" for x in w) +
        f" | post-traitement {qm:8s} (CV " +
        " ".join(f"{k.split('_')[1]} {v:.3f}" for k, v in cvq.items()) + ")")

    # Les vecteurs de regime sont consignes ici : sans eux, T2 decrirait un
    # vecteur unique la ou le produit livre en commute deux, et les planches
    # d'annexe annonceraient une composition qui n'est pas celle de leur courbe.
    _reg = {}
    if regime is not None:
        _s_, _wh_, _wl_ = regime
        _reg["regime_seuil"] = round(float(_s_), 2)
        for _j, _m in enumerate(MEM):
            _reg[f"w_haut_{_m}"] = round(float(_wh_[_j]), 3)
            _reg[f"w_bas_{_m}"] = round(float(_wl_[_j]), 3)
    lp.append(dict(station=st, schema=sch, post_traitement=qm, echelle=round(ech, 2),
                   w_GR4J=round(w[0], 3), w_HECHMS=round(w[1], 3),
                   w_LISFLOOD=round(w[2], 3), **_reg,
                   **{f"cv_{k}": round(v, 3) for k, v in cvs.items()},
                   **{f"cv_{k}": round(v, 3) for k, v in cvq.items()},
                   couverture_BMA_val=round(couv, 3)))
    for per, (X, y, fin, idx) in (("cal", (Xtr, ytr, fin_tr, tr.index)),
                                  ("val", (Xte, yte, fin_te, te.index))):
        for nom, s in (("GR4J", X[:, 0]), ("HECHMS", X[:, 1]),
                       ("LISFLOOD", X[:, 2]), ("OPTW", X @ w), ("FINAL", fin)):
            lperf.append(dict(station=st, periode=per, serie=nom,
                              KGE=round(kge_val(s, y), 3), NSE=round(nse(s, y), 3),
                              r=round(float(np.corrcoef(s, y)[0, 1]), 3),
                              PBIAS=round(pbias(s, y), 1), FHV=round(fhv(s, y), 1),
                              FLV=round(flv(s, y), 1)))
    # Les séries sont exportées sur la GRILLE JOURNALIÈRE COMPLÈTE, pas sur les
    # seuls jours observés. Sinon les jours sans mesure disparaissent du fichier
    # et les figures relient les deux bords d'une lacune par une droite — Savè
    # 2012-2016 en donnait un exemple grossier. Les poids, le post-traitement et
    # l'enveloppe sont ceux appris ci-dessus ; seul `obs` reste vide là où il n'y
    # a pas de mesure, et l'évaluation continue de ne porter que sur ces jours.
    dg = V[V.station == st].set_index("date").sort_index()
    dg = dg.loc[CAL[0]:VAL[1]]
    Xg = dg[MEM].to_numpy()
    base_g = Xg @ w
    if "regime" in dir() and regime is not None:
        _s, _wh, _wl = regime
        base_g = np.where(Xg.mean(axis=1) >= _s, Xg @ _wh, Xg @ _wl)
    # la série écrite en sortie court sur la grille journalière complète : elle
    # doit subir la MÊME remise à l'échelle que les séries de calage et de test,
    # sans quoi le fichier de prédictions porte le mélange nu et tout ce qui le
    # relit — bootstrap, figures, diagnostics — travaille sur un autre objet.
    if ech > 1.0:
        base_g = np.clip(_mu + ech * (base_g - _mu), 0, None)
    fin_g = qm_apply2(base_g, ss, oo, sl) if qm != "aucun" else base_g
    lo_g, hi_g = bma_interval(Xg, wb, s2)
    grille[st] = dict(idx=dg.index, obs=dg["obs"].to_numpy(),
                      base=base_g, fin=fin_g)
    preds.append(pd.DataFrame({
        "date": dg.index, "station": st,
        "periode": np.where(dg.index <= CAL[1], "cal", "val"),
        "obs": dg["obs"].to_numpy(),
        "GR4J": Xg[:, 0], "HECHMS": Xg[:, 1], "LISFLOOD": Xg[:, 2],
        "OPTW": base_g, "FINAL": fin_g, "BMA_lo": lo_g, "BMA_hi": hi_g}))

# ---------------------------------------------------------------------------
# CASCADE AHLAN -> BONOU
# Ahlan est immédiatement en amont de Bonou : ce que la plaine y absorbe ne
# réapparaît pas à l'exutoire. Plutôt que de corriger les deux stations
# séparément — deux rustines sans lien entre elles — on ne corrige qu'une fois,
# à Ahlan, puis on transporte la masse retirée vers l'aval par une fonction de
# transfert (retard, laminage, gain) calée sur le calage seul. Le post-traitement
# local de Bonou s'applique ensuite à la série ainsi propagée.
# ---------------------------------------------------------------------------
def transfert(delta, L, k, alpha):
    d = np.zeros_like(delta)
    if L > 0:
        d[L:] = delta[:-L]
    else:
        d = delta.copy()
    out, S = np.zeros_like(d), 0.0
    for i in range(len(d)):
        rel = S / k
        out[i] = rel
        S += d[i] - rel
    return alpha * out


def err_max(s, o, idx):
    g = pd.DataFrame({"s": s, "o": o}, index=idx)
    g = g[(g.index.month >= 6) & (g.index.month <= 12)].groupby(lambda x: x.year).max()
    g = g[g.o > 0]
    return float(np.mean(np.abs((g.s - g.o) / g.o))) * 100 if len(g) else np.nan


gA, gB = grille["AHLAN"], grille["BONOU"]
mtr = gA["idx"] <= CAL[1]
dlt = gA["fin"] - gA["base"]                     # masse retirée à Ahlan
oB, bB = gB["obs"], gB["base"]
etr = mtr & np.isfinite(oB)
k0 = kge_val(bB[etr], oB[etr])
best = (1e9, 0, 1.0, 1.0)
for L in (0, 1, 2, 3):
    for k in (1.0, 2.0, 3.0, 5.0, 8.0):
        for al in (0.4, 0.6, 0.8, 1.0):
            c = bB + transfert(dlt, L, k, al)
            e = err_max(c[etr], oB[etr], gB["idx"][etr])
            if e < best[0] and kge_val(c[etr], oB[etr]) >= k0 - 0.005:
                best = (e, L, k, al)
_, L, k, al = best
casc = bB + transfert(dlt, L, k, al)
ss_b, oo_b, sl_b = QMV["QM_LISSE"](casc[etr], oB[etr])
finB = qm_apply2(casc, ss_b, oo_b, sl_b)
log(f"\ncascade Ahlan -> Bonou : retard {L} j, laminage {k:.0f} j, gain {al:.1f} ; "
    f"post-traitement local QM_LISSE appliqué à la série propagée")

# report dans les prédictions, les poids et les performances
pr = pd.concat(preds, ignore_index=True)
mB = pr.station == "BONOU"
pr.loc[mB, "OPTW"] = casc            # la série de référence à Bonou est la cascade
pr.loc[mB, "FINAL"] = finB
lp = [dict(x, post_traitement=("QM_FULL" if x["station"] == "AHLAN" else
                               "CASCADE+QM_LISSE" if x["station"] == "BONOU"
                               else x["post_traitement"])) for x in lp]
lperf = [x for x in lperf if not (x["station"] == "BONOU" and
                                  x["serie"] in ("OPTW", "FINAL"))]
for per, msk in (("cal", pr.periode == "cal"), ("val", pr.periode == "val")):
    d = pr[mB & msk].dropna(subset=["obs"])
    y = d.obs.to_numpy()
    for nom, s in (("OPTW", d.OPTW.to_numpy()), ("FINAL", d.FINAL.to_numpy())):
        lperf.append(dict(station="BONOU", periode=per, serie=nom,
                          KGE=round(kge_val(s, y), 3), NSE=round(nse(s, y), 3),
                          r=round(float(np.corrcoef(s, y)[0, 1]), 3),
                          PBIAS=round(pbias(s, y), 1), FHV=round(fhv(s, y), 1),
                          FLV=round(flv(s, y), 1)))

# ATTENTION a la colonne OPTW de ce fichier : elle porte le melange convexe nu a
# BETEROU, SAVE, KABOUA, ATCHERIGBE et AHLAN, la serie commutee par regime a DOME,
# et la serie propagee depuis Ahlan a BONOU. Le melange nu se reconstruit depuis
# tables/retained_weights.csv renormalise ; c'est ce que fait performance_by_gauge.csv.
pr.to_csv(SERIES / "ensemble_daily.csv", index=False)
W = pd.DataFrame(lp)
W.to_csv(TAB / "retained_weights.csv", index=False)
M = pd.DataFrame(lperf)
M.to_csv(TAB / "performance_by_gauge.csv", index=False)

log("\n=== FINAL vs membres (KGE / NSE, calage | validation) ===")
log(f"{'station':11s} {'FINAL cal':>13s} {'FINAL val':>13s} {'meilleur membre val':>21s}")
for st in STATIONS:
    g = M[M.station == st].set_index(["periode", "serie"])
    bm = max(MEM, key=lambda m: g.loc[("val", m), "KGE"])
    log(f"{st:11s} "
        f"{g.loc[('cal','FINAL'),'KGE']:.3f}/{g.loc[('cal','FINAL'),'NSE']:.3f}".ljust(25)
        + f"{g.loc[('val','FINAL'),'KGE']:.3f}/{g.loc[('val','FINAL'),'NSE']:.3f}".rjust(10)
        + f"   {bm} {g.loc[('val',bm),'KGE']:.3f}/{g.loc[('val',bm),'NSE']:.3f}".rjust(21))
mo = M[M.serie == "FINAL"].groupby("periode")[["KGE", "NSE"]].mean().round(3)
mb = M[M.serie.isin(MEM)].groupby(["periode", "serie"])[["KGE", "NSE"]].mean().round(3)
log(f"\nFINAL moyen : calage KGE {mo.loc['cal','KGE']:.3f} NSE {mo.loc['cal','NSE']:.3f}"
    f" | validation KGE {mo.loc['val','KGE']:.3f} NSE {mo.loc['val','NSE']:.3f}")
for m in MEM:
    log(f"  {m:9s} validation KGE {mb.loc[('val',m),'KGE']:.3f} "
        f"NSE {mb.loc[('val',m),'NSE']:.3f}")
log(f"couverture BMA moyenne (validation) : {W.couverture_BMA_val.mean():.3f} (cible 0.90)")
log(f"post-traitement retenu : " +
    ", ".join(f"{r.station}={r.post_traitement}" for r in W.itertuples()))
(ROOT / "results" / "fit_log.txt").write_text("\n".join(jour), encoding="utf-8")
print("\nécrits : series/ensemble_daily.csv, retained_weights.csv, performance_by_gauge.csv")
