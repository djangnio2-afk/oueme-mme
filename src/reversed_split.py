#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reversed-split robustness test: weights (and the guarded QM) are trained on
2006-2016, FROZEN, and evaluated on 1997-2005. If the conclusions of the main
split (OPTW robust > best member; unconstrained GR-C over-fits; guarded QM helps
peaks) hold in both directions, they are not artefacts of the split choice.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from mme_lib import (TAB, MEMBERS, STATIONS, SERIES, kge_val, metrics, fit_optw,
                     qm_fit, qm_apply, load_long)

TRAIN = ("2006-01-01", "2016-12-31")   # reversed
TEST  = ("1997-01-01", "2005-12-31")

def nested_qm_decision(Xtr, ytr, k=5):
    n = len(ytr); idx = np.array_split(np.arange(n), k)
    p_optw = np.full(n, np.nan); p_qm = np.full(n, np.nan)
    for f in range(k):
        te = idx[f]; tr = np.concatenate([idx[j] for j in range(k) if j != f])
        w = fit_optw(Xtr[tr], ytr[tr], N=40, refine=False)
        base_te = Xtr[te] @ w; p_optw[te] = base_te
        ss, oo = qm_fit(Xtr[tr] @ w, ytr[tr])
        p_qm[te] = qm_apply(base_te, ss, oo)
    return kge_val(p_optw, ytr), kge_val(p_qm, ytr)

long = load_long()
rows = []
for st in STATIONS:
    d = long[long.station == st].dropna(subset=["obs"] + MEMBERS).set_index("date").sort_index()
    tr = d.loc[TRAIN[0]:TRAIN[1]]; te = d.loc[TEST[0]:TEST[1]]
    if len(tr) < 200 or len(te) < 200:
        print(f"[skip] {st} (train {len(tr)}, test {len(te)})"); continue
    Xtr, ytr = tr[MEMBERS].to_numpy(), tr["obs"].to_numpy()
    Xte, yte = te[MEMBERS].to_numpy(), te["obs"].to_numpy(); dte = te.index

    ktr = {m: kge_val(tr[m].to_numpy(), ytr) for m in MEMBERS}
    best = max(ktr, key=ktr.get); bi = MEMBERS.index(best)
    preds = {"B0_best": Xte[:, bi], "B1_mean": Xte.mean(axis=1)}
    w2 = np.array([max(ktr[m], 0) for m in MEMBERS]); w2 /= w2.sum()
    preds["B2_kgew"] = Xte @ w2
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr]); cc, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    preds["GRC"] = np.hstack([np.ones((len(Xte), 1)), Xte]) @ cc
    w = fit_optw(Xtr, ytr); base_te = Xte @ w
    preds["OPTW"] = base_te
    # guarded QM (same gate as finalize.py)
    base_tr = Xtr @ w; thr98 = np.percentile(ytr, 98); h = ytr >= thr98
    fhv_tr = 100 * (base_tr[h].sum() - ytr[h].sum()) / ytr[h].sum()
    # garde-fou STRICTEMENT identique a celui de finalize.py : sur-estimation
    # marquee des pics, partagee par les TROIS membres, et gain confirme par
    # validation croisee emboitee. La valeur absolue laissait auparavant passer
    # les stations a fort DEFICIT de pointe, que le garde direct refuse.
    memb_fhv = 100 * (Xtr[h].sum(axis=0) - ytr[h].sum()) / ytr[h].sum()
    consensus_over = bool((memb_fhv > 0).all())
    k_o, k_q = nested_qm_decision(Xtr, ytr)
    if (k_q > k_o + 0.01) and (fhv_tr > 15) and consensus_over:
        ss, oo = qm_fit(base_tr, ytr)
        preds["FINAL"] = qm_apply(base_te, ss, oo); qm_on = True
    else:
        preds["FINAL"] = base_te; qm_on = False
    for name, p in preds.items():
        m = metrics(p, yte, dte)
        rows.append(dict(station=st, method=name, KGE=round(m["KGE"], 3),
                         NSE=round(m["NSE"], 3),
                         FHV=round(m["FHV"], 1), PBIAS=round(m["PBIAS"], 1),
                         qm=qm_on if name == "FINAL" else None,
                         w_str="/".join(f"{x:.2f}" for x in w) if name == "OPTW" else None))

R = pd.DataFrame(rows)
R.to_csv(TAB / "reversed_split.csv", index=False)
piv = R.pivot(index="station", columns="method", values="KGE")[["B0_best","B1_mean","B2_kgew","GRC","OPTW","FINAL"]]
piv.loc["MEAN"] = piv.mean()
print("=== REVERSED SPLIT: train 2006-2016 -> frozen test 1997-2005 ===")
print(piv.round(3).to_string())
print("\nOPTW weights (GR4J/HEC/LIS) and QM decisions:")
print(R[R.method=="OPTW"][["station","w_str"]].to_string(index=False))
print(R[R.method=="FINAL"][["station","qm"]].to_string(index=False))
print(f"\nSaved -> {SERIES/'reversed_split.csv'}")
