#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spatial transferability of the convex weights, leaving one station out.

The learning window of the six other stations is pooled, each station
normalized by the mean of its own members over that window, which is a
non-leaking scale since it is computed from the members alone. One convex
weight vector is fitted on the pooled set, then applied to the test window of
the station left out, brought back to its own scale. The comparison is against
the weighting fitted locally on that same station.

Reads  data/member_simulations_daily.csv
Writes results/tables/transferability_loso.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from mme_lib import TAB, DATA, SERIES, MEMBERS, STATIONS, fit_optw, kge_val

TRAIN = ("1995-01-01", "2005-12-31")
TEST = ("2006-01-01", "2016-12-31")

long = pd.read_csv(DATA / "member_simulations_daily.csv", parse_dates=["date"])
d = {st: long[long.station == st].dropna(subset=["obs"] + MEMBERS)
                                 .set_index("date").sort_index() for st in STATIONS}
scale = {st: d[st].loc[TRAIN[0]:TRAIN[1], MEMBERS].mean(axis=1).mean() for st in STATIONS}

rows = []
for held in STATIONS:
    Xp, yp = [], []
    for st in STATIONS:
        if st == held:
            continue
        tr = d[st].loc[TRAIN[0]:TRAIN[1]]
        Xp.append(tr[MEMBERS].to_numpy() / scale[st])
        yp.append(tr["obs"].to_numpy() / scale[st])
    w_pool = fit_optw(np.vstack(Xp), np.concatenate(yp))

    te = d[held].loc[TEST[0]:TEST[1]]
    k_loso = kge_val(te[MEMBERS].to_numpy() @ w_pool, te["obs"].to_numpy())

    tr = d[held].loc[TRAIN[0]:TRAIN[1]]
    w_loc = fit_optw(tr[MEMBERS].to_numpy(), tr["obs"].to_numpy())
    k_spec = kge_val(te[MEMBERS].to_numpy() @ w_loc, te["obs"].to_numpy())

    rows.append(dict(station=held,
                     KGE_OPTW_stationspecific=round(k_spec, 3),
                     KGE_OPTW_LOSO=round(k_loso, 3),
                     w_pool="/".join(f"{x:.2f}" for x in w_pool)))

R = pd.DataFrame(rows)
R.to_csv(TAB / "transferability_loso.csv", index=False)
loss = R.KGE_OPTW_stationspecific - R.KGE_OPTW_LOSO
print(R.to_string(index=False))
print(f"\nperte moyenne du transfert : {loss.mean():+.3f} KGE")
print(f"stations perdant 0.01 ou moins : {int((loss <= 0.01).sum())}/{len(R)}")
print(f"-> {SERIES / 'transferability_loso.csv'}")
