#!/usr/bin/env python3
"""
V2 — Phase 0 : forme reelle de la courbe de degradation
========================================================

Pourquoi cette mesure avant d'implementer quoi que ce soit
-----------------------------------------------------------
La V1 mesure la degradation sur des fenetres d'age CUMULATIVES :

    ages 3-10  -> 0.0737 s/tour   (MEDIUM)
    ages 3-inf -> 0.0429 s/tour

La seconde valeur INCLUT deja la premiere. Les utiliser comme deux segments
d'une fonction par morceaux serait faux : on prendrait une moyenne globale
pour une pente de fin de relais.

Ce script mesure les pentes sur des fenetres DISJOINTES (3-10, 11-20, 21-30,
31+), ce qui donne la vraie forme de la courbe et permet de choisir en
connaissance de cause entre :

  * lineaire par morceaux  : une pente par segment
  * logarithmique          : t = a * ln(1 + b*age)     -> decelere
  * exponentielle saturante: t = a * (1 - exp(-b*age)) -> plateau

Methode identique a 02_calibrate_pace.py : effets fixes (GP x tour) et
(GP x pilote) retires par projections alternees, donc l'effet carburant,
l'evolution de piste et la meteo sont absorbes. Seul l'age du pneu subsiste.

Aucune ecriture hors du rapport. Lancer depuis la racine du projet V2 :
    python scripts/v2_00_measure_degradation_curve.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GHOSTS = ROOT / "data" / "raw" / "all_drivers_dataset.parquet"
OUT = ROOT / "docs" / "v2_degradation_curve.md"

SLICKS = ["SOFT", "MEDIUM", "HARD"]
WINDOWS = [(3, 10), (11, 20), (21, 30), (31, 45), (46, 99)]
MIN_ROWS = 150

lines = ["# V2 — Forme de la courbe de degradation\n"]


def emit(t=""):
    print(t)
    lines.append(t)


def clean(g):
    g = g.dropna(subset=["lap_time_s", "tyre_age_laps", "tyre_compound", "stint_number"])
    g = g[g["lap_time_s"] > 0].sort_values(["driver", "lap_number"])
    ns = g.groupby("driver")["stint_number"].diff().fillna(0) != 0
    g = g[~(ns | ns.shift(-1, fill_value=False))]
    g = g[(g["tyre_age_laps"] != 1) & (g["lap_number"] > 1)]
    for c in ("is_safety_car", "is_vsc"):
        if c in g.columns:
            g = g[~g[c].astype("boolean").fillna(False).to_numpy(dtype=bool)]
    med = g["lap_time_s"].median()
    return g[(g["lap_time_s"] < 1.07 * med) & (g["lap_time_s"] > 0.93 * med)]


def demean(cols, keys, n_iter=30):
    out = cols.copy()
    for _ in range(n_iter):
        before = out.copy()
        for k in keys:
            out = out - pd.DataFrame(out).groupby(k, observed=True).transform("mean").to_numpy()
        if np.abs(out - before).max() < 1e-9:
            break
    return out


def slope(d, compound, lo, hi):
    """Pente locale sur une fenetre d'age DISJOINTE, effets fixes retires."""
    g = d[(d["tyre_compound"] == compound)
          & (d["tyre_age_laps"] >= lo) & (d["tyre_age_laps"] <= hi)]
    if len(g) < MIN_ROWS or g["tyre_age_laps"].nunique() < 3:
        return None, len(g)
    gl = pd.factorize(pd.Series(list(zip(g["season"], g["event"], g["lap_number"]))))[0]
    gd = pd.factorize(pd.Series(list(zip(g["season"], g["event"], g["driver"]))))[0]
    M = demean(np.column_stack([g["lap_time_s"].to_numpy(float),
                                g["tyre_age_laps"].to_numpy(float)]), [gl, gd])
    y, x = M[:, 0], M[:, 1]
    if x.std() < 1e-9:
        return None, len(g)
    return float(np.polyfit(x, y, 1)[0]), len(g)


df = pd.read_parquet(GHOSTS)
need = {"season", "event", "driver", "lap_number", "lap_time_s",
        "tyre_compound", "tyre_age_laps", "stint_number"}
if need - set(df.columns):
    sys.exit(f"colonnes manquantes : {need - set(df.columns)}")

data = pd.concat([clean(g) for _, g in df.groupby(["season", "event"])],
                 ignore_index=True)
emit(f"{len(data):,} tours exploitables\n")

emit("## Pentes locales sur fenetres d'age DISJOINTES (s/tour)\n")
emit("| compose | " + " | ".join(f"{lo}-{hi if hi < 99 else '+'}" for lo, hi in WINDOWS) + " |")
emit("|---|" + "---|" * len(WINDOWS))

curve = {}
for c in SLICKS:
    cells, vals = [], {}
    for lo, hi in WINDOWS:
        s, n = slope(data, c, lo, hi)
        if s is None:
            cells.append(f"– ({n})")
        else:
            cells.append(f"**{s:.4f}** ({n})")
            vals[f"{lo}-{hi}"] = round(s, 5)
    curve[c] = vals
    emit(f"| {c} | " + " | ".join(cells) + " |")
emit("\n_(entre parentheses : nombre de tours ayant servi a l'estimation)_\n")

emit("## Forme de la courbe\n")
verdicts = []
for c in SLICKS:
    v = list(curve[c].values())
    if len(v) < 2:
        emit(f"- **{c}** : trop peu de fenetres exploitables pour conclure.")
        continue
    first, last = v[0], v[-1]
    ratio = last / first if abs(first) > 1e-6 else float("nan")
    if ratio < 0.7:
        forme = "DECELERE — la degradation ralentit avec l'age du pneu"
    elif ratio > 1.3:
        forme = "ACCELERE — la degradation s'aggrave avec l'age (falaise)"
    else:
        forme = "LINEAIRE — pente stable, le modele actuel est adapte"
    verdicts.append(forme.split(" —")[0])
    emit(f"- **{c}** : {first:.4f} -> {last:.4f} s/tour (rapport {ratio:.2f}) — {forme}")

emit("\n## Lecture et suite\n")
if verdicts and all(v == "LINEAIRE" for v in verdicts):
    emit("Les pentes locales sont stables : **la degradation non lineaire n'est "
         "pas justifiee par les donnees**. Le point 2 de la feuille de route V2 "
         "peut etre abandonne, et l'ecart 0.0737 / 0.0429 s'explique alors "
         "entierement par l'effet de fenetre cumulative.")
elif verdicts and all(v == "DECELERE" for v in verdicts):
    emit("Les pentes decroissent avec l'age : retenir une forme **logarithmique** "
         "`delta_t = a * ln(1 + b * age)` ou lineaire par morceaux sur les "
         "fenetres ci-dessus. Le modele lineaire actuel sous-estime le debut de "
         "relais et surestime la fin.")
elif verdicts and all(v == "ACCELERE" for v in verdicts):
    emit("Les pentes croissent avec l'age : c'est le profil de **falaise** "
         "(cliff) connu en F1. Retenir une forme convexe, par exemple "
         "`delta_t = a * age + c * max(0, age - seuil)^2`.")
else:
    emit("Les composes n'ont pas la meme forme de courbe. Modeliser chacun "
         "separement, ou retenir le lineaire par morceaux qui s'adapte aux trois "
         "sans imposer de forme fonctionnelle commune.")

emit("\n> **Attention aux fenetres a faible effectif.** Une pente estimee sur "
     "moins de 300 tours est peu fiable ; les relais tres longs (46+) ne sont "
     "observes que sur les strategies a 0 ou 1 arret, donc sur un echantillon "
     "biaise vers les circuits a faible degradation.\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
(ROOT / "data" / "processed" / "v2_degradation_curve.json").write_text(
    json.dumps({"windows": [list(w) for w in WINDOWS], "slopes_s_per_lap": curve},
               indent=2), encoding="utf-8")
print(f"\nRapport : {OUT}")
print(f"Donnees : {ROOT / 'data' / 'processed' / 'v2_degradation_curve.json'}")
