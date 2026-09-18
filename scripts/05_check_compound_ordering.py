#!/usr/bin/env python3
"""
Diagnostic — l'ordre SOFT > MEDIUM > HARD est-il estimable ?
============================================================

Constat : trois calibrations successives donnent SOFT ~ MEDIUM (ecart < 3%),
alors que la physique du sport impose SOFT > MEDIUM > HARD.

Hypothese : biais de selection par longueur de relais. Les equipes font des
relais COURTS en SOFT et s'arretent avant la chute de performance ; MEDIUM et
HARD sont vus jusqu'a des ages eleves. Les pentes sont donc estimees sur des
FENETRES D'AGE DIFFERENTES et ne sont pas comparables.

Ce script :
  1. affiche la distribution des ages observes par compose,
  2. reestime les pentes sur une FENETRE COMMUNE (ages 3 a N), pour plusieurs
     valeurs de N, et regarde si l'ordre apparait.

Lecture :
  * l'ordre apparait sur une fenetre commune -> utiliser cette fenetre dans
    02_calibrate_pace.py (variable AGE_WINDOW).
  * l'ordre n'apparait sur aucune fenetre -> la donnee ne permet pas de
    separer les composes. Le documenter comme limite et passer a la suite.

Aucune ecriture.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "raw" / "all_drivers_dataset.parquet")
SLICKS = ["SOFT", "MEDIUM", "HARD"]


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


data = pd.concat([clean(g) for _, g in df.groupby(["season", "event"])],
                 ignore_index=True)
print(f"{len(data):,} tours exploitables\n")

print("=" * 70)
print("DISTRIBUTION DES AGES DE PNEU OBSERVES")
print("=" * 70)
print(f"{'compose':>8} {'n':>8} {'p10':>6} {'mediane':>9} {'p90':>6} {'max':>6}")
for c in SLICKS:
    s = data.loc[data["tyre_compound"] == c, "tyre_age_laps"]
    if len(s) < 50:
        continue
    print(f"{c:>8} {len(s):>8} {s.quantile(.10):>6.0f} {s.median():>9.0f} "
          f"{s.quantile(.90):>6.0f} {s.max():>6.0f}")
print("\nSi les p90 different fortement, les pentes ne sont pas comparables :")
print("elles sont estimees sur des portions differentes de la courbe.\n")


def demean(cols, keys, n_iter=25):
    out = cols.copy()
    for _ in range(n_iter):
        before = out.copy()
        for k in keys:
            out = out - pd.DataFrame(out).groupby(k, observed=True).transform("mean").to_numpy()
        if np.abs(out - before).max() < 1e-9:
            break
    return out


def slopes_in_window(d, age_max):
    g = d[(d["tyre_age_laps"] > 2) & (d["tyre_age_laps"] <= age_max)]
    comps = [c for c in SLICKS if (g["tyre_compound"] == c).sum() >= 200]
    if len(comps) < 3:
        return None, len(g)
    gl = pd.factorize(pd.Series(list(zip(g["season"], g["event"], g["lap_number"]))))[0]
    gd = pd.factorize(pd.Series(list(zip(g["season"], g["event"], g["driver"]))))[0]
    age = g["tyre_age_laps"].to_numpy(float) - 2
    cols = [g["lap_time_s"].to_numpy(float)]
    for c in comps:
        cols.append(age * (g["tyre_compound"] == c).to_numpy(float))
    M = demean(np.column_stack(cols), [gl, gd])
    coef, *_ = np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)
    return {c: float(coef[i]) for i, c in enumerate(comps)}, len(g)


print("=" * 70)
print("PENTES SUR FENETRE D'AGE COMMUNE")
print("=" * 70)
print(f"{'fenetre':>12} {'n tours':>9} {'SOFT':>9} {'MEDIUM':>9} {'HARD':>9}  ordre ?")
found = False
for amax in (10, 15, 20, 25, 30, 40, 999):
    sl, n = slopes_in_window(data, amax)
    if sl is None:
        print(f"{'3-' + str(amax):>12} {n:>9}   pas assez de donnees pour 3 composes")
        continue
    ordered = sl["SOFT"] > sl["MEDIUM"] > sl["HARD"]
    found |= ordered
    label = "3-" + ("inf" if amax == 999 else str(amax))
    print(f"{label:>12} {n:>9} {sl['SOFT']:>9.4f} {sl['MEDIUM']:>9.4f} "
          f"{sl['HARD']:>9.4f}  {'OUI' if ordered else 'non'}")

print("\n" + "=" * 70)
if found:
    print("CONCLUSION : l'ordre apparait sur au moins une fenetre commune.")
    print("-> fixer AGE_WINDOW dans 02_calibrate_pace.py sur cette fenetre,")
    print("   et documenter le choix (biais de longueur de relais).")
else:
    print("CONCLUSION : l'ordre n'apparait sur AUCUNE fenetre.")
    print("-> la donnee ne permet pas de separer les composes. Deux options :")
    print("   (a) garder les pentes estimees et documenter que le choix de")
    print("       compose n'est pas un levier strategique du modele ;")
    print("   (b) imposer un ecart relatif issu de donnees publiques Pirelli,")
    print("       en l'assumant explicitement comme hypothese non calibree.")
    print("   L'option (a) est plus honnete et suffit pour la certification :")
    print("   l'agent apprendra QUAND s'arreter, pas AVEC QUEL pneu.")
