#!/usr/bin/env python3
"""
Diagnostic — ou se trouve le temps d'arret dans les donnees ?
=============================================================

La penalite de tour de sortie est ressortie NEGATIVE de 02_calibrate_pace.py,
ce qui est physiquement impossible. Hypothese : le temps de passage au stand
(~23 s) est attribue au tour de SORTIE (age = 1) et non au tour d'ENTREE
(dernier tour du relais precedent). Dans ce cas mon filtre 93-107% elimine
tous les vrais tours de sortie, et le coefficient est estime sur un
echantillon biaise.

Ce script trace le profil du temps au tour en fonction de l'age du pneu,
en ecart relatif a la mediane du GP. Lecture :

  * pic massif (>115%) a age = 1  -> le temps d'arret est sur le tour de
    SORTIE. Il faut exclure age = 1 et estimer la penalite sur age = 2 seul,
    ou corriger l'attribution en amont.
  * pic massif a "dernier tour du relais"  -> attribution au tour d'ENTREE
    (convention attendue). Il suffit alors d'elargir la bande pour les tours
    de sortie.
  * leger surcout (102-105%) a age = 1-2 et rien d'autre -> les deux
    conventions sont deja gerees, le biais vient d'ailleurs.

Aucune ecriture : ce script ne fait que lire et afficher.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GHOSTS = ROOT / "data" / "raw" / "all_drivers_dataset.parquet"

df = pd.read_parquet(GHOSTS)
need = {"season", "event", "driver", "lap_number", "lap_time_s",
        "tyre_age_laps", "stint_number"}
if need - set(df.columns):
    sys.exit(f"colonnes manquantes : {need - set(df.columns)}")

# on travaille sur le temps BRUT si 01_clean_laps l'a conserve
col = "lap_time_s_raw" if "lap_time_s_raw" in df.columns else "lap_time_s"
print(f"Colonne utilisee : {col}\n")

df = df.dropna(subset=[col, "tyre_age_laps", "stint_number"])
df = df[(df[col] > 0) & (df["lap_number"] > 1)]
for c in ("is_safety_car", "is_vsc"):
    if c in df.columns:
        df = df[~df[c].astype("boolean").fillna(False).to_numpy(dtype=bool)]

med = df.groupby(["season", "event"])[col].median().rename("med")
df = df.merge(med, on=["season", "event"])
df = df[df[col] < 5 * df["med"]]          # retire les artefacts sesT
df["ratio"] = 100 * df[col] / df["med"]

# marquage du dernier tour de chaque relais (tour d'entree presume)
df = df.sort_values(["season", "event", "driver", "lap_number"])
g = df.groupby(["season", "event", "driver"])["stint_number"]
df["is_out_lap"] = (g.diff().fillna(0) != 0)
df["is_in_lap"] = df["is_out_lap"].shift(-1, fill_value=False)

print("=" * 66)
print("PROFIL PAR AGE DE PNEU (ecart a la mediane du GP, en %)")
print("=" * 66)
print(f"{'age':>5} {'n':>7} {'mediane':>9} {'p25':>8} {'p75':>8} {'p95':>8}")
for age in list(range(1, 11)) + [15, 20, 25, 30]:
    s = df.loc[df["tyre_age_laps"] == age, "ratio"]
    if len(s) < 10:
        continue
    print(f"{age:>5} {len(s):>7} {s.median():>9.1f} {s.quantile(.25):>8.1f} "
          f"{s.quantile(.75):>8.1f} {s.quantile(.95):>8.1f}")

print("\n" + "=" * 66)
print("OU EST LE TEMPS D'ARRET ?")
print("=" * 66)
for label, mask in (("tour de sortie (1er du relais)", df["is_out_lap"]),
                    ("tour d'entree (dernier du relais)", df["is_in_lap"]),
                    ("tour courant (ni l'un ni l'autre)",
                     ~df["is_out_lap"] & ~df["is_in_lap"])):
    s = df.loc[mask, "ratio"]
    if len(s) < 10:
        continue
    over = 100 * (s > 115).mean()
    print(f"{label:<36} n={len(s):>6}  mediane {s.median():>6.1f}%  "
          f"{over:>5.1f}% au-dessus de 115%")

out = df.loc[df["is_out_lap"], "ratio"].median()
inl = df.loc[df["is_in_lap"], "ratio"].median()
print("\nCONCLUSION")
if out > 115:
    print("  Le temps d'arret est sur le TOUR DE SORTIE.")
    print("  -> exclure age = 1 de la regression et estimer la penalite de")
    print("     pneu froid sur age = 2 uniquement (elle sera sous-estimee ;")
    print("     la documenter comme borne basse).")
elif inl > 115:
    print("  Le temps d'arret est sur le TOUR D'ENTREE (convention attendue).")
    print("  -> elargir la bande haute a 120% pour les tours de sortie dans")
    print("     clean_subset : ils sont legitimement lents, pas aberrants.")
else:
    print("  Aucun des deux ne porte un surcout marque : l'attribution du")
    print("  temps d'arret est faite ailleurs (colonne dediee ?) ou les tours")
    print("  d'arret sont deja retires de ce jeu de donnees.")
print(f"\n  mediane tour de sortie : {out:.1f}%  |  tour d'entree : {inl:.1f}%")
