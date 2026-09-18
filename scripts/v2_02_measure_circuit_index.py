#!/usr/bin/env python3
"""
V2 — Phase 2a : index de degradation par circuit
=================================================

Pourquoi
---------
12 des 19 GP hors pool ne sont battus par aucun algorithme et aucune graine
face a l'oracle, Monza en tete ou les trois s'abstiennent identiquement.
L'espace d'observation ne contient aucune variable decrivant la severite
abrasive du trace : l'agent doit l'inferer de l'age du pneu seul, et il y
echoue sur les profils qu'il n'a pas rencontres.

Cet index fournit cette information a priori, sous forme d'un scalaire
statique par GP.

Ce n'est PAS une fuite de donnees. La severite pneumatique d'un circuit est
connue de toutes les equipes avant le depart : Pirelli la publie, les equipes
l'historisent, et c'est l'entree principale d'un stratege. L'agent recoit donc
une information dont disposerait n'importe quel decideur reel, pas un resultat
futur. A documenter comme tel dans le dossier.

Methode
--------
Regression par circuit sur la fenetre d'age 3-20 tours uniquement. Au-dela,
la Phase 0 a montre que les pentes ne sont pas estimables : elles deviennent
negatives sur HARD a cause d'une censure informative (un pneu n'atteint 30
tours que lorsque tout va bien, donc les relais longs survivants sont
selectionnes sur leur faible degradation).

Deux choix pour la robustesse :

  * les trois composes secs sont utilises, ponderes par effectif, plutot que
    MEDIUM seul. Sur les GP ou le peloton a peu roule en MEDIUM, une
    estimation mono-compose serait instable ou absente.
  * la normalisation est bornee aux percentiles 5 et 95 avant tronquage a
    [0, 1] : une seule valeur aberrante ecraserait sinon toute l'echelle.

Test de plausibilite
---------------------
Le classement doit etre coherent avec la connaissance du sport. Bahrein et
l'Espagne sont des circuits notoirement abrasifs ; Monza et Bakou, domines
par les lignes droites, le sont peu. Si Monza ressort en tete, l'index est
faux et ne doit pas etre integre.

Aucune ecriture hors du rapport et du JSON. Lancer depuis la racine V2 :
    python scripts/v2_02_measure_circuit_index.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GHOSTS = ROOT / "data" / "raw" / "all_drivers_dataset.parquet"
OUT_JSON = ROOT / "data" / "processed" / "circuit_degradation_index.json"
OUT_MD = ROOT / "docs" / "v2_circuit_index.md"

AGE_MIN, AGE_MAX = 3, 20          # fenetre fiable, cf. Phase 0
SLICKS = ["SOFT", "MEDIUM", "HARD"]
MIN_ROWS = 120                    # par compose et par circuit
P_LO, P_HI = 5, 95                # percentiles de normalisation

# Reperes publics, pour le test de plausibilite uniquement — n'entrent pas
# dans le calcul.
EXPECTED_HIGH = ["Bahrain", "Spanish", "Qatar", "Austrian", "Hungarian"]
EXPECTED_LOW = ["Italian", "Azerbaijan", "Las Vegas", "Belgian", "Monaco"]

lines = ["# V2 — Index de degradation par circuit\n"]


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


def circuit_slope(g):
    """Pente ponderee par effectif sur les 3 composes secs, fenetre 3-20."""
    g = g[(g["tyre_age_laps"] >= AGE_MIN) & (g["tyre_age_laps"] <= AGE_MAX)]
    num, den, detail = 0.0, 0, {}
    for c in SLICKS:
        sub = g[g["tyre_compound"] == c]
        if len(sub) < MIN_ROWS or sub["tyre_age_laps"].nunique() < 3:
            continue
        # effets fixes tour et pilote, a l'interieur du GP
        kl = pd.factorize(sub["lap_number"])[0]
        kd = pd.factorize(sub["driver"])[0]
        M = demean(np.column_stack([sub["lap_time_s"].to_numpy(float),
                                    sub["tyre_age_laps"].to_numpy(float)]), [kl, kd])
        if M[:, 1].std() < 1e-9:
            continue
        s = float(np.polyfit(M[:, 1], M[:, 0], 1)[0])
        detail[c] = (round(s, 5), len(sub))
        num += s * len(sub)
        den += len(sub)
    if den == 0:
        return None, 0, detail
    return num / den, den, detail


df = pd.read_parquet(GHOSTS)
need = {"season", "event", "driver", "lap_number", "lap_time_s",
        "tyre_compound", "tyre_age_laps", "stint_number"}
if need - set(df.columns):
    sys.exit(f"colonnes manquantes : {need - set(df.columns)}")

emit(f"Fenetre d'age retenue : {AGE_MIN}-{AGE_MAX} tours (zone fiable, cf. Phase 0)\n")

rows = []
for (s, e), g in df.groupby(["season", "event"]):
    slope, n, detail = circuit_slope(clean(g))
    if slope is None:
        emit(f"- **{s} {e}** : ecarte, moins de {MIN_ROWS} tours exploitables par compose")
        continue
    rows.append({"season": int(s), "event": e, "slope": slope, "n": n, "detail": detail})

if len(rows) < 5:
    sys.exit("Trop peu de circuits exploitables pour normaliser.")

slopes = np.array([r["slope"] for r in rows])
lo, hi = np.percentile(slopes, P_LO), np.percentile(slopes, P_HI)
emit(f"\nNormalisation bornee aux percentiles {P_LO}/{P_HI} : "
     f"[{lo:.4f}, {hi:.4f}] s/tour\n")
for r in rows:
    r["index"] = float(np.clip((r["slope"] - lo) / (hi - lo), 0.0, 1.0)) if hi > lo else 0.5

emit("## Classement par severite\n")
emit("| circuit | pente (s/tour) | index | n tours | detail par compose |")
emit("|---|---|---|---|---|")
for r in sorted(rows, key=lambda x: -x["index"]):
    det = ", ".join(f"{c} {v[0]:.4f}" for c, v in sorted(r["detail"].items()))
    emit(f"| {r['season']} {r['event']} | {r['slope']:.4f} | **{r['index']:.2f}** "
         f"| {r['n']} | {det} |")

# --- test de plausibilite ---------------------------------------------------
emit("\n## Test de plausibilite\n")
n = len(rows)
ranked = sorted(rows, key=lambda x: -x["index"])
top_third = {r["event"] for r in ranked[: max(n // 3, 1)]}
bottom_third = {r["event"] for r in ranked[-max(n // 3, 1):]}

hits, misses = [], []
for name in EXPECTED_HIGH:
    match = [r for r in rows if name in r["event"]]
    if not match:
        continue
    pos = [i for i, r in enumerate(ranked) if r["event"] == match[0]["event"]][0]
    (hits if match[0]["event"] in top_third else misses).append(
        f"{name} attendu ABRASIF, classe {pos + 1}/{n} (index {match[0]['index']:.2f})")
for name in EXPECTED_LOW:
    match = [r for r in rows if name in r["event"]]
    if not match:
        continue
    pos = [i for i, r in enumerate(ranked) if r["event"] == match[0]["event"]][0]
    (hits if match[0]["event"] in bottom_third else misses).append(
        f"{name} attendu PEU ABRASIF, classe {pos + 1}/{n} (index {match[0]['index']:.2f})")

for h in hits:
    emit(f"- OK — {h}")
for m in misses:
    emit(f"- **ECART** — {m}")

ratio = len(hits) / max(len(hits) + len(misses), 1)
emit(f"\n**{len(hits)}/{len(hits) + len(misses)} reperes conformes "
     f"({100 * ratio:.0f} %).**\n")
if ratio >= 0.6:
    emit("L'index est coherent avec la connaissance du sport : il peut etre "
         "integre a l'observation (Phase 2b).")
else:
    emit("**NE PAS INTEGRER.** L'index contredit la severite connue des "
         "circuits ; il mesure autre chose que l'abrasivite — rythme de la "
         "voiture, conditions du jour, ou artefact d'echantillon. Revoir la "
         "methode avant d'aller plus loin.")

emit("\n> Les reperes publics servent UNIQUEMENT au controle de plausibilite. "
     "Ils n'entrent pas dans le calcul de l'index, qui est derive des seules "
     "donnees de temps au tour.\n")

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "age_window": [AGE_MIN, AGE_MAX],
    "percentiles": [P_LO, P_HI],
    "bounds_s_per_lap": [float(lo), float(hi)],
    "plausibility_ratio": ratio,
    "index": {f"{r['season']}|{r['event']}": round(r["index"], 4) for r in rows},
    "slopes_s_per_lap": {f"{r['season']}|{r['event']}": round(r["slope"], 5) for r in rows},
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nRapport : {OUT_MD}")
print(f"Index   : {OUT_JSON}")
