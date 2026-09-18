#!/usr/bin/env python3
"""
Etape 2 — Calibration de la degradation, en SECONDES
=====================================================

Pourquoi une regression multi-pilotes et pas mono-pilote :
    Sur un seul pilote, `tyre_age_laps` et `lap_number` avancent de 1 en 1
    ensemble a l'interieur d'un relais. Apres effets fixes de relais ils sont
    strictement colineaires : la matrice est de rang 1, et les deux
    coefficients ne sont PAS identifiables (d'ou les `nan` de H1bis).

    Sur les 20 pilotes, au tour 30 certains sont sur un pneu de 5 tours et
    d'autres de 25. Un effet fixe PAR TOUR absorbe alors tout ce qui est
    commun au peloton a cet instant -- carburant, evolution de piste, meteo,
    neutralisations -- et l'age du pneu redevient identifiable.

Modele estime, par GP :
    lap_time = a_pilote + b_tour + somme_c ( pente_c * age * 1{compose=c} ) + e

    a_pilote : effet fixe pilote (rythme voiture)
    b_tour   : effet fixe tour   (carburant + piste + meteo + neutralisations)
    pente_c  : DEGRADATION du compose c, en SECONDES PAR TOUR <- l'inconnue

Le terme carburant n'a plus besoin d'etre estime separement : il est absorbe
par b_tour. C'est exactement ce qui le rend non-reinitialisable par un arret,
contrairement a l'age du pneu.

La penalite de tour de sortie est estimee a part, sur les tours d'age <= 2
(exclus de la regression de pente), comme ecart au rythme attendu.

Sortie : data/processed/pace_calibration.json  +  rapport_calibration.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GHOSTS = DATA / "raw" / "all_drivers_dataset.parquet"
OUT_JSON = DATA / "processed" / "pace_calibration.json"
REPORT = ROOT / "rapport_calibration.md"

SLICKS = ["SOFT", "MEDIUM", "HARD"]
WETS = ["INTERMEDIATE", "WET"]
MIN_ROWS_PER_GP = 60
MIN_ROWS_PER_COMPOUND = 25

lines = ["# Calibration du modele de rythme (en secondes)\n"]


def emit(t=""):
    print(t)
    lines.append(t)


df = pd.read_parquet(GHOSTS)
need = {"season", "event", "driver", "lap_number", "lap_time_s",
        "tyre_compound", "tyre_age_laps", "stint_number"}
missing = need - set(df.columns)
if missing:
    sys.exit(
        f"ERREUR : colonnes manquantes dans all_drivers_dataset.parquet : {missing}\n"
        "La calibration multi-pilotes exige le compose et l'age du pneu pour TOUS\n"
        "les pilotes. Si ces colonnes n'existent que pour le pilote de reference,\n"
        "il faut d'abord etendre l'extraction (extract_tracinginsights.py) aux 20\n"
        "pilotes. Sans cela, la degradation n'est pas identifiable."
    )

emit(f"Donnees : {len(df):,} tours, {df['driver'].nunique()} pilotes, "
     f"{df.groupby(['season', 'event']).ngroups} GP\n")


def clean_subset(g: pd.DataFrame) -> pd.DataFrame:
    """Tours exploitables pour la regression.

    ORDRE CRITIQUE : les tours d'entree/sortie sont reperes AVANT le filtrage
    hors bande. Le tour d'arret est par construction hors bande ; si on filtre
    d'abord, le changement de relais se decale sur le tour d'age 2, qui se
    fait exclure a son tour -- et il ne reste plus aucun tour froid pour
    estimer la penalite (bug constate sur donnees de controle).
    """
    g = g.dropna(subset=["lap_time_s", "tyre_age_laps", "tyre_compound", "stint_number"])
    g = g[g["lap_time_s"] > 0].sort_values(["driver", "lap_number"])

    # Le diagnostic 04_diagnose_outlap.py montre que le temps de passage au
    # stand est reparti sur DEUX tours dans ces donnees :
    #   tour de sortie (age = 1)          : mediane 122.9% de la reference
    #   tour d'entree (dernier du relais) : mediane 105.2%, 25.6% > 115%
    # Les deux sont exclus. La penalite de pneu froid est estimee sur l'age 2
    # (mediane 101.0%), seul tour "froid" non contamine.
    new_stint = g.groupby("driver")["stint_number"].diff().fillna(0) != 0
    in_lap = new_stint.shift(-1, fill_value=False)
    g = g[~(new_stint | in_lap)]
    g = g[g["tyre_age_laps"] != 1]

    g = g[g["lap_number"] > 1]
    for c in ("is_safety_car", "is_vsc"):
        if c in g.columns:
            g = g[~g[c].astype("boolean").fillna(False).to_numpy(dtype=bool)]

    med = g["lap_time_s"].median()
    return g[(g["lap_time_s"] < 1.07 * med) & (g["lap_time_s"] > 0.93 * med)]


def _demean(cols, keys, n_iter=25):
    """Transformation "within" a deux dimensions par projections alternees.

    Retire simultanement les effets fixes (GP x tour) et (GP x pilote) sans
    construire les milliers d'indicatrices correspondantes. Apres cette
    transformation, une simple MCO sur les pentes donne le meme resultat
    qu'une regression a effets fixes complets (theoreme de Frisch-Waugh).

    Pourquoi c'est necessaire : estimer les pentes GP par GP puis prendre une
    mediane compare des composes observes sur des ENSEMBLES DE GP DIFFERENTS
    (SOFT sur 15 GP, MEDIUM sur 27). L'ordre SOFT > MEDIUM > HARD peut alors
    s'inverser par simple effet de composition. Ici tous les composes sont
    compares a l'interieur du meme GP et du meme tour de course.
    """
    out = cols.copy()
    for _ in range(n_iter):
        before = out.copy()
        for key in keys:
            means = pd.DataFrame(out).groupby(key, observed=True).transform("mean").to_numpy()
            out = out - means
        if np.abs(out - before).max() < 1e-9:
            break
    return out


def pooled_slopes(df: pd.DataFrame):
    """Pentes de degradation, estimees conjointement sur tous les GP."""
    g = df[df["tyre_age_laps"] > 2].copy()
    comps = [c for c in SLICKS + WETS
             if (g["tyre_compound"] == c).sum() >= MIN_ROWS_PER_COMPOUND]
    if not comps:
        return {}, np.nan, np.nan

    gp_lap = pd.factorize(pd.Series(list(zip(g["season"], g["event"], g["lap_number"]))))[0]
    gp_drv = pd.factorize(pd.Series(list(zip(g["season"], g["event"], g["driver"]))))[0]

    age = g["tyre_age_laps"].to_numpy(float) - 2
    cols = [g["lap_time_s"].to_numpy(float)]
    for c in comps:
        cols.append(age * (g["tyre_compound"] == c).to_numpy(float))
    M = _demean(np.column_stack(cols), [gp_lap, gp_drv])

    y, X = M[:, 0], M[:, 1:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return ({c: float(coef[i]) for i, c in enumerate(comps)},
            float(r2), float(resid.std()))


def cold_penalty_local(df: pd.DataFrame, slopes: dict):
    """Penalite de pneu froid, par contraste LOCAL a l'interieur du relais.

    On compare le tour d'age 2 aux tours d'age 3 a 5 du MEME relais, corriges
    de la degradation attendue. Contrairement au residu vis-a-vis de la droite
    globale, ce contraste est insensible a la courbure de la degradation : la
    droite ajustee sur les ages 3-30 et prolongee jusqu'a l'age 2 surestime le
    temps attendu des que la degradation accelere avec l'age, ce qui produisait
    une penalite negative -- physiquement impossible.

    Il est aussi insensible au carburant et a l'evolution de piste, les tours
    compares etant separes de 1 a 3 tours seulement.
    """
    out = {}
    ref_ages = (3, 4, 5)
    g = df[df["tyre_age_laps"].isin((2,) + ref_ages)]
    keys = ["season", "event", "driver", "stint_number"]
    for c in SLICKS + WETS:
        deltas = []
        sub = g[g["tyre_compound"] == c]
        slope = slopes.get(c, 0.05)
        for _, st in sub.groupby(keys, observed=True):
            cold = st.loc[st["tyre_age_laps"] == 2, "lap_time_s"]
            ref = st[st["tyre_age_laps"].isin(ref_ages)]
            if cold.empty or len(ref) < 2:
                continue
            # temps attendu a l'age 2 d'apres les tours de reference du relais
            expected = (ref["lap_time_s"] - slope * (ref["tyre_age_laps"] - 2)).median()
            deltas.append(float(cold.iloc[0]) - expected)
        if len(deltas) >= 20:
            out[c] = float(np.median(deltas))
        else:
            out[c] = None
    return out


def age_profile(df: pd.DataFrame):
    """Profil observe par age, en % de la mediane du GP -- revele la courbure."""
    g = df.copy()
    med = g.groupby(["season", "event"])["lap_time_s"].transform("median")
    g["ratio"] = 100 * g["lap_time_s"] / med
    rows = []
    for c in SLICKS:
        sub = g[g["tyre_compound"] == c]
        if len(sub) < 100:
            continue
        cells = []
        for lo, hi in ((2, 2), (3, 5), (6, 10), (11, 15), (16, 25), (26, 40)):
            s = sub.loc[sub["tyre_age_laps"].between(lo, hi), "ratio"]
            cells.append(f"{s.median():.2f}" if len(s) >= 20 else "-")
        rows.append((c, cells))
    return rows


clean = pd.concat([clean_subset(g) for _, g in df.groupby(["season", "event"])],
                  ignore_index=True)
emit(f"Tours exploitables apres nettoyage : {len(clean):,}\n")

emit("## Profil observe par age de pneu (% de la mediane du GP)\n")
emit("| compose | age 2 | 3-5 | 6-10 | 11-15 | 16-25 | 26-40 |")
emit("|---|---|---|---|---|---|---|")
for c, cells in age_profile(clean):
    emit(f"| {c} | " + " | ".join(cells) + " |")
emit("\nUn profil non lineaire justifie le contraste local pour le pneu froid "
     "(cf. cold_penalty_local).\n")

calib_slopes, r2, noise = pooled_slopes(clean)
emit("## Pentes de degradation retenues (estimation conjointe)\n")
emit(f"Regression unique sur {len(clean):,} tours, effets fixes (GP x tour) "
     f"et (GP x pilote). R2 = {r2:.3f}\n")
emit("| compose | pente (s/tour) | n tours | plausible ? |")
emit("|---|---|---|---|")
for c in SLICKS + WETS:
    if c not in calib_slopes:
        emit(f"| {c} | - | 0 | pas de donnees |")
        continue
    n = int((clean["tyre_compound"] == c).sum())
    lo, hi = (0.02, 0.20) if c in SLICKS else (0.01, 0.15)
    ok = "oui" if lo <= calib_slopes[c] <= hi else f"**HORS PLAGE ({lo}-{hi})**"
    emit(f"| {c} | {calib_slopes[c]:.4f} | {n} | {ok} |")

if all(c in calib_slopes for c in SLICKS):
    ordered = calib_slopes["SOFT"] > calib_slopes["MEDIUM"] > calib_slopes["HARD"]
    emit(f"\nOrdre SOFT > MEDIUM > HARD respecte : "
         f"**{'oui' if ordered else 'NON — a investiguer avant de continuer'}**\n")

outlap_raw = cold_penalty_local(clean, calib_slopes)

emit("## Penalite de pneu froid (age = 2)\n")
emit("Contraste LOCAL : tour d'age 2 compare aux tours d'age 3-5 du MEME "
     "relais, corriges de la degradation attendue. Insensible a la courbure "
     "de la degradation, au carburant et a l'evolution de piste.\n")
emit("_(ancienne methode, abandonnee : residu MEDIAN des tours d'age 2 par rapport au modele "
     "ajuste sur les tours chauds (effets fixes pilote + tour + pente). "
     "Estimer un coefficient dedie echouait : sur quelques dizaines de "
     "lignes, l'indicatrice devient quasi colineaire avec les effets fixes "
     "de tour et le coefficient explose. Les residus sont mutualises sur "
     "tous les GP.\n")
emit("> **Borne basse assumee.** Le tour de sortie (age = 1) porte le temps "
     "de passage au stand (mediane 122.9% de la reference, cf. "
     "04_diagnose_outlap.py) et ne permet pas d'isoler l'effet thermique. "
     "La penalite est donc estimee sur l'age 2 seul, ou le pneu est deja "
     "partiellement monte en temperature. La vraie penalite a l'age 1 est "
     "superieure. Consequence : le modele SOUS-ESTIME legerement le cout "
     "d'un arret, ce qui joue en faveur des strategies a arret. A citer "
     "comme limite dans le dossier.\n")
emit("| compose | penalite (s) | n tours | plausible ? |")
emit("|---|---|---|---|")

outlap = {}
for c in SLICKS + WETS:
    pen = outlap_raw.get(c)
    if pen is None:
        emit(f"| {c} | - | - | trop peu de relais exploitables |")
        continue
    outlap[c] = pen
    ok = "oui" if 0.2 <= pen <= 4.0 else "**HORS PLAGE (0.2-4.0 s)**"
    emit(f"| {c} | {pen:.3f} | - | {ok} |")

noise = float(noise) if noise == noise else 0.5
emit(f"\n## Bruit par tour\n\nEcart-type residuel median : **{noise:.3f} s**\n")
emit("Remplace `noise_std_units = 0.15` exprime en unites de gp_std, qui "
     "valait jusqu'a +/- 2.9 s sur les GP a gp_std eleve.\n")

calib = {
    "degradation_s_per_lap": calib_slopes,
    "out_lap_penalty_s": outlap,
    "noise_std_s": round(noise, 4),
    "pit_stop_cost_mean_s": 23.23,
    "pit_stop_cost_std_s": 5.38,
    "method": "effets fixes pilote + tour, multi-pilotes, par GP ; "
              "l'effet carburant est absorbe par l'effet fixe de tour",
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8")
REPORT.write_text("\n".join(lines), encoding="utf-8")
emit(f"\nCalibration : {OUT_JSON}\nRapport : {REPORT}")
print("\nEtape suivante : python scripts/03_check_breakeven.py")
