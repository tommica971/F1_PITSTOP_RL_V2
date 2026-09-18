#!/usr/bin/env python3
"""
Diagnostic de l'environnement F1PitStopEnv — deux hypotheses a confirmer
========================================================================

A placer dans F1_PITSTOP_RL/scripts/ et lancer depuis la racine du projet :
    python scripts/diagnose_env.py

H1 -- La degradation pneu est trop faible pour rendre un arret rentable.
      On calcule le point mort d'un arret (gain de degradation vs cout de
      pit + tours de sortie) avec les constantes ACTUELLES, GP par GP.
      Si le point mort n'est jamais atteint, le 0-stop est l'optimum de
      l'environnement et l'agent a raison.

H1bis -- La pente calibree confond degradation pneu et allegement carburant.
      On re-estime la pente en separant l'age du pneu (reinitialisable par
      un arret) du numero de tour (effet carburant, NON reinitialisable).
      Si le coefficient d'age remonte nettement une fois le carburant
      controle, c'est la confirmation.

H2 -- Sous SC/VSC, l'agent recoit un temps au tour constant (baseline x
      multiplicateur) alors que les fantomes rejouent leurs vrais temps.
      On compare les deux pour chiffrer le gain (ou la perte) gratuit par
      tour d'anomalie.

Sortie : tableaux console + diagnostic_env.md pour le dossier.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "env"))
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "config"))

from pace_model import (  # noqa: E402
    DEGRADATION_SLOPE, OUT_LAP_PENALTY, PIT_STOP_COST_MEAN_S,
)
from gp_pool_config import GP_POOL, DRIVER_CODE  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "diagnostic_env.md"

features = pd.read_parquet(DATA / "processed" / "features_dataset.parquet")
ghosts = pd.read_parquet(DATA / "raw" / "all_drivers_dataset.parquet")

lines = ["# Diagnostic de l'environnement F1PitStopEnv\n"]


def section(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    lines.append(f"\n## {title}\n")


def emit(text):
    print(text)
    lines.append(text)


# ---------------------------------------------------------------------------
# H1 : point mort d'un arret avec les constantes actuelles
# ---------------------------------------------------------------------------
section("H1 - Un arret est-il rentable dans l'environnement actuel ?")

emit(f"Cout d'un arret : {PIT_STOP_COST_MEAN_S:.2f} s (moyenne calibree)\n")
emit("| GP | gp_std (s) | degr. HARD (s/tour) | degr. cumulee 40 tours (s) | "
     "surcout tours de sortie (s) | cout total d'un arret (s) | rentable ? |")
emit("|---|---|---|---|---|---|---|")

rows = []
for gp in GP_POOL:
    sub = features[(features["season"] == gp["season"])
                   & (features["event"] == gp["event"])]
    usable = sub[sub["usable_for_reward"]]
    if usable.empty:
        continue
    std = float(usable["lap_time_s"].std())

    slope_s = DEGRADATION_SLOPE["HARD"] * std          # s par tour d'age
    cum40 = sum(slope_s * (a - 2) for a in range(3, 41))  # s perdues sur 40 tours

    # ce que coute un arret : pit + 2 tours de sortie, MOINS la degradation
    # qu'on aurait subie sur ces 2 tours si on etait reste en piste
    outlap_extra = 2 * (OUT_LAP_PENALTY["HARD"] * std - slope_s * 38)
    total_cost = PIT_STOP_COST_MEAN_S + max(outlap_extra, 0.0)
    ok = "OUI" if cum40 > total_cost else "**NON**"

    emit(f"| {gp['season']} {gp['event']} | {std:.2f} | {slope_s:.4f} | "
         f"{cum40:.1f} | {outlap_extra:.1f} | {total_cost:.1f} | {ok} |")
    rows.append((gp, std, slope_s, cum40, total_cost, ok))

n_no = sum(1 for r in rows if r[5] != "OUI")
emit(f"\n**{n_no}/{len(rows)} GP ou aucun arret n'est rentable.** "
     "Si ce nombre est eleve, le 0-stop est l'optimum de l'environnement "
     "et l'agent ne peut pas apprendre autre chose.\n")

emit("Ordre de grandeur reel en F1 : 0.05 a 0.12 s/tour de degradation slick. "
     "Comparer a la colonne 'degr. HARD (s/tour)' ci-dessus.\n")


# ---------------------------------------------------------------------------
# H1bis : degradation pneu vs allegement carburant
# ---------------------------------------------------------------------------
section("H1bis - La pente calibree confond-elle pneu et carburant ?")

emit("Regression par GP : lap_time ~ tyre_age + lap_number, avec effets fixes "
     "de relais (on retire la moyenne de chaque relais).\n")
emit("- coef tyre_age  = degradation pneu, REINITIALISABLE par un arret\n"
     "- coef lap_number = allegement carburant, NON reinitialisable\n")
emit("\n| GP | pente nette (s/tour) | coef pneu (s/tour) | coef carburant (s/tour) | n tours |")
emit("|---|---|---|---|---|")

net_all, tyre_all, fuel_all = [], [], []
for gp in GP_POOL:
    sub = features[(features["season"] == gp["season"])
                   & (features["event"] == gp["event"])
                   & (features["usable_for_reward"])].copy()
    needed = {"lap_time_s", "tyre_age_laps", "lap_number", "stint_number"}
    if sub.empty or not needed.issubset(sub.columns) or len(sub) < 15:
        continue

    sub = sub.dropna(subset=list(needed))
    sub = sub[sub["tyre_age_laps"] > 2]           # hors tours de sortie
    if sub["stint_number"].nunique() < 2 or len(sub) < 15:
        continue

    # effets fixes de relais : on centre dans chaque relais
    g = sub.groupby("stint_number")
    y = (sub["lap_time_s"] - g["lap_time_s"].transform("mean")).to_numpy()
    a = (sub["tyre_age_laps"] - g["tyre_age_laps"].transform("mean")).to_numpy()
    n = (sub["lap_number"] - g["lap_number"].transform("mean")).to_numpy()

    # pente nette : age seul (ce que fait la calibration actuelle)
    net = np.polyfit(a, y, 1)[0] if np.std(a) > 1e-9 else np.nan

    # modele a deux regresseurs
    X = np.column_stack([a, n])
    if np.linalg.matrix_rank(X) < 2:
        tyre = fuel = np.nan
    else:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        tyre, fuel = coef

    emit(f"| {gp['season']} {gp['event']} | {net:.4f} | {tyre:.4f} | "
         f"{fuel:.4f} | {len(sub)} |")
    for lst, v in ((net_all, net), (tyre_all, tyre), (fuel_all, fuel)):
        if not np.isnan(v):
            lst.append(v)

if tyre_all:
    emit(f"\n**Mediane** : pente nette {np.median(net_all):.4f} s/tour | "
         f"pneu {np.median(tyre_all):.4f} s/tour | "
         f"carburant {np.median(fuel_all):.4f} s/tour\n")
    ratio = np.median(tyre_all) / np.median(net_all) if np.median(net_all) else float("nan")
    emit(f"Rapport pneu / nette = {ratio:.1f}x. "
         "Un rapport nettement superieur a 1 confirme que la calibration "
         "actuelle sous-estime la degradation en absorbant l'effet carburant.\n")
    emit("Attention : `tyre_age` et `lap_number` sont fortement correles a "
         "l'interieur d'un relais. Les deux coefficients ne sont separables "
         "que si les relais couvrent des phases de course differentes. "
         "Verifier la colinearite avant de conclure (voir ci-dessous).\n")
    # diagnostic de colinearite, honnete
    allsub = features[features["usable_for_reward"]].dropna(
        subset=["tyre_age_laps", "lap_number", "stint_number"])
    if not allsub.empty:
        gg = allsub.groupby(["season", "event", "stint_number"])
        aa = (allsub["tyre_age_laps"] - gg["tyre_age_laps"].transform("mean"))
        nn = (allsub["lap_number"] - gg["lap_number"].transform("mean"))
        r = np.corrcoef(aa, nn)[0, 1]
        emit(f"Correlation intra-relais age/tour : r = {r:.3f}. "
             f"{'Separation fiable.' if abs(r) < 0.95 else 'TROP COLINEAIRE - les deux coefficients ne sont pas identifiables, il faut une autre source (donnees multi-pilotes).'}\n")


# ---------------------------------------------------------------------------
# H2 : temps au tour de l'agent vs peloton sous SC/VSC
# ---------------------------------------------------------------------------
section("H2 - L'agent gagne-t-il du temps gratuit sous SC/VSC ?")

VSC_MULT, SC_MULT = 1.03, 1.40
emit(f"Multiplicateurs de l'env : SC = {SC_MULT}, VSC = {VSC_MULT}\n")
emit("| GP | type | n tours | mult. reel peloton | mult. env | "
     "ecart par tour (s) | ecart total (s) |")
emit("|---|---|---|---|---|---|---|")

total_free = 0.0
for gp in GP_POOL:
    sub = features[(features["season"] == gp["season"])
                   & (features["event"] == gp["event"])]
    if sub.empty:
        continue
    usable = sub[sub["usable_for_reward"]]
    if usable.empty:
        continue
    baseline = float(usable["lap_time_s"].median())

    gsub = ghosts[(ghosts["season"] == gp["season"])
                  & (ghosts["event"] == gp["event"])
                  & (ghosts["driver"] != DRIVER_CODE)]

    for kind, col, mult in (("SC", "is_safety_car", SC_MULT),
                            ("VSC", "is_vsc", VSC_MULT)):
        if col not in sub.columns:
            continue
        laps = sub.loc[sub[col].astype(bool), "lap_number"].astype(int).tolist()
        if not laps:
            continue
        field = gsub[gsub["lap_number"].isin(laps)]
        field = field[field["lap_time_s"] > 0]
        if field.empty:
            continue
        real_mult = float(field["lap_time_s"].median()) / baseline
        per_lap = (real_mult - mult) * baseline      # >0 = temps gratuit pour l'agent
        total = per_lap * len(laps)
        total_free += total
        emit(f"| {gp['season']} {gp['event']} | {kind} | {len(laps)} | "
             f"{real_mult:.3f} | {mult:.3f} | {per_lap:+.2f} | {total:+.1f} |")

emit(f"\n**Total sur le pool d'entrainement : {total_free:+.1f} s** offerts "
     "(positif) ou retires (negatif) a l'agent, sans lien avec ses decisions.\n")
emit("Rappel : 1 position vaut typiquement 2 a 6 s dans un peloton F1. "
     "Un ecart de quelques dizaines de secondes deplace l'agent de plusieurs "
     "rangs sans qu'aucune strategie ne soit en cause.\n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"\n-> {OUT}")
