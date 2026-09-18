"""
Phase 4.1 — Modele de rythme calibre, VERSION 2 (en secondes)
==============================================================

CHANGEMENT STRUCTURANT par rapport a la v1 : toutes les constantes sont
exprimees en SECONDES, plus en unites de `gp_std`.

Motivation (diagnostic_env.md) : la v1 calculait
    lap_time = gp_baseline + perf_target * gp_std
ou `perf_target` agregeait degradation, penalite de tour de sortie et bruit,
tous exprimes en ecarts-types du GP. Or `gp_std` mesure par construction la
DISPERSION des temps au tour du GP, qui depend surtout de la quantite de
tours perturbes (safety car, arrets) que le filtre `usable_for_reward`
laissait passer -- pas de la physique des pneus. Resultat : sur le pool,
gp_std allait de 3.3 s (Turquie 2021) a 22.5 s (Australie 2025), et la meme
constante "calibree" signifiait des choses differentes selon le GP :

    penalite de tour de sortie HARD (1.48 unite) :  4.9 s en Turquie,
                                                   28.3 s en Australie
    bruit par tour (0.15 unite)                 : +/-0.50 s en Turquie,
                                                  +/-2.9 s en Australie

Un bruit de +/-2.9 s par tour produit une marche aleatoire de ~20 s sur une
course, soit +/-5 positions de pur hasard : la position finale de l'agent
etait en grande partie un tirage au sort, non correle a ses decisions.

En v2, `gp_std` ne sert plus qu'a un seul usage legitime : documenter la
dispersion du GP. Il n'intervient plus dans le calcul du temps au tour.

Formule v2 :
    lap_time = gp_baseline
             + degradation_s[compound] * max(age - 2, 0)
             + out_lap_penalty_s[compound]   si age <= 2
             + weather_mismatch_penalty_s
             + N(0, noise_std_s)

Provenance :
    - degradation_s, out_lap_penalty_s, noise_std_s : calibres par
      02_calibrate_pace.py (regression a effets fixes pilote + tour sur les
      20 pilotes). L'effet carburant est absorbe par l'effet fixe de tour,
      ce qui garantit que `degradation_s` mesure bien un effet d'AGE DU PNEU
      (reinitialisable par un arret) et non un effet de temps de course
      (non reinitialisable) -- c'etait la confusion de la v1.
    - pit_stop_cost : inchange (notebook 03, deja en secondes)
    - weather_mismatch_penalty_s : toujours NON calibre, hypothese assumee,
      mais desormais en secondes donc constante d'un GP a l'autre.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIB_PATH = (Path(__file__).resolve().parents[3] / "data" / "processed"
              / "pace_calibration.json")

# Valeurs de repli si le fichier de calibration est absent. Ordres de grandeur
# publics F1 -- utilisables pour un test, PAS pour un resultat du dossier.
_FALLBACK = {
    "degradation_s_per_lap": {"SOFT": 0.095, "MEDIUM": 0.070, "HARD": 0.050,
                              "INTERMEDIATE": 0.040, "WET": 0.040},
    "out_lap_penalty_s": {"SOFT": 0.90, "MEDIUM": 1.00, "HARD": 1.30,
                          "INTERMEDIATE": 0.60, "WET": 0.60},
    "noise_std_s": 0.45,
    "pit_stop_cost_mean_s": 23.23,
    "pit_stop_cost_std_s": 5.38,
}


def _load_calibration() -> dict:
    if not CALIB_PATH.exists():
        import warnings
        warnings.warn(
            f"pace_calibration.json introuvable ({CALIB_PATH}) -- valeurs de "
            "repli utilisees. Lancer scripts/02_calibrate_pace.py avant tout "
            "entrainement destine au dossier.", RuntimeWarning, stacklevel=2)
        return dict(_FALLBACK)
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    for key, default in _FALLBACK.items():
        if key not in calib:
            calib[key] = default
        elif isinstance(default, dict):
            calib[key] = {**default, **calib[key]}   # complete les composes manquants
    return calib


_CALIB = _load_calibration()

DEGRADATION_S = _CALIB["degradation_s_per_lap"]
OUT_LAP_PENALTY_S = _CALIB["out_lap_penalty_s"]
NOISE_STD_S = _CALIB["noise_std_s"]
PIT_STOP_COST_MEAN_S = _CALIB["pit_stop_cost_mean_s"]
PIT_STOP_COST_STD_S = _CALIB["pit_stop_cost_std_s"]

ALPINE_TEAM_OFFSET_S = 0.0  # cf. v1 : gp_baseline encode deja le rythme reel

SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

# Hypotheses NON calibrees, desormais en secondes (v1 : en unites de gp_std,
# donc variables d'un GP a l'autre d'un facteur 7).
WEATHER_MISMATCH_PENALTY_S = {
    ("slick_on_wet_track", "heavy"): 8.0,
    ("slick_on_wet_track", "light"): 3.0,
    ("wet_tyre_on_dry_track", "any"): 2.5,
}


@dataclass
class GPPaceContext:
    """Contexte de calibration d'un GP.

    `gp_std_s` est conserve pour la tracabilite et le diagnostic, mais
    N'INTERVIENT PLUS dans predict_lap_time (cf. en-tete du module).
    """
    gp_baseline_s: float
    gp_std_s: float
    race_total_laps: int


def predict_lap_time(
    context: GPPaceContext,
    tyre_compound: str,
    tyre_age_laps: int,
    is_raining: bool,
    rain_intensity_recent: float,
    rng: np.random.Generator,
    noise_std_s: float | None = None,
) -> float:
    """Temps au tour predit de l'agent, en secondes."""
    delta_s = ALPINE_TEAM_OFFSET_S

    if tyre_age_laps <= 2:
        delta_s += OUT_LAP_PENALTY_S.get(tyre_compound, 1.0)
    else:
        delta_s += DEGRADATION_S.get(tyre_compound, 0.06) * (tyre_age_laps - 2)

    if is_raining and tyre_compound in SLICK_COMPOUNDS:
        severity = "heavy" if rain_intensity_recent >= 0.5 else "light"
        delta_s += WEATHER_MISMATCH_PENALTY_S[("slick_on_wet_track", severity)]
    elif (not is_raining and tyre_compound in WET_COMPOUNDS
          and rain_intensity_recent < 0.1):
        delta_s += WEATHER_MISMATCH_PENALTY_S[("wet_tyre_on_dry_track", "any")]

    std = NOISE_STD_S if noise_std_s is None else noise_std_s
    delta_s += rng.normal(0, std)

    return float(max(context.gp_baseline_s + delta_s, context.gp_baseline_s * 0.5))


def sample_pit_stop_cost(rng: np.random.Generator) -> float:
    """Cout stochastique d'un arret, en secondes (calibre, notebook 03)."""
    return float(max(rng.normal(PIT_STOP_COST_MEAN_S, PIT_STOP_COST_STD_S), 10.0))


def degradation_cost_s(compound: str, age_start: int, n_laps: int) -> float:
    """Secondes perdues en restant `n_laps` tours de plus sur un pneu.

    Utilitaire de diagnostic (cf. 03_check_breakeven.py) : permet de comparer
    directement le cout de rester en piste au cout d'un arret, sans simuler.
    """
    slope = DEGRADATION_S.get(compound, 0.06)
    return float(sum(slope * max(age_start + i - 2, 0) for i in range(n_laps)))
