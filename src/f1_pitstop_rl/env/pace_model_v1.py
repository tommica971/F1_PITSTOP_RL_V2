"""
Phase 2.1 — Modèle de rythme calibré (pace_model)
====================================================
Traduit les résultats calibrés en Phase 1.3/1.4 (notebooks 01, 02, 03) en une
fonction de prédiction du temps au tour de l'agent. Séparé de F1PitStopEnv
pour rester testable isolément (Phase 2.3).

Formule :
    lap_time = gp_baseline + team_offset + degradation(compound, age)
               + weather_mismatch_penalty(compound, is_raining) + bruit

Provenance de chaque terme -- IMPORTANT pour la traçabilité (dossier écrit) :
    - gp_baseline, gp_std     : calculés depuis les données réelles du GP
                                 (médiane/écart-type des tours exploitables)
    - team_offset             : calibré, notebook 02 (rythme Alpine vs peloton,
                                 saison 2024)
    - degradation(compound)   : calibré, notebook 03 section 4 (pentes hors
                                 tour de sortie, effets fixes par relais)
    - out_lap_penalty         : calibré, notebook 03 (pic observé tours 1-2)
    - weather_mismatch_penalty: NON calibré sur données réelles -- valeurs
                                 assumées, documentées comme telles, à affiner
                                 si le temps le permet (Phase 2.2/perspective)
"""

from dataclasses import dataclass, field

import numpy as np

# --- Valeurs calibrées (notebooks 01/02/03) ---------------------------------

# Pente de dégradation par composé, en unités perf_target (écarts-types du GP)
# par tour, hors tour de sortie (age > 2). Source : 03_analyses_avancees.ipynb,
# section 4, régression à effets fixes par relais.
DEGRADATION_SLOPE = {
    "SOFT": 0.00361,
    "MEDIUM": 0.00584,
    "HARD": 0.00670,
    "INTERMEDIATE": 0.00298,
    "WET": 0.00298,  # WET non observé isolément dans le pool -- valeur INTER reprise par défaut, à documenter comme hypothèse
}

# Pénalité de tour de sortie (pneu froid), en unités perf_target, observée en
# moyenne sur le bin (0,2] tours d'âge -- source : 03_analyses_avancees.ipynb
OUT_LAP_PENALTY = {
    "SOFT": 0.75,
    "MEDIUM": 0.70,
    "HARD": 1.48,
    "INTERMEDIATE": 0.16,
    "WET": 0.16,
}

# Décalage d'équipe Alpine vs médiane peloton -- RETIRÉ (double comptage) :
# GPPaceContext.gp_baseline_s est déjà calculé comme la médiane des PROPRES
# tours de Gasly sur CE GP précis (cf. f1_pitstop_env._load_gp_data), donc il
# encode déjà toute sa compétitivité réelle ce jour-là. Ajouter un offset
# supplémentaire ici pénalisait l'agent deux fois pour le même effet -- bug
# identifié et corrigé via un test de sanité (rejeu de la stratégie réelle de
# Gasly à Bahreïn 2023, comparé à son résultat réel P9).
ALPINE_TEAM_OFFSET = 0.0

# Coût d'un arrêt au stand -- source : 03_analyses_avancees.ipynb section 2
PIT_STOP_COST_MEAN_S = 23.23
PIT_STOP_COST_STD_S = 5.38

# --- Valeurs NON calibrées (hypothèses documentées, à affiner) -------------
# Pénalité relative si le composé est inadapté aux conditions piste. Pas de
# données réelles suffisantes dans le pool pour calibrer précisément un
# "mauvais choix de pneu" (peu d'occurrences observées) -- valeurs choisies
# pour refléter l'ordre de grandeur qualitatif connu du sport (aquaplanage
# sur slick en forte pluie = très pénalisant, surchauffe de pneu pluie sur
# piste sèche = pénalisant mais moins dramatique), à recalibrer si des
# données suffisantes sont trouvées.
SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

WEATHER_MISMATCH_PENALTY = {
    ("slick_on_wet_track", "heavy"): 3.5,   # slick sur piste très mouillée : très pénalisant (aquaplaning)
    ("slick_on_wet_track", "light"): 1.2,   # slick sur piste légèrement humide : pénalisant
    ("wet_tyre_on_dry_track", "any"): 0.9,  # pneu pluie sur piste sèche : surchauffe, pénalisant modéré
}


@dataclass
class GPPaceContext:
    """Contexte de calibration propre à un GP (calculé depuis les données réelles)."""
    gp_baseline_s: float   # médiane des temps au tour exploitables du GP
    gp_std_s: float        # écart-type des temps au tour exploitables du GP
    race_total_laps: int


def predict_lap_time(
    context: GPPaceContext,
    tyre_compound: str,
    tyre_age_laps: int,
    is_raining: bool,
    rain_intensity_recent: float,
    rng: np.random.Generator,
    noise_std_units: float = 0.15,
) -> float:
    """
    Prédit le temps au tour de l'agent (en secondes) pour un tour donné.

    Args:
        context: contexte de calibration du GP en cours
        tyre_compound: composé actuel de l'agent
        tyre_age_laps: âge du train de pneus actuel (tours)
        is_raining: pluie détectée au tour courant (donnée exogène historique)
        rain_intensity_recent: proxy de densité de pluie récente (0-1, cf.
            notebook 01 -- moyenne glissante sur les derniers tours), utilisé
            pour moduler la pénalité de mauvais choix de pneu
        rng: générateur aléatoire (pour le bruit stochastique)
        noise_std_units: écart-type du bruit, en unités perf_target

    Returns:
        Temps au tour prédit, en secondes.
    """
    perf_target = ALPINE_TEAM_OFFSET

    if tyre_age_laps <= 2:
        perf_target += OUT_LAP_PENALTY.get(tyre_compound, 0.5)
    else:
        perf_target += DEGRADATION_SLOPE.get(tyre_compound, 0.005) * (tyre_age_laps - 2)

    # Pénalité de mauvais choix de pneu (hypothèse documentée, non calibrée)
    if is_raining and tyre_compound in SLICK_COMPOUNDS:
        severity = "heavy" if rain_intensity_recent >= 0.5 else "light"
        perf_target += WEATHER_MISMATCH_PENALTY[("slick_on_wet_track", severity)]
    elif not is_raining and tyre_compound in WET_COMPOUNDS and rain_intensity_recent < 0.1:
        perf_target += WEATHER_MISMATCH_PENALTY[("wet_tyre_on_dry_track", "any")]

    perf_target += rng.normal(0, noise_std_units)

    lap_time_s = context.gp_baseline_s + perf_target * context.gp_std_s
    return float(max(lap_time_s, context.gp_baseline_s * 0.5))  # garde-fou anti-valeur absurde


def sample_pit_stop_cost(rng: np.random.Generator) -> float:
    """Coût stochastique d'un arrêt au stand, en secondes (calibré, notebook 03)."""
    cost = rng.normal(PIT_STOP_COST_MEAN_S, PIT_STOP_COST_STD_S)
    return float(max(cost, 10.0))  # borne physique basse (cf. filtre 10-60s du calibrage)
