"""
Phase 2.2 — Fonction de récompense (équivalent-temps)
========================================================
Extrait de F1PitStopEnv.step() pour être testable isolément (Phase 2.3) et
pour centraliser la traçabilité des valeurs calibrées.

Principe (Phase 0.2) : récompense dense, en équivalent-temps (1 seconde
perdue = -1.0 point), combinant :
    1. Temps perdu/gagné ce tour vs la référence du GP (inclut le coût de pit
       stop calibré, cf. pace_model.py)
    2. Delta de position (gain/perte de rang), pondéré par l'écart TYPIQUE
       entre deux positions adjacentes -- calibré, pas arbitraire (cf. ci-dessous)
    3. Pénalité d'air sale (proximité prolongée d'un rival devant)
    4. Bonus de position finale, OU pénalité de disqualification si la règle
       des 2 composés (Article 30.7) n'est pas respectée

Provenance de chaque constante :
    - POSITION_DELTA_WEIGHT_S : médiane de `gap_avant` observée sur le pool
      RL (features_dataset.parquet, Phase 1.3) = 1.868s. Interprétation :
      gagner une position "vaut" en moyenne l'écart typique qui sépare deux
      voitures consécutives sur ce pool -- ancrage réel plutôt qu'arbitraire.
    - DIRTY_AIR_THRESHOLD_S, DIRTY_AIR_PENALTY : NON calibrés sur données
      réelles (pas de mesure directe de perte d'appui aérodynamique dans les
      données disponibles) -- hypothèses documentées, ordre de grandeur
      qualitatif raisonnable, à affiner si des données de perte de temps en
      sillage deviennent disponibles.
    - POINTS_TABLE / POINTS_SCALE (Phase 3.4, remplace l'ancien bonus
      linéaire (21-position)*poids) : barème de points réel du championnat
      pilotes F1 (P1=25 ... P10=1, P11+ =0). Le bonus terminal reflète
      maintenant qu'en réalité, finir P11 ou P20 ne rapporte STRICTEMENT
      RIEN de plus au classement pilotes -- contrairement au barème linéaire
      précédent qui donnait à l'agent un gradient artificiel entre P11 et
      P20. Le terme dense par tour (position_delta_component) N'EST PAS
      supprimé : il reste le seul signal d'apprentissage exploitable hors du
      top 10 (sans lui, l'agent perdrait tout gradient tant qu'il n'atteint
      pas la P10, notamment en début d'entraînement). POINTS_SCALE est un
      choix de modélisation ASSUMÉ, pas calibré sur données réelles (il
      n'existe pas de "taux de change" points<->secondes dans la réalité) :
      fixé à 4.0 pour que le bonus maximal (P1 : 25*4=100) reste égal à
      l'ancien bonus max ((21-1)*5.0=100), pour rester cohérent en ordre de
      grandeur avec le reste de la récompense (cf. smoke tests Phase 2.1).
    - DISQUALIFICATION_PENALTY : calibré pour excéder strictement le bonus de
      position finale maximal, afin qu'une stratégie 1-composé ne soit JAMAIS
      compétitive avec une stratégie conforme, même en cas de victoire de
      facto sur la piste (Article 30.7 -- une disqualification annule le
      classement, cf. retour utilisateur Phase 2.1). Invariant toujours vrai
      après Phase 3.4 : 150 > max(POINTS_TABLE)*POINTS_SCALE = 25*4 = 100.
"""

from dataclasses import dataclass

# Cf. en-tête -- médiane gap_avant, features_dataset.parquet (Phase 1.3)
POSITION_DELTA_WEIGHT_S = 1.868

# Hypothèses non calibrées (documentées) -- cf. en-tête
DIRTY_AIR_THRESHOLD_S = 1.0
DIRTY_AIR_PENALTY = 0.10

# Ancien bonus linéaire -- conservé pour référence/comparaison dans le
# dossier écrit (Phase 3.1 vs 3.4), plus utilisé dans le calcul (Phase 3.4).
FINAL_POSITION_BONUS_WEIGHT = 5.0

# Barème de points réel du championnat pilotes F1 (Phase 3.4) -- cf. en-tête.
POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
POINTS_SCALE = 4.0  # hypothèse assumée, non calibrée -- cf. en-tête

DISQUALIFICATION_PENALTY = 150.0  # > max(POINTS_TABLE)*POINTS_SCALE = 25*4 = 100

# Pénalité progressive de non-conformité (Phase 3.2, ajoutée suite à un taux
# de DSQ ~40% observé même après avoir rendu la regle observable -- la seule
# pénalité terminale est trop difficile à assigner sur un horizon de 50-70
# tours pour les algorithmes testés avec un budget d'entraînement limité).
# Commence à s'appliquer dans les derniers COMPLIANCE_WARNING_LAPS tours de
# la course si un seul composé a été utilisé, avec une intensité croissante
# à mesure que la fin de course approche -- donne un gradient d'apprentissage
# exploitable AVANT le couperet final, plutôt qu'un signal unique et tardif.
COMPLIANCE_WARNING_LAPS = 15
COMPLIANCE_WARNING_MAX_PENALTY = 2.0


@dataclass
class RewardBreakdown:
    """Décomposition de la récompense d'un tour, pour diagnostic/tests (Phase 2.3)."""
    time_component: float
    position_delta_component: float
    dirty_air_component: float
    compliance_component: float
    terminal_component: float
    total: float


def compute_step_reward(
    lap_time_s: float,
    gp_baseline_s: float,
    delta_position: int,
    gap_avant_s: float,
    is_terminal: bool,
    own_final_position: int,
    is_invalid_strategy: bool,
    tours_restants: int = 999,
    n_compounds_used: int = 2,
    race_had_rain: bool = True,
) -> RewardBreakdown:
    """
    Calcule la récompense d'un tour.

    Args:
        lap_time_s: temps au tour réalisé par l'agent (inclut coût de pit
            stop éventuel, déjà calculé par pace_model.py)
        gp_baseline_s: temps de référence du GP (médiane des tours réels
            exploitables de l'agent sur ce GP, cf. GPPaceContext)
        delta_position: changement de position ce tour (positif = gain de
            rang, négatif = perte de rang)
        gap_avant_s: écart (secondes) avec le rival directement devant, APRÈS
            le tour -- utilisé pour la pénalité d'air sale
        is_terminal: True si dernier tour de la course
        own_final_position: position finale (pertinent seulement si is_terminal)
        is_invalid_strategy: True si la règle des 2 composés (Article 30.7)
            n'est pas respectée (course sèche, un seul composé utilisé)
        tours_restants: tours restants avant la fin de la course (pour le
            signal de conformité progressif)
        n_compounds_used: nombre de composés distincts utilisés jusqu'ici
        race_had_rain: True si la course a connu de la pluie (règle levée)

    Returns:
        RewardBreakdown avec le détail par composante et le total.
    """
    time_component = -(lap_time_s - gp_baseline_s)
    position_delta_component = delta_position * POSITION_DELTA_WEIGHT_S

    dirty_air_component = 0.0
    if gap_avant_s < DIRTY_AIR_THRESHOLD_S:
        dirty_air_component = -DIRTY_AIR_PENALTY

    compliance_component = 0.0
    if (
        not race_had_rain
        and n_compounds_used < 2
        and not is_terminal
        and tours_restants <= COMPLIANCE_WARNING_LAPS
    ):
        urgency = (COMPLIANCE_WARNING_LAPS - tours_restants) / COMPLIANCE_WARNING_LAPS
        compliance_component = -urgency * COMPLIANCE_WARNING_MAX_PENALTY

    terminal_component = 0.0
    if is_terminal:
        if is_invalid_strategy:
            terminal_component = -DISQUALIFICATION_PENALTY
        else:
            # Phase 3.4 : barème de points réel (P1-P10) au lieu du bonus
            # linéaire -- finir P11 ou P20 rapporte 0 dans les deux cas,
            # comme au vrai championnat pilotes (cf. en-tête du module).
            terminal_component = POINTS_TABLE.get(own_final_position, 0) * POINTS_SCALE

    total = (
        time_component + position_delta_component + dirty_air_component
        + compliance_component + terminal_component
    )

    return RewardBreakdown(
        time_component=time_component,
        position_delta_component=position_delta_component,
        dirty_air_component=dirty_air_component,
        compliance_component=compliance_component,
        terminal_component=terminal_component,
        total=total,
    )
