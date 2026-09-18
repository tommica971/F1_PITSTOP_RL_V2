"""
Configuration du pool de Grands Prix — Projet F1PitStopEnv
Phase 0.3 : pool validé sur données réelles (voir historique de conversation / dossier écrit).

Pilote étudié : Pierre Gasly (Alpine F1 Team), code FastF1 "GAS".

Chaque entrée du pool porte :
- season       : année de la saison FastF1
- event        : nom exact de l'événement (nomenclature FastF1 / TracingInsights)
- role         : rôle du GP dans le pipeline ML
                 "train_wet"  -> entraînement, météo variable
                 "train_dry"  -> entraînement, baseline sec
                 "test_wet"   -> test de généralisation, météo extrême jamais vue à l'entraînement
                 "test_reg"   -> test de généralisation, nouvelle réglementation 2026
- rain_pct     : % d'échantillons météo avec pluie détectée pendant la course (calculé Phase 0.3)
- known_issues : anomalies connues nécessitant un traitement spécifique en nettoyage (Phase 1.1)
"""

DRIVER_CODE = "GAS"
TEAM_NAME = "Alpine"

GP_POOL = [
    # --- Entraînement : météo variable ---
    {
        "season": 2025, "event": "British Grand Prix", "role": "train_wet",
        "rain_pct": 18.1, "known_issues": [],
    },
    {
        "season": 2025, "event": "Australian Grand Prix", "role": "train_wet",
        "rain_pct": 32.6, "known_issues": ["multiple_safety_car"],
    },
    {
        "season": 2024, "event": "Canadian Grand Prix", "role": "train_wet",
        "rain_pct": 27.4, "known_issues": ["safety_car"],
    },
    {
        "season": 2023, "event": "Dutch Grand Prix", "role": "train_wet",
        "rain_pct": 23.0, "known_issues": ["red_flag", "safety_car", "vsc"],
    },
    {
        # Ajouté Phase 3.5 bis : profil absent du pool jusqu'ici -- piste
        # humide au départ (démarrage sur INTERMEDIATE, humidité 92-95%)
        # mais qui NE REPLEUT JAMAIS sur toute la course (is_raining=False,
        # 0/58 tours). Les 3 autres GP train_wet démarrant sur pneu pluie
        # (British/Australian/Canadian) voient tous la pluie se confirmer
        # ensuite -- aucun n'enseigne "la piste peut rester sèche après un
        # départ humide". Identifié suite à l'échec de généralisation sur
        # Belgique 2025 (role test_wet, même profil) -- cf. notebook 07,
        # diagnostic 06bis/07. Vérifié sans anomalie de piste (0 SC/VSC/
        # drapeau rouge, rcm.json).
        "season": 2021, "event": "Turkish Grand Prix", "role": "train_wet",
        "rain_pct": 0.0, "known_issues": [],
    },

    # --- Entraînement : baseline sec ---
    {
        "season": 2023, "event": "Bahrain Grand Prix", "role": "train_dry",
        "rain_pct": 0.0, "known_issues": [],
    },
    {
        "season": 2023, "event": "Spanish Grand Prix", "role": "train_dry",
        "rain_pct": 0.0, "known_issues": [],
    },
    {
        "season": 2023, "event": "Abu Dhabi Grand Prix", "role": "train_dry",
        "rain_pct": 0.0, "known_issues": [],
    },

    # --- Test de généralisation ---
    {
        "season": 2025, "event": "Belgian Grand Prix", "role": "test_wet",
        "rain_pct": 45.7, "known_issues": ["red_flag"],  # volontairement NON filtré : c'est le but du test
    },
    {
        "season": 2026, "event": "Japanese Grand Prix", "role": "test_reg",
        "rain_pct": 0.0, "known_issues": [],
    },
]

# GP de réserve écarté du pool principal — Gasly DNF au tour 1, aucune donnée stratégique exploitable
EXCLUDED_GP = [
    {
        "season": 2024, "event": "British Grand Prix",
        "reason": "Gasly DNF tour 1 — 1 seul tour enregistré, aucune donnée stratégique exploitable. "
                  "Remplacé par 2025 British Grand Prix dans le pool train_wet.",
    },
]
