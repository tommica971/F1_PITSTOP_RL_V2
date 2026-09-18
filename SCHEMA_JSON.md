# Schéma JSON — Dashboard F1_PITSTOP_RL

Trois fichiers JSON pré-calculés, générés par `run_season_inference.py`.
À placer dans `F1_PITSTOP_RL/dashboard/data/` :

- `season_2025.json`     → alimente l'onglet Comparatif Saison
- `race_<GP>_2025.json`  → un par GP, alimente l'onglet Command Center (tour par tour)
- `undercut_sim.json`    → alimente l'onglet Simulateur Undercut

---

## 1. `season_2025.json`

```json
{
  "driver": "Pierre Gasly",
  "team": "Alpine F1 Team",
  "season": 2025,
  "model": "a2c_extended_pool_5000k",
  "races": [
    {
      "round": 1,
      "gp_name": "Australie",
      "date": "2025-03-16",
      "seen_in_training": true,
      "real": {
        "finish_position": 11,
        "points": 0,
        "n_pitstops": 2,
        "compounds_used": ["MEDIUM", "HARD"]
      },
      "agent": {
        "finish_position": 9,
        "points": 2,
        "n_pitstops": 2,
        "compounds_used": ["MEDIUM", "HARD"],
        "mean_decision_confidence": 0.81
      },
      "delta_points": 2
    }
  ],
  "summary": {
    "real_total_points": 20,
    "agent_total_points": 34,
    "real_avg_position": 12.4,
    "agent_avg_position": 10.1,
    "races_improved": 14,
    "races_worse": 3,
    "races_equal": 7,
    "unseen_races_only": {
      "real_total_points": 6,
      "agent_total_points": 9,
      "n_races": 8
    }
  }
}
```

Notes :
- `seen_in_training: true` pour les GP dont le tracé/profil météo est dans le pool d'entraînement (GB 2025, Australie 2025 — cf. `gp_pool_config.py`). Le dashboard doit afficher un badge visuel distinct pour ces courses (ex. "Vu à l'entraînement" vs "Généralisation").
- `summary.unseen_races_only` isole les GP jamais vus — c'est le chiffre à mettre en avant devant le jury, pas le total brut.
- `delta_points = agent.points - real.points` (peut être négatif).

---

## 2. `race_<GP>_2025.json` (ex. `race_belgique_2025.json`)

```json
{
  "gp_name": "Belgique",
  "date": "2025-07-27",
  "seen_in_training": false,
  "total_laps": 44,
  "weather_summary": "Pluie intermittente, piste humide en début de course",
  "laps": [
    {
      "lap": 1,
      "position_real": 15,
      "position_agent": 15,
      "tire_age_real": 0,
      "tire_age_agent": 0,
      "compound_real": "INTERMEDIATE",
      "compound_agent": "INTERMEDIATE",
      "gap_ahead_s": 1.2,
      "gap_behind_s": 0.8,
      "is_raining": true,
      "track_status": "green",
      "agent_action": "stay",
      "agent_action_confidence": 0.94,
      "agent_action_probs": {
        "stay": 0.94,
        "pit_soft": 0.01,
        "pit_medium": 0.01,
        "pit_hard": 0.01,
        "pit_intermediate": 0.02,
        "pit_wet": 0.01
      },
      "pit_real_this_lap": false,
      "pit_agent_this_lap": false
    }
  ],
  "pit_events": {
    "real": [{"lap": 18, "compound_after": "MEDIUM", "duration_s": 22.4}],
    "agent": [{"lap": 15, "compound_after": "MEDIUM", "duration_s": 22.4, "confidence": 0.87}]
  },
  "final": {
    "real": {"position": 19, "points": 0},
    "agent": {"position": 2, "points": 18}
  }
}
```

Notes :
- `agent_action_probs` alimente l'indicateur de confiance affiché à chaque tour (barres ou jauge).
- `track_status` : `green | yellow | sc | vsc | red` (cohérent avec les événements SC déterministes du simulateur).
- Le dashboard rejoue les tours un par un (lecture automatique ou scrubber manuel) — pas de calcul en direct, tout est déjà dans le fichier.

---

## 3. `undercut_sim.json`

```json
{
  "scenario_name": "Undercut vs overcut — GP Belgique, tour 14",
  "context": {
    "gp_name": "Belgique",
    "lap": 14,
    "position": 5,
    "gap_ahead_s": 3.1,
    "tire_age": 13,
    "compound": "INTERMEDIATE"
  },
  "options": [
    {
      "strategy": "undercut_immediate",
      "label": "Pit immédiat (undercut)",
      "pit_lap": 14,
      "simulated_outcome": {
        "position_at_lap_20": 3,
        "confidence": 0.78
      }
    },
    {
      "strategy": "stay_out_3_laps",
      "label": "Rester 3 tours (overcut)",
      "pit_lap": 17,
      "simulated_outcome": {
        "position_at_lap_20": 5,
        "confidence": 0.62
      }
    },
    {
      "strategy": "agent_choice",
      "label": "Choix réel de l'agent",
      "pit_lap": 15,
      "simulated_outcome": {
        "position_at_lap_20": 2,
        "confidence": 0.87
      }
    }
  ]
}
```

Notes :
- Ce fichier est un export **illustratif** d'un seul moment de décision (pas tour par tour complet) — sert à montrer visuellement le raisonnement de l'agent sur un dilemme stratégique classique.
- `simulated_outcome` vient de plusieurs relances de l'environnement (mêmes seeds/conditions, action forcée différente) — pas une vraie incertitude du modèle, à annoncer comme tel en soutenance.
