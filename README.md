# F1_PITSTOP_RL

Optimisation dynamique de la decision de pit stop en Formule 1 par apprentissage
par renforcement -- cas d'usage Pierre Gasly (Alpine F1 Team).

Voir la feuille de route complete dans le dossier ecrit (docs/dossier_ecrit/).

## Structure du projet

- src/f1_pitstop_rl/config/     -- pool de GP, configuration d'observation figee (C5.3.1)
- src/f1_pitstop_rl/data/       -- extraction FastF1 / TracingInsights, nettoyage, features (Phase 1)
- src/f1_pitstop_rl/env/        -- F1PitStopEnv, Gymnasium (Phase 2)
- src/f1_pitstop_rl/training/   -- DQN / PPO / A2C, tuning Optuna (Phase 3)
- src/f1_pitstop_rl/evaluation/ -- tests de generalisation, comparaison algos
- src/f1_pitstop_rl/monitoring/ -- derive de donnees, Evidently AI (Phase 4)
- tests/                        -- tests unitaires de l'environnement (Phase 2.3)
- docs/                         -- dossier ecrit + support de soutenance (Phase 5)

## Installation

    pip install -r requirements.txt

## Pipeline d'extraction

    python -m f1_pitstop_rl.data.extract_fastf1
    python -m f1_pitstop_rl.data.extract_tracinginsights
