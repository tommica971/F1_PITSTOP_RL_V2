"""
Phase 3.2 — Utilitaires communs d'entraînement
=================================================
Factorise ce qui est identique entre train_dqn.py / train_ppo.py / train_a2c.py :
création de l'environnement, callback de suivi des épisodes, sauvegarde.

Protocole retenu pour cette première passe (Phase 3.2, cf. échange de
cadrage) :
    - Hyperparamètres par défaut de Stable-Baselines3 (le réglage fin via
      Optuna est la Phase 3.3, étape suivante)
    - Entraînement sur le pool train_wet + train_dry (7 GP), un GP tiré
      aléatoirement à chaque épisode (cf. F1PitStopEnv, gp_roles par défaut)
    - 150 000 timesteps par algorithme (~2000-3500 épisodes selon le GP,
      calibré via un test de vitesse préalable : ~9-14s/5000 steps)
    - Pas de parallélisation (VecEnv) à ce stade -- Phase 4.1
"""

import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "env"))
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "config"))

from f1_pitstop_env import F1PitStopEnv

TOTAL_TIMESTEPS = 500_000
TRAIN_GP_ROLES = ("train_wet", "train_dry")
SEED = 0

MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "models" / "training_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def make_train_env(seed: int = SEED, gp_weights: dict | None = None,
                   min_stint_laps: int | None = None) -> Monitor:
    """min_stint_laps : None -> valeur du module ; 0 -> contrainte desactivee
    (ablation 2x2, cf. scripts/v2_05_apply_ablation_switches.py)."""
    env = F1PitStopEnv(gp_roles=TRAIN_GP_ROLES, seed=seed, gp_weights=gp_weights,
                       min_stint_laps=min_stint_laps)
    return Monitor(env)


# Preset (Phase 3.5 bis) : sur-pondere les GP secs (train_dry) a ~60% du
# tirage au lieu de ~43% en tirage uniforme (3 GP secs / 7 au total) --
# cf. f1_pitstop_env.py pour la justification complete. Poids par role,
# pas par GP individuel : chaque GP au sein d'un role garde un poids egal.
GP_WEIGHTS_FAVOR_DRY = {"train_dry": 2.0, "train_wet": 1.0}


class EpisodeLogCallback(BaseCallback):
    """Enregistre récompense, longueur et position finale de chaque épisode
    terminé, pour tracer les courbes d'apprentissage après coup."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_positions: list[int] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_timesteps.append(self.num_timesteps)
                self.episode_positions.append(info.get("position", np.nan))
        return True

    def save(self, path: Path):
        np.savez(
            path,
            rewards=np.array(self.episode_rewards),
            lengths=np.array(self.episode_lengths),
            positions=np.array(self.episode_positions),
            timesteps=np.array(self.episode_timesteps),
        )
