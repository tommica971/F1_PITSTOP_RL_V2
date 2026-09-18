"""Vérifie que tous les modèles se rechargent dans le venv courant.

Usage (à la racine du projet, venv 3.12 activé) :
    python check_models.py
"""
from pathlib import Path
import sys

import numpy, torch, gymnasium, stable_baselines3
from stable_baselines3 import A2C, DQN, PPO

print(f"Python {sys.version.split()[0]} | numpy {numpy.__version__} | "
      f"torch {torch.__version__} | SB3 {stable_baselines3.__version__} | "
      f"gymnasium {gymnasium.__version__}\n")

ALGOS = {"dqn": DQN, "a2c": A2C, "ppo": PPO}

n_ok, n_ko = 0, 0
for path in sorted(Path("models").rglob("*.zip")):
    name = path.stem.lower()
    algo = next((cls for key, cls in ALGOS.items() if key in name), None)
    if algo is None:
        print(f"[?]  {path} — algorithme non reconnu, ignoré")
        continue
    try:
        model = algo.load(path, device="cpu")
        obs = model.observation_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        print(f"[OK] {path} — gamma={model.gamma:.4f}, action test={action}")
        n_ok += 1
    except Exception as e:
        print(f"[KO] {path} — {type(e).__name__}: {e}")
        n_ko += 1

print(f"\nBilan : {n_ok} OK, {n_ko} KO")
