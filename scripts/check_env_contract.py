"""Verifie que l'environnement respecte le contrat Gymnasium (check_env).

Usage, depuis la racine du projet :
    python scripts/check_env_contract.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for sub in ("env", "config", "data", "training"):
    sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / sub))

from gymnasium.utils.env_checker import check_env  # noqa: E402
from f1_pitstop_env import F1PitStopEnv  # noqa: E402

env = F1PitStopEnv()
check_env(env, skip_render_check=True)
print("Contrat Gymnasium respecte.")
