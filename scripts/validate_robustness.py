"""
F1_PITSTOP_RL/scripts/validate_robustness.py

Repond a une question simple avant la soutenance : "le resultat affiche
(ex. Monaco P19 -> P4) est-il une vraie capacite de l'agent, ou un coup de
chance d'un seul tirage stochastique ?"

Pour chaque GP demande :
    - Rejoue N episodes en mode stochastique (model.predict(deterministic=False))
      et rapporte moyenne +/- ecart-type de la position/points finaux.
    - Rejoue 1 episode en mode deterministe (l'action la plus probable, pas un
      tirage) -- c'est la version "politique pure", reproductible.
    - Detecte les sauts de position suspects (> SUSPICIOUS_JUMP positions en un
      seul tour) qui peuvent signaler un bug de calcul plutot qu'une vraie
      remontee stategique -- imprime un avertissement pour inspection manuelle,
      ne corrige rien automatiquement.

Reutilise les memes imports que run_season_inference.py (aucune nouvelle
logique d'extraction : suppose que features_dataset.parquet contient deja
les GP demandes).

Usage :
    cd F1_PITSTOP_RL/scripts
    python validate_robustness.py --gp "Monaco Grand Prix" --n-runs 10
    python validate_robustness.py --gp "Monaco Grand Prix" "Belgian Grand Prix" --n-runs 10
    python validate_robustness.py --all-dashboard-gps --n-runs 10   # tous les GP deja dans dashboard/data/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src" / "f1_pitstop_rl"
sys.path.insert(0, str(SRC_DIR / "config"))
sys.path.insert(0, str(SRC_DIR / "data"))
sys.path.insert(0, str(SRC_DIR / "env"))

from gp_pool_config import GP_POOL                      # noqa: E402
from f1_pitstop_env import F1PitStopEnv                  # noqa: E402
from reward import POINTS_TABLE                          # noqa: E402

DATA_DIR = SCRIPT_DIR.parent / "data"
FEATURES_PATH = DATA_DIR / "processed" / "features_dataset.parquet"
MODEL_PATH = SCRIPT_DIR.parent / "models" / "a2c" / "a2c_extended_pool_5000k.zip"
DASHBOARD_DATA_DIR = SCRIPT_DIR.parent / "dashboard" / "data"

SEASON = 2025
SUSPICIOUS_JUMP = 5  # positions gagnees/perdues en un seul tour -- au-dela, a inspecter manuellement


def gp_role_in_pool(season: int, event: str):
    for gp in GP_POOL:
        if gp["season"] == season and gp["event"] == event:
            return gp["role"]
    return None


def run_one_episode(model, season: int, event: str, deterministic: bool, seed: int):
    gp_dict = {"season": season, "event": event, "role": gp_role_in_pool(season, event) or "validation",
               "known_issues": []}
    env = F1PitStopEnv(fixed_gp=gp_dict, seed=seed)
    obs, _ = env.reset(seed=seed)

    positions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        positions.append(info["position"])

    final_position = positions[-1]
    final_points = POINTS_TABLE.get(final_position, 0)

    # Detection de sauts suspects (hors premier tour, qui reflete l'etat reel herite)
    jumps = []
    for i in range(1, len(positions)):
        delta = abs(positions[i] - positions[i - 1])
        if delta >= SUSPICIOUS_JUMP:
            jumps.append({"lap_index": i, "from": positions[i - 1], "to": positions[i], "delta": delta})

    return final_position, final_points, jumps


def validate_gp(model, season: int, event: str, n_runs: int):
    print(f"\n=== {event} {season} ===")

    # Version deterministe (politique pure, reproductible)
    det_pos, det_pts, det_jumps = run_one_episode(model, season, event, deterministic=True, seed=42)
    print(f"  Deterministe (politique pure) : P{det_pos} ({det_pts} pts)")
    if det_jumps:
        print(f"    [!] Sauts de position suspects (>= {SUSPICIOUS_JUMP} places en 1 tour) :")
        for j in det_jumps:
            print(f"        tour idx {j['lap_index']} : P{j['from']} -> P{j['to']} (delta {j['delta']})")

    # N runs stochastiques (seeds differentes pour varier a la fois l'action ET
    # l'episode si le seed influence autre chose que le tirage d'action -- ici
    # seul le tirage d'action varie, l'episode est fixe via fixed_gp)
    stoch_positions, stoch_points = [], []
    all_jumps = []
    for i in range(n_runs):
        pos, pts, jumps = run_one_episode(model, season, event, deterministic=False, seed=1000 + i)
        stoch_positions.append(pos)
        stoch_points.append(pts)
        if jumps:
            all_jumps.append((i, jumps))

    mean_pos, std_pos = np.mean(stoch_positions), np.std(stoch_positions)
    mean_pts, std_pts = np.mean(stoch_points), np.std(stoch_points)
    print(f"  Stochastique ({n_runs} runs)   : P{mean_pos:.1f} +/- {std_pos:.1f}  "
          f"({mean_pts:.1f} +/- {std_pts:.1f} pts)")
    print(f"    positions obtenues : {sorted(stoch_positions)}")

    if std_pos >= 3:
        print(f"    [!] Ecart-type eleve (>= 3 positions) -- le resultat stochastique affiche au dashboard "
              f"peut etre un tirage chanceux, pas une capacite fiable. Preferer la moyenne ou le "
              f"deterministe pour le dossier/soutenance.")

    if all_jumps:
        print(f"    [!] Sauts suspects detectes sur {len(all_jumps)}/{n_runs} runs stochastiques "
              f"-- a inspecter (bug de calcul de position au retour de pit ?)")

    return {
        "gp_name": event, "deterministic": {"position": det_pos, "points": det_pts, "suspicious_jumps": det_jumps},
        "stochastic": {"n_runs": n_runs, "mean_position": mean_pos, "std_position": std_pos,
                        "mean_points": mean_pts, "std_points": std_pts, "positions": stoch_positions},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp", nargs="+", default=None, help="Un ou plusieurs GP a valider")
    parser.add_argument("--all-dashboard-gps", action="store_true",
                         help="Valide tous les GP deja presents dans dashboard/data/")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--out", type=str, default=None, help="Chemin JSON optionnel pour sauver le rapport")
    args = parser.parse_args()

    if args.all_dashboard_gps:
        events = []
        for p in sorted(DASHBOARD_DATA_DIR.glob("race_*_2025.json")):
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # Fichiers generes avant le correctif d'encodage (ecriture par
                # defaut Windows = cp1252 au lieu d'UTF-8) -- lecture de secours.
                text = p.read_text(encoding="cp1252")
            events.append(json.loads(text)["gp_name"])
    elif args.gp:
        events = args.gp
    else:
        raise SystemExit("Precise --gp \"Nom du GP\" ou --all-dashboard-gps")

    print(f"Chargement du modele : {MODEL_PATH}")
    model = A2C.load(str(MODEL_PATH), device="cpu")

    results = [validate_gp(model, SEASON, event, args.n_runs) for event in events]

    high_variance = [r["gp_name"] for r in results if r["stochastic"]["std_position"] >= 3]
    with_jumps = [r["gp_name"] for r in results
                  if r["deterministic"]["suspicious_jumps"]]

    print("\n" + "=" * 60)
    print(f"RESUME ({len(results)} GP valides)")
    if high_variance:
        print(f"  Variance elevee (resultat peu fiable a l'echelle d'un seul run) : {high_variance}")
    if with_jumps:
        print(f"  Sauts de position suspects en deterministe : {with_jumps}")
    if not high_variance and not with_jumps:
        print("  Aucun signal d'alerte -- les resultats semblent robustes sur ces GP.")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRapport detaille : {args.out}")


if __name__ == "__main__":
    main()
