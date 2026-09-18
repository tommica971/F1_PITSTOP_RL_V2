"""
F1_PITSTOP_RL/scripts/extract_model_data.py

Lit les logs d'entrainement (models/training_logs/*.npz) et les resultats
Optuna (*_optuna_trials.csv, *_optuna_best_params.json) pour produire un JSON
compact destine a l'onglet "Modele" du dashboard : courbe de reward (moyenne
glissante, sous-echantillonnee), courbe de position finale par episode, et
resume des essais Optuna.

Les logs bruts (ex. a2c_extended_pool_5000k_log.npz) contiennent ~83 000
episodes -- beaucoup trop pour un graphique web. Sous-echantillonnage par
bucket (moyenne glissante) a TARGET_POINTS points.

Usage :
    cd F1_PITSTOP_RL/scripts
    python extract_model_data.py --model a2c_extended_pool_5000k --algo a2c
    python extract_model_data.py --model a2c_extended_pool_5000k --algo a2c \
        --compare a2c_baseline a2c_tuned    # ajoute des courbes de comparaison

Sortie :
    dashboard/data/model_training.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
LOGS_DIR = SCRIPT_DIR.parent / "models" / "training_logs"
DASHBOARD_DATA_DIR = SCRIPT_DIR.parent / "dashboard" / "data"

TARGET_POINTS = 200  # nombre de points apres sous-echantillonnage pour chaque courbe


def downsample(values: np.ndarray, target: int = TARGET_POINTS):
    """Moyenne par bucket -- preserve la tendance sans exporter tous les points bruts."""
    n = len(values)
    if n <= target:
        return values.tolist()
    bucket_size = n // target
    n_full = bucket_size * target
    buckets = values[:n_full].reshape(target, bucket_size)
    out = buckets.mean(axis=1)
    return out.tolist()


def load_training_curve(model_name: str):
    path = LOGS_DIR / f"{model_name}_log.npz"
    if not path.exists():
        print(f"  [!] Introuvable : {path}")
        return None
    d = np.load(path)
    rewards, positions, timesteps = d["rewards"], d["positions"], d["timesteps"]
    return {
        "n_episodes": int(len(rewards)),
        "final_timesteps": int(timesteps[-1]),
        "reward_curve": downsample(rewards),
        "position_curve": downsample(positions.astype(float)),
        "timesteps_curve": downsample(timesteps.astype(float)),
        # Moyenne glissante sur les 500 derniers episodes, pour un indicateur
        # de convergence simple (reward et position stabilisees ou non)
        "reward_last_500_mean": float(np.mean(rewards[-500:])) if len(rewards) >= 500 else float(np.mean(rewards)),
        "position_last_500_mean": float(np.mean(positions[-500:])) if len(positions) >= 500 else float(np.mean(positions)),
    }


def load_optuna(algo: str):
    trials_path = LOGS_DIR / f"{algo}_optuna_trials.csv"
    best_path = LOGS_DIR / f"{algo}_optuna_best_params.json"
    if not trials_path.exists() or not best_path.exists():
        print(f"  [!] Optuna introuvable pour {algo}")
        return None

    import csv
    trials = []
    with open(trials_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("state") != "COMPLETE":
                continue
            trials.append({
                "number": int(row["number"]),
                "value": float(row["value"]),
                "params": {k.replace("params_", ""): float(v) for k, v in row.items()
                           if k.startswith("params_") and v not in (None, "")},
            })
    best = json.loads(best_path.read_text(encoding="utf-8"))
    return {"trials": trials, "best_value": best["best_value"], "best_params": best["best_params"],
            "timesteps_per_trial": best.get("timesteps_per_trial")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Nom du modele final (ex. a2c_extended_pool_5000k)")
    parser.add_argument("--algo", type=str, required=True, help="Algo pour les donnees Optuna (ex. a2c)")
    parser.add_argument("--compare", nargs="*", default=[], help="Autres modeles a inclure pour comparaison")
    args = parser.parse_args()

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Modele principal : {args.model}")
    main_curve = load_training_curve(args.model)

    comparisons = {}
    for name in args.compare:
        print(f"Comparaison : {name}")
        curve = load_training_curve(name)
        if curve:
            comparisons[name] = curve

    print(f"Optuna : {args.algo}")
    optuna_data = load_optuna(args.algo)

    output = {
        "main_model": args.model,
        "training_curve": main_curve,
        "comparisons": comparisons,
        "optuna": optuna_data,
    }

    out_path = DASHBOARD_DATA_DIR / "model_training.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOK -> {out_path}")


if __name__ == "__main__":
    main()
