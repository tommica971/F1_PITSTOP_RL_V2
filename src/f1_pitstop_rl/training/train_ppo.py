"""Phase 3.2 — Entraînement PPO sur le pool d'entraînement (train_wet + train_dry).

Phase 3.5 bis : option --extended-pool pour entraîner sur le pool étendu à
10 GP (Turkish Grand Prix 2021 ajouté -- profil "piste humide au départ, ne
repleut jamais", absent du pool jusqu'ici et identifié comme cause de
l'échec de généralisation d'A2C/PPO sur Belgique 2025, cf. notebook 07).
Réutilise les hyperparamètres déjà tunés (ppo_optuna_best_params.json),
pour isoler l'effet du pool. Sauvegardé sous ppo_extended_pool.zip --
n'écrase pas ppo_baseline/ppo_tuned.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import make_train_env, EpisodeLogCallback, TOTAL_TIMESTEPS, SEED, MODELS_DIR, LOGS_DIR

from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extended-pool", action="store_true",
        help="Entraîne sur le pool étendu à 10 GP (Turkish Grand Prix 2021 ajouté), "
             "hyperparamètres déjà tunés inchangés. Sauvegardé sous ppo_extended_pool.zip.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Budget de timesteps (défaut : {TOTAL_TIMESTEPS}).",
    )
    args = parser.parse_args()

    if args.extended_pool:
        best_params_path = LOGS_DIR / "ppo_optuna_best_params.json"
        with open(best_params_path) as f:
            hyperparams = json.load(f)["best_params"]
        print(f"--extended-pool actif : pool a 8 GP d'entrainement, hyperparametres repris de {best_params_path}")
        model_name = "ppo_extended_pool"
    else:
        hyperparams = {}
        model_name = "ppo_baseline"

    if args.timesteps != TOTAL_TIMESTEPS:
        model_name += f"_{args.timesteps // 1000}k"

    env = make_train_env(seed=SEED)
    # device='cpu' : SB3 deconseille le GPU pour MlpPolicy (cf. warning
    # SB3 lors des runs precedents) -- aucun gain de vitesse ici, et le GPU
    # introduit une non-determinisme (meme seed, resultats differents d'un
    # run a l'autre) qui expliquait l'ecart observe entre deux baselines
    # PPO supposees identiques (Phase 3.4).
    model = PPO("MlpPolicy", env, verbose=1, seed=SEED, device="cpu", **hyperparams)

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    out_dir = MODELS_DIR / "ppo"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / model_name)
    callback.save(LOGS_DIR / f"{model_name}_log.npz")

    print(f"\nPPO entraîné ({model_name}) : {len(callback.episode_rewards)} épisodes joués sur {args.timesteps} timesteps")
    print(f"Modèle sauvegardé -> {out_dir / (model_name + '.zip')}")


if __name__ == "__main__":
    main()
