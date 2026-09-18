"""Phase 3.2 — Entraînement A2C sur le pool d'entraînement (train_wet + train_dry).

Phase 3.5 bis : option --favor-dry pour tester si le sur-tirage des GP secs
(train_dry) à l'entraînement corrige l'échec de généralisation observé sur
Belgique 2025 (A2C tuné n'apprend quasiment jamais à pit volontairement --
cf. notebook 06, section 6bis). Réutilise les hyperparamètres déjà trouvés
par Optuna pour A2C tuné (a2c_optuna_best_params.json) : seule la
pondération du tirage de GP change, pour isoler cet effet précis plutôt que
mélanger plusieurs changements. Sauvegardé sous un nom distinct
(a2c_favor_dry.zip) -- n'écrase ni la baseline ni le tuné actuels, pour
garder une comparaison propre.

Résultat --favor-dry : n'a pas corrigé Belgique (cf. notebook 07) --
diagnostic plus poussé (tour par tour, données réelles) a montré que le
pool ne contenait aucun GP avec le profil "piste humide au départ, ne
repleut jamais" (celui de Belgique). Turkish Grand Prix 2021 a ce profil
exact et a été ajouté au pool (gp_pool_config.py, GP_POOL passe de 9 à 10,
8 GP d'entraînement au lieu de 7). --extended-pool entraîne sur ce nouveau
pool, hyperparamètres tuned inchangés (isole l'effet de la couverture du
pool, comme --favor-dry isolait l'effet de sa pondération). Aucun
gp_weights nécessaire : make_train_env() filtre déjà par rôle, donc
Turkish Grand Prix (train_wet) est automatiquement inclus.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import make_train_env, EpisodeLogCallback, TOTAL_TIMESTEPS, SEED, MODELS_DIR, LOGS_DIR, GP_WEIGHTS_FAVOR_DRY

from stable_baselines3 import A2C


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--favor-dry", action="store_true",
        help="Sur-pondère les GP secs à l'entraînement (~60%% du tirage au lieu de ~43%%), "
             "en réutilisant les hyperparamètres déjà tunés. Sauvegardé sous a2c_favor_dry.zip.",
    )
    group.add_argument(
        "--extended-pool", action="store_true",
        help="Entraîne sur le pool étendu à 10 GP (Turkish Grand Prix 2021 ajouté), "
             "hyperparamètres déjà tunés inchangés. Sauvegardé sous a2c_extended_pool.zip.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Budget de timesteps (défaut : {TOTAL_TIMESTEPS}).",
    )
    args = parser.parse_args()

    if args.favor_dry:
        gp_weights = GP_WEIGHTS_FAVOR_DRY
        best_params_path = LOGS_DIR / "a2c_optuna_best_params.json"
        with open(best_params_path) as f:
            hyperparams = json.load(f)["best_params"]
        print(f"--favor-dry actif : tirage pondéré {gp_weights}, hyperparamètres repris de {best_params_path}")
        model_name = "a2c_favor_dry"
    elif args.extended_pool:
        gp_weights = None  # tirage uniforme sur le pool a 8 GP d'entrainement (deja mis a jour dans GP_POOL)
        best_params_path = LOGS_DIR / "a2c_optuna_best_params.json"
        with open(best_params_path) as f:
            hyperparams = json.load(f)["best_params"]
        print(f"--extended-pool actif : pool a 8 GP d'entrainement, hyperparametres repris de {best_params_path}")
        model_name = "a2c_extended_pool"
    else:
        gp_weights = None
        hyperparams = {}
        model_name = "a2c_baseline"

    if args.timesteps != TOTAL_TIMESTEPS:
        model_name += f"_{args.timesteps // 1000}k"

    env = make_train_env(seed=SEED, gp_weights=gp_weights)
    # device='cpu' : cf. train_ppo.py -- reproductibilite, pas de gain GPU pour MlpPolicy
    model = A2C("MlpPolicy", env, verbose=1, seed=SEED, device="cpu", **hyperparams)

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    out_dir = MODELS_DIR / "a2c"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / model_name)
    callback.save(LOGS_DIR / f"{model_name}_log.npz")

    print(f"\nA2C entraîné ({model_name}) : {len(callback.episode_rewards)} épisodes joués sur {args.timesteps} timesteps")
    print(f"Modèle sauvegardé -> {out_dir / (model_name + '.zip')}")


if __name__ == "__main__":
    main()
