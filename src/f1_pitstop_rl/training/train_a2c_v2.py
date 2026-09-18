"""Phase 4.3 — Entrainement A2C sur l'environnement v3, hyperparametres revus.

Pourquoi ce script existe a cote de train_a2c.py
-------------------------------------------------
Le run court sur l'environnement v3, avec les hyperparametres Optuna herites,
a produit une politique EFFONDREE : P(pit) mediane a 3.5e-06, entropie a
0.0002 contre log(6)=1.79, zero arret choisi meme en echantillonnant sur
20 episodes, recompense strictement egale au script 0-stop sur les 8 GP.

Diagnostic (scripts/diagnose_policy.py) :

  gamma = 0.9193  -> horizon effectif 1/(1-gamma) = 12.4 tours. Un arret
      coute 23 s immediatement et se rembourse sur 25-30 tours. Sur Bahrein,
      le benefice brut vaut ~2.7/tour sur 28 tours :
          actualise a gamma=0.919 : 30.8  contre 23 de cout -> net +7.8
          actualise a gamma=0.99  : 65.9  contre 23 de cout -> net +42.9
      A 0.919, le gain est noye sous le bruit (l'ecart-type du cout d'arret
      vaut 5.38 s a lui seul). L'agent ne PEUT PAS voir le remboursement.

  ent_coef = 1.1e-06 -> aucune pression d'exploration. Les premieres
      tentatives de pit sont punies de -23, le gradient pousse P(pit) vers 0,
      et rien ne l'en fait ressortir.

  n_steps = 5 -> rollout tres court, estimation d'avantage myope.

Ces valeurs viennent d'une etude Optuna menee sur l'ANCIEN environnement, ou
le 0-stop etait reellement optimal. Optuna a donc selectionne les reglages
qui convergent le plus vite vers "ne jamais pitter". Les reutiliser apres
avoir change la dynamique est une erreur methodologique -- a citer comme
telle dans le dossier.

Valeurs retenues ici, et leur justification
--------------------------------------------
  gamma = 0.99      horizon ~100 tours, superieur a la course la plus longue
                    du pool (72 tours). L'agent voit la fin de course depuis
                    le debut, ce qu'exige une decision d'arret.
  ent_coef = 0.01   valeur usuelle SB3 pour A2C discret. Maintient une
                    exploration residuelle sur les 6 actions.
  n_steps = 32      rollout plus long, avantage mieux estime sur un horizon
                    ou l'effet d'un arret devient visible.
  learning_rate, gae_lambda, vf_coef : repris d'Optuna. Ils ne portent pas
                    l'hypothese fautive sur l'horizon, et les conserver
                    limite le nombre de variables modifiees a la fois.

Usage, depuis la racine du projet :
    python src/f1_pitstop_rl/training/train_a2c_v2.py
    python src/f1_pitstop_rl/training/train_a2c_v2.py --timesteps 1000000
    python src/f1_pitstop_rl/training/train_a2c_v2.py --gamma 0.995 --ent-coef 0.02
    python src/f1_pitstop_rl/training/train_a2c_v2.py --inherit-optuna   # temoin
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (make_train_env, EpisodeLogCallback, SEED,  # noqa: E402
                    MODELS_DIR, LOGS_DIR)

from stable_baselines3 import A2C  # noqa: E402

# Hyperparametres revus pour l'environnement v3 -- cf. en-tete
HYPERPARAMS_V2 = {
    "learning_rate": 0.0008482974741031032,   # repris d'Optuna
    "gamma": 0.99,                            # 0.9193 -> horizon trop court
    "n_steps": 32,                            # 5 -> rollout trop court
    "ent_coef": 0.01,                         # 1.11e-06 -> effondrement
    "gae_lambda": 0.9098464906699114,         # repris d'Optuna
    "vf_coef": 0.2769352092003568,            # repris d'Optuna
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--ent-coef", type=float, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--inherit-optuna", action="store_true",
                    help="Temoin : reprend les hyperparametres Optuna tels "
                         "quels, pour documenter l'ecart dans le dossier.")
    args = ap.parse_args()

    if args.inherit_optuna:
        p = LOGS_DIR / "a2c_optuna_best_params.json"
        if not p.exists():
            sys.exit(f"Introuvable : {p}\n"
                     "Copier depuis models_v1/training_logs/ si besoin.")
        hyperparams = json.loads(p.read_text(encoding="utf-8"))["best_params"]
        default_name = "a2c_v3env_optuna_herite"
    else:
        hyperparams = dict(HYPERPARAMS_V2)
        for key, val in (("gamma", args.gamma), ("ent_coef", args.ent_coef),
                         ("n_steps", args.n_steps)):
            if val is not None:
                hyperparams[key] = val
        default_name = "a2c_v3env"

    name = args.name or f"{default_name}_{args.timesteps // 1000}k"

    gamma = hyperparams["gamma"]
    print(f"Modele        : {name}")
    print(f"Timesteps     : {args.timesteps:,}")
    print(f"Horizon (1/(1-gamma)) : {1 / (1 - gamma):.0f} tours "
          f"(course la plus longue du pool : 72 tours)")
    if 1 / (1 - gamma) < 72:
        print("  ATTENTION : horizon inferieur a la course la plus longue. "
              "L'agent ne verra pas la fin de course depuis le debut.")
    print("Hyperparametres :")
    for k, v in sorted(hyperparams.items()):
        print(f"   {k:<16} {v}")
    print()

    env = make_train_env(seed=args.seed)
    model = A2C("MlpPolicy", env, verbose=1, seed=args.seed, device="cpu",
                **hyperparams)

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    out_dir = MODELS_DIR / "a2c"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / name)
    callback.save(LOGS_DIR / f"{name}_log.npz")

    print(f"\n{len(callback.episode_rewards)} episodes joues sur "
          f"{args.timesteps:,} timesteps")
    print(f"Modele -> {out_dir / (name + '.zip')}")
    print(f"\nVerification :")
    print(f"  python scripts/diagnose_policy.py --model models/a2c/{name}.zip")
    print(f"  python scripts/eval_pit_behaviour.py --model models/a2c/{name}.zip")


if __name__ == "__main__":
    main()
