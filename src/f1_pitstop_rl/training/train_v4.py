"""Phase 4.4 — Entrainement v4 : a priori sur STAY a l'initialisation.

Le probleme, en une ligne
--------------------------
Cinq des six actions sont des arrets. Une politique initialisee uniformement
s'arrete ~83% des tours, soit ~50 arrets par course. Cette region de l'espace
des politiques est si catastrophique que le gradient ecrase P(pit) a zero
avant tout apprentissage utile.

Mesure : 95% du progres atteint a l'episode 8 sur 8291 (A2C, 500k pas), et a
l'episode 1505 sur 82968 pour le run historique de 5M pas -- 98% du budget de
calcul n'apprenait rien. Trois tentatives de correction par hyperparametres
(ent_coef x10000, gamma 0.919 -> 0.99, n_steps 5 -> 32, normalize_advantage)
n'ont pas suffi : le terme d'entropie plafonne a ent_coef * log(6) = 0.018
face a des avantages de plusieurs dizaines.

Les deux correctifs structurels
--------------------------------
1. MIN_STINT_LAPS dans l'environnement (scripts/apply_patch_min_stint.py) :
   plafonne les arrets a ~11 par course au lieu de 57.

2. A PRIORI SUR STAY, ici. Le biais de la derniere couche du reseau de
   politique est decale pour que P(stay) = --stay-prior (0.97 par defaut)
   a l'initialisation, au lieu de 1/6. L'agent demarre alors a ~1.7 arret
   par course -- un point de depart physiquement plausible.

   Ce n'est PAS de l'apprentissage par imitation : on n'indique ni quand ni
   avec quel pneu s'arreter, on deplace seulement le point de depart hors
   d'une region degeneree. La politique reste libre d'apprendre n'importe
   quelle strategie. L'initialisation est identique pour les trois
   algorithmes, donc la comparaison reste equitable.

   Logit applique : log(p * 5 / (1 - p)) sur l'action 0, les 5 actions de
   pit se partageant (1 - p) a parts egales.

Usage, depuis la racine du projet :
    python src/f1_pitstop_rl/training/train_v4.py --algo a2c
    python src/f1_pitstop_rl/training/train_v4.py --algo dqn
    python src/f1_pitstop_rl/training/train_v4.py --algo a2c --stay-prior 0.90
    python src/f1_pitstop_rl/training/train_v4.py --algo a2c --stay-prior 0.1667  # temoin uniforme
"""

import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (make_train_env, EpisodeLogCallback, SEED,  # noqa: E402
                    MODELS_DIR, LOGS_DIR)

from stable_baselines3 import A2C, DQN, PPO  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[3]

N_ACTIONS = 6
N_PIT_ACTIONS = N_ACTIONS - 1


class RewardScale(gym.RewardWrapper):
    """Homothetie positive : ne change pas la politique optimale, seulement
    l'echelle des gradients et des cibles de valeur."""

    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


A2C_PARAMS = {
    "learning_rate": 0.0008482974741031032,
    "gamma": 0.99, "n_steps": 32, "ent_coef": 0.05,
    "gae_lambda": 0.9098464906699114, "vf_coef": 0.2769352092003568,
    "normalize_advantage": True,
}
PPO_PARAMS = {
    "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "batch_size": 64,
    "ent_coef": 0.02, "gae_lambda": 0.95, "n_epochs": 10,
}
DQN_PARAMS = {
    "learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 200_000,
    "learning_starts": 10_000, "batch_size": 64, "train_freq": 4,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.40, "exploration_final_eps": 0.10,
}
PARAMS = {"a2c": A2C_PARAMS, "ppo": PPO_PARAMS, "dqn": DQN_PARAMS}
CLASSES = {"a2c": A2C, "ppo": PPO, "dqn": DQN}


def apply_stay_prior(model, algo: str, p_stay: float) -> bool:
    """Decale le biais de sortie pour que P(stay) = p_stay a l'initialisation.

    A2C / PPO : logits de politique -> softmax. On ajoute un offset a
    l'action 0 tel que le softmax donne exactement p_stay, les 5 actions de
    pit se partageant le reste.

    DQN : il n'y a pas de politique parametree, l'action greedy est
    l'argmax des Q. On rehausse le biais de sortie de l'action 0 pour que
    STAY soit l'action greedy au depart ; l'exploration reste assuree par
    epsilon-greedy, dont le calendrier est independant du gradient.
    """
    if not 0 < p_stay < 1:
        sys.exit("--stay-prior doit etre strictement entre 0 et 1.")
    offset = math.log(p_stay * N_PIT_ACTIONS / (1 - p_stay))

    with torch.no_grad():
        if algo in ("a2c", "ppo"):
            layer = getattr(model.policy, "action_net", None)
            if layer is None or layer.bias is None:
                return False
            layer.bias[0] += offset
            return True
        # DQN
        q = getattr(model.policy, "q_net", None)
        net = getattr(q, "q_net", None) if q is not None else None
        if net is None or not hasattr(net[-1], "bias"):
            return False
        # echelle des Q : un offset en unites de recompense par tour suffit
        net[-1].bias[0] += abs(offset)
        model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
        return True


def measure_initial_pit_rate(model, env, n_laps: int = 500) -> float:
    """Taux d'arret d'une politique fraichement initialisee, pour verifier
    que l'a priori a bien ete applique."""
    obs, _ = env.reset(seed=123)
    pits = 0
    for _ in range(n_laps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, _, info = env.step(int(action))
        pits += int(bool(info.get("is_pit")))
        if done:
            obs, _ = env.reset()
    return pits / n_laps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=list(CLASSES), default="a2c")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--stay-prior", type=float, default=0.97)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--min-stint-laps", type=int, default=None,
                    help="Longueur minimale de relais. 0 desactive "
                         "la contrainte (ablation). Defaut : valeur "
                         "du module.")
    ap.add_argument("--params-json", default=None,
                    help="Hyperparametres issus d'une etude Optuna. "
                         "Evite la recopie manuelle et garantit que le "
                         "modele entraine est bien celui selectionne.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()

    if args.reward_scale <= 0:
        sys.exit("--reward-scale doit etre strictement positif.")

    params = dict(PARAMS[args.algo])
    if args.params_json:
        import json
        p = Path(args.params_json)
        if not p.is_absolute():
            p = ROOT_DIR / p
        if not p.exists():
            sys.exit(f"Fichier d'hyperparametres introuvable : {p}")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        loaded = loaded.get("best_params", loaded)
        # Seules les cles connues de l'algorithme sont reprises : le JSON
        # d'Optuna contient aussi des metadonnees (score, graines, GP de
        # validation) que SB3 rejetterait.
        unknown = [k for k in loaded if k not in params]
        params.update({k: v for k, v in loaded.items() if k in params})
        print(f"Hyperparametres lus depuis {p.name}")
        if unknown:
            print(f"  cles ignorees (metadonnees) : {', '.join(unknown)}")
    name = args.name or f"{args.algo}_v4_{args.timesteps // 1000}k"

    expected = (1 - args.stay_prior) * 57
    print(f"Modele        : {name}")
    print(f"Algorithme    : {args.algo.upper()}")
    print(f"Timesteps     : {args.timesteps:,}")
    print(f"A priori STAY : {args.stay_prior:.4f} "
          f"-> ~{expected:.1f} arrets par course a l'initialisation "
          f"(uniforme : {5 / 6 * 57:.0f})")
    print(f"Reward scale  : {args.reward_scale}")
    print()

    env = make_train_env(seed=args.seed,
                         min_stint_laps=args.min_stint_laps)
    if args.reward_scale != 1.0:
        env = RewardScale(env, args.reward_scale)

    model = CLASSES[args.algo]("MlpPolicy", env, verbose=1, seed=args.seed,
                               device="cpu", **params)

    if not apply_stay_prior(model, args.algo, args.stay_prior):
        sys.exit("ERREUR : impossible d'atteindre la couche de sortie du "
                 "reseau. Structure de politique SB3 inattendue — ne pas "
                 "poursuivre, le run serait identique aux precedents.")

    rate = measure_initial_pit_rate(model, make_train_env(seed=args.seed + 99))
    print(f"Taux d'arret mesure avant entrainement : {rate:.1%} "
          f"({rate * 57:.1f} arrets par course)")
    if args.algo in ("a2c", "ppo") and rate > 0.25:
        print("  ATTENTION : taux beaucoup plus eleve qu'attendu. "
              "L'a priori n'a probablement pas ete applique correctement.")
    print()

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    out_dir = MODELS_DIR / args.algo
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / name)
    callback.save(LOGS_DIR / f"{name}_log.npz")

    r = np.array(callback.episode_rewards)
    print(f"\n{len(r)} episodes | recompense : debut {r[:50].mean():.1f}, "
          f"fin {r[-50:].mean():.1f}")
    print(f"Modele -> {out_dir / (name + '.zip')}")
    print("\nVerification, dans cet ordre :")
    print(f"  python scripts/inspect_training_log.py "
          f"--log models/training_logs/{name}_log.npz")
    print(f"  python scripts/eval_pit_behaviour.py "
          f"--model models/{args.algo}/{name}.zip --algo {args.algo}")


if __name__ == "__main__":
    main()
