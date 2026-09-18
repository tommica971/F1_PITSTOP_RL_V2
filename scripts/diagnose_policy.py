#!/usr/bin/env python3
"""
Diagnostic de politique — pourquoi l'agent ne s'arrete-t-il jamais ?
====================================================================

Constat (eval_pit_behaviour.py) : la recompense de l'agent est EXACTEMENT
celle du script 0-stop sur les 8 GP. La politique joue donc `stay` a chaque
tour, en deterministe. Reste a savoir si elle est effondree ou seulement
penchee.

Trois mesures, qui menent a des correctifs differents :

  1. HYPERPARAMETRES du modele (ent_coef, gamma, n_steps, learning_rate).
     Ils viennent d'Optuna, regle sur l'ANCIEN environnement ou le 0-stop
     etait reellement optimal : Optuna a pu selectionner les reglages qui
     convergent le plus vite vers "ne jamais pitter".

  2. PROBABILITES D'ACTION le long d'une course. Si P(pit) vaut 1e-8, la
     politique est effondree : aucun budget d'entrainement ne la ramenera,
     il faut relancer avec plus d'entropie. Si P(pit) vaut 0.05-0.20 avec
     un pic quelque part au milieu de course, la politique a appris quelque
     chose mais l'argmax reste sur `stay` -- un simple manque de temps.

  3. COMPORTEMENT STOCHASTIQUE. En echantillonnant au lieu de prendre
     l'argmax, l'agent s'arrete-t-il parfois ? Si oui, le signal existe.

Usage, depuis la racine du projet :
    python scripts/diagnose_policy.py --model models/a2c/a2c_extended_pool.zip
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("src/f1_pitstop_rl/env", "src/f1_pitstop_rl/config"):
    sys.path.insert(0, str(ROOT / p))

from f1_pitstop_env import F1PitStopEnv, ACTIONS  # noqa: E402
from gp_pool_config import GP_POOL  # noqa: E402

ALGOS = {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="a2c", choices=list(ALGOS))
    ap.add_argument("--gp", default="Bahrain Grand Prix")
    ap.add_argument("--n-stochastic", type=int, default=20)
    args = ap.parse_args()

    import stable_baselines3 as sb3
    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    model = getattr(sb3, ALGOS[args.algo]).load(str(path), device="cpu")

    # --- 1. hyperparametres ------------------------------------------------
    print("=" * 74)
    print("1. HYPERPARAMETRES DU MODELE")
    print("=" * 74)
    keys = ["learning_rate", "gamma", "n_steps", "ent_coef", "vf_coef",
            "gae_lambda", "max_grad_norm", "use_rms_prop", "normalize_advantage",
            "batch_size", "exploration_fraction", "exploration_final_eps"]
    for k in keys:
        v = getattr(model, k, None)
        if v is not None:
            print(f"   {k:<24} {v() if callable(v) else v}")

    json_path = ROOT / "models" / "training_logs" / f"{args.algo}_optuna_best_params.json"
    if json_path.exists():
        best = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"\n   Optuna (regle sur l'ANCIEN environnement) :")
        print(f"   {best.get('best_params')}")
        print(f"   valeur optimisee : {best.get('best_value')}")

    # --- 2. probabilites d'action ------------------------------------------
    print("\n" + "=" * 74)
    print(f"2. PROBABILITES D'ACTION — {args.gp}")
    print("=" * 74)

    gp = next((g for g in GP_POOL if g["event"] == args.gp), None)
    if gp is None:
        sys.exit(f"GP absent du pool : {args.gp}")

    env = F1PitStopEnv(fixed_gp=gp, seed=0)
    obs, _ = env.reset(seed=0)

    rows, p_pit_all, entropies = [], [], []
    done = False
    while not done:
        lap, age = env.current_lap, env.own_tyre_age
        probs = None
        try:
            t, _ = model.policy.obs_to_tensor(obs)
            dist = model.policy.get_distribution(t).distribution
            probs = dist.probs.detach().cpu().numpy().ravel()
            entropies.append(float(dist.entropy().item()))
        except AttributeError:
            pass  # DQN : pas de distribution, on se rabat sur les Q-valeurs

        if probs is not None:
            p_pit = float(probs[1:].sum())
            p_pit_all.append(p_pit)
            rows.append((lap, age, p_pit, int(np.argmax(probs))))

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))

    if rows:
        print(f"{'tour':>6}{'age pneu':>10}{'P(pit)':>12}{'argmax':>16}")
        step = max(len(rows) // 14, 1)
        for lap, age, p, am in rows[::step]:
            print(f"{lap:>6}{age:>10}{p:>12.2e}{ACTIONS[am]:>16}")

        pmax, pmed = max(p_pit_all), float(np.median(p_pit_all))
        print(f"\n   P(pit) : mediane {pmed:.2e}  |  maximum {pmax:.2e} "
              f"(tour {rows[int(np.argmax(p_pit_all))][0]})")
        if entropies:
            print(f"   Entropie moyenne : {np.mean(entropies):.4f} "
                  f"(maximum possible pour 6 actions : {np.log(6):.4f})")

        print()
        if pmax < 1e-3:
            print("   -> POLITIQUE EFFONDREE. Aucun budget supplementaire ne la")
            print("      ramenera : il faut reentrainer avec plus d'entropie.")
        elif pmax < 0.05:
            print("   -> Politique fortement penchee vers `stay`, mais pas nulle.")
            print("      Plus d'entropie et plus de pas devraient suffire.")
        else:
            print("   -> La politique envisage serieusement l'arret ; c'est")
            print("      l'argmax qui reste sur `stay`. Manque de temps plutot")
            print("      que d'exploration.")

    # --- 3. comportement stochastique --------------------------------------
    print("\n" + "=" * 74)
    print(f"3. ECHANTILLONNAGE STOCHASTIQUE ({args.n_stochastic} episodes)")
    print("=" * 74)

    counts = []
    for s in range(args.n_stochastic):
        env = F1PitStopEnv(fixed_gp=gp, seed=s)
        obs, _ = env.reset(seed=s)
        n, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _, info = env.step(int(action))
            if info["is_pit"] and not info.get("forced_compliance_pit"):
                n += 1
        counts.append(n)

    print(f"   Arrets choisis par episode : moyenne {np.mean(counts):.2f}, "
          f"maximum {max(counts)}, episodes avec au moins un arret : "
          f"{sum(1 for c in counts if c > 0)}/{len(counts)}")
    if max(counts) == 0:
        print("\n   Meme en echantillonnant, l'agent ne s'arrete JAMAIS.")
        print("   Confirme l'effondrement : la masse de probabilite sur les")
        print("   5 actions de pit est numeriquement nulle.")


if __name__ == "__main__":
    main()
