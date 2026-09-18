#!/usr/bin/env python3
"""
Evaluation du comportement d'arret — entrainement ET generalisation
====================================================================

Repond a deux questions :

  1. L'agent s'arrete-t-il de son propre chef ? On distingue les arrets
     CHOISIS des arrets imposes par forced_compliance_pit (Article 30.7),
     qui ne sont pas des decisions : l'ancien agent affichait "1 arret" sur
     19 GP de la saison 2025 alors qu'aucun n'en etait un.

  2. Ce comportement tient-il sur les GP JAMAIS VUS a l'entrainement ?
     C'est la question que le jury posera en premier. Les roles sont lus
     depuis gp_pool_config et les resultats separes entrainement /
     generalisation.

Criteres, par ordre d'importance :
  - ARRETS CHOISIS par course. Proche de 0 -> l'agent ne decide rien.
  - TOUR d'arret. Groupes au tour 3 ou au dernier tour -> artefact.
    Repartis vers le milieu et variables selon le circuit -> strategie.
  - RECOMPENSE vs 0-stop et vs meilleur script.
  - ECART entrainement / generalisation -> surapprentissage eventuel.

Usage, depuis la racine du projet :
    python scripts/eval_pit_behaviour.py --model models/a2c/a2c_v4_500k.zip
    python scripts/eval_pit_behaviour.py --model <...> --algo dqn
    python scripts/eval_pit_behaviour.py --model <...> --roles train_dry test_wet
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("src/f1_pitstop_rl/env", "src/f1_pitstop_rl/config"):
    sys.path.insert(0, str(ROOT / p))

from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND  # noqa: E402
from gp_pool_config import GP_POOL  # noqa: E402

ALGOS = {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"}
COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}
TRAIN_ROLES = {"train_wet", "train_dry"}
SCRIPTS = ([0.40], [0.50], [0.60], [0.33, 0.66])


def run_agent(model, gp, seed):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    obs, _ = env.reset(seed=seed)
    cum, chosen, forced, done, info = 0.0, [], [], False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(action))
        cum += r
        if info.get("is_pit"):
            (forced if info.get("forced_compliance_pit") else chosen).append(info["lap"])
    return cum, chosen, forced


def run_script(gp, seed, pit_fracs):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    env.reset(seed=seed)
    total, cum, done = env.race_total_laps, 0.0, False
    while not done:
        action = 0
        for f in pit_fracs:
            if env.current_lap == max(int(round(f * total)), 3):
                unused = [c for c in ("MEDIUM", "HARD", "SOFT")
                          if c not in env.compounds_used]
                action = COMPOUND_TO_ACTION[unused[0] if unused else "HARD"]
        _, r, done, _, _ = env.step(action)
        cum += r
    return cum


def evaluate(model, gp, n_seeds):
    chosen_n, forced_n, laps, rewards = [], [], [], []
    for s in range(n_seeds):
        r, chosen, forced = run_agent(model, gp, s)
        chosen_n.append(len(chosen))
        forced_n.append(len(forced))
        laps += chosen
        rewards.append(r)
    zero = float(np.mean([run_script(gp, s, []) for s in range(n_seeds)]))
    best = max(float(np.mean([run_script(gp, s, f) for s in range(n_seeds)]))
               for f in SCRIPTS)
    return {"chosen": float(np.mean(chosen_n)), "forced": float(np.mean(forced_n)),
            "laps": laps, "reward": float(np.mean(rewards)),
            "zero": zero, "best": best}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="a2c", choices=list(ALGOS))
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--roles", nargs="*", default=None)
    args = ap.parse_args()

    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"Modele introuvable : {path}")

    import stable_baselines3 as sb3
    model = getattr(sb3, ALGOS[args.algo]).load(str(path), device="cpu")

    roles_present = [r for r in dict.fromkeys(g.get("role") for g in GP_POOL) if r]
    roles = args.roles or roles_present
    unknown = set(roles) - set(roles_present)
    if unknown:
        sys.exit(f"Roles absents du pool : {sorted(unknown)}\n"
                 f"Disponibles : {roles_present}")

    print(f"Modele : {path.name}  |  {args.n_seeds} graines par GP")
    print(f"Roles  : {', '.join(roles)}\n")

    groups = defaultdict(list)
    for gp in GP_POOL:
        if gp.get("role") in roles:
            groups["ENTRAINEMENT" if gp["role"] in TRAIN_ROLES
                   else "GENERALISATION"].append(gp)

    summary = {}
    for group in ("ENTRAINEMENT", "GENERALISATION"):
        gps = groups.get(group)
        if not gps:
            continue
        print("=" * 104)
        print(f"{group}  ({len(gps)} GP)")
        print("=" * 104)
        print(f"{'GP':<26}{'role':<11}{'choisis':>9}{'forces':>8}"
              f"{'tour median':>15}{'agent':>10}{'0-stop':>10}{'best script':>13}")
        print("-" * 104)

        rows = []
        for gp in gps:
            try:
                m = evaluate(model, gp, args.n_seeds)
            except Exception as exc:                        # noqa: BLE001
                print(f"{gp['event'][:25]:<26}{gp['role']:<11}  echec : {exc}")
                continue
            lap_txt = (f"{np.median(m['laps']):.0f} [{min(m['laps'])}-{max(m['laps'])}]"
                       if m["laps"] else "-")
            print(f"{gp['event'][:25]:<26}{gp['role']:<11}{m['chosen']:>9.1f}"
                  f"{m['forced']:>8.1f}{lap_txt:>15}{m['reward']:>10.1f}"
                  f"{m['zero']:>10.1f}{m['best']:>13.1f}")
            rows.append(m)

        if not rows:
            continue
        mean_chosen = float(np.mean([r["chosen"] for r in rows]))
        bz = sum(1 for r in rows if r["reward"] > r["zero"])
        bb = sum(1 for r in rows if r["reward"] > r["best"])
        summary[group] = (mean_chosen, bz, bb, len(rows))
        print(f"\n  arrets choisis {mean_chosen:.2f}/course  |  bat le 0-stop "
              f"{bz}/{len(rows)}  |  bat le meilleur script {bb}/{len(rows)}\n")

    print("=" * 104)
    tr, ge = summary.get("ENTRAINEMENT"), summary.get("GENERALISATION")
    if not ge:
        print("Aucun GP de generalisation evalue.")
    elif not tr:
        print(f"Generalisation seule : {ge[0]:.2f} arret/course, "
              f"bat le 0-stop {ge[1]}/{ge[3]}")
    else:
        print(f"Arrets choisis  — entrainement {tr[0]:.2f}  vs  generalisation {ge[0]:.2f}")
        print(f"Bat le 0-stop   — entrainement {tr[1]}/{tr[3]}  vs  generalisation {ge[1]}/{ge[3]}")
        print(f"Bat le script   — entrainement {tr[2]}/{tr[3]}  vs  generalisation {ge[2]}/{ge[3]}\n")
        if ge[0] < 0.3 * max(tr[0], 1e-9):
            print("VERDICT : le comportement d'arret s'effondre hors du pool.")
            print("Surapprentissage — l'agent a memorise ses GP d'entrainement.")
        elif ge[1] < ge[3] / 2:
            print("VERDICT : l'agent s'arrete toujours hors du pool mais ses")
            print("resultats se degradent. Generalisation partielle, a documenter")
            print("en nommant les GP concernes.")
        else:
            print("VERDICT : le comportement tient hors du pool.")
            print("Generalisation confirmee sur les GP jamais vus.")
    print("=" * 104)


if __name__ == "__main__":
    main()
