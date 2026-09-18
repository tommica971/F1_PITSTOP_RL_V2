#!/usr/bin/env python3
"""
Etape 3 — Test d'oracle : l'environnement recompense-t-il l'arret ?
===================================================================

C'est le verrou avant tout reentrainement. On ignore completement les modeles
entraines et on evalue des strategies SCRIPTEES sur le pool. Deux issues, qui
menent a des actions opposees :

  * le 0-stop gagne toujours  -> l'environnement est encore faux, l'agent
    aurait raison de ne jamais s'arreter. Ne PAS reentrainer.
  * une strategie a 1 ou 2 arrets gagne -> l'environnement est sain et
    l'echec precedent etait un echec d'apprentissage. Reentrainer.

Deux niveaux d'analyse :
  A. Point mort analytique, sans simulation : cout de rester en piste
     (degradation cumulee) vs cout d'un arret (pit + tour de sortie).
  B. Simulation complete dans F1PitStopEnv, strategies scriptees, N graines,
     comparaison des recompenses ET des positions finales.

La partie B est aussi un livrable de validation d'environnement reutilisable
tel quel dans le dossier (competence C5.2).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("src/f1_pitstop_rl/env", "src/f1_pitstop_rl/config", "src"):
    sys.path.insert(0, str(ROOT / p))

from pace_model import (  # noqa: E402
    DEGRADATION_S, OUT_LAP_PENALTY_S, PIT_STOP_COST_MEAN_S, degradation_cost_s,
)

N_SEEDS = 20
SLICKS = ["SOFT", "MEDIUM", "HARD"]

print("=" * 78)
print("A. POINT MORT ANALYTIQUE (sans simulation)")
print("=" * 78)
print("\nQuestion : sur une course de L tours, rester sur un seul train "
      "coute-t-il\nplus cher que s'arreter une fois ?\n")
print(f"Cout d'un arret : {PIT_STOP_COST_MEAN_S:.2f} s de passage au stand\n")
print(f"{'course':>8} {'compose':>8} {'0-stop (s)':>12} {'1-stop (s)':>12} "
      f"{'ecart':>9}  verdict")

any_pit_wins = False
for total in (44, 57, 70):
    for c in SLICKS:
        cost0 = degradation_cost_s(c, 1, total)
        half = total // 2
        cost1 = (degradation_cost_s(c, 1, half)
                 + degradation_cost_s(c, 1, total - half)
                 + PIT_STOP_COST_MEAN_S + OUT_LAP_PENALTY_S.get(c, 1.0) * 2)
        gain = cost0 - cost1
        wins = gain > 0
        any_pit_wins |= wins
        print(f"{total:>8} {c:>8} {cost0:>12.1f} {cost1:>12.1f} {gain:>+9.1f}  "
              f"{'ARRET GAGNANT' if wins else 'rester en piste'}")

print(f"\n=> {'Au moins une configuration rend l arret rentable.' if any_pit_wins else 'AUCUNE configuration ne rend l arret rentable : ne pas reentrainer.'}")
print("\nRappel des constantes en vigueur :")
for c in SLICKS:
    print(f"   {c:<7} degradation {DEGRADATION_S.get(c, float('nan')):.4f} s/tour"
          f" | tour de sortie {OUT_LAP_PENALTY_S.get(c, float('nan')):.2f} s")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("B. SIMULATION DANS F1PitStopEnv")
print("=" * 78)

try:
    from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND
    from gp_pool_config import GP_POOL
except ImportError as exc:
    print(f"\nEnvironnement non importable ici ({exc}).")
    print("Lancer ce script depuis la racine du projet F1_PITSTOP_RL.")
    sys.exit(0)

COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}


def make_strategy(name, pit_fracs, compound="HARD"):
    """Renvoie une fonction (lap, total, env) -> action."""
    def policy(lap, total, env):
        for f in pit_fracs:
            if lap == max(int(round(f * total)), 3):
                # choisit un compose sec pas encore utilise si possible
                unused = [c for c in ("MEDIUM", "HARD", "SOFT")
                          if c not in env.compounds_used]
                return COMPOUND_TO_ACTION[unused[0] if unused else compound]
        return 0
    return name, policy


STRATEGIES = [
    make_strategy("0-stop", []),
    make_strategy("1-stop @40%", [0.40]),
    make_strategy("1-stop @50%", [0.50]),
    make_strategy("1-stop @60%", [0.60]),
    make_strategy("2-stop @33/66%", [0.33, 0.66]),
]


def run(env, policy, seed):
    obs, info = env.reset(seed=seed)
    total = env.race_total_laps
    cum, done = 0.0, False
    while not done:
        action = policy(env.current_lap, total, env)
        obs, r, done, trunc, info = env.step(action)
        cum += r
    return cum, info["position"], info.get("invalid_strategy", False)


results = defaultdict(list)
positions = defaultdict(list)

for gp in GP_POOL:
    if gp.get("role") not in ("train_wet", "train_dry"):
        continue
    env = F1PitStopEnv(fixed_gp=gp)
    for name, policy in STRATEGIES:
        rs, ps = [], []
        for seed in range(N_SEEDS):
            try:
                r, pos, invalid = run(env, policy, seed)
            except Exception as exc:                       # noqa: BLE001
                print(f"  [{gp['event']}] {name} : echec ({exc})")
                break
            rs.append(r)
            ps.append(pos)
        if rs:
            results[(gp["event"], name)] = rs
            positions[(gp["event"], name)] = ps

print(f"\nRecompense moyenne sur {N_SEEDS} graines (ecart-type entre parentheses)\n")
header = f"{'GP':<28}" + "".join(f"{n:>18}" for n, _ in STRATEGIES)
print(header)
print("-" * len(header))

wins = defaultdict(int)
events = sorted({k[0] for k in results})
for ev in events:
    row = f"{ev[:27]:<28}"
    best, best_name = -1e18, None
    for name, _ in STRATEGIES:
        rs = results.get((ev, name))
        if not rs:
            row += f"{'-':>18}"
            continue
        m, s = float(np.mean(rs)), float(np.std(rs))
        row += f"{m:>11.1f}({s:>4.1f})"
        if m > best:
            best, best_name = m, name
    if best_name:
        wins[best_name] += 1
    print(row + f"   <- {best_name}")

print("\nStrategie gagnante par GP :")
for name, _ in STRATEGIES:
    print(f"   {name:<16} {wins[name]} GP")

zero = wins["0-stop"]
n_gp = sum(wins.values())
print("\n" + "=" * 78)
if n_gp == 0:
    print("VERDICT : aucune simulation exploitable -- verifier les chemins de donnees.")
elif zero == n_gp:
    print("VERDICT : le 0-stop gagne PARTOUT. L'environnement ne recompense")
    print("toujours pas l'arret. NE PAS REENTRAINER -- reprendre la calibration.")
elif zero > n_gp / 2:
    print(f"VERDICT : le 0-stop gagne encore sur {zero}/{n_gp} GP. C'est peut-etre")
    print("legitime sur les circuits a faible degradation, mais verifier que les")
    print("GP a forte degradation (Bahrein, Espagne) ne sont pas dans ce lot.")
else:
    print(f"VERDICT : une strategie a arret gagne sur {n_gp - zero}/{n_gp} GP.")
    print("L'environnement recompense desormais la strategie. Reentrainement justifie.")
print("=" * 78)
print("\nEtape suivante : test de sanite Bahrein 2023, puis reentrainement.")
