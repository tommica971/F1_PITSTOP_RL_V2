#!/usr/bin/env python3
"""
V2 — Controle : d'ou viennent les recompenses positives ?
==========================================================

Constat apres la correction du biais de rythme : certaines recompenses totales
deviennent POSITIVES — Monza +16.6 (DQN s2), Abu Dhabi +3.6, Singapour +0.7,
et l'oracle lui-meme a +15.4 sur Monza.

La recompense est exprimee en equivalent-temps (1 seconde perdue = -1 point),
plus un bonus terminal de position (bareme F1 x 4, jusqu'a +100). Un total
positif est donc possible en principe : il suffit que le bonus terminal
depasse le temps perdu. Mais il peut aussi signaler un defaut du correctif
introduit en Phase 1 — si le differentiel d'age sur-credite un agent qui
pitte tot, il gagnerait du temps sans raison physique.

Les deux cas se distinguent en decomposant la recompense.

Ce que le script fait
----------------------
Pour chaque GP demande, rejoue le 0-stop et une strategie a un arret, puis
somme separement les composantes de RewardBreakdown :

    time_component            temps gagne/perdu vs la reference du tour
    position_delta_component  gains de rang en cours de course
    dirty_air_component       penalite d'air sale
    compliance_component      penalite progressive Article 30.7
    terminal_component        bonus de position finale (bareme x 4)

Lecture :
  * total positif porte par terminal_component -> comportement ATTENDU,
    l'agent finit haut et le bareme domine le temps perdu.
  * total positif porte par time_component -> ANOMALIE : l'agent gagnerait du
    temps sur la reference du peloton tour apres tour, ce qui n'a pas de sens
    physique pour une Alpine. Le differentiel d'age sur-credite.

Aucune ecriture. Lancer depuis la racine V2 :
    python scripts/v2_04_check_reward_sign.py
    python scripts/v2_04_check_reward_sign.py --gp "Italian Grand Prix"
"""
import argparse
import dataclasses
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in ("src/f1_pitstop_rl/env", "src/f1_pitstop_rl/config"):
    sys.path.insert(0, str(ROOT / p))

from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND  # noqa: E402
from gp_pool_config import GP_POOL  # noqa: E402

COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}
INFERENCE_ROLE = "season_2025_inference"
# GP ou des recompenses positives ont ete observees, plus deux temoins
DEFAULT_GPS = ["Italian Grand Prix", "Abu Dhabi Grand Prix",
               "Singapore Grand Prix", "Bahrain Grand Prix"]
FIELDS = ["time_component", "position_delta_component", "dirty_air_component",
          "compliance_component", "terminal_component"]


def role_of(event):
    for gp in GP_POOL:
        if gp["event"] == event:
            return gp["role"]
    return INFERENCE_ROLE


def run(event, fracs, seed):
    gp = {"season": 2025, "event": event, "role": role_of(event), "known_issues": []}
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    env.reset(seed=seed)
    total, done, parts = env.race_total_laps, False, defaultdict(float)
    pits, ages = [], []
    while not done:
        action = 0
        for f in fracs:
            if env.current_lap == max(int(round(f * total)), 3):
                unused = [c for c in ("MEDIUM", "HARD", "SOFT")
                          if c not in env.compounds_used]
                action = COMPOUND_TO_ACTION[unused[0] if unused else "HARD"]
        lap, age = env.current_lap, env.own_tyre_age
        pack = env.pace_ctx.pack_age_for(lap) if hasattr(env.pace_ctx, "pack_age_for") else float("nan")
        _, r, done, _, info = env.step(action)
        if pack == pack:
            ages.append(age - pack)
        bd = info.get("reward_breakdown")
        if bd is not None:
            for k, v in dataclasses.asdict(bd).items():
                if k in FIELDS:
                    parts[k] += v
        if info.get("is_pit"):
            pits.append(lap)
    return parts, info.get("position"), pits, ages


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gp", nargs="*", default=DEFAULT_GPS)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    print("=" * 96)
    print("DECOMPOSITION DE LA RECOMPENSE — d'ou vient le signe ?")
    print("=" * 96)

    anomalies = []
    for event in args.gp:
        print(f"\n### {event}")
        print(f"{'strategie':<16}{'total':>9}{'temps':>10}{'rangs':>9}{'air sale':>10}"
              f"{'conformite':>12}{'terminal':>10}{'pos':>6}{'arrets':>10}")
        print("-" * 96)
        for name, fracs in (("0-stop", []), ("1-stop @50%", [0.50])):
            runs = [run(event, fracs, s) for s in range(args.n_seeds)]
            agg = {k: float(np.mean([r[0][k] for r in runs])) for k in FIELDS}
            total = sum(agg.values())
            pos = float(np.median([r[1] for r in runs]))
            pits = runs[0][2]
            print(f"{name:<16}{total:>9.1f}{agg['time_component']:>10.1f}"
                  f"{agg['position_delta_component']:>9.1f}"
                  f"{agg['dirty_air_component']:>10.1f}"
                  f"{agg['compliance_component']:>12.1f}"
                  f"{agg['terminal_component']:>10.1f}{pos:>6.0f}"
                  f"{str(pits):>10}")
            if total > 0 and agg["time_component"] > 0:
                anomalies.append((event, name, agg["time_component"]))

        # ecart d'age a la reference, sur la strategie a un arret
        _, _, _, ages = run(event, [0.50], 0)
        if ages:
            a = np.array(ages)
            print(f"{'':<16}ecart d'age a la reference : mediane {np.median(a):+.1f}, "
                  f"min {a.min():+.0f}, max {a.max():+.0f} tours "
                  f"({100 * np.mean(np.abs(a) >= 20):.0f} % a la borne)")

    print("\n" + "=" * 96)
    if anomalies:
        print("ANOMALIE : recompense positive portee par le TEMPS, pas par le bonus")
        print("terminal. L'agent gagnerait du temps sur la reference du peloton tour")
        print("apres tour, ce qui n'a pas de sens physique. Le differentiel d'age")
        print("introduit en Phase 1 sur-credite probablement les pneus jeunes.\n")
        for ev, st, t in anomalies:
            print(f"   {ev} — {st} : composante temps {t:+.1f}")
        print("\nPiste : verifier la ligne 'ecart d'age'. Si l'agent est durablement")
        print("plus jeune que le peloton (mediane tres negative), il encaisse un")
        print("credit permanent. Une correction possible est de ne creer que")
        print("l'ecart POSITIF (pneu plus vieux = penalite) sans crediter l'ecart")
        print("negatif, ou de recentrer sur l'age median de l'agent lui-meme.")
    else:
        print("PAS D'ANOMALIE. Les recompenses positives sont portees par le bonus")
        print("terminal (bareme de points x 4, jusqu'a +100), pas par la composante")
        print("temps. C'est le comportement attendu : un agent qui finit dans les")
        print("points peut afficher un total positif malgre du temps perdu.")
        print("\nA noter pour le dossier : les recompenses V1 et V2 ne sont donc pas")
        print("comparables en valeur absolue. Seuls les TAUX (bat le 0-stop, bat")
        print("l'oracle, potentiel capte) le sont.")
    print("=" * 96)


if __name__ == "__main__":
    main()
