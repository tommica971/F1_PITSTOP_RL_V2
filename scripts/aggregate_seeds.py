#!/usr/bin/env python3
"""
Agregation multi-graines — la comparaison d'algorithmes est-elle reproductible ?
================================================================================

Pourquoi ce script
-------------------
Les resultats de comparaison reposaient sur UN run par algorithme. Or la
variabilite entre configurations proches s'est revelee importante : DQN est
passe de 1/8 a 4/8 contre le meilleur script entre deux versions, et A2C de
10/19 a 4/19 en generalisation apres un simple changement de borne
d'observation.

Affirmer "A2C generalise mieux que DQN et PPO" sur une graine par algorithme
n'est pas defendable devant un jury de certification. Ce script agrege
plusieurs graines et rapporte moyenne et ecart-type, ce qui permet de dire si
l'ecart entre algorithmes depasse le bruit.

Il lit les rapports produits par eval_generalization.py (generalisation_*.json)
et n'exige aucun recalcul.

Usage, depuis la racine du projet :
    python scripts/aggregate_seeds.py
    python scripts/aggregate_seeds.py --pattern "generalisation_*_v4_*.json"
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def algo_of(name: str) -> str:
    for a in ("a2c", "ppo", "dqn"):
        if name.lower().startswith(a):
            return a.upper()
    return "?"


def seed_of(name: str) -> str:
    m = re.search(r"_s(\d+)", name)
    return m.group(1) if m else "0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="generalisation_*.json")
    ap.add_argument("--dir", default=".")
    args = ap.parse_args()

    files = sorted((ROOT / args.dir).glob(args.pattern))
    if not files:
        raise SystemExit(f"Aucun rapport trouve ({args.pattern}). "
                         "Lancer d'abord eval_generalization.py.")

    runs = defaultdict(list)
    per_gp = defaultdict(lambda: defaultdict(list))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        algo = d.get("algo", algo_of(d["model"])).upper()
        s = d["summary"]
        n = s["n_gp"]
        runs[algo].append({
            "model": d["model"], "seed": seed_of(d["model"]),
            "chosen": s["mean_chosen"],
            "beats_zero": 100 * s["beats_zero"] / n,
            "beats_best": 100 * s["beats_best"] / n,
            "never": 100 * s["never_pits"] / n,
        })
        for row in d["generalisation"]:
            per_gp[row["event"]][algo].append(row["reward"] > row["best"])

    print(f"{len(files)} rapports | "
          + " ".join(f"{a}:{len(v)}" for a, v in sorted(runs.items())) + "\n")

    print("=" * 88)
    print("RESULTATS PAR ALGORITHME (moyenne ± ecart-type sur les graines)")
    print("=" * 88)
    print(f"{'algo':<7}{'n':>4}{'arrets choisis':>20}{'bat 0-stop (%)':>20}"
          f"{'bat script (%)':>20}{'sans arret (%)':>17}")
    print("-" * 88)

    stats = {}
    for algo, rs in sorted(runs.items()):
        def ms(k):
            v = np.array([r[k] for r in rs], dtype=float)
            return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        c, bz, bb, nv = (ms("chosen"), ms("beats_zero"),
                         ms("beats_best"), ms("never"))
        stats[algo] = {"chosen": c, "beats_zero": bz, "beats_best": bb}
        print(f"{algo:<7}{len(rs):>4}{f'{c[0]:.2f} ± {c[1]:.2f}':>20}"
              f"{f'{bz[0]:.0f} ± {bz[1]:.0f}':>20}"
              f"{f'{bb[0]:.0f} ± {bb[1]:.0f}':>20}"
              f"{f'{nv[0]:.0f} ± {nv[1]:.0f}':>17}")

    if any(len(v) < 2 for v in runs.values()):
        print("\n  Une seule graine pour au moins un algorithme : les ecarts-types")
        print("  valent 0 par construction et ne mesurent rien. Relancer avec")
        print("  plusieurs graines avant de conclure.")
    else:
        print("\n" + "=" * 88)
        print("L'ECART ENTRE ALGORITHMES DEPASSE-T-IL LE BRUIT ?")
        print("=" * 88)
        algos = sorted(stats)
        for i, a in enumerate(algos):
            for b in algos[i + 1:]:
                for k, label in (("beats_best", "bat le script"),
                                 ("beats_zero", "bat le 0-stop")):
                    ma, sa = stats[a][k]
                    mb, sb = stats[b][k]
                    pooled = np.sqrt(sa ** 2 + sb ** 2)
                    d = abs(ma - mb)
                    verdict = ("ecart significatif" if pooled > 0 and d > 2 * pooled
                               else "dans le bruit")
                    print(f"  {label:<16} {a} {ma:5.0f}%  vs  {b} {mb:5.0f}%   "
                          f"ecart {d:4.0f} pts, bruit ±{pooled:.0f}  -> {verdict}")

    print("\n" + "=" * 88)
    print("GP OU AUCUN ALGORITHME NE BAT LE MEILLEUR SCRIPT")
    print("=" * 88)
    hopeless = [gp for gp, d in per_gp.items()
                if not any(any(v) for v in d.values())]
    for gp in sorted(hopeless):
        print(f"  {gp}")
    print(f"\n  {len(hopeless)}/{len(per_gp)} GP. Ce sont les cas a documenter")
    print("  comme limite du modele, et non comme faiblesse d'un algorithme.")


if __name__ == "__main__":
    main()
