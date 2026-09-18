#!/usr/bin/env python3
"""
Inspection d'un log d'entrainement — l'effondrement est-il precoce ?
====================================================================

Hypothese a confirmer : 5 des 6 actions sont des arrets. A l'initialisation
la politique est quasi uniforme, l'agent pitte donc ~83% des tours, soit une
cinquantaine d'arrets par course (~-1300 de recompense). La premiere lecon
apprise est "ne jamais pitter", et elle est ecrasante. Une fois P(pit) a
zero, un bonus d'entropie de 0.01 ne pese plus rien face a des avantages de
plusieurs dizaines (le terme vaut au mieux 0.01 * log(6) = 0.018).

Signature attendue dans la courbe : une recompense initiale tres basse, une
remontee brutale sur les premieres centaines d'episodes, puis un plateau
plat jusqu'a la fin. Un plateau atteint en moins de 5% du budget signifie
que 95% de l'entrainement n'apprend plus rien.

Usage, depuis la racine du projet :
    python scripts/inspect_training_log.py --log models/training_logs/a2c_v3env_500k_log.npz
    python scripts/inspect_training_log.py --log <...> --compare <autre.npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(p: Path):
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        sys.exit(f"Log introuvable : {p}")
    d = np.load(p)
    return p.name, d["rewards"], d["lengths"], d["timesteps"]


def summarise(name, rewards, lengths, timesteps):
    n = len(rewards)
    print(f"\n### {name}")
    print(f"{n:,} episodes, {timesteps[-1]:,} timesteps, "
          f"longueur mediane {np.median(lengths):.0f} tours\n")

    print(f"{'tranche':>16}{'episodes':>10}{'recompense moyenne':>22}{'ecart-type':>13}")
    edges = [0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = int(lo * n), int(hi * n)
        if b <= a:
            continue
        seg = rewards[a:b]
        print(f"{f'{lo:.0%}-{hi:.0%}':>16}{b - a:>10}{seg.mean():>22.1f}{seg.std():>13.1f}")

    # a quel moment la courbe atteint-elle 95% de son niveau final ?
    final = rewards[int(0.9 * n):].mean()
    start = rewards[:max(int(0.01 * n), 10)].mean()
    span = final - start
    idx = None
    if span != 0:
        w = max(n // 100, 20)
        smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")
        target = start + 0.95 * span
        hits = np.where(smooth >= target)[0] if span > 0 else np.where(smooth <= target)[0]
        idx = int(hits[0]) if len(hits) else None

    print(f"\n   depart (1% initial) : {start:.1f}")
    print(f"   plateau (10% final) : {final:.1f}")
    if idx is not None:
        pct = 100 * idx / n
        print(f"   95% du progres atteint a l'episode {idx:,} ({pct:.1f}% du budget)")
        if pct < 5:
            print("\n   -> EFFONDREMENT PRECOCE CONFIRME. La politique se fige dans")
            print("      les premiers pour-cents du budget ; le reste de")
            print("      l'entrainement n'apprend plus rien. Augmenter le budget")
            print("      ne servira a rien -- il faut changer l'echelle de")
            print("      recompense et/ou l'exploration.")
        elif pct < 25:
            print("\n   -> Convergence rapide mais pas instantanee ; la politique")
            print("      apprend encore un peu apres le plateau initial.")
        else:
            print("\n   -> Progression etalee : l'agent apprend tout au long du run.")
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--compare", nargs="*", default=[])
    args = ap.parse_args()

    results = [summarise(*load(Path(args.log)))]
    for c in args.compare:
        results.append(summarise(*load(Path(c))))

    if len(results) > 1:
        print("\n" + "=" * 62)
        print("Plateaux compares :", "  |  ".join(f"{r:.1f}" for r in results))


if __name__ == "__main__":
    main()
