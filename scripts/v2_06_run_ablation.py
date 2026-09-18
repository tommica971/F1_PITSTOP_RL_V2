#!/usr/bin/env python3
"""
V2 — Phase 3b : ablation 2x2 des correctifs structurels
========================================================

Question
---------
Deux correctifs ont sorti les agents de l'effondrement de politique
(95 % du progres atteint a l'episode 8 sur 8 291, zero arret choisi sur 20
episodes stochastiques). Ils ont ete appliques CONJOINTEMENT :

  MIN_STINT_LAPS   plafonne les arrets a ~11 par course au lieu de 57
  a priori STAY    P(stay) = 0.97 a l'initialisation, au lieu de 1/6

Lequel porte l'effet ? Sans ablation, on ne peut pas repondre, et le dossier
signale cette zone comme ouverte.

Protocole
----------
Matrice 2x2, 3 graines par cellule, soit 12 entrainements :

  baseline         min_stint = 0     stay_prior = 1/6
  min_stint seul   min_stint = 5     stay_prior = 1/6
  stay_prior seul  min_stint = 0     stay_prior = 0.97
  complet          min_stint = 5     stay_prior = 0.97

A2C par defaut : c'est sur lui que l'effondrement a ete diagnostique, et il
est le plus sensible aux deux correctifs. DQN a une exploration epsilon-greedy
imposee qui le rend moins dependant de l'a priori — l'y inclure diluerait le
signal.

Le script GENERE les commandes plutot que de lancer lui-meme : chaque run
dure 20 a 30 minutes, et tu dois pouvoir les repartir sur plusieurs fenetres,
en relancer un seul, ou t'arreter en cours.

Usage, depuis la racine V2 :
    python scripts/v2_06_run_ablation.py                 # affiche le plan
    python scripts/v2_06_run_ablation.py --write ablation.bat
    python scripts/v2_06_run_ablation.py --analyse       # apres les runs
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRAIN = "src\\f1_pitstop_rl\\training\\train_v4.py"
UNIFORM = 0.1667          # 1/6 : initialisation uniforme exacte

CELLS = [
    ("baseline",    0, UNIFORM, "aucun correctif"),
    ("minstint",    5, UNIFORM, "MIN_STINT_LAPS seul"),
    ("stayprior",   0, 0.97,    "a priori STAY seul"),
    ("complet",     5, 0.97,    "les deux correctifs"),
]
SEEDS = (1, 2, 3)


def name(cell, seed, algo):
    return f"{algo}_abl_{cell}_s{seed}"


def plan(algo, timesteps):
    out = []
    for cell, ms, sp, _ in CELLS:
        for s in SEEDS:
            out.append(
                f"python {TRAIN} --algo {algo} --seed {s} "
                f"--timesteps {timesteps} --stay-prior {sp} "
                f"--min-stint-laps {ms} --name {name(cell, s, algo)}")
    for cell, _, _, _ in CELLS:
        for s in SEEDS:
            out.append(f"python scripts\\eval_generalization.py "
                       f"--model models/{algo}/{name(cell, s, algo)}.zip --algo {algo}")
    return out


def analyse(algo):
    rows = defaultdict(list)
    for cell, _, _, label in CELLS:
        for s in SEEDS:
            p = ROOT / f"generalisation_{name(cell, s, algo)}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            su, n = d["summary"], d["summary"]["n_gp"]
            rows[cell].append({
                "chosen": su["mean_chosen"],
                "beats_zero": 100 * su["beats_zero"] / n,
                "beats_best": 100 * su["beats_best"] / n,
                "never": 100 * su["never_pits"] / n,
            })

    if not rows:
        sys.exit("Aucun rapport trouve. Lancer d'abord les entrainements et "
                 "les evaluations (voir le plan sans --analyse).")

    print("=" * 88)
    print(f"ABLATION 2x2 — {algo.upper()}  (moyenne ± ecart-type sur les graines)")
    print("=" * 88)
    print(f"{'configuration':<22}{'n':>3}{'arrets choisis':>18}{'bat 0-stop (%)':>18}"
          f"{'sans arret (%)':>18}")
    print("-" * 88)

    stats = {}
    for cell, ms, sp, label in CELLS:
        rs = rows.get(cell)
        if not rs:
            print(f"{label:<22}{'—':>3}   (aucun run)")
            continue

        def ms_(k):
            v = np.array([r[k] for r in rs], float)
            return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        c, bz, nv = ms_("chosen"), ms_("beats_zero"), ms_("never")
        stats[cell] = {"chosen": c, "beats_zero": bz}
        print(f"{label:<22}{len(rs):>3}{f'{c[0]:.2f} ± {c[1]:.2f}':>18}"
              f"{f'{bz[0]:.0f} ± {bz[1]:.0f}':>18}{f'{nv[0]:.0f} ± {nv[1]:.0f}':>18}")

    if len(stats) < 4:
        print("\nMatrice incomplete : les effets ne peuvent pas etre separes.")
        return

    print("\n" + "=" * 88)
    print("EFFETS PRINCIPAUX ET INTERACTION (sur 'bat le 0-stop')")
    print("=" * 88)
    b = {k: stats[k]["beats_zero"][0] for k in stats}
    noise = float(np.mean([stats[k]["beats_zero"][1] for k in stats]))

    eff_ms = ((b["minstint"] - b["baseline"]) + (b["complet"] - b["stayprior"])) / 2
    eff_sp = ((b["stayprior"] - b["baseline"]) + (b["complet"] - b["minstint"])) / 2
    inter = (b["complet"] - b["stayprior"]) - (b["minstint"] - b["baseline"])

    for label, val in (("MIN_STINT_LAPS", eff_ms), ("a priori STAY", eff_sp),
                       ("interaction", inter)):
        verdict = "significatif" if abs(val) > 2 * noise else "dans le bruit"
        print(f"  {label:<18}{val:>+8.1f} pts   (bruit ±{noise:.0f})   {verdict}")

    print("\nLecture : un effet principal est la moyenne des deux contrastes ou")
    print("le facteur passe de inactif a actif. L'interaction mesure si les deux")
    print("correctifs se renforcent (positive) ou se substituent (negative) —")
    print("une interaction fortement negative signifierait qu'un seul des deux")
    print("suffit, et que le second n'apporte rien une fois le premier en place.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", default="a2c", choices=["a2c", "ppo", "dqn"])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--write", default=None, help="Ecrit les commandes dans un .bat")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()

    if args.analyse:
        analyse(args.algo)
        return

    cmds = plan(args.algo, args.timesteps)
    print(f"ABLATION 2x2 — {args.algo.upper()}, {len(SEEDS)} graines, "
          f"{args.timesteps:,} pas\n")
    print(f"{'configuration':<22}{'min_stint':>11}{'stay_prior':>12}")
    print("-" * 45)
    for cell, ms, sp, label in CELLS:
        print(f"{label:<22}{ms:>11}{sp:>12}")
    print(f"\n{len(CELLS) * len(SEEDS)} entrainements + autant d'evaluations.")
    print("Compter 20 a 30 min par entrainement A2C.\n")
    print("-" * 78)
    for c in cmds:
        print(c)

    if args.write:
        out = Path(args.write)
        if not out.is_absolute():
            out = ROOT / out
        out.write_text("@echo off\r\n" + "\r\n".join(cmds) + "\r\n", encoding="utf-8")
        print(f"\n-> {out}")
    print("\nApres les runs :")
    print(f"  python scripts/v2_06_run_ablation.py --algo {args.algo} --analyse")


if __name__ == "__main__":
    main()
