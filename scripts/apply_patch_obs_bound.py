#!/usr/bin/env python3
"""
Patch — borne de tyre_age_laps dans l'espace d'observation
===========================================================

Bug. OBS_HIGH borne `tyre_age_laps` a 45 tours. La borne a ete fixee en
Phase 2.1 pour des relais de longueur normale, sans anticiper qu'un agent
capable de 0-stop atteindrait 70 tours sur le meme train. L'observation est
`np.clip(obs, OBS_LOW, OBS_HIGH)` : au-dela de 45 tours, l'agent ne percoit
donc PLUS son pneu vieillir. Le signal qui devrait declencher l'arret est
gele.

Mesure (scripts/diagnose_trigger.py, modele a2c_v4_500k) :

    Italian Grand Prix     P(pit) max au tour 45, age pneu affiche 45 (sature)
    Mexico City Grand Prix P(pit) max au tour 51, age pneu affiche 45 (sature)

Les deux font partie des 9 GP ou l'agent ne s'arrete jamais.

Correctif : borne portee a 80, soit au-dela de la course la plus longue du
pool (72 tours). Aucun episode ne peut plus saturer cette dimension.

ATTENTION : l'espace d'observation change, donc les modeles entraines avant
ce patch ne sont plus compatibles. Il faut reentrainer.

Le script est idempotent et n'ecrit rien si le texte attendu est absent.

Usage, depuis la racine du projet :
    python scripts/apply_patch_obs_bound.py --dry-run
    python scripts/apply_patch_obs_bound.py
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "src" / "f1_pitstop_rl" / "env" / "f1_pitstop_env.py"
BACKUP = ENV.with_suffix(".py.bak_preobsbound")

OLD = "    45,     # tyre_age_laps (borne large)"
NEW = ("    80,     # tyre_age_laps -- porte de 45 a 80 (cf.\n"
       "            # scripts/apply_patch_obs_bound.py). A 45, l'observation\n"
       "            # saturait des qu'un relais depassait 45 tours, ce qui\n"
       "            # arrive systematiquement en 0-stop : l'agent cessait de\n"
       "            # percevoir le vieillissement de son pneu. Mesure sur\n"
       "            # a2c_v4_500k : Monza et Mexico atteignaient leur P(pit)\n"
       "            # maximale avec un age affiche bloque a 45.\n"
       "            # 80 > course la plus longue du pool (72 tours).")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not ENV.exists():
        sys.exit(f"ERREUR : {ENV} introuvable.\n"
                 "Lancer depuis la racine du projet F1_PITSTOP_RL.")
    src = ENV.read_text(encoding="utf-8")

    if "porte de 45 a 80" in src:
        print("Patch deja applique -- rien a faire.")
        return

    n = src.count(OLD)
    if n != 1:
        print(f"  ECHEC : ligne attendue trouvee {n} fois (attendu 1).")
        print(f"  Texte cherche : {OLD!r}")
        print("\nRien n'a ete ecrit. Modifier OBS_HIGH a la main : passer la")
        print("premiere valeur (tyre_age_laps) de 45 a 80.")
        sys.exit(1)

    out = src.replace(OLD, NEW)
    import ast
    try:
        ast.parse(out)
    except SyntaxError as exc:
        sys.exit(f"ERREUR : resultat invalide ({exc}). Rien n'a ete ecrit.")

    print("  OK      borne tyre_age_laps : 45 -> 80")
    if args.dry_run:
        print("\n--dry-run : le bloc s'applique, rien n'est ecrit.")
        return

    if not BACKUP.exists():
        shutil.copy(ENV, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    ENV.write_text(out, encoding="utf-8")
    print(f"Patch applique : {ENV}")
    print("\nL'espace d'observation a change : les modeles v4 ne sont plus")
    print("compatibles. Reentrainer avant toute evaluation :")
    print("  python src/f1_pitstop_rl/training/train_v4.py --algo a2c "
          "--name a2c_v5_500k")
    print("  python scripts/eval_generalization.py "
          "--model models/a2c/a2c_v5_500k.zip")


if __name__ == "__main__":
    main()
