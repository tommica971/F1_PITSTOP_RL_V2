"""Patch : oracle 0-stop inclus, et meme jeu de GP hors pool partout.

1. Oracle sans le 0-stop (eval_generalization.py, build_comparison_data.py)
   L'oracle etait la meilleure des 4 fenetres d'arret, 0-stop EXCLU. Sur une
   course ou ne pas s'arreter est optimal (Miami 2025 : 0-stop -36.1, meilleure
   fenetre -40.1), un agent qui s'abstient "battait l'oracle" sans avoir rien
   decide. Correctif : oracle = meilleure strategie connue a posteriori,
   0-stop INCLUS. Battre l'oracle exige alors de faire strictement mieux que
   toutes les strategies scriptees, abstention comprise.

2. Jeu hors pool different (build_comparison_data.py)
   Le resume "hors pool" excluait seulement les GP d'ENTRAINEMENT : la
   Belgique 2025, GP de TEST du pool (role test_wet), y etait comptee -> 20 GP,
   contre 19 dans eval_generalization.py. Correctif : exclusion de tout GP de
   GP_POOL (entrainement et test), comme dans l'evaluation de reference.
   role_of() compare deja (saison, course) : pas de confusion Japon 2025/2026.

Usage, depuis la racine du projet :
    python scripts/apply_patch_oracle_pool.py
"""
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent

PATCHES = {
    "eval_generalization.py": [
        (
            '"reward": float(np.mean(rewards)), "zero": zero, "best": best}',
            '"reward": float(np.mean(rewards)), "zero": zero,\n'
            '            # oracle = meilleure strategie a posteriori, 0-stop INCLUS\n'
            '            "best": max(best, zero), "best_script": best}',
        ),
    ],
    "build_comparison_data.py": [
        (
            '        oracle = float(np.mean([r["reward"] for r in oracle_runs]))\n',
            '        oracle = float(np.mean([r["reward"] for r in oracle_runs]))\n'
            '        # Oracle = meilleure strategie connue a posteriori, 0-stop INCLUS.\n'
            '        # Sans cela, sur une course ou ne pas s\'arreter est optimal, un\n'
            '        # agent qui s\'abstient "bat l\'oracle" sans rien decider (Miami).\n'
            '        if zero >= oracle:\n'
            '            oracle_name, oracle_runs, oracle = "0-stop", scripts["0-stop"], zero\n',
        ),
        (
            '        in_pool = role_of(event) in TRAIN_ROLES\n',
            '        in_pool = role_of(event) in TRAIN_ROLES\n'
            '        # GP_POOL entier (entrainement + test) : meme jeu hors pool que\n'
            '        # eval_generalization.py (la Belgique 2025 est un GP de test)\n'
            '        in_gp_pool = role_of(event) is not None\n',
        ),
        (
            '        print(f"{event[:29]:<30}{\'oui\' if in_pool else \'-\':>6}{a_mean:>10.1f}"',
            '        pool_txt = "train" if in_pool else "test" if in_gp_pool else "-"\n'
            '        print(f"{event[:29]:<30}{pool_txt:>6}{a_mean:>10.1f}"',
        ),
        (
            '            "in_train_pool": in_pool,\n',
            '            "in_train_pool": in_pool,\n'
            '            "in_gp_pool": in_gp_pool,\n',
        ),
        (
            '    out_pool = [r for r in rows if not r["in_train_pool"]]',
            '    out_pool = [r for r in rows if not r["in_gp_pool"]]',
        ),
        (
            '"Borne haute choisie A POSTERIORI parmi 4 fenetres. Aucun "',
            '"Borne haute choisie A POSTERIORI parmi 4 fenetres ET le 0-stop. Aucun "',
        ),
    ],
}


def main() -> int:
    plans = {}
    for name, reps in PATCHES.items():
        path = HERE / name
        src = path.read_text(encoding="utf-8")
        if "0-stop INCLUS" in src:
            print(f"{name} : deja patche")
            continue
        for old, new in reps:
            n = src.count(old)
            if n != 1:
                print(f"ECHEC {name} : motif trouve {n} fois (attendu 1) :\n  {old[:100]}")
                print("AUCUN fichier modifie.")
                return 1
            src = src.replace(old, new)
        plans[path] = src
    for path, src in plans.items():            # ecriture seulement si tout est OK
        shutil.copy2(path, path.with_suffix(".py.bak_oracle"))
        path.write_text(src, encoding="utf-8")
        compile(src, str(path), "exec")
        print(f"Patche : {path.name}  (sauvegarde .bak_oracle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
