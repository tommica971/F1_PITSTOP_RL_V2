"""Patch : extract_model_data.py lit l'etude Optuna choisie (V2 par defaut en V2).

Probleme
--------
load_optuna() lisait un nom de fichier fixe, <algo>_optuna_trials.csv et
<algo>_optuna_best_params.json : l'etude V1. Le dashboard V2 aurait affiche les
hyperparametres V1 (gamma 0.919, ent_coef 1e-6) a cote d'un modele V2.
De plus, le JSON V2 nomme sa cle "best_value_heldout" (objectif mesure sur les
GP hors pool) et non "best_value" : lecture impossible en l'etat.

Correctif
---------
- option --optuna-study (defaut "optuna_v2") : lit <algo>_<study>_*.csv/json
- cle best_value ou best_value_heldout, avec le type d'objectif explicite
- essais sans valeur (PRUNED) ecartes ; nombre d'essais elagues reporte
- metadonnees V2 conservees (graines, GP de validation, a priori STAY)

Usage, depuis la racine du projet :
    python scripts/apply_patch_extract_optuna_v2.py
"""
from pathlib import Path
import shutil
import sys

TARGET = Path(__file__).resolve().parent / "extract_model_data.py"

REPLACEMENTS = [
    (
        'def load_optuna(algo: str):\n'
        '    trials_path = LOGS_DIR / f"{algo}_optuna_trials.csv"\n'
        '    best_path = LOGS_DIR / f"{algo}_optuna_best_params.json"',
        'def load_optuna(algo: str, study: str = "optuna_v2"):\n'
        '    trials_path = LOGS_DIR / f"{algo}_{study}_trials.csv"\n'
        '    best_path = LOGS_DIR / f"{algo}_{study}_best_params.json"',
    ),
    (
        '            if row.get("state") != "COMPLETE":\n'
        '                continue',
        '            if row.get("state", "COMPLETE") != "COMPLETE" or row.get("value") in (None, ""):\n'
        '                n_pruned += 1\n'
        '                continue',
    ),
    (
        '    trials = []\n'
        '    with open(trials_path, encoding="utf-8") as f:',
        '    trials = []\n'
        '    n_pruned = 0\n'
        '    with open(trials_path, encoding="utf-8") as f:',
    ),
    (
        '    return {"trials": trials, "best_value": best["best_value"], "best_params": best["best_params"],\n'
        '            "timesteps_per_trial": best.get("timesteps_per_trial")}',
        '    heldout = "best_value_heldout" in best\n'
        '    return {\n'
        '        "study": study,\n'
        '        "trials": trials,\n'
        '        "n_pruned": n_pruned,\n'
        '        "best_value": best.get("best_value_heldout", best.get("best_value")),\n'
        '        # heldout : objectif mesure sur les GP hors pool ; train : sur le pool\n'
        '        "objective": "heldout" if heldout else "train",\n'
        '        "best_params": best["best_params"],\n'
        '        "timesteps_per_trial": best.get("timesteps_per_trial"),\n'
        '        "n_train_seeds": best.get("n_train_seeds"),\n'
        '        "validation_gps": best.get("validation_gps"),\n'
        '        "stay_prior": best.get("stay_prior"),\n'
        '    }',
    ),
    (
        '    parser.add_argument("--compare", nargs="*", default=[], help="Autres modeles a inclure pour comparaison")\n',
        '    parser.add_argument("--compare", nargs="*", default=[], help="Autres modeles a inclure pour comparaison")\n'
        '    parser.add_argument("--optuna-study", default="optuna_v2",\n'
        '                        help="Etude Optuna a lire : optuna_v2 (defaut) ou optuna (etude V1)")\n',
    ),
    (
        '    print(f"Optuna : {args.algo}")\n'
        '    optuna_data = load_optuna(args.algo)',
        '    print(f"Optuna : {args.algo} / {args.optuna_study}")\n'
        '    optuna_data = load_optuna(args.algo, args.optuna_study)',
    ),
]


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    if "--optuna-study" in src:
        print("Deja patche, rien a faire.")
        return 0
    for old, new in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            print(f"ECHEC : motif trouve {n} fois (attendu 1) :\n{old[:120]}")
            print("Fichier NON modifie.")
            return 1
        src = src.replace(old, new)
    backup = TARGET.with_suffix(".py.bak_optuna_v1")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")
    print(f"Patche : {TARGET}\nSauvegarde : {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
