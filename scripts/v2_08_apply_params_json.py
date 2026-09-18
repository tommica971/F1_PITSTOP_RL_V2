#!/usr/bin/env python3
"""
V2 — Ajoute --params-json a train_v4.py
========================================

Pourquoi un fichier plutot que la ligne de commande
-----------------------------------------------------
L'etude Optuna produit six hyperparametres a quatorze decimales. Les recopier
a la main dans une commande est une source d'erreur silencieuse : un chiffre
perdu et le modele entraine n'est plus celui que l'etude a selectionne, sans
qu'aucun message ne le signale.

Lire directement le JSON produit par l'etude supprime cette transcription.
Pour la tracabilite (C5.3.1), cela permet d'affirmer que le modele final est
entraine depuis le fichier de l'etude, sans intervention manuelle.

Comportement
-------------
--params-json <fichier> remplace les hyperparametres par defaut par ceux du
fichier. Les options explicites de la ligne de commande (--gamma, --ent-coef,
--n-steps) restent prioritaires, pour pouvoir deroger ponctuellement sans
editer le fichier.

Le script accepte les deux formats produits par le projet :
  {"best_params": {...}}   sortie d'Optuna
  {...}                    dictionnaire d'hyperparametres direct

Idempotent.

Usage, depuis la racine V2 :
    python scripts/v2_08_apply_params_json.py --dry-run
    python scripts/v2_08_apply_params_json.py

Puis :
    python src/f1_pitstop_rl/training/train_v4.py --algo a2c --seed 1 ^
        --params-json models/training_logs/a2c_optuna_v2_best_params.json ^
        --name a2c_v2opt_s1
"""
import argparse
import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "src" / "f1_pitstop_rl" / "training" / "train_v4.py"
BACKUP = TRAIN.with_suffix(".py.bak_preparamsjson")
MARKER = "--params-json"

OLD_ARG = '    ap.add_argument("--seed", type=int, default=SEED)'
NEW_ARG = ('    ap.add_argument("--params-json", default=None,\n'
           '                    help="Hyperparametres issus d\'une etude Optuna. "\n'
           '                         "Evite la recopie manuelle et garantit que le "\n'
           '                         "modele entraine est bien celui selectionne.")\n'
           '    ap.add_argument("--seed", type=int, default=SEED)')

OLD_PARAMS = "    params = dict(PARAMS[args.algo])"
NEW_PARAMS = '''    params = dict(PARAMS[args.algo])
    if args.params_json:
        import json
        p = Path(args.params_json)
        if not p.is_absolute():
            p = ROOT_DIR / p
        if not p.exists():
            sys.exit(f"Fichier d'hyperparametres introuvable : {p}")
        loaded = json.loads(p.read_text(encoding="utf-8"))
        loaded = loaded.get("best_params", loaded)
        # Seules les cles connues de l'algorithme sont reprises : le JSON
        # d'Optuna contient aussi des metadonnees (score, graines, GP de
        # validation) que SB3 rejetterait.
        unknown = [k for k in loaded if k not in params]
        params.update({k: v for k, v in loaded.items() if k in params})
        print(f"Hyperparametres lus depuis {p.name}")
        if unknown:
            print(f"  cles ignorees (metadonnees) : {', '.join(unknown)}")'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not TRAIN.exists():
        sys.exit(f"ERREUR : {TRAIN} introuvable.")
    src = TRAIN.read_text(encoding="utf-8")
    if MARKER in src:
        print("Patch deja applique -- rien a faire.")
        return

    out = src
    for label, old, new in (("option --params-json", OLD_ARG, NEW_ARG),
                            ("lecture du fichier", OLD_PARAMS, NEW_PARAMS)):
        if out.count(old) != 1:
            sys.exit(f"  ECHEC   {label} : motif trouve {out.count(old)} fois "
                     "(attendu 1).\nRien n'a ete ecrit.")
        out = out.replace(old, new)
        print(f"  OK      {label}")

    # ROOT_DIR : la racine du projet, pour resoudre un chemin relatif
    if "ROOT_DIR" not in out:
        anchor = "sys.path.insert(0, str(Path(__file__).parent))"
        if out.count(anchor) != 1:
            sys.exit("  ECHEC   ancre de chemin introuvable.\nRien n'a ete ecrit.")
        out = out.replace(anchor, anchor + "\nROOT_DIR = Path(__file__).resolve().parents[3]")
        print("  OK      racine du projet resolue")

    try:
        ast.parse(out)
    except SyntaxError as exc:
        sys.exit(f"  ECHEC   resultat invalide ({exc}).\nRien n'a ete ecrit.")

    if args.dry_run:
        print("\n--dry-run : tous les blocs s'appliquent, rien n'est ecrit.")
        return
    if not BACKUP.exists():
        shutil.copy(TRAIN, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    TRAIN.write_text(out, encoding="utf-8")
    print(f"Patch applique : {TRAIN}")
    print("\nEntrainement des 3 graines avec les hyperparametres de l'etude :")
    for s in (1, 2, 3):
        print(f"  python src\\f1_pitstop_rl\\training\\train_v4.py --algo a2c "
              f"--seed {s} --params-json "
              f"models/training_logs/a2c_optuna_v2_best_params.json "
              f"--name a2c_v2opt_s{s}")


if __name__ == "__main__":
    main()
