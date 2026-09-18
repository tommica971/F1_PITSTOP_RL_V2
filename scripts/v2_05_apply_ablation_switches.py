#!/usr/bin/env python3
"""
V2 — Phase 3a : rendre MIN_STINT_LAPS desactivable pour l'ablation
===================================================================

Pourquoi
---------
Deux correctifs structurels ont sorti les agents de l'effondrement de
politique, et ils ont ete appliques CONJOINTEMENT :

  MIN_STINT_LAPS = 5   une action de pit avant 5 tours d'age est convertie
                       en STAY. Plafonne les arrets a ~11 par course au lieu
                       de 57, ramenant la catastrophe initiale de -1300 a
                       environ -250.
  a priori STAY        le biais de sortie du reseau est decale pour que
                       P(stay) = 0.97 a l'initialisation, au lieu de 1/6.
                       L'agent demarre a ~1.7 arret par course au lieu de 50.

Leur effet respectif n'a jamais ete isole. Le dossier signale cette zone
comme ouverte. L'ablation 2x2 y repond, mais elle exige de pouvoir DESACTIVER
MIN_STINT_LAPS, aujourd'hui code en dur.

Ce que le patch fait
---------------------
MIN_STINT_LAPS reste la valeur par defaut du module. F1PitStopEnv accepte en
plus un parametre `min_stint_laps` : passe a 0, la contrainte est inactive et
l'environnement retrouve le comportement d'avant correctif. Aucun appel
existant n'est modifie — le defaut preserve le comportement actuel.

L'a priori STAY est deja parametrable via `--stay-prior` de train_v4.py
(0.1667 reproduit exactement l'initialisation uniforme), il n'y a rien a
patcher de ce cote.

Idempotent. N'ecrit rien si un bloc echoue.

Usage, depuis la racine V2 :
    python scripts/v2_05_apply_ablation_switches.py --dry-run
    python scripts/v2_05_apply_ablation_switches.py
"""
import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "src" / "f1_pitstop_rl" / "env" / "f1_pitstop_env.py"
COMMON = ROOT / "src" / "f1_pitstop_rl" / "training" / "common.py"
TRAIN = ROOT / "src" / "f1_pitstop_rl" / "training" / "train_v4.py"
BACKUP = ENV.with_suffix(".py.bak_preablation")
MARKER = "self.min_stint_laps"

# --- common.py : make_train_env doit pouvoir transmettre le parametre -------
COMMON_OLD = """def make_train_env(seed: int = SEED, gp_weights: dict | None = None) -> Monitor:
    env = F1PitStopEnv(gp_roles=TRAIN_GP_ROLES, seed=seed, gp_weights=gp_weights)
    return Monitor(env)"""
COMMON_NEW = """def make_train_env(seed: int = SEED, gp_weights: dict | None = None,
                   min_stint_laps: int | None = None) -> Monitor:
    \"\"\"min_stint_laps : None -> valeur du module ; 0 -> contrainte desactivee
    (ablation 2x2, cf. scripts/v2_05_apply_ablation_switches.py).\"\"\"
    env = F1PitStopEnv(gp_roles=TRAIN_GP_ROLES, seed=seed, gp_weights=gp_weights,
                       min_stint_laps=min_stint_laps)
    return Monitor(env)"""

# --- train_v4.py : exposer l'option en ligne de commande --------------------
TRAIN_OLD_ARG = '    ap.add_argument("--reward-scale", type=float, default=0.1)'
TRAIN_NEW_ARG = ('    ap.add_argument("--reward-scale", type=float, default=0.1)\n'
                 '    ap.add_argument("--min-stint-laps", type=int, default=None,\n'
                 '                    help="Longueur minimale de relais. 0 desactive "\n'
                 '                         "la contrainte (ablation). Defaut : valeur "\n'
                 '                         "du module.")')
TRAIN_OLD_ENV = "    env = make_train_env(seed=args.seed)"
TRAIN_NEW_ENV = ("    env = make_train_env(seed=args.seed,\n"
                 "                         min_stint_laps=args.min_stint_laps)")

OLD_STEP = ('        if action != 0 and not is_last_lap '
            'and self.own_tyre_age < MIN_STINT_LAPS:')
NEW_STEP = ('        if (action != 0 and not is_last_lap\n'
            '                and self.own_tyre_age < self.min_stint_laps):')


def patch_init(src: str):
    """Ajoute min_stint_laps a la signature de __init__ et l'enregistre."""
    m = re.search(r"(    def __init__\(\s*self,\s*)(.*?)(\n    \):|\):)", src, re.S)
    if not m:
        return None, "signature de __init__ introuvable"
    head, params, close = m.groups()
    if "min_stint_laps" in params:
        return src, None
    sep = ",\n        " if "\n" in params else ", "
    new_params = params.rstrip().rstrip(",") + sep + "min_stint_laps: int | None = None"
    src = src[:m.start()] + head + new_params + close + src[m.end():]

    # enregistrement : juste apres la fin de la signature
    anchor = head + new_params + close
    idx = src.index(anchor) + len(anchor)
    assign = (
        "\n        # Longueur minimale de relais. None -> valeur du module.\n"
        "        # 0 desactive la contrainte : sert a l'ablation 2x2\n"
        "        # (scripts/v2_05_apply_ablation_switches.py).\n"
        "        self.min_stint_laps = (MIN_STINT_LAPS if min_stint_laps is None\n"
        "                               else int(min_stint_laps))")
    return src[:idx] + assign + src[idx:], None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not ENV.exists():
        sys.exit(f"ERREUR : {ENV} introuvable.")
    src = ENV.read_text(encoding="utf-8")
    if MARKER in src:
        print("Patch deja applique -- rien a faire.")
        return
    if "MIN_STINT_LAPS" not in src:
        sys.exit("ERREUR : MIN_STINT_LAPS absent. Appliquer d'abord "
                 "apply_patch_min_stint.py.")

    out, err = patch_init(src)
    if err:
        sys.exit(f"  ECHEC   {err}\nRien n'a ete ecrit.")
    print("  OK      parametre min_stint_laps ajoute a __init__")

    if out.count(OLD_STEP) != 1:
        sys.exit(f"  ECHEC   test dans step() trouve {out.count(OLD_STEP)} fois "
                 "(attendu 1).\nRien n'a ete ecrit.")
    out = out.replace(OLD_STEP, NEW_STEP)
    print("  OK      test de step() bascule sur self.min_stint_laps")

    try:
        ast.parse(out)
    except SyntaxError as exc:
        sys.exit(f"  ECHEC   resultat invalide ({exc}).\nRien n'a ete ecrit.")

    # --- common.py et train_v4.py ------------------------------------------
    extra = []
    for path, blocks, label in (
        (COMMON, [(COMMON_OLD, COMMON_NEW)], "common.py"),
        (TRAIN, [(TRAIN_OLD_ARG, TRAIN_NEW_ARG), (TRAIN_OLD_ENV, TRAIN_NEW_ENV)],
         "train_v4.py"),
    ):
        if not path.exists():
            sys.exit(f"  ECHEC   {path} introuvable.\nRien n'a ete ecrit.")
        txt = path.read_text(encoding="utf-8")
        if "min_stint_laps" in txt:
            print(f"  deja OK      {label}")
            continue
        for old, new in blocks:
            if txt.count(old) != 1:
                sys.exit(f"  ECHEC   {label} : bloc trouve {txt.count(old)} fois "
                         "(attendu 1).\nRien n'a ete ecrit.")
            txt = txt.replace(old, new)
        try:
            ast.parse(txt)
        except SyntaxError as exc:
            sys.exit(f"  ECHEC   {label} : resultat invalide ({exc}).")
        print(f"  OK           {label}")
        extra.append((path, txt))

    if args.dry_run:
        print("\n--dry-run : tous les blocs s'appliquent, rien n'est ecrit.")
        return
    for path, txt in extra:
        bak = path.with_suffix(".py.bak_preablation")
        if not bak.exists():
            shutil.copy(path, bak)
        path.write_text(txt, encoding="utf-8")
        print(f"   -> {path.name}")
    if not BACKUP.exists():
        shutil.copy(ENV, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    ENV.write_text(out, encoding="utf-8")
    print(f"Patch applique : {ENV}")
    print("\nControle de non-regression — les chiffres doivent etre INCHANGES :")
    print("  python scripts/03_check_breakeven.py")
    print("  python scripts/06_sanity_replay.py")


if __name__ == "__main__":
    main()
