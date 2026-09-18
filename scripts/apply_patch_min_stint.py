#!/usr/bin/env python3
"""
Patch — longueur minimale de relais (MIN_STINT_LAPS)
=====================================================

Ajoute a F1PitStopEnv une contrainte dure : une action de pit avant que le
pneu ait MIN_STINT_LAPS tours est convertie en STAY.

Justification metier. Aucune equipe de F1 ne s'arrete a nouveau 1 ou 2 tours
apres un arret : le temps de passage au stand (~23 s) ne peut pas etre
rembourse sur un relais aussi court, quelle que soit la degradation. C'est
une contrainte du domaine, au meme titre que la regle des 2 composes --
et elle est implementee de la meme facon que forced_compliance_pit, donc
algorithme-agnostique : la comparaison DQN / PPO / A2C reste equitable.

Justification d'apprentissage. Cinq des six actions sont des arrets. Une
politique uniforme s'arrete ~83% des tours, soit ~50 arrets par course
(~-1300 de recompense). Cette region de l'espace des politiques est si
catastrophique que le gradient ecrase P(pit) a zero dans les premieres
secondes et n'en ressort jamais -- mesure : 95% du progres atteint a
l'episode 8 sur 8291 (scripts/inspect_training_log.py). La contrainte
plafonne le nombre d'arrets a total_laps / MIN_STINT_LAPS (~11 au lieu de
57), ce qui ramene la catastrophe initiale autour de -250.

Le script est IDEMPOTENT et n'ecrit rien si un bloc echoue.

Usage, depuis la racine du projet :
    python scripts/apply_patch_min_stint.py --dry-run
    python scripts/apply_patch_min_stint.py
    python scripts/apply_patch_min_stint.py --min-stint 6
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "src" / "f1_pitstop_rl" / "env" / "f1_pitstop_env.py"
BACKUP = ENV.with_suffix(".py.bak_preminstint")

ANCHOR_CONST = 'VSC_PACE_MULTIPLIER = 1.03'
ANCHOR_STEP = ("        is_last_lap = self.current_lap >= self.race_total_laps\n"
               "        forced_compliance_pit = False\n")
ANCHOR_INFO = '            "forced_compliance_pit": forced_compliance_pit,\n'

MARKER = "MIN_STINT_LAPS"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-stint", type=int, default=5,
                    help="Tours minimum avant de pouvoir se rearreter (defaut 5).")
    args = ap.parse_args()

    if not ENV.exists():
        sys.exit(f"ERREUR : {ENV} introuvable.\n"
                 "Lancer depuis la racine du projet F1_PITSTOP_RL.")
    src = ENV.read_text(encoding="utf-8")

    if MARKER in src:
        print("Patch deja applique -- rien a faire.")
        return

    const_block = (
        f"MIN_STINT_LAPS = {args.min_stint}   # cf. scripts/apply_patch_min_stint.py\n"
        "# Longueur minimale d'un relais, en tours. Une action de pit avant cet\n"
        "# age de pneu est convertie en STAY. Contrainte dure du domaine (aucune\n"
        "# equipe ne se rearrete 1-2 tours apres un arret : 23 s ne se remboursent\n"
        "# pas sur un relais aussi court), implementee comme forced_compliance_pit\n"
        "# et donc algorithme-agnostique.\n"
        + ANCHOR_CONST
    )

    step_block = (
        "        is_last_lap = self.current_lap >= self.race_total_laps\n"
        "\n"
        "        # Longueur minimale de relais (cf. MIN_STINT_LAPS en tete de\n"
        "        # module). Exception au dernier tour, ou le garde-fou de\n"
        "        # conformite Article 30.7 doit pouvoir imposer un arret.\n"
        "        blocked_by_min_stint = False\n"
        "        if action != 0 and not is_last_lap and self.own_tyre_age < MIN_STINT_LAPS:\n"
        "            action = 0\n"
        "            blocked_by_min_stint = True\n"
        "\n"
        "        forced_compliance_pit = False\n"
    )

    info_block = (
        '            "forced_compliance_pit": forced_compliance_pit,\n'
        '            "blocked_by_min_stint": blocked_by_min_stint,\n'
    )

    out, failed = src, []
    for label, anchor, block in (
        ("constante MIN_STINT_LAPS", ANCHOR_CONST, const_block),
        ("blocage dans step()", ANCHOR_STEP, step_block),
        ("trace dans info", ANCHOR_INFO, info_block),
    ):
        n = out.count(anchor)
        if n != 1:
            failed.append(f"{label}  ({n} occurrences, attendu 1)")
            continue
        out = out.replace(anchor, block)
        print(f"  OK      {label}")

    for f in failed:
        print(f"  ECHEC   {f}")
    if failed:
        print("\nRien n'a ete ecrit. Le fichier ne correspond pas au texte "
              "attendu — applique le patch a la main ou envoie-moi le fichier.")
        sys.exit(1)

    import ast
    try:
        ast.parse(out)
    except SyntaxError as exc:
        sys.exit(f"\nERREUR : resultat invalide ({exc}). Rien n'a ete ecrit.")

    if args.dry_run:
        print(f"\n--dry-run : les 3 blocs s'appliquent "
              f"(MIN_STINT_LAPS={args.min_stint}), rien n'est ecrit.")
        return

    if not BACKUP.exists():
        shutil.copy(ENV, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    ENV.write_text(out, encoding="utf-8")
    print(f"Patch applique (MIN_STINT_LAPS={args.min_stint}) : {ENV}")
    print("\nVerification — l'oracle doit rester favorable a l'arret :")
    print("  python scripts/03_check_breakeven.py")
    print("  python scripts/06_sanity_replay.py")


if __name__ == "__main__":
    main()
