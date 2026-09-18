#!/usr/bin/env python3
"""
Applique automatiquement le patch v3 a f1_pitstop_env.py
=========================================================

Quatre modifications, decrites dans PATCH_ENV_V3.md :
  1. import de build_lap_reference
  2. construction de la reference par tour dans _load_gp_data
  3. utilisation de la reference dans step() (remplace les multiplicateurs SC/VSC)
  4. meme reference passee a compute_step_reward

Le script est IDEMPOTENT : relance sans effet si le patch est deja applique.
Il sauvegarde l'original en f1_pitstop_env.py.bak_prev3 et n'ecrit le fichier
que si les QUATRE modifications ont reussi -- en cas d'echec partiel, rien
n'est modifie et le script indique quel bloc n'a pas ete trouve.

Usage, depuis la racine du projet :
    python scripts/apply_patch_v3.py
    python scripts/apply_patch_v3.py --dry-run     # montre sans ecrire
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "src" / "f1_pitstop_rl" / "env" / "f1_pitstop_env.py"
BACKUP = ENV.with_suffix(".py.bak_prev3")

# (nom, [variantes de texte a chercher], remplacement)
PATCHES = [
    (
        "1. import de build_lap_reference",
        ["from pace_model import GPPaceContext, predict_lap_time, sample_pit_stop_cost"],
        "from pace_model import (GPPaceContext, predict_lap_time, sample_pit_stop_cost,\n"
        "                        build_lap_reference)",
    ),
    (
        "2. reference par tour dans _load_gp_data",
        [
            'usable = gasly[gasly["usable_for_reward"]]\n'
            "        self.pace_ctx = GPPaceContext(\n"
            '            gp_baseline_s=float(usable["lap_time_s"].median()),\n'
            '            gp_std_s=float(usable["lap_time_s"].std()),\n'
            "            race_total_laps=self.race_total_laps,\n"
            "        )",
            'usable = gasly[gasly[flag]]\n'
            "        self.pace_ctx = GPPaceContext(\n"
            '            gp_baseline_s=float(usable["lap_time_s"].median()),\n'
            '            gp_std_s=float(usable["lap_time_s"].std()),\n'
            "            race_total_laps=self.race_total_laps,\n"
            "        )",
        ],
        'flag = ("usable_for_reward_v2" if "usable_for_reward_v2" in gasly.columns\n'
        '                else "usable_for_reward")\n'
        "        usable = gasly[gasly[flag]]\n"
        "\n"
        "        # Reference de rythme tour par tour, derivee du peloton reel.\n"
        "        # Absorbe pluie, Safety Car, VSC, sechage de piste et evolution\n"
        "        # de grip, sans aucune hypothese -- la v2 n'avait aucun terme\n"
        "        # pour l'etat de la piste une fois usable_for_reward nettoye.\n"
        "        lap_reference = build_lap_reference(self._ghosts_df, season, event)\n"
        "\n"
        "        self.pace_ctx = GPPaceContext(\n"
        '            gp_baseline_s=float(usable["lap_time_s"].median()),\n'
        '            gp_std_s=float(usable["lap_time_s"].std()),\n'
        "            race_total_laps=self.race_total_laps,\n"
        "            lap_reference_s=lap_reference,\n"
        "        )",
    ),
    (
        "3. calcul du temps au tour dans step()",
        [
            "if under_caution and not is_pit:\n"
            '            multiplier = SC_PACE_MULTIPLIER if weather["is_safety_car"] else VSC_PACE_MULTIPLIER\n'
            "            lap_time = self.pace_ctx.gp_baseline_s * multiplier\n"
            "        elif under_caution and is_pit:\n"
            "            lap_time = self.pace_ctx.gp_baseline_s  # pas de malus SC supplémentaire, le pit_cost suffit\n"
            "        else:\n"
            "            lap_time = predict_lap_time(\n"
            "                context=self.pace_ctx,\n"
            "                tyre_compound=self.own_tyre_compound,\n"
            "                tyre_age_laps=self.own_tyre_age,\n"
            '                is_raining=weather["is_raining"],\n'
            '                rain_intensity_recent=max(weather["delta_pluie_3tours"], 0.0) + float(weather["is_raining"]) * 0.3,\n'
            "                rng=self.rng,\n"
            "            )",
        ],
        "# Plus de multiplicateur SC/VSC : la reference du tour porte deja\n"
        "        # le ralentissement reel du peloton, mesure au lieu d'etre suppose.\n"
        "        if under_caution and is_pit:\n"
        "            # S'arreter sous neutralisation coute relativement moins cher :\n"
        "            # on ne cumule pas le ralentissement du peloton avec le cout de\n"
        "            # pit complet (double comptage, test de sanite Phase 2.2).\n"
        "            lap_time = self.pace_ctx.reference_for(self.current_lap)\n"
        "        else:\n"
        "            lap_time = predict_lap_time(\n"
        "                context=self.pace_ctx,\n"
        "                tyre_compound=self.own_tyre_compound,\n"
        "                tyre_age_laps=self.own_tyre_age,\n"
        '                is_raining=weather["is_raining"],\n'
        '                rain_intensity_recent=max(weather["delta_pluie_3tours"], 0.0) + float(weather["is_raining"]) * 0.3,\n'
        "                rng=self.rng,\n"
        "                lap=self.current_lap,\n"
        "            )",
    ),
    (
        "4. meme reference dans compute_step_reward",
        [
            "lap_time_s=lap_time,\n"
            "            gp_baseline_s=self.pace_ctx.gp_baseline_s,",
        ],
        "lap_time_s=lap_time,\n"
        "            # Reference du TOUR, pas constante de course : sinon chaque\n"
        "            # tour de Safety Car deviendrait une penalite massive sans\n"
        "            # lien avec les decisions de l'agent.\n"
        "            gp_baseline_s=self.pace_ctx.reference_for(self.current_lap),",
    ),
]

MARKERS = ["build_lap_reference", "lap_reference_s=lap_reference",
           "lap=self.current_lap", "reference_for(self.current_lap)"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not ENV.exists():
        sys.exit(f"ERREUR : {ENV} introuvable.\n"
                 "Lancer ce script depuis la racine du projet F1_PITSTOP_RL.")

    src = ENV.read_text(encoding="utf-8")

    if all(m in src for m in MARKERS):
        print("Patch v3 deja applique -- rien a faire.")
        return

    out, failed, applied = src, [], []
    for name, variants, replacement in PATCHES:
        hit = next((v for v in variants if v in out), None)
        if hit is None:
            # deja applique isolement ?
            if any(m in out for m in MARKERS if m.split("(")[0] in replacement):
                applied.append(f"{name}  (deja en place)")
                continue
            failed.append(name)
            continue
        if out.count(hit) != 1:
            failed.append(f"{name}  ({out.count(hit)} occurrences, attendu 1)")
            continue
        out = out.replace(hit, replacement)
        applied.append(name)

    for a in applied:
        print(f"  OK      {a}")
    for f in failed:
        print(f"  ECHEC   {f}")

    if failed:
        print("\nAucune modification ecrite : le fichier ne correspond pas au")
        print("texte attendu (deja edite a la main ?). Applique PATCH_ENV_V3.md")
        print("manuellement pour les blocs en echec, ou envoie-moi le fichier.")
        sys.exit(1)

    import ast
    try:
        ast.parse(out)
    except SyntaxError as exc:
        sys.exit(f"\nERREUR : le resultat n'est pas du Python valide ({exc}).\n"
                 "Rien n'a ete ecrit.")

    if args.dry_run:
        print("\n--dry-run : les 4 blocs s'appliquent proprement, rien n'est ecrit.")
        return

    if not BACKUP.exists():
        shutil.copy(ENV, BACKUP)
        print(f"\nSauvegarde : {BACKUP.name}")
    ENV.write_text(out, encoding="utf-8")
    print(f"Patch applique : {ENV}")
    print("\nVerification :")
    print("  python scripts/06_sanity_replay.py")
    print("  python scripts/03_check_breakeven.py")


if __name__ == "__main__":
    main()
