#!/usr/bin/env python3
"""
V2 — Phase 1 : correction du biais de rythme par differentiel d'age
====================================================================

Le probleme
------------
La reference de rythme de l'environnement est la mediane des temps du peloton
au tour courant (`build_lap_reference`). Cette mediane contient DEJA la
degradation moyenne des 20 voitures a cet instant. Le modele y ajoute ensuite
la degradation propre de l'agent : la degradation est comptee deux fois.

Mesure (06_sanity_replay.py, V1) : en rejouant la strategie REELLE de Gasly,
erreur temporelle mediane de +0.95 %, et un classement a P15 a Silverstone
(P6 en realite), P18 aux Pays-Bas (P4), P20 en Belgique (P10). L'ecart est
donc present quelle que soit la strategie jouee.

Pourquoi un differentiel plutot qu'une soustraction absolue
------------------------------------------------------------
Retirer la degradation moyenne ABSOLUE du peloton exigerait d'evaluer la
courbe de degradation a des ages de pneu eleves, souvent au-dela de 20 tours.
Or la Phase 0 (v2_00_measure_degradation_curve.py) a montre que les pentes
au-dela de 20 tours ne sont pas estimables : elles deviennent negatives sur
HARD (-0.0122, -0.0407, -0.0842 s/tour), ce qui est physiquement impossible.
La cause est une censure informative — un pneu n'atteint 30 ou 40 tours que
lorsque tout va bien, donc les relais longs survivants sont selectionnes sur
leur faible degradation.

La correction retenue ne manipule que des ECARTS d'age :

    lap_time = reference(tour)
             + pente(compose) * (age_agent - age_median_peloton)
             + penalite_pneu_froid
             + bruit

Quand l'agent a exactement l'age median du peloton, la correction est nulle et
il tourne au rythme de reference — ce qui est le comportement attendu. Quand
il a un pneu plus vieux, il perd ; plus jeune, il gagne. La degradation n'est
plus comptee deux fois, et l'ecart d'age reste dans la plage ou les pentes
sont fiables.

Ce que ce script fait
----------------------
1. calcule l'age median du pneu du peloton par (GP, tour) depuis
   all_drivers_dataset.parquet, et l'ecrit dans un JSON de calibration
2. patche pace_model.py : GPPaceContext porte la nouvelle table, et
   predict_lap_time applique le differentiel
3. patche f1_pitstop_env.py : la table est chargee au _load_gp_data

Idempotent. N'ecrit rien si un bloc echoue.

Usage, depuis la racine du projet V2 :
    python scripts/v2_01_apply_pace_debias.py --dry-run
    python scripts/v2_01_apply_pace_debias.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "src" / "f1_pitstop_rl" / "env" / "f1_pitstop_env.py"
PACE = ROOT / "src" / "f1_pitstop_rl" / "env" / "pace_model.py"
GHOSTS = ROOT / "data" / "raw" / "all_drivers_dataset.parquet"
AGES_JSON = ROOT / "data" / "processed" / "pack_median_age.json"

MARKER = "pack_median_age"

# --- pace_model.py ----------------------------------------------------------
PACE_OLD_CTX = '''    gp_baseline_s: float
    gp_std_s: float
    race_total_laps: int
    lap_reference_s: dict = field(default_factory=dict)

    def reference_for(self, lap: int) -> float:
        return float(self.lap_reference_s.get(lap, self.gp_baseline_s))'''

PACE_NEW_CTX = '''    gp_baseline_s: float
    gp_std_s: float
    race_total_laps: int
    lap_reference_s: dict = field(default_factory=dict)
    pack_median_age: dict = field(default_factory=dict)

    def reference_for(self, lap: int) -> float:
        return float(self.lap_reference_s.get(lap, self.gp_baseline_s))

    def pack_age_for(self, lap: int) -> float:
        """Age median du pneu du peloton a ce tour.

        La reference de rythme etant la mediane des temps du peloton, elle
        contient deja la degradation correspondant a CET age. Le modele ne
        doit donc facturer a l'agent que son ECART a cet age, sous peine de
        compter la degradation deux fois (cf. v2_01_apply_pace_debias.py).
        Repli a NaN : correction desactivee, comportement V1.
        """
        return float(self.pack_median_age.get(lap, float("nan")))'''

PACE_OLD_PRED = '''    if tyre_age_laps <= 2:
        delta_s += OUT_LAP_PENALTY_S.get(tyre_compound, 0.5)
    else:
        delta_s += DEGRADATION_S.get(tyre_compound, 0.045) * (tyre_age_laps - 2)'''

PACE_NEW_PRED = '''    if tyre_age_laps <= 2:
        delta_s += OUT_LAP_PENALTY_S.get(tyre_compound, 0.5)
    else:
        slope = DEGRADATION_S.get(tyre_compound, 0.045)
        pack_age = context.pack_age_for(lap) if lap is not None else float("nan")
        if pack_age == pack_age:            # non NaN : correction active
            # Differentiel : on ne facture que l'ECART a l'age median du
            # peloton, deja contenu dans la reference de rythme. Borne a la
            # plage ou les pentes sont estimables (cf. Phase 0 : au-dela de
            # 20 tours d'ecart, la censure informative rend la pente non
            # identifiable, elle devient meme negative sur HARD).
            delta_age = float(np.clip(tyre_age_laps - pack_age, -20.0, 20.0))
            delta_s += slope * delta_age
        else:
            delta_s += slope * (tyre_age_laps - 2)'''

# --- f1_pitstop_env.py ------------------------------------------------------
ENV_OLD = '''        self.pace_ctx = GPPaceContext(
            gp_baseline_s=float(usable["lap_time_s"].median()),
            gp_std_s=float(usable["lap_time_s"].std()),
            race_total_laps=self.race_total_laps,
            lap_reference_s=lap_reference,
        )'''

ENV_NEW = '''        # Age median du pneu du peloton, tour par tour. Sert a ne facturer a
        # l'agent que son ECART a cet age : la reference de rythme contient
        # deja la degradation du peloton (cf. v2_01_apply_pace_debias.py).
        pack_median_age = _load_pack_median_age(season, event)

        self.pace_ctx = GPPaceContext(
            gp_baseline_s=float(usable["lap_time_s"].median()),
            gp_std_s=float(usable["lap_time_s"].std()),
            race_total_laps=self.race_total_laps,
            lap_reference_s=lap_reference,
            pack_median_age=pack_median_age,
        )'''

ENV_HELPER = '''

_PACK_AGE_CACHE = None


def _load_pack_median_age(season, event) -> dict:
    """Charge {tour: age median du peloton} pour un GP.

    Produit par scripts/v2_01_apply_pace_debias.py. Absent -> dict vide, la
    correction est alors inactive et le modele retrouve le comportement V1.
    """
    global _PACK_AGE_CACHE
    if _PACK_AGE_CACHE is None:
        p = (Path(__file__).resolve().parents[3] / "data" / "processed"
             / "pack_median_age.json")
        try:
            _PACK_AGE_CACHE = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            import warnings
            warnings.warn(
                f"pack_median_age.json introuvable ({p}) — correction du biais "
                "de rythme INACTIVE, comportement V1.", RuntimeWarning,
                stacklevel=2)
            _PACK_AGE_CACHE = {}
    return {int(k): float(v)
            for k, v in _PACK_AGE_CACHE.get(f"{season}|{event}", {}).items()}
'''


def build_ages():
    df = pd.read_parquet(GHOSTS)
    need = {"season", "event", "lap_number", "tyre_age_laps"}
    if need - set(df.columns):
        sys.exit(f"colonnes manquantes : {need - set(df.columns)}")
    df = df.dropna(subset=["tyre_age_laps"])
    out, stats = {}, []
    for (s, e), g in df.groupby(["season", "event"]):
        med = g.groupby("lap_number")["tyre_age_laps"].median()
        out[f"{s}|{e}"] = {str(int(k)): float(v) for k, v in med.items()}
        stats.append((f"{s} {e}", float(med.median()), float(med.max())))
    return out, stats


def patch(path: Path, blocks, label):
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"  deja patche   {label}")
        return src, True, False
    out = src
    for old, new in blocks:
        if out.count(old) != 1:
            print(f"  ECHEC         {label} : bloc trouve {out.count(old)} fois")
            return src, False, False
        out = out.replace(old, new)
    import ast
    try:
        ast.parse(out)
    except SyntaxError as exc:
        print(f"  ECHEC         {label} : resultat invalide ({exc})")
        return src, False, False
    print(f"  OK            {label}")
    return out, True, True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in (ENV, PACE, GHOSTS):
        if not p.exists():
            sys.exit(f"ERREUR : {p} introuvable. Lancer depuis la racine du projet.")

    print("1. Calcul de l'age median du pneu du peloton, par GP et par tour")
    ages, stats = build_ages()
    print(f"   {len(ages)} GP\n")
    print(f"   {'GP':<34}{'age median':>12}{'age max':>10}")
    for name, med, mx in sorted(stats)[:6]:
        print(f"   {name[:33]:<34}{med:>12.1f}{mx:>10.1f}")
    print(f"   ... ({len(stats)} GP au total)\n")

    print("2. Patch des modules")
    pace_out, pace_ok, pace_changed = patch(
        PACE, [(PACE_OLD_CTX, PACE_NEW_CTX), (PACE_OLD_PRED, PACE_NEW_PRED)],
        "pace_model.py")
    env_src = ENV.read_text(encoding="utf-8")
    env_blocks = [(ENV_OLD, ENV_NEW)]
    if "_load_pack_median_age" not in env_src:
        anchor = "\n\nclass F1PitStopEnv"
        if env_src.count(anchor) == 1:
            env_blocks.append((anchor, ENV_HELPER + "\nclass F1PitStopEnv"))
        else:
            print(f"  ECHEC         f1_pitstop_env.py : ancre de classe "
                  f"trouvee {env_src.count(anchor)} fois")
            sys.exit(1)
    env_out, env_ok, env_changed = patch(ENV, env_blocks, "f1_pitstop_env.py")

    if not (pace_ok and env_ok):
        sys.exit("\nRien n'a ete ecrit.")
    if args.dry_run:
        print("\n--dry-run : les deux modules se patchent, rien n'est ecrit.")
        return

    AGES_JSON.parent.mkdir(parents=True, exist_ok=True)
    AGES_JSON.write_text(json.dumps(ages), encoding="utf-8")
    print(f"\n   -> {AGES_JSON}")

    for path, out, changed in ((PACE, pace_out, pace_changed),
                               (ENV, env_out, env_changed)):
        if not changed:
            continue
        bak = path.with_suffix(".py.bak_predebias")
        if not bak.exists():
            shutil.copy(path, bak)
        path.write_text(out, encoding="utf-8")
        print(f"   -> {path.name} (sauvegarde {bak.name})")

    print("\nVerification, dans cet ordre :")
    print("  python scripts/06_sanity_replay.py --json docs/sanity_v2_apres.json")
    print("  python scripts/03_check_breakeven.py")
    print("\nCriteres : erreur de position mediane <= 2 places, temps <= 0.30 %.")
    print("Si non atteints : ne pas poursuivre l'ajustement, restaurer depuis")
    print("les .bak_predebias et conserver la comparaison interne aux baselines.")


if __name__ == "__main__":
    main()
