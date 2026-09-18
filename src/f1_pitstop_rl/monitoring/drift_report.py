"""Rapport de derive des donnees d'entree de l'agent (C5.3.3).

Question posee
--------------
Les situations de course que l'agent rencontre en inference ressemblent-elles
a celles sur lesquelles il a ete entraine ? Si la distribution d'une variable
d'observation derive, les decisions de l'agent sur cette variable reposent sur
une extrapolation, et sa fiabilite baisse. C'est la perte de 10 % mesuree hors
pool par eval_generalization.py : ce rapport en cherche l'origine cote donnees.

Reference et courant
--------------------
- reference : les GP du pool d'entrainement (role "train_*" dans GP_POOL)
- courant   : les GP absents du pool, hors courses tronquees (< 25 tours),
              soit exactement le jeu de generalisation de eval_generalization.py

Variables surveillees
---------------------
Uniquement celles qui alimentent l'observation (F1PitStopEnv._build_observation).
Surveiller lap_time_s, par exemple, n'aurait aucun sens : il varie d'un circuit
a l'autre par construction. n_compounds_used n'existe pas dans le dataset ; il
est reconstruit tour par tour (nombre de composes distincts deja utilises).

Limite assumee
--------------
Les donnees sont celles de la course reelle du pilote, pas les trajectoires de
l'agent : l'age pneu ou la position observes par l'agent dependent de ses
propres decisions. Le rapport mesure donc la derive des conditions de course
(circuits, meteo, rivaux, format), pas celle des etats visites par la politique.

Methode : taille d'effet plutot que p-value
-------------------------------------------
Par defaut, Evidently choisit des tests statistiques (K-S, chi2, Z) et declare
une derive si p < 0,05. Avec ~500 tours de reference et ~1 150 courants, ces
tests detectent des ecarts minimes : ils repondent a "les distributions
sont-elles strictement identiques ?", ce qui n'est jamais le cas entre deux
saisons. La methode par defaut ici mesure l'AMPLEUR de l'ecart :
  - numeriques   : distance de Wasserstein normalisee (seuil 0,1)
  - categorielles: distance de Jensen-Shannon (seuil 0,1)
--method pvalue retablit les tests statistiques, pour comparaison.

Seuils calibres sur les donnees (analyse par GP)
------------------------------------------------
Comparer UNE course au pool melange signale presque tout : une course, c'est
une temperature, une longueur, un melange de pneus. Les courses d'entrainement
elles-memes deriveraient les unes par rapport aux autres. Le seuil par
variable est donc calibre : chaque GP d'entrainement est compare aux 7 autres
(leave-one-out), et le seuil retenu est la plus grande distance observee --
la variabilite naturelle entre deux courses que l'agent connait (plancher 0,1).
Un GP d'inference n'est signale sur une variable que s'il s'ecarte davantage
que ne s'ecartent entre elles les courses d'entrainement.

Sorties (dossier --out, reports/drift par defaut)
-------------------------------------------------
- drift_report.html  : rapport Evidently interactif, reference vs courant
- drift_summary.json : resultat par variable et part de derive par GP

Usage, depuis la racine du projet :
    python src/f1_pitstop_rl/monitoring/drift_report.py
    python src/f1_pitstop_rl/monitoring/drift_report.py --fail-share 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Colonnes constantes (ex. pluie nulle partout) : numpy avertit sur 0/0.
# Sans consequence sur le resultat, on ne pollue pas la sortie.
warnings.filterwarnings("ignore", message="invalid value encountered")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "config"))
from gp_pool_config import GP_POOL  # noqa: E402

FEATURES_PATH = ROOT / "data" / "processed" / "features_dataset.parquet"
MIN_LAPS = 25  # meme seuil que eval_generalization.py (Monaco 2025 : 8 tours)

# Variables de l'observation, dans l'ordre de _build_observation
NUMERICAL = [
    "tyre_age_laps",        # own_tyre_age
    "tours_restants",       # race_total_laps - current_lap
    "position",             # own_position
    "gap_avant",
    "gap_arriere",
    "track_temp_c",
    "delta_pluie_3tours",
    "n_compounds_used",     # reconstruit
]
CATEGORICAL = [
    "is_raining",
    "rival_avant_vient_de_pitter",    # rival_avant_pit  (changement de relais
    "rival_arriere_vient_de_pitter",  # rival_arriere_pit  entre tour-1 et tour)
    "tyre_compound",                # one-hot dans l'observation
]
COLUMNS = NUMERICAL + CATEGORICAL
DEFINITION = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)


# --- Donnees -----------------------------------------------------------------

def _cumulative_unique(s: pd.Series) -> pd.Series:
    seen: set = set()
    out = []
    for v in s:
        if pd.notna(v):
            seen.add(v)
        out.append(float(len(seen)))
    return pd.Series(out, index=s.index)


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(FEATURES_PATH)
    df = df.sort_values(["season", "event", "driver", "lap_number"])
    df["n_compounds_used"] = (
        df.groupby(["season", "event", "driver"])["tyre_compound"].transform(_cumulative_unique)
    )
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)

    train = {(g["season"], g["event"]) for g in GP_POOL if g["role"].startswith("train")}
    pool = {(g["season"], g["event"]) for g in GP_POOL}
    key = pd.Series(list(zip(df["season"], df["event"])), index=df.index)
    n_laps = df.groupby(["season", "event"])["lap_number"].transform("count")

    reference = df[key.isin(train)]
    current = df[~key.isin(pool) & (n_laps >= MIN_LAPS)]
    return reference, current


# --- Evidently ---------------------------------------------------------------

METHODS = {
    "distance": dict(num_method="wasserstein", cat_method="jensenshannon",
                     num_threshold=0.1, cat_threshold=0.1),
    "pvalue": {},  # choix automatique d'Evidently (K-S, chi2, Z ; p < 0,05)
}


def run_report(current: pd.DataFrame, reference: pd.DataFrame, method: str = "distance",
               thresholds: dict | None = None):
    kwargs = dict(METHODS[method])
    if thresholds:
        kwargs["per_column_threshold"] = thresholds
    return Report([DataDriftPreset(**kwargs)]).run(
        Dataset.from_pandas(current[COLUMNS], data_definition=DEFINITION),
        Dataset.from_pandas(reference[COLUMNS], data_definition=DEFINITION),
    )


def _clean(v):
    """Convertit en float natif (JSON) ; NaN / inf -> None."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def parse(snapshot) -> dict:
    """Extrait part de derive et resultat par variable du resultat Evidently."""
    out = {"drifted_count": None, "drifted_share": None, "columns": {}}
    for m in snapshot.dict()["metrics"]:
        cfg = m.get("config", {})
        kind = cfg.get("type", "")
        if kind.endswith("DriftedColumnsCount"):
            out["drifted_count"] = int(m["value"]["count"])
            out["drifted_share"] = float(m["value"]["share"])
        elif kind.endswith("ValueDrift"):
            method, threshold, value = cfg.get("method", ""), cfg.get("threshold"), _clean(m["value"])
            if value is None:
                drift = None
            elif "p_value" in method:   # test statistique : derive si p < seuil
                drift = bool(value < threshold)
            else:                       # distance : derive si distance >= seuil
                drift = bool(value >= threshold)
            out["columns"][cfg["column"]] = {
                "method": method, "threshold": threshold, "value": value, "drift": drift,
            }
    return out


MIN_THRESHOLD = 0.1  # plancher : variabilite naturelle nulle -> seuil par defaut


def calibrate(reference: pd.DataFrame, method: str) -> tuple[dict, dict]:
    """Seuil par variable = distance max entre un GP d'entrainement et les autres."""
    loo: dict[str, list] = {c: [] for c in COLUMNS}
    for _, g in reference.groupby(["season", "event"]):
        rest = reference.drop(g.index)
        cols = parse(run_report(g, rest, method))["columns"]
        for c in COLUMNS:
            v = cols.get(c, {}).get("value")
            if v is not None:
                loo[c].append(v)
    thresholds = {c: max([MIN_THRESHOLD] + loo[c]) for c in COLUMNS}
    return thresholds, loo


# --- Main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="reports/drift", help="dossier de sortie")
    ap.add_argument("--method", choices=sorted(METHODS), default="distance",
                    help="distance (ampleur, defaut) ou pvalue (tests statistiques)")
    ap.add_argument("--fail-share", type=float, default=None,
                    help="code de sortie 1 si la part de variables en derive depasse ce seuil")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reference, current = load_split()
    n_ref = reference.groupby(["season", "event"]).ngroups
    n_cur = current.groupby(["season", "event"]).ngroups

    snapshot = run_report(current, reference, args.method)
    snapshot.save_html(str(out / "drift_report.html"))
    overall = parse(snapshot)

    # Seuils calibres (uniquement pertinent en mode distance)
    calibrated = args.method == "distance"
    if calibrated:
        thresholds, loo = calibrate(reference, args.method)
    else:
        thresholds, loo = None, {}

    per_gp = []
    for (season, event), g in current.groupby(["season", "event"]):
        try:
            r = parse(run_report(g, reference, args.method, thresholds))
            drifted = sorted(c for c, v in r["columns"].items() if v["drift"])
            per_gp.append({"season": int(season), "event": event, "laps": len(g),
                           "drifted_count": len(drifted), "drifted_columns": drifted,
                           "distances": {c: v["value"] for c, v in r["columns"].items()}})
        except Exception as e:  # un GP ne doit pas faire tomber le rapport
            per_gp.append({"season": int(season), "event": event, "laps": len(g),
                           "error": f"{type(e).__name__}: {e}"})
    per_gp.sort(key=lambda r: -(r.get("drifted_count") or 0))

    # Frequence de chaque variable parmi les GP signales
    freq = {c: sum(c in r.get("drifted_columns", []) for r in per_gp) for c in COLUMNS}

    summary = {
        "method": args.method,
        "reference": {"gp": n_ref, "laps": len(reference), "definition": "GP_POOL role train_*"},
        "current": {"gp": n_cur, "laps": len(current),
                    "definition": f"GP hors GP_POOL, >= {MIN_LAPS} tours"},
        "overall": overall,
        "calibration": {"thresholds": thresholds, "leave_one_out": loo,
                        "floor": MIN_THRESHOLD} if calibrated else None,
        "per_gp_frequency": freq,
        "per_gp": per_gp,
    }
    (out / "drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Console ---
    w = 86
    n_gp = len(per_gp)
    print("=" * w)
    print(f"DERIVE DES ENTREES [{args.method}] — reference {n_ref} GP ({len(reference)} tours) "
          f"vs courant {n_cur} GP ({len(current)} tours)")
    print("=" * w)
    head = "seuil calib." if calibrated else "seuil"
    print(f"{'variable':<30}{'methode':<15}{'global':>9}{head:>14}  {'GP signales':>12}")
    print("-" * w)
    for col in COLUMNS:
        r = overall["columns"].get(col)
        if r is None:
            print(f"{col:<30}(absente du resultat)")
            continue
        val = "n/a" if r["value"] is None else f"{r['value']:.3g}"
        thr = thresholds[col] if calibrated else r["threshold"]
        print(f"{col:<30}{r['method'][:14]:<15}{val:>9}{thr:>14.3g}  "
              f"{freq[col]:>6}/{n_gp}")
    print("-" * w)
    if calibrated:
        print("seuil calib. : plus grande distance entre un GP d'entrainement et les 7 autres")
        print("GP signales  : GP d'inference qui s'ecartent PLUS que les GP d'entrainement entre eux")
    print()
    print(f"{'GP (courant)':<34}{'tours':>6}{'signal':>8}  variables au-dela du seuil")
    print("-" * w)
    for r in per_gp:
        name = f"{r['season']} {r['event']}"
        if "error" in r:
            print(f"{name:<34}{r['laps']:>6}{'err':>8}  {r['error'][:40]}")
        else:
            print(f"{name:<34}{r['laps']:>6}{r['drifted_count']:>5}/{len(COLUMNS)}  "
                  f"{', '.join(r['drifted_columns']) or '-'}")
    print("=" * w)
    print(f"Rapport : {(out / 'drift_report.html').resolve()}")
    print(f"Resume  : {(out / 'drift_summary.json').resolve()}")

    ok = [r for r in per_gp if "drifted_count" in r]
    mean_share = sum(r["drifted_count"] for r in ok) / (len(ok) * len(COLUMNS)) if ok else 0.0
    print(f"Part moyenne de variables signalees par GP : {mean_share:.0%}")
    if args.fail_share is not None and mean_share > args.fail_share:
        print(f"ECHEC : {mean_share:.0%} > seuil {args.fail_share:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
