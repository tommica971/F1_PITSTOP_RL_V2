"""
F1_PITSTOP_RL/scripts/run_season_inference.py

Fait tourner le modele A2C entraine (INFERENCE uniquement, pas de reentrainement)
sur les GP de la saison 2025 de Gasly, et exporte les JSON pre-calcules pour le
dashboard de soutenance (cf. SCHEMA_JSON.md).

Reutilise integralement les modules existants du repo (aucune nouvelle logique
d'extraction/feature engineering) :
    - extract_tracinginsights.py : laps Gasly (season/event) -> master_dataset
    - extract_plateau.py         : laps tous pilotes (season/event) -> all_drivers_dataset
    - extract_initial_gaps.py    : ecarts reels tour 1 -> initial_gaps
    - features.py                : feature engineering -> features_dataset
    - f1_pitstop_env.py          : F1PitStopEnv(fixed_gp=...) pour l'inference
    - reward.py                  : POINTS_TABLE (meme bareme que l'entrainement)

Convention du projet respectee : device='cpu' partout.

Usage :
    cd F1_PITSTOP_RL/scripts
    python run_season_inference.py                     # les 24 GP de la saison
    python run_season_inference.py --gp "Belgian Grand Prix"   # un seul GP (debug)
    python run_season_inference.py --skip-extraction    # si les parquets sont deja a jour

Sortie :
    dashboard/data/season_2025.json
    dashboard/data/race_<gp_slug>_2025.json  (un par GP)

ATTENTION : le nom d'"event" doit matcher EXACTEMENT le nom de dossier utilise
par le miroir TracingInsights(-Archive) pour la saison 2025 (meme convention
que gp_pool_config.py, ex. "Belgian Grand Prix"). La liste SEASON_2025_CALENDAR
ci-dessous est construite sur les noms officiels FIA -- si un fetch echoue avec
"pas de liste pilotes / GP ignore", verifie l'orthographe exacte contre le repo
TracingInsights-Archive/2025 sur GitHub avant de creuser plus loin.
"""

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / "src" / "f1_pitstop_rl"
sys.path.insert(0, str(SRC_DIR / "config"))
sys.path.insert(0, str(SRC_DIR / "data"))
sys.path.insert(0, str(SRC_DIR / "env"))

from gp_pool_config import GP_POOL, DRIVER_CODE, TEAM_NAME          # noqa: E402
from extract_tracinginsights import (                                # noqa: E402
    extract_driver_laps, extract_track_status, clean_laps as clean_laps_gasly,
)
from extract_plateau import extract_all_drivers_gp, clean_plateau    # noqa: E402
from extract_initial_gaps import get_driver_list, fetch_json, _repo_base  # noqa: E402
from features import (                                               # noqa: E402
    get_all_drivers_laps, build_gap_and_undercut_features,
    add_rolling_rival_pit_flags, add_weather_trend, add_race_progress,
)
from f1_pitstop_env import F1PitStopEnv                               # noqa: E402
from reward import POINTS_TABLE                                       # noqa: E402

DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DASHBOARD_DATA_DIR = SCRIPT_DIR.parent / "dashboard" / "data"

MASTER_PATH = RAW_DIR / "master_dataset.parquet"
PLATEAU_PATH = RAW_DIR / "all_drivers_dataset.parquet"
GAPS_PATH = RAW_DIR / "initial_gaps.parquet"
FEATURES_PATH = PROCESSED_DIR / "features_dataset.parquet"

MODEL_PATH = SCRIPT_DIR.parent / "models" / "a2c" / "a2c_extended_pool_5000k.zip"

INFERENCE_ROLE = "season_2025_inference"  # role neutre pour les GP hors pool d'entrainement
ACTION_NAMES = ["stay", "pit_soft", "pit_medium", "pit_hard", "pit_intermediate", "pit_wet"]
ACTION_TO_COMPOUND_STR = {0: None, 1: "SOFT", 2: "MEDIUM", 3: "HARD", 4: "INTERMEDIATE", 5: "WET"}

# Calendrier officiel FIA 2025 (24 courses). A verifier contre les dossiers
# reels du repo TracingInsights-Archive/2025 si un fetch echoue (cf. docstring).
SEASON_2025_CALENDAR = [
    "Australian Grand Prix", "Chinese Grand Prix", "Japanese Grand Prix",
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix",
    "Canadian Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Belgian Grand Prix", "Hungarian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]
SEASON = 2025


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def gp_role_in_pool(season: int, event: str) -> str | None:
    for gp in GP_POOL:
        if gp["season"] == season and gp["event"] == event:
            return gp["role"]
    return None


def load_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def gp_already_extracted(df: pd.DataFrame, season: int, event: str) -> bool:
    if df.empty:
        return False
    return ((df["season"] == season) & (df["event"] == event)).any()


# --------------------------------------------------------------------------
# Etape 1 : extraction brute + feature engineering pour un GP absent du pool
# --------------------------------------------------------------------------

MIN_USABLE_LAPS = 5  # cf. EXCLUDED_GP dans gp_pool_config.py : un DNF prematuré
                      # (ex. 2024 British GP, 1 seul tour) ne laisse aucune donnee
                      # strategique exploitable -- pas un echec de script, un GP a exclure.


def extract_and_build_features_for_gp(season: int, event: str):
    role = gp_role_in_pool(season, event) or INFERENCE_ROLE
    print(f"  Extraction brute (Gasly)...")
    laps = extract_driver_laps(season, event)
    if laps is None:
        print(f"    -> [SKIP] pas de donnees Gasly pour {event}")
        return False
    if len(laps) < MIN_USABLE_LAPS:
        print(f"    -> [SKIP] seulement {len(laps)} tour(s) enregistre(s) pour Gasly "
              f"(abandon precoce probable) -- GP exclu, aucune donnee strategique exploitable")
        return False

    status = extract_track_status(season, event)
    laps = laps.merge(status, left_on="lap", right_on="lap", how="left")
    for col in ["is_red_flag", "is_safety_car", "is_vsc"]:
        if col not in laps.columns:
            laps[col] = False
        laps[col] = laps[col].fillna(False).astype(bool)
    laps = clean_laps_gasly(laps)
    # Garde-fou defensif : clean_laps() (extract_tracinginsights.py) ne force pas
    # les colonnes numeriques -- decouvert sur Miami 2025, ou des valeurs "None"
    # litterales (donnee source, probablement liees au drapeau rouge de cette
    # course) font echouer l'ecriture Parquet en aval (ArrowInvalid), colonne par
    # colonne (vu sur stint_number puis tyre_age_laps -- generalise a toutes les
    # colonnes numeriques attendues plutot que corriger au cas par cas). Patch
    # scope a ce script uniquement ; le meme gap existe dans extract_tracinginsights.py.
    _NUMERIC_COLS = ["lap_number", "stint_number", "tyre_age_laps", "lap_time_s",
                      "air_temp_c", "humidity_pct", "pressure_hpa", "track_temp_c",
                      "wind_direction_deg", "wind_speed_kmh"]
    for col in _NUMERIC_COLS:
        if col in laps.columns:
            laps[col] = pd.to_numeric(laps[col], errors="coerce")
    laps["role"] = role
    laps["known_issues"] = ""

    master = load_or_empty(MASTER_PATH)
    master = pd.concat([master, laps], ignore_index=True) if not master.empty else laps
    # dedup : une tentative precedente peut avoir ecrit ce GP avant d'echouer
    # plus loin dans le pipeline (vu sur Miami -- triplication des lignes)
    master = master.drop_duplicates(subset=["season", "event", "driver", "lap_number"], keep="last")
    master.to_parquet(MASTER_PATH, index=False)

    print(f"  Extraction plateau (tous pilotes)...")
    plateau_raw = extract_all_drivers_gp(season, event, role)
    if plateau_raw.empty:
        print(f"    -> [SKIP] pas de plateau pour {event}")
        return False
    plateau = clean_plateau(plateau_raw)
    all_drivers_df = load_or_empty(PLATEAU_PATH)
    all_drivers_df = pd.concat([all_drivers_df, plateau], ignore_index=True) if not all_drivers_df.empty else plateau
    all_drivers_df = all_drivers_df.drop_duplicates(subset=["season", "event", "driver", "lap_number"], keep="last")
    all_drivers_df.to_parquet(PLATEAU_PATH, index=False)

    print(f"  Ecarts reels tour 1...")
    gap_rows = []
    for drv in get_driver_list(season, event):
        url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{drv}/laptimes.json"
        raw = fetch_json(url)
        if raw is None or not raw.get("lap"):
            continue
        try:
            idx = raw["lap"].index(1)
            ses_t = raw["sesT"][idx]
            if ses_t is not None:
                gap_rows.append({"season": season, "event": event, "driver": drv, "sesT_lap1": float(ses_t)})
        except (ValueError, IndexError, TypeError):
            continue
    gaps_df = load_or_empty(GAPS_PATH)
    new_gaps = pd.DataFrame(gap_rows)
    gaps_df = pd.concat([gaps_df, new_gaps], ignore_index=True) if not gaps_df.empty else new_gaps
    gaps_df = gaps_df.drop_duplicates(subset=["season", "event", "driver"], keep="last")
    gaps_df.to_parquet(GAPS_PATH, index=False)

    print(f"  Feature engineering...")
    gp_base = master[(master["season"] == season) & (master["event"] == event)].copy()
    all_drivers = get_all_drivers_laps(season, event)
    if all_drivers.empty:
        print(f"    -> [SKIP] gaps non calculables pour {event}")
        return False
    gap_df = build_gap_and_undercut_features(all_drivers)
    if gap_df.empty:
        print(f"    -> [SKIP] aucun tour Gasly retrouve dans le plateau multi-pilotes pour {event} "
              f"(donnees de position manquantes/incompletes) -- GP exclu")
        return False
    gap_df = add_rolling_rival_pit_flags(gap_df)
    merged = gp_base.merge(gap_df, left_on="lap_number", right_on="lap", how="left")
    merged = merged.drop(columns=["lap"], errors="ignore")

    features_df = load_or_empty(FEATURES_PATH)
    features_df = pd.concat([features_df, merged], ignore_index=True) if not features_df.empty else merged
    features_df = add_weather_trend(features_df)
    features_df = add_race_progress(features_df)
    # Comble les trous ("None" litteral cote source -> NaN apres coercion, vu sur
    # Miami 2025 -- tyre_age_laps/stint_number) par report de la derniere valeur
    # connue, PAR GP (season, event) pour ne jamais faire deborder d'un GP a
    # l'autre. F1PitStopEnv relit ce fichier directement (son propre pd.read_parquet
    # interne) -- le nettoyage doit avoir lieu ICI, avant l'ecriture, pas seulement
    # dans une copie locale en aval.
    # tyre_compound porte le meme defaut sous forme de chaine "None" litterale
    # (pas un NaN pandas) -- il faut la convertir avant de pouvoir la combler.
    if "tyre_compound" in features_df.columns:
        features_df["tyre_compound"] = features_df["tyre_compound"].replace("None", np.nan)
    # stint_number : ffill/bfill sur (season,event) est raisonnable ici, les
    # changements de relais sont rares (2-4 par course), un trou meme large a
    # peu de chances de chevaucher plusieurs vrais changements de relais.
    if "stint_number" in features_df.columns:
        features_df["stint_number"] = features_df.groupby(["season", "event"])["stint_number"].transform(
            lambda s: s.ffill().bfill()
        )
    if "tyre_compound" in features_df.columns:
        features_df["tyre_compound"] = features_df.groupby(["season", "event"])["tyre_compound"].transform(
            lambda s: s.ffill().bfill()
        )
    # tyre_age_laps : PAS de ffill/bfill ici -- decouvert sur Miami 2025, ou un
    # trou couvrant tout un relais fait remonter (bfill) la valeur du relais
    # SUIVANT (qui redemarre a 1 apres un arret), figeant l'age du pneu a 1 sur
    # tout le premier relais. L'agent croit alors rouler sur un pneu neuf en
    # permanence -> le modele de rythme predit des tours irrealistes -> gain de
    # position artificiel (constate : P18 -> P1 en un seul tour sur Miami).
    # Fix correct : recalculer l'age comme un compteur par relais (stint_number
    # doit d'abord etre fiable, cf. ci-dessus), independant de la fenetre du trou.
    if "tyre_age_laps" in features_df.columns and "stint_number" in features_df.columns:
        features_df = features_df.sort_values(["season", "event", "lap_number"])
        features_df["tyre_age_laps"] = features_df.groupby(
            ["season", "event", "stint_number"]
        ).cumcount() + 1
    features_df = features_df.drop_duplicates(subset=["season", "event", "lap_number"], keep="last")
    features_df.to_parquet(FEATURES_PATH, index=False)

    print(f"    -> OK, {len(merged)} tours ajoutes")
    return True


# --------------------------------------------------------------------------
# Etape 2 : inference tour par tour sur un GP deja present dans les parquets
# --------------------------------------------------------------------------

def run_inference_on_gp(model, season: int, event: str, seed: int = 42):
    features_df = pd.read_parquet(FEATURES_PATH)
    real = features_df[
        (features_df["season"] == season) & (features_df["event"] == event)
    ].sort_values("lap_number").reset_index(drop=True)
    if real.empty:
        print(f"    -> [SKIP] pas de features pour {event}, extraction necessaire d'abord")
        return None
    # Comble les trous ponctuels ("None" litteral cote source, coerce en NaN a
    # l'extraction -- vu sur Miami 2025) par report de la derniere valeur connue :
    # plus robuste pour l'affichage dashboard qu'un plantage sur un seul tour glitche.
    _FFILL_COLS = ["position", "tyre_age_laps", "tyre_compound"]
    for col in _FFILL_COLS:
        if col in real.columns:
            real[col] = real[col].ffill().bfill()

    gp_dict = {"season": season, "event": event, "role": gp_role_in_pool(season, event) or INFERENCE_ROLE,
               "known_issues": []}
    env = F1PitStopEnv(fixed_gp=gp_dict, seed=seed)
    obs, _ = env.reset(seed=seed)

    lap1 = real.iloc[0]
    # cum_time_real_s demarre a 0, aligne sur l'episode (tour 2), PAS sur le
    # temps de session brut du tour 1 (~3500s, artefact structurel documente
    # en tete de f1_pitstop_env.py) -- sinon l'ecart avec cum_time_agent_s
    # (qui demarre bien a 0 au reset) n'aurait aucun sens.
    cum_time_real = 0.0
    laps_out = [{
        "lap": int(lap1["lap_number"]),
        "position_real": int(lap1["position"]), "position_agent": int(lap1["position"]),
        "tire_age_real": int(lap1["tyre_age_laps"]), "tire_age_agent": int(lap1["tyre_age_laps"]),
        "compound_real": str(lap1["tyre_compound"]), "compound_agent": str(lap1["tyre_compound"]),
        "is_raining": bool(lap1["is_raining"]),
        "track_status": "sc" if lap1.get("is_safety_car") else ("vsc" if lap1.get("is_vsc") else "green"),
        "agent_action": "stay", "agent_action_confidence": None, "agent_action_probs": None,
        "pit_real_this_lap": False, "pit_agent_this_lap": False,
        "air_temp_c": float(lap1["air_temp_c"]) if pd.notna(lap1.get("air_temp_c")) else None,
        "track_temp_c": float(lap1["track_temp_c"]) if pd.notna(lap1.get("track_temp_c")) else None,
        "humidity_pct": float(lap1["humidity_pct"]) if pd.notna(lap1.get("humidity_pct")) else None,
        "cum_time_real_s": cum_time_real, "cum_time_agent_s": 0.0,
        "reward_breakdown": None,
    }]

    done = False
    compound_before = str(lap1["tyre_compound"])
    tire_age_before = int(lap1["tyre_age_laps"])

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()

        lap_num = env.current_lap
        real_row = real[real["lap_number"] == lap_num]

        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        real_pos = int(real_row["position"].iloc[0]) if not real_row.empty else None
        real_compound = str(real_row["tyre_compound"].iloc[0]) if not real_row.empty else None
        real_tire_age = int(real_row["tyre_age_laps"].iloc[0]) if not real_row.empty else None
        real_raining = bool(real_row["is_raining"].iloc[0]) if not real_row.empty else False
        real_lap_time = float(real_row["lap_time_s"].iloc[0]) if (not real_row.empty and pd.notna(real_row["lap_time_s"].iloc[0])) else None
        if real_lap_time is not None:
            cum_time_real += real_lap_time

        breakdown = info.get("reward_breakdown")
        breakdown_dict = dataclasses.asdict(breakdown) if breakdown is not None else None

        laps_out.append({
            "lap": lap_num,
            "position_real": real_pos, "position_agent": int(info["position"]),
            "tire_age_real": real_tire_age, "tire_age_agent": tire_age_before,
            "compound_real": real_compound, "compound_agent": compound_before,
            "is_raining": real_raining,
            "track_status": "sc" if (not real_row.empty and real_row["is_safety_car"].iloc[0])
                             else ("vsc" if (not real_row.empty and real_row["is_vsc"].iloc[0]) else "green"),
            "agent_action": ACTION_NAMES[int(action)],
            "agent_action_confidence": float(np.max(probs)),
            "agent_action_probs": {n: float(p) for n, p in zip(ACTION_NAMES, probs)},
            "pit_real_this_lap": False,  # cf. pit_events.real pour les arrets reels (pit_time_loss)
            "pit_agent_this_lap": bool(info["is_pit"]),
            "forced_compliance_pit": bool(info.get("forced_compliance_pit", False)),
            "compound_after_pit": None,  # rempli juste apres si pit_agent_this_lap
            "air_temp_c": float(real_row["air_temp_c"].iloc[0]) if (not real_row.empty and pd.notna(real_row["air_temp_c"].iloc[0])) else None,
            "track_temp_c": float(real_row["track_temp_c"].iloc[0]) if (not real_row.empty and pd.notna(real_row["track_temp_c"].iloc[0])) else None,
            "humidity_pct": float(real_row["humidity_pct"].iloc[0]) if (not real_row.empty and pd.notna(real_row["humidity_pct"].iloc[0])) else None,
            "cum_time_real_s": cum_time_real, "cum_time_agent_s": float(env.own_cum_time),
            "reward_breakdown": breakdown_dict,
        })

        if info["is_pit"]:
            # IMPORTANT : ne PAS deriver le composé depuis l'`action` predite par
            # model.predict() -- en cas de forced_compliance_pit (dernier tour,
            # course seche, < 2 composés utilisés), l'environnement SUBSTITUE en
            # interne une action differente (cf. f1_pitstop_env.py, bloc
            # forced_compliance_pit) sans jamais renvoyer cette action corrigee a
            # l'appelant. Se fier a `action` ici affichait a tort le MEME composé
            # qu'avant le pit (repere sur le cas Chine 2025 : pit forcé tour 56
            # affiche comme "-> MEDIUM" alors que l'environnement force en realite
            # le premier composé non utilise, ici HARD). `env.own_tyre_compound`
            # est mis a jour par l'environnement lui-meme avec l'action reellement
            # appliquee (forcee ou non) -- c'est la seule source fiable.
            compound_before = str(env.own_tyre_compound)
            tire_age_before = 1
            laps_out[-1]["compound_after_pit"] = compound_before
        else:
            tire_age_before += 1

    final_position_agent = laps_out[-1]["position_agent"]
    final_position_real = int(real.iloc[-1]["position"])
    # Un arret reel = un changement de stint_number (pas de pit_time_loss dans
    # features_dataset, cette colonne n'existe que dans all_drivers_dataset).
    real_sorted = real.sort_values("lap_number")
    stint_changes = real_sorted["stint_number"].ne(real_sorted["stint_number"].shift())
    real_pit_laps = real_sorted.loc[stint_changes & (real_sorted["lap_number"] != real_sorted["lap_number"].min()), "lap_number"].tolist()
    agent_pit_laps = [l["lap"] for l in laps_out if l["pit_agent_this_lap"]]

    race_json = {
        "gp_name": event, "season": season,
        "seen_in_training": gp_role_in_pool(season, event) in ("train_wet", "train_dry"),
        "total_laps": len(laps_out),
        "laps": laps_out,
        "pit_events": {
            "real": [
                {"lap": int(l), "compound_after": str(real_sorted[real_sorted["lap_number"] == l]["tyre_compound"].iloc[0])}
                for l in real_pit_laps
            ],
            "agent": [{"lap": l["lap"], "compound_after": l["compound_after_pit"],
                       "confidence": l["agent_action_confidence"], "forced_compliance": l["forced_compliance_pit"]}
                      for l in laps_out if l["pit_agent_this_lap"]],
        },
        "final": {
            "real": {"position": final_position_real, "points": POINTS_TABLE.get(final_position_real, 0)},
            "agent": {"position": final_position_agent, "points": POINTS_TABLE.get(final_position_agent, 0)},
        },
    }
    return race_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp", type=str, default=None, help="Un seul GP (debug), sinon toute la saison")
    parser.add_argument("--skip-extraction", action="store_true",
                         help="Suppose que master/plateau/gaps/features sont deja a jour pour tous les GP demandes")
    args = parser.parse_args()

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Chargement du modele : {MODEL_PATH}")
    model = A2C.load(str(MODEL_PATH), device="cpu")

    events = [args.gp] if args.gp else SEASON_2025_CALENDAR
    # On verifie sur features_dataset.parquet (l'artefact final que lit
    # run_inference_on_gp), pas sur master_dataset.parquet : un run precedent
    # peut avoir ecrit une ligne dans master puis echoue avant d'atteindre les
    # features (ex. Saudi Arabian Grand Prix, DNF tour 1) -- se fier a master
    # ferait sauter l'extraction a tort et l'inference echouerait ensuite.
    features_existing = load_or_empty(FEATURES_PATH)

    failed_gps = []
    for i, event in enumerate(events, 1):
        print(f"\n--- [{i}/{len(events)}] {event} {SEASON} ---")
        try:
            if not args.skip_extraction and not gp_already_extracted(features_existing, SEASON, event):
                ok = extract_and_build_features_for_gp(SEASON, event)
                if not ok:
                    failed_gps.append(event)
                    continue
                features_existing = load_or_empty(FEATURES_PATH)
            else:
                print("  Deja extrait, on passe directement a l'inference.")

            race_json = run_inference_on_gp(model, SEASON, event)
            if race_json is None:
                failed_gps.append(event)
                continue
        except Exception as e:
            # Un GP en echec (donnees manquantes/malformees cote miroir, etc.)
            # ne doit pas faire perdre la progression sur les 23 autres.
            print(f"    -> [ERREUR] {event} : {type(e).__name__}: {e}")
            failed_gps.append(event)
            continue

        slug = slugify(event)
        out_path = DASHBOARD_DATA_DIR / f"race_{slug}_{SEASON}.json"
        out_path.write_text(json.dumps(race_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {out_path}")

    # --- Agregation season_2025.json a partir des race_*.json ecrits ---
    season_races = []
    for i, event in enumerate(events, 1):
        slug = slugify(event)
        race_path = DASHBOARD_DATA_DIR / f"race_{slug}_{SEASON}.json"
        if not race_path.exists():
            continue
        race_json = json.loads(race_path.read_text(encoding="utf-8"))
        real_pos, agent_pos = race_json["final"]["real"]["position"], race_json["final"]["agent"]["position"]
        real_pts, agent_pts = race_json["final"]["real"]["points"], race_json["final"]["agent"]["points"]
        confs = [l["agent_action_confidence"] for l in race_json["laps"] if l["agent_action_confidence"] is not None]
        season_races.append({
            "round": i, "gp_name": event, "seen_in_training": race_json["seen_in_training"],
            "real": {"finish_position": real_pos, "points": real_pts,
                     "n_pitstops": len(race_json["pit_events"]["real"])},
            "agent": {"finish_position": agent_pos, "points": agent_pts,
                      "n_pitstops": len(race_json["pit_events"]["agent"]),
                      "mean_decision_confidence": float(np.mean(confs)) if confs else None},
            "delta_points": agent_pts - real_pts,
        })

    if not season_races:
        print("\nAucun GP traite avec succes -- season_2025.json non genere.")
        return

    real_total = sum(r["real"]["points"] for r in season_races)
    agent_total = sum(r["agent"]["points"] for r in season_races)
    unseen = [r for r in season_races if not r["seen_in_training"]]

    season_json = {
        "driver": "Pierre Gasly", "team": TEAM_NAME, "season": SEASON,
        "model": MODEL_PATH.stem, "races": season_races,
        "summary": {
            "real_total_points": real_total, "agent_total_points": agent_total,
            "real_avg_position": float(np.mean([r["real"]["finish_position"] for r in season_races])),
            "agent_avg_position": float(np.mean([r["agent"]["finish_position"] for r in season_races])),
            "races_improved": sum(1 for r in season_races if r["delta_points"] > 0),
            "races_worse": sum(1 for r in season_races if r["delta_points"] < 0),
            "races_equal": sum(1 for r in season_races if r["delta_points"] == 0),
            "unseen_races_only": {
                "real_total_points": sum(r["real"]["points"] for r in unseen),
                "agent_total_points": sum(r["agent"]["points"] for r in unseen),
                "n_races": len(unseen),
            },
        },
    }
    out_path = DASHBOARD_DATA_DIR / f"season_{SEASON}.json"
    out_path.write_text(json.dumps(season_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== Termine : {out_path} ===")
    print(f"Points reels : {real_total}  |  Points agent : {agent_total}  |  GP traites : {len(season_races)}/{len(events)}")
    if failed_gps:
        print(f"GP exclus/en echec ({len(failed_gps)}) : {', '.join(failed_gps)}")


if __name__ == "__main__":
    main()
