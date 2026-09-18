"""
Jeu de calibrage étendu — saison 2024 complète (Phase 2.2, hors pool RL)
===========================================================================
Le pool RL (F1_PITSTOP_RL/src/f1_pitstop_rl/config/gp_pool_config.py, 9 GP)
reste figé -- décision actée en Phase 0.3, ne pas y toucher.

Ce script sert UNIQUEMENT à isoler statistiquement l'effet de dégradation
pneu (vs effet carburant / évolution piste) avec plus de volume de données,
pour calibrer plus finement la fonction de récompense en Phase 2.2. Les 24
GP de la saison 2024 sont utilisés ici comme échantillon de calibrage, pas
comme extension du pool d'entraînement de l'agent RL.

Sortie : data/raw/calibration_2024_dataset.parquet
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_tracinginsights import extract_track_status

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASON_2024_EVENTS = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
    "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
    "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
    "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix",
    "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]


def _repo_base(season: int) -> str:
    return f"https://raw.githubusercontent.com/TracingInsights-Archive/{season}/main"


def fetch_json(url: str) -> dict | None:
    result = subprocess.run(
        ["curl", "-s", "-w", "\n%{http_code}", url],
        capture_output=True, text=True, timeout=30,
    )
    *body_lines, status_code = result.stdout.split("\n")
    body = "\n".join(body_lines)
    if status_code.strip() != "200" or not body.strip():
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def get_driver_team_map(season: int, event: str) -> dict[str, str]:
    url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/drivers.json"
    raw = fetch_json(url)
    if raw is None:
        return {}
    return {d["driver"]: d["team"] for d in raw.get("drivers", [])}


def extract_gp(season: int, event: str) -> pd.DataFrame:
    driver_team_map = get_driver_team_map(season, event)
    if not driver_team_map:
        return pd.DataFrame()

    frames = []
    for drv, team in driver_team_map.items():
        url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{drv}/laptimes.json"
        raw = fetch_json(url)
        if raw is None:
            continue
        df = pd.DataFrame(raw)
        keep = ["lap", "pos", "sesT", "compound", "stint", "life", "del"]
        df = df[[c for c in keep if c in df.columns]]
        df["driver"] = drv
        df["team"] = team
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    plateau = pd.concat(frames, ignore_index=True)
    plateau["season"] = season
    plateau["event"] = event

    status = extract_track_status(season, event)
    if not status.empty:
        plateau = plateau.merge(status, left_on="lap", right_on="lap", how="left")
    for col in ["is_red_flag", "is_safety_car", "is_vsc"]:
        if col not in plateau.columns:
            plateau[col] = False
        plateau[col] = plateau[col].fillna(False).astype(bool)

    return plateau


def clean_gp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["season", "event", "driver", "lap"]).reset_index(drop=True)
    df["lap_time_s"] = df.groupby(["season", "event", "driver"])["sesT"].diff()
    first_lap_mask = df["lap"] == df.groupby(["season", "event", "driver"])["lap"].transform("min")
    df.loc[first_lap_mask, "lap_time_s"] = df.loc[first_lap_mask, "sesT"]

    df["is_deleted"] = df["del"].astype(bool)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["life"] = pd.to_numeric(df["life"], errors="coerce")
    df["stint"] = pd.to_numeric(df["stint"], errors="coerce")

    df = df.rename(columns={
        "lap": "lap_number", "pos": "position", "compound": "tyre_compound",
        "stint": "stint_number", "life": "tyre_age_laps",
    })

    # Position de depart = position au tour 1 -- deja disponible dans les
    # donnees deja recuperees, aucun appel reseau supplementaire necessaire.
    # df est trie par lap_number croissant (cf. sort_values ci-dessus), donc
    # .iloc[0] par groupe correspond bien au tour 1.
    df["starting_position"] = df.groupby(["season", "event", "driver"])["position"].transform("first")

    df["usable"] = (
        df["lap_time_s"].gt(0)
        & df["lap_time_s"].lt(400)
        & ~df["is_deleted"].astype(bool)
        & ~df["is_safety_car"].astype(bool)
        & ~df["is_vsc"].astype(bool)
        & (df["lap_number"] != df.groupby(["season", "event", "driver"])["lap_number"].transform("min"))
    )

    keep_cols = [
        "season", "event", "driver", "team", "lap_number", "position",
        "starting_position", "stint_number", "tyre_compound", "tyre_age_laps",
        "lap_time_s", "usable",
    ]
    return df[[c for c in keep_cols if c in df.columns]]


def run_extraction() -> pd.DataFrame:
    frames = []
    print(f"Extraction du jeu de calibrage : saison 2024 complète ({len(SEASON_2024_EVENTS)} GP)\n")
    for event in SEASON_2024_EVENTS:
        print(f"  {event}...", end=" ")
        raw = extract_gp(2024, event)
        if raw.empty:
            print("echec")
            continue
        cleaned = clean_gp(raw)
        print(f"{cleaned['driver'].nunique()} pilotes, {len(cleaned)} tours")
        frames.append(cleaned)

    full = pd.concat(frames, ignore_index=True)
    out_path = RAW_DIR / "calibration_2024_dataset.parquet"
    full.to_parquet(out_path, index=False)
    print(f"\nDataset de calibrage : {full.shape} -> {out_path}")
    return full


if __name__ == "__main__":
    run_extraction()
