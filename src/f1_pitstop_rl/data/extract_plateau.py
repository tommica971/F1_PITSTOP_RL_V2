"""
Extraction du plateau complet — tous pilotes, pool de GP (Phase 0.3)
=======================================================================
Objectif double :
  1. Nourrir l'analyse du plateau (rythme par équipe, stratégies pneus,
     patterns stratégiques vs position de départ) -- compréhension du sport.
  2. Fournir les données nécessaires à la calibration des 19 adversaires
     fantômes de F1PitStopEnv (Phase 2) -- pas des bots inventés, un rejeu
     réaliste basé sur le pace et la stratégie réels de chaque équipe.

Réutilise la logique d'extraction déjà écrite dans features.py
(get_driver_list, fetch_json) mais persiste TOUS les pilotes au lieu de ne
garder que les gaps calculés pour Gasly.

Sortie : data/raw/all_drivers_dataset.parquet (1 ligne = 1 tour x 1 pilote)
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
sys.path.insert(0, str(Path(__file__).parent))
from gp_pool_config import GP_POOL
from extract_tracinginsights import extract_track_status

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _repo_base(season: int) -> str:
    if season == 2026:
        return "https://raw.githubusercontent.com/TracingInsights/2026/main"
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


def get_starting_grid(season: int, event: str, driver_team_map: dict[str, str]) -> dict[str, int]:
    """Position de départ = position au tour 1 (proxy simple, cf. limite assumée)."""
    grid = {}
    for drv in driver_team_map:
        url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{drv}/laptimes.json"
        raw = fetch_json(url)
        if raw is None or not raw.get("lap"):
            continue
        try:
            idx = raw["lap"].index(1)
            pos = raw["pos"][idx]
            if pos not in (None, "None"):
                grid[drv] = int(pos)
        except (ValueError, IndexError):
            continue
    return grid


def extract_all_drivers_gp(season: int, event: str, role: str) -> pd.DataFrame:
    driver_team_map = get_driver_team_map(season, event)
    if not driver_team_map:
        print(f"    -> pas de liste pilotes, GP ignoré")
        return pd.DataFrame()

    starting_grid = get_starting_grid(season, event, driver_team_map)

    frames = []
    for drv, team in driver_team_map.items():
        url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{drv}/laptimes.json"
        raw = fetch_json(url)
        if raw is None:
            continue
        df = pd.DataFrame(raw)
        keep = ["lap", "pos", "sesT", "compound", "stint", "life", "pin", "pout", "del", "iacc"]
        df = df[[c for c in keep if c in df.columns]]
        df["driver"] = drv
        df["team"] = team
        df["starting_position"] = starting_grid.get(drv)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    plateau = pd.concat(frames, ignore_index=True)
    plateau["season"] = season
    plateau["event"] = event
    plateau["role"] = role

    status = extract_track_status(season, event)
    plateau = plateau.merge(status, left_on="lap", right_on="lap", how="left")
    for col in ["is_red_flag", "is_safety_car", "is_vsc"]:
        if col not in plateau.columns:
            plateau[col] = False
        # cf. bug dtype découvert Phase 1.3 : cast bool explicite obligatoire
        plateau[col] = plateau[col].fillna(False).astype(bool)

    return plateau


def clean_plateau(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["season", "event", "driver", "lap"]).reset_index(drop=True)
    df["lap_time_s"] = df.groupby(["season", "event", "driver"])["sesT"].diff()
    first_lap_mask = df["lap"] == df.groupby(["season", "event", "driver"])["lap"].transform("min")
    df.loc[first_lap_mask, "lap_time_s"] = df.loc[first_lap_mask, "sesT"]

    df["is_deleted"] = df["del"].astype(bool)
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["life"] = pd.to_numeric(df["life"], errors="coerce")
    df["stint"] = pd.to_numeric(df["stint"], errors="coerce")

    # Cout reel d'un arret : pin = temps (sesT) d'entree dans la voie des stands,
    # pout = temps (sesT) de sortie. La difference approxime le temps passe en
    # voie des stands (proxy documente -- pas exactement le "temps perdu vs etre
    # reste en piste" que rapporte FastF1 nativement, qui necessite une reference
    # de temps de trajet a vitesse normale sur ce meme troncon).
    df["pin_num"] = pd.to_numeric(df["pin"], errors="coerce")
    df["pout_num"] = pd.to_numeric(df["pout"], errors="coerce")
    pit_in = df.dropna(subset=["pin_num"])[["season", "event", "driver", "lap", "pin_num"]]
    pit_out = df.dropna(subset=["pout_num"])[["season", "event", "driver", "lap", "pout_num"]]
    # pout est generalement enregistre au tour suivant le pin -> on associe le
    # pout du tour N ou N+1 au pin du tour N (le plus proche dans le temps)
    pit_events = pit_in.merge(
        pit_out, on=["season", "event", "driver"], suffixes=("_in", "_out")
    )
    pit_events = pit_events[pit_events["pout_num"] > pit_events["pin_num"]]
    pit_events["pit_gap"] = pit_events["pout_num"] - pit_events["pin_num"]
    # garder la sortie la plus proche dans le temps pour chaque entree
    pit_events = pit_events.sort_values("pit_gap").drop_duplicates(
        subset=["season", "event", "driver", "lap_in"], keep="first"
    )
    pit_events = pit_events.rename(columns={"lap_in": "lap_number", "pit_gap": "pit_time_loss"})
    df = df.rename(columns={"lap": "lap_number"})
    df = df.merge(
        pit_events[["season", "event", "driver", "lap_number", "pit_time_loss"]],
        on=["season", "event", "driver", "lap_number"], how="left",
    )

    df = df.rename(columns={
        "pos": "position", "compound": "tyre_compound",
        "stint": "stint_number", "life": "tyre_age_laps",
    })

    # Même garde-fou que master_dataset.parquet (Phase 1.3, debug) : le tour 1
    # n'est jamais un vrai temps au tour (sesT cumulé), exclu structurellement.
    df["usable_for_pace"] = (
        df["lap_time_s"].gt(0)
        & df["lap_time_s"].lt(400)
        & ~df["is_deleted"].astype(bool)
        & ~df["is_safety_car"].astype(bool)
        & ~df["is_vsc"].astype(bool)
        & (df["lap_number"] != df.groupby(["season", "event", "driver"])["lap_number"].transform("min"))
    )

    keep_cols = [
        "season", "event", "role", "driver", "team", "starting_position",
        "lap_number", "position", "stint_number", "tyre_compound",
        "tyre_age_laps", "lap_time_s", "is_deleted", "usable_for_pace",
        "pit_time_loss", "is_red_flag", "is_safety_car", "is_vsc",
    ]
    return df[[c for c in keep_cols if c in df.columns]]


def run_plateau_extraction() -> pd.DataFrame:
    all_frames = []
    print(f"Extraction du plateau complet pour {len(GP_POOL)} GP\n")

    for gp in GP_POOL:
        season, event, role = gp["season"], gp["event"], gp["role"]
        print(f"[{role}] {season} {event}")
        raw = extract_all_drivers_gp(season, event, role)
        if raw.empty:
            continue
        cleaned = clean_plateau(raw)
        n_drivers = cleaned["driver"].nunique()
        print(f"    -> {n_drivers} pilotes, {len(cleaned)} tours")
        all_frames.append(cleaned)

    full = pd.concat(all_frames, ignore_index=True)
    out_path = RAW_DIR / "all_drivers_dataset.parquet"
    full.to_parquet(out_path, index=False)
    print(f"\nDataset plateau complet : {full.shape} -> {out_path}")
    return full


if __name__ == "__main__":
    df = run_plateau_extraction()
    print("\nAperçu :")
    print(df.head(10).to_string())
    print(f"\nPilotes uniques : {df['driver'].nunique()}")
    print(f"Équipes uniques : {df['team'].nunique()}")
