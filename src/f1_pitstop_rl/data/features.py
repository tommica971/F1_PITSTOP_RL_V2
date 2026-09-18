"""
Phase 1.2 — Jeu de variables (état/observation) — C5.1.2 / C5.2.1
===================================================================
Enrichit le dataset brut de Gasly (Phase 1.1) avec les variables d'état
définies dans la formalisation MDP (Phase 0.2) :

    - gap_avant / gap_arriere        : écart temps (s) avec les voitures
                                        directement devant/derrière
    - rival_avant_vient_de_pitter    : le rival juste devant a pitté il y a
                                        <= PIT_REACTION_WINDOW tours
    - rival_arriere_vient_de_pitter  : idem pour le rival juste derrière
    - fenetre_undercut               : gap_avant sous le seuil undercut
                                        calibré par circuit
    - delta_pluie_3tours             : tendance météo (pluie qui s'installe /
                                        se dissipe) sur les 3 derniers tours
    - tours_restants                 : tours restants avant la fin de course

Nécessite les temps de tous les pilotes (pas seulement Gasly) pour calculer
les écarts -> requêtes supplémentaires vers le miroir TracingInsights.

Limite méthodologique assumée et documentée : les écarts sont approximés par
différence de temps de session cumulé (sesT) à numéro de tour identique. Pour
un pilote en délta d'un tour (lapped), l'approximation peut être légèrement
biaisée -- acceptable pour l'usage RL visé, à affiner si besoin via les temps
de passage réels (FastF1 get_position_data) dans une itération ultérieure.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from gp_pool_config import GP_POOL, DRIVER_CODE

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PIT_REACTION_WINDOW = 2       # tours pour considérer un pit "récent" (undercut/overcut réactif)
UNDERCUT_GAP_THRESHOLD_S = 2.5  # seuil générique ; à recalibrer par circuit en Phase 1.3


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


def get_driver_list(season: int, event: str) -> list[str]:
    url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/drivers.json"
    raw = fetch_json(url)
    if raw is None:
        return []
    return [d["driver"] for d in raw.get("drivers", [])]


def get_all_drivers_laps(season: int, event: str) -> pd.DataFrame:
    """Récupère lap/pos/sesT/pin/pout/wR pour tous les pilotes d'une course."""
    drivers = get_driver_list(season, event)
    frames = []
    for drv in drivers:
        url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{drv}/laptimes.json"
        raw = fetch_json(url)
        if raw is None:
            continue
        df = pd.DataFrame(raw)
        keep = ["lap", "pos", "sesT", "pin", "pout", "drv", "wR"]
        df = df[[c for c in keep if c in df.columns]]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_gap_and_undercut_features(all_drivers: pd.DataFrame, driver: str = DRIVER_CODE) -> pd.DataFrame:
    """Calcule gap_avant/gap_arriere et flags undercut pour un pilote donné, tour par tour."""
    all_drivers = all_drivers.copy()
    all_drivers["pos"] = pd.to_numeric(all_drivers["pos"], errors="coerce")
    all_drivers["pit_this_lap"] = all_drivers["pin"].apply(lambda x: x not in (None, "None"))

    rows = []
    for lap_num, lap_group in all_drivers.groupby("lap"):
        lap_group = lap_group.dropna(subset=["pos"]).sort_values("pos")
        self_row = lap_group[lap_group["drv"] == driver]
        if self_row.empty:
            continue
        self_row = self_row.iloc[0]
        self_pos = self_row["pos"]
        self_sesT = self_row["sesT"]

        ahead = lap_group[lap_group["pos"] == self_pos - 1]
        behind = lap_group[lap_group["pos"] == self_pos + 1]

        gap_avant = float(self_sesT - ahead["sesT"].iloc[0]) if not ahead.empty else np.nan
        gap_arriere = float(behind["sesT"].iloc[0] - self_sesT) if not behind.empty else np.nan

        rows.append({
            "lap": lap_num,
            "position": int(self_pos),
            "gap_avant": gap_avant,
            "gap_arriere": gap_arriere,
            "rival_avant_drv": ahead["drv"].iloc[0] if not ahead.empty else None,
            "rival_arriere_drv": behind["drv"].iloc[0] if not behind.empty else None,
            "rival_avant_pitte_ce_tour": bool(ahead["pit_this_lap"].iloc[0]) if not ahead.empty else False,
            "rival_arriere_pitte_ce_tour": bool(behind["pit_this_lap"].iloc[0]) if not behind.empty else False,
        })

    return pd.DataFrame(rows)


def add_rolling_rival_pit_flags(gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Etend le flag 'pitte ce tour' en fenêtre glissante (PIT_REACTION_WINDOW tours),
    pour capturer un undercut/overcut réactif même si l'agent réagit 1-2 tours après.
    """
    gap_df = gap_df.sort_values("lap").reset_index(drop=True)
    gap_df["rival_avant_vient_de_pitter"] = (
        gap_df["rival_avant_pitte_ce_tour"]
        .rolling(PIT_REACTION_WINDOW, min_periods=1).max().astype(bool)
    )
    gap_df["rival_arriere_vient_de_pitter"] = (
        gap_df["rival_arriere_pitte_ce_tour"]
        .rolling(PIT_REACTION_WINDOW, min_periods=1).max().astype(bool)
    )
    gap_df["fenetre_undercut"] = gap_df["gap_avant"] < UNDERCUT_GAP_THRESHOLD_S
    return gap_df


def add_weather_trend(df: pd.DataFrame) -> pd.DataFrame:
    """delta_pluie_3tours : tendance pluie sur les 3 derniers tours (installation/dissipation)."""
    df = df.sort_values(["season", "event", "lap_number"]).reset_index(drop=True)
    df["is_raining_int"] = df["is_raining"].astype(int)
    df["delta_pluie_3tours"] = df.groupby(["season", "event"])["is_raining_int"] \
        .transform(lambda s: s.diff(periods=3))
    df = df.drop(columns=["is_raining_int"])
    return df


def add_race_progress(df: pd.DataFrame) -> pd.DataFrame:
    """tours_restants : tours restants avant la fin de la course (basé sur le dernier tour observé)."""
    df = df.copy()
    max_lap = df.groupby(["season", "event"])["lap_number"].transform("max")
    df["tours_restants"] = max_lap - df["lap_number"]
    return df


def run_feature_pipeline() -> pd.DataFrame:
    base = pd.read_parquet(RAW_DIR / "master_dataset.parquet")
    enriched_frames = []

    print(f"Construction des features pour {len(GP_POOL)} GP\n")

    for gp in GP_POOL:
        season, event, role = gp["season"], gp["event"], gp["role"]
        print(f"[{role}] {season} {event}")

        gp_base = base[(base["season"] == season) & (base["event"] == event)].copy()
        if gp_base.empty:
            print("    -> pas de données de base, GP ignoré")
            continue

        all_drivers = get_all_drivers_laps(season, event)
        if all_drivers.empty:
            print("    -> impossible de récupérer les autres pilotes, gaps non calculés")
            enriched_frames.append(gp_base)
            continue

        gap_df = build_gap_and_undercut_features(all_drivers)
        gap_df = add_rolling_rival_pit_flags(gap_df)

        merged = gp_base.merge(gap_df, left_on="lap_number", right_on="lap", how="left")
        merged = merged.drop(columns=["lap"], errors="ignore")

        n_gap_ok = merged["gap_avant"].notna().sum()
        print(f"    -> {len(merged)} tours | gap_avant calculé sur {n_gap_ok} tours "
              f"| {merged['fenetre_undercut'].sum()} tours en fenêtre undercut")

        enriched_frames.append(merged)

    full = pd.concat(enriched_frames, ignore_index=True)
    full = add_weather_trend(full)
    full = add_race_progress(full)

    out_path = PROCESSED_DIR / "features_dataset.parquet"
    full.to_parquet(out_path, index=False)
    print(f"\nDataset enrichi : {full.shape} -> {out_path}")
    return full


if __name__ == "__main__":
    df = run_feature_pipeline()

    print("\n=== Colonnes du jeu de variables final ===")
    for c in df.columns:
        print(f"  - {c} ({df[c].dtype})")

    print("\n=== Valeurs manquantes ===")
    na = df.isna().sum()
    print(na[na > 0])
