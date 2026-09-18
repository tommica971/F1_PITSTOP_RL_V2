"""
Écarts réels au tour 1 (Phase 2.2 — remplace l'approximation constante)
===========================================================================
F1PitStopEnv approximait l'écart initial entre l'agent et chaque fantôme par
une constante (AVG_GAP_PER_POSITION_S * différence de position). Un test de
sanité (rejeu de la stratégie réelle de Gasly à Bahreïn 2023 contre son vrai
résultat P9) a montré que c'est le principal facteur d'erreur résiduel une
fois le modèle de rythme calibré correctement.

Ce script récupère le temps de session cumulé (sesT) réel de tous les
pilotes au tour 1, pour chaque GP du pool RL -- donnée déjà présente dans
laptimes.json mais jamais persistée jusqu'ici (remplacée par un diff de
temps au tour dans les pipelines existants).

Sortie : data/raw/initial_gaps.parquet (season, event, driver, sesT_lap1)
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from gp_pool_config import GP_POOL

RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"


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


def run_extraction() -> pd.DataFrame:
    rows = []
    print(f"Extraction des ecarts reels au tour 1 pour {len(GP_POOL)} GP\n")

    for gp in GP_POOL:
        season, event = gp["season"], gp["event"]
        print(f"  {season} {event}...", end=" ")
        drivers = get_driver_list(season, event)
        n_found = 0
        for drv in drivers:
            url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{drv}/laptimes.json"
            raw = fetch_json(url)
            if raw is None or not raw.get("lap"):
                continue
            try:
                idx = raw["lap"].index(1)
                ses_t = raw["sesT"][idx]
                if ses_t is not None:
                    rows.append({"season": season, "event": event, "driver": drv, "sesT_lap1": float(ses_t)})
                    n_found += 1
            except (ValueError, IndexError, TypeError):
                continue
        print(f"{n_found} pilotes")

    df = pd.DataFrame(rows)
    out_path = RAW_DIR / "initial_gaps.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n{len(df)} lignes -> {out_path}")
    return df


if __name__ == "__main__":
    run_extraction()
