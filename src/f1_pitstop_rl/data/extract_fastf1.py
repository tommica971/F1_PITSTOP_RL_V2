"""
Phase 1.1 — Extraction et nettoyage des données (C5.2.1)
=========================================================
Pipeline CANONIQUE utilisant la librairie officielle FastF1.

C'est ce script qui doit être exécuté sur une machine avec accès réseau standard
(FastF1 interroge livetiming.formula1.com et l'API Ergast/Jolpica). Il n'est pas
exécuté dans l'environnement de développement utilisé pour ce document (accès
réseau restreint), mais produit un schéma de sortie strictement identique à
extract_tracinginsights.py, dont il constitue la référence de vérité.

Installation :
    pip install fastf1 pandas pyarrow

Documentation FastF1 : https://docs.fastf1.dev
"""

import sys
from pathlib import Path

import fastf1
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gp_pool_config import GP_POOL, DRIVER_CODE

# --- Configuration du cache FastF1 (obligatoire : évite de re-télécharger à
#     chaque exécution, requis par la librairie) ---
CACHE_DIR = Path(__file__).parent / ".fastf1_cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path(__file__).parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_session(season: int, event: str, driver: str = DRIVER_CODE) -> pd.DataFrame | None:
    """
    Charge une session de course FastF1 et extrait, pour un pilote donné,
    les données tour-par-tour fusionnées avec la météo et le statut piste.
    """
    try:
        session = fastf1.get_session(season, event, "R")
        # laps + car_data + weather : nécessaires pour l'espace d'observation
        session.load(laps=True, telemetry=False, weather=True, messages=True)
    except Exception as exc:
        print(f"  [!] Impossible de charger {season} {event} : {exc}")
        return None

    driver_laps = session.laps.pick_drivers(driver).copy()
    if driver_laps.empty:
        print(f"  [!] Aucun tour pour {driver} sur {season} {event}")
        return None

    # --- Fusion météo : jointure asof sur le temps de session, comme FastF1
    #     le recommande pour aligner les tours avec les échantillons météo
    #     (fréquence d'échantillonnage météo != fréquence des tours) ---
    weather = session.weather_data.copy()
    driver_laps = pd.merge_asof(
        driver_laps.sort_values("Time"),
        weather.sort_values("Time"),
        on="Time",
        direction="nearest",
    )

    # --- Statut piste (Red Flag / Safety Car / VSC) par tour, via les
    #     messages de direction de course (Race Control Messages) ---
    rcm = session.race_control_messages.copy()

    def _lap_has_flag(lap_number: int, keyword_cat: str, keyword_val: str) -> bool:
        lap_msgs = rcm[rcm["Lap"] == lap_number]
        if keyword_cat == "flag":
            return (lap_msgs["Flag"] == keyword_val).any()
        return lap_msgs["Message"].str.upper().str.contains(keyword_val, na=False).any()

    driver_laps["is_red_flag"] = driver_laps["LapNumber"].apply(
        lambda ln: _lap_has_flag(ln, "flag", "RED")
    )
    driver_laps["is_safety_car"] = driver_laps["LapNumber"].apply(
        lambda ln: _lap_has_flag(ln, "msg", "SAFETY CAR DEPLOYED")
        or _lap_has_flag(ln, "msg", "SAFETY CAR IN THIS LAP")
    )
    driver_laps["is_vsc"] = driver_laps["LapNumber"].apply(
        lambda ln: _lap_has_flag(ln, "msg", "VIRTUAL SAFETY CAR")
    )

    driver_laps["season"] = season
    driver_laps["event"] = event
    return driver_laps


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage et renommage — schéma identique à extract_tracinginsights.clean_laps."""
    df = df.copy()

    df = df.rename(columns={
        "LapNumber": "lap_number", "Compound": "tyre_compound", "Stint": "stint_number",
        "TyreLife": "tyre_age_laps", "Team": "team", "Driver": "driver",
        "AirTemp": "air_temp_c", "Humidity": "humidity_pct", "Pressure": "pressure_hpa",
        "Rainfall": "is_raining", "TrackTemp": "track_temp_c",
        "WindDirection": "wind_direction_deg", "WindSpeed": "wind_speed_kmh",
        "FreshTyre": "is_fresh_tyre", "Deleted": "is_deleted", "IsAccurate": "is_inaccurate_raw",
    })

    df["lap_time_s"] = df["LapTime"].dt.total_seconds()
    # IsAccurate=True veut dire "fiable" dans FastF1 -> on stocke l'inverse pour
    # rester cohérent avec le nom "is_inaccurate" du pipeline TracingInsights
    df["is_inaccurate"] = ~df["is_inaccurate_raw"].fillna(False)

    keep_cols = [
        "season", "event", "driver", "team", "lap_number", "stint_number",
        "tyre_compound", "tyre_age_laps", "is_fresh_tyre", "lap_time_s",
        "air_temp_c", "humidity_pct", "pressure_hpa", "is_raining",
        "track_temp_c", "wind_direction_deg", "wind_speed_kmh",
        "is_deleted", "is_inaccurate", "is_red_flag", "is_safety_car", "is_vsc",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Cf. note méthodologique et historique de debug dans extract_tracinginsights.py
    # (Phase 1.3) : IsAccurate n'est PAS utilisé comme filtre d'exclusion dur, et le
    # tour 1 (potentiellement non représentatif selon la référence de temps FastF1)
    # ainsi que les tours SC/VSC sont exclus structurellement, pas via un simple
    # seuil de durée -- cohérence garantie avec le pipeline TracingInsights.
    df["usable_for_reward"] = (
        df["lap_time_s"].gt(0)
        & df["lap_time_s"].lt(400)
        & ~df["is_deleted"].fillna(False)
        & ~df["is_safety_car"].fillna(False)
        & ~df["is_vsc"].fillna(False)
        & (df["lap_number"] != df.groupby(["season", "event"])["lap_number"].transform("min"))
    )

    return df


def run_pipeline() -> pd.DataFrame:
    all_gp_frames = []
    print(f"Extraction FastF1 du pool de {len(GP_POOL)} GP pour le pilote {DRIVER_CODE}\n")

    for gp in GP_POOL:
        season, event, role = gp["season"], gp["event"], gp["role"]
        print(f"[{role}] {season} {event}")

        raw = extract_session(season, event)
        if raw is None:
            continue

        laps = clean_laps(raw)
        laps["role"] = role
        laps["known_issues"] = ",".join(gp["known_issues"]) if gp["known_issues"] else ""

        out_path = OUTPUT_DIR / f"{season}_{event.replace(' ', '_')}_{DRIVER_CODE}.parquet"
        laps.to_parquet(out_path, index=False)
        all_gp_frames.append(laps)

    master = pd.concat(all_gp_frames, ignore_index=True)
    master_path = OUTPUT_DIR / "master_dataset.parquet"
    master.to_parquet(master_path, index=False)
    print(f"\nDataset consolidé : {len(master)} lignes -> {master_path}")
    return master


if __name__ == "__main__":
    df = run_pipeline()
    print("\n=== Statistiques de validation (C5.2.1) ===")
    print(df.groupby(["role", "event"]).agg(
        n_laps=("lap_number", "count"),
        n_usable=("usable_for_reward", "sum"),
        pct_raining=("is_raining", "mean"),
    ).round(3))
