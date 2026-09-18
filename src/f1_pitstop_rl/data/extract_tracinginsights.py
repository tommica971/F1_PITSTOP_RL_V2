"""
Phase 1.1 — Extraction et nettoyage des données (C5.2.1)
=========================================================
Pipeline FONCTIONNEL, exécutable dans cet environnement.

Source de données : miroir TracingInsights (github.com/TracingInsights[-Archive]),
qui republie les données FastF1 en JSON statique. Utilisé ici car l'environnement
d'exécution n'a pas d'accès réseau direct aux serveurs live timing FastF1
(livetiming.formula1.com, Ergast, etc.) — seul raw.githubusercontent.com est autorisé.

Le schéma de sortie (colonnes, types, granularité = 1 ligne par tour par pilote)
est identique à celui produit par extract_fastf1.py, afin que les deux pipelines
soient interchangeables selon l'environnement d'exécution.

Sortie : un fichier Parquet par GP + un fichier Parquet consolidé, dans ./data/raw/
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from gp_pool_config import GP_POOL, DRIVER_CODE

OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Anomalies de piste à signaler au niveau tour (masquage en Phase 2, pas ici)
TRACK_ANOMALY_FLAGS = {"RED", "SAFETY CAR", "VIRTUAL SAFETY CAR"}


def _repo_base(season: int) -> str:
    """Retourne l'URL de base du repo TracingInsights selon la saison."""
    if season == 2026:
        return "https://raw.githubusercontent.com/TracingInsights/2026/main"
    return f"https://raw.githubusercontent.com/TracingInsights-Archive/{season}/main"


def fetch_json(url: str) -> dict | None:
    """Récupère et parse un fichier JSON distant. Retourne None si indisponible."""
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


def extract_driver_laps(season: int, event: str, driver: str = DRIVER_CODE) -> pd.DataFrame | None:
    """
    Extrait les données tour-par-tour d'un pilote pour une course donnée.
    laptimes.json contient déjà les données météo alignées par tour (wT, wAT, wH,
    wP, wR, wTT, wWD, wWS) — pas besoin de fusion séparée avec weather.json.
    """
    url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/{driver}/laptimes.json"
    raw = fetch_json(url)
    if raw is None:
        print(f"  [!] Pas de données laptimes pour {season} {event} ({driver})")
        return None

    df = pd.DataFrame(raw)
    df["season"] = season
    df["event"] = event
    return df


def extract_track_status(season: int, event: str) -> pd.DataFrame:
    """
    Extrait les messages de direction de course (rcm.json) et construit
    une table tour -> anomalies (Red Flag / Safety Car / VSC) détectées.
    """
    url = f"{_repo_base(season)}/{event.replace(' ', '%20')}/Race/rcm.json"
    raw = fetch_json(url)
    if raw is None:
        return pd.DataFrame(columns=["lap", "is_red_flag", "is_safety_car", "is_vsc"])

    rcm = pd.DataFrame(raw)

    # Certains messages (ex: pre-course) n'ont pas de tour associe (lap='None'
    # textuel) -- decouvert sur Miami 2024. On les exclut avant le groupby pour
    # eviter un dtype incoherent (int/str mixe) qui casse les fusions en aval.
    rcm["lap"] = pd.to_numeric(rcm["lap"], errors="coerce")
    rcm = rcm.dropna(subset=["lap"])
    if rcm.empty:
        return pd.DataFrame(columns=["lap", "is_red_flag", "is_safety_car", "is_vsc"])
    rcm["lap"] = rcm["lap"].astype(int)

    # Détection via champs structurés (cat/flag/status) plutôt que regex sur le texte
    # libre "msg" : une recherche naïve de "RED FLAG" dans msg matche à tort
    # "CHEQUERED FLAG" (qui contient la sous-chaîne "...ERED FLAG"). Les champs
    # catégoriels sont sans ambiguïté.
    rcm["is_red_flag_row"] = rcm["flag"].astype(str).str.upper().eq("RED")
    rcm["is_safety_car_row"] = (
        rcm["cat"].astype(str).eq("SafetyCar")
        & rcm["status"].astype(str).isin(["DEPLOYED", "IN THIS LAP"])
    )
    rcm["is_vsc_row"] = rcm["msg"].astype(str).str.upper().str.contains("VIRTUAL SAFETY CAR", na=False)

    per_lap = rcm.groupby("lap").agg(
        is_red_flag=("is_red_flag_row", "any"),
        is_safety_car=("is_safety_car_row", "any"),
        is_vsc=("is_vsc_row", "any"),
    ).reset_index()

    return per_lap


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage : typage, calcul du temps au tour, gestion des valeurs manquantes,
    exclusion des tours non exploitables (formation lap, données incomplètes).
    """
    df = df.copy()

    # sesT = temps de session cumulé (s) -> temps au tour par différence
    df = df.sort_values(["event", "season", "lap"]).reset_index(drop=True)
    df["lap_time_s"] = df.groupby(["season", "event"])["sesT"].diff()
    # premier tour : pas de tour précédent, on garde sesT tel quel comme approximation
    first_lap_mask = df.groupby(["season", "event"])["lap"].transform("min") == df["lap"]
    df.loc[first_lap_mask, "lap_time_s"] = df.loc[first_lap_mask, "sesT"]

    # Flags qualité FastF1 déjà présents dans la donnée source
    df["is_deleted"] = df["del"].astype(bool)
    df["is_inaccurate"] = df["iacc"].astype(bool)

    # Renommage explicite (documentation C5.2.1 : traçabilité des variables)
    df = df.rename(columns={
        "lap": "lap_number", "compound": "tyre_compound", "stint": "stint_number",
        "life": "tyre_age_laps", "team": "team", "drv": "driver",
        "wAT": "air_temp_c", "wH": "humidity_pct", "wP": "pressure_hpa",
        "wR": "is_raining", "wTT": "track_temp_c", "wWD": "wind_direction_deg",
        "wWS": "wind_speed_kmh", "fresh": "is_fresh_tyre", "pos": "position_raw",
    })

    keep_cols = [
        "season", "event", "driver", "team", "lap_number", "stint_number",
        "tyre_compound", "tyre_age_laps", "is_fresh_tyre", "lap_time_s",
        "air_temp_c", "humidity_pct", "pressure_hpa", "is_raining",
        "track_temp_c", "wind_direction_deg", "wind_speed_kmh",
        "is_deleted", "is_inaccurate", "is_red_flag", "is_safety_car", "is_vsc",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # NOTE methodologique (a documenter dans le dossier, C5.2.1) :
    # Le flag "iacc" (IsAccurate natif FastF1) est TRES restrictif -- sur certains GP
    # pluvieux il exclurait jusqu'a 60% des tours. Il signale une incertitude sur la
    # reconstruction du temps theorique optimal (utile pour du chronometrage officiel),
    # mais n'invalide pas le temps au tour lui-meme pour un usage RL. On le conserve
    # donc comme FEATURE de confiance plutot que comme filtre d'exclusion strict.
    #
    # Historique de debug (Phase 1.3, notebook 01_exploration_donnees.ipynb) -- a
    # documenter dans le dossier ecrit :
    #   1. usable_for_reward ne s'appuyait au depart que sur un seuil "< 400s" pour
    #      detecter l'artefact du tour 1 (temps de session cumule sesT, jamais un
    #      vrai temps au tour -- cf. ligne ~120). Ce seuil supposait que l'artefact
    #      serait toujours tres grand (3000-8000s sur la plupart des GP).
    #   2. Sur le Canada 2024, cet artefact valait 366.5s -- SOUS le seuil -- et est
    #      passe a travers le filtre, faussant severement la calibration en aval
    #      (valide via GroupKFold : R² de generalisation -0.586 avant correctif,
    #      +0.289 apres).
    #   3. Les tours sous Safety Car / VSC n'etaient pas exclus non plus alors que
    #      la decision Phase 0.2 le prevoyait explicitement.
    # -> Le tour 1 est desormais exclu structurellement (par numero de tour, pas par
    #    seuil de duree), ainsi que les tours SC/VSC. Le seuil <400s est conserve en
    #    filet de securite supplementaire, pas comme mecanisme de detection principal.
    df["usable_for_reward"] = (
        df["lap_time_s"].gt(0)
        & df["lap_time_s"].lt(400)
        & ~df["is_deleted"].astype(bool)
        & ~df["is_safety_car"].astype(bool)
        & ~df["is_vsc"].astype(bool)
        & (df["lap_number"] != df.groupby(["season", "event"])["lap_number"].transform("min"))
    )

    return df


def run_pipeline() -> pd.DataFrame:
    all_gp_frames = []

    print(f"Extraction du pool de {len(GP_POOL)} GP pour le pilote {DRIVER_CODE}\n")

    for gp in GP_POOL:
        season, event, role = gp["season"], gp["event"], gp["role"]
        print(f"[{role}] {season} {event}")

        laps = extract_driver_laps(season, event)
        if laps is None:
            continue

        status = extract_track_status(season, event)
        laps = laps.merge(status, left_on="lap", right_on="lap", how="left")
        for col in ["is_red_flag", "is_safety_car", "is_vsc"]:
            if col not in laps.columns:
                laps[col] = False
            # IMPORTANT : fillna seul laisse la colonne en dtype "object" après un
            # merge avec valeurs manquantes -- l'opérateur "~" sur un object dtype
            # fait un NON binaire (~True=-2, ~False=-1, tous deux "vrais"), pas un
            # NON logique. Sans ce cast explicite, le filtre usable_for_reward
            # n'exclut RIEN silencieusement (bug découvert lors de la régénération
            # du dataset, Phase 1.3).
            laps[col] = laps[col].fillna(False).astype(bool)

        laps = clean_laps(laps)
        laps["role"] = role
        laps["known_issues"] = ",".join(gp["known_issues"]) if gp["known_issues"] else ""

        n_total = len(laps)
        n_usable = laps["usable_for_reward"].sum()
        n_anomaly = (laps.get("is_red_flag", False) | laps.get("is_safety_car", False)).sum() \
            if "is_red_flag" in laps.columns else 0
        print(f"    -> {n_total} tours | {n_usable} exploitables | {n_anomaly} tours sous anomalie piste")

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
    agg_spec = {
        "n_laps": ("lap_number", "count"),
        "n_usable": ("usable_for_reward", "sum"),
        "pct_raining": ("is_raining", "mean"),
    }
    if "is_red_flag" in df.columns:
        agg_spec["n_red_flag_laps"] = ("is_red_flag", "sum")
    if "is_safety_car" in df.columns:
        agg_spec["n_sc_laps"] = ("is_safety_car", "sum")

    print(df.groupby(["role", "event"]).agg(**agg_spec).round(3))

    print("\nValeurs manquantes par colonne :")
    print(df.isna().sum()[df.isna().sum() > 0])
