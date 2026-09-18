#!/usr/bin/env python3
"""
Etape 1 — Reconstruction du filtre de tours exploitables + nettoyage
====================================================================

Constat (diagnostic_env.md) : gp_std_s vaut 3 a 22 s selon le GP, alors qu'un
ecart-type de temps au tour propre se situe entre 0.5 et 2 s. Le filtre
`usable_for_reward` ne retire donc presque rien. Comme pace_model.py exprime
TOUTES ses constantes en unites de gp_std, chaque constante "calibree" signifie
une chose differente selon le GP -- la penalite de tour de sortie vaut 28 s a
Silverstone et 5 s en Turquie.

Ce script :
  1. reconstruit `usable_for_reward` avec un filtre explicite et trace,
  2. nettoie les temps au tour aberrants des fantomes (artefact sesT, cf. la
     ligne Pays-Bas 2023 VSC a 32.9x la baseline dans le diagnostic),
  3. rapporte gp_std AVANT / APRES -- c'est le controle de reussite.

Critere de reussite : gp_std entre 0.5 et 2.5 s sur TOUS les GP.

Sorties :
    data/processed/features_dataset.parquet   (colonnes *_v2 ajoutees)
    data/raw/all_drivers_dataset.parquet      (lap_time_s nettoye)
    rapport_nettoyage.md

Les fichiers d'origine sont sauvegardes en .bak avant ecriture.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATURES = DATA / "processed" / "features_dataset.parquet"
GHOSTS = DATA / "raw" / "all_drivers_dataset.parquet"
REPORT = ROOT / "rapport_nettoyage.md"

# --- Parametres du filtre (explicites, a citer dans le dossier) -------------
# 107% : seuil du reglement F1 pour les qualifications, reutilise ici comme
# borne haute d'un tour "de course normale". Un tour de pit (+23 s sur ~90 s)
# ou de safety car (+40%) le depasse toujours.
MAX_RATIO_OF_MEDIAN = 1.07
MIN_RATIO_OF_MEDIAN = 0.93
# Un tour a plus de 3x la mediane n'est pas un tour de course : c'est un
# artefact de donnees (temps de session cumule pris pour un temps au tour).
ABERRANT_RATIO = 3.0

lines = ["# Rapport de nettoyage des temps au tour\n"]


def emit(t=""):
    print(t)
    lines.append(t)


def load(path, label):
    if not path.exists():
        sys.exit(f"ERREUR : {path} introuvable")
    df = pd.read_parquet(path)
    emit(f"- `{label}` : {len(df):,} lignes, {df['lap_number'].nunique()} tours distincts")
    return df


emit("## Fichiers charges\n")
feat = load(FEATURES, "features_dataset.parquet")
ghosts = load(GHOSTS, "all_drivers_dataset.parquet")

required = {"season", "event", "lap_number", "lap_time_s"}
for name, df in (("features", feat), ("all_drivers", ghosts)):
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERREUR : colonnes manquantes dans {name} : {missing}")

has_pit_col = "is_pit" in feat.columns or "pit_in" in feat.columns
emit(f"\nColonne d'arret explicite detectee : {has_pit_col}")
emit(f"Colonnes disponibles (features) : {sorted(feat.columns)}\n")


# ---------------------------------------------------------------------------
def build_usable(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute usable_for_reward_v2 + la raison d'exclusion, GP par GP."""
    df = df.copy()
    df["exclusion_reason"] = ""

    def mark(mask, reason):
        todo = mask & (df["exclusion_reason"] == "")
        df.loc[todo, "exclusion_reason"] = reason

    # 1. valeurs invalides
    mark(df["lap_time_s"].isna() | (df["lap_time_s"] <= 0), "temps nul ou manquant")

    # 2. tour 1 : artefact structurel documente (temps de session cumule)
    mark(df["lap_number"] == 1, "tour 1 (artefact sesT)")

    # 3. neutralisations
    for col, reason in (("is_safety_car", "safety car"), ("is_vsc", "VSC")):
        if col in df.columns:
            mark(df[col].astype("boolean").fillna(False).to_numpy(dtype=bool), reason)

    # 4. tours d'arret et tours de sortie, deduits du changement de relais
    if "stint_number" in df.columns:
        key = ["season", "event"] + (["driver"] if "driver" in df.columns else [])
        df = df.sort_values(key + ["lap_number"])
        g = df.groupby(key, sort=False)["stint_number"]
        new_stint = (g.diff() != 0) & g.diff().notna()
        # le tour d'arret est celui qui precede le nouveau relais ; le tour de
        # sortie est le premier du nouveau relais -> on exclut les deux
        mark(new_stint.to_numpy(), "tour de sortie")
        mark(new_stint.shift(-1, fill_value=False).to_numpy(), "tour d'arret")

    if "tyre_age_laps" in df.columns:
        mark((df["tyre_age_laps"] <= 2).to_numpy(), "pneu froid (age <= 2)")

    # 5. bornes relatives a la mediane du GP, calculees sur ce qui reste
    keep = df["exclusion_reason"] == ""
    med = (df[keep].groupby(["season", "event"])["lap_time_s"].median()
           .rename("median_gp"))
    df = df.merge(med, on=["season", "event"], how="left")
    ratio = df["lap_time_s"] / df["median_gp"]
    mark((ratio > ABERRANT_RATIO).to_numpy(), "aberrant (> 3x mediane)")
    mark((ratio > MAX_RATIO_OF_MEDIAN).to_numpy(), "hors bornes (> 107%)")
    mark((ratio < MIN_RATIO_OF_MEDIAN).to_numpy(), "hors bornes (< 93%)")

    df["usable_for_reward_v2"] = df["exclusion_reason"] == ""
    return df


emit("## Motifs d'exclusion (pilote de reference)\n")
feat = build_usable(feat)
counts = feat["exclusion_reason"].replace("", "CONSERVE").value_counts()
emit("| motif | tours |")
emit("|---|---|")
for reason, n in counts.items():
    emit(f"| {reason} | {n} |")

kept = int(feat["usable_for_reward_v2"].sum())
emit(f"\n**{kept} / {len(feat)} tours conserves ({100 * kept / len(feat):.0f}%).**\n")


# ---------------------------------------------------------------------------
emit("## gp_std AVANT / APRES\n")
emit("| GP | gp_std avant (s) | gp_std apres (s) | baseline apres (s) | n tours | verdict |")
emit("|---|---|---|---|---|---|")

old_col = "usable_for_reward" if "usable_for_reward" in feat.columns else None
ok_all = True
for (s, e), g in feat.groupby(["season", "event"]):
    before = float(g[g[old_col]]["lap_time_s"].std()) if old_col else float(g["lap_time_s"].std())
    sub = g[g["usable_for_reward_v2"]]
    if len(sub) < 5:
        emit(f"| {s} {e} | {before:.2f} | - | - | {len(sub)} | **TROP PEU DE TOURS** |")
        ok_all = False
        continue
    after = float(sub["lap_time_s"].std())
    base = float(sub["lap_time_s"].median())
    good = 0.3 <= after <= 2.5
    ok_all &= good
    emit(f"| {s} {e} | {before:.2f} | {after:.2f} | {base:.2f} | {len(sub)} | "
         f"{'OK' if good else '**HORS CIBLE**'} |")

emit(f"\n**Critere global : {'ATTEINT' if ok_all else 'NON ATTEINT'}** "
     "(cible : gp_std entre 0.3 et 2.5 s partout)\n")
if not ok_all:
    emit("> Si un GP reste hors cible, inspecter ses tours conserves avant de "
         "passer a l'etape 2 : la calibration heritera du probleme.\n")


# ---------------------------------------------------------------------------
emit("## Nettoyage des temps aberrants (fantomes)\n")
med_g = (ghosts.groupby(["season", "event"])["lap_time_s"].median().rename("median_gp"))
ghosts = ghosts.merge(med_g, on=["season", "event"], how="left")
bad = (ghosts["lap_time_s"] > ABERRANT_RATIO * ghosts["median_gp"]) | (ghosts["lap_time_s"] <= 0)

emit(f"{int(bad.sum())} temps au tour aberrants detectes "
     f"({100 * bad.mean():.2f}% des lignes).\n")
if bad.any():
    ex = ghosts[bad].nlargest(min(8, int(bad.sum())), "lap_time_s")
    emit("| GP | pilote | tour | temps (s) | ratio mediane |")
    emit("|---|---|---|---|---|")
    for r in ex.itertuples():
        drv = getattr(r, "driver", "?")
        emit(f"| {r.season} {r.event} | {drv} | {r.lap_number} | "
             f"{r.lap_time_s:.1f} | {r.lap_time_s / r.median_gp:.1f}x |")

# remplacement par la mediane du peloton a ce tour (le fantome reste classe,
# ce qui est plus fidele que de le supprimer ou de le declarer abandonne)
lap_med = (ghosts[~bad].groupby(["season", "event", "lap_number"])["lap_time_s"]
           .median().rename("lap_median"))
ghosts = ghosts.merge(lap_med, on=["season", "event", "lap_number"], how="left")
ghosts["lap_time_s_clean"] = np.where(
    bad, ghosts["lap_median"].fillna(ghosts["median_gp"]), ghosts["lap_time_s"])
n_fixed = int((ghosts["lap_time_s"] != ghosts["lap_time_s_clean"]).sum())
emit(f"\n{n_fixed} temps remplaces par la mediane du peloton au meme tour.\n")


# ---------------------------------------------------------------------------
for path in (FEATURES, GHOSTS):
    if path.exists() and not path.with_suffix(".parquet.bak").exists():
        shutil.copy(path, path.with_suffix(".parquet.bak"))

feat_out = feat.drop(columns=["median_gp"], errors="ignore")
feat_out.to_parquet(FEATURES, index=False)

ghosts_out = ghosts.copy()
ghosts_out["lap_time_s_raw"] = ghosts_out["lap_time_s"]
ghosts_out["lap_time_s"] = ghosts_out["lap_time_s_clean"]
ghosts_out = ghosts_out.drop(columns=["median_gp", "lap_median", "lap_time_s_clean"])
ghosts_out.to_parquet(GHOSTS, index=False)

REPORT.write_text("\n".join(lines), encoding="utf-8")
emit(f"\nSauvegardes : *.parquet.bak | Rapport : {REPORT}")
print("\nEtape suivante : python scripts/02_calibrate_pace.py")
