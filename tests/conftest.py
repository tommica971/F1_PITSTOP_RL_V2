"""Configuration pytest partagee.

Deux roles :

1. Rendre les modules du projet importables. Le projet n'est pas installe en
   package : chaque module est importe par son nom court (`f1_pitstop_env`,
   `gp_pool_config`), ce qui impose d'ajouter leurs dossiers au sys.path.

2. Ecarter automatiquement les tests qui dependent des fichiers Parquet.
   Ceux-ci ne sont pas versionnes (volumineux, regenerables par la chaine
   d'ingestion), donc absents en integration continue. Sans ce mecanisme, les
   25 tests de test_data_pipeline.py echouent sur un `FileNotFoundError` qui
   ne signale aucune regression reelle.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
for _sub in ("env", "config", "data", "training"):
    sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / _sub))

DATA_DIR = ROOT / "data"

# Fichiers sans lesquels les tests de pipeline n'ont rien a verifier.
REQUIRED_DATA = (
    DATA_DIR / "raw" / "master_dataset.parquet",
    DATA_DIR / "raw" / "all_drivers_dataset.parquet",
    DATA_DIR / "processed" / "features_dataset.parquet",
)


def data_available() -> bool:
    return all(p.exists() for p in REQUIRED_DATA)


def pytest_collection_modifyitems(config, items):
    """Marque et ecarte les tests de pipeline quand les Parquet sont absents.

    Tout test dont le fichier contient "data_pipeline", ou marque
    requires_data, est considere comme dependant des donnees.
    """
    if data_available():
        return
    skip = pytest.mark.skip(
        reason="fichiers Parquet absents — non versionnes, regenerables par "
               "la chaine d'ingestion (extract_* puis 01_clean_laps.py)")
    for item in items:
        if "data_pipeline" in str(item.fspath) or "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not data_available():
        pytest.skip("fichiers Parquet absents")
    return DATA_DIR
