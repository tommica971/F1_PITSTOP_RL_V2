"""Configuration pytest partagée -- ajoute les modules du projet au sys.path."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "env"))
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "config"))
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "data"))

DATA_DIR = ROOT / "data"
