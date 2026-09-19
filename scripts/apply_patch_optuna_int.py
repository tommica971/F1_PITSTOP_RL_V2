"""Panneau Optuna du dashboard : les hyperparametres entiers (n_steps...)
s'affichaient avec toPrecision(4), soit "32.00". Les entiers sont desormais
affiches tels quels ; les reels gardent 4 chiffres significatifs.
Usage : python scripts/apply_patch_optuna_int.py [racine_du_depot]
"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
f = root / "dashboard" / "dashboard_template.html"
src = f.read_text(encoding="utf-8")

old = "typeof v==='number' ? v.toPrecision(4) : v"
new = "typeof v==='number' ? (Number.isInteger(v) ? v : v.toPrecision(4)) : v"

if new in src:
    sys.exit(f"Deja patche : {f}")
if src.count(old) != 1:
    sys.exit(f"Motif introuvable (ou multiple) : {f}")
shutil.copy2(f, f.with_name(f.name + ".bak_optuna_int"))
f.write_text(src.replace(old, new), encoding="utf-8")
print(f"Patche : {f}  (sauvegarde .bak_optuna_int)")