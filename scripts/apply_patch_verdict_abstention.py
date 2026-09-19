"""Verdict console de build_comparison_data.py : une abstention n'est pas
"sous le 0-stop". Quand l'agent ne s'arrete sur aucune graine, il joue la
course du 0-stop et obtient exactement son score (delta = 0). Libelle seul :
les compteurs utilisent delta_vs_zero > 0 et ne changent pas.
Usage : python scripts/apply_patch_verdict_abstention.py [racine_du_depot]
"""
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
f = root / "scripts" / "build_comparison_data.py"
src = f.read_text(encoding="utf-8")

old = ('        verdict = ("bat l\'oracle" if a_mean > oracle\n'
       '                   else "bat le 0-stop" if a_mean > zero else "sous le 0-stop")\n')
new = ('        if abs(a_mean - zero) < 1e-9:\n'
       '            verdict = "abstention"      # = score du 0-stop\n'
       '        elif a_mean > oracle:\n'
       '            verdict = "bat l\'oracle"\n'
       '        elif a_mean > zero:\n'
       '            verdict = "bat le 0-stop"\n'
       '        else:\n'
       '            verdict = "sous le 0-stop"\n')

if 'verdict = "abstention"' in src:
    sys.exit(f"Deja patche : {f}")
if src.count(old) != 1:
    sys.exit(f"Motif introuvable (ou multiple) : {f}")
shutil.copy2(f, f.with_name(f.name + ".bak_verdict"))
f.write_text(src.replace(old, new), encoding="utf-8")
print(f"Patche : {f}  (sauvegarde .bak_verdict)")
