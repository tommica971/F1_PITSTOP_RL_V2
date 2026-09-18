"""Patch du template du dashboard : GP de test, cumul, abstentions.

1. Jeu hors pool (lignes ~551 et ~600) : le template filtrait sur
   in_train_pool, ce qui comptait la Belgique 2025 (GP de TEST du pool) comme
   "jamais vue" -> cumul "sur 20 GP" alors que le KPI annonce 19.
   Il utilise desormais in_gp_pool (ajoute par apply_patch_oracle_pool.py),
   avec repli sur in_train_pool pour les anciens fichiers.
2. Badge (ligne ~620) : trois statuts, "entrainement", "test (pool)",
   "jamais vu".
3. Couleur de l'ecart au 0-stop (ligne ~622) : une abstention (ecart nul)
   s'affichait en rouge comme un echec. Elle s'affiche en gris.

Usage, depuis la racine du projet :
    python scripts/apply_patch_template_pool.py
"""
from pathlib import Path
import re
import shutil
import sys

TARGET = Path(__file__).resolve().parent.parent / "dashboard" / "dashboard_template.html"

# (motif regex, remplacement, nombre d'occurrences attendu)
PATCHES = [
    (
        r"d\.races\.filter\(r=>!r\.in_train_pool\)",
        "d.races.filter(r=>!(r.in_gp_pool ?? r.in_train_pool))",
        2,
    ),
    (
        r"const badge = r\.in_train_pool \? '<span class=\"badge\">entraînement</span>'\s*"
        r": '<span class=\"badge\">jamais vu</span>';",
        "const badge = r.in_train_pool ? '<span class=\"badge\">entraînement</span>'\n"
        "          : (r.in_gp_pool ? '<span class=\"badge\">test (pool)</span>'\n"
        "                          : '<span class=\"badge\">jamais vu</span>');",
        1,
    ),
    (
        r"col = dz>0 \? 'var\(--pos\)' : 'var\(--neg\)';",
        "col = Math.abs(dz) < 0.05 ? 'var(--muted, #8b95a7)'  /* abstention = 0-stop */\n"
        "          : dz>0 ? 'var(--pos)' : 'var(--neg)';",
        1,
    ),
]


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    if "in_gp_pool" in src:
        print("Deja patche, rien a faire.")
        return 0
    for pattern, repl, expected in PATCHES:
        n = len(re.findall(pattern, src))
        if n != expected:
            print(f"ECHEC : motif trouve {n} fois (attendu {expected}) :\n  {pattern[:90]}")
            print("Template NON modifie.")
            return 1
        src = re.sub(pattern, lambda _m: repl, src)
    shutil.copy2(TARGET, TARGET.with_suffix(".html.bak_pool"))
    TARGET.write_text(src, encoding="utf-8")
    print(f"Patche : {TARGET}\nSauvegarde : {TARGET.with_suffix('.html.bak_pool')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
