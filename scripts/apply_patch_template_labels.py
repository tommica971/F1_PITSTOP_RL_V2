"""Patch du template : panneau Optuna explicite, codes de course.

1. Panneau Optuna. Il affichait les meilleurs hyperparametres de l'etude a
   cote du modele presente, qui ne les utilise PAS (reglages par defaut).
   Un lecteur en deduisait que le modele presente etait le modele optimise.
   Titre et mention explicites : etude testee, non retenue, avec l'ecart mesure.
2. Codes de course. Le graphique tronquait le nom a 3 lettres : Australian et
   Austrian donnaient tous deux "AUS". Codes explicites (AUS / AUT, GBR, NED...).

Usage, depuis la racine du projet :
    python scripts/apply_patch_template_labels.py
"""
from pathlib import Path
import shutil
import sys

TARGET = Path(__file__).resolve().parent.parent / "dashboard" / "dashboard_template.html"

GP_CODES = '''function gpCode(name){
  const codes = {"Abu Dhabi":"ABU","Australian":"AUS","Austrian":"AUT","Azerbaijan":"AZE",
    "Bahrain":"BHR","Belgian":"BEL","British":"GBR","Canadian":"CAN","Chinese":"CHN",
    "Dutch":"NED","Emilia Romagna":"EMI","Hungarian":"HUN","Italian":"ITA","Japanese":"JPN",
    "Las Vegas":"LVG","Mexico City":"MEX","Miami":"MIA","Monaco":"MON","Qatar":"QAT",
    "Singapore":"SGP","Spanish":"ESP","São Paulo":"BRA","United States":"USA"};
  const key = name.replace(" Grand Prix","");
  return codes[key] || key.slice(0,3).toUpperCase();
}
'''

OPTUNA_NOTE = (
    '`<div style="color:var(--text-dim);margin-bottom:10px;line-height:1.6;">'
    'Hyperparamètres <b>testés et non retenus</b>. Le modèle présenté '
    '(${modelData.main_model}) utilise les réglages par défaut : '
    'récompense moyenne hors pool −38,8 ± 6,4 contre −46,6 ± 7,1 avec ces réglages '
    '(3 graines chacun), bien que l’étude ait été conduite sur ces mêmes GP. '
    'Étude : ${modelData.optuna.trials.length} essais complétés'
    '${modelData.optuna.n_pruned!=null ? ", "+modelData.optuna.n_pruned+" élagués" : ""}.'
    '</div>` + '
)

PATCHES = [
    ('<p class="panel-title">Optimisation Optuna — meilleurs hyperparamètres</p>',
     '<p class="panel-title">Étude Optuna V2 — réglages testés, non retenus</p>', 1),
    ("    optunaEl.innerHTML = `<div style=\"color:var(--text-dim);margin-bottom:6px;\">Meilleure valeur",
     "    optunaEl.innerHTML = " + OPTUNA_NOTE +
     "`<div style=\"color:var(--text-dim);margin-bottom:6px;\">Meilleure valeur", 1),
    ("r.gp_name.slice(0,3).toUpperCase()", "gpCode(r.gp_name)", 2),
    ("document.getElementById('teamName').textContent",
     GP_CODES + "document.getElementById('teamName').textContent", 1),
]


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    if "function gpCode" in src:
        print("Deja patche, rien a faire.")
        return 0
    for old, new, expected in PATCHES:
        n = src.count(old)
        if n != expected:
            print(f"ECHEC : motif trouve {n} fois (attendu {expected}) :\n  {old[:90]}")
            print("Template NON modifie.")
            return 1
        src = src.replace(old, new)
    shutil.copy2(TARGET, TARGET.with_suffix(".html.bak_labels"))
    TARGET.write_text(src, encoding="utf-8")
    print(f"Patche : {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
