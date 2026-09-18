"""
F1_PITSTOP_RL/dashboard/build_dashboard.py  (v2)

Embarque les JSON de dashboard/data/ directement dans le HTML final -- pas de
fetch() runtime, donc aucune dependance a un serveur local ni risque CORS en
double-clic. C'est le choix le plus fiable pour une soutenance.

Nouveau en v2 : injecte aussi comparison_*.json, produit par
scripts/build_comparison_data.py, qui alimente l'onglet "Strategie" (vue
d'accueil). Voir apply_patch_dashboard.py pour la raison : le comparatif
agent vs pilote reel n'est pas une metrique valide, un biais de +0.95% sur le
temps de course deplacant l'agent de 5 a 9 positions quelle que soit sa
strategie.

season_2025.json et les race_*.json restent embarques : ils alimentent le
Command Center et l'onglet Sim-to-Real, qui gardent leur utilite (visualisation
tour par tour, analyse de l'ecart simulation/realite).

Usage :
    cd F1_PITSTOP_RL/dashboard
    python build_dashboard.py
    python build_dashboard.py --comparison data/comparison_a2c_v4_s1.json
"""

import argparse
import json
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
DATA_DIR = DASHBOARD_DIR / "data"
TEMPLATE_PATH = DASHBOARD_DIR / "dashboard_template.html"
OUTPUT_PATH = DASHBOARD_DIR / "dashboard.html"


def load(path: Path, label: str, required: bool = False):
    if not path.exists():
        msg = f"[!] {path.name} introuvable -- {label}"
        if required:
            raise SystemExit(msg)
        print(msg)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", default=None,
                    help="Chemin du comparison_*.json (defaut : le plus recent)")
    ap.add_argument("--multi-seed", default=None,
                    help="Resultat agrege sur plusieurs graines d'entrainement, "
                         "affiche sous le KPI principal. Exemple : "
                         "--multi-seed \"88 %% +/- 8 (~17/19)\". A prendre dans la "
                         "sortie de scripts/aggregate_seeds.py — le KPI principal "
                         "reste celui de la graine affichee, cette mention evite "
                         "de presenter la meilleure graine comme le resultat.")
    args = ap.parse_args()

    season = load(DATA_DIR / "season_2025.json",
                  "lance scripts/run_season_inference_v2.py d'abord", required=True)

    races = {}
    for p in sorted(DATA_DIR.glob("race_*_2025.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        races[r["gp_name"]] = r
    missing = [r["gp_name"] for r in season["races"] if r["gp_name"] not in races]
    if missing:
        print(f"[!] {len(missing)} GP dans season_2025.json sans fichier race_*.json : {missing}")

    if args.comparison:
        comp_path = Path(args.comparison)
        if not comp_path.is_absolute():
            comp_path = DASHBOARD_DIR / comp_path
    else:
        found = sorted(DATA_DIR.glob("comparison_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        comp_path = found[0] if found else DATA_DIR / "comparison_absent.json"
        if found and len(found) > 1:
            print(f"[i] {len(found)} fichiers comparison_*.json — le plus recent retenu "
                  f"({comp_path.name}). --comparison pour en choisir un autre.")
    comparison = load(comp_path, "onglet Strategie vide "
                      "(lance scripts/build_comparison_data.py)")

    if comparison and args.multi_seed:
        comparison["multi_seed"] = args.multi_seed

    # circuit_reference.json n'est plus lu : seul renderCircuitTab l'utilisait,
    # et son onglet a ete retire (donnees jamais verifiees, cf. passe 1).
    circuit = None
    model = load(DATA_DIR / "model_training.json",
                 "onglet Modele incomplet (lance extract_model_data.py)")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if "/*__EMBEDDED_DATA__*/" not in template:
        raise SystemExit("Marqueur /*__EMBEDDED_DATA__*/ absent du template.")

    def js(name, obj):
        return f"const {name} = " + json.dumps(obj, ensure_ascii=False) + ";\n"

    injected = (js("seasonData", season) + js("raceCache", races)
                + js("circuitRef", circuit) + js("modelData", model)
                + js("comparisonData", comparison))
    OUTPUT_PATH.write_text(
        template.replace("/*__EMBEDDED_DATA__*/", injected), encoding="utf-8")

    print(f"\nOK -> {OUTPUT_PATH}")
    print(f"   {len(races)} GP embarques")
    if comparison:
        s = comparison["summary_out_of_pool"]
        print(f"   Strategie : {comparison['model']} — bat le 0-stop "
              f"{s['beats_zero_stop']}/{s['n_gp']} ({s['pct_beats_zero_stop']}%), "
              f"atteint l'oracle {s['beats_oracle']}/{s['n_gp']}")
    else:
        print("   Onglet Strategie VIDE — le dashboard s'ouvrira sans sa vue d'accueil.")


if __name__ == "__main__":
    main()
