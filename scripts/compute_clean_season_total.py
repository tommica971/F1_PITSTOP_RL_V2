"""
F1_PITSTOP_RL/scripts/compute_clean_season_total.py

Prend le rapport produit par validate_robustness.py (--all-dashboard-gps) et le
croise avec season_2025.json pour produire trois totaux de points, au lieu
d'un seul chiffre brut potentiellement gonfle par l'artefact de transition
SC/VSC (cf. analyse Miami/Sao Paulo -- delta pace-model injustifie au sortir
d'un tour sous Safety Car/VSC).

Classification par GP (deterministe = reference reproductible, PAS le tirage
stochastique affiche au dashboard qui peut varier d'un run a l'autre) :
    - "propre"  : aucun saut de position suspect en deterministe ET
                  ecart-type stochastique < VARIANCE_THRESHOLD
    - "suspect" : au moins un saut detecte OU variance elevee

Trois totaux calcules a partir des points DETERMINISTES (reproductibles) :
    1. Total brut       : somme sur tous les GP traites
    2. Total "propre"   : somme sur les GP sans aucun signal d'alerte
    3. Total "suspect"  : somme sur les GP avec saut ou variance elevee
    (1 = 2 + 3, par construction)

Usage :
    cd F1_PITSTOP_RL/scripts
    python compute_clean_season_total.py
    python compute_clean_season_total.py --report validation_report.json \
        --season-json ../dashboard/data/season_2025.json \
        --out clean_total_report.json --out-md clean_total_report.md
"""

import argparse
import json
from pathlib import Path

VARIANCE_THRESHOLD = 3.0  # ecart-type de position stochastique, au-dela = peu fiable a l'echelle d'un seul run


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # Fichiers generes avant le correctif d'encodage (cf. run_season_inference.py)
        return json.loads(path.read_text(encoding="cp1252"))


def classify(entry: dict) -> str:
    has_jumps = len(entry["deterministic"]["suspicious_jumps"]) > 0
    high_variance = entry["stochastic"]["std_position"] >= VARIANCE_THRESHOLD
    return "suspect" if (has_jumps or high_variance) else "propre"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=str, default="validation_report.json")
    parser.add_argument("--season-json", type=str, default="../dashboard/data/season_2025.json")
    parser.add_argument("--out", type=str, default="clean_total_report.json")
    parser.add_argument("--out-md", type=str, default="clean_total_report.md")
    args = parser.parse_args()

    report = load_json(Path(args.report))
    season = load_json(Path(args.season_json))

    real_points_by_gp = {r["gp_name"]: r["real"]["points"] for r in season["races"]}
    seen_by_gp = {r["gp_name"]: r["seen_in_training"] for r in season["races"]}

    rows = []
    for entry in report:
        name = entry["gp_name"]
        status = classify(entry)
        rows.append({
            "gp_name": name,
            "seen_in_training": seen_by_gp.get(name),
            "classification": status,
            "real_points": real_points_by_gp.get(name),
            "agent_points_deterministic": entry["deterministic"]["points"],
            "agent_position_deterministic": entry["deterministic"]["position"],
            "n_suspicious_jumps": len(entry["deterministic"]["suspicious_jumps"]),
            "stochastic_std_position": entry["stochastic"]["std_position"],
        })

    total_real = sum(r["real_points"] for r in rows if r["real_points"] is not None)
    total_brut = sum(r["agent_points_deterministic"] for r in rows)
    total_propre = sum(r["agent_points_deterministic"] for r in rows if r["classification"] == "propre")
    total_suspect = sum(r["agent_points_deterministic"] for r in rows if r["classification"] == "suspect")
    n_propre = sum(1 for r in rows if r["classification"] == "propre")
    n_suspect = sum(1 for r in rows if r["classification"] == "suspect")

    print(f"{'GP':<28} {'Statut':<14} {'Classe':<9} {'Réel':>5} {'Agent':>6} {'Sauts':>6} {'Std':>6}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: -x["agent_points_deterministic"]):
        print(f"{r['gp_name']:<28} "
              f"{'Entraînement' if r['seen_in_training'] else 'Généralisation':<14} "
              f"{r['classification']:<9} "
              f"{r['real_points']:>5} "
              f"{r['agent_points_deterministic']:>6} "
              f"{r['n_suspicious_jumps']:>6} "
              f"{r['stochastic_std_position']:>6.1f}")

    print("\n" + "=" * 80)
    print(f"Total réel (saison)                          : {total_real} pts")
    print(f"Total agent BRUT (déterministe, {len(rows)} GP)          : {total_brut} pts")
    print(f"  dont \"propre\" ({n_propre} GP, aucun signal d'alerte) : {total_propre} pts")
    print(f"  dont \"suspect\" ({n_suspect} GP, saut ou variance)    : {total_suspect} pts")

    output = {
        "rows": rows,
        "totals": {
            "real": total_real,
            "agent_gross": total_brut,
            "agent_clean": total_propre,
            "agent_suspect": total_suspect,
            "n_gp_clean": n_propre,
            "n_gp_suspect": n_suspect,
        },
    }
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport JSON : {args.out}")

    md_lines = [
        "# Décomposition du gain de points — saison 2025",
        "",
        "Trois totaux, calculés à partir des résultats **déterministes** (politique la "
        "plus probable, reproductible — pas un tirage stochastique isolé) :",
        "",
        f"- **Total réel de Gasly (saison)** : {total_real} pts",
        f"- **Total agent, brut** ({len(rows)} GP traités) : {total_brut} pts",
        f"- **Total agent, sous-ensemble \"propre\"** ({n_propre} GP sans saut de position détecté "
        f"ni variance élevée entre runs) : {total_propre} pts",
        f"- **Total agent, sous-ensemble \"suspect\"** ({n_suspect} GP avec au moins un saut "
        f"de position ≥5 places en un tour, ou un écart-type ≥{VARIANCE_THRESHOLD:.0f} positions "
        "sur 5 tirages stochastiques) : " + f"{total_suspect} pts",
        "",
        "Le sous-ensemble \"suspect\" corrèle systématiquement avec les tours suivant une "
        "période de Safety Car/VSC (cf. analyse Miami/São Paulo/Monaco) : le modèle de "
        "rythme calibré en Phase 2.2 traite la sortie de SC/VSC comme un retour instantané "
        "au rythme normal, alors qu'en réalité le peloton reste resserré 1 à 2 tours de plus. "
        "Le sous-total \"propre\" est la mesure la plus défendable de la capacité stratégique "
        "green-flag de l'agent.",
        "",
        "## Détail par Grand Prix",
        "",
        "| GP | Statut | Classe | Réel | Agent (déterministe) | Sauts détectés | Écart-type stochastique |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["classification"], -x["agent_points_deterministic"])):
        md_lines.append(
            f"| {r['gp_name']} | {'Entraînement' if r['seen_in_training'] else 'Généralisation'} "
            f"| {r['classification']} | {r['real_points']} | {r['agent_points_deterministic']} "
            f"| {r['n_suspicious_jumps']} | {r['stochastic_std_position']:.1f} |"
        )
    Path(args.out_md).write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Rapport Markdown (pretes a coller dans le dossier) : {args.out_md}")


if __name__ == "__main__":
    main()
