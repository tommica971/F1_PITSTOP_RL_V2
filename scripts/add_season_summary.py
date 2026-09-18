"""Complete season_2025.json avec les champs attendus par le dashboard.

Probleme
--------
run_season_inference_v2.py produit season, model, algo, deterministic,
excluded_gp, races. Le template lit aussi seasonData.driver (en-tete) et
seasonData.summary.* (KPI de l'onglet Sim-to-Real). L'absence de summary
faisait planter le script de la page (TypeError ligne 791), et tout ce qui
s'affiche ensuite restait vide : KPI de points, graphique, onglet
Apprentissage RL.

Ce script ajoute :
- driver : "Pierre Gasly"
- summary : points reels et agent sur la saison, les memes restreints aux GP
  hors pool, et le decompte des courses ameliorees / degradees / egales.

"Hors pool" = absent de GP_POOL (entrainement ET test), par (saison, course) :
meme definition que eval_generalization.py et build_comparison_data.py
(19 GP ; la Belgique 2025, GP de test, n'y est pas).

Rappel : ces points ne mesurent pas l'agent (biais de rythme de +0,95 %,
5 a 9 positions). Ils alimentent l'onglet Sim-to-Real, qui l'affiche.

Usage, depuis la racine du projet, APRES run_season_inference_v2.py :
    python scripts/add_season_summary.py
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "config"))
from gp_pool_config import GP_POOL  # noqa: E402

SEASON_PATH = ROOT / "dashboard" / "data" / "season_2025.json"
DRIVER = "Pierre Gasly"


def points_key(d: dict, side: str) -> str:
    keys = [k for k in d if "point" in k.lower() and isinstance(d[k], (int, float))]
    if len(keys) != 1:
        sys.exit(f"Cle de points introuvable ou ambigue dans race['{side}'] : {list(d)}")
    return keys[0]


def main() -> int:
    data = json.loads(SEASON_PATH.read_text(encoding="utf-8"))
    races = data["races"]
    season = data.get("season", 2025)
    pool = {(g["season"], g["event"]) for g in GP_POOL}

    k_real = points_key(races[0]["real"], "real")
    k_agent = points_key(races[0]["agent"], "agent")

    def pts(r):
        return r["real"][k_real], r["agent"][k_agent]

    def delta(r):
        rp, ap = pts(r)
        return r.get("delta_points", ap - rp)

    unseen = [r for r in races if (season, r["gp_name"]) not in pool]
    for r in races:
        r["in_gp_pool"] = (season, r["gp_name"]) in pool

    data["driver"] = DRIVER
    data["summary"] = {
        "real_total_points": sum(pts(r)[0] for r in races),
        "agent_total_points": sum(pts(r)[1] for r in races),
        "unseen_races_only": {
            "n_races": len(unseen),
            "real_total_points": sum(pts(r)[0] for r in unseen),
            "agent_total_points": sum(pts(r)[1] for r in unseen),
        },
        "races_improved": sum(1 for r in races if delta(r) > 0),
        "races_worse": sum(1 for r in races if delta(r) < 0),
        "races_equal": sum(1 for r in races if delta(r) == 0),
        "note": "Comparaison au pilote reel NON valide (biais de rythme +0,95 %, "
                "5 a 9 positions) : affichee dans l'onglet Sim-to-Real a titre "
                "de diagnostic, pas comme mesure de l'agent.",
    }
    SEASON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    s = data["summary"]
    print(f"Cles de points : real['{k_real}'], agent['{k_agent}']")
    print(f"Saison : reel {s['real_total_points']} pts | agent {s['agent_total_points']} pts "
          f"({len(races)} GP)")
    u = s["unseen_races_only"]
    print(f"Hors pool : {u['n_races']} GP | reel {u['real_total_points']} | agent {u['agent_total_points']}")
    print(f"Ameliorees {s['races_improved']} | degradees {s['races_worse']} | egales {s['races_equal']}")
    print(f"-> {SEASON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
