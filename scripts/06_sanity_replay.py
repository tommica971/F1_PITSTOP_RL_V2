#!/usr/bin/env python3
"""
Etape 5 — Test de sanite : rejeu de la strategie reelle  (V2, erreur de position)
==================================================================================

L'oracle (03_check_breakeven.py) verifie que l'environnement RECOMPENSE
correctement l'arret. Il ne verifie pas qu'il reste FIDELE a la realite : un
environnement peut classer les strategies dans le bon ordre tout en simulant
un Bahrein a 6200 s au lieu de 5600 s. Ce sont deux proprietes independantes,
et c'est celle-ci que le present script controle.

Principe : on rejoue dans F1PitStopEnv la strategie REELLE de Gasly (ses
arrets reels, aux tours reels, vers les composes reels), puis on compare le
resultat simule au resultat reel.

Nouveau en V2 : l'ERREUR DE POSITION devient le critere principal
-----------------------------------------------------------------
La V1 ne rapportait que l'erreur temporelle et considerait la position comme
un indicateur secondaire. C'etait une erreur de jugement : l'erreur de
position est precisement ce que la correction du biais de rythme doit reduire,
et elle est directement interpretable.

Mesure V1 (avant correction) : erreur temporelle mediane +0.95%, et un rejeu
de la strategie reelle qui classe Gasly P15 a Silverstone (P6 en realite),
P18 aux Pays-Bas (P4), P20 en Belgique (P10).

CRITERES D'ACCEPTATION DE LA CORRECTION (poses AVANT de corriger, pour ne pas
ajuster un parametre jusqu'a obtenir le resultat souhaite) :

    erreur de position mediane   <= 2 places
    erreur temporelle mediane    <= 0.30 %

REGLE D'ARRET : si ces tolerances ne sont pas atteintes apres correction, on
n'insiste pas. On conserve la comparaison interne aux baselines, qui reste
valide, et on documente la limite.

Usage, depuis la racine du projet :
    python scripts/06_sanity_replay.py
    python scripts/06_sanity_replay.py --json docs/sanity_v2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in ("src/f1_pitstop_rl/env", "src/f1_pitstop_rl/config"):
    sys.path.insert(0, str(ROOT / p))

try:
    from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND
    from gp_pool_config import GP_POOL, DRIVER_CODE
except ImportError as exc:
    sys.exit(f"Environnement non importable ({exc}).\n"
             "Lancer ce script depuis la racine du projet.")

N_SEEDS = 15
MAX_POS_ERROR = 2.0     # places, mediane
MAX_TIME_ERROR = 0.30   # %, mediane
COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}

features = pd.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")


def real_strategy(season, event):
    """Arrets reels de Gasly, temps de course reel et position reelle.

    Un arret exige un changement de relais ET une remise a zero de l'age du
    pneu : le seul changement de stint_number produisait des arrets fantomes
    (Australie 5 au lieu de 2, Canada 4 au lieu de 1) sur les courses
    pluvieuses.
    """
    g = features[(features["season"] == season) & (features["event"] == event)]
    if "driver" in g.columns:
        g = g[g["driver"] == DRIVER_CODE]
    g = g.sort_values("lap_number").reset_index(drop=True)
    if g.empty:
        return None, None, None

    pits = {}
    for i in range(1, len(g)):
        if (g.loc[i, "stint_number"] != g.loc[i - 1, "stint_number"]
                and g.loc[i, "tyre_age_laps"] < g.loc[i - 1, "tyre_age_laps"]):
            pits[int(g.loc[i, "lap_number"])] = str(g.loc[i, "tyre_compound"])

    laps = g[g["lap_number"] >= 2]
    med = laps["lap_time_s"].median()
    usable = laps[(laps["lap_time_s"] > 0) & (laps["lap_time_s"] < 3 * med)]
    return pits, float(usable["lap_time_s"].sum()), int(g.iloc[-1]["position"])


def replay(env, pits, seed):
    env.reset(seed=seed)
    done, info = False, {}
    while not done:
        compound = pits.get(env.current_lap)
        action = COMPOUND_TO_ACTION.get(compound, 0) if compound else 0
        _, _, done, _, info = env.step(action)
    return env.own_cum_time, info["position"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="Chemin d'export du rapport JSON")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    print("=" * 94)
    print("TEST DE SANITE — rejeu de la strategie reelle de Gasly")
    print("=" * 94)
    print(f"\n{args.n_seeds} graines par GP. Temps compare a partir du tour 2.")
    print(f"Criteres : erreur de position mediane <= {MAX_POS_ERROR:.0f} places, "
          f"erreur temporelle mediane <= {MAX_TIME_ERROR:.2f} %\n")
    print(f"{'GP':<28}{'simule (s)':>12}{'reel (s)':>11}{'err %':>8}"
          f"{'pos sim':>9}{'pos reel':>10}{'err pos':>9}")
    print("-" * 94)

    rows = []
    for gp in GP_POOL:
        pits, real_time, real_pos = real_strategy(gp["season"], gp["event"])
        if pits is None or not real_time:
            print(f"{gp['event'][:27]:<28}{'donnees indisponibles':>50}")
            continue
        try:
            env = F1PitStopEnv(fixed_gp=gp)
            runs = [replay(env, pits, s) for s in range(args.n_seeds)]
        except Exception as exc:                                # noqa: BLE001
            print(f"{gp['event'][:27]:<28}  echec : {exc}")
            continue

        sim_time = float(np.mean([r[0] for r in runs]))
        sim_pos = float(np.median([r[1] for r in runs]))
        pct = 100 * (sim_time - real_time) / real_time
        pos_err = sim_pos - real_pos
        label = f"{gp['season']} {gp['event']}".replace(" Grand Prix", "")
        print(f"{label[:27]:<28}{sim_time:>12.1f}{real_time:>11.1f}{pct:>+8.2f}"
              f"{sim_pos:>9.0f}{real_pos:>10}{pos_err:>+9.0f}")
        rows.append({"gp": label, "sim_time_s": sim_time, "real_time_s": real_time,
                     "time_error_pct": pct, "sim_position": sim_pos,
                     "real_position": real_pos, "position_error": pos_err})

    print("-" * 94)
    if not rows:
        sys.exit("\nAucun GP exploitable — verifier les chemins de donnees.")

    t_med = float(np.median([abs(r["time_error_pct"]) for r in rows]))
    t_max = float(np.max([abs(r["time_error_pct"]) for r in rows]))
    p_med = float(np.median([abs(r["position_error"]) for r in rows]))
    p_max = float(np.max([abs(r["position_error"]) for r in rows]))
    p_bias = float(np.median([r["position_error"] for r in rows]))

    print(f"\nErreur temporelle absolue : mediane {t_med:.2f} %  |  pire cas {t_max:.2f} %")
    print(f"Erreur de position absolue : mediane {p_med:.1f} places  |  pire cas {p_max:.0f}")
    print(f"Biais de position (signe)  : {p_bias:+.1f} place(s) — "
          f"{'agent systematiquement recule' if p_bias > 0 else 'agent systematiquement avance' if p_bias < 0 else 'pas de biais directionnel'}")

    ok_t, ok_p = t_med <= MAX_TIME_ERROR, p_med <= MAX_POS_ERROR
    print("\n" + "=" * 94)
    if ok_t and ok_p:
        print("VERDICT : CRITERES ATTEINTS. La comparaison au pilote reel redevient")
        print("une metrique valide. Le pivot methodologique de la V1 peut etre")
        print("reconsidere — ce qui implique de revoir le dashboard et le dossier.")
    else:
        manque = []
        if not ok_p:
            manque.append(f"position {p_med:.1f} > {MAX_POS_ERROR:.0f}")
        if not ok_t:
            manque.append(f"temps {t_med:.2f} % > {MAX_TIME_ERROR:.2f} %")
        print(f"VERDICT : CRITERES NON ATTEINTS ({', '.join(manque)}).")
        print("Regle d'arret : ne pas poursuivre l'ajustement. Conserver la")
        print("comparaison interne aux baselines, valide car les deux termes")
        print("subissent le meme biais, et documenter la limite.")
    print("=" * 94)

    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "n_seeds": args.n_seeds,
            "criteria": {"max_position_error": MAX_POS_ERROR,
                         "max_time_error_pct": MAX_TIME_ERROR},
            "summary": {"time_error_median_pct": t_med, "time_error_max_pct": t_max,
                        "position_error_median": p_med, "position_error_max": p_max,
                        "position_bias": p_bias, "criteria_met": bool(ok_t and ok_p)},
            "races": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
