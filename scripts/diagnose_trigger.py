#!/usr/bin/env python3
"""
Qu'est-ce qui declenche l'arret, et pourquoi certains GP ne le franchissent jamais ?
====================================================================================

Constat. Sur les 19 GP hors pool, A2C a un comportement BINAIRE : 1.0 arret
choisi ou 0.0, presque sans variance entre graines.

    quand il s'arrete (10 GP)  : a +/- 6 points du meilleur script
    quand il ne s'arrete pas (9 GP) : 60 a 70 points de retard

Et diagnose_policy.py montre un basculement franc, pas une hesitation :
Bahrein passe de P(pit)=4.0e-04 au tour 22 a 0.977 au tour 26. La Belgique
plafonne a 4.1e-04 au tour 28 -- exactement le niveau de Bahrein juste avant
son basculement -- puis redescend. L'agent approche du seuil sans le franchir.

Le probleme n'est donc ni l'exploration ni le budget. C'est : QUELLE REGION
DE L'OBSERVATION declenche l'arret, et pourquoi certains GP n'y entrent
jamais ?

Ce que le script fait
----------------------
  1. rejoue chaque GP en deterministe, en enregistrant a chaque tour
     l'observation complete et P(pit)
  2. identifie la "region de declenchement" : l'observation moyenne au tour
     ou P(pit) bascule, sur les GP qui s'arretent
  3. pour chaque GP qui ne s'arrete pas, mesure l'ecart a cette region,
     dimension par dimension, et nomme la dimension la plus eloignee
  4. teste explicitement l'hypothese du COMPOSE DE DEPART

Sortie : tableaux console + trigger_<modele>.json

Usage, depuis la racine du projet :
    python scripts/diagnose_trigger.py --model models/a2c/a2c_v4_500k.zip
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

from f1_pitstop_env import F1PitStopEnv  # noqa: E402
from gp_pool_config import GP_POOL  # noqa: E402

ALGOS = {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"}
INFERENCE_ROLE = "season_2025_inference"
MIN_LAPS = 25
TRIGGER = 0.5   # P(pit) au-dela de laquelle l'agent bascule

OBS_NAMES = ["age_pneu", "tours_restants", "position", "gap_avant",
             "gap_arriere", "track_temp", "delta_pluie", "is_raining",
             "rival_av_pit", "rival_ar_pit", "n_composes",
             "c_SOFT", "c_MEDIUM", "c_HARD", "c_INTER", "c_WET"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def trace(model, gp):
    """Rejoue un GP, renvoie (observations, P(pit) par tour, compose de depart)."""
    env = F1PitStopEnv(fixed_gp=gp, seed=0)
    obs, _ = env.reset(seed=0)
    start_compound = env.own_tyre_compound
    obs_list, p_list, done = [], [], False
    while not done:
        obs_list.append(np.asarray(obs, dtype=float).copy())
        try:
            t, _ = model.policy.obs_to_tensor(obs)
            probs = model.policy.get_distribution(t).distribution.probs
            p_list.append(float(probs.detach().cpu().numpy().ravel()[1:].sum()))
        except AttributeError:
            p_list.append(float("nan"))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
    return np.array(obs_list), np.array(p_list), start_compound


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="a2c", choices=list(ALGOS))
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"Modele introuvable : {path}")
    import stable_baselines3 as sb3
    model = getattr(sb3, ALGOS[args.algo]).load(str(path), device="cpu")

    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")
    pool = {(g["season"], g["event"]) for g in GP_POOL}
    counts = feat[feat["season"] == args.season].groupby("event")["lap_number"].max()

    gps = [{"season": args.season, "event": e, "role": INFERENCE_ROLE,
            "known_issues": []}
           for e, n in counts.items()
           if (args.season, e) not in pool and n >= MIN_LAPS]
    gps += [g for g in GP_POOL]   # le pool aussi, pour la region de reference

    print(f"Modele : {path.name}  |  {len(gps)} GP\n")

    records = []
    for gp in gps:
        try:
            obs, p, start = trace(model, gp)
        except Exception as exc:                            # noqa: BLE001
            print(f"  {gp['event'][:28]:<30} echec : {exc}")
            continue
        fired = bool((p >= TRIGGER).any())
        idx = int(np.argmax(p))
        records.append({
            "event": gp["event"], "season": gp["season"],
            "in_pool": (gp["season"], gp["event"]) in pool,
            "start_compound": start, "fired": fired,
            "p_max": float(p[idx]), "lap_of_max": idx + 2,
            "obs_at_max": obs[idx].tolist(),
        })

    fired = [r for r in records if r["fired"]]
    missed = [r for r in records if not r["fired"]]
    if not fired:
        sys.exit("Aucun GP ne declenche l'arret — rien a comparer.")

    # --- 1. hypothese du compose de depart ---------------------------------
    print("=" * 84)
    print("1. HYPOTHESE : LE COMPOSE DE DEPART")
    print("=" * 84)
    print(f"{'compose de depart':<20}{'declenche':>12}{'ne declenche pas':>20}{'taux':>10}")
    print("-" * 84)
    for c in COMPOUNDS:
        f = sum(1 for r in fired if r["start_compound"] == c)
        m = sum(1 for r in missed if r["start_compound"] == c)
        if f + m == 0:
            continue
        print(f"{c:<20}{f:>12}{m:>20}{100 * f / (f + m):>9.0f}%")
    print()
    slick = [r for r in records if r["start_compound"] in ("SOFT", "MEDIUM", "HARD")]
    wet = [r for r in records if r["start_compound"] in ("INTERMEDIATE", "WET")]
    if wet and slick:
        rs = 100 * sum(r["fired"] for r in slick) / len(slick)
        rw = 100 * sum(r["fired"] for r in wet) / len(wet)
        print(f"   depart en slick : {rs:.0f}% de declenchement ({len(slick)} GP)")
        print(f"   depart en pneu pluie : {rw:.0f}% ({len(wet)} GP)")
        print(f"\n   -> {'HYPOTHESE CONFIRMEE' if rs - rw > 30 else 'hypothese non confirmee : le compose de depart n explique pas tout'}")

    # --- 2. region de declenchement ----------------------------------------
    print("\n" + "=" * 84)
    print("2. REGION DE DECLENCHEMENT (observation au tour du basculement)")
    print("=" * 84)
    A = np.array([r["obs_at_max"] for r in fired])
    B = np.array([r["obs_at_max"] for r in missed]) if missed else None
    scale = np.where(A.std(axis=0) > 1e-6, A.std(axis=0), 1.0)

    print(f"{'dimension':<18}{'declenche (moy±ec)':>24}"
          f"{'ne declenche pas':>20}{'ecart (ec-types)':>19}")
    print("-" * 84)
    gaps = []
    for i, nm in enumerate(OBS_NAMES):
        a_m, a_s = A[:, i].mean(), A[:, i].std()
        if B is None:
            continue
        b_m = B[:, i].mean()
        d = abs(b_m - a_m) / scale[i]
        gaps.append((d, nm, a_m, b_m))
        print(f"{nm:<18}{f'{a_m:8.2f} ± {a_s:5.2f}':>24}{b_m:>20.2f}{d:>19.2f}")

    print("\n   Dimensions les plus discriminantes :")
    for d, nm, a_m, b_m in sorted(gaps, reverse=True)[:4]:
        print(f"      {nm:<18} declenche {a_m:7.2f}  vs  {b_m:7.2f}  "
              f"({d:.1f} ecarts-types)")

    # --- 3. detail par GP ---------------------------------------------------
    print("\n" + "=" * 84)
    print("3. DETAIL PAR GP")
    print("=" * 84)
    print(f"{'GP':<30}{'pool':>6}{'depart':<14}{'declenche':>11}"
          f"{'P(pit) max':>12}{'tour':>7}{'age pneu':>10}")
    print("-" * 84)
    for r in sorted(records, key=lambda x: (-x["fired"], x["event"])):
        print(f"{r['event'][:29]:<30}{'oui' if r['in_pool'] else '-':>6}"
              f"{r['start_compound']:<14}{'OUI' if r['fired'] else 'non':>11}"
              f"{r['p_max']:>12.2e}{r['lap_of_max']:>7}"
              f"{r['obs_at_max'][0]:>10.0f}")

    out = ROOT / f"trigger_{path.stem}.json"
    out.write_text(json.dumps(
        {"model": path.name, "trigger_threshold": TRIGGER,
         "obs_names": OBS_NAMES, "records": records}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
