#!/usr/bin/env python3
"""
Inference saison — version 2 (environnement v4, modele parametrable)
=====================================================================

Remplace run_season_inference.py pour la generation des donnees du dashboard.
Ne fait PLUS d'extraction : lit uniquement les parquets, deja nettoyes par
scripts/01_clean_laps.py. Relancer l'extraction reecrirait ces fichiers et
reintroduirait les artefacts sesT (temps a 8291 s au tour 1, Belgique 2025).
Si un GP manque, l'ajouter avec l'ancien script SANS --skip-extraction, puis
relancer 01_clean_laps.py, puis revenir ici.

Corrections par rapport a la v1 (toutes identifiees par scripts/audit_dashboard_data.py)
-----------------------------------------------------------------------------------------
1. --model / --algo : le modele n'est plus code en dur. Supporte A2C, PPO, DQN
   (DQN n'expose pas de distribution de politique : les probabilites sont
   derivees des Q-valeurs par softmax, et signalees comme telles).

2. pit_real_this_lap etait code en dur a False sur TOUS les tours, alors que
   40 arrets reels existaient : le dashboard ne pouvait marquer aucun arret du
   pilote reel sur la timeline. Desormais renseigne depuis les changements de
   relais.

3. tire_age_agent etait gele au tour 2 sur les 23 GP : tire_age_before etait
   initialise a l'age du tour 1, alors que l'environnement fait +1 au reset
   (l'episode demarre au tour 2). Les pneus de l'agent avaient donc
   systematiquement un tour de moins que ceux du pilote reel, sur toute la
   course.

4. Arrets reels fantomes : un changement de stint_number sans remise a zero de
   l'age pneu n'est pas un arret (Australie comptait 5 arrets au lieu de 2,
   Canada 4 au lieu de 1). Un arret exige desormais une remise a zero.

5. Courses tronquees : Monaco 2025 (8 tours sur 78, abandon) etait traite comme
   une course complete et l'agent y "gagnait" 12 points en passant P18 -> P1 en
   trois tours sans aucune action. Ces GP sont desormais ECARTES et listes avec
   leur raison dans season_2025.json, pour affichage explicite au dashboard.

6. deterministe par defaut : la v1 echantillonnait (deterministic=False), donc
   le dashboard changeait a chaque execution. --stochastic pour l'ancien
   comportement.

Usage, depuis la racine du projet :
    python scripts/run_season_inference_v2.py --model models/dqn/dqn_v4_s1.zip --algo dqn
    python scripts/run_season_inference_v2.py --model <...> --gp "Bahrain Grand Prix"
"""
import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "f1_pitstop_rl"
for p in ("config", "env"):
    sys.path.insert(0, str(SRC / p))

from gp_pool_config import GP_POOL  # noqa: E402
from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND  # noqa: E402
from reward import POINTS_TABLE  # noqa: E402

FEATURES_PATH = ROOT / "data" / "processed" / "features_dataset.parquet"
DASHBOARD_DATA_DIR = ROOT / "dashboard" / "data"
INFERENCE_ROLE = "season_2025_inference"
SEASON = 2025
MIN_LAPS = 25          # en deca : course tronquee (abandon), non evaluable
ALGOS = {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"}
ACTION_NAMES = ["STAY"] + [f"PIT_{ACTION_TO_COMPOUND[i]}"
                           for i in sorted(ACTION_TO_COMPOUND) if i != 0]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def role_in_pool(season, event):
    for gp in GP_POOL:
        if gp["season"] == season and gp["event"] == event:
            return gp["role"]
    return None


def action_probs(model, obs, algo):
    """Probabilites d'action. DQN n'a pas de politique parametree : on derive
    une distribution des Q-valeurs par softmax, pour l'affichage uniquement."""
    import torch
    t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        if algo == "dqn":
            q = model.policy.q_net(t).cpu().numpy().ravel()
            e = np.exp(q - q.max())
            return e / e.sum(), True
        d = model.policy.get_distribution(t).distribution
        return d.probs.cpu().numpy().ravel(), False


def real_pit_laps(real: pd.DataFrame) -> list[int]:
    """Arrets reels : changement de relais AVEC remise a zero de l'age pneu.

    Un simple changement de stint_number produisait des arrets fantomes
    (Australie 5 au lieu de 2, Canada 4 au lieu de 1) : dans les courses
    pluvieuses, la source enregistre plusieurs changements consecutifs sans
    qu'aucun arret n'ait eu lieu.
    """
    r = real.sort_values("lap_number").reset_index(drop=True)
    laps = []
    for i in range(1, len(r)):
        stint_changed = r.loc[i, "stint_number"] != r.loc[i - 1, "stint_number"]
        age_reset = r.loc[i, "tyre_age_laps"] < r.loc[i - 1, "tyre_age_laps"]
        if stint_changed and age_reset:
            laps.append(int(r.loc[i, "lap_number"]))
    return laps


def run_inference(model, algo, season, event, seed, deterministic):
    feat = pd.read_parquet(FEATURES_PATH)
    real = (feat[(feat["season"] == season) & (feat["event"] == event)]
            .sort_values("lap_number").reset_index(drop=True))
    if real.empty:
        return None, "aucune donnee dans features_dataset"
    if len(real) < MIN_LAPS:
        return None, f"course tronquee ({len(real)} tours) — abandon, non evaluable"

    for col in ("position", "tyre_age_laps", "tyre_compound"):
        if col in real.columns:
            real[col] = real[col].ffill().bfill()

    pit_laps_real = real_pit_laps(real)
    gp = {"season": season, "event": event,
          "role": role_in_pool(season, event) or INFERENCE_ROLE, "known_issues": []}
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    obs, _ = env.reset(seed=seed)

    lap1 = real.iloc[0]
    cum_real = 0.0
    laps_out = [{
        "lap": int(lap1["lap_number"]),
        "position_real": int(lap1["position"]), "position_agent": int(lap1["position"]),
        "tire_age_real": int(lap1["tyre_age_laps"]),
        "tire_age_agent": int(lap1["tyre_age_laps"]),
        "compound_real": str(lap1["tyre_compound"]),
        "compound_agent": str(lap1["tyre_compound"]),
        "is_raining": bool(lap1["is_raining"]),
        "track_status": "sc" if lap1.get("is_safety_car") else ("vsc" if lap1.get("is_vsc") else "green"),
        "agent_action": "STAY", "agent_action_confidence": None,
        "agent_action_probs": None, "probs_from_q_values": False,
        "pit_real_this_lap": False, "pit_agent_this_lap": False,
        "forced_compliance_pit": False, "compound_after_pit": None,
        "air_temp_c": None, "track_temp_c": None, "humidity_pct": None,
        "cum_time_real_s": 0.0, "cum_time_agent_s": 0.0, "reward_breakdown": None,
    }]

    # +1 : l'environnement fait own_tyre_age = age(tour 1) + 1 au reset, puisque
    # l'episode demarre au tour 2. Sans ce +1, l'age de l'agent restait fige au
    # tour 2 puis accusait un tour de retard sur le reel toute la course.
    age_agent = int(lap1["tyre_age_laps"]) + 1
    compound_agent = str(lap1["tyre_compound"])
    done = False

    while not done:
        probs, from_q = action_probs(model, obs, algo)
        action, _ = model.predict(obs, deterministic=deterministic)
        lap_num = env.current_lap
        row = real[real["lap_number"] == lap_num]
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        def val(col, cast=float):
            if row.empty or col not in row or pd.isna(row[col].iloc[0]):
                return None
            return cast(row[col].iloc[0])

        lt = val("lap_time_s")
        if lt is not None:
            cum_real += lt

        bd = info.get("reward_breakdown")
        laps_out.append({
            "lap": lap_num,
            "position_real": val("position", int), "position_agent": int(info["position"]),
            "tire_age_real": val("tyre_age_laps", int), "tire_age_agent": age_agent,
            "compound_real": (str(row["tyre_compound"].iloc[0]) if not row.empty else None),
            "compound_agent": compound_agent,
            "is_raining": bool(row["is_raining"].iloc[0]) if not row.empty else False,
            "track_status": ("sc" if (not row.empty and row["is_safety_car"].iloc[0])
                             else ("vsc" if (not row.empty and row["is_vsc"].iloc[0]) else "green")),
            "agent_action": ACTION_NAMES[int(action)],
            "agent_action_confidence": float(np.max(probs)),
            "agent_action_probs": {n: float(p) for n, p in zip(ACTION_NAMES, probs)},
            "probs_from_q_values": bool(from_q),
            "pit_real_this_lap": lap_num in pit_laps_real,
            "pit_agent_this_lap": bool(info["is_pit"]),
            "forced_compliance_pit": bool(info.get("forced_compliance_pit", False)),
            "blocked_by_min_stint": bool(info.get("blocked_by_min_stint", False)),
            "compound_after_pit": None,
            "air_temp_c": val("air_temp_c"), "track_temp_c": val("track_temp_c"),
            "humidity_pct": val("humidity_pct"),
            "cum_time_real_s": cum_real, "cum_time_agent_s": float(env.own_cum_time),
            "reward_breakdown": dataclasses.asdict(bd) if bd is not None else None,
        })

        if info["is_pit"]:
            # env.own_tyre_compound et non l'action predite : en cas de
            # forced_compliance_pit l'environnement substitue une autre action
            # sans la renvoyer a l'appelant (cas Chine 2025).
            compound_agent = str(env.own_tyre_compound)
            age_agent = 1
            laps_out[-1]["compound_after_pit"] = compound_agent
        else:
            age_agent += 1

    pos_agent = laps_out[-1]["position_agent"]
    pos_real = int(real.iloc[-1]["position"])
    rs = real.sort_values("lap_number")
    agent_pits = [l for l in laps_out if l["pit_agent_this_lap"]]

    return {
        "gp_name": event, "season": season,
        "model": Path(model_path_global).name, "algo": algo.upper(),
        "deterministic": deterministic,
        "seen_in_training": role_in_pool(season, event) in ("train_wet", "train_dry"),
        "total_laps": len(laps_out), "laps": laps_out,
        "pit_events": {
            "real": [{"lap": l, "compound_after":
                      str(rs[rs["lap_number"] == l]["tyre_compound"].iloc[0])}
                     for l in pit_laps_real],
            "agent": [{"lap": l["lap"], "compound_after": l["compound_after_pit"],
                       "confidence": l["agent_action_confidence"],
                       "forced_compliance": l["forced_compliance_pit"]}
                      for l in agent_pits],
        },
        "final": {
            "real": {"position": pos_real, "points": POINTS_TABLE.get(pos_real, 0)},
            "agent": {"position": pos_agent, "points": POINTS_TABLE.get(pos_agent, 0)},
        },
    }, None


model_path_global = ""


def main():
    global model_path_global
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="a2c", choices=list(ALGOS))
    ap.add_argument("--gp", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stochastic", action="store_true",
                    help="Echantillonne au lieu de prendre l'action la plus "
                         "probable (comportement de la v1, non reproductible).")
    args = ap.parse_args()

    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"Modele introuvable : {path}")
    model_path_global = str(path)

    import stable_baselines3 as sb3
    model = getattr(sb3, ALGOS[args.algo]).load(str(path), device="cpu")
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    feat = pd.read_parquet(FEATURES_PATH)
    events = ([args.gp] if args.gp else
              sorted(feat[feat["season"] == SEASON]["event"].unique()))

    print(f"Modele : {path.name} ({args.algo.upper()})")
    print(f"Mode   : {'stochastique' if args.stochastic else 'deterministe'}")
    print(f"{len(events)} GP\n")

    races, excluded = [], []
    for event in events:
        race, reason = run_inference(model, args.algo, SEASON, event,
                                     args.seed, not args.stochastic)
        if race is None:
            print(f"  [ECARTE] {event} — {reason}")
            excluded.append({"gp_name": event, "reason": reason})
            continue
        out = DASHBOARD_DATA_DIR / f"race_{slugify(event)}_{SEASON}.json"
        out.write_text(json.dumps(race, indent=2, ensure_ascii=False), encoding="utf-8")
        f = race["final"]
        n_chosen = sum(1 for e in race["pit_events"]["agent"] if not e["forced_compliance"])
        print(f"  {event[:34]:<36} reel P{f['real']['position']:>2} "
              f"({f['real']['points']:>2} pts) | agent P{f['agent']['position']:>2} "
              f"({f['agent']['points']:>2} pts) | {n_chosen} arret(s) choisi(s)")
        races.append(race)

    season_json = {
        "season": SEASON, "model": path.name, "algo": args.algo.upper(),
        "deterministic": not args.stochastic,
        "excluded_gp": excluded,
        "races": [{
            "round": i + 1, "gp_name": r["gp_name"],
            "seen_in_training": r["seen_in_training"],
            "real": {"finish_position": r["final"]["real"]["position"],
                     "points": r["final"]["real"]["points"],
                     "n_pitstops": len(r["pit_events"]["real"])},
            "agent": {"finish_position": r["final"]["agent"]["position"],
                      "points": r["final"]["agent"]["points"],
                      "n_pitstops": len(r["pit_events"]["agent"]),
                      "n_pitstops_chosen": sum(1 for e in r["pit_events"]["agent"]
                                               if not e["forced_compliance"])},
            "delta_points": (r["final"]["agent"]["points"] - r["final"]["real"]["points"]),
        } for i, r in enumerate(races)],
    }
    (DASHBOARD_DATA_DIR / f"season_{SEASON}.json").write_text(
        json.dumps(season_json, indent=2, ensure_ascii=False), encoding="utf-8")

    tr = sum(r["final"]["real"]["points"] for r in races)
    ta = sum(r["final"]["agent"]["points"] for r in races)
    chosen = sum(s["agent"]["n_pitstops_chosen"] for s in season_json["races"])
    print(f"\n{len(races)} GP retenus, {len(excluded)} ecartes")
    print(f"Total reel {tr} pts  |  total agent {ta} pts  |  ecart {ta - tr:+d}")
    print(f"Arrets choisis par l'agent : {chosen} sur {len(races)} GP")
    print(f"\n-> {DASHBOARD_DATA_DIR}")
    print("Etape suivante : python dashboard/build_dashboard.py")


if __name__ == "__main__":
    main()
