#!/usr/bin/env python3
"""
Donnees du comparatif de strategies — alimente le dashboard
============================================================

Pourquoi ce script remplace le comparatif "agent vs pilote reel"
----------------------------------------------------------------
Le test de sanite (06_sanity_replay.py) montre que rejouer la strategie REELLE
de Gasly dans l'environnement le classe P15 a Silverstone (P6 en realite),
P18 aux Pays-Bas (P4), P20 en Belgique (P10). L'ecart vient d'un biais de
+0.95% sur le temps total de course, soit ~54 s sur 57 tours -- environ 5 a 9
positions dans un peloton compact. Ce biais s'applique quelle que soit la
strategie jouee.

La position finale et le total de points face au pilote reel ne sont donc PAS
des metriques valides de l'agent. En revanche, comparer l'agent a des
strategies scriptees evaluees DANS LE MEME ENVIRONNEMENT reste valide : le
biais affecte identiquement les deux termes de la comparaison.

Ce script produit donc, par GP, l'agent face a deux references :
    0-stop         strategie naive, reference LOYALE (aucune connaissance
                   prealable requise)
    meilleur script ORACLE choisi a posteriori parmi 4 fenetres d'arret --
                   aucun stratege ne connait d'avance la bonne fenetre. C'est
                   une borne haute, pas un concurrent realiste.

Ne pas confondre les deux dans le discours : sur 19 GP hors pool, DQN bat le
0-stop sur 18 et atteint le niveau de l'oracle sur 2.

Sortie : dashboard/data/comparison_<modele>.json

Usage, depuis la racine du projet :
    python scripts/build_comparison_data.py --model models/dqn/dqn_v4_s1.zip --algo dqn
    python scripts/build_comparison_data.py --model <...> --seeds models/dqn/dqn_v4_s1.zip models/dqn/dqn_v4_s2.zip models/dqn/dqn_v4_s3.zip
"""
import argparse
import json
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

FEATURES_PATH = ROOT / "data" / "processed" / "features_dataset.parquet"
OUT_DIR = ROOT / "dashboard" / "data"
INFERENCE_ROLE = "season_2025_inference"
SEASON = 2025
MIN_LAPS = 25
ALGOS = {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"}
COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}
TRAIN_ROLES = {"train_wet", "train_dry"}

SCRIPTS = {
    "0-stop": [],
    "1-stop @40%": [0.40],
    "1-stop @50%": [0.50],
    "1-stop @60%": [0.60],
    "2-stop @33/66%": [0.33, 0.66],
}


def role_of(event):
    for gp in GP_POOL:
        if gp["season"] == SEASON and gp["event"] == event:
            return gp["role"]
    return None


def gp_dict(event):
    return {"season": SEASON, "event": event,
            "role": role_of(event) or INFERENCE_ROLE, "known_issues": []}


def run_agent(model, gp, seed):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    obs, _ = env.reset(seed=seed)
    cum, chosen, forced, done, info = 0.0, [], [], False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(action))
        cum += r
        if info.get("is_pit"):
            entry = {"lap": info["lap"], "compound": str(env.own_tyre_compound)}
            (forced if info.get("forced_compliance_pit") else chosen).append(entry)
    return {"reward": cum, "chosen": chosen, "forced": forced,
            "total_time_s": float(env.own_cum_time),
            "final_position": int(info.get("position", 0))}


def run_script(gp, seed, fracs):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    env.reset(seed=seed)
    total, cum, done, pits = env.race_total_laps, 0.0, False, []
    while not done:
        action = 0
        for f in fracs:
            if env.current_lap == max(int(round(f * total)), 3):
                unused = [c for c in ("MEDIUM", "HARD", "SOFT")
                          if c not in env.compounds_used]
                action = COMPOUND_TO_ACTION[unused[0] if unused else "HARD"]
        lap = env.current_lap
        _, r, done, _, info = env.step(action)
        cum += r
        if info.get("is_pit"):
            pits.append({"lap": lap, "compound": str(env.own_tyre_compound)})
    return {"reward": cum, "pits": pits, "total_time_s": float(env.own_cum_time)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Modele principal, affiche par defaut")
    ap.add_argument("--algo", default="dqn", choices=list(ALGOS))
    ap.add_argument("--seeds", nargs="*", default=[],
                    help="Modeles supplementaires (autres graines) pour la bande d'incertitude")
    ap.add_argument("--n-seeds", type=int, default=5, help="Graines d'environnement par GP")
    args = ap.parse_args()

    import stable_baselines3 as sb3
    cls = getattr(sb3, ALGOS[args.algo])

    def load(rel):
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"Modele introuvable : {p}")
        return p, cls.load(str(p), device="cpu")

    main_path, main_model = load(args.model)
    extras = [load(s) for s in args.seeds if Path(s).name != main_path.name]

    feat = pd.read_parquet(FEATURES_PATH)
    counts = feat[feat["season"] == SEASON].groupby("event")["lap_number"].max()

    events, excluded = [], []
    for event, n in counts.sort_values().items():
        (events if n >= MIN_LAPS else excluded).append((event, int(n)))
    for event, n in excluded:
        print(f"  [ECARTE] {event} — course tronquee ({n} tours), abandon")

    print(f"\nModele principal : {main_path.name} ({args.algo.upper()})")
    print(f"Graines supplementaires : {len(extras)}")
    print(f"{len(events)} GP retenus, {len(excluded)} ecartes\n")

    print(f"{'GP':<30}{'pool':>6}{'agent':>10}{'0-stop':>10}{'oracle':>10}"
          f"{'arrets':>9}{'verdict':>16}")
    print("-" * 91)

    rows = []
    for event, n_laps in events:
        gp = gp_dict(event)
        in_pool = role_of(event) in TRAIN_ROLES
        try:
            agent = [run_agent(main_model, gp, s) for s in range(args.n_seeds)]
            scripts = {name: [run_script(gp, s, f) for s in range(args.n_seeds)]
                       for name, f in SCRIPTS.items()}
        except Exception as exc:                                # noqa: BLE001
            print(f"{event[:29]:<30}  echec : {exc}")
            continue

        a_mean = float(np.mean([r["reward"] for r in agent]))
        a_std = float(np.std([r["reward"] for r in agent]))
        zero = float(np.mean([r["reward"] for r in scripts["0-stop"]]))
        oracle_name, oracle_runs = max(
            ((n, rs) for n, rs in scripts.items() if n != "0-stop"),
            key=lambda kv: np.mean([r["reward"] for r in kv[1]]))
        oracle = float(np.mean([r["reward"] for r in oracle_runs]))

        # bande d'incertitude entre modeles (graines d'entrainement)
        seed_rewards = [a_mean]
        for _, m in extras:
            try:
                seed_rewards.append(float(np.mean(
                    [run_agent(m, gp, s)["reward"] for s in range(args.n_seeds)])))
            except Exception:                                   # noqa: BLE001
                pass

        chosen = agent[0]["chosen"]
        verdict = ("bat l'oracle" if a_mean > oracle
                   else "bat le 0-stop" if a_mean > zero else "sous le 0-stop")
        print(f"{event[:29]:<30}{'oui' if in_pool else '-':>6}{a_mean:>10.1f}"
              f"{zero:>10.1f}{oracle:>10.1f}{len(chosen):>9}{verdict:>16}")

        rows.append({
            "gp_name": event, "total_laps": n_laps,
            "in_train_pool": in_pool,
            "agent": {
                "reward_mean": a_mean, "reward_std": a_std,
                "pit_laps": [p["lap"] for p in chosen],
                "pit_compounds": [p["compound"] for p in chosen],
                "forced_pits": [p["lap"] for p in agent[0]["forced"]],
                "total_time_s": float(np.mean([r["total_time_s"] for r in agent])),
                "final_position": agent[0]["final_position"],
            },
            "zero_stop": {
                "reward_mean": zero,
                "total_time_s": float(np.mean([r["total_time_s"] for r in scripts["0-stop"]])),
            },
            "oracle": {
                "name": oracle_name, "reward_mean": oracle,
                "pit_laps": [p["lap"] for p in oracle_runs[0]["pits"]],
                "total_time_s": float(np.mean([r["total_time_s"] for r in oracle_runs])),
            },
            "all_scripts": {n: float(np.mean([r["reward"] for r in rs]))
                            for n, rs in scripts.items()},
            "seeds": {
                "n_models": len(seed_rewards),
                "reward_mean": float(np.mean(seed_rewards)),
                "reward_std": float(np.std(seed_rewards, ddof=1)) if len(seed_rewards) > 1 else 0.0,
                "reward_min": float(np.min(seed_rewards)),
                "reward_max": float(np.max(seed_rewards)),
            },
            "delta_vs_zero": a_mean - zero,
            "delta_vs_oracle": a_mean - oracle,
        })

    out_pool = [r for r in rows if not r["in_train_pool"]]
    n = len(out_pool)
    bz = sum(1 for r in out_pool if r["delta_vs_zero"] > 0)
    bo = sum(1 for r in out_pool if r["delta_vs_oracle"] > 0)

    payload = {
        "model": main_path.name, "algo": args.algo.upper(),
        "season": SEASON, "n_env_seeds": args.n_seeds,
        "seed_models": [main_path.name] + [p.name for p, _ in extras],
        "excluded_gp": [{"gp_name": e, "laps": l,
                         "reason": f"course tronquee ({l} tours) — abandon, non evaluable"}
                        for e, l in excluded],
        "methodology": {
            "why_not_real_driver":
                "Le test de sanite montre qu'en rejouant la strategie REELLE de "
                "Gasly, l'environnement le classe P15 a Silverstone (P6 reel), "
                "P18 aux Pays-Bas (P4 reel). Un biais de +0.95% sur le temps "
                "total de course (~54 s sur 57 tours, soit ~0.9 s/tour) deplace "
                "l'agent de 5 a 9 positions quelle que soit sa strategie. La "
                "position finale face au pilote reel n'est donc pas une metrique "
                "valide ; la comparaison a des strategies evaluees dans le meme "
                "environnement l'est, le biais affectant identiquement les deux.",
            "zero_stop": "Reference LOYALE : ne demande aucune connaissance prealable.",
            "oracle": "Borne haute choisie A POSTERIORI parmi 4 fenetres. Aucun "
                      "stratege ne connait d'avance la bonne fenetre : ce n'est "
                      "pas un concurrent realiste.",
        },
        "summary_out_of_pool": {
            "n_gp": n, "beats_zero_stop": bz, "beats_oracle": bo,
            "pct_beats_zero_stop": round(100 * bz / n, 1) if n else 0,
            "pct_beats_oracle": round(100 * bo / n, 1) if n else 0,
            "mean_delta_vs_zero": round(float(np.mean([r["delta_vs_zero"] for r in out_pool])), 2) if n else 0,
        },
        "races": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"comparison_{main_path.stem}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    s = payload["summary_out_of_pool"]
    print(f"\n{'=' * 91}")
    print(f"HORS POOL ({n} GP) — bat le 0-stop {bz}/{n} ({s['pct_beats_zero_stop']}%)"
          f"  |  atteint l'oracle {bo}/{n} ({s['pct_beats_oracle']}%)")
    print(f"Gain moyen face au 0-stop : {s['mean_delta_vs_zero']:+.1f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
