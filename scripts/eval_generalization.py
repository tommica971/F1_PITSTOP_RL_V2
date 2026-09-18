#!/usr/bin/env python3
"""
Generalisation sur toute la saison 2025 hors pool d'entrainement
=================================================================

Pourquoi ce script plutot que eval_pit_behaviour.py seul
---------------------------------------------------------
Le pool ne contient que 2 GP de test (Belgique 2025, Japon 2026). Sur cet
echantillon, le dernier modele donne 0.0 arret choisi en Belgique et 0.6 au
Japon : impossible d'en conclure quoi que ce soit, ni dans un sens ni dans
l'autre. Deux points ne font pas une mesure.

features_dataset.parquet contient une trentaine de GP. Tous ceux qui ne sont
pas dans GP_POOL n'ont JAMAIS ete vus a l'entrainement : c'est un jeu de
generalisation d'une vingtaine de courses, deja disponible, qui transforme
l'anecdote en mesure.

Ce que le script fait
----------------------
  - liste les GP presents dans features_dataset et absents de GP_POOL
  - ecarte les courses tronquees (abandon simule comme une course complete,
    cf. Monaco 2025 : 8 tours sur 78 -- l'agent y "gagnait" 12 points)
  - evalue l'agent et les strategies scriptees sur chacun, N graines
  - compare l'agregat aux GP d'entrainement pour chiffrer la perte

Sortie : tableau console + generalisation_<modele>.json pour le dossier.

Usage, depuis la racine du projet :
    python scripts/eval_generalization.py --model models/a2c/a2c_v4_500k.zip
    python scripts/eval_generalization.py --model <...> --algo dqn --n-seeds 5
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

from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND  # noqa: E402
from gp_pool_config import GP_POOL  # noqa: E402

ALGOS = {"a2c": "A2C", "ppo": "PPO", "dqn": "DQN"}
COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}
INFERENCE_ROLE = "season_2025_inference"
SCRIPTS = ([0.40], [0.50], [0.60], [0.33, 0.66])
TRAIN_ROLES = {"train_wet", "train_dry"}
MIN_LAPS = 25   # en deca, la course est tronquee (abandon) -> non evaluable


def run_agent(model, gp, seed):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    obs, _ = env.reset(seed=seed)
    cum, chosen, forced, done, info = 0.0, [], [], False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(action))
        cum += r
        if info.get("is_pit"):
            (forced if info.get("forced_compliance_pit") else chosen).append(info["lap"])
    return cum, chosen, forced


def run_script(gp, seed, pit_fracs):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    env.reset(seed=seed)
    total, cum, done = env.race_total_laps, 0.0, False
    while not done:
        action = 0
        for f in pit_fracs:
            if env.current_lap == max(int(round(f * total)), 3):
                unused = [c for c in ("MEDIUM", "HARD", "SOFT")
                          if c not in env.compounds_used]
                action = COMPOUND_TO_ACTION[unused[0] if unused else "HARD"]
        _, r, done, _, _ = env.step(action)
        cum += r
    return cum


def evaluate(model, gp, n_seeds):
    chosen_n, forced_n, laps, rewards = [], [], [], []
    for s in range(n_seeds):
        r, chosen, forced = run_agent(model, gp, s)
        chosen_n.append(len(chosen))
        forced_n.append(len(forced))
        laps += chosen
        rewards.append(r)
    zero = float(np.mean([run_script(gp, s, []) for s in range(n_seeds)]))
    best = max(float(np.mean([run_script(gp, s, f) for s in range(n_seeds)]))
               for f in SCRIPTS)
    return {"event": gp["event"], "chosen": float(np.mean(chosen_n)),
            "forced": float(np.mean(forced_n)), "laps": laps,
            "reward": float(np.mean(rewards)), "zero": zero, "best": best}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="a2c", choices=list(ALGOS))
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.model)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"Modele introuvable : {path}")

    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")
    pool = {(g["season"], g["event"]) for g in GP_POOL}

    counts = (feat[feat["season"] == args.season]
              .groupby("event")["lap_number"].max().sort_values())
    candidates, truncated = [], []
    for event, n_laps in counts.items():
        if (args.season, event) in pool:
            continue
        (truncated if n_laps < MIN_LAPS else candidates).append((event, int(n_laps)))

    if not candidates:
        sys.exit(f"Aucun GP hors pool trouve pour la saison {args.season}.")

    import stable_baselines3 as sb3
    model = getattr(sb3, ALGOS[args.algo]).load(str(path), device="cpu")

    print(f"Modele : {path.name}  |  {args.n_seeds} graines par GP")
    print(f"{len(candidates)} GP hors pool, saison {args.season}")
    if truncated:
        print(f"{len(truncated)} GP ecartes (course tronquee, < {MIN_LAPS} tours) : "
              + ", ".join(f"{e} ({n}t)" for e, n in truncated))
    print()

    print("=" * 100)
    print(f"{'GP':<30}{'tours':>7}{'choisis':>9}{'forces':>8}{'tour median':>15}"
          f"{'agent':>10}{'0-stop':>10}{'best script':>12}")
    print("-" * 100)

    rows = []
    for event, n_laps in candidates:
        gp = {"season": args.season, "event": event,
              "role": INFERENCE_ROLE, "known_issues": []}
        try:
            m = evaluate(model, gp, args.n_seeds)
        except Exception as exc:                            # noqa: BLE001
            print(f"{event[:29]:<30}{n_laps:>7}  echec : {exc}")
            continue
        m["total_laps"] = n_laps
        lap_txt = (f"{np.median(m['laps']):.0f} [{min(m['laps'])}-{max(m['laps'])}]"
                   if m["laps"] else "-")
        print(f"{event[:29]:<30}{n_laps:>7}{m['chosen']:>9.1f}{m['forced']:>8.1f}"
              f"{lap_txt:>15}{m['reward']:>10.1f}{m['zero']:>10.1f}{m['best']:>12.1f}")
        rows.append(m)

    if not rows:
        sys.exit("\nAucun GP evaluable.")

    chosen = float(np.mean([r["chosen"] for r in rows]))
    never = sum(1 for r in rows if r["chosen"] == 0)
    bz = sum(1 for r in rows if r["reward"] > r["zero"])
    bb = sum(1 for r in rows if r["reward"] > r["best"])
    n = len(rows)

    print("\n" + "=" * 100)
    print(f"GENERALISATION — {n} GP jamais vus a l'entrainement\n")
    print(f"  arrets choisis par course : {chosen:.2f}")
    print(f"  GP sans aucun arret choisi : {never}/{n}")
    print(f"  bat le 0-stop              : {bz}/{n}  ({100 * bz / n:.0f}%)")
    print(f"  bat le meilleur script     : {bb}/{n}  ({100 * bb / n:.0f}%)")

    # reference : les memes mesures sur le pool d'entrainement
    train = [g for g in GP_POOL if g.get("role") in TRAIN_ROLES]
    tr_rows = []
    for gp in train:
        try:
            tr_rows.append(evaluate(model, gp, args.n_seeds))
        except Exception:                                   # noqa: BLE001
            continue
    if tr_rows:
        t_chosen = float(np.mean([r["chosen"] for r in tr_rows]))
        t_bz = sum(1 for r in tr_rows if r["reward"] > r["zero"])
        t_bb = sum(1 for r in tr_rows if r["reward"] > r["best"])
        m = len(tr_rows)
        print(f"\n  pour comparaison, sur les {m} GP d'entrainement :")
        print(f"  arrets choisis {t_chosen:.2f}  |  bat le 0-stop {t_bz}/{m}  "
              f"|  bat le script {t_bb}/{m}")
        drop = 1 - chosen / t_chosen if t_chosen else 1.0
        print(f"\n  perte sur les arrets choisis : {100 * drop:.0f}%")

    print()
    if chosen < 0.4:
        print("VERDICT : le comportement d'arret ne se transfere pas. L'agent a")
        print("appris ses GP d'entrainement plutot qu'une politique d'arret.")
    elif bz < n / 2:
        print("VERDICT : l'agent s'arrete hors du pool mais fait moins bien que")
        print("le 0-stop sur la majorite des GP. Transfert partiel, insuffisant")
        print("pour un resultat de dossier sans nuance.")
    elif bb < n / 3:
        print("VERDICT : l'agent bat le 0-stop hors du pool mais reste loin des")
        print("strategies scriptees. Generalisation reelle mais faible, a")
        print("presenter comme telle.")
    else:
        print("VERDICT : le comportement se transfere aux GP jamais vus.")
    print("=" * 100)

    out = Path(args.out) if args.out else ROOT / f"generalisation_{path.stem}.json"
    out.write_text(json.dumps({
        "model": path.name, "algo": args.algo, "season": args.season,
        "n_seeds": args.n_seeds, "min_laps": MIN_LAPS,
        "excluded_truncated": [{"event": e, "laps": l} for e, l in truncated],
        "generalisation": rows,
        "summary": {"mean_chosen": chosen, "n_gp": n, "never_pits": never,
                    "beats_zero": bz, "beats_best": bb},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {out}")


if __name__ == "__main__":
    main()
