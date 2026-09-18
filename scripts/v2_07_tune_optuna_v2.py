#!/usr/bin/env python3
"""
V2 — Phase 4 : reglage des hyperparametres, version corrigee
=============================================================

Cinq defauts de l'etude V1, et ce qui change
---------------------------------------------

1. L'OBJECTIF EVALUAIT SUR LE POOL D'ENTRAINEMENT.
   `evaluate_across_pool(model, TRAIN_GPS)` : Optuna optimisait donc la
   performance sur les GP deja vus, c'est-a-dire le surapprentissage. Cela
   explique une large part de l'ecart entrainement/generalisation mesure
   ensuite (A2C : 1.00 arret choisi a l'entrainement, 0.51 hors pool).
   -> l'objectif porte desormais sur des GP JAMAIS VUS.

2. AUCUNE GRAINE SUR LE SAMPLER.
   `create_study` sans `sampler` : l'etude n'etait pas rejouable, ce qui est
   un manquement direct a la tracabilite (C5.3.1).
   -> TPESampler(seed=42).

3. gamma ECHANTILLONNE DANS [0.90, 0.999].
   C'est ainsi que 0.919 a ete retenu, soit un horizon effectif de 12 tours
   sur un probleme de 70 pas ou l'arret se rembourse sur 25 a 30 tours.
   -> [0.98, 0.999] : horizon minimal de 50 tours.

4. ent_coef D'A2C DANS [1e-8, 1e-1].
   C'est ainsi que 1.11e-06 a ete retenu, valeur a laquelle l'entropie
   s'effondre a 0.0002 contre log(6)=1.79 possible, et l'agent ne s'arrete
   plus jamais.
   -> borne basse relevee a 1e-4. n_steps d'A2C elargi pour inclure 32, la
   valeur qui fonctionne et que l'espace V1 excluait ([5, 8] seulement).

5. L'OBJECTIF N'APPLIQUAIT AUCUN CORRECTIF v4.
   Ni a priori STAY, ni reward_scale, ni min_stint_laps. Optuna entrainait
   donc des agents qui s'effondrent, et selectionnait les hyperparametres qui
   s'effondrent le mieux. C'est le defaut bloquant : relancer sans le
   corriger reproduirait le probleme a l'identique.
   -> l'objectif reutilise RewardScale et apply_stay_prior de train_v4.py,
   pour que les hyperparametres trouves soient transferables au protocole
   d'entrainement reel.

Plusieurs graines d'entrainement par essai
-------------------------------------------
L'ablation 2x2 a mesure une variabilite inter-graines de +/-16 points sur le
taux de victoire, avec une configuration allant de 21 % a 74 % selon la seule
graine d'entrainement. Evaluer chaque essai sur UNE graine reviendrait donc a
optimiser en grande partie du bruit : un mauvais jeu d'hyperparametres avec une
graine favorable scorerait mieux qu'un bon avec une graine defavorable.

Chaque essai entraine donc N_TRAIN_SEEDS agents et l'objectif est leur moyenne.
C'est un meilleur emploi du temps de calcul qu'un nombre d'essais plus eleve a
une seule graine : on reduit le bruit de la fonction objectif au lieu d'affiner
une recherche sur ce bruit.

Le pruner ne suit que la premiere graine — des rapports intermediaires
provenant de plusieurs entrainements se chevaucheraient et fausseraient la
comparaison aux medianes.

Choix des GP de validation
---------------------------
Evaluer sur les 19 GP hors pool a chaque checkpoint coute trop cher (16
evaluations par essai x 20 essais). Un sous-ensemble de 6 GP est utilise pour
les evaluations intermediaires qui alimentent le pruner, et l'evaluation
FINALE de chaque essai porte sur les 19.

Ces 6 GP ne sont pas choisis a la main : ils sont stratifies sur l'ecart
entre le 0-stop et l'oracle, c'est-a-dire sur l'importance reelle de la
strategie sur ce circuit. Deux GP a fort ecart, deux medians, deux a faible
ecart. La selection est calculee une fois puis mise en cache, donc
reproductible.

Usage, depuis la racine V2 :
    python scripts/v2_07_tune_optuna_v2.py --algo a2c --n-trials 20
    python scripts/v2_07_tune_optuna_v2.py --algo dqn --n-trials 20 --timesteps 500000
    python scripts/v2_07_tune_optuna_v2.py --select-only    # montre les 6 GP
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parent.parent
for p in ("src/f1_pitstop_rl/env", "src/f1_pitstop_rl/config",
          "src/f1_pitstop_rl/training"):
    sys.path.insert(0, str(ROOT / p))

from f1_pitstop_env import F1PitStopEnv, ACTION_TO_COMPOUND  # noqa: E402
from gp_pool_config import GP_POOL  # noqa: E402
from common import make_train_env, LOGS_DIR, MODELS_DIR, SEED  # noqa: E402
from train_v4 import RewardScale, apply_stay_prior  # noqa: E402

SAMPLER_SEED = 42
STAY_PRIOR = 0.97
REWARD_SCALE = 0.1
EVAL_SEEDS = (100, 101)
EVAL_FREQ = 50_000
N_TRAIN_SEEDS = 2      # graines d'entrainement par essai, cf. en-tete
N_VALIDATION_GPS = 6
SEASON = 2025
MIN_LAPS = 25
INFERENCE_ROLE = "season_2025_inference"
CACHE = ROOT / "data" / "processed" / "optuna_validation_gps.json"
COMPOUND_TO_ACTION = {c: a for a, c in ACTION_TO_COMPOUND.items()}
TRAIN_ROLES = {"train_wet", "train_dry"}
ALGOS = {"a2c": A2C, "dqn": DQN, "ppo": PPO}


# ---------------------------------------------------------------------------
def held_out_gps() -> list[dict]:
    feat = pd.read_parquet(ROOT / "data" / "processed" / "features_dataset.parquet")
    pool = {(g["season"], g["event"]) for g in GP_POOL if g.get("role") in TRAIN_ROLES}
    counts = feat[feat["season"] == SEASON].groupby("event")["lap_number"].max()
    return [{"season": SEASON, "event": e, "role": INFERENCE_ROLE, "known_issues": []}
            for e, n in counts.items()
            if (SEASON, e) not in pool and n >= MIN_LAPS]


def script_reward(gp, seed, fracs):
    env = F1PitStopEnv(fixed_gp=gp, seed=seed)
    env.reset(seed=seed)
    total, cum, done = env.race_total_laps, 0.0, False
    while not done:
        action = 0
        for f in fracs:
            if env.current_lap == max(int(round(f * total)), 3):
                unused = [c for c in ("MEDIUM", "HARD", "SOFT")
                          if c not in env.compounds_used]
                action = COMPOUND_TO_ACTION[unused[0] if unused else "HARD"]
        _, r, done, _, _ = env.step(action)
        cum += r
    return cum


def select_validation_gps(all_gps, verbose=True):
    """6 GP stratifies sur l'ecart 0-stop / oracle, mis en cache."""
    if CACHE.exists():
        names = json.loads(CACHE.read_text(encoding="utf-8"))["events"]
        return [g for g in all_gps if g["event"] in names]

    if verbose:
        print(f"Selection des {N_VALIDATION_GPS} GP de validation "
              f"(ecart 0-stop / oracle, calcule une fois)...")
    gaps = []
    for gp in all_gps:
        try:
            zero = np.mean([script_reward(gp, s, []) for s in EVAL_SEEDS])
            best = max(np.mean([script_reward(gp, s, f) for s in EVAL_SEEDS])
                       for f in ([0.40], [0.50], [0.60], [0.33, 0.66]))
            gaps.append((float(best - zero), gp))
        except Exception:                                   # noqa: BLE001
            continue
    gaps.sort(key=lambda x: -x[0])
    n = len(gaps)
    # 2 a fort ecart, 2 medians, 2 a faible ecart
    picks = [gaps[0], gaps[1],
             gaps[n // 2 - 1], gaps[n // 2],
             gaps[-2], gaps[-1]][:N_VALIDATION_GPS]
    if verbose:
        print(f"{'GP':<32}{'ecart 0-stop / oracle':>24}")
        for g, gp in picks:
            print(f"{gp['event'][:31]:<32}{g:>24.1f}")
        print()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({
        "criterion": "ecart entre le 0-stop et le meilleur script, par GP",
        "eval_seeds": list(EVAL_SEEDS),
        "events": [gp["event"] for _, gp in picks],
        "gaps": {gp["event"]: round(g, 2) for g, gp in picks},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return [gp for _, gp in picks]


def evaluate(model, gps) -> float:
    rewards = []
    for gp in gps:
        for seed in EVAL_SEEDS:
            env = F1PitStopEnv(fixed_gp=gp, seed=seed)
            obs, _ = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _, _ = env.step(int(action))
                total += r
            rewards.append(total)
    return float(np.mean(rewards))


class TrialEvalCallback(BaseCallback):
    def __init__(self, trial, gps, eval_freq):
        super().__init__(0)
        self.trial, self.gps, self.eval_freq, self.idx = trial, gps, eval_freq, 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.idx += 1
            self.trial.report(evaluate(self.model, self.gps), self.idx)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


# --- espaces de recherche ---------------------------------------------------
def sample_a2c(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
        # V1 : [5, 8] seulement — la valeur qui fonctionne (32) etait exclue
        "n_steps": trial.suggest_categorical("n_steps", [8, 16, 32, 64]),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),      # V1 : 0.90-0.999
        "gae_lambda": trial.suggest_float("gae_lambda", 0.85, 0.99),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),  # V1 : 1e-8
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "normalize_advantage": True,   # sans cela ent_coef reste cosmetique
    }


def sample_ppo(trial):
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "n_steps": n_steps,
        "batch_size": min(trial.suggest_categorical("batch_size", [32, 64, 128]), n_steps),
        "n_epochs": trial.suggest_int("n_epochs", 4, 15),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.85, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True),
    }


def sample_dqn(trial):
    buf = trial.suggest_categorical("buffer_size", [100_000, 200_000, 400_000])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
        "buffer_size": buf,
        "learning_starts": trial.suggest_categorical("learning_starts", [5_000, 10_000, 20_000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [4, 8]),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1_000, 5_000]),
        # exploration_final_eps plus bas qu'en V1 : a 0.10 l'agent tire encore
        # 10 % d'actions aleatoires en fin d'entrainement, dont 83 % sont des
        # arrets — d'ou les episodes a 13 arrets par course observes.
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.2, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.10),
    }


SAMPLERS = {"a2c": sample_a2c, "ppo": sample_ppo, "dqn": sample_dqn}


def make_objective(algo, timesteps, val_gps, all_gps):
    cls, sampler = ALGOS[algo], SAMPLERS[algo]

    def objective(trial):
        params = sampler(trial)
        full_scores, val_scores = [], []

        for k, train_seed in enumerate(range(SEED, SEED + N_TRAIN_SEEDS)):
            env = make_train_env(seed=train_seed)
            if REWARD_SCALE != 1.0:
                env = RewardScale(env, REWARD_SCALE)
            model = cls("MlpPolicy", env, verbose=0, seed=train_seed,
                        device="cpu", **params)
            if not apply_stay_prior(model, algo, STAY_PRIOR):
                sys.exit("ERREUR : apply_stay_prior a echoue — structure de "
                         "politique SB3 inattendue. Sans a priori STAY, l'etude "
                         "reproduirait l'effondrement de la V1. Arret.")

            # Le pruner ne suit que la PREMIERE graine : sur les suivantes, les
            # rapports intermediaires se chevaucheraient et fausseraient la
            # comparaison aux medianes.
            cb = TrialEvalCallback(trial, val_gps, EVAL_FREQ) if k == 0 else None
            try:
                model.learn(total_timesteps=timesteps, callback=cb)
            except optuna.TrialPruned:
                raise
            except (ValueError, AssertionError) as exc:
                print(f"Essai {trial.number} rejete (parametres invalides) : {exc}")
                return float("-inf")

            full_scores.append(evaluate(model, all_gps))
            val_scores.append(evaluate(model, val_gps))
            if k == 0:
                tmp = MODELS_DIR / "optuna_trials_tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                model.save(tmp / f"{algo}_trial_{trial.number}")

        # Moyenne sur les graines d'entrainement : c'est ce qui reduit le bruit
        # de l'objectif, mesure a +/-16 points de taux de victoire lors de
        # l'ablation 2x2.
        trial.set_user_attr("score_validation", float(np.mean(val_scores)))
        trial.set_user_attr("scores_par_graine", [round(x, 2) for x in full_scores])
        trial.set_user_attr("ecart_type_graines",
                            float(np.std(full_scores, ddof=1)) if len(full_scores) > 1 else 0.0)
        return float(np.mean(full_scores))

    return objective


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=list(ALGOS), default="a2c")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--select-only", action="store_true",
                    help="Affiche les GP de validation et s'arrete.")
    args = ap.parse_args()

    all_gps = held_out_gps()
    val_gps = select_validation_gps(all_gps)
    print(f"{len(all_gps)} GP hors pool | {len(val_gps)} en validation intermediaire : "
          + ", ".join(g["event"].replace(" Grand Prix", "") for g in val_gps))
    if args.select_only:
        return

    print(f"\nAlgo {args.algo.upper()} | {args.n_trials} essais | "
          f"{args.timesteps:,} pas x {N_TRAIN_SEEDS} graines par essai")
    print(f"soit {args.n_trials * N_TRAIN_SEEDS} entrainements au total")
    print(f"a priori STAY {STAY_PRIOR} | reward_scale {REWARD_SCALE} | "
          f"sampler seed {SAMPLER_SEED}\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SAMPLER_SEED),     # V1 : aucune graine
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=4),
        study_name=f"{args.algo}_tuning_v2",
    )
    study.optimize(make_objective(args.algo, args.timesteps, val_gps, all_gps),
                   n_trials=args.n_trials)

    print(f"\n=== Meilleur essai ({args.algo.upper()}) ===")
    print(f"Essai #{study.best_trial.number}")
    print(f"Score sur les {len(all_gps)} GP hors pool : {study.best_value:.1f}")
    print(f"Score sur les {len(val_gps)} GP de validation : "
          f"{study.best_trial.user_attrs.get('score_validation', float('nan')):.1f}")
    print(f"Par graine : {study.best_trial.user_attrs.get('scores_par_graine')} "
          f"(ecart-type {study.best_trial.user_attrs.get('ecart_type_graines', 0):.1f})")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / f"{args.algo}_optuna_v2_best_params.json").write_text(json.dumps({
        "best_value_heldout": study.best_value,
        "best_params": study.best_params,
        "timesteps_per_trial": args.timesteps,
        "sampler_seed": SAMPLER_SEED, "n_train_seeds": N_TRAIN_SEEDS,
        "scores_par_graine": study.best_trial.user_attrs.get("scores_par_graine"),
        "stay_prior": STAY_PRIOR, "reward_scale": REWARD_SCALE,
        "validation_gps": [g["event"] for g in val_gps],
        "n_heldout_gps": len(all_gps),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    study.trials_dataframe().to_csv(
        LOGS_DIR / f"{args.algo}_optuna_v2_trials.csv", index=False)

    print(f"\n-> {LOGS_DIR / f'{args.algo}_optuna_v2_best_params.json'}")
    print("\nEntrainer ensuite les 3 graines avec ces parametres, puis evaluer.")


if __name__ == "__main__":
    main()
