"""
Phase 3.3/3.5 — Réglage des hyperparamètres (Optuna) pour DQN, PPO et A2C
====================================================================
DQN avait été écarté de la suite du projet en Phase 3.2 (cf. docs/
dossier_ecrit/phase3_2_resultats_preliminaires.md), motif principal : taux
de DSQ persistant (~29%) lié à la règle des 2 composés. Ce motif ne tient
plus depuis le garde-fou de conformité ajouté en Phase 3.4 (f1_pitstop_env.py,
dernier tour, algorithme-agnostique) : réévalué avec le nouvel environnement,
DQN baseline tombe à 0% de DSQ, comme PPO et A2C (cf. notebook 06). DQN a
donc été réintégré au tuning Optuna.

A2C avait été envisagé pour exclusion (Phase 3.5, premier diagnostic) après
un run où 70% des essais (14/20) s'effondraient dans le même optimum local
dégénéré ("ne jamais pit volontairement"), sans qu'aucun ne soit élagué par
le MedianPruner. En creusant (cf. a2c_optuna_trials.csv de ce run), le
piège n'est pas une faiblesse générale d'A2C mais une zone précise de
l'espace de recherche : à n_steps=8, tous les essais avec learning_rate
> 4e-4 échappent au piège (4/4), tous ceux < 3e-5 y restent (2/2) --
séparation nette. Pour n_steps in {16, 32, 64}, le piège persiste presque
systématiquement même à learning_rate élevé -- donc ce n'est pas juste une
question de budget insuffisant (contrairement au piège initial de PPO,
résolu en augmentant le budget de 100k à 150k), c'est l'espace de
recherche qui incluait des zones structurellement défavorables. A2C est
donc réintégré, avec `sample_a2c_params` resserré en conséquence plutôt
que réévalué sur le même espace qui avait produit 70% d'échecs.

Budget d'entraînement : porté à 500 000 timesteps par essai par défaut
(Phase 3.5), pour aligner le tuning sur le budget des baselines
(TOTAL_TIMESTEPS dans common.py) -- avant cela, les modèles "tunés"
s'entraînaient en fait sur MOINS de données que les baselines (150k vs
500k), faussant la comparaison. Paramétrable via --timesteps pour explorer
un budget encore plus large (ex: 1 000 000) sans modifier le script.

Protocole (deuxième révision -- cf. phase3_2_resultats_preliminaires.md) :
    1. Un premier essai à budget réduit (100 000 timesteps) a montré que la
       corrélation entre le piège d'optimum local ("rester en piste" tout le
       temps) et un hyperparamètre isolé (ex: learning_rate) n'est PAS
       propre -- certains essais à learning_rate élevé restent piégés,
       d'autres à learning_rate très faible s'en échappent. Resserrer la
       recherche sur une hypothèse mono-hyperparamètre n'est donc pas
       justifié par les données.
    2. Décision retenue à la place (budget de calcul disponible) : chaque
       essai utilise le budget COMPLET (150 000 timesteps, identique à
       l'entraînement final), qui s'est montré fiable pour s'échapper du
       piège. Plus coûteux par essai, mais plus honnête qu'un seuil
       intermédiaire arbitraire.
    3. Élagage précoce (MedianPruner) conservé pour ne pas gaspiller de
       temps sur les essais clairement mauvais dès les premières évaluations
       intermédiaires.
    4. Le meilleur essai est directement le modèle final -- pas besoin d'un
       entraînement complet séparé après coup puisque chaque essai EST déjà
       un entraînement complet (contrairement à la première révision).
    5. (3e révision) L'évaluation initiale (un seul environnement à seed
       fixe) laissait un essai "bien tomber" sur un sous-ensemble favorable
       de GP/bruit stochastique -- écart observé entre le score Optuna du
       meilleur essai (-331.2) et sa vraie performance sur une évaluation
       plus large (-374.2, 14% de DSQ). Remplacée par une évaluation qui
       parcourt explicitement les 7 GP d'entraînement × 2 seeds fixes
       (14 épisodes), à chaque checkpoint intermédiaire ET à la fin de
       chaque essai.

Usage :
    python tune_optuna.py --algo dqn --n_trials 20 --timesteps 500000
    python tune_optuna.py --algo ppo --n_trials 20 --timesteps 500000
    python tune_optuna.py --algo a2c --n_trials 20 --timesteps 500000
    # --timesteps est optionnel (defaut 500000) ; ex. pour un budget encore
    # plus large : --timesteps 1000000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import make_train_env, EpisodeLogCallback, TRAIN_GP_ROLES, SEED, MODELS_DIR, LOGS_DIR

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "env"))
sys.path.insert(0, str(ROOT / "src" / "f1_pitstop_rl" / "config"))
from f1_pitstop_env import F1PitStopEnv
from gp_pool_config import GP_POOL

TRAIN_GPS = [gp for gp in GP_POOL if gp["role"] in TRAIN_GP_ROLES]

TRIAL_BUDGET_TIMESTEPS_DEFAULT = 500_000  # Phase 3.5 : aligne par defaut le tuning sur le budget des baselines (cf. en-tete). Modifiable via --timesteps.
EVAL_FREQ = 30_000

# Evaluation robuste (Phase 3.3, 3e revision) : la version precedente
# evaluait chaque essai sur un seul environnement a seed fixe (SEED+1000),
# dont la selection interne de GP n'etait pas garantie de couvrir les 7 GP
# du pool -- un essai pouvait "bien tomber" sur un sous-ensemble favorable
# de GP/bruit stochastique, faussant la comparaison (cf. ecart observe entre
# le score Optuna du meilleur essai (-331.2) et sa vraie performance sur une
# evaluation plus large (-374.2, DSQ 14%), signe de surapprentissage sur le
# scenario d'evaluation). Remplace par une evaluation qui parcourt
# EXPLICITEMENT tous les GP d'entrainement, sur plusieurs seeds fixes.
EVAL_SEEDS = [100, 101]  # 2 seeds x 7 GP = 14 episodes par evaluation

TRIALS_TMP_DIR = MODELS_DIR / "optuna_trials_tmp"


def evaluate_across_pool(model, gp_list: list[dict]) -> float:
    """Evalue un modele (politique deterministe) sur tous les GP du pool
    d'entrainement, chacun avec plusieurs seeds fixes -- couverture complete
    et reproductible, plutot qu'un tirage aleatoire potentiellement biaise."""
    rewards = []
    for gp in gp_list:
        for seed in EVAL_SEEDS:
            env = F1PitStopEnv(fixed_gp=gp, seed=seed)
            obs, info = env.reset(seed=seed)
            total_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                if terminated:
                    break
            rewards.append(total_reward)
    return float(np.mean(rewards))


class TrialEvalCallback(BaseCallback):
    """Callback personnalise (n'utilise plus stable_baselines3.EvalCallback,
    remplacee par evaluate_across_pool) : reporte la performance moyenne sur
    l'ensemble du pool d'entrainement a Optuna, et declenche l'elagage
    precoce si l'essai est peu prometteur."""

    def __init__(self, trial: optuna.Trial, gp_list: list[dict], eval_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.gp_list = gp_list
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            self.last_mean_reward = evaluate_across_pool(self.model, self.gp_list)
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


def sample_ppo_params(trial: optuna.Trial) -> dict:
    # Espace resserre suite a la premiere recherche (10 essais, cf.
    # ppo_optuna_trials.csv) : les essais tombes dans le piege d'optimum
    # local precoce ("ne jamais s'arreter") avaient systematiquement
    # learning_rate < 2e-4 ET ent_coef < 5e-6. L'essai qui a le mieux
    # echappe avait learning_rate=8e-4, ent_coef=5e-4 -- les deux bornes
    # basses sont donc relevees en consequence plutot que resserrees au
    # hasard.
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    batch_size = min(batch_size, n_steps)  # garde-fou : batch_size ne peut pas depasser n_steps
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True),
    }


def sample_a2c_params(trial: optuna.Trial) -> dict:
    # Resserre suite au diagnostic Phase 3.5 (cf. a2c_optuna_trials.csv,
    # premier run large-espace, 70% des essais piegés dans l'optimum local
    # "ne jamais pit") : a n_steps=8, tous les essais avec learning_rate
    # < 3e-5 restent piégés (2/2), tous ceux avec learning_rate > 4e-4
    # echappent (4/4) -- separation nette. Pour n_steps in {16, 32, 64},
    # le piege persiste presque systematiquement MEME a learning_rate eleve
    # (ex: n_steps=32, lr=9.5e-4, toujours piege) -- donc pas juste une
    # question de budget/lr, ces valeurs de n_steps sont structurellement
    # defavorables ici et sont retirees plutot que compensees.
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [5, 8]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
    }


def sample_dqn_params(trial: optuna.Trial) -> dict:
    # Espace standard sb3 pour DQN, adapte a l'horizon court de l'episode
    # (~50-70 tours) et au budget d'entrainement (150k timesteps, identique
    # aux autres algos pour rester comparable). Pas d'a priori resserre
    # ici (contrairement a sample_ppo_params) : DQN n'a jamais ete tune
    # avant (ecarte en Phase 3.2), donc pas d'echec de recherche precedent
    # a exploiter.
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    batch_size = min(batch_size, buffer_size)  # garde-fou : batch_size ne peut pas depasser le buffer
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": buffer_size,
        "learning_starts": trial.suggest_categorical("learning_starts", [1_000, 5_000, 10_000]),
        "batch_size": batch_size,
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 2, 4]),
        "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1_000, 5_000]),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
    }


ALGO_REGISTRY = {
    "dqn": (DQN, sample_dqn_params),
    "ppo": (PPO, sample_ppo_params),
    "a2c": (A2C, sample_a2c_params),
}


def make_objective(algo_name: str, total_timesteps: int):
    algo_cls, sampler = ALGO_REGISTRY[algo_name]

    def objective(trial: optuna.Trial) -> float:
        params = sampler(trial)
        env = make_train_env(seed=SEED)

        # device='cpu' : cf. train_ppo.py -- reproductibilite des essais
        model = algo_cls("MlpPolicy", env, verbose=0, seed=SEED, device="cpu", **params)

        callback = TrialEvalCallback(trial, TRAIN_GPS, eval_freq=EVAL_FREQ, verbose=0)
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
        except optuna.TrialPruned:
            raise
        except (ValueError, AssertionError) as e:
            # combinaison d'hyperparametres invalide (ex: batch_size > buffer) -> essai rejete
            print(f"Essai rejete (parametres invalides) : {e}")
            return float("-inf")

        TRIALS_TMP_DIR.mkdir(parents=True, exist_ok=True)
        model.save(TRIALS_TMP_DIR / f"{algo_name}_trial_{trial.number}")

        # Evaluation finale complete (peut differer legerement de
        # last_mean_reward si le dernier checkpoint d'evaluation ne
        # correspond pas exactement a la fin de l'entrainement)
        final_score = evaluate_across_pool(model, TRAIN_GPS)
        return final_score

    return objective


def promote_best_trial(algo_name: str, best_trial_number: int):
    """Copie le modele deja entraine de l'essai gagnant comme modele final
    '_tuned', au lieu de relancer un entrainement complet redondant (chaque
    essai etant deja a budget complet, cf. protocole revise en-tete)."""
    import shutil

    src = TRIALS_TMP_DIR / f"{algo_name}_trial_{best_trial_number}.zip"
    out_dir = MODELS_DIR / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{algo_name}_tuned.zip"
    shutil.copy(src, dst)
    print(f"\nModele du meilleur essai (#{best_trial_number}) copie -> {dst}")

    # nettoyage des modeles temporaires des autres essais (non bloquant : un
    # verrou de synchronisation OneDrive/antivirus peut empecher la
    # suppression du dossier lui-meme, ce n'est pas grave, seuls les fichiers
    # .zip comptent et sont deja supprimes a ce stade)
    if TRIALS_TMP_DIR.exists():
        for f in TRIALS_TMP_DIR.glob(f"{algo_name}_trial_*.zip"):
            f.unlink()
        try:
            if not any(TRIALS_TMP_DIR.iterdir()):
                TRIALS_TMP_DIR.rmdir()
        except OSError:
            pass  # dossier verrouille (OneDrive, antivirus...) -- sans consequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo", "a2c"], required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument(
        "--timesteps", type=int, default=TRIAL_BUDGET_TIMESTEPS_DEFAULT,
        help="Budget de timesteps par essai (defaut : 500000, aligne sur les baselines).",
    )
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=f"{args.algo}_tuning",
    )
    study.optimize(make_objective(args.algo, args.timesteps), n_trials=args.n_trials)

    print(f"\n=== Meilleur essai ({args.algo.upper()}) ===")
    print(f"Numero d'essai : {study.best_trial.number}")
    print(f"Valeur (récompense moyenne) : {study.best_value:.1f}")
    print("Hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    results_path = LOGS_DIR / f"{args.algo}_optuna_best_params.json"
    with open(results_path, "w") as f:
        json.dump(
            {"best_value": study.best_value, "best_params": study.best_params, "timesteps_per_trial": args.timesteps},
            f, indent=2,
        )
    print(f"\nRésultats sauvegardés -> {results_path}")

    trials_csv_path = LOGS_DIR / f"{args.algo}_optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_csv_path, index=False)
    print(f"Détail de tous les essais -> {trials_csv_path}")

    promote_best_trial(args.algo, study.best_trial.number)


if __name__ == "__main__":
    main()
