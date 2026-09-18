# F1_PITSTOP_RL — document de reprise (RNCP Bloc 5)

À coller en début de nouvelle conversation. Mis à jour le 18/09/2026, fin de la
session « industrialisation ». Remplace le récapitulatif précédent.

---

## 1. Le projet en une phrase

Agent RL (stable-baselines3 : A2C, DQN, PPO) décidant à chaque tour de rester
en piste ou de s'arrêter (5 composés) pour Pierre Gasly, dans un environnement
Gymnasium construit sur des données FastF1. Évaluation contre des baselines
internes : **0-stop** (référence loyale) et **oracle** (meilleure de 4 fenêtres
choisie a posteriori, borne haute et non concurrent réaliste). La comparaison
au pilote réel en positions/points est **invalide** (biais structurel de
5 à 9 positions même en rejouant sa vraie stratégie, cf. §3).

Deux projets :
- **V1** `C:\dev\F1_PITSTOP_RL` — dépôt GitHub `tommica971/F1_PITSTOP_RL`
- **V2** `C:\dev\F1_PITSTOP_RL_V2` — git local, **pas encore sur GitHub**

---

## 2. Acquis des sessions précédentes (résumé)

- Audit : l'ancien comparatif « 181 points contre 18 » était un artefact
  (Monaco tronqué, arrêts réels codés en dur, 19 GP sur 23 sans aucune décision).
- Environnement V1 corrigé : filtre des temps reconstruit (`gp_std` 3-22 s →
  0,6-2,8 s), dégradation calibrée en secondes par régression multi-pilotes,
  référence de rythme par tour (fin du multiplicateur SC fixe),
  `MIN_STINT_LAPS = 5`, a priori STAY (P(stay) = 0,97 à l'init).
- Élargir `OBS_HIGH` de `tyre_age_laps` (45 → 80) a dégradé la généralisation
  (10/19 → 4/19) : annulé.
- Sanity replay : la stratégie réelle de Gasly rejouée finit P15/P18/P20 au
  lieu de P6/P4/P10 → biais de +0,95 % sur le temps total → pivot vers les
  baselines internes.
- V2 : correction du biais de rythme (différentiel d'âge, erreur temporelle
  0,95 % → 0,58 %) mais erreur de position inchangée (8 places) → le décalage
  vient de l'absence d'interactions en piste, pas de la calibration.
- V2, abandonnés après mesure : dégradation non linéaire (censure
  informative, pentes négatives) et index de dégradation par circuit (biais de
  confusion, 44 % de conformité au test de plausibilité).
- Ablation 2×2 (MIN_STINT × STAY) : effets dans le bruit à 3 graines.

---

## 3. Résultats de référence

### V1 — modèle retenu `dqn_v4_s1` (dans l'image Docker)

19 GP hors pool (Monaco écarté, tronqué) : **bat le 0-stop 18/19**, bat
l'oracle 1/19, 1 GP sans arrêt (Monza), 1,24 arrêt/course, perte de 10 % sur
les arrêts choisis par rapport aux GP d'entraînement.
V1 3 graines : DQN 88 % ± 8 contre le 0-stop, A2C 53 % ± 9, PPO 51 % ± 8.

### V2 — 3 graines, 19 GP hors pool (session du 18/09)

| | récompense moy. | bat 0-stop | bat oracle | abstention | catastrophes |
|---|---|---|---|---|---|
| **A2C, réglages par défaut (`a2c_v2env`)** | **−38,8 ± 6,4** | 13 ± 2 | **4 ± 1** | 6 ± 3 | **0** |
| A2C, Optuna V2 (`a2c_v2opt`) | −46,6 ± 7,1 | 9 ± 3 | 2 ± 1 | 9 ± 3 | 0 |
| DQN (`dqn_v2env`) | −56,0 ± 5,4 | 13 ± 2 | 1 ± 2 | 2 ± 1 | 1 ± 2 |
| PPO (`ppo_v2env`) | −61,0 ± 10,9 | 11 ± 2 | 1 ± 1 | 4 ± 2 | 3 ± 1 |

Plancher 0-stop moyen −71,4, oracle moyen −18,8. Catastrophe = agent pire que
le 0-stop de plus de 20 points.

Conclusions :
1. **Optuna V2 sans gain** (tendance défavorable, dans le bruit), alors même
   qu'il a été réglé sur ces 20 GP hors pool. Son estimation était juste
   (−44,5 annoncé, −46,6 mesuré) : la recherche en 20 essais n'a pas trouvé
   mieux que les valeurs par défaut.
2. **Sur V2, A2C passe devant DQN** : à égalité sur « bat le 0-stop » (13/19),
   mais −38,8 contre −56,0 en récompense (≈ 3 écarts-types entre graines) et
   seul algorithme sans catastrophe sur 57 courses. DQN graine 2 et PPO graine 3
   font des boucles de 8 à 12 arrêts (Espagne, Autriche, Canada, Mexique).
3. **Leçon de méthode** : une métrique de comptage (« bat le 0-stop ») ne voit
   pas l'ampleur des échecs. L'avance de DQN en V1 reposait sur elle.
4. **Ne jamais comparer V1 et V2 en valeurs absolues** : l'environnement, donc
   les récompenses et le plancher, ont changé.

Modèle V2 à retenir : **`a2c_v2env`**, présenté par la moyenne sur 3 graines.
Graine à livrer choisie sur les GP d'entraînement (pas sur les 19 GP) :
s2 et s3 à égalité (6/8) → règle retenue : la plus basse, **s2**.

---

## 4. Industrialisation — état au 18/09 (V1)

### Environnement figé (C5.2, était BLOQUANT → levé)
Versions lues dans `system_info.txt` à l'intérieur des `.zip` des modèles :
Python 3.12.7 (entraînement) / 3.12.10 (local et Docker), SB3 2.7.1,
torch 2.11.0 (CUDA à l'entraînement, **CPU** à l'inférence), numpy 2.4.4,
gymnasium 1.2.3, cloudpickle 3.1.2, pandas 3.0.6, pyarrow 25.0.1.

- venv local : `.venv312` (Python 3.12 **python.org**, pas Anaconda : la base
  Anaconda provoquait `WinError 1114` sur `c10.dll`)
- `requirements.txt` (inférence, épinglé, index CPU PyTorch intégré),
  `requirements-dev.txt` (pytest, ruff, optuna), `requirements-data.txt`
  (fastf1, **version à épingler** : 3.3.0 incompatible numpy 2),
  `requirements-monitoring.txt` (evidently 0.7.23), `requirements.lock.txt`
- `check_models.py` : recharge tous les modèles de `models/` (V1 20/20, V2 35/35)

### Reproductibilité démontrée
`dqn_v4_s1`, évaluation de généralisation : **identique à l'octet** entre la
machine d'entraînement (Windows, GPU) et le poste (Windows, CPU), **identique
en contenu JSON** dans le conteneur Linux. Preuves :
`docs/resultats/generalisation_dqn_v4_s1.json` et `..._docker.json`.

### Docker
`python:3.12.10-slim`, cibles `service` (défaut, CMD `check_models.py`, modèle
livré `models/dqn/dqn_v4_s1.zip`) et `notebook` (Jupyter). Utilisateur non root.
`.dockerignore` (contexte 777 Ko). `docker-compose.yml` : `train` (volumes
data/models/reports) et `notebook` (127.0.0.1:8888).

### Git / GitHub
Dépôt `tommica971/F1_PITSTOP_RL`, historique linéaire, tags `v1-final`,
`v1-infra`, `v1-ci`. `.gitattributes` (LF, binaires). Badge CI dans le README.
Parquet **versionnés** (< 1 Mo au total) : `test_env.py` et l'environnement ont
besoin de `features_dataset.parquet` et `all_drivers_dataset.parquet`.

### CI/CD (verte)
- `ci.yml` (push) : ubuntu-24.04, checkout@v5, setup-python@v6, Python 3.12.10.
  Job tests : `check_models.py`, **76 tests**, `scripts/check_env_contract.py`
  (contrat Gymnasium respecté), ruff non bloquant. Job Docker : build service,
  modèle rechargé dans l'image, build notebook, taille.
- `smoke-test.yml` (hebdomadaire + manuel) : A2C 20 000 pas via `train_v4.py`,
  rechargement de `models/a2c/ci_smoke.zip` (vérifié en local).
- `drift.yml` (mensuel + manuel) : rapport de dérive publié en artefact.

### Monitoring de dérive (C5.3.3, était un TODO → livré)
`src/f1_pitstop_rl/monitoring/drift_report.py`, Evidently 0.7.23.
- Référence : 8 GP `train_*` de `GP_POOL` (490 tours). Courant : 19 GP hors
  pool (1 153 tours).
- Variables : les 12 entrées de `_build_observation` (dont `n_compounds_used`
  reconstruit, drapeaux rivaux = `rival_*_vient_de_pitter`).
- Méthode : Wasserstein normalisée / Jensen-Shannon (ampleur), **seuils
  calibrés par leave-one-out** sur le pool (plancher 0,1). Les p-values
  signalaient 9/12 variables (significativité ≠ importance) : rejetées.
- Résultats (GP signalés sur 19) : `track_temp_c` 8 (seule variable dont la
  distance globale dépasse le seuil calibré), `tyre_age_laps` 8, `position` 6,
  `rival_arriere` 6 (bruité, événements rares), `tyre_compound` 4 ; pluie 0,
  `tours_restants` 0 (contrôle). Part moyenne signalée : 17 %.
- Lien performance (`dqn_v4_s1`) : 3 des 4 échecs (Monza, Azerbaïdjan, Canada)
  parmi les 5 GP les plus dérivés ; potentiel capté médian 11 % (≥ 3 variables)
  contre 92 %. **Non significatif** (Spearman ρ = −0,32, p = 0,18) : hypothèse
  cohérente, pas démonstration.
- Hypothèse retirée : « pool surreprésenté en pluie » (démentie par le
  calibrage).
- Preuve : `docs/resultats/drift_summary.json`.

---

## 5. Reste à faire

1. **V2 sur GitHub** avec son infrastructure (copier celle de V1, Dockerfile à
   faire pointer sur `models/a2c/a2c_v2env_s2.zip`, vérifier les tests V2).
2. **`model_training.json`** : pointe encore sur `a2c_extended_pool_5000k`.
3. **Dashboard** : régénérer sur l'environnement V2 avec `a2c_v2env`.
4. `fastf1` à épingler dans `requirements-data.txt`.
5. Vérifier que `reports/drift/` est dans le `.gitignore` de V1.
6. Lancer une fois à la main `smoke-test.yml` et `drift.yml` sur GitHub.

---

## 6. Choix de présentation

- Récit : **V1** démonstrateur qui a révélé les biais ; **V2** réévaluation
  corrigée. Ne pas présenter V2 comme une refonte architecturale.
- Formulations à ne pas rater : « bat la stratégie naïve » ≠ « atteint
  l'oracle » ; résultats **par moyenne sur 3 graines** ; résultats négatifs
  (Optuna V2, deux extensions) présentés comme tels.
- Dashboard : onglet « Stratégie » en accueil (agent / 0-stop / oracle),
  4 onglets, comparaison au pilote réel dans « Sim-to-Real & limites ».
- Perspectives : action « programmer l'arrêt au tour N » ; garde-fou
  d'inférence (plafond d'arrêts) contre les boucles DQN/PPO ; pool enrichi en
  courses chaudes et à relais longs (monitoring) ; plus de graines (≥ 10) pour
  trancher les écarts dans le bruit.
