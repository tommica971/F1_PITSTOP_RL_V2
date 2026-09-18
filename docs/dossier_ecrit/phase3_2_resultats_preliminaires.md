# Phase 3.2 — Résultats préliminaires et décision budgétaire pour la Phase 3.3

## 1. Contexte

Trois versions successives de l'environnement/récompense ont été entraînées
(150 000 timesteps, hyperparamètres par défaut SB3, pool train_wet+train_dry) :

| Version | Correctif appliqué |
|---|---|
| V1 (baseline) | Espace d'observation à 15 dimensions, récompense terminale seule pour la règle des 2 composés |
| V2 | + `n_compounds_used` ajouté à l'observation (nécessité Markov, cf. `f1_pitstop_env.py`) |
| V3 | + signal de conformité progressif dans les 15 derniers tours si non-conforme (cf. `reward.py`, `COMPLIANCE_WARNING_LAPS`) |

## 2. Résultats — taux de disqualification (règle Article 30.7)

| Algo | V1 | V2 | V3 |
|---|---|---|---|
| DQN | 29% | 43% | **29%** |
| PPO | 43% | 43% | **0%** |
| A2C | 43% | 43% | **10%** |

**V2 seule n'a rien changé** : rendre la règle *observable* ne suffit pas si
le signal d'apprentissage reste uniquement terminal -- confirme que le
problème était bien l'assignation de crédit sur un horizon long (50-70
tours), pas un manque d'information.

**V3 a résolu le problème quasi intégralement pour PPO (0%) et A2C (10%),
mais seulement partiellement pour DQN (29%, inchangé par rapport à V1).**

## 3. Interprétation

L'écart entre familles d'algorithmes est cohérent avec leurs mécanismes
internes respectifs :

- **PPO et A2C** (actor-critic) utilisent une estimation d'avantage (GAE)
  qui propage un signal de récompense dense sur plusieurs pas de temps de
  façon relativement directe -- le signal de conformité progressif, présent
  à chaque tour dès que la course entre dans sa phase finale, est exploité
  efficacement.
- **DQN** (value-based, bootstrap TD) propage l'information plus lentement à
  travers les mises à jour de la fonction Q, tour par tour -- un horizon de
  50-70 pas reste difficile à intégrer même avec un signal dense, sans
  garantie de convergence rapide au même budget d'entraînement.

Ce n'est pas une surprise complète : la Phase 3.1 avait déjà noté que DQN
"peut être moins stable sur des récompenses bruitées que les méthodes
actor-critic modernes" -- ce résultat empirique confirme cette anticipation
théorique sur un point précis et mesurable.

## 4. Résultats globaux (récompense moyenne, évaluation déterministe, V3)

| Algo | Récompense moyenne | Position moyenne | DSQ |
|---|---|---|---|
| **PPO** | **-347.7** | 12.8 | 0% |
| A2C | -352.4 | 12.7 | 10% |
| DQN | -464.9 | 13.8 | 29% |

PPO ressort en tête sur les trois critères simultanément à ce stade.

## 5. Décision finale — DQN écarté de la suite du projet

Après cette analyse, **DQN est retiré de la comparaison à partir de la
Phase 3.3**. Ce n'est pas un abandon prématuré : la comparaison empirique
prévue en Phase 3.1 a été menée à son terme (les 3 algorithmes ont bien été
entraînés et évalués dans les mêmes conditions, cf. sections 2 et 4), et le
résultat est suffisamment net pour trancher :

- Taux de DSQ résiduel de 29% après correction de la récompense (V3), contre
  0% (PPO) et 10% (A2C) -- écart qui persiste malgré un signal
  d'apprentissage dense, signe d'une limite structurelle et non d'un simple
  défaut de réglage (cf. section 3)
- Récompense moyenne nettement en retrait (-464.9 contre -347.7 pour PPO et
  -352.4 pour A2C)

**La suite du projet (Phase 3.3 Optuna, Phase 3.4 test de généralisation,
Phase 3.5 sélection finale) se concentre donc sur PPO et A2C uniquement.**
Le code et les résultats DQN sont conservés dans le dépôt (`train_dqn.py`,
`models/dqn/`) à titre de preuve de la démarche comparative pour le dossier
écrit (C5.1.3/C5.2.3 : justifier un choix technologique par la comparaison,
y compris en documentant pourquoi un candidat a été écarté en cours de
route), mais ne font plus l'objet de développement actif.

## 6. Point de vigilance pour la suite

Le taux de DSQ et la récompense moyenne constituent déjà, à ce stade, un
signal fort en faveur de PPO pour la sélection finale (Phase 3.5) -- à
confirmer ou nuancer une fois le réglage Optuna et le test de généralisation
(GP jamais vus à l'entraînement) réalisés sur PPO et A2C.
