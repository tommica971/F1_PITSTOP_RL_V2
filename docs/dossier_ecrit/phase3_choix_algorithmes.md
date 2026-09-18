# Phase 3.1 — Sélection et justification des algorithmes (C5.1.3)

## 1. Critères de sélection

Le choix des algorithmes candidats découle directement des caractéristiques
de `F1PitStopEnv`, figées en Phase 2 :

| Caractéristique de l'environnement | Conséquence sur le choix d'algorithme |
|---|---|
| Espace d'action **discret** (6 actions) | Élimine d'emblée les algorithmes continus purs (SAC, DDPG, TD3) |
| Épisodes **courts à moyens** (43-72 tours) | Peu de contrainte sur l'algorithme, mais favorise les méthodes qui convergent vite avec peu d'épisodes |
| **Stochasticité** (coût de pit stop échantillonné ~N(23.2, 5.4), bruit du modèle de rythme) | Nécessite un algorithme robuste au bruit de récompense, pas uniquement optimal en environnement déterministe |
| Récompense **dense + terminale** (temps-équivalent tour par tour + bonus/DSQ en fin de course) | Pas de problème de récompense extrêmement sparse -- les algorithmes classiques (sans reward shaping exotique) sont adaptés |
| **Diversité d'entraînement limitée** (7 GP d'entraînement : 4 météo variable + 3 secs) | Favorise les algorithmes échantillon-efficaces (réutilisation des données) |
| Un seul agent, **adversaires figés** (rejeu historique, décision Phase 0) | Pas de non-stationnarité liée à des adversaires apprenants -- simplifie l'apprentissage par rapport à un cadre multi-agent réel |

Ces critères motivent la comparaison de trois algorithmes classiques et
bien documentés, couvrant deux familles différentes (value-based vs
actor-critic), plutôt qu'un choix unique a priori -- l'objectif de la
Phase 3 est justement de trancher **empiriquement** (C5.2.3), sur la base
d'une comparaison rigoureuse plutôt que d'une seule intuition.

## 2. Présentation des trois candidats

### DQN (Deep Q-Network)
Algorithme **value-based**, hors-politique (off-policy). Apprend une fonction
Q(s,a) approximée par un réseau de neurones, avec rejeu d'expérience
(*experience replay*) et réseau cible (*target network*) pour stabiliser
l'apprentissage.

- **Atouts pour ce projet** : hors-politique → réutilise les transitions
  passées via le buffer de rejeu, ce qui est précieux vu la diversité
  d'entraînement limitée (7 GP). Naturellement adapté à un espace d'action
  discret de petite taille (6 actions) comme le nôtre.
- **Limites connues** : peut surestimer les valeurs Q (biais optimiste),
  sensible aux hyperparamètres d'exploration (ε-greedy), historiquement
  moins stable sur des récompenses bruitées que les méthodes actor-critic
  modernes -- point de vigilance direct compte tenu du caractère stochastique
  de notre fonction de récompense (Phase 2.2).

### PPO (Proximal Policy Optimization)
Algorithme **actor-critic**, sur-politique (on-policy), avec objectif clippé
qui limite l'amplitude des mises à jour de politique pour éviter les
effondrements d'apprentissage.

- **Atouts pour ce projet** : reconnu pour sa robustesse aux hyperparamètres
  et sa stabilité de convergence, y compris en présence de bruit dans la
  récompense -- pertinent vu le coût de pit stop stochastique. Gère bien les
  épisodes de longueur variable (43 à 72 tours selon le GP).
- **Limites connues** : sur-politique → ne réutilise pas les anciennes
  transitions, donc potentiellement moins échantillon-efficace que DQN sur
  un pool d'entraînement restreint. Coût de calcul par itération plus élevé
  (plusieurs époques de descente de gradient par lot collecté).

### A2C (Advantage Actor-Critic)
Version synchrone et simplifiée de A3C, actor-critic sur-politique, sans le
clipping de PPO. Sert ici de **référence/baseline** pour mesurer l'apport
réel du clipping PPO sur notre problème spécifique.

- **Atouts pour ce projet** : plus simple et plus rapide par itération que
  PPO (une seule époque de mise à jour), utile comme point de comparaison
  pour vérifier si la sophistication de PPO se justifie réellement sur ce
  problème précis, ou si un algorithme plus simple suffit.
- **Limites connues** : généralement moins stable que PPO sur des tâches
  bruitées (pas de mécanisme de clipping), plus sensible au taux
  d'apprentissage.

## 3. Synthèse comparative

| Critère | DQN | PPO | A2C |
|---|---|---|---|
| Famille | Value-based | Actor-critic | Actor-critic |
| Politique | Hors-politique | Sur-politique | Sur-politique |
| Échantillon-efficacité | Élevée (replay buffer) | Moyenne | Moyenne-faible |
| Robustesse au bruit de récompense | Moyenne | Élevée | Moyenne |
| Simplicité de réglage | Moyenne (ε-greedy, buffer) | Élevée | Moyenne |
| Coût de calcul par itération | Faible-moyen | Moyen-élevé | Faible |
| Adapté à l'action discrète (6) | ✅ Natif | ✅ (politique catégorielle) | ✅ (politique catégorielle) |

## 4. Décision

Les **trois algorithmes seront entraînés et comparés empiriquement** sur le
pool de GP d'entraînement (Phase 3.2-3.5), plutôt que d'en écarter un a
priori sur la seule base de cette analyse théorique. Cette approche est
délibérée :

- Elle constitue en elle-même une démonstration de compétence (C5.1.3 :
  justifier un choix technologique ne signifie pas figer une seule option
  sans preuve, mais motiver une démarche de comparaison rigoureuse)
- Le caractère stochastique de notre récompense (Phase 2.2) rend risqué de
  parier a priori sur DQN malgré son avantage théorique d'échantillon-
  efficacité -- PPO pourrait s'avérer plus fiable en pratique
- La comparaison sera faite sur des critères objectifs définis en amont
  (Phase 3.5) : performance moyenne sur le pool d'entraînement, stabilité de
  convergence, et surtout **capacité de généralisation** sur un GP non vu
  (Phase 3.4) -- le critère le plus important au vu de l'objectif du projet

**Implémentation retenue** : Stable-Baselines3 (cf. `requirements.txt`),
bibliothèque de référence, activement maintenue, avec des implémentations
validées des trois algorithmes candidats et une API homogène (`.learn()`,
`.predict()`) qui simplifie la comparaison à protocole d'entraînement égal.

---

**Suite donnée (Phase 3.2)** : les 3 algorithmes ont été entraînés et
comparés empiriquement comme prévu ci-dessus. DQN a été écarté de la suite
du projet sur la base de ces résultats -- cf.
`phase3_2_resultats_preliminaires.md` pour le détail et la justification.
