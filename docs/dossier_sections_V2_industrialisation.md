# Sections du dossier — résultats V2, industrialisation, monitoring

---

## X. Réévaluation V2 : comparaison des algorithmes

### X.1 Protocole

Les trois algorithmes (A2C, DQN, PPO) ont été entraînés sur l'environnement V2,
qui corrige le biais de rythme identifié en V1, avec trois graines
d'entraînement chacun et des hyperparamètres identiques à V1. Une quatrième
configuration, A2C avec les hyperparamètres issus de l'étude Optuna V2, a été
entraînée dans les mêmes conditions. Chaque modèle est évalué sur les 19 Grands
Prix de la saison 2025 absents du pool d'entraînement (Monaco est écarté :
course tronquée à 8 tours), avec cinq graines d'évaluation par course.

Quatre indicateurs sont rapportés, en moyenne et écart-type sur les trois
graines d'entraînement : la récompense moyenne, le nombre de courses où l'agent
bat la stratégie sans arrêt (0-stop), le nombre de courses où il bat l'oracle
(meilleure stratégie scriptée choisie a posteriori parmi quatre fenêtres d'arrêt
et le 0-stop), et le nombre de
« catastrophes », courses où l'agent fait pire que le 0-stop de plus de
20 points.

Les valeurs absolues ne sont pas comparables à celles de V1 : la correction de
l'environnement modifie la récompense et le plancher 0-stop.

### X.2 Résultats

| Configuration | Récompense moy. | Bat le 0-stop | Bat l'oracle | Abstention | Catastrophes |
|---|---|---|---|---|---|
| A2C, réglages par défaut | **−38,8 ± 6,4** | 13 ± 2 / 19 | **2,7 ± 1,2** | 6 ± 3 | **0** |
| A2C, réglages Optuna V2 | −46,6 ± 7,1 | 9 ± 3 / 19 | 0,7 ± 1,2 | 9 ± 3 | 0 |
| DQN | −56,0 ± 5,4 | 13 ± 2 / 19 | 1,0 ± 1,7 | 2 ± 1 | 1 ± 2 |
| PPO | −61,0 ± 10,9 | 11 ± 2 / 19 | 0,7 ± 0,6 | 4 ± 2 | 3 ± 1 |

Repères : récompense moyenne du 0-stop −71,4 ; de l'oracle −18,8.

### X.3 Analyse

**L'optimisation Optuna V2 n'apporte pas de gain mesurable.** L'A2C réglé par
Optuna obtient une récompense moyenne inférieure à celle de l'A2C aux réglages
par défaut (−46,6 contre −38,8) et s'abstient plus souvent. Cet écart reste dans
le bruit à trois graines, mais sa direction est significative à un autre
titre : l'étude Optuna avait été conduite sur ces mêmes courses hors pool, ce
qui aurait dû avantager la configuration optimisée. L'estimation produite par
l'étude était pourtant fiable (−44,5 annoncé, −46,6 mesuré) : c'est la
recherche, limitée à vingt essais, qui n'a pas trouvé de configuration
supérieure aux valeurs par défaut : neuf essais ont été élagués, deux se sont
effondrés sur la stratégie sans arrêt, et le meilleur essai, le cinquième, n'a
été amélioré par aucun des quinze suivants. Ce résultat négatif est conservé tel quel,
au même titre que les deux extensions de modélisation abandonnées après mesure.

**A2C devient l'algorithme le plus fiable sur V2.** A2C et DQN battent la
stratégie sans arrêt sur le même nombre de courses (13 sur 19). Cet indicateur
masque pourtant une différence majeure : une graine de DQN et une graine de PPO
entrent dans des boucles de huit à douze arrêts sur plusieurs courses, avec des
récompenses jusqu'à cinq fois plus mauvaises que le 0-stop. La récompense
moyenne départage nettement les algorithmes (−38,8 pour A2C contre −56,0 pour
DQN, soit un écart d'environ trois écarts-types entre graines), A2C atteint
plus souvent le niveau de l'oracle (2,7 courses contre 1,0), et il est le
seul à ne produire aucune catastrophe sur 57 courses évaluées.

**Enseignement méthodologique.** Un indicateur de comptage (« bat le 0-stop »)
mesure la fréquence des succès mais ignore l'ampleur des échecs. En V1, la
supériorité de DQN reposait principalement sur cet indicateur. L'évaluation
retenue associe désormais un indicateur de fréquence, un indicateur d'ampleur
(récompense moyenne) et un indicateur de risque (catastrophes).

**Modèle retenu pour V2 : A2C aux réglages par défaut**, présenté par sa moyenne
sur trois graines. La graine livrée est choisie sur les courses d'entraînement,
et non sur les courses d'évaluation, afin de ne pas introduire de biais de
sélection.

### X.4 Définition de l'oracle

Une première version de l'oracle excluait la stratégie sans arrêt. Sur une
course où ne pas s'arrêter est optimal (Miami 2025), un agent qui s'abstenait
était alors compté comme battant l'oracle sans avoir pris de décision. L'oracle
est désormais la meilleure stratégie connue a posteriori, sans arrêt comprise ;
les résultats ci-dessus utilisent cette définition. Ce défaut surévaluait
surtout les configurations qui s'abstiennent le plus (A2C Optuna, A2C par
défaut) ; il ne modifie aucune des conclusions.

### X.5 Limites

Trois graines par configuration ne suffisent pas à trancher les écarts
inférieurs à environ deux écarts-types ; une dizaine de graines serait
nécessaire. Les boucles d'arrêts observées chez DQN et PPO appellent un
garde-fou à l'inférence (plafond d'arrêts par course), proposé en perspective.

---

## Y. Industrialisation et reproductibilité (C5.2)

### Y.1 Environnement figé à partir des modèles eux-mêmes

Les versions des dépendances ne sont pas reconstituées de mémoire : elles sont
lues dans le fichier `system_info.txt` que stable-baselines3 enregistre à
l'intérieur de chaque modèle sauvegardé. Tous les modèles livrés ont été
entraînés sous Python 3.12.7, stable-baselines3 2.7.1, PyTorch 2.11.0,
NumPy 2.4.4 et Gymnasium 1.2.3. Ces versions sont épinglées dans
`requirements.txt`, qui ne contient que les dépendances d'inférence ; les
dépendances de collecte de données, de développement et de monitoring sont
isolées dans des fichiers distincts, et l'environnement complet validé est
conservé dans un fichier de verrouillage.

L'inférence utilise la version CPU de PyTorch : un modèle entraîné sur GPU se
recharge sans modification sur CPU, et l'image de déploiement s'en trouve
nettement allégée.

### Y.2 Reproductibilité démontrée sur trois plateformes

L'évaluation de généralisation du modèle livré a été rejouée sur trois
environnements : la machine d'entraînement (Windows, GPU), un poste de travail
(Windows, CPU) et le conteneur de déploiement (Linux, CPU). Les résultats sont
identiques à l'octet entre les deux environnements Windows, et identiques
valeur par valeur dans le conteneur Linux, sur les 19 courses et 5 graines
d'évaluation. Le comportement du modèle ne dépend donc ni du système, ni du
matériel, ni du passage GPU vers CPU.

### Y.3 Conteneurisation

L'image Docker repose sur une image de base Python dont la version est figée
(3.12.10). Elle propose deux cibles : une cible de service, dont la commande par
défaut vérifie que le modèle livré se recharge, et une cible de démonstration
intégrant Jupyter. Les données sont montées en volume et non figées dans
l'image ; le conteneur s'exécute avec un utilisateur non privilégié.

### Y.4 Intégration continue

À chaque modification poussée sur le dépôt, la chaîne d'intégration continue :
installe l'environnement figé, vérifie que les vingt modèles versionnés se
rechargent, exécute les 76 tests unitaires et d'intégration, contrôle que
l'environnement respecte le contrat de l'API Gymnasium, construit l'image
Docker et vérifie que le modèle livré s'y recharge. Le système d'exploitation
de la chaîne et les versions des actions utilisées sont figés, par cohérence
avec le reste du projet.

Un réentraînement complet en intégration continue n'est pas réaliste
(plusieurs heures par algorithme, limite de six heures par exécution). Il est
remplacé par un test de fumée hebdomadaire : un entraînement court qui vérifie
que la chaîne d'apprentissage démarre et produit un modèle rechargeable, afin
de détecter les régressions silencieuses sans prétendre valider une
performance.

La chaîne a elle-même révélé un défaut : lors de sa première exécution, trente
tests échouaient parce que l'environnement dépendait de fichiers de données non
versionnés. Ces fichiers pesant moins d'un mégaoctet au total, ils ont été
ajoutés au dépôt, ce qui permet de reproduire l'ensemble des résultats à partir
d'un simple clone.

---

## Z. Surveillance de la dérive des données (C5.3.3)

### Z.1 Question posée

Les situations de course rencontrées par l'agent en inférence ressemblent-elles
à celles sur lesquelles il a été entraîné ? Si la distribution d'une variable
d'observation s'écarte trop, les décisions de l'agent reposent sur une
extrapolation. Le rapport compare les huit courses d'entraînement (référence)
aux dix-neuf courses d'évaluation (courant), sur les douze variables qui
composent réellement l'observation de l'agent.

### Z.2 Choix de méthode

Une première version utilisait les tests statistiques par défaut de
l'outil (Kolmogorov-Smirnov, khi-deux) et signalait neuf variables sur douze.
Ces tests répondent à la question « les distributions sont-elles strictement
identiques ? », à laquelle la réponse est toujours négative entre deux saisons
dès que les échantillons sont assez grands : la significativité statistique
n'est pas l'importance pratique. La méthode retenue mesure l'ampleur de l'écart
(distance de Wasserstein normalisée pour les variables numériques, distance de
Jensen-Shannon pour les variables catégorielles).

Le seuil de décision n'est pas fixé arbitrairement. Chaque course
d'entraînement est comparée aux sept autres, et le seuil retenu pour chaque
variable est la plus grande distance observée : c'est la variabilité naturelle
entre deux courses que l'agent connaît. Une course d'évaluation n'est signalée
sur une variable que si elle s'en écarte davantage que les courses
d'entraînement ne s'écartent entre elles.

### Z.3 Résultats

En moyenne, 17 % des variables sont signalées par course. La température de
piste (8 courses sur 19) est la seule variable dont l'écart global dépasse même
la variabilité naturelle du pool : la saison 2025 couvre des conditions
thermiques absentes de l'entraînement. Suivent l'âge des pneumatiques
(8 courses, relais plus longs) et la position (6 courses, compétitivité
différente de la voiture). Les variables de pluie ne sont signalées sur aucune
course, contrairement à ce que suggérait une lecture non calibrée ; le nombre de
tours restants, qui sert de contrôle de cohérence, ne l'est pas non plus.

Le rapport a été croisé avec les performances du modèle livré. Trois des quatre
courses où l'agent échoue nettement figurent parmi les cinq courses les plus
dérivées, et le potentiel capté médian est de 11 % sur les courses signalées sur
au moins trois variables, contre 92 % sur les autres. La corrélation de rang
n'est toutefois pas significative sur dix-neuf courses (ρ = −0,32 ; p = 0,18) :
le lien entre dérive et échec est une hypothèse cohérente, non une
démonstration.

### Z.4 Exploitation

Le rapport est produit automatiquement chaque mois par la chaîne d'intégration
continue et publié en artefact téléchargeable. Il ne bloque pas les
livraisons : une dérive entre saisons est attendue. Il fonde deux décisions :
une règle opérationnelle (une course signalée sur au moins trois variables est
traitée comme une décision à faible confiance, soumise à revue ou ramenée à la
stratégie de référence) et une orientation de réentraînement ciblée (enrichir
le pool en courses chaudes et à relais longs plutôt que réentraîner à
l'aveugle).
