# V2 — Deux extensions du modèle abandonnées après mesure

> Section destinée au dossier RNCP Bloc 5.
> Tous les chiffres proviennent de scripts du projet, indiqués entre parenthèses.

## Pourquoi cette section existe

La V2 devait étendre le modèle de rythme sur deux points : une dégradation
pneumatique non linéaire, et un indicateur de sévérité abrasive propre à chaque
circuit. Les deux ont été **mesurés avant d'être implémentés**, et les deux ont
été abandonnés parce que les données ne les supportent pas.

Ce choix mérite d'être exposé plutôt que masqué. Alimenter un agent
d'apprentissage par renforcement avec un coefficient dont on sait qu'il mesure
un biais d'échantillonnage plutôt qu'un phénomène physique produirait une
politique apprise sur du bruit. Les deux refus reposent sur des mécanismes
statistiques différents, et ils convergent vers la même conclusion : **les
données disponibles permettent une calibration agrégée, pas une calibration à
granularité fine.**

---

## 1. Dégradation non linéaire — censure informative

### 1.1 L'hypothèse de départ

La calibration V1 mesurait la pente de dégradation sur des fenêtres d'âge
**cumulatives** :

| fenêtre | pente MEDIUM |
|---|---|
| âges 3-10 | 0,0737 s/tour |
| âges 3 et plus | 0,0429 s/tour |

L'écart suggérait une dégradation qui ralentit avec l'âge du pneu, donc une
forme logarithmique plutôt que linéaire. Mais ces deux valeurs ne sont pas
comparables : la seconde **inclut** la première. Les utiliser comme deux
segments d'une fonction par morceaux aurait consisté à prendre une moyenne
globale pour une pente de fin de relais.

### 1.2 La mesure sur fenêtres disjointes

`scripts/v2_00_measure_degradation_curve.py` reprend la méthode de calibration
— effets fixes (GP × tour) et (GP × pilote) retirés par projections alternées,
donc nets de l'effet carburant et de l'évolution de piste — mais sur des
fenêtres d'âge **disjointes**, sur 28 904 tours.

| composé | 3-10 | 11-20 | 21-30 | 31-45 | 46+ |
|---|---|---|---|---|---|
| SOFT | 0,0871 (1627) | 0,0583 (1578) | 0,0537 (578) | – (100) | – (4) |
| MEDIUM | 0,0977 (3703) | 0,0341 (4015) | 0,0366 (1736) | **0,1079** (433) | – (37) |
| HARD | 0,0576 (3419) | 0,0566 (4077) | **−0,0122** (2729) | **−0,0407** (1255) | **−0,0842** (210) |

*(entre parenthèses : nombre de tours ayant servi à l'estimation)*

### 1.3 Pourquoi ces valeurs sont inexploitables

Deux résultats sont physiquement impossibles.

**Les pentes HARD deviennent négatives et le restent**, de plus en plus : un
pneu qui accélérerait en vieillissant, à un rythme croissant.

**MEDIUM décrit une courbe en U** : 0,0977, puis 0,0341, puis 0,1079 — une
remontée au-dessus de la valeur initiale qui ne correspond à aucun comportement
pneumatique connu.

### 1.4 Le mécanisme : une censure informative

Les relais longs ne constituent pas un échantillon aléatoire. **Un pneu
n'atteint 30 ou 40 tours que lorsque tout va bien** : circuit peu abrasif,
pilotage économe, absence de trafic, conditions fraîches. Dès qu'un relais se
dégrade vite, l'équipe s'arrête — et ces tours n'existent jamais dans la
fenêtre 31-45.

Les survivants d'une fenêtre d'âge élevée sont donc sélectionnés sur leur faible
dégradation, ce qui tire mécaniquement la pente vers le bas. L'effet s'aggrave
avec l'âge, d'où des valeurs de plus en plus négatives sur HARD.

Les effets fixes ne corrigent pas ce biais : ils retirent ce qui est commun au
peloton à un instant donné, pas le mécanisme de sélection des observations
elles-mêmes.

### 1.5 Conséquence de l'implémenter malgré tout

Une courbe ajustée sur ces valeurs ferait **accélérer le pneu au-delà de 30
tours**. Dans l'environnement, rester en piste deviendrait gratuit puis
avantageux — ce qui ramènerait exactement au problème diagnostiqué en V1, où le
0-stop était l'optimum et où aucun agent ne s'arrêtait jamais.

### 1.6 Décision

La dégradation reste **linéaire, calibrée sur la fenêtre fiable 3-20 tours**,
où les effectifs vont de 1 578 à 4 077 tours et où les trois composés décroissent
de façon cohérente : SOFT 0,0871 → 0,0583, MEDIUM 0,0977 → 0,0341, HARD 0,0576 →
0,0566.

Corriger le biais demanderait un modèle de survie ou une pondération inverse de
la probabilité de survie du relais — un chantier à part entière, hors périmètre.

---

## 2. Index de dégradation par circuit — biais de confusion

### 2.1 L'hypothèse de départ

Sur 19 Grands Prix hors pool, **12 ne sont battus par aucun algorithme et aucune
graine** face à l'oracle. Monza est le cas d'école : les trois algorithmes s'y
abstiennent identiquement alors que l'écart entre 0-stop et oracle y vaut 39,8
points.

L'explication avancée était que l'espace d'observation ne contient aucune
variable décrivant la sévérité abrasive du tracé : l'agent doit l'inférer de
l'âge du pneu seul, et il y échoue sur les profils qu'il n'a pas rencontrés.

Un indicateur scalaire statique par circuit devait combler ce manque.

### 2.2 Le protocole, et son garde-fou

`scripts/v2_02_measure_circuit_index.py` calcule une pente par circuit sur la
fenêtre 3-20 tours, pondérée par effectif sur les trois composés secs, puis la
normalise entre les percentiles 5 et 95 avec troncature à [0, 1].

**Un test de plausibilité a été posé avant le calcul** : le classement obtenu
doit être cohérent avec la sévérité connue du sport. Cinq circuits notoirement
abrasifs (Bahreïn, Espagne, Qatar, Autriche, Hongrie) et cinq qui ne le sont pas
(Monza, Bakou, Las Vegas, Spa, Monaco) servent de repères. Ils n'entrent pas
dans le calcul. Seuil d'acceptation : 60 % de repères conformes.

### 2.3 Résultat

**4 repères sur 9 conformes, soit 44 %.** Le seuil n'est pas atteint.

Trois valeurs invalident l'index à elles seules :

| observation | valeur | commentaire |
|---|---|---|
| Japon 2026, MEDIUM | **−0,9323 s/tour** | le pneu gagnerait presque une seconde par tour d'usure, sur 382 tours |
| Abu Dhabi 2023, MEDIUM | **0,3897 s/tour** | dix fois la pente globale calibrée ; le même circuit en 2025 donne 0,0187, soit un rapport de 20 entre deux éditions |
| Pays-Bas 2025 | HARD **+0,0495**, SOFT **−0,0661** | deux composés de signes opposés sur la même course |

Et le classement contredit directement la connaissance du sport : **Monaco
ressort 6ᵉ sur 26**, plus abrasif que l'Autriche (14ᵉ), tandis que le Qatar,
circuit à forte contrainte pneumatique, arrive 25ᵉ.

### 2.4 Le mécanisme : trois effets confondus

La mesure par circuit repose sur 200 à 900 tours, contre 28 904 pour la
régression globale. À ce volume, trois effets deviennent indissociables :

**L'écurie.** Les effets fixes par pilote absorbent le rythme moyen de chaque
voiture, mais pas sa dégradation propre. Si les écuries à forte usure roulent
majoritairement en MEDIUM sur un Grand Prix donné, la pente MEDIUM capte la
différence entre écuries, pas l'abrasivité du tracé.

**L'allocation Pirelli.** Les étiquettes SOFT, MEDIUM et HARD sont relatives à
l'allocation de chaque week-end : le manufacturier choisit trois composés dans
une gamme C1 à C6. Le MEDIUM de Monza et celui de Monaco ne sont pas le même
pneu. C'est ce qui explique l'écart d'un facteur 20 entre deux éditions d'Abu
Dhabi.

**Les conditions du jour.** Température de piste, évolution du grip, incidents
de course varient d'une édition à l'autre du même tracé.

Ce que l'index mesurait n'était donc pas la sévérité intrinsèque du circuit,
mais le produit **circuit × composé alloué × écurie × conditions**.

### 2.5 Conséquence de l'intégrer malgré tout

L'agent aurait reçu, sur les circuits jamais vus, un signal statique le
désignant comme plus ou moins abrasif — avec Monaco annoncé plus sévère que
l'Autriche. Sur les Grands Prix mal classés, cet indicateur aurait
**activement dérouté** la décision d'arrêt plutôt que de l'informer.

### 2.6 Décision

L'index n'est pas intégré à l'observation. La cause des 12 Grands Prix hors de
portée reste identifiée — absence d'information sur la dégradation propre au
circuit — mais elle n'est pas corrigeable avec les données disponibles.

**La voie réaliste** serait de construire l'indicateur depuis les allocations
Pirelli C1-C6 par Grand Prix, qui encodent directement la sévérité anticipée
par le manufacturier et sont publiées avant chaque course. Cette donnée n'est
pas dans le jeu de données du projet ; c'est la première perspective
d'amélioration du modèle physique.

---

## 3. Ce que ces deux refus établissent

Les deux mécanismes diffèrent — censure informative dans un cas, confusion de
facteurs dans l'autre — mais la conclusion converge :

> Les données de temps au tour, agrégées sur 31 Grands Prix et 20 pilotes,
> permettent une calibration **globale** fiable de la dégradation pneumatique.
> Elles ne permettent pas de descendre au niveau du circuit ni à celui des
> relais longs, où le volume chute et où les effets de sélection et de
> confusion dominent le signal.

Cette limite explique directement pourquoi une part des Grands Prix reste hors
de portée de l'agent, et elle borne ce qu'il est raisonnable d'attendre du
modèle sans données supplémentaires.

**Ce que la démarche a permis d'éviter** : deux extensions qui auraient dégradé
l'environnement au lieu de l'améliorer, avec dans le premier cas un retour
probable au comportement 0-stop diagnostiqué en V1.

### Précédent dans le projet

Une troisième extension avait déjà été écartée sur le même principe, mais
**après** implémentation plutôt qu'avant : élargir la borne de `tyre_age_laps`
de 45 à 80 tours dans l'espace d'observation, pour supprimer une saturation
mesurée sur deux Grands Prix. Le résultat a fait chuter la généralisation de
10/19 à 4/19 Grands Prix, et le correctif a été annulé.

La différence de coût entre les deux approches est instructive : l'ablation de
la borne a demandé neuf entraînements pour aboutir à un refus, les deux refus
documentés ici n'ont demandé que deux scripts de mesure. **Mesurer avant
d'implémenter est le principal enseignement méthodologique de la V2.**
