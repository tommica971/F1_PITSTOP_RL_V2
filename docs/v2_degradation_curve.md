# V2 — Forme de la courbe de degradation

28,904 tours exploitables

## Pentes locales sur fenetres d'age DISJOINTES (s/tour)

| compose | 3-10 | 11-20 | 21-30 | 31-45 | 46-+ |
|---|---|---|---|---|---|
| SOFT | **0.0871** (1627) | **0.0583** (1578) | **0.0537** (578) | – (100) | – (4) |
| MEDIUM | **0.0977** (3703) | **0.0341** (4015) | **0.0366** (1736) | **0.1079** (433) | – (37) |
| HARD | **0.0576** (3419) | **0.0566** (4077) | **-0.0122** (2729) | **-0.0407** (1255) | **-0.0842** (210) |

_(entre parentheses : nombre de tours ayant servi a l'estimation)_

## Forme de la courbe

- **SOFT** : 0.0871 -> 0.0537 s/tour (rapport 0.62) — DECELERE — la degradation ralentit avec l'age du pneu
- **MEDIUM** : 0.0977 -> 0.1079 s/tour (rapport 1.10) — LINEAIRE — pente stable, le modele actuel est adapte
- **HARD** : 0.0576 -> -0.0842 s/tour (rapport -1.46) — DECELERE — la degradation ralentit avec l'age du pneu

## Lecture et suite

Les composes n'ont pas la meme forme de courbe. Modeliser chacun separement, ou retenir le lineaire par morceaux qui s'adapte aux trois sans imposer de forme fonctionnelle commune.

> **Attention aux fenetres a faible effectif.** Une pente estimee sur moins de 300 tours est peu fiable ; les relais tres longs (46+) ne sont observes que sur les strategies a 0 ou 1 arret, donc sur un echantillon biaise vers les circuits a faible degradation.
