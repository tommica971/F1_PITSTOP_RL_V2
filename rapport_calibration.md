# Calibration du modele de rythme (en secondes)

Donnees : 35,086 tours, 35 pilotes, 31 GP

Tours exploitables apres nettoyage : 28,904

## Profil observe par age de pneu (% de la mediane du GP)

| compose | age 2 | 3-5 | 6-10 | 11-15 | 16-25 | 26-40 |
|---|---|---|---|---|---|---|
| SOFT | 100.96 | 100.30 | 99.96 | 99.92 | 99.88 | 99.88 |
| MEDIUM | 100.61 | 100.41 | 100.37 | 100.38 | 100.22 | 100.24 |
| HARD | 99.62 | 99.74 | 99.81 | 99.77 | 99.60 | 99.63 |

Un profil non lineaire justifie le contraste local pour le pneu froid (cf. cold_penalty_local).

## Pentes de degradation retenues (estimation conjointe)

Regression unique sur 28,904 tours, effets fixes (GP x tour) et (GP x pilote). R2 = 0.129

| compose | pente (s/tour) | n tours | plausible ? |
|---|---|---|---|
| SOFT | 0.0419 | 4012 | oui |
| MEDIUM | 0.0429 | 10312 | oui |
| HARD | 0.0356 | 12034 | oui |
| INTERMEDIATE | 0.0187 | 2546 | oui |
| WET | - | 0 | pas de donnees |

Ordre SOFT > MEDIUM > HARD respecte : **NON — a investiguer avant de continuer**

## Penalite de pneu froid (age = 2)

Contraste LOCAL : tour d'age 2 compare aux tours d'age 3-5 du MEME relais, corriges de la degradation attendue. Insensible a la courbure de la degradation, au carburant et a l'evolution de piste.

_(ancienne methode, abandonnee : residu MEDIAN des tours d'age 2 par rapport au modele ajuste sur les tours chauds (effets fixes pilote + tour + pente). Estimer un coefficient dedie echouait : sur quelques dizaines de lignes, l'indicatrice devient quasi colineaire avec les effets fixes de tour et le coefficient explose. Les residus sont mutualises sur tous les GP.

> **Borne basse assumee.** Le tour de sortie (age = 1) porte le temps de passage au stand (mediane 122.9% de la reference, cf. 04_diagnose_outlap.py) et ne permet pas d'isoler l'effet thermique. La penalite est donc estimee sur l'age 2 seul, ou le pneu est deja partiellement monte en temperature. La vraie penalite a l'age 1 est superieure. Consequence : le modele SOUS-ESTIME legerement le cout d'un arret, ce qui joue en faveur des strategies a arret. A citer comme limite dans le dossier.

| compose | penalite (s) | n tours | plausible ? |
|---|---|---|---|
| SOFT | 0.490 | - | oui |
| MEDIUM | 0.386 | - | oui |
| HARD | 0.097 | - | **HORS PLAGE (0.2-4.0 s)** |
| INTERMEDIATE | -0.126 | - | **HORS PLAGE (0.2-4.0 s)** |
| WET | - | - | trop peu de relais exploitables |

## Bruit par tour

Ecart-type residuel median : **0.608 s**

Remplace `noise_std_units = 0.15` exprime en unites de gp_std, qui valait jusqu'a +/- 2.9 s sur les GP a gp_std eleve.
