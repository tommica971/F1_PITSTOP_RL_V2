# V2 — Index de degradation par circuit

Fenetre d'age retenue : 3-20 tours (zone fiable, cf. Phase 0)

- **2021 Turkish Grand Prix** : ecarte, moins de 120 tours exploitables par compose
- **2024 Canadian Grand Prix** : ecarte, moins de 120 tours exploitables par compose
- **2025 Australian Grand Prix** : ecarte, moins de 120 tours exploitables par compose
- **2025 Belgian Grand Prix** : ecarte, moins de 120 tours exploitables par compose
- **2025 British Grand Prix** : ecarte, moins de 120 tours exploitables par compose

Normalisation bornee aux percentiles 5/95 : [-0.0010, 0.1372] s/tour

## Classement par severite

| circuit | pente (s/tour) | index | n tours | detail par compose |
|---|---|---|---|---|
| 2023 Abu Dhabi Grand Prix | 0.1500 | **1.00** | 854 | HARD 0.0558, MEDIUM 0.3897 |
| 2025 Miami Grand Prix | 0.1416 | **1.00** | 206 | HARD 0.1416 |
| 2025 Bahrain Grand Prix | 0.1240 | **0.90** | 593 | HARD 0.1434, MEDIUM 0.1138 |
| 2023 Bahrain Grand Prix | 0.1212 | **0.88** | 790 | HARD 0.1123, SOFT 0.1314 |
| 2025 Spanish Grand Prix | 0.0889 | **0.65** | 850 | MEDIUM 0.0603, SOFT 0.1088 |
| 2025 Monaco Grand Prix | 0.0862 | **0.63** | 658 | HARD 0.0569, MEDIUM 0.1291 |
| 2025 Hungarian Grand Prix | 0.0842 | **0.62** | 778 | HARD 0.0424, MEDIUM 0.1354 |
| 2025 Emilia Romagna Grand Prix | 0.0802 | **0.59** | 719 | HARD 0.0983, MEDIUM 0.0436 |
| 2025 São Paulo Grand Prix | 0.0743 | **0.55** | 511 | MEDIUM 0.0743 |
| 2025 Saudi Arabian Grand Prix | 0.0730 | **0.54** | 334 | HARD 0.0730 |
| 2025 Chinese Grand Prix | 0.0628 | **0.46** | 636 | HARD 0.0630, MEDIUM 0.0624 |
| 2025 Italian Grand Prix | 0.0612 | **0.45** | 279 | HARD 0.0612 |
| 2023 Spanish Grand Prix | 0.0582 | **0.43** | 940 | HARD 0.0797, MEDIUM 0.0704, SOFT 0.0186 |
| 2025 Austrian Grand Prix | 0.0564 | **0.42** | 691 | HARD 0.0538, MEDIUM 0.0582 |
| 2025 Mexico City Grand Prix | 0.0535 | **0.39** | 711 | MEDIUM 0.0302, SOFT 0.0681 |
| 2025 United States Grand Prix | 0.0491 | **0.36** | 605 | MEDIUM 0.0628, SOFT 0.0356 |
| 2025 Azerbaijan Grand Prix | 0.0469 | **0.35** | 535 | HARD 0.0276, MEDIUM 0.0735 |
| 2025 Japanese Grand Prix | 0.0457 | **0.34** | 324 | HARD 0.0457 |
| 2025 Canadian Grand Prix | 0.0394 | **0.29** | 750 | HARD 0.0352, MEDIUM 0.0486 |
| 2025 Singapore Grand Prix | 0.0281 | **0.21** | 548 | HARD 0.0319, MEDIUM 0.0248 |
| 2025 Abu Dhabi Grand Prix | 0.0220 | **0.17** | 713 | HARD 0.0251, MEDIUM 0.0187 |
| 2025 Las Vegas Grand Prix | 0.0201 | **0.15** | 331 | HARD 0.0200 |
| 2023 Dutch Grand Prix | 0.0042 | **0.04** | 359 | SOFT 0.0042 |
| 2025 Dutch Grand Prix | 0.0035 | **0.03** | 452 | HARD 0.0495, SOFT -0.0661 |
| 2025 Qatar Grand Prix | -0.0025 | **0.00** | 474 | MEDIUM -0.0025 |
| 2026 Japanese Grand Prix | -0.9323 | **0.00** | 382 | MEDIUM -0.9323 |

## Test de plausibilite

- OK — Bahrain attendu ABRASIF, classe 3/26 (index 0.88)
- OK — Spanish attendu ABRASIF, classe 5/26 (index 0.43)
- OK — Hungarian attendu ABRASIF, classe 7/26 (index 0.62)
- OK — Las Vegas attendu PEU ABRASIF, classe 22/26 (index 0.15)
- **ECART** — Qatar attendu ABRASIF, classe 25/26 (index 0.00)
- **ECART** — Austrian attendu ABRASIF, classe 14/26 (index 0.42)
- **ECART** — Italian attendu PEU ABRASIF, classe 12/26 (index 0.45)
- **ECART** — Azerbaijan attendu PEU ABRASIF, classe 17/26 (index 0.35)
- **ECART** — Monaco attendu PEU ABRASIF, classe 6/26 (index 0.63)

**4/9 reperes conformes (44 %).**

**NE PAS INTEGRER.** L'index contredit la severite connue des circuits ; il mesure autre chose que l'abrasivite — rythme de la voiture, conditions du jour, ou artefact d'echantillon. Revoir la methode avant d'aller plus loin.

> Les reperes publics servent UNIQUEMENT au controle de plausibilite. Ils n'entrent pas dans le calcul de l'index, qui est derive des seules donnees de temps au tour.
