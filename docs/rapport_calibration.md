# Calibration du modele de rythme (en secondes)

Donnees : 35,086 tours, 35 pilotes, 31 GP

## Pentes de degradation par GP (s/tour)

| GP | n tours | R2 | SOFT | MEDIUM | HARD | INTERMEDIATE | WET | residu (s) |
|---|---|---|---|---|---|---|---|---|
| 2021 Turkish Grand Prix | 1082 | 0.676 | - | - | - | 0.0246 | - | 0.848 |
| 2023 Abu Dhabi Grand Prix | 1064 | 0.868 | - | 0.0938 | 0.0602 | - | - | 0.386 |
| 2023 Bahrain Grand Prix | 894 | 0.851 | 0.1017 | - | 0.1096 | - | - | 0.493 |
| 2023 Dutch Grand Prix | 805 | 0.928 | 0.0346 | 0.0575 | 0.0238 | 0.0486 | - | 0.492 |
| 2023 Spanish Grand Prix | 1197 | 0.791 | 0.0426 | 0.0463 | 0.0373 | - | - | 0.536 |
| 2024 Canadian Grand Prix | 640 | 0.824 | - | 0.1060 | - | -0.0157 | - | 0.908 |
| 2025 Abu Dhabi Grand Prix | 1082 | 0.797 | - | 0.0764 | 0.0511 | - | - | 0.443 |
| 2025 Australian Grand Prix | 527 | 0.887 | - | - | - | -0.2450 | - | 0.791 |
| 2025 Austrian Grand Prix | 1005 | 0.769 | 0.0417 | 0.0545 | 0.0499 | - | - | 0.454 |
| 2025 Azerbaijan Grand Prix | 851 | 0.882 | - | 0.0116 | 0.0165 | - | - | 0.477 |
| 2025 Bahrain Grand Prix | 959 | 0.771 | 0.1127 | 0.1095 | 0.1253 | - | - | 0.498 |
| 2025 Belgian Grand Prix | 542 | 0.778 | - | 0.0915 | 0.1103 | - | - | 0.597 |
| 2025 British Grand Prix | 359 | 0.829 | - | - | - | 0.0076 | - | 0.953 |
| 2025 Canadian Grand Prix | 1197 | 0.746 | - | 0.0307 | 0.0226 | - | - | 0.521 |
| 2025 Chinese Grand Prix | 991 | 0.828 | - | 0.0346 | 0.0306 | - | - | 0.466 |
| 2025 Dutch Grand Prix | 1020 | 0.811 | 0.0179 | 0.0293 | 0.0257 | - | - | 0.490 |
| 2025 Emilia Romagna Grand Prix | 927 | 0.771 | - | 0.0455 | 0.0521 | - | - | 0.540 |
| 2025 Hungarian Grand Prix | 1289 | 0.767 | 0.0428 | 0.0516 | 0.0386 | - | - | 0.602 |
| 2025 Italian Grand Prix | 915 | 0.875 | - | 0.0181 | 0.0153 | - | - | 0.370 |
| 2025 Japanese Grand Prix | 997 | 0.902 | 0.0533 | 0.0444 | 0.0314 | - | - | 0.380 |
| 2025 Las Vegas Grand Prix | 754 | 0.811 | - | 0.0384 | 0.0315 | - | - | 0.496 |
| 2025 Mexico City Grand Prix | 1153 | 0.776 | 0.0550 | 0.0612 | 0.0491 | - | - | 0.561 |
| 2025 Miami Grand Prix | 449 | 0.691 | - | 0.1482 | 0.1142 | - | - | 0.577 |
| 2025 Monaco Grand Prix | 1264 | 0.587 | 0.0566 | 0.0398 | 0.0292 | - | - | 1.239 |
| 2025 Qatar Grand Prix | 905 | 0.907 | 0.0186 | 0.0050 | -0.0080 | - | - | 0.440 |
| 2025 Saudi Arabian Grand Prix | 809 | 0.830 | - | 0.0332 | 0.0262 | - | - | 0.518 |
| 2025 Singapore Grand Prix | 1154 | 0.649 | 0.0427 | 0.0409 | 0.0560 | - | - | 0.815 |
| 2025 Spanish Grand Prix | 986 | 0.820 | 0.0755 | 0.0724 | - | - | - | 0.578 |
| 2025 São Paulo Grand Prix | 1042 | 0.812 | 0.0655 | 0.0508 | 0.0676 | - | - | 0.411 |
| 2025 United States Grand Prix | 966 | 0.783 | 0.0585 | 0.0516 | 0.0797 | - | - | 0.504 |
| 2026 Japanese Grand Prix | 916 | 0.905 | - | 0.0457 | 0.0349 | - | - | 0.476 |

## Pentes retenues (mediane ponderee par le nombre de tours)

| compose | pente (s/tour) | n GP | plausible ? |
|---|---|---|---|
| SOFT | 0.0533 | 15 | oui |
| MEDIUM | 0.0463 | 27 | oui |
| HARD | 0.0373 | 26 | oui |
| INTERMEDIATE | 0.0246 | 5 | oui |
| WET | - | 0 | pas de donnees |

Ordre SOFT > MEDIUM > HARD respecte : **oui**

## Penalite de tour de sortie (pneu froid, age <= 2)

Estimee dans la meme regression que la pente, donc nette de l'effet carburant absorbe par l'effet fixe de tour.

| compose | penalite (s) | n GP | plausible ? |
|---|---|---|---|
| SOFT | -0.011 | 15 | **HORS PLAGE (0.2-4.0 s)** |
| MEDIUM | -0.267 | 27 | **HORS PLAGE (0.2-4.0 s)** |
| HARD | -0.093 | 26 | **HORS PLAGE (0.2-4.0 s)** |
| INTERMEDIATE | -0.005 | 5 | **HORS PLAGE (0.2-4.0 s)** |
| WET | - | 0 | pas de donnees |

## Bruit par tour

Ecart-type residuel median : **0.504 s**

Remplace `noise_std_units = 0.15` exprime en unites de gp_std, qui valait jusqu'a +/- 2.9 s sur les GP a gp_std eleve.
