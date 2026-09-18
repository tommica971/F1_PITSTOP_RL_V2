# Rapport de nettoyage des temps au tour

## Fichiers charges

- `features_dataset.parquet` : 1,748 lignes, 72 tours distincts
- `all_drivers_dataset.parquet` : 35,086 lignes, 78 tours distincts

Colonne d'arret explicite detectee : False
Colonnes disponibles (features) : ['air_temp_c', 'delta_pluie_3tours', 'driver', 'event', 'fenetre_undercut', 'gap_arriere', 'gap_avant', 'humidity_pct', 'is_deleted', 'is_fresh_tyre', 'is_inaccurate', 'is_raining', 'is_red_flag', 'is_safety_car', 'is_vsc', 'known_issues', 'lap_number', 'lap_time_s', 'position', 'pressure_hpa', 'rival_arriere_drv', 'rival_arriere_pitte_ce_tour', 'rival_arriere_vient_de_pitter', 'rival_avant_drv', 'rival_avant_pitte_ce_tour', 'rival_avant_vient_de_pitter', 'role', 'season', 'stint_number', 'team', 'tours_restants', 'track_temp_c', 'tyre_age_laps', 'tyre_compound', 'usable_for_reward', 'wind_direction_deg', 'wind_speed_kmh']

## Motifs d'exclusion (pilote de reference)

| motif | tours |
|---|---|
| CONSERVE | 1376 |
| pneu froid (age <= 2) | 80 |
| hors bornes (> 107%) | 79 |
| safety car | 51 |
| tour de sortie | 50 |
| tour d'arret | 42 |
| tour 1 (artefact sesT) | 30 |
| hors bornes (< 93%) | 28 |
| VSC | 12 |

**1376 / 1748 tours conserves (79%).**

## gp_std AVANT / APRES

| GP | gp_std avant (s) | gp_std apres (s) | baseline apres (s) | n tours | verdict |
|---|---|---|---|---|---|
| 2021 Turkish Grand Prix | 3.34 | 0.82 | 94.29 | 53 | OK |
| 2023 Abu Dhabi Grand Prix | 4.09 | 0.84 | 90.24 | 50 | OK |
| 2023 Bahrain Grand Prix | 4.73 | 1.48 | 98.12 | 46 | OK |
| 2023 Dutch Grand Prix | 14.02 | 1.16 | 75.83 | 38 | OK |
| 2023 Spanish Grand Prix | 3.79 | 1.16 | 80.14 | 59 | OK |
| 2024 Canadian Grand Prix | 9.15 | 2.80 | 88.43 | 31 | **HORS CIBLE** |
| 2025 Abu Dhabi Grand Prix | 4.25 | 0.98 | 89.88 | 50 | OK |
| 2025 Australian Grand Prix | 22.49 | 2.28 | 92.34 | 34 | OK |
| 2025 Austrian Grand Prix | 5.68 | 0.70 | 71.14 | 60 | OK |
| 2025 Azerbaijan Grand Prix | 11.03 | 1.46 | 107.28 | 43 | OK |
| 2025 Bahrain Grand Prix | 6.90 | 0.82 | 98.20 | 46 | OK |
| 2025 Belgian Grand Prix | 17.28 | 0.91 | 107.86 | 31 | OK |
| 2025 British Grand Prix | 19.15 | 2.15 | 104.31 | 22 | OK |
| 2025 Canadian Grand Prix | 8.04 | 0.99 | 77.68 | 60 | OK |
| 2025 Chinese Grand Prix | 2.66 | 1.00 | 98.01 | 51 | OK |
| 2025 Dutch Grand Prix | 10.36 | 0.65 | 76.70 | 55 | OK |
| 2025 Emilia Romagna Grand Prix | 12.26 | 0.93 | 81.63 | 46 | OK |
| 2025 Hungarian Grand Prix | 2.29 | 0.85 | 83.39 | 64 | OK |
| 2025 Italian Grand Prix | 2.98 | 0.83 | 84.12 | 47 | OK |
| 2025 Japanese Grand Prix | 3.96 | 1.21 | 92.80 | 48 | OK |
| 2025 Las Vegas Grand Prix | 6.27 | 0.59 | 97.00 | 44 | OK |
| 2025 Mexico City Grand Prix | 3.58 | 1.03 | 84.10 | 63 | OK |
| 2025 Miami Grand Prix | 1.77 | 0.68 | 91.59 | 26 | OK |
| 2025 Monaco Grand Prix | 13.37 | - | - | 3 | **TROP PEU DE TOURS** |
| 2025 Qatar Grand Prix | 8.56 | 1.24 | 87.37 | 47 | OK |
| 2025 Singapore Grand Prix | 3.47 | 1.37 | 98.66 | 53 | OK |
| 2025 Spanish Grand Prix | 7.47 | 1.18 | 81.68 | 53 | OK |
| 2025 São Paulo Grand Prix | 6.76 | 0.77 | 74.80 | 59 | OK |
| 2025 United States Grand Prix | 3.09 | 0.79 | 101.11 | 49 | OK |
| 2026 Japanese Grand Prix | 14.63 | 0.92 | 94.46 | 45 | OK |

**Critere global : NON ATTEINT** (cible : gp_std entre 0.3 et 2.5 s partout)

> Si un GP reste hors cible, inspecter ses tours conserves avant de passer a l'etape 2 : la calibration heritera du probleme.

## Nettoyage des temps aberrants (fantomes)

639 temps au tour aberrants detectes (1.82% des lignes).

| GP | pilote | tour | temps (s) | ratio mediane |
|---|---|---|---|---|
| 2025 Belgian Grand Prix | ALO | 1 | 8291.1 | 76.7x |
| 2025 Belgian Grand Prix | ANT | 1 | 8288.8 | 76.7x |
| 2025 Belgian Grand Prix | HAM | 1 | 8286.0 | 76.7x |
| 2025 Belgian Grand Prix | SAI | 1 | 8283.9 | 76.7x |
| 2025 Belgian Grand Prix | STR | 1 | 8281.5 | 76.6x |
| 2025 Belgian Grand Prix | COL | 1 | 8279.9 | 76.6x |
| 2025 Belgian Grand Prix | HUL | 1 | 8279.0 | 76.6x |
| 2025 Belgian Grand Prix | GAS | 1 | 8276.8 | 76.6x |

639 temps remplaces par la mediane du peloton au meme tour.
