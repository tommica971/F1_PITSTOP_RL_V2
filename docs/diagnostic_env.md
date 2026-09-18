# Diagnostic de l'environnement F1PitStopEnv


## H1 - Un arret est-il rentable dans l'environnement actuel ?

Cout d'un arret : 23.23 s (moyenne calibree)

| GP | gp_std (s) | degr. HARD (s/tour) | degr. cumulee 40 tours (s) | surcout tours de sortie (s) | cout total d'un arret (s) | rentable ? |
|---|---|---|---|---|---|---|
| 2025 British Grand Prix | 19.15 | 0.1283 | 95.1 | 46.9 | 70.2 | OUI |
| 2025 Australian Grand Prix | 22.49 | 0.1507 | 111.6 | 55.1 | 78.3 | OUI |
| 2024 Canadian Grand Prix | 9.15 | 0.0613 | 45.4 | 22.4 | 45.7 | **NON** |
| 2023 Dutch Grand Prix | 14.02 | 0.0940 | 69.6 | 34.4 | 57.6 | OUI |
| 2021 Turkish Grand Prix | 3.34 | 0.0224 | 16.6 | 8.2 | 31.4 | **NON** |
| 2023 Bahrain Grand Prix | 4.73 | 0.0317 | 23.5 | 11.6 | 34.8 | **NON** |
| 2023 Spanish Grand Prix | 3.79 | 0.0254 | 18.8 | 9.3 | 32.5 | **NON** |
| 2023 Abu Dhabi Grand Prix | 4.09 | 0.0274 | 20.3 | 10.0 | 33.2 | **NON** |
| 2025 Belgian Grand Prix | 17.28 | 0.1157 | 85.8 | 42.3 | 65.6 | OUI |
| 2026 Japanese Grand Prix | 14.63 | 0.0980 | 72.6 | 35.8 | 59.1 | OUI |

**5/10 GP ou aucun arret n'est rentable.** Si ce nombre est eleve, le 0-stop est l'optimum de l'environnement et l'agent ne peut pas apprendre autre chose.

Ordre de grandeur reel en F1 : 0.05 a 0.12 s/tour de degradation slick. Comparer a la colonne 'degr. HARD (s/tour)' ci-dessus.


## H1bis - La pente calibree confond-elle pneu et carburant ?

Regression par GP : lap_time ~ tyre_age + lap_number, avec effets fixes de relais (on retire la moyenne de chaque relais).

- coef tyre_age  = degradation pneu, REINITIALISABLE par un arret
- coef lap_number = allegement carburant, NON reinitialisable


| GP | pente nette (s/tour) | coef pneu (s/tour) | coef carburant (s/tour) | n tours |
|---|---|---|---|---|
| 2025 British Grand Prix | -2.0055 | nan | nan | 39 |
| 2025 Australian Grand Prix | -1.0568 | nan | nan | 47 |
| 2024 Canadian Grand Prix | -0.4031 | nan | nan | 59 |
| 2023 Dutch Grand Prix | -0.8555 | nan | nan | 60 |
| 2021 Turkish Grand Prix | -0.0341 | nan | nan | 54 |
| 2023 Bahrain Grand Prix | 0.2367 | nan | nan | 47 |
| 2023 Spanish Grand Prix | 0.0346 | nan | nan | 61 |
| 2023 Abu Dhabi Grand Prix | 0.0293 | nan | nan | 50 |
| 2025 Belgian Grand Prix | -0.2031 | nan | nan | 39 |
| 2026 Japanese Grand Prix | -0.6314 | nan | nan | 47 |

## H2 - L'agent gagne-t-il du temps gratuit sous SC/VSC ?

Multiplicateurs de l'env : SC = 1.4, VSC = 1.03

| GP | type | n tours | mult. reel peloton | mult. env | ecart par tour (s) | ecart total (s) |
|---|---|---|---|---|---|---|
| 2025 British Grand Prix | SC | 6 | 1.315 | 1.400 | -8.91 | -53.5 |
| 2025 British Grand Prix | VSC | 4 | 1.122 | 1.030 | +9.63 | +38.5 |
| 2025 Australian Grand Prix | SC | 6 | 1.413 | 1.400 | +1.18 | +7.1 |
| 2024 Canadian Grand Prix | SC | 4 | 1.399 | 1.400 | -0.06 | -0.2 |
| 2023 Dutch Grand Prix | SC | 3 | 1.556 | 1.400 | +11.90 | +35.7 |
| 2023 Dutch Grand Prix | VSC | 1 | 32.948 | 1.030 | +2438.00 | +2438.0 |
| 2023 Bahrain Grand Prix | SC | 1 | 1.269 | 1.400 | -12.97 | -13.0 |
| 2023 Bahrain Grand Prix | VSC | 2 | 1.040 | 1.030 | +0.98 | +2.0 |
| 2026 Japanese Grand Prix | SC | 2 | 1.397 | 1.400 | -0.29 | -0.6 |

**Total sur le pool d'entrainement : +2454.0 s** offerts (positif) ou retires (negatif) a l'agent, sans lien avec ses decisions.

Rappel : 1 position vaut typiquement 2 a 6 s dans un peloton F1. Un ecart de quelques dizaines de secondes deplace l'agent de plusieurs rangs sans qu'aucune strategie ne soit en cause.
