# Décomposition du gain de points — saison 2025

Trois totaux, calculés à partir des résultats **déterministes** (politique la plus probable, reproductible — pas un tirage stochastique isolé) :

- **Total réel de Gasly (saison)** : 18 pts
- **Total agent, brut** (23 GP traités) : 181 pts
- **Total agent, sous-ensemble "propre"** (9 GP sans saut de position détecté ni variance élevée entre runs) : 50 pts
- **Total agent, sous-ensemble "suspect"** (14 GP avec au moins un saut de position ≥5 places en un tour, ou un écart-type ≥3 positions sur 5 tirages stochastiques) : 131 pts

Le sous-ensemble "suspect" corrèle systématiquement avec les tours suivant une période de Safety Car/VSC (cf. analyse Miami/São Paulo/Monaco) : le modèle de rythme calibré en Phase 2.2 traite la sortie de SC/VSC comme un retour instantané au rythme normal, alors qu'en réalité le peloton reste resserré 1 à 2 tours de plus. Le sous-total "propre" est la mesure la plus défendable de la capacité stratégique green-flag de l'agent.

## Détail par Grand Prix

| GP | Statut | Classe | Réel | Agent (déterministe) | Sauts détectés | Écart-type stochastique |
|---|---|---|---|---|---|---|
| Australian Grand Prix | Entraînement | propre | 0 | 25 | 0 | 0.0 |
| Belgian Grand Prix | Généralisation | propre | 1 | 25 | 0 | 0.0 |
| Abu Dhabi Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.0 |
| Canadian Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.0 |
| Hungarian Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.0 |
| Italian Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.0 |
| Japanese Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.8 |
| Mexico City Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.0 |
| Singapore Grand Prix | Généralisation | propre | 0 | 0 | 0 | 0.0 |
| British Grand Prix | Entraînement | suspect | 8 | 25 | 7 | 0.8 |
| Emilia Romagna Grand Prix | Généralisation | suspect | 0 | 25 | 2 | 0.0 |
| São Paulo Grand Prix | Généralisation | suspect | 1 | 25 | 1 | 0.0 |
| Spanish Grand Prix | Généralisation | suspect | 2 | 25 | 1 | 0.0 |
| Bahrain Grand Prix | Généralisation | suspect | 6 | 18 | 2 | 1.4 |
| Monaco Grand Prix | Généralisation | suspect | 0 | 12 | 1 | 3.4 |
| Dutch Grand Prix | Généralisation | suspect | 0 | 1 | 10 | 5.9 |
| Austrian Grand Prix | Généralisation | suspect | 0 | 0 | 1 | 2.3 |
| Azerbaijan Grand Prix | Généralisation | suspect | 0 | 0 | 0 | 3.2 |
| Chinese Grand Prix | Généralisation | suspect | 0 | 0 | 1 | 0.0 |
| Las Vegas Grand Prix | Généralisation | suspect | 0 | 0 | 2 | 0.0 |
| Miami Grand Prix | Généralisation | suspect | 0 | 0 | 1 | 0.0 |
| Qatar Grand Prix | Généralisation | suspect | 0 | 0 | 4 | 2.5 |
| United States Grand Prix | Généralisation | suspect | 0 | 0 | 2 | 0.0 |