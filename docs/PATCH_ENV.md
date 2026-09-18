# Étape 4 — Correctifs à appliquer dans `f1_pitstop_env.py`

Trois modifications. Chacune est donnée en avant/après exact.

---

## 4.1 — Utiliser le filtre reconstruit

`_load_gp_data`, calcul du `GPPaceContext`.

**Avant**
```python
usable = gasly[gasly["usable_for_reward"]]
```

**Après**
```python
# usable_for_reward laissait passer les tours de SC, d'arrêt et le tour 1,
# d'où un gp_std de 3 à 22 s selon le GP (cf. diagnostic_env.md).
# usable_for_reward_v2 est reconstruit par scripts/01_clean_laps.py.
flag = "usable_for_reward_v2" if "usable_for_reward_v2" in gasly.columns else "usable_for_reward"
usable = gasly[gasly[flag]]
```

---

## 4.2 — Safety Car : le rythme réel du peloton, plus un multiplicateur assumé

C'est le correctif qui supprime les gains de position gratuits.

Dans la v1, sous neutralisation, l'agent reçoit `gp_baseline_s * multiplicateur`,
une constante déterministe, pendant que les 19 fantômes rejouent leurs **vrais**
temps. Le diagnostic mesure l'écart réel : le peloton tourne à 1,32× à
Silverstone et 1,56× aux Pays-Bas, contre 1,40× fixe dans l'env. Chaque tour de
SC offre donc ±10 s à l'agent, sans lien avec ses décisions.

La correction consiste à prendre la médiane des temps réels du peloton à ce
tour précis — donnée déjà présente dans `all_drivers_dataset.parquet`.

**À ajouter dans `_load_gp_data`**, après la construction de `self.ghost_laps` :

```python
# Rythme réel du peloton tour par tour, pour les tours neutralisés.
# Remplace SC_PACE_MULTIPLIER / VSC_PACE_MULTIPLIER, qui étaient des
# hypothèses assumées : l'agent subit désormais exactement le même
# ralentissement que ses adversaires, ni plus ni moins.
field = ghosts[ghosts["lap_time_s"] > 0]
self.field_median_lap = (
    field.groupby("lap_number")["lap_time_s"].median().to_dict()
)
```

**Dans `step()`**, remplacer le bloc de calcul du temps au tour :

**Avant**
```python
if under_caution and not is_pit:
    multiplier = SC_PACE_MULTIPLIER if weather["is_safety_car"] else VSC_PACE_MULTIPLIER
    lap_time = self.pace_ctx.gp_baseline_s * multiplier
elif under_caution and is_pit:
    lap_time = self.pace_ctx.gp_baseline_s
else:
    lap_time = predict_lap_time(
        context=self.pace_ctx,
        tyre_compound=self.own_tyre_compound,
        tyre_age_laps=self.own_tyre_age,
        is_raining=weather["is_raining"],
        rain_intensity_recent=max(weather["delta_pluie_3tours"], 0.0) + float(weather["is_raining"]) * 0.3,
        rng=self.rng,
    )
```

**Après**
```python
if under_caution:
    # Rythme réel médian du peloton à ce tour (données, pas hypothèse).
    # Repli sur l'ancien multiplicateur si le tour est absent des données.
    fallback_mult = SC_PACE_MULTIPLIER if weather["is_safety_car"] else VSC_PACE_MULTIPLIER
    lap_time = self.field_median_lap.get(
        self.current_lap, self.pace_ctx.gp_baseline_s * fallback_mult
    )
    if is_pit:
        # S'arrêter sous neutralisation est moins coûteux relativement : on
        # ne cumule pas le ralentissement du peloton avec le coût de pit
        # complet (bug de double comptage, test de sanité Phase 2.2).
        lap_time = self.pace_ctx.gp_baseline_s
else:
    lap_time = predict_lap_time(
        context=self.pace_ctx,
        tyre_compound=self.own_tyre_compound,
        tyre_age_laps=self.own_tyre_age,
        is_raining=weather["is_raining"],
        rain_intensity_recent=max(weather["delta_pluie_3tours"], 0.0) + float(weather["is_raining"]) * 0.3,
        rng=self.rng,
    )
```

Conserve `SC_PACE_MULTIPLIER` et `VSC_PACE_MULTIPLIER` en haut du module : ils
ne servent plus que de repli, et leur présence documente la démarche dans le
dossier (hypothèse assumée → valeur mesurée).

---

## 4.3 — Nettoyage des temps fantômes aberrants

`01_clean_laps.py` corrige déjà le fichier Parquet, mais un garde-fou dans
l'env évite qu'un artefact futur ne réintroduise le problème silencieusement.

**Dans `step()`**, boucle de rejeu des fantômes :

**Avant**
```python
if ghost_lap_time is not None and ghost_lap_time > 0:
    self.ghost_cum_time[drv] += ghost_lap_time
```

**Après**
```python
# Garde-fou : un temps > 3x la référence n'est pas un tour de course, c'est
# un artefact de données (temps de session cumulé pris pour un temps au
# tour). Observé aux Pays-Bas 2023 : un tour à 32,9x la baseline ajoutait
# ~2400 s au temps cumulé du fantôme, plaçant l'agent P1 mécaniquement.
MAX_PLAUSIBLE = 3.0 * self.pace_ctx.gp_baseline_s
if ghost_lap_time is not None and 0 < ghost_lap_time < MAX_PLAUSIBLE:
    self.ghost_cum_time[drv] += ghost_lap_time
elif ghost_lap_time is not None and ghost_lap_time >= MAX_PLAUSIBLE:
    self.ghost_cum_time[drv] += self.field_median_lap.get(
        self.current_lap, self.pace_ctx.gp_baseline_s
    )
```

Attention : le `else` existant, qui gère les abandons, doit rester atteignable
uniquement quand `ghost_lap_time is None` ou `<= 0`. Vérifie l'enchaînement des
branches après édition.

---

## Ce que le correctif ne règle pas

Le bruit par tour passe de ±2,9 s à ±0,43 s, donc la marche aléatoire tombe de
~20 s à ~3 s sur une course. C'est le bon ordre de grandeur, mais **la variance
de position ne disparaît pas** : le coût d'arrêt reste tiré dans une loi
d'écart-type 5,38 s, ce qui est calibré et légitime. Attends-toi encore à des
écarts de 1 à 2 positions entre deux graines. C'est de la variance réelle du
sport, pas un bug — mais ça veut dire qu'une évaluation sur une seule graine
ne vaut rien. Évalue sur 20 graines minimum et rapporte moyenne et écart-type.

---

## Ordre d'exécution

```
python scripts/01_clean_laps.py        # -> rapport_nettoyage.md
python scripts/02_calibrate_pace.py    # -> pace_calibration.json
# remplacer src/f1_pitstop_rl/env/pace_model.py par la v2
# appliquer 4.1, 4.2, 4.3 sur f1_pitstop_env.py
python scripts/03_check_breakeven.py   # verrou : ne pas continuer si 0-stop gagne partout
# rejouer le test de sanité Bahreïn 2023
# réentraîner
```

Les points de contrôle, dans l'ordre :

1. `gp_std` entre 0,5 et 2,5 s sur les 10 GP
2. dégradation entre 0,02 et 0,20 s/tour, ordre SOFT > MEDIUM > HARD respecté
3. le test d'oracle donne une stratégie à arrêt gagnante sur la majorité des GP
4. le test de sanité Bahreïn 2023 reste dans sa tolérance (~0,2 % sur le temps agrégé)

Si le point 3 échoue, ne réentraîne pas : le problème est encore dans la
calibration, et un entraînement de plus ne fera que le confirmer plus cher.
