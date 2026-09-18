# Patch ENV v3 — référence de rythme par tour

**Ce patch remplace la section 4.2 de `PATCH_ENV.md`.** Si tu as déjà appliqué
4.2 (le bloc `field_median_lap` sous neutralisation), ce patch le généralise :
la même idée, étendue à *tous* les tours et plus seulement aux tours de SC.

Les sections 4.1 (`usable_for_reward_v2`) et 4.3 (garde-fou sur les temps
aberrants) restent valables telles quelles.

Motivation : le test de sanité montre des courses de pluie simulées 1,5 à
8,3 % trop rapides, parce que `gp_baseline_s` est devenu — à juste titre — la
médiane des tours verts et propres, sans qu'aucun terme ne représente l'état
de la piste. Voir l'en-tête de `pace_model_v3.py` pour le détail.

---

## 1. Installer le modèle v3

```cmd
copy src\f1_pitstop_rl\env\pace_model.py src\f1_pitstop_rl\env\pace_model_v2.py.bak
copy pace_model_v3.py src\f1_pitstop_rl\env\pace_model.py
```

## 2. Importer la nouvelle fonction

**Avant**
```python
from pace_model import GPPaceContext, predict_lap_time, sample_pit_stop_cost
```

**Après**
```python
from pace_model import (GPPaceContext, predict_lap_time, sample_pit_stop_cost,
                        build_lap_reference)
```

## 3. Construire la référence par tour

Dans `_load_gp_data`, au niveau du calcul du `GPPaceContext`.

**Avant**
```python
usable = gasly[gasly["usable_for_reward"]]
self.pace_ctx = GPPaceContext(
    gp_baseline_s=float(usable["lap_time_s"].median()),
    gp_std_s=float(usable["lap_time_s"].std()),
    race_total_laps=self.race_total_laps,
)
```

**Après**
```python
flag = ("usable_for_reward_v2" if "usable_for_reward_v2" in gasly.columns
        else "usable_for_reward")
usable = gasly[gasly[flag]]

# Référence de rythme tour par tour, dérivée du peloton réel. Absorbe
# pluie, Safety Car, VSC, séchage de piste et évolution de grip, sans
# aucune hypothèse — là où la v2 n'avait aucun terme pour l'état de la
# piste une fois `usable_for_reward` nettoyé.
lap_reference = build_lap_reference(self._ghosts_df, season, event)

self.pace_ctx = GPPaceContext(
    gp_baseline_s=float(usable["lap_time_s"].median()),
    gp_std_s=float(usable["lap_time_s"].std()),
    race_total_laps=self.race_total_laps,
    lap_reference_s=lap_reference,
)
```

Note : `self._ghosts_df` contient tous les pilotes, agent compris. C'est
volontaire — on veut la médiane du peloton entier, pas des 19 adversaires.

## 4. Remplacer le bloc de calcul du temps au tour

Dans `step()`. Si tu as appliqué le patch 4.2, remplace le bloc que tu avais
mis en place. Sinon, remplace le bloc d'origine.

**Après**
```python
# Plus de multiplicateur SC/VSC : la référence du tour porte déjà le
# ralentissement réel du peloton, mesuré au lieu d'être supposé.
if under_caution and is_pit:
    # S'arrêter sous neutralisation coûte relativement moins cher : on ne
    # cumule pas le ralentissement du peloton avec le coût de pit complet
    # (double comptage identifié au test de sanité Phase 2.2).
    lap_time = self.pace_ctx.reference_for(self.current_lap)
else:
    lap_time = predict_lap_time(
        context=self.pace_ctx,
        tyre_compound=self.own_tyre_compound,
        tyre_age_laps=self.own_tyre_age,
        is_raining=weather["is_raining"],
        rain_intensity_recent=max(weather["delta_pluie_3tours"], 0.0)
                              + float(weather["is_raining"]) * 0.3,
        rng=self.rng,
        lap=self.current_lap,
    )
```

`SC_PACE_MULTIPLIER` et `VSC_PACE_MULTIPLIER` ne sont plus utilisés. Garde-les
déclarés en haut du module : leur présence documente le passage d'une
hypothèse assumée à une valeur mesurée, ce qui est une bonne page de dossier.

## 5. La récompense doit suivre la même référence

`compute_step_reward` compare le temps au tour à `gp_baseline_s`. Si la
référence bouge tour à tour mais que la récompense garde une constante, chaque
tour sous Safety Car devient une pénalité massive sans rapport avec les
décisions de l'agent.

**Avant**
```python
breakdown = compute_step_reward(
    lap_time_s=lap_time,
    gp_baseline_s=self.pace_ctx.gp_baseline_s,
```

**Après**
```python
breakdown = compute_step_reward(
    lap_time_s=lap_time,
    gp_baseline_s=self.pace_ctx.reference_for(self.current_lap),
```

Aucune modification de `reward.py` : le paramètre garde son nom, seule la
valeur passée change. Le terme `time_component` mesure désormais l'écart au
peloton **à cet instant**, ce qui est plus proche de son intention d'origine.

---

## Vérification

```cmd
python scripts\06_sanity_replay.py
```

Attendu : les courses de pluie rentrent dans la même plage que les courses
sèches. Cible — médiane sous 1 %, pire cas sous 2,5 %.

Puis, parce que le modèle de rythme a changé :

```cmd
python scripts\03_check_breakeven.py
```

Le verdict doit rester favorable à l'arrêt. Si l'un des deux échoue, arrête-toi
et envoie la sortie plutôt que de lancer un entraînement.

## Ce que ce patch ne corrige pas

La médiane du peloton inclut la dégradation moyenne des autres voitures : une
petite part de dégradation est donc comptée deux fois. L'effet est faible et
surtout **identique pour toutes les stratégies évaluées**, donc il ne biaise
pas la comparaison de stratégies — qui est l'objet de l'environnement. À citer
comme limite assumée plutôt qu'à corriger.
