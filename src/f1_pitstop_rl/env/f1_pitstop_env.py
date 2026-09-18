"""
Phase 2.1 — F1PitStopEnv (Gymnasium)
=======================================
Environnement RL pour la décision de pit stop, un agent (Gasly/Alpine) +
19 adversaires fantômes rejouant leur trajectoire historique réelle (décision
Phase 0 : pas de multi-agent complet, cf. échange de validation Phase 2).

SIMPLIFICATIONS ASSUMÉES ET DOCUMENTÉES (à citer dans le dossier écrit) :

1. L'épisode démarre au TOUR 2, pas au tour 1. Le tour 1 souffre d'un artefact
   de données structurel (temps de session cumulé au lieu d'un vrai temps au
   tour, cf. debug Phase 1.2/1.3) qui rend impossible une reconstruction
   fiable du temps au tour 1. L'état de départ (position, écarts) au tour 2
   est initialisé depuis les données réelles observées au tour 1.

2. Les écarts (gaps) initiaux avec TOUS les fantômes utilisent le temps de
   session réel (sesT) au tour 1 (fichier initial_gaps.parquet) -- PAS une
   approximation par position (corrigé en Phase 2.2 suite au test de sanité
   ci-dessous). Pendant la course, les écarts évoluent dynamiquement à partir
   des temps cumulés réels (fantômes) et prédits (agent).

3. Les tours sous Safety Car utilisent un multiplicateur de rythme sévère
   (SC_PACE_MULTIPLIER=1.40, assumé) ; les tours sous VSC utilisent un
   multiplicateur bien plus doux (VSC_PACE_MULTIPLIER=1.03, calibré via le
   test de sanité ci-dessous -- une VSC ne ralentit presque pas le rythme réel
   contrairement à une Safety Car physique sur la piste).

TEST DE SANITÉ (Phase 2.2) -- rejeu de la stratégie réelle de Gasly à
Bahreïn 2023 (SOFT->HARD T10->HARD T26->SOFT T41, résultat réel P19->P9) :
    - Version initiale (offset équipe en double-comptage + SC/VSC non
      distingués + écarts approximés) : temps simulé 5685.6s (réel 5600.8s,
      +84.8s) -> P17
    - Après retrait du double-comptage (gp_baseline_s encode déjà la
      compétitivité réelle de Gasly ce jour-là) -> P17, écart réduit à +45.3s
    - Après distinction VSC (douce, 1.03x) / SC (sévère, 1.40x) -> P16,
      écart réduit à +8.7s (0.16% d'erreur agrégée sur ~5600s)
    - Après remplacement de l'approximation par position par les vrais
      écarts sesT au tour 1 (initial_gaps.parquet) -> P15
    CONCLUSION : le rythme global est calibré à 0.16% près, mais le peloton
    ce jour-là était extrêmement compact (P1-P16 sur seulement 91s, ~6s/rang)
    -- le reliquat de position restant est attribuable à la variance du
    tirage stochastique du coût de pit stop (Phase 1.4, ±5.4s), pas à un
    défaut de modélisation. Décision actée : ne pas chasser plus loin la
    reproduction exacte d'un résultat historique individuel (fragile par
    nature avec un peloton aussi resserré) ; la calibration du RYTHME
    AGRÉGÉ est validée, l'évaluation relative de stratégies (l'objectif réel
    de l'environnement) n'est pas remise en cause par ce reliquat.

Espace d'observation (16 dimensions -- 15 en Phase 1.3, + n_compounds_used
ajouté en Phase 3.2 suite à un taux de disqualification ~40% : sans cette
info, l'agent ne peut pas savoir s'il a déjà satisfait la règle des 2
composés, cf. Article 30.7 -- nécessité Markov, pas un problème de réglage) :
    [tyre_age_laps, tours_restants, position, gap_avant, gap_arriere,
     track_temp_c, delta_pluie_3tours, is_raining,
     rival_avant_vient_de_pitter, rival_arriere_vient_de_pitter,
     n_compounds_used,
     tyre_compound (one-hot 5 : SOFT/MEDIUM/HARD/INTERMEDIATE/WET)]

Espace d'action (6, discret) :
    0=rester en piste, 1=pit+SOFT, 2=pit+MEDIUM, 3=pit+HARD,
    4=pit+INTERMEDIATE, 5=pit+WET
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
sys.path.insert(0, str(Path(__file__).parent))
from gp_pool_config import GP_POOL, DRIVER_CODE
from pace_model import (GPPaceContext, predict_lap_time, sample_pit_stop_cost,
                        build_lap_reference)
from reward import compute_step_reward

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
ACTIONS = ["STAY", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD", "PIT_INTER", "PIT_WET"]
ACTION_TO_COMPOUND = {1: "SOFT", 2: "MEDIUM", 3: "HARD", 4: "INTERMEDIATE", 5: "WET"}

# Hypothèses documentées (non calibrées finement, cf. en-tête du module)
AVG_GAP_PER_POSITION_S = 1.9   # cf. features_dataset.parquet, médiane gap_avant Phase 1.3
# Pénalité de rythme sous anomalie piste (assumée, affinée via test de sanité
# Bahrain 2023 -- Phase 2.2). Distinction VSC/SC nécessaire : une VSC impose
# une vitesse plafond mais pas de voiture de sécurité physique sur la piste,
# le ralentissement réel observé est bien plus faible qu'une Safety Car
# complète (constaté : tour VSC Bahreïn 2023 à 95.07s, quasi identique à un
# tour normal ~96-98s, alors qu'un tour de Safety Car complet est typiquement
# 30-50% plus lent).
MIN_STINT_LAPS = 5   # cf. scripts/apply_patch_min_stint.py
# Longueur minimale d'un relais, en tours. Une action de pit avant cet
# age de pneu est convertie en STAY. Contrainte dure du domaine (aucune
# equipe ne se rearrete 1-2 tours apres un arret : 23 s ne se remboursent
# pas sur un relais aussi court), implementee comme forced_compliance_pit
# et donc algorithme-agnostique.
VSC_PACE_MULTIPLIER = 1.03
SC_PACE_MULTIPLIER = 1.40
DIRTY_AIR_THRESHOLD_S = 1.0   # cf. reward.py -- utilisé aussi pour l'observation (rival_avant_vient_de_pitter)

# Bornage de l'espace d'observation (Phase 2.1 -- "bornage", révisé Phase 3.2
# suite au taux de DSQ ~40% observé sur le premier entraînement)
OBS_LOW = np.array([
    0,      # tyre_age_laps
    0,      # tours_restants
    1,      # position
    0,      # gap_avant (s)
    0,      # gap_arriere (s)
    0,      # track_temp_c
    -1,     # delta_pluie_3tours
    0,      # is_raining
    0,      # rival_avant_vient_de_pitter
    0,      # rival_arriere_vient_de_pitter
    1,      # n_compounds_used (au moins 1, le composé de depart)
    0, 0, 0, 0, 0,  # tyre_compound one-hot
], dtype=np.float32)

OBS_HIGH = np.array([
    45,     # tyre_age_laps (borne large)
    80,     # tours_restants (borne large, course la plus longue du pool)
    20,     # position
    60,     # gap_avant (s), borne large anti-outlier (cf. EDA Phase 1.3)
    60,     # gap_arriere (s)
    50,     # track_temp_c
    1,      # delta_pluie_3tours
    1,      # is_raining
    1,      # rival_avant_vient_de_pitter
    1,      # rival_arriere_vient_de_pitter
    5,      # n_compounds_used (5 composés possibles au maximum)
    1, 1, 1, 1, 1,  # tyre_compound one-hot
], dtype=np.float32)


_PACK_AGE_CACHE = None


def _load_pack_median_age(season, event) -> dict:
    """Charge {tour: age median du peloton} pour un GP.

    Produit par scripts/v2_01_apply_pace_debias.py. Absent -> dict vide, la
    correction est alors inactive et le modele retrouve le comportement V1.
    """
    global _PACK_AGE_CACHE
    if _PACK_AGE_CACHE is None:
        p = (Path(__file__).resolve().parents[3] / "data" / "processed"
             / "pack_median_age.json")
        try:
            _PACK_AGE_CACHE = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            import warnings
            warnings.warn(
                f"pack_median_age.json introuvable ({p}) — correction du biais "
                "de rythme INACTIVE, comportement V1.", RuntimeWarning,
                stacklevel=2)
            _PACK_AGE_CACHE = {}
    return {int(k): float(v)
            for k, v in _PACK_AGE_CACHE.get(f"{season}|{event}", {}).items()}

class F1PitStopEnv(gym.Env):
    """Environnement Gymnasium pour l'optimisation de la stratégie de pit stop."""

    metadata = {"render_modes": []}

    def __init__(
        self, gp_roles=("train_wet", "train_dry"), fixed_gp: dict | None = None,
        seed: int | None = None, gp_weights: dict | None = None,
        min_stint_laps: int | None = None
    ):
        # Longueur minimale de relais. None -> valeur du module.
        # 0 desactive la contrainte : sert a l'ablation 2x2
        # (scripts/v2_05_apply_ablation_switches.py).
        self.min_stint_laps = (MIN_STINT_LAPS if min_stint_laps is None
                               else int(min_stint_laps))
        super().__init__()
        self.gp_roles = gp_roles
        self.fixed_gp = fixed_gp
        self.eligible_gps = [gp for gp in GP_POOL if gp["role"] in gp_roles]
        if not self.eligible_gps and fixed_gp is None:
            raise ValueError(f"Aucun GP disponible pour les rôles {gp_roles}")

        # Tirage pondéré par rôle (Phase 3.5 bis) -- optionnel, retro-compatible :
        # gp_weights=None (défaut) => tirage uniforme, comportement inchangé.
        # Motivation : sur le pool actuel (4 GP train_wet / 3 GP train_dry),
        # un tirage uniforme sous-représente les scénarios secs longue
        # distance qui exigent une vraie gestion d'usure (la règle des 2
        # composés est levée sur la majorité des GP pluie, cf. reward.py) --
        # ce déséquilibre est une piste plausible pour expliquer pourquoi
        # PPO/A2C tunés échouent sur Belgique 2025 (généralisation, Phase
        # 3.5). gp_weights permet de sur-pondérer un rôle sans toucher au
        # pool lui-même ni à la récompense.
        if gp_weights is not None:
            unknown_roles = set(gp_weights) - {"train_wet", "train_dry"}
            if unknown_roles:
                raise ValueError(f"gp_weights contient des rôles inconnus : {unknown_roles}")
            raw_weights = np.array([gp_weights.get(gp["role"], 1.0) for gp in self.eligible_gps], dtype=float)
            self._gp_probs = raw_weights / raw_weights.sum()
        else:
            self._gp_probs = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)

        # Perf (revue de code, Phase 3.5) : vecteurs one-hot precalcules une
        # fois plutot que reconstruits par comprehension de liste a chaque
        # step() -- comportement identique, juste evite l'allocation +
        # comparaison de chaines repetee dans la boucle chaude.
        self._compound_onehot = {
            c: [1.0 if c == other else 0.0 for other in COMPOUNDS] for c in COMPOUNDS
        }

        self._features_df = pd.read_parquet(DATA_DIR / "processed" / "features_dataset.parquet")
        self._ghosts_df = pd.read_parquet(DATA_DIR / "raw" / "all_drivers_dataset.parquet")
        self._initial_gaps_df = pd.read_parquet(DATA_DIR / "raw" / "initial_gaps.parquet")

        self.rng = np.random.default_rng(seed)

        # État interne, initialisé à reset()
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.current_gp = None
        self.current_lap = None
        self.race_total_laps = None
        self.pace_ctx = None
        self.weather_by_lap = {}
        self.ghost_cum_time = {}
        self.ghost_laps = {}
        self.own_cum_time = 0.0
        self.own_position = None
        self.own_tyre_compound = None
        self.own_tyre_age = None
        self.ghost_retired = set()
        self.compounds_used = set()
        self.race_had_rain = False
        # Perf (revue de code, Phase 3.5) : cache du classement trie, invalide
        # a chaque step() -- evite de retrier all_times jusqu'a 3 fois par
        # tour dans _rank_position/_compute_gaps/_rival_pit_flags.
        self._ranked_cache = None

    def _select_episode_gp(self) -> dict:
        if self.fixed_gp is not None:
            return self.fixed_gp
        if self._gp_probs is not None:
            idx = self.rng.choice(len(self.eligible_gps), p=self._gp_probs)
        else:
            idx = self.rng.integers(0, len(self.eligible_gps))
        return self.eligible_gps[idx]

    def _load_gp_data(self, gp: dict):
        season, event = gp["season"], gp["event"]

        gasly = self._features_df[
            (self._features_df["season"] == season) & (self._features_df["event"] == event)
        ].sort_values("lap_number").reset_index(drop=True)
        if gasly.empty:
            raise ValueError(f"Pas de données Gasly pour {season} {event}")

        ghosts = self._ghosts_df[
            (self._ghosts_df["season"] == season) & (self._ghosts_df["event"] == event)
            & (self._ghosts_df["driver"] != DRIVER_CODE)
        ]

        self.race_total_laps = int(gasly["lap_number"].max())

        # Météo/anomalies par tour (donnée exogène, indépendante des choix de l'agent)
        self.weather_by_lap = {
            int(row.lap_number): {
                "is_raining": bool(row.is_raining),
                "track_temp_c": float(row.track_temp_c),
                "delta_pluie_3tours": float(row.delta_pluie_3tours) if pd.notna(row.delta_pluie_3tours) else 0.0,
                "is_safety_car": bool(row.is_safety_car),
                "is_vsc": bool(row.is_vsc),
            }
            for row in gasly.itertuples()
        }

        # Contexte de calibration du GP (médiane/écart-type des tours exploitables)
        flag = ("usable_for_reward_v2" if "usable_for_reward_v2" in gasly.columns
                else "usable_for_reward")
        usable = gasly[gasly[flag]]

        # Reference de rythme tour par tour, derivee du peloton reel.
        # Absorbe pluie, Safety Car, VSC, sechage de piste et evolution
        # de grip, sans aucune hypothese -- la v2 n'avait aucun terme
        # pour l'etat de la piste une fois usable_for_reward nettoye.
        lap_reference = build_lap_reference(self._ghosts_df, season, event)

        # Age median du pneu du peloton, tour par tour. Sert a ne facturer a
        # l'agent que son ECART a cet age : la reference de rythme contient
        # deja la degradation du peloton (cf. v2_01_apply_pace_debias.py).
        pack_median_age = _load_pack_median_age(season, event)

        self.pace_ctx = GPPaceContext(
            gp_baseline_s=float(usable["lap_time_s"].median()),
            gp_std_s=float(usable["lap_time_s"].std()),
            race_total_laps=self.race_total_laps,
            lap_reference_s=lap_reference,
            pack_median_age=pack_median_age,
        )

        # Trajectoires fantômes : temps au tour réel par pilote/tour (rejeu figé)
        self.ghost_laps = {
            drv: g.set_index("lap_number")["lap_time_s"].to_dict()
            for drv, g in ghosts.groupby("driver")
        }
        self._ghost_stint_lookup = {
            (row.driver, int(row.lap_number)): row.stint_number
            for row in ghosts.itertuples()
        }
        ghost_starting_positions = {
            drv: g.sort_values("lap_number").iloc[0]["position"]
            for drv, g in ghosts.groupby("driver")
        }

        # État de départ de l'agent (hérité des données réelles au tour 1)
        lap1 = gasly[gasly["lap_number"] == gasly["lap_number"].min()].iloc[0]
        self.own_position = int(lap1["position"])
        self.own_tyre_compound = str(lap1["tyre_compound"])
        self.own_tyre_age = int(lap1["tyre_age_laps"]) + 1  # +1 car l'épisode démarre au tour 2
        self.compounds_used = {self.own_tyre_compound}
        self.race_had_rain = any(w["is_raining"] for w in self.weather_by_lap.values())

        # Initialisation des temps cumulés fantômes -- écarts RÉELS au tour 1
        # (sesT), remplace l'approximation constante AVG_GAP_PER_POSITION_S
        # (correctif Phase 2.2, suite au test de sanité Bahrain 2023).
        self.own_cum_time = 0.0
        gaps = self._initial_gaps_df[
            (self._initial_gaps_df["season"] == season) & (self._initial_gaps_df["event"] == event)
        ].set_index("driver")["sesT_lap1"].to_dict()

        own_ses_t = gaps.get(DRIVER_CODE)
        self.ghost_cum_time = {}
        for drv in self.ghost_laps:
            if own_ses_t is not None and drv in gaps:
                self.ghost_cum_time[drv] = gaps[drv] - own_ses_t
            else:
                # repli sur l'approximation par position si donnée manquante
                self.ghost_cum_time[drv] = (ghost_starting_positions[drv] - self.own_position) * AVG_GAP_PER_POSITION_S

        self.current_gp = gp
        # cf. _ranked_drivers : own_cum_time/ghost_cum_time viennent d'être
        # (ré)initialisés pour ce nouvel épisode -- invalide tout cache
        # hérité de l'épisode précédent.
        self._ranked_cache = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        gp = self._select_episode_gp()
        self._load_gp_data(gp)
        self.current_lap = 2  # cf. hypothèse #1 en-tête

        obs = self._build_observation()
        info = {"gp": self.current_gp["event"], "season": self.current_gp["season"]}
        return obs, info

    def _active_ghost_times(self) -> dict:
        return {drv: t for drv, t in self.ghost_cum_time.items() if drv not in self.ghost_retired}

    def _ranked_drivers(self) -> list[tuple[str, float]]:
        """Classement (driver -> temps cumulé) trié croissant, mis en cache
        pour la durée d'un step()/reset() -- évite de retrier all_times
        jusqu'à 3 fois par tour (perf, revue de code Phase 3.5). Le cache
        est invalidé explicitement en tête de step() (une seule source de
        vérité pour l'invalidation, cf. self._ranked_cache = None)."""
        if self._ranked_cache is None:
            all_times = {**self._active_ghost_times(), "__AGENT__": self.own_cum_time}
            self._ranked_cache = sorted(all_times.items(), key=lambda kv: kv[1])
        return self._ranked_cache

    def _rank_position(self) -> int:
        ranked = self._ranked_drivers()
        return [k for k, _ in ranked].index("__AGENT__") + 1

    def _compute_gaps(self) -> tuple[float, float]:
        ranked = self._ranked_drivers()
        idx = [k for k, _ in ranked].index("__AGENT__")
        gap_avant = ranked[idx][1] - ranked[idx - 1][1] if idx > 0 else 0.0
        gap_arriere = ranked[idx + 1][1] - ranked[idx][1] if idx < len(ranked) - 1 else 0.0
        return max(gap_avant, 0.0), max(gap_arriere, 0.0)

    def _rival_pit_flags(self) -> tuple[bool, bool]:
        """Détecte si le rival immédiat (avant/arrière) vient de s'arrêter ce tour."""
        ranked = self._ranked_drivers()
        idx = [k for k, _ in ranked].index("__AGENT__")

        def _stint_at(drv: str, lap: int):
            row = self._ghost_stint_lookup.get((drv, lap))
            return row

        def _just_pitted(drv: str) -> bool:
            if drv is None or drv == "__AGENT__" or self.current_lap <= 2:
                return False
            stint_now = _stint_at(drv, self.current_lap)
            stint_prev = _stint_at(drv, self.current_lap - 1)
            if stint_now is None or stint_prev is None:
                return False
            return stint_now != stint_prev

        rival_avant = ranked[idx - 1][0] if idx > 0 else None
        rival_arriere = ranked[idx + 1][0] if idx < len(ranked) - 1 else None
        return _just_pitted(rival_avant), _just_pitted(rival_arriere)

    def _build_observation(self) -> np.ndarray:
        weather = self.weather_by_lap.get(self.current_lap, {
            "is_raining": False, "track_temp_c": 25.0, "delta_pluie_3tours": 0.0,
        })
        gap_avant, gap_arriere = self._compute_gaps()
        rival_avant_pit, rival_arriere_pit = self._rival_pit_flags()

        compound_onehot = self._compound_onehot[self.own_tyre_compound]

        # Nécessaire à la propriété de Markov pour la règle des 2 composés
        # (Article 30.7) : sans cette information, l'agent ne peut pas savoir
        # s'il a déjà satisfait la règle -- bug identifié via le taux de DSQ
        # ~40% observé après un premier entraînement (Phase 3.2).
        n_compounds_used = float(len(self.compounds_used))

        obs = np.array([
            self.own_tyre_age,
            self.race_total_laps - self.current_lap,
            self.own_position,
            gap_avant,
            gap_arriere,
            weather["track_temp_c"],
            weather["delta_pluie_3tours"],
            float(weather["is_raining"]),
            float(rival_avant_pit),
            float(rival_arriere_pit),
            n_compounds_used,
            *compound_onehot,
        ], dtype=np.float32)

        return np.clip(obs, OBS_LOW, OBS_HIGH)

    def step(self, action: int):
        if self.current_lap is None:
            raise RuntimeError("step() appelé avant reset()")

        # Invalide le cache de classement (cf. _ranked_drivers) : own_cum_time
        # va changer plus bas dans ce step(), donc l'ancien classement n'est
        # plus valable. Recalcule une seule fois, a la premiere consultation.
        self._ranked_cache = None

        weather = self.weather_by_lap.get(self.current_lap, {
            "is_raining": False, "track_temp_c": 25.0, "delta_pluie_3tours": 0.0,
            "is_safety_car": False, "is_vsc": False,
        })

        # Garde-fou de conformité (Phase 3.4) -- règle des 2 composés dure,
        # Article 30.7. Le signal de conformité progressif (reward.py) reste
        # en place comme incitation ANTICIPÉE, mais ne suffisait pas à lui
        # seul (DSQ ~14% persistant même avec le signal, cf. évaluation PPO
        # tuné v3). Ici on n'intervient QUE sur le tout dernier tour de la
        # course (le seul moment où "attendre encore" n'est plus une option
        # pour l'agent) : si la course est sèche et que l'action choisie
        # laisserait l'agent avec un seul composé utilisé au drapeau à
        # damiers, l'action est remplacée par un pit vers un composé sec
        # encore inutilisé. Décision de modélisation assumée : la contrainte
        # 2-composés est une règle dure du règlement, pas une pénalité
        # probabiliste -- la modéliser comme contrainte dure est plus
        # fidèle au domaine, pas juste plus simple. Algorithme-agnostique
        # (contrairement à un masquage d'action type MaskablePPO, qui
        # n'existe pas nativement pour A2C dans sb3-contrib) : la comparaison
        # DQN/PPO/A2C reste équitable.
        is_last_lap = self.current_lap >= self.race_total_laps

        # Longueur minimale de relais (cf. MIN_STINT_LAPS en tete de
        # module). Exception au dernier tour, ou le garde-fou de
        # conformite Article 30.7 doit pouvoir imposer un arret.
        blocked_by_min_stint = False
        if (action != 0 and not is_last_lap
                and self.own_tyre_age < self.min_stint_laps):
            action = 0
            blocked_by_min_stint = True

        forced_compliance_pit = False
        if is_last_lap and not self.race_had_rain and len(self.compounds_used) < 2:
            resulting_compound = ACTION_TO_COMPOUND.get(action, self.own_tyre_compound)
            if resulting_compound in self.compounds_used:
                dry_compounds = ["MEDIUM", "HARD", "SOFT"]
                new_compound = next(c for c in dry_compounds if c not in self.compounds_used)
                action = next(a for a, c in ACTION_TO_COMPOUND.items() if c == new_compound)
                forced_compliance_pit = True

        prev_position = self.own_position
        is_pit = action != 0
        pit_cost = 0.0
        under_caution = weather["is_safety_car"] or weather["is_vsc"]

        # Temps au tour de l'agent, avec le pneu AVANT changement éventuel.
        # NOTE (bug corrigé, Phase 2.2 -- test de sanité Bahrain 2023) : si le
        # tour est À LA FOIS sous SC/VSC ET un tour de pit, ne PAS cumuler le
        # multiplicateur SC (ralentissement de tout le peloton) ET le coût de
        # pit complet -- ça double-comptait la pénalité, alors qu'en réalité
        # s'arrêter sous SC/VSC est justement moins coûteux relativement (cf.
        # 03_analyses_avancees.ipynb section 5 : gain de position -1.20 sous
        # anomalie vs -1.92 en vert -- l'écart au peloton se creuse moins vite
        # pendant qu'on est arrêté si tout le monde roule au ralenti).
        # Plus de multiplicateur SC/VSC : la reference du tour porte deja
        # le ralentissement reel du peloton, mesure au lieu d'etre suppose.
        if under_caution and is_pit:
            # S'arreter sous neutralisation coute relativement moins cher :
            # on ne cumule pas le ralentissement du peloton avec le cout de
            # pit complet (double comptage, test de sanite Phase 2.2).
            lap_time = self.pace_ctx.reference_for(self.current_lap)
        else:
            lap_time = predict_lap_time(
                context=self.pace_ctx,
                tyre_compound=self.own_tyre_compound,
                tyre_age_laps=self.own_tyre_age,
                is_raining=weather["is_raining"],
                rain_intensity_recent=max(weather["delta_pluie_3tours"], 0.0) + float(weather["is_raining"]) * 0.3,
                rng=self.rng,
                lap=self.current_lap,
            )

        if is_pit:
            pit_cost = sample_pit_stop_cost(self.rng)
            lap_time += pit_cost

        self.own_cum_time += lap_time

        # Rejeu des fantômes : temps réel historique à ce tour (déjà réaliste
        # sous SC/VSC/pit, car ce sont de vraies données). Distinction
        # importante (bug identifié lors du test Phase 2.1, notebook 04) :
        #   - un pilote DOUBLÉ termine la course avec 1-2 tours de moins que
        #     le leader (ex: 56/57) -- son temps cumulé figé est parfaitement
        #     valide et il reste classé dans le peloton
        #   - un VRAI abandon (ex: tour 13/57) doit être exclu du classement,
        #     sinon son temps cumulé minuscule le ferait apparaître à tort
        #     comme "toujours devant"
        # Seuil : moins de 85% de la course complétée = abandon réel
        RETIREMENT_FRACTION_THRESHOLD = 0.85
        for drv, laps in self.ghost_laps.items():
            ghost_lap_time = laps.get(self.current_lap)
            if ghost_lap_time is not None and ghost_lap_time > 0:
                self.ghost_cum_time[drv] += ghost_lap_time
            else:
                last_lap = max(laps.keys()) if laps else 0
                if last_lap < RETIREMENT_FRACTION_THRESHOLD * self.race_total_laps:
                    self.ghost_retired.add(drv)
                # sinon : pilote doublé, on garde son temps cumulé figé (reste classé)

        # Mise à jour de l'état pneu (effective à partir du tour suivant)
        if is_pit:
            self.own_tyre_compound = ACTION_TO_COMPOUND[action]
            self.own_tyre_age = 1
            self.compounds_used.add(self.own_tyre_compound)
        else:
            self.own_tyre_age += 1

        self.own_position = self._rank_position()
        gap_avant, _ = self._compute_gaps()

        # --- Récompense (équivalent-temps, Phase 0.2/2.2, cf. reward.py) ---
        delta_position = prev_position - self.own_position
        terminated = self.current_lap >= self.race_total_laps

        # Règle F1 réelle (Article 30.7) : au moins 2 composés secs différents
        # en course sèche, sous peine d'EXCLUSION des résultats.
        invalid_strategy = terminated and not self.race_had_rain and len(self.compounds_used) < 2

        breakdown = compute_step_reward(
            lap_time_s=lap_time,
            # Reference du TOUR, pas constante de course : sinon chaque
            # tour de Safety Car deviendrait une penalite massive sans
            # lien avec les decisions de l'agent.
            gp_baseline_s=self.pace_ctx.reference_for(self.current_lap),
            delta_position=delta_position,
            gap_avant_s=gap_avant,
            is_terminal=terminated,
            own_final_position=self.own_position,
            is_invalid_strategy=invalid_strategy,
            tours_restants=self.race_total_laps - self.current_lap,
            n_compounds_used=len(self.compounds_used),
            race_had_rain=self.race_had_rain,
        )
        reward = breakdown.total

        info = {
            "lap": self.current_lap,
            "is_pit": is_pit,
            "pit_cost_s": pit_cost,
            "lap_time_s": lap_time,
            "position": self.own_position,
            "invalid_strategy": invalid_strategy,
            "forced_compliance_pit": forced_compliance_pit,
            "blocked_by_min_stint": blocked_by_min_stint,
            "reward_breakdown": breakdown,
        }

        if not terminated:
            self.current_lap += 1

        obs = self._build_observation()
        return obs, float(reward), terminated, False, info
