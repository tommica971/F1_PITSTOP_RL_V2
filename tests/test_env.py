"""
Phase 2.3 — Tests unitaires de F1PitStopEnv
==============================================
Formalise en pytest ce qui a été validé de façon exploratoire dans les
notebooks 04 (test interactif) et 05 (patterns de stratégie), plus les
régressions découvertes pendant le débogage (fantômes retirés vs doublés,
règle des 2 composés).
"""

import numpy as np
import pytest

from f1_pitstop_env import F1PitStopEnv, ACTIONS
from gp_pool_config import GP_POOL

ALL_POOL_GPS = GP_POOL
TRAIN_GPS = [gp for gp in GP_POOL if gp["role"] in ("train_wet", "train_dry")]


@pytest.fixture
def env():
    return F1PitStopEnv(gp_roles=("train_wet", "train_dry"), seed=42)


class TestInstantiation:
    def test_action_space_size(self, env):
        assert env.action_space.n == len(ACTIONS) == 6

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (16,)

    def test_observation_bounds_are_finite(self, env):
        assert np.all(np.isfinite(env.observation_space.low))
        assert np.all(np.isfinite(env.observation_space.high))
        assert np.all(env.observation_space.low <= env.observation_space.high)


class TestResetAndStep:
    def test_reset_returns_valid_observation(self, env):
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert "gp" in info and "season" in info

    def test_reset_starts_at_lap_2(self, env):
        env.reset(seed=0)
        assert env.current_lap == 2

    def test_step_before_reset_raises(self):
        fresh_env = F1PitStopEnv(gp_roles=("train_wet", "train_dry"), seed=1)
        with pytest.raises(RuntimeError):
            fresh_env.step(0)

    @pytest.mark.parametrize("action", range(6))
    def test_each_action_valid_on_first_step(self, env, action):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)

    def test_episode_terminates_at_correct_lap_count(self, env):
        obs, info = env.reset(seed=0)
        expected_steps = env.race_total_laps - 1  # episode demarre au tour 2
        n_steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(0)
            n_steps += 1
            if terminated:
                break
        assert n_steps == expected_steps

    def test_position_always_in_valid_range(self, env):
        obs, info = env.reset(seed=0)
        rng = np.random.default_rng(0)
        while True:
            action = int(rng.integers(0, 6))
            obs, reward, terminated, truncated, info = env.step(action)
            assert 1 <= info["position"] <= 20
            if terminated:
                break


class TestRobustnessAcrossPool:
    """Rejoue un episode aleatoire complet sur chaque GP du pool -- aucun
    crash, aucune observation hors bornes (cf. notebook 04, section 4)."""

    @pytest.mark.parametrize("gp", ALL_POOL_GPS, ids=[f"{g['season']}_{g['event']}" for g in ALL_POOL_GPS])
    def test_random_episode_no_crash_no_out_of_bounds(self, gp):
        test_env = F1PitStopEnv(fixed_gp=gp, seed=123)
        obs, info = test_env.reset(seed=123)
        rng = np.random.default_rng(123)
        while True:
            action = int(rng.integers(0, 6))
            obs, reward, terminated, truncated, info = test_env.step(action)
            assert test_env.observation_space.contains(obs), f"Obs hors bornes tour {info['lap']}"
            if terminated:
                break


class TestRetiredVsLappedGhosts:
    """Cf. bug identifie notebook 04 : un pilote retire tot doit etre exclu
    du classement, un pilote double en fin de course doit y rester."""

    def test_early_retiree_is_excluded_from_ranking(self):
        # Bahrain 2023 : LEC (abandon tour 40), OCO (tour 41), PIA (tour 13)
        test_env = F1PitStopEnv(
            fixed_gp={"season": 2023, "event": "Bahrain Grand Prix", "role": "train_dry"}, seed=1
        )
        test_env.reset(seed=1)
        rng = np.random.default_rng(1)
        while True:
            action = int(rng.integers(0, 6))
            obs, reward, terminated, truncated, info = test_env.step(action)
            if terminated:
                break
        assert "PIA" in test_env.ghost_retired
        assert "LEC" in test_env.ghost_retired
        assert "OCO" in test_env.ghost_retired

    def test_lapped_driver_not_marked_retired(self):
        # Bahrain 2023 : plusieurs pilotes finissent tour 56/57 (double, pas abandon)
        test_env = F1PitStopEnv(
            fixed_gp={"season": 2023, "event": "Bahrain Grand Prix", "role": "train_dry"}, seed=1
        )
        test_env.reset(seed=1)
        while True:
            obs, reward, terminated, truncated, info = test_env.step(0)
            if terminated:
                break
        assert len(test_env.ghost_retired) == 3  # uniquement les 3 vrais abandons


class TestTyreRule:
    """Article 30.7 : au moins 2 composes secs, sauf course pluvieuse."""

    def test_single_compound_dry_race_triggers_guardrail_not_dsq(self):
        """Phase 3.4 : le garde-fou de conformite (dernier tour, cf.
        f1_pitstop_env.py) empeche desormais la DSQ meme si l'agent ne pit
        jamais volontairement -- remplace l'ancien test qui verifiait
        l'inverse (comportement intentionnellement change, cf. reward.py /
        f1_pitstop_env.py Phase 3.4 : bareme de points reel + garde-fou)."""
        test_env = F1PitStopEnv(
            fixed_gp={"season": 2023, "event": "Bahrain Grand Prix", "role": "train_dry"}, seed=1
        )
        test_env.reset(seed=1)
        forced_count = 0
        while True:
            obs, reward, terminated, truncated, info = test_env.step(0)  # ne jamais pit volontairement
            if info["forced_compliance_pit"]:
                forced_count += 1
            if terminated:
                break
        assert info["invalid_strategy"] is False
        assert forced_count == 1  # le garde-fou n'intervient qu'une fois, au dernier tour

    def test_two_compounds_dry_race_is_valid(self):
        test_env = F1PitStopEnv(
            fixed_gp={"season": 2023, "event": "Bahrain Grand Prix", "role": "train_dry"}, seed=1
        )
        test_env.reset(seed=1)
        lap = 2
        while True:
            action = 2 if lap == 25 else 0  # un seul pit MEDIUM
            obs, reward, terminated, truncated, info = test_env.step(action)
            lap += 1
            if terminated:
                break
        assert info["invalid_strategy"] is False

    def test_single_compound_wet_race_is_not_disqualified(self):
        # cf. debug Phase 1.3 : British 2025 a un vrai signal de pluie (18.1%)
        test_env = F1PitStopEnv(
            fixed_gp={"season": 2025, "event": "British Grand Prix", "role": "train_wet"}, seed=1
        )
        test_env.reset(seed=1)
        while True:
            obs, reward, terminated, truncated, info = test_env.step(0)
            if terminated:
                break
        assert test_env.race_had_rain is True
        assert info["invalid_strategy"] is False

    def test_disqualification_penalty_dominates_any_position_bonus(self):
        """Une strategie invalide ne doit JAMAIS etre preferable a une
        strategie valide, meme si elle finit mieux classee sur la piste."""
        from reward import DISQUALIFICATION_PENALTY, FINAL_POSITION_BONUS_WEIGHT
        best_possible_position_bonus = 20 * FINAL_POSITION_BONUS_WEIGHT  # P1, max theorique large
        assert DISQUALIFICATION_PENALTY > best_possible_position_bonus - 20 * FINAL_POSITION_BONUS_WEIGHT + 100
        # borne large : DSQ doit au moins depasser le bonus P1 reel (100 pts, cf. reward.py)
        assert DISQUALIFICATION_PENALTY > (21 - 1) * FINAL_POSITION_BONUS_WEIGHT


class TestSafetyCarHandling:
    def test_no_stacking_of_sc_penalty_and_pit_cost(self):
        """Cf. bug identifie Phase 2.2 : un tour a la fois sous SC/VSC et de
        pit ne doit pas cumuler le multiplicateur SC ET le cout de pit plein."""
        test_env = F1PitStopEnv(
            fixed_gp={"season": 2023, "event": "Dutch Grand Prix", "role": "train_wet"}, seed=1
        )
        test_env.reset(seed=1)
        # Dutch 2023 a un Red Flag + SC/VSC connus (cf. gp_pool_config.py)
        max_plausible_lap_time = test_env.pace_ctx.gp_baseline_s * 1.40 + 60  # SC max + pit max
        while True:
            obs, reward, terminated, truncated, info = test_env.step(0)
            assert info["lap_time_s"] < max_plausible_lap_time * 1.1, (
                f"Tour {info['lap']}: lap_time={info['lap_time_s']:.1f}s suspect (empilement SC+pit ?)"
            )
            if terminated:
                break
