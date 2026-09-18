"""Phase 2.3 — Tests unitaires de reward.py (isolé de la mécanique Gymnasium)."""

import pytest

from reward import (
    compute_step_reward,
    POSITION_DELTA_WEIGHT_S,
    DIRTY_AIR_THRESHOLD_S,
    DIRTY_AIR_PENALTY,
    FINAL_POSITION_BONUS_WEIGHT,
    DISQUALIFICATION_PENALTY,
)

GP_BASELINE = 95.0


class TestTimeComponent:
    def test_faster_than_baseline_gives_positive_time_component(self):
        b = compute_step_reward(
            lap_time_s=90.0, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
        )
        assert b.time_component > 0

    def test_slower_than_baseline_gives_negative_time_component(self):
        b = compute_step_reward(
            lap_time_s=120.0, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
        )
        assert b.time_component < 0

    def test_time_component_magnitude_equals_seconds_delta(self):
        b = compute_step_reward(
            lap_time_s=105.0, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
        )
        assert b.time_component == pytest.approx(-10.0)


class TestPositionDeltaComponent:
    def test_gaining_position_is_rewarded(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=1,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
        )
        assert b.position_delta_component == pytest.approx(POSITION_DELTA_WEIGHT_S)

    def test_losing_position_is_penalized(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=-2,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
        )
        assert b.position_delta_component == pytest.approx(-2 * POSITION_DELTA_WEIGHT_S)

    def test_no_change_gives_zero(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
        )
        assert b.position_delta_component == 0.0


class TestDirtyAirComponent:
    def test_below_threshold_triggers_penalty(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=DIRTY_AIR_THRESHOLD_S - 0.1, is_terminal=False,
            own_final_position=10, is_invalid_strategy=False,
        )
        assert b.dirty_air_component == pytest.approx(-DIRTY_AIR_PENALTY)

    def test_above_threshold_no_penalty(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=DIRTY_AIR_THRESHOLD_S + 0.1, is_terminal=False,
            own_final_position=10, is_invalid_strategy=False,
        )
        assert b.dirty_air_component == 0.0

    def test_exactly_at_threshold_no_penalty(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=DIRTY_AIR_THRESHOLD_S, is_terminal=False,
            own_final_position=10, is_invalid_strategy=False,
        )
        assert b.dirty_air_component == 0.0


class TestTerminalComponent:
    def test_non_terminal_step_has_no_terminal_component(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=1, is_invalid_strategy=False,
        )
        assert b.terminal_component == 0.0

    def test_p1_finish_gives_maximal_bonus(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=True, own_final_position=1, is_invalid_strategy=False,
        )
        assert b.terminal_component == pytest.approx(20 * FINAL_POSITION_BONUS_WEIGHT)

    def test_bonus_decreases_with_worse_position(self):
        b_p1 = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=True, own_final_position=1, is_invalid_strategy=False,
        )
        b_p10 = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=True, own_final_position=10, is_invalid_strategy=False,
        )
        assert b_p1.terminal_component > b_p10.terminal_component

    def test_invalid_strategy_gives_disqualification_penalty_not_position_bonus(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=True, own_final_position=1, is_invalid_strategy=True,
        )
        # Meme en P1 (meilleure position possible), une strategie invalide
        # doit etre penalisee, pas recompensee -- pas de classement possible.
        assert b.terminal_component == pytest.approx(-DISQUALIFICATION_PENALTY)

    def test_disqualification_worse_than_worst_valid_finish(self):
        b_dsq = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=True, own_final_position=1, is_invalid_strategy=True,
        )
        b_p20 = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=True, own_final_position=20, is_invalid_strategy=False,
        )
        assert b_dsq.terminal_component < b_p20.terminal_component


class TestTotalConsistency:
    def test_total_equals_sum_of_components(self):
        b = compute_step_reward(
            lap_time_s=100.0, gp_baseline_s=GP_BASELINE, delta_position=2,
            gap_avant_s=0.5, is_terminal=True, own_final_position=5, is_invalid_strategy=False,
        )
        expected = (
            b.time_component + b.position_delta_component + b.dirty_air_component
            + b.compliance_component + b.terminal_component
        )
        assert b.total == pytest.approx(expected)


class TestComplianceComponent:
    def test_no_penalty_far_from_race_end(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
            tours_restants=30, n_compounds_used=1, race_had_rain=False,
        )
        assert b.compliance_component == 0.0

    def test_penalty_appears_near_race_end_if_single_compound(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
            tours_restants=5, n_compounds_used=1, race_had_rain=False,
        )
        assert b.compliance_component < 0.0

    def test_no_penalty_if_already_compliant(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
            tours_restants=5, n_compounds_used=2, race_had_rain=False,
        )
        assert b.compliance_component == 0.0

    def test_no_penalty_if_wet_race(self):
        b = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
            tours_restants=5, n_compounds_used=1, race_had_rain=True,
        )
        assert b.compliance_component == 0.0

    def test_penalty_grows_as_race_end_approaches(self):
        b_far = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
            tours_restants=14, n_compounds_used=1, race_had_rain=False,
        )
        b_near = compute_step_reward(
            lap_time_s=GP_BASELINE, gp_baseline_s=GP_BASELINE, delta_position=0,
            gap_avant_s=10.0, is_terminal=False, own_final_position=10, is_invalid_strategy=False,
            tours_restants=1, n_compounds_used=1, race_had_rain=False,
        )
        assert b_near.compliance_component < b_far.compliance_component
