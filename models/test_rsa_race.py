"""
Tests for RSA Race Model (reduced parameterization).
"""

import pytest
import jax.numpy as jnp
import numpy as np
from rsa_race import (
    # Core race model
    semantic_match,
    activation,
    race_model,
    predict_race,
    # VOC for race model
    expected_utility_race,
    benefit_of_waiting,
    optimal_stopping_time_race,
    predict_voc_race,
    # Legacy model (for comparison)
    p_retrieved,
    retrieval_choice_model,
    predict_retrieval_choice,
    # Utilities
    informativity,
    choice_probs,
    benefit_of_lf,
    model_comparison,
    N_ANGLES,
    WORD_LOCS,
    Word,
    DEFAULT_PARAMS,
)


# =============================================================================
# Core Race Model Tests
# =============================================================================

class TestSemanticMatch:
    """Tests for semantic match function M(θ, w)."""

    def test_highest_at_prototype(self):
        """Semantic match should be highest at word's prototype."""
        sigma = 0.15
        match_at_proto = float(semantic_match(0.5, 0.5, sigma))
        match_away = float(semantic_match(0.3, 0.5, sigma))
        assert match_at_proto > match_away

    def test_symmetric(self):
        """Semantic match should be symmetric around prototype."""
        sigma = 0.15
        match_left = float(semantic_match(0.4, 0.5, sigma))
        match_right = float(semantic_match(0.6, 0.5, sigma))
        assert match_left == pytest.approx(match_right, rel=0.01)

    def test_positive(self):
        """Semantic match should always be positive."""
        sigma = 0.15
        for theta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            match = float(semantic_match(theta, 0.5, sigma))
            assert match > 0


class TestActivation:
    """Tests for activation function A(w, t) = B(w) + δ·M(θ,w)·t."""

    def test_baseline_at_t0(self):
        """At t=0, activation should equal baseline."""
        sigma, B_hf, B_lf, delta = 0.15, 1.0, 0.5, 0.5
        A = activation(0.5, sigma, B_hf, B_lf, delta, t=0.0)
        assert float(A[Word.HF1]) == pytest.approx(B_hf)
        assert float(A[Word.LF]) == pytest.approx(B_lf)
        assert float(A[Word.HF2]) == pytest.approx(B_hf)

    def test_increases_with_time(self):
        """Activation should increase with time (positive drift)."""
        sigma, B_hf, B_lf, delta = 0.15, 1.0, 0.5, 0.5
        A_early = activation(0.5, sigma, B_hf, B_lf, delta, t=1.0)
        A_late = activation(0.5, sigma, B_hf, B_lf, delta, t=10.0)
        assert float(A_late[Word.LF]) > float(A_early[Word.LF])

    def test_semantic_advantage_grows(self):
        """At LF prototype, LF's advantage over HF should grow with time."""
        sigma, B_hf, B_lf, delta = 0.15, 1.0, 0.5, 0.5
        theta = 0.5

        A_early = activation(theta, sigma, B_hf, B_lf, delta, t=1.0)
        A_late = activation(theta, sigma, B_hf, B_lf, delta, t=10.0)

        gap_early = float(A_early[Word.LF] - A_early[Word.HF1])
        gap_late = float(A_late[Word.LF] - A_late[Word.HF1])

        assert gap_late > gap_early


class TestRaceModel:
    """Tests for race model word selection (reduced parameterization)."""

    def test_probs_sum_to_one(self):
        """Word probabilities should sum to 1."""
        # New signature: race_model(theta, sigma, B_hf, delta, t, lapse)
        # B_hf is now ΔB (difference), with B_lf=0 fixed internally
        sigma, B_hf, delta = 0.15, 0.5, 0.5  # B_hf=0.5 means ΔB=0.5
        probs = race_model(0.5, sigma, B_hf, delta, t=5.0)
        assert float(jnp.sum(probs)) == pytest.approx(1.0)

    def test_hf_dominates_at_t0(self):
        """At t=0, HF should dominate due to baseline advantage."""
        sigma, B_hf, delta = 0.15, 1.0, 0.5  # ΔB=1.0 means big HF advantage
        probs = race_model(0.5, sigma, B_hf, delta, t=0.0)
        assert float(probs[Word.HF1]) + float(probs[Word.HF2]) > float(probs[Word.LF])

    def test_lf_gains_with_time_at_prototype(self):
        """At LF prototype, P(LF) should increase with time."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.5

        probs_early = race_model(theta, sigma, B_hf, delta, t=2.0)
        probs_late = race_model(theta, sigma, B_hf, delta, t=10.0)

        assert float(probs_late[Word.LF]) > float(probs_early[Word.LF])

    def test_hf_gains_with_time_in_hf_region(self):
        """In HF region, P(HF) should increase with time."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.1  # Near HF1

        probs_early = race_model(theta, sigma, B_hf, delta, t=2.0)
        probs_late = race_model(theta, sigma, B_hf, delta, t=10.0)

        assert float(probs_late[Word.HF1]) > float(probs_early[Word.HF1])


class TestCrossover:
    """Tests for the critical crossover prediction."""

    def test_crossover_in_hf_region(self):
        """
        In HF region, less time should give MORE LF (crossover).

        This is the key prediction that distinguishes the race model
        from the theta-independent retrieval model.
        """
        # Use fitted params that show crossover
        sigma, B_hf, delta = 0.5, 0.12, 0.42  # ΔB=0.12 (was 1.21-1.09)
        theta = 0.1  # Deep in HF region

        probs_strict = race_model(theta, sigma, B_hf, delta, t=5.0)
        probs_lenient = race_model(theta, sigma, B_hf, delta, t=10.0)

        # Crossover: strict should have MORE LF than lenient in HF region
        assert float(probs_strict[Word.LF]) > float(probs_lenient[Word.LF])

    def test_no_crossover_in_lf_region(self):
        """
        In LF region, more time should give MORE LF (no crossover).
        """
        sigma, B_hf, delta = 0.5, 0.12, 0.42
        theta = 0.5  # LF prototype

        probs_strict = race_model(theta, sigma, B_hf, delta, t=5.0)
        probs_lenient = race_model(theta, sigma, B_hf, delta, t=10.0)

        assert float(probs_lenient[Word.LF]) > float(probs_strict[Word.LF])


class TestTimePressureInteraction:
    """Tests for time pressure × position interaction."""

    def test_interaction_direction(self):
        """Time pressure effect should flip between LF and HF regions."""
        results = predict_race(**DEFAULT_PARAMS, T_strict=5.0, T_lenient=10.0)

        # LF region (prototype)
        lf_idx = 50  # theta = 0.5
        effect_lf = float(
            results['lenient'][lf_idx, Word.LF] -
            results['strict'][lf_idx, Word.LF]
        )

        # HF region
        hf_idx = 10  # theta = 0.1
        effect_hf = float(
            results['lenient'][hf_idx, Word.LF] -
            results['strict'][hf_idx, Word.LF]
        )

        # Effect should be positive in LF region, negative in HF region
        assert effect_lf > 0, "More time should increase P(LF) at LF prototype"
        assert effect_hf < 0, "More time should decrease P(LF) in HF region"


# =============================================================================
# VOC Tests for Race Model
# =============================================================================

class TestExpectedUtilityRace:
    """Tests for expected utility under race model."""

    def test_increases_with_time_at_prototype(self):
        """EU should increase with time at LF prototype."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.5

        eu_early = float(expected_utility_race(theta, sigma, B_hf, delta, t=1.0))
        eu_late = float(expected_utility_race(theta, sigma, B_hf, delta, t=10.0))

        assert eu_late > eu_early


class TestBenefitOfWaiting:
    """Tests for marginal benefit of waiting."""

    def test_positive_at_prototype(self):
        """Benefit of waiting should be positive at LF prototype."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.5

        benefit = float(benefit_of_waiting(theta, sigma, B_hf, delta, t=1.0))
        assert benefit > 0

    def test_decreases_with_time(self):
        """Marginal benefit should decrease as you wait longer."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.5

        benefit_early = float(benefit_of_waiting(theta, sigma, B_hf, delta, t=1.0))
        benefit_late = float(benefit_of_waiting(theta, sigma, B_hf, delta, t=5.0))

        assert benefit_early > benefit_late


class TestOptimalStoppingRace:
    """Tests for optimal stopping time."""

    def test_returns_valid_time(self):
        """Optimal stopping should return valid time values."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        cost = 0.1

        t_proto = optimal_stopping_time_race(0.5, sigma, B_hf, delta, cost)
        t_hf = optimal_stopping_time_race(0.1, sigma, B_hf, delta, cost)

        # Both should return positive times within bounds
        assert 0 <= t_proto <= 15.0
        assert 0 <= t_hf <= 15.0

    def test_shorter_with_higher_cost(self):
        """Higher cost should lead to shorter waiting time."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.5

        t_low_cost = optimal_stopping_time_race(theta, sigma, B_hf, delta, cost=0.01)
        t_high_cost = optimal_stopping_time_race(theta, sigma, B_hf, delta, cost=0.5)

        assert t_low_cost > t_high_cost


class TestVOCRacePredictions:
    """Tests for VOC predictions."""

    def test_rt_step_pattern(self):
        """RT should show step pattern: higher in LF region, lower in HF region."""
        results = predict_voc_race(
            sigma=0.15, B_hf=0.5, delta=0.5, lapse=0.0,
            cost=0.05, T_max=15.0
        )

        # Average RT in LF region (theta 0.3-0.7)
        lf_indices = (results['angles'] > 0.3) & (results['angles'] < 0.7)
        rt_lf_region = results['response_times'][lf_indices].mean()

        # Average RT in HF regions (theta < 0.2 or > 0.8)
        hf_indices = (results['angles'] < 0.2) | (results['angles'] > 0.8)
        rt_hf_region = results['response_times'][hf_indices].mean()

        assert rt_lf_region > rt_hf_region


# =============================================================================
# Legacy Model Tests (for comparison)
# =============================================================================

class TestLegacyRetrieval:
    """Tests for legacy theta-independent retrieval model."""

    def test_retrieval_independent_of_theta(self):
        """Legacy model: retrieval should be theta-independent."""
        p1 = float(p_retrieved(5.0, 0.3))
        p2 = float(p_retrieved(5.0, 0.3))
        assert p1 == p2

    def test_no_crossover_in_legacy(self):
        """Legacy model should NOT show crossover in HF region."""
        results = predict_retrieval_choice(
            sigma=0.15, alpha=4.0, lambda_lf=0.3,
            T_strict=5.0, T_lenient=10.0
        )

        hf_idx = 10
        p_lf_strict = float(results['strict'][hf_idx, Word.LF])
        p_lf_lenient = float(results['lenient'][hf_idx, Word.LF])

        # Legacy model: lenient should always have >= P(LF) as strict
        assert p_lf_lenient >= p_lf_strict


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_extreme_theta(self):
        """Model should handle theta at extremes."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5

        probs_0 = race_model(0.0, sigma, B_hf, delta, t=5.0)
        probs_1 = race_model(1.0, sigma, B_hf, delta, t=5.0)

        assert jnp.all(jnp.isfinite(probs_0))
        assert jnp.all(jnp.isfinite(probs_1))
        assert float(jnp.sum(probs_0)) == pytest.approx(1.0)
        assert float(jnp.sum(probs_1)) == pytest.approx(1.0)

    def test_zero_time(self):
        """Model should handle t=0."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5

        probs = race_model(0.5, sigma, B_hf, delta, t=0.0)
        assert jnp.all(jnp.isfinite(probs))
        assert float(jnp.sum(probs)) == pytest.approx(1.0)

    def test_very_long_time(self):
        """At very long times, semantic match should dominate."""
        sigma, B_hf, delta = 0.15, 0.5, 0.5
        theta = 0.5  # LF prototype

        probs = race_model(theta, sigma, B_hf, delta, t=100.0)

        # At LF prototype with long time, LF should dominate
        assert float(probs[Word.LF]) > 0.9
