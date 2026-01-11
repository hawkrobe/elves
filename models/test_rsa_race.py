"""
Tests for RSA Race Model (memo implementation).

Tests verify:
1. Basic model outputs (probabilities sum to 1, correct shape)
2. Qualitative predictions (LF highest at prototype, frequency effects)
3. Time pressure interaction (key empirical prediction)
"""

import pytest
import jax.numpy as jnp
import numpy as np
from rsa_race import (
    semantic_match,
    activation,
    retrieval_utility,
    S1_race,
    S1_voc,
    predict_time_pressure,
    predict_voc,
    N_ANGLES,
    N_TIMES,
    WORD_LOCS,
    WORD_FREQ,
    Word,
)


class TestSemanticMatch:
    """Tests for semantic match computation."""

    def test_perfect_match_at_center(self):
        """Semantic match should be 1.0 at word's center."""
        sigma = 0.12
        # LF word is at position 0.5
        match = float(semantic_match(0.5, Word.LF, sigma))
        assert match == pytest.approx(1.0, rel=0.01)

    def test_match_decreases_with_distance(self):
        """Match should decrease as target moves away from word center."""
        sigma = 0.12
        match_close = float(semantic_match(0.45, Word.LF, sigma))
        match_far = float(semantic_match(0.3, Word.LF, sigma))
        assert match_close > match_far

    def test_match_symmetric(self):
        """Match should be symmetric around word center."""
        sigma = 0.12
        match_left = float(semantic_match(0.4, Word.LF, sigma))
        match_right = float(semantic_match(0.6, Word.LF, sigma))
        assert match_left == pytest.approx(match_right, rel=0.01)

    def test_match_bounded(self):
        """Match should be in [0, 1]."""
        sigma = 0.12
        for theta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for w in [Word.HF1, Word.LF, Word.HF2]:
                match = float(semantic_match(theta, w, sigma))
                assert 0 <= match <= 1


class TestActivation:
    """Tests for activation dynamics."""

    def test_activation_increases_with_time(self):
        """Activation should increase over time."""
        sigma, baseline_hf, baseline_lf, drift_rate = 0.12, 0.4, 0.25, 2.0
        theta = 0.5

        act_early = float(activation(theta, Word.LF, 2, sigma, baseline_hf, baseline_lf, drift_rate))
        act_late = float(activation(theta, Word.LF, 8, sigma, baseline_hf, baseline_lf, drift_rate))
        assert act_late > act_early

    def test_hf_higher_baseline(self):
        """HF words should have higher activation at t=0."""
        sigma, baseline_hf, baseline_lf, drift_rate = 0.12, 0.4, 0.25, 2.0
        theta = 0.5

        act_hf = float(activation(theta, Word.HF1, 0, sigma, baseline_hf, baseline_lf, drift_rate))
        act_lf = float(activation(theta, Word.LF, 0, sigma, baseline_hf, baseline_lf, drift_rate))
        assert act_hf > act_lf

    def test_lf_catches_up_at_prototype(self):
        """At LF prototype, LF should overtake HF given enough time."""
        sigma, baseline_hf, baseline_lf, drift_rate = 0.12, 0.4, 0.25, 2.0
        theta = 0.5  # LF prototype

        act_hf_late = float(activation(theta, Word.HF1, 10, sigma, baseline_hf, baseline_lf, drift_rate))
        act_lf_late = float(activation(theta, Word.LF, 10, sigma, baseline_hf, baseline_lf, drift_rate))
        assert act_lf_late > act_hf_late


class TestS1Race:
    """Tests for race model speaker."""

    def test_output_shape(self):
        """S1_race should return [N_ANGLES, N_TIMES, N_WORDS] array."""
        pred = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                       drift_rate=2.0, temperature=0.25)
        assert pred.shape == (N_ANGLES, N_TIMES, 3)

    def test_probabilities_sum_to_one(self):
        """Word probabilities should sum to 1 for each (angle, time)."""
        pred = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                       drift_rate=2.0, temperature=0.25)
        sums = pred.sum(axis=2)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_probabilities_non_negative(self):
        """All probabilities should be non-negative."""
        pred = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                       drift_rate=2.0, temperature=0.25)
        assert jnp.all(pred >= 0)

    def test_lf_highest_at_prototype(self):
        """P(LF) should be highest at LF prototype."""
        pred = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                       drift_rate=2.0, temperature=0.25)

        # At time=10 (lenient), LF should dominate at prototype
        prototype_idx = 50  # theta = 0.5
        boundary_idx = 25   # theta = 0.25

        p_lf_prototype = float(pred[prototype_idx, 10, Word.LF])
        p_lf_boundary = float(pred[boundary_idx, 10, Word.LF])

        assert p_lf_prototype > p_lf_boundary


class TestTimePressure:
    """Tests for time pressure effects - the key empirical prediction."""

    def test_time_pressure_reduces_p_lf(self):
        """Strict deadline should reduce P(LF) at prototype."""
        results = predict_time_pressure(
            sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
            drift_rate=2.0, temperature=0.25,
            time_strict=8, time_lenient=10
        )

        prototype_idx = 50
        p_lf_lenient = float(results['lenient'][prototype_idx, Word.LF])
        p_lf_strict = float(results['strict'][prototype_idx, Word.LF])

        assert p_lf_lenient >= p_lf_strict

    def test_interaction_exists(self):
        """Time pressure effect should differ by position (interaction)."""
        results = predict_time_pressure(
            sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
            drift_rate=2.0, temperature=0.25,
            time_strict=8, time_lenient=10
        )

        # Compute effect at different positions
        prototype_idx = 50
        boundary_idx = 25

        effect_prototype = float(
            results['lenient'][prototype_idx, Word.LF] -
            results['strict'][prototype_idx, Word.LF]
        )
        effect_boundary = float(
            results['lenient'][boundary_idx, Word.LF] -
            results['strict'][boundary_idx, Word.LF]
        )

        # Effects should be different (interaction)
        assert effect_prototype != pytest.approx(effect_boundary, abs=0.001)


class TestVOC:
    """Tests for Value of Computation model."""

    def test_voc_output_shape(self):
        """S1_voc should return [N_ANGLES, N_WORDS] array."""
        pred = S1_voc(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                      drift_rate=2.0, alpha=1.0, beta_cost=1.0)
        assert pred.shape == (N_ANGLES, 3)

    def test_voc_probabilities_sum_to_one(self):
        """VOC probabilities should sum to 1."""
        pred = S1_voc(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                      drift_rate=2.0, alpha=1.0, beta_cost=1.0)
        sums = pred.sum(axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

    def test_voc_lf_high_at_prototype(self):
        """VOC model should predict high P(LF) at prototype."""
        results = predict_voc(
            sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
            drift_rate=2.0, alpha=1.0, beta_cost=1.0
        )

        prototype_idx = 50
        p_lf = float(results['predictions'][prototype_idx, Word.LF])
        assert p_lf > 0.5


class TestFrequencyEffects:
    """Tests for frequency manipulation effects."""

    def test_higher_baseline_increases_hf_advantage(self):
        """Larger HF baseline should increase HF selection."""
        pred_small = S1_race(sigma=0.12, baseline_hf=0.3, baseline_lf=0.25,
                             drift_rate=2.0, temperature=0.25)
        pred_large = S1_race(sigma=0.12, baseline_hf=0.6, baseline_lf=0.25,
                             drift_rate=2.0, temperature=0.25)

        # At boundary, HF advantage should be larger with higher baseline
        boundary_idx = 25
        p_hf_small = float(pred_small[boundary_idx, 5, Word.HF1])
        p_hf_large = float(pred_large[boundary_idx, 5, Word.HF1])

        assert p_hf_large > p_hf_small


class TestEdgeCases:
    """Tests for edge cases and parameter extremes."""

    def test_extreme_theta_values(self):
        """Model should handle theta at extremes (0 and 1)."""
        pred = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                       drift_rate=2.0, temperature=0.25)

        # Should not have NaN or Inf
        assert jnp.all(jnp.isfinite(pred[0, :, :]))    # theta = 0
        assert jnp.all(jnp.isfinite(pred[-1, :, :]))   # theta = 1

    def test_very_low_temperature(self):
        """Low temperature should make choices more deterministic."""
        pred_low = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                           drift_rate=2.0, temperature=0.1)
        pred_high = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                            drift_rate=2.0, temperature=1.0)

        # Low temperature should have higher max probability
        max_prob_low = float(pred_low[50, 10, :].max())
        max_prob_high = float(pred_high[50, 10, :].max())

        assert max_prob_low > max_prob_high

    def test_zero_drift_rate(self):
        """With zero drift, only baseline matters."""
        pred = S1_race(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                       drift_rate=0.0, temperature=0.25)

        # Early and late should be the same
        assert jnp.allclose(pred[:, 0, :], pred[:, 10, :], atol=1e-5)
