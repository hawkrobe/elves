"""
Tests for RSA + retrieval model.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from rsa_race import (
    p_retrieved,
    informativity,
    choice_probs,
    nearest_hf,
    retrieval_choice_model,
    fixed_resource_model,
    informativity_only_model,
    predict_retrieval_choice,
    predict_fixed_resource,
    model_comparison,
    benefit_of_lf,
    optimal_stopping_time,
    voc_model,
    voc_response_time,
    predict_voc,
    N_ANGLES,
    WORD_LOCS,
    Word,
)


class TestRetrieval:
    """Tests for retrieval probability."""

    def test_retrieval_zero_at_t0(self):
        """P(retrieved | t=0) should be 0."""
        assert float(p_retrieved(0.0, 0.3)) == pytest.approx(0.0)

    def test_retrieval_increases_with_time(self):
        """P(retrieved) should increase with time."""
        p_early = float(p_retrieved(2.0, 0.3))
        p_late = float(p_retrieved(10.0, 0.3))
        assert p_late > p_early

    def test_retrieval_approaches_one(self):
        """P(retrieved) should approach 1 as t → ∞."""
        p = float(p_retrieved(100.0, 0.3))
        assert p > 0.99

    def test_higher_lambda_faster_retrieval(self):
        """Higher λ should mean faster retrieval."""
        p_slow = float(p_retrieved(5.0, 0.2))
        p_fast = float(p_retrieved(5.0, 0.5))
        assert p_fast > p_slow


class TestInformativity:
    """Tests for informativity (log likelihood)."""

    def test_informativity_highest_at_prototype(self):
        """Informativity should be highest at word's prototype."""
        sigma = 0.15
        info_at_proto = float(informativity(0.5, Word.LF, sigma))
        info_away = float(informativity(0.3, Word.LF, sigma))
        assert info_at_proto > info_away

    def test_informativity_symmetric(self):
        """Informativity should be symmetric around prototype."""
        sigma = 0.15
        info_left = float(informativity(0.4, Word.LF, sigma))
        info_right = float(informativity(0.6, Word.LF, sigma))
        assert info_left == pytest.approx(info_right, rel=0.01)


class TestChoiceProbs:
    """Tests for choice probabilities given all words available."""

    def test_probs_sum_to_one(self):
        """Choice probabilities should sum to 1."""
        probs = choice_probs(0.5, sigma=0.15, alpha=4.0)
        assert float(jnp.sum(probs)) == pytest.approx(1.0)

    def test_lf_highest_at_prototype(self):
        """P(LF) should be highest at LF prototype."""
        probs = choice_probs(0.5, sigma=0.15, alpha=4.0)
        assert float(probs[Word.LF]) > float(probs[Word.HF1])
        assert float(probs[Word.LF]) > float(probs[Word.HF2])

    def test_hf1_highest_at_left(self):
        """P(HF1) should be highest near θ=0."""
        probs = choice_probs(0.1, sigma=0.15, alpha=4.0)
        assert float(probs[Word.HF1]) > float(probs[Word.LF])
        assert float(probs[Word.HF1]) > float(probs[Word.HF2])

    def test_higher_alpha_more_peaked(self):
        """Higher α should make distribution more peaked."""
        probs_low = choice_probs(0.5, sigma=0.15, alpha=1.0)
        probs_high = choice_probs(0.5, sigma=0.15, alpha=10.0)
        # Higher alpha should give more probability to LF at prototype
        assert float(probs_high[Word.LF]) > float(probs_low[Word.LF])


class TestNearestHF:
    """Tests for nearest HF selection."""

    def test_hf1_selected_left(self):
        """HF1 should be selected for θ < 0.5."""
        probs = nearest_hf(0.3)
        assert float(probs[Word.HF1]) == 1.0
        assert float(probs[Word.LF]) == 0.0
        assert float(probs[Word.HF2]) == 0.0

    def test_hf2_selected_right(self):
        """HF2 should be selected for θ >= 0.5."""
        probs = nearest_hf(0.7)
        assert float(probs[Word.HF1]) == 0.0
        assert float(probs[Word.LF]) == 0.0
        assert float(probs[Word.HF2]) == 1.0


class TestRetrievalChoiceModel:
    """Tests for the full retrieval × choice model."""

    def test_probs_sum_to_one(self):
        """Production probabilities should sum to 1."""
        probs = retrieval_choice_model(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)
        assert float(jnp.sum(probs)) == pytest.approx(1.0)

    def test_time_pressure_reduces_lf(self):
        """Shorter deadline should reduce P(LF)."""
        probs_lenient = retrieval_choice_model(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)
        probs_strict = retrieval_choice_model(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=5.0)
        assert float(probs_strict[Word.LF]) < float(probs_lenient[Word.LF])

    def test_lf_higher_at_prototype_than_boundary(self):
        """P(LF) should be higher at prototype than boundary."""
        probs_proto = retrieval_choice_model(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)
        probs_boundary = retrieval_choice_model(0.25, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)
        assert float(probs_proto[Word.LF]) > float(probs_boundary[Word.LF])


class TestTimePressureInteraction:
    """Tests for the critical time pressure × position interaction."""

    def test_interaction_exists(self):
        """Time pressure effect should be larger at prototype than boundary."""
        results = predict_retrieval_choice(
            sigma=0.15, alpha=4.0, lambda_lf=0.3,
            T_strict=5.0, T_lenient=10.0
        )

        prototype_idx = 50  # θ = 0.5
        boundary_idx = 25   # θ = 0.25

        # Effect at prototype
        effect_proto = float(
            results['lenient'][prototype_idx, Word.LF] -
            results['strict'][prototype_idx, Word.LF]
        )

        # Effect at boundary
        effect_boundary = float(
            results['lenient'][boundary_idx, Word.LF] -
            results['strict'][boundary_idx, Word.LF]
        )

        # Interaction: effect should be larger at prototype
        assert effect_proto > effect_boundary

    def test_fixed_model_no_interaction(self):
        """Fixed resource model should show NO interaction (parallel shift)."""
        angles = jnp.linspace(0, 1, N_ANGLES)

        # Two "conditions" with different fixed retrieval probabilities
        pred_high = predict_fixed_resource(sigma=0.15, alpha=4.0, p_retrieve_fixed=0.9)
        pred_low = predict_fixed_resource(sigma=0.15, alpha=4.0, p_retrieve_fixed=0.6)

        prototype_idx = 50
        boundary_idx = 25

        # Compute effects
        effect_proto = float(
            pred_high['predictions'][prototype_idx, Word.LF] -
            pred_low['predictions'][prototype_idx, Word.LF]
        )
        effect_boundary = float(
            pred_high['predictions'][boundary_idx, Word.LF] -
            pred_low['predictions'][boundary_idx, Word.LF]
        )

        # For fixed model, effects should be proportional (parallel shift)
        # The ratio of effects should equal ratio of P(LF|all) at each position
        ratio_proto = float(pred_high['predictions'][prototype_idx, Word.LF] /
                           pred_low['predictions'][prototype_idx, Word.LF])
        ratio_boundary = float(pred_high['predictions'][boundary_idx, Word.LF] /
                              pred_low['predictions'][boundary_idx, Word.LF])

        # Ratios should be similar (parallel shift)
        assert ratio_proto == pytest.approx(ratio_boundary, rel=0.1)


class TestModelComparison:
    """Tests comparing the three models."""

    def test_informativity_only_no_frequency_effect(self):
        """Informativity-only model should show no frequency asymmetry."""
        results = model_comparison(sigma=0.15, alpha=4.0)

        # At prototype (0.5), P(LF) should be high for all models
        proto_idx = 50
        p_lf_info = float(results['informativity_only'][proto_idx, Word.LF])
        p_lf_retrieval = float(results['retrieval_choice'][proto_idx, Word.LF])

        # Informativity-only should have HIGHER P(LF) because no retrieval cost
        assert p_lf_info >= p_lf_retrieval

    def test_retrieval_choice_shows_frequency_effect(self):
        """Retrieval-choice model should show frequency effect (LF suppressed)."""
        results = model_comparison(sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)

        # Compare to informativity-only at prototype
        proto_idx = 50
        p_lf_retrieval = float(results['retrieval_choice'][proto_idx, Word.LF])
        p_lf_info = float(results['informativity_only'][proto_idx, Word.LF])

        # Retrieval cost should reduce P(LF)
        assert p_lf_retrieval < p_lf_info


class TestEdgeCases:
    """Tests for edge cases."""

    def test_extreme_theta_values(self):
        """Model should handle θ at extremes."""
        probs_0 = retrieval_choice_model(0.0, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)
        probs_1 = retrieval_choice_model(1.0, sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0)

        assert jnp.all(jnp.isfinite(probs_0))
        assert jnp.all(jnp.isfinite(probs_1))
        assert float(jnp.sum(probs_0)) == pytest.approx(1.0)
        assert float(jnp.sum(probs_1)) == pytest.approx(1.0)

    def test_very_high_alpha(self):
        """Very high α should make choices nearly deterministic."""
        probs = choice_probs(0.5, sigma=0.15, alpha=100.0)
        assert float(probs[Word.LF]) > 0.99

    def test_zero_lambda(self):
        """λ=0 should mean LF never retrieved."""
        probs = retrieval_choice_model(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.0, T=10.0)
        # With LF never retrieved, should get HF2 (nearest to 0.5)
        assert float(probs[Word.LF]) == pytest.approx(0.0)
        assert float(probs[Word.HF2]) == pytest.approx(1.0)


class TestVOC:
    """Tests for Value of Computation (optimal stopping) model."""

    def test_benefit_higher_at_prototype(self):
        """Benefit of LF should be higher at prototype than boundary."""
        benefit_proto = float(benefit_of_lf(0.5, sigma=0.15, alpha=4.0))
        benefit_boundary = float(benefit_of_lf(0.25, sigma=0.15, alpha=4.0))
        assert benefit_proto > benefit_boundary

    def test_benefit_positive_at_prototype(self):
        """Benefit should be positive at LF prototype."""
        benefit = float(benefit_of_lf(0.5, sigma=0.15, alpha=4.0))
        assert benefit > 0

    def test_stopping_time_longer_at_prototype(self):
        """Should wait longer at prototype (high benefit) than boundary."""
        t_proto = float(optimal_stopping_time(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1))
        t_boundary = float(optimal_stopping_time(0.25, sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1))
        assert t_proto > t_boundary

    def test_higher_cost_reduces_waiting(self):
        """Higher cost should reduce optimal stopping time."""
        t_low_cost = float(optimal_stopping_time(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.05))
        t_high_cost = float(optimal_stopping_time(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.5))
        assert t_low_cost > t_high_cost

    def test_voc_probs_sum_to_one(self):
        """VOC model probabilities should sum to 1."""
        probs = voc_model(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1, T_max=15.0)
        assert float(jnp.sum(probs)) == pytest.approx(1.0)

    def test_voc_response_time_capped_by_deadline(self):
        """Response time should be capped by deadline."""
        rt = float(voc_response_time(0.5, sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.01, T_max=5.0))
        assert rt <= 5.0

    def test_voc_rt_varies_with_position(self):
        """Response times should vary with position."""
        results = predict_voc(sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1, T_max=15.0)

        rt_proto = float(results['response_times'][50])  # θ = 0.5
        rt_boundary = float(results['response_times'][25])  # θ = 0.25

        # Should wait longer at prototype
        assert rt_proto > rt_boundary

    def test_voc_lf_higher_at_prototype(self):
        """P(LF) should still be higher at prototype under VOC."""
        results = predict_voc(sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1, T_max=15.0)

        p_lf_proto = float(results['predictions'][50, Word.LF])
        p_lf_boundary = float(results['predictions'][25, Word.LF])

        assert p_lf_proto > p_lf_boundary
