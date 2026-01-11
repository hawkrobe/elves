"""
RSA with frequency-driven retrieval.

Retrieval: P(LF retrieved by t) = 1 - exp(-λt)  [frequency-driven, θ-independent]
Choice: P(w | θ, available) ∝ exp(α × Info(w | θ))  [standard RSA]
Production: P(LF | θ, T) = P(retrieved) × P(choose LF | all available)
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from enum import IntEnum

N_ANGLES = 101

Angle = jnp.linspace(0, 1, N_ANGLES)

class Word(IntEnum):
    HF1 = 0
    LF = 1
    HF2 = 2

WORD_LOCS = jnp.array([0.0, 0.5, 1.0])


# =============================================================================
# Retrieval (frequency-driven, θ-independent)
# =============================================================================

@jax.jit
def p_retrieved(t, lambda_lf):
    """P(LF retrieved by time t) = 1 - exp(-λt). HF always available."""
    return 1.0 - jnp.exp(-lambda_lf * t)


# =============================================================================
# Informativity (standard RSA)
# =============================================================================

@jax.jit
def informativity(theta, w, sigma):
    """Info(w | θ) = log N(θ | μ_w, σ)"""
    return jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)


@jax.jit
def choice_probs(theta, sigma, alpha):
    """P(choose w | θ, all available) ∝ exp(α × Info(w | θ))"""
    info = jnp.array([informativity(theta, w, sigma) for w in Word])
    log_probs = alpha * info
    log_probs = log_probs - jax.scipy.special.logsumexp(log_probs)
    return jnp.exp(log_probs)


@jax.jit
def nearest_hf(theta):
    """Returns probability distribution over words when only HF available."""
    # Deterministically choose nearest HF
    p_hf1 = (theta < 0.5).astype(float)
    p_hf2 = (theta >= 0.5).astype(float)
    return jnp.array([p_hf1, 0.0, p_hf2])


# =============================================================================
# Production Models
# =============================================================================

@jax.jit
def retrieval_choice_model(theta, sigma, alpha, lambda_lf, T):
    """
    Full model: retrieval × choice.

    P(w | θ, T) = P(LF retrieved) × P(choose w | all available)
                + P(LF not retrieved) × P(choose w | only HF)
    """
    p_ret = p_retrieved(T, lambda_lf)
    p_choice = choice_probs(theta, sigma, alpha)
    p_hf_only = nearest_hf(theta)

    return p_ret * p_choice + (1 - p_ret) * p_hf_only


@jax.jit
def fixed_resource_model(theta, sigma, alpha, p_retrieve_fixed):
    """
    Fixed retrieval probability (no time/deadline sensitivity).
    Predicts main effect of frequency but NO interaction with position.
    """
    p_choice = choice_probs(theta, sigma, alpha)
    p_hf_only = nearest_hf(theta)

    return p_retrieve_fixed * p_choice + (1 - p_retrieve_fixed) * p_hf_only


@jax.jit
def informativity_only_model(theta, sigma, alpha):
    """
    No retrieval cost, just softmax over informativity.
    Predicts no frequency effect.
    """
    return choice_probs(theta, sigma, alpha)


# =============================================================================
# Vectorized prediction functions
# =============================================================================

def predict_retrieval_choice(sigma=0.15, alpha=4.0, lambda_lf=0.3,
                              T_strict=5.0, T_lenient=10.0):
    """Generate predictions for strict vs lenient deadline."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    # Vectorize over angles
    pred_fn = jax.vmap(lambda th: retrieval_choice_model(th, sigma, alpha, lambda_lf, T_strict))
    pred_strict = pred_fn(angles)

    pred_fn = jax.vmap(lambda th: retrieval_choice_model(th, sigma, alpha, lambda_lf, T_lenient))
    pred_lenient = pred_fn(angles)

    return {
        'strict': pred_strict,
        'lenient': pred_lenient,
        'angles': angles,
        'p_retrieved_strict': float(p_retrieved(T_strict, lambda_lf)),
        'p_retrieved_lenient': float(p_retrieved(T_lenient, lambda_lf)),
    }


def predict_fixed_resource(sigma=0.15, alpha=4.0, p_retrieve_fixed=0.7):
    """Generate predictions for fixed resource model."""
    angles = jnp.linspace(0, 1, N_ANGLES)
    pred_fn = jax.vmap(lambda th: fixed_resource_model(th, sigma, alpha, p_retrieve_fixed))

    return {
        'predictions': pred_fn(angles),
        'angles': angles,
    }


def predict_informativity_only(sigma=0.15, alpha=4.0):
    """Generate predictions for informativity-only model."""
    angles = jnp.linspace(0, 1, N_ANGLES)
    pred_fn = jax.vmap(lambda th: informativity_only_model(th, sigma, alpha))

    return {
        'predictions': pred_fn(angles),
        'angles': angles,
    }


def predict_frequency_manipulation(sigma=0.15, alpha=4.0, T=10.0,
                                    lambda_1to2=0.5, lambda_1to4=0.2):
    """Compare different frequency ratios (λ values)."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_fn_1to2 = jax.vmap(lambda th: retrieval_choice_model(th, sigma, alpha, lambda_1to2, T))
    pred_fn_1to4 = jax.vmap(lambda th: retrieval_choice_model(th, sigma, alpha, lambda_1to4, T))

    return {
        'ratio_1to2': pred_fn_1to2(angles),
        'ratio_1to4': pred_fn_1to4(angles),
        'angles': angles,
    }


def model_comparison(sigma=0.15, alpha=4.0, lambda_lf=0.3, T=10.0, p_fixed=0.7):
    """Compare all three models."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_retrieval = jax.vmap(lambda th: retrieval_choice_model(th, sigma, alpha, lambda_lf, T))(angles)
    pred_fixed = jax.vmap(lambda th: fixed_resource_model(th, sigma, alpha, p_fixed))(angles)
    pred_info = jax.vmap(lambda th: informativity_only_model(th, sigma, alpha))(angles)

    return {
        'retrieval_choice': pred_retrieval,
        'fixed_resource': pred_fixed,
        'informativity_only': pred_info,
        'angles': angles,
    }
