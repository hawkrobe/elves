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


# =============================================================================
# VOC: Value of Computation (Optimal Stopping)
# =============================================================================

@jax.jit
def expected_utility_all_available(theta, sigma, alpha):
    """E[Info(w | θ)] when all words available, choosing by RSA."""
    probs = choice_probs(theta, sigma, alpha)
    info = jnp.array([informativity(theta, w, sigma) for w in Word])
    return jnp.sum(probs * info)


@jax.jit
def utility_hf_only(theta, sigma):
    """Utility of responding with nearest HF (deterministic)."""
    # nearest HF
    hf_idx = jnp.where(theta < 0.5, Word.HF1, Word.HF2)
    return informativity(theta, hf_idx, sigma)


@jax.jit
def benefit_of_lf(theta, sigma, alpha):
    """
    Improvement in expected utility if LF becomes available.

    benefit(θ) = E[Info | all available] - Info(nearest HF | θ)

    High at prototype (LF much better than HF).
    Low at boundary (LF only marginally better).
    """
    eu_all = expected_utility_all_available(theta, sigma, alpha)
    eu_hf = utility_hf_only(theta, sigma)
    return eu_all - eu_hf


@jax.jit
def optimal_stopping_time(theta, sigma, alpha, lambda_lf, cost):
    """
    Optimal time to stop waiting for LF.

    Stop when: marginal cost ≥ marginal benefit
    i.e., when: c ≥ λ × P(not yet retrieved) × benefit(θ)

    For exponential retrieval, this gives:
    t* = (1/λ) × log(λ × benefit(θ) / c)  [if benefit > c/λ, else t*=0]
    """
    benefit = benefit_of_lf(theta, sigma, alpha)

    # If benefit is low enough, respond immediately
    # Otherwise, wait until marginal benefit = marginal cost
    # Marginal benefit at time t: λ × exp(-λt) × benefit
    # Set equal to c: λ × exp(-λt) × benefit = c
    # Solve: t = (1/λ) × log(λ × benefit / c)

    threshold = cost / lambda_lf
    t_star = jnp.where(
        benefit > threshold,
        (1.0 / lambda_lf) * jnp.log(lambda_lf * benefit / cost),
        0.0
    )
    return jnp.maximum(t_star, 0.0)


@jax.jit
def voc_model(theta, sigma, alpha, lambda_lf, cost, T_max):
    """
    VOC model: speaker chooses optimal stopping time, capped by deadline.

    Returns P(w | θ) under optimal stopping.
    """
    # Optimal stopping time (unconstrained)
    t_star = optimal_stopping_time(theta, sigma, alpha, lambda_lf, cost)

    # Actual response time is min(t_star, T_max)
    t_respond = jnp.minimum(t_star, T_max)

    # Production probabilities at response time
    return retrieval_choice_model(theta, sigma, alpha, lambda_lf, t_respond)


@jax.jit
def voc_response_time(theta, sigma, alpha, lambda_lf, cost, T_max):
    """Returns the optimal response time (capped by deadline)."""
    t_star = optimal_stopping_time(theta, sigma, alpha, lambda_lf, cost)
    return jnp.minimum(t_star, T_max)


def predict_voc(sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1, T_max=15.0):
    """
    Generate VOC model predictions.

    Returns word probabilities and response times across positions.
    """
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_fn = jax.vmap(lambda th: voc_model(th, sigma, alpha, lambda_lf, cost, T_max))
    rt_fn = jax.vmap(lambda th: voc_response_time(th, sigma, alpha, lambda_lf, cost, T_max))
    benefit_fn = jax.vmap(lambda th: benefit_of_lf(th, sigma, alpha))

    return {
        'predictions': pred_fn(angles),
        'response_times': rt_fn(angles),
        'benefit': benefit_fn(angles),
        'angles': angles,
    }


def predict_voc_deadline_comparison(sigma=0.15, alpha=4.0, lambda_lf=0.3, cost=0.1,
                                     T_strict=5.0, T_lenient=15.0):
    """
    Compare VOC model under different deadlines.

    Key prediction: Response times should be faster at boundaries
    (low benefit of waiting) than at prototypes (high benefit).
    """
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_strict = jax.vmap(lambda th: voc_model(th, sigma, alpha, lambda_lf, cost, T_strict))(angles)
    pred_lenient = jax.vmap(lambda th: voc_model(th, sigma, alpha, lambda_lf, cost, T_lenient))(angles)

    rt_strict = jax.vmap(lambda th: voc_response_time(th, sigma, alpha, lambda_lf, cost, T_strict))(angles)
    rt_lenient = jax.vmap(lambda th: voc_response_time(th, sigma, alpha, lambda_lf, cost, T_lenient))(angles)

    return {
        'strict': pred_strict,
        'lenient': pred_lenient,
        'rt_strict': rt_strict,
        'rt_lenient': rt_lenient,
        'angles': angles,
    }
