"""
RSA Race Model: Lexical access as activation race.

Model:
    A(w, t) = B(w) + δ · M(θ, w) · t

Where:
    - B(w): baseline activation (higher for HF due to frequency)
    - M(θ, w): semantic match between target θ and word meaning
    - δ: drift rate (how fast semantic information accumulates)
    - t: time available for processing

Selection via softmax over activations at response time.

Key insight: At t=0, only baseline matters (frequency effect).
As t increases, semantic match dominates (informativity effect).
The "cost" of LF is the TIME needed for semantic information to overcome
the frequency disadvantage.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from enum import IntEnum
import numpy as np

N_ANGLES = 101


class Word(IntEnum):
    HF1 = 0
    LF = 1
    HF2 = 2


WORD_LOCS = jnp.array([0.0, 0.5, 1.0])


# =============================================================================
# Core Race Model
# =============================================================================

@jax.jit
def semantic_match(theta, word_loc, sigma):
    """
    M(θ, w): Semantic similarity between target and word meaning.

    Uses Gaussian similarity centered at word's prototype.
    """
    return norm.pdf(theta, loc=word_loc, scale=sigma)


@jax.jit
def activation(theta, sigma, B_hf, B_lf, delta, t):
    """
    A(w, t) = B(w) + δ · M(θ, w) · t

    Returns activation for all three words [HF1, LF, HF2].
    """
    baselines = jnp.array([B_hf, B_lf, B_hf])
    M = jnp.array([semantic_match(theta, loc, sigma) for loc in WORD_LOCS])
    return baselines + delta * M * t


@jax.jit
def race_model(theta, sigma, B_hf, delta, t, lapse=0.0):
    """
    Race model: select word via softmax over activations.

    P(w | θ, t) = (1-lapse) · softmax(A(w,t)) + lapse · uniform

    Uses reduced parameterization with α=1 and B_lf=0 fixed.
    """
    A = activation(theta, sigma, B_hf, B_lf=0.0, delta=delta, t=t)
    log_probs = A  # α=1, so no scaling
    log_probs = log_probs - jax.scipy.special.logsumexp(log_probs)
    probs = jnp.exp(log_probs)
    # Mix with uniform distribution (1/3 for each word)
    return (1 - lapse) * probs + lapse * (1.0 / 3.0)


# =============================================================================
# Informativity (for compatibility and VOC analysis)
# =============================================================================

@jax.jit
def informativity(theta, w, sigma):
    """Info(w | θ) = log P(θ | w) - log likelihood under word meaning."""
    return jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)


@jax.jit
def choice_probs(theta, sigma, alpha):
    """P(choose w | θ) based on informativity (RSA baseline)."""
    info = jnp.array([informativity(theta, w, sigma) for w in Word])
    log_probs = alpha * info
    log_probs = log_probs - jax.scipy.special.logsumexp(log_probs)
    return jnp.exp(log_probs)


# =============================================================================
# Legacy: Theta-independent retrieval model (for comparison)
# =============================================================================

@jax.jit
def p_retrieved(t, lambda_lf):
    """P(LF retrieved by time t) = 1 - exp(-λt). Legacy model."""
    return 1.0 - jnp.exp(-lambda_lf * t)


@jax.jit
def nearest_hf(theta):
    """Distribution over words when only HF available."""
    p_hf1 = (theta < 0.5).astype(float)
    p_hf2 = (theta >= 0.5).astype(float)
    return jnp.array([p_hf1, 0.0, p_hf2])


@jax.jit
def retrieval_choice_model(theta, sigma, alpha, lambda_lf, T):
    """
    Legacy model: theta-independent retrieval.

    P(w | θ, T) = P(ret) × P(choice | all) + (1-P(ret)) × P(choice | HF only)
    """
    p_ret = p_retrieved(T, lambda_lf)
    p_choice = choice_probs(theta, sigma, alpha)
    p_hf_only = nearest_hf(theta)
    return p_ret * p_choice + (1 - p_ret) * p_hf_only


# =============================================================================
# Vectorized Prediction Functions
# =============================================================================

def predict_race(sigma=0.206, B_hf=0.464, delta=0.467, lapse=0.150,
                 T_strict=4.75, T_lenient=7.39):
    """Generate race model predictions for strict vs lenient deadline."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_strict = jax.vmap(
        lambda th: race_model(th, sigma, B_hf, delta, T_strict, lapse)
    )(angles)

    pred_lenient = jax.vmap(
        lambda th: race_model(th, sigma, B_hf, delta, T_lenient, lapse)
    )(angles)

    return {
        'strict': pred_strict,
        'lenient': pred_lenient,
        'angles': angles,
    }


def predict_retrieval_choice(sigma=0.15, alpha=4.0, lambda_lf=0.3,
                              T_strict=5.0, T_lenient=10.0):
    """Generate predictions for legacy retrieval model."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_strict = jax.vmap(
        lambda th: retrieval_choice_model(th, sigma, alpha, lambda_lf, T_strict)
    )(angles)

    pred_lenient = jax.vmap(
        lambda th: retrieval_choice_model(th, sigma, alpha, lambda_lf, T_lenient)
    )(angles)

    return {
        'strict': pred_strict,
        'lenient': pred_lenient,
        'angles': angles,
        'p_retrieved_strict': float(p_retrieved(T_strict, lambda_lf)),
        'p_retrieved_lenient': float(p_retrieved(T_lenient, lambda_lf)),
    }


def predict_fixed_resource(sigma=0.15, alpha=4.0, p_retrieve_fixed=0.7):
    """Generate predictions for fixed resource model (no time sensitivity)."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    def fixed_model(theta):
        p_choice = choice_probs(theta, sigma, alpha)
        p_hf_only = nearest_hf(theta)
        return p_retrieve_fixed * p_choice + (1 - p_retrieve_fixed) * p_hf_only

    return {
        'predictions': jax.vmap(fixed_model)(angles),
        'angles': angles,
    }


def model_comparison(sigma=0.206, B_hf=0.464, delta=0.467, lapse=0.150,
                     lambda_lf=0.3, T=10.0, p_fixed=0.7, alpha=4.0):
    """Compare race model, retrieval model, and fixed model."""
    angles = jnp.linspace(0, 1, N_ANGLES)

    pred_race = jax.vmap(
        lambda th: race_model(th, sigma, B_hf, delta, T, lapse)
    )(angles)

    pred_retrieval = jax.vmap(
        lambda th: retrieval_choice_model(th, sigma, alpha, lambda_lf, T)
    )(angles)

    def fixed_model(theta):
        p_choice = choice_probs(theta, sigma, alpha)
        p_hf_only = nearest_hf(theta)
        return p_fixed * p_choice + (1 - p_fixed) * p_hf_only

    pred_fixed = jax.vmap(fixed_model)(angles)

    return {
        'race': pred_race,
        'retrieval_choice': pred_retrieval,
        'fixed_resource': pred_fixed,
        'angles': angles,
    }


# =============================================================================
# VOC: Value of Computation for Race Model
# =============================================================================

@jax.jit
def expected_utility_race(theta, sigma, B_hf, delta, t, lapse=0.0):
    """
    Expected informativity under race model at time t.

    E[Info] = Σ_w P(w | θ, t) × Info(w | θ)
    """
    probs = race_model(theta, sigma, B_hf, delta, t, lapse)
    info = jnp.array([informativity(theta, w, sigma) for w in Word])
    return jnp.sum(probs * info)


@jax.jit
def benefit_of_waiting(theta, sigma, B_hf, delta, t, lapse=0.0, dt=0.1):
    """
    Marginal benefit of waiting additional dt seconds.

    benefit = E[Info | t + dt] - E[Info | t]
    """
    eu_now = expected_utility_race(theta, sigma, B_hf, delta, t, lapse)
    eu_later = expected_utility_race(theta, sigma, B_hf, delta, t + dt, lapse)
    return (eu_later - eu_now) / dt


@jax.jit
def benefit_of_lf(theta, sigma, alpha):
    """
    Benefit of LF being the most informative word.

    For VOC analysis: how much better is LF than nearest HF?
    """
    info_lf = informativity(theta, Word.LF, sigma)
    hf_idx = jnp.where(theta < 0.5, Word.HF1, Word.HF2)
    info_hf = informativity(theta, hf_idx, sigma)
    return jnp.maximum(info_lf - info_hf, 0.0)


def optimal_stopping_time_race(theta, sigma, B_hf, delta, cost, lapse=0.0,
                                T_max=15.0, dt=0.1):
    """
    Find optimal stopping time for race model.

    Stop when marginal benefit of waiting ≤ marginal cost.
    Uses discrete search (not differentiable).
    """
    times = np.arange(0, T_max, dt)

    for t in times:
        benefit = float(benefit_of_waiting(theta, sigma, B_hf, delta, t, lapse, dt))
        if benefit < cost:
            return t

    return T_max


def predict_voc_race(sigma=0.206, B_hf=0.464, delta=0.467, lapse=0.150,
                     cost=0.1, T_max=15.0):
    """
    Generate VOC predictions for race model.

    Returns word probabilities and response times across positions.
    """
    angles = np.linspace(0, 1, N_ANGLES)

    response_times = []
    predictions = []

    for theta in angles:
        t_opt = optimal_stopping_time_race(
            theta, sigma, B_hf, delta, cost, lapse, T_max
        )
        response_times.append(t_opt)
        pred = race_model(theta, sigma, B_hf, delta, t_opt, lapse)
        predictions.append(np.array(pred))

    return {
        'predictions': np.array(predictions),
        'response_times': np.array(response_times),
        'angles': angles,
    }


def predict_voc_race_comparison(sigma=0.206, B_hf=0.464, delta=0.467, lapse=0.150,
                                 cost_strict=0.2, cost_lenient=0.05, T_max=15.0):
    """
    Compare VOC predictions under different costs (time pressure manipulation).

    Higher cost = more time pressure = respond sooner
    """
    angles = np.linspace(0, 1, N_ANGLES)

    rt_strict, rt_lenient = [], []
    pred_strict, pred_lenient = [], []

    for theta in angles:
        # Strict: higher cost, respond sooner
        t_s = optimal_stopping_time_race(
            theta, sigma, B_hf, delta, cost_strict, lapse, T_max
        )
        rt_strict.append(t_s)
        pred_strict.append(np.array(
            race_model(theta, sigma, B_hf, delta, t_s, lapse)
        ))

        # Lenient: lower cost, can wait longer
        t_l = optimal_stopping_time_race(
            theta, sigma, B_hf, delta, cost_lenient, lapse, T_max
        )
        rt_lenient.append(t_l)
        pred_lenient.append(np.array(
            race_model(theta, sigma, B_hf, delta, t_l, lapse)
        ))

    return {
        'strict': np.array(pred_strict),
        'lenient': np.array(pred_lenient),
        'rt_strict': np.array(rt_strict),
        'rt_lenient': np.array(rt_lenient),
        'angles': angles,
    }


# =============================================================================
# Default parameters (fitted to data)
# =============================================================================

# Reduced parameterization: 4 free parameters
#   sigma  - semantic width of Gaussian similarity
#   B_hf   - HF baseline advantage (ΔB, since B_lf=0)
#   delta  - drift rate (semantic accumulation speed)
#   lapse  - response noise (motor errors, incomplete learning)
#
# Fixed (baked into race_model): α=1, B_lf=0

DEFAULT_PARAMS = {
    'sigma': 0.206,
    'B_hf': 0.464,      # 1:2 condition; use 0.800 for 1:4
    'delta': 0.467,
    'lapse': 0.150,
}
