"""
Resource-Rational RSA with Race Model Retrieval and VOC Analysis

Integrates:
1. RSA informativity (why speakers care about word choice)
2. Race model dynamics (how retrieval cost depends on time)
3. Value of Computation (when to stop deliberating)

The key insight: cost is not a fixed penalty but the TIME required for
semantic information to overcome frequency-based baseline activation.
"""

from memo import memo
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from enum import IntEnum

# Discretization
N_ANGLES = 101
N_TIMES = 11  # deliberation time steps

Angle = jnp.arange(N_ANGLES)
Time = jnp.arange(N_TIMES)

class Word(IntEnum):
    HF1 = 0  # HF word at position 0.0
    LF = 1   # LF word at position 0.5
    HF2 = 2  # HF word at position 1.0

WORD_LOCS = jnp.array([0.0, 0.5, 1.0])
WORD_FREQ = jnp.array([1, 0, 1])  # 1 = HF, 0 = LF


# =============================================================================
# Race Model Components (JAX-compatible)
# =============================================================================

@jax.jit
def semantic_match(theta, w, sigma):
    """Gaussian semantic match between target θ and word w."""
    loc = WORD_LOCS[w]
    match = norm.pdf(theta, loc=loc, scale=sigma)
    max_match = norm.pdf(0.0, loc=0.0, scale=sigma)
    return match / max_match


@jax.jit
def activation(theta, w, t, sigma, baseline_hf, baseline_lf, drift_rate):
    """
    Activation(w, t) = baseline(w) + drift × semantic_match(θ, w) × t

    HF words: high baseline, immediate availability
    LF words: low baseline, rely on semantic accumulation over time
    """
    baseline = jnp.where(WORD_FREQ[w] == 1, baseline_hf, baseline_lf)
    sem_match = semantic_match(theta, w, sigma)
    time_scaled = t / (N_TIMES - 1)  # normalize to [0, 1]
    return baseline + drift_rate * sem_match * time_scaled


@jax.jit
def retrieval_utility(w, a, t, sigma, baseline_hf, baseline_lf, drift_rate, temperature):
    """
    Utility for selecting word w at angle a after time t.
    Uses softmax over activations (Luce choice rule).
    """
    theta = a / (N_ANGLES - 1)
    act = activation(theta, w, t, sigma, baseline_hf, baseline_lf, drift_rate)
    return jnp.exp(act / temperature)


# =============================================================================
# Informativity (RSA Literal Listener)
# =============================================================================

@jax.jit
def informativity(w, a, sigma):
    """Log probability that literal listener recovers θ from word w."""
    theta = a / (N_ANGLES - 1)
    log_L0 = jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)
    return log_L0


# =============================================================================
# VOC: Value of Computation
# =============================================================================

@jax.jit
def voc_utility(w, a, t, sigma, baseline_hf, baseline_lf, drift_rate,
                temperature, alpha, time_cost):
    """
    VOC-based utility combining:
    - Benefit: informativity of chosen word (α × log L0)
    - Cost: time spent deliberating (time_cost × t)

    U(w, t | θ) = α × log L0(θ|w) - time_cost × t

    But word selection depends on retrieval dynamics (race model).
    """
    theta = a / (N_ANGLES - 1)

    # Informativity benefit
    info = informativity(w, a, sigma)

    # Retrieval probability from race model
    act = activation(theta, w, t, sigma, baseline_hf, baseline_lf, drift_rate)

    # Combined: race model determines WHICH word, informativity determines VALUE
    return jnp.exp((alpha * info + act) / temperature)


# =============================================================================
# Memo Models
# =============================================================================

@memo
def S1_race[a: Angle, t: Time, w: Word](sigma, baseline_hf, baseline_lf, drift_rate, temperature):
    """
    Race model speaker with fixed deliberation time.
    P(w | θ, t) determined by activation dynamics.
    """
    speaker: knows(a, t)
    speaker: chooses(w in Word, wpp=retrieval_utility(
        w, a, t, sigma, baseline_hf, baseline_lf, drift_rate, temperature
    ))
    return Pr[speaker.w == w]


@memo
def S1_voc[a: Angle, w: Word](sigma, baseline_hf, baseline_lf, drift_rate, temperature, alpha, time_cost):
    """
    VOC speaker who chooses word, marginalizing over deliberation time.

    The speaker implicitly chooses how long to deliberate based on VOC:
    - At prototype: waiting is valuable (LF much more informative)
    - At boundary: waiting provides little benefit, respond quickly with HF

    Returns P(w | θ) with time marginalized out.
    """
    speaker: knows(a)
    speaker: chooses(t in Time, wpp=1.0)  # uniform prior on time
    speaker: chooses(w in Word, wpp=voc_utility(
        w, a, t, sigma, baseline_hf, baseline_lf, drift_rate,
        temperature, alpha, time_cost
    ))
    return Pr[speaker.w == w]


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_time_pressure(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                          drift_rate=2.0, temperature=0.25,
                          time_strict=8, time_lenient=10):
    """
    Generate predictions for strict vs lenient time pressure conditions.

    Returns dict with predictions for each condition.
    """
    # S1_race returns shape [N_ANGLES, N_TIMES, N_WORDS]
    all_preds = S1_race(
        sigma=sigma, baseline_hf=baseline_hf,
        baseline_lf=baseline_lf, drift_rate=drift_rate, temperature=temperature
    )

    # Extract predictions at specific time indices
    pred_strict = all_preds[:, time_strict, :]   # shape [N_ANGLES, N_WORDS]
    pred_lenient = all_preds[:, time_lenient, :]

    return {
        'strict': pred_strict,
        'lenient': pred_lenient,
        'angles': jnp.linspace(0, 1, N_ANGLES),
    }


def predict_voc(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                drift_rate=2.0, temperature=0.25, alpha=1.0, time_cost=0.1):
    """
    Generate VOC model predictions.

    Returns P(w | θ) marginalized over deliberation time.
    """
    pred = S1_voc(
        sigma=sigma, baseline_hf=baseline_hf, baseline_lf=baseline_lf,
        drift_rate=drift_rate, temperature=temperature,
        alpha=alpha, time_cost=time_cost
    )

    return {
        'predictions': pred,
        'angles': jnp.linspace(0, 1, N_ANGLES),
    }


