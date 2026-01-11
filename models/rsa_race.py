"""
RSA with race model retrieval dynamics.

Activation(w, t) = baseline(w) + drift × semantic_match(θ, w) × t
"""

from memo import memo
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from enum import IntEnum

N_ANGLES = 101
N_TIMES = 11

Angle = jnp.arange(N_ANGLES)
Time = jnp.arange(N_TIMES)

class Word(IntEnum):
    HF1 = 0
    LF = 1
    HF2 = 2

WORD_LOCS = jnp.array([0.0, 0.5, 1.0])
WORD_FREQ = jnp.array([1, 0, 1])  # 1 = HF, 0 = LF


@jax.jit
def semantic_match(theta, w, sigma):
    """Gaussian semantic match between target θ and word w, normalized to [0, 1]."""
    loc = WORD_LOCS[w]
    match = norm.pdf(theta, loc=loc, scale=sigma)
    max_match = norm.pdf(0.0, loc=0.0, scale=sigma)
    return match / max_match


@jax.jit
def activation(theta, w, t, sigma, baseline_hf, baseline_lf, drift_rate):
    """Activation(w, t) = baseline(w) + drift × semantic_match(θ, w) × t"""
    baseline = jnp.where(WORD_FREQ[w] == 1, baseline_hf, baseline_lf)
    sem_match = semantic_match(theta, w, sigma)
    time_scaled = t / (N_TIMES - 1)
    return baseline + drift_rate * sem_match * time_scaled


@jax.jit
def retrieval_utility(w, a, t, sigma, baseline_hf, baseline_lf, drift_rate, temperature):
    """Softmax utility for word selection based on activation."""
    theta = a / (N_ANGLES - 1)
    act = activation(theta, w, t, sigma, baseline_hf, baseline_lf, drift_rate)
    return jnp.exp(act / temperature)


@jax.jit
def informativity(w, a, sigma):
    """Log probability that literal listener recovers θ from word w."""
    theta = a / (N_ANGLES - 1)
    log_L0 = jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)
    return log_L0


@jax.jit
def voc_utility(w, a, t, sigma, baseline_hf, baseline_lf, drift_rate,
                alpha, beta_cost):
    """Utility = α × (informativity + β × activation)."""
    theta = a / (N_ANGLES - 1)
    info = informativity(w, a, sigma)
    act = activation(theta, w, t, sigma, baseline_hf, baseline_lf, drift_rate)
    return jnp.exp(alpha * (info + beta_cost * act))


@memo
def S1_race[a: Angle, t: Time, w: Word](sigma, baseline_hf, baseline_lf, drift_rate, temperature):
    """Race model speaker: P(w | θ, t) from activation dynamics."""
    speaker: knows(a, t)
    speaker: chooses(w in Word, wpp=retrieval_utility(
        w, a, t, sigma, baseline_hf, baseline_lf, drift_rate, temperature
    ))
    return Pr[speaker.w == w]


@memo
def S1_voc[a: Angle, w: Word](sigma, baseline_hf, baseline_lf, drift_rate, alpha, beta_cost):
    """VOC speaker: P(w | θ) marginalized over deliberation time."""
    speaker: knows(a)
    speaker: chooses(t in Time, wpp=1.0)
    speaker: chooses(w in Word, wpp=voc_utility(
        w, a, t, sigma, baseline_hf, baseline_lf, drift_rate,
        alpha, beta_cost
    ))
    return Pr[speaker.w == w]


def predict_time_pressure(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                          drift_rate=2.0, temperature=0.25,
                          time_strict=8, time_lenient=10):
    """Predictions for strict vs lenient time pressure conditions."""
    all_preds = S1_race(
        sigma=sigma, baseline_hf=baseline_hf,
        baseline_lf=baseline_lf, drift_rate=drift_rate, temperature=temperature
    )
    return {
        'strict': all_preds[:, time_strict, :],
        'lenient': all_preds[:, time_lenient, :],
        'angles': jnp.linspace(0, 1, N_ANGLES),
    }


def predict_voc(sigma=0.12, baseline_hf=0.4, baseline_lf=0.25,
                drift_rate=2.0, alpha=1.0, beta_cost=1.0):
    """VOC predictions: P(w | θ) marginalized over time."""
    pred = S1_voc(
        sigma=sigma, baseline_hf=baseline_hf, baseline_lf=baseline_lf,
        drift_rate=drift_rate, alpha=alpha, beta_cost=beta_cost
    )
    return {
        'predictions': pred,
        'angles': jnp.linspace(0, 1, N_ANGLES),
    }
