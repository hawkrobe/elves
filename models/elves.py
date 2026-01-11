"""
RSA Speaker Models for Word Choice Under Resource Constraints

Implements three model variants comparing accounts of frequency effects
in referential communication. Verified against original WebPPL implementation.
"""

from memo import memo
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from enum import IntEnum

N_ANGLES = 101
Angle = jnp.arange(N_ANGLES)

class Word(IntEnum):
    W1 = 0  # HF, μ = 0.0
    W2 = 1  # LF, μ = 0.5
    W3 = 2  # HF, μ = 1.0

WORD_LOCS = jnp.array([0.0, 0.5, 1.0])
WORD_COSTS = jnp.array([0.0, 1.0, 0.0])


@jax.jit
def utility_info(w, a, sigma, alpha):
    """U(w|θ) = α · log L0(θ|w)"""
    theta = a / (N_ANGLES - 1)
    log_L0 = jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)
    return jnp.exp(alpha * log_L0)


@jax.jit
def utility_rr(w, a, sigma, alpha, lambda_cost):
    """U(w|θ) = α · [log L0(θ|w) - λ·C(w)]"""
    theta = a / (N_ANGLES - 1)
    log_L0 = jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)
    return jnp.exp(alpha * (log_L0 - lambda_cost * WORD_COSTS[w]))


@jax.jit
def utility_fixed(w, a, sigma, alpha, p_retrieve_lf):
    """P(w|θ) ∝ L0(θ|w)^α · P_retrieve(w)"""
    theta = a / (N_ANGLES - 1)
    log_L0 = jnp.log(norm.pdf(theta, loc=WORD_LOCS[w], scale=sigma) + 1e-10)
    retrieve = jnp.array([1.0, p_retrieve_lf, 1.0])[w]
    return jnp.exp(alpha * log_L0) * retrieve


@memo
def S1_info[a: Angle, w: Word](sigma, alpha):
    """Informativity-only speaker: P(w|θ) ∝ L0(θ|w)^α"""
    speaker: knows(a)
    speaker: chooses(w in Word, wpp=utility_info(w, a, sigma, alpha))
    return Pr[speaker.w == w]


@memo
def S1_rr[a: Angle, w: Word](sigma, alpha, lambda_cost):
    """Resource-rational speaker: P(w|θ) ∝ exp(α·[log L0 - λ·C(w)])"""
    speaker: knows(a)
    speaker: chooses(w in Word, wpp=utility_rr(w, a, sigma, alpha, lambda_cost))
    return Pr[speaker.w == w]


@memo
def S1_fixed[a: Angle, w: Word](sigma, alpha, p_retrieve_lf):
    """Fixed-resource speaker: P(w|θ) ∝ L0(θ|w)^α · P_retrieve(w)"""
    speaker: knows(a)
    speaker: chooses(w in Word, wpp=utility_fixed(w, a, sigma, alpha, p_retrieve_lf))
    return Pr[speaker.w == w]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sigma, alpha = 0.25, 1.0
    lambda_cost, p_retrieve_lf = 0.5, 0.6

    info = S1_info(sigma, alpha)
    rr = S1_rr(sigma, alpha, lambda_cost)
    fixed = S1_fixed(sigma, alpha, p_retrieve_lf)

    angles = jnp.linspace(0, 1, N_ANGLES)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    titles = ['Informativity-only', 'Resource-rational', 'Fixed-resource']
    preds = [info, rr, fixed]

    for ax, title, pred in zip(axes, titles, preds):
        ax.plot(angles, pred[:, 0], label='word1 (HF)')
        ax.plot(angles, pred[:, 1], label='word2 (LF)')
        ax.plot(angles, pred[:, 2], label='word3 (HF)')
        ax.set_title(title)
        ax.set_xlabel('θ')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel('P(word | θ)')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()
