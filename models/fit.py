"""
RSA Race Model Fitting

Fits the race model to Experiment 1 data (2x2 design: time pressure × frequency ratio).

Model: A(w, t) = B(w) + δ · M(θ, w) · t

Parameters:
  - Shared across conditions: σ (semantic width), δ (drift rate), λ (lapse)
  - Varies by frequency ratio: B_HF (baseline advantage for HF words)
  - Varies by condition: T (effective processing time)

Usage:
    python fit.py

Output:
    - Fitted parameters and fit statistics printed to console
    - race_model_fit.png: Model fit visualization
    - fitted_params.npz: Saved parameters
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from pathlib import Path

import jax
import jax.numpy as jnp
from rsa_race import race_model

# Vectorized race model for batch computation
_race_model_vec = jax.vmap(race_model, in_axes=(0, None, 0, None, 0, None))


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and preprocess Experiment 1 data."""
    data_dir = Path(__file__).parent.parent / "data" / "full-production" / "processed_data"

    production = pd.read_csv(data_dir / "production.csv")
    participants = pd.read_csv(data_dir / "participants.csv")

    df = production.merge(
        participants[["subject_id", "time_pressure", "frequency_ratio"]],
        on="subject_id"
    )

    # Compute distance to nearest LF trained angle
    df["dist_to_lf_deg"] = np.where(
        df["nearestFreq"] == "LF",
        np.abs(df["distance"]),
        np.abs(df["targetAngle"] - df["nextNearestAngle"])
    )
    df["dist_to_lf_deg"] = np.where(
        df["dist_to_lf_deg"] > 180,
        360 - df["dist_to_lf_deg"],
        df["dist_to_lf_deg"]
    )

    # Convert to model's theta space (0.5 = LF prototype, 0/1 = HF prototypes)
    df["theta"] = 0.5 - df["dist_to_lf_deg"] / 90

    # Filter to critical trials
    df = df[df["between_freqs"]]              # Between-frequency trials only
    df = df[df["timed_out"] == 0]              # Exclude timeouts
    df = df[(df["rt"] > 200) &                 # Exclude very fast responses
            (df["rt"] < df["time_pressure"] * 1000)]  # Exclude responses after deadline

    df["is_lf"] = (df["responseFreq"] == "LF").astype(int)

    return df


# =============================================================================
# Loss Functions
# =============================================================================

def neg_log_likelihood(params, df):
    """
    Compute negative log-likelihood on trial-level data (vectorized).

    Parameters (9 total):
        - sigma, B_hf_12, B_hf_14, delta, lapse
        - T_12_5, T_12_10, T_14_5, T_14_10
    """
    sigma, B_hf_12, B_hf_14, delta, lapse, T_12_5, T_12_10, T_14_5, T_14_10 = params

    # Parameter bounds
    if sigma <= 0.05 or sigma > 2.0:
        return 1e10
    if delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in [T_12_5, T_12_10, T_14_5, T_14_10]):
        return 1e10

    # Vectorized condition mapping
    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values

    T = np.where(
        (tp == 5) & fr_is_12, T_12_5,
        np.where((tp == 10) & fr_is_12, T_12_10,
            np.where(tp == 5, T_14_5, T_14_10)))
    B_hf = np.where(fr_is_12, B_hf_12, B_hf_14)

    # Vectorized race model computation
    theta = df["theta"].values
    try:
        probs = _race_model_vec(
            jnp.array(theta), sigma,
            jnp.array(B_hf), delta,
            jnp.array(T), lapse
        )
        p_lf = np.array(probs[:, 1])  # LF is index 1
    except:
        return 1e10

    # Clip and compute NLL
    eps = 1e-10
    p_lf = np.clip(p_lf, eps, 1 - eps)
    y = df["is_lf"].values
    return -np.sum(y * np.log(p_lf) + (1 - y) * np.log(1 - p_lf))


# =============================================================================
# Fitting
# =============================================================================

def fit_model(df):
    """Fit Interaction model using trial-level negative log-likelihood."""
    x0 = [0.2, 0.5, 0.9, 0.5, 0.15, 4.0, 6.0, 1.5]  # sigma, B_12, B_14, delta, lapse, T_5, T_10, gamma
    bounds = [
        (0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0),
        (0.0, 0.3), (0.1, 30.0), (0.1, 30.0), (-10.0, 10.0),
    ]

    print(f"Fitting Interaction model ({len(df):,} trials)...")
    result = minimize(nll_interaction, x0, args=(df,), method='L-BFGS-B', bounds=bounds)

    param_names = ["sigma", "B_hf_12", "B_hf_14", "delta", "lapse",
                   "T_5", "T_10", "gamma"]
    params = {name: val for name, val in zip(param_names, result.x)}

    params["T_12_5"] = params["T_5"]
    params["T_12_10"] = params["T_10"]
    params["T_14_5"] = params["T_5"] + params["gamma"]
    params["T_14_10"] = params["T_10"]

    params["nll"] = result.fun
    params["n_trials"] = len(df)

    return params


# =============================================================================
# Model Comparison
# =============================================================================

def _compute_nll_vec(theta, T, B_hf, y, sigma, delta, lapse):
    """Vectorized NLL computation."""
    try:
        probs = _race_model_vec(
            jnp.array(theta), sigma,
            jnp.array(B_hf), delta,
            jnp.array(T), lapse
        )
        p_lf = np.array(probs[:, 1])
    except:
        return 1e10

    eps = 1e-10
    p_lf = np.clip(p_lf, eps, 1 - eps)
    return -np.sum(y * np.log(p_lf) + (1 - y) * np.log(1 - p_lf))


def nll_shared_B(params, df):
    """Model with shared B_HF across frequency ratios (8 params)."""
    sigma, B_hf, delta, lapse, T_12_5, T_12_10, T_14_5, T_14_10 = params

    if sigma <= 0.05 or sigma > 2.0 or delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in [T_12_5, T_12_10, T_14_5, T_14_10]):
        return 1e10

    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where((tp == 5) & fr_is_12, T_12_5,
            np.where((tp == 10) & fr_is_12, T_12_10,
                np.where(tp == 5, T_14_5, T_14_10)))
    B = np.full(len(df), B_hf)

    return _compute_nll_vec(df["theta"].values, T, B, df["is_lf"].values, sigma, delta, lapse)


def nll_shared_T(params, df):
    """Model with shared T across frequency ratios (7 params)."""
    sigma, B_hf_12, B_hf_14, delta, lapse, T_5, T_10 = params

    if sigma <= 0.05 or sigma > 2.0 or delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5 or T_5 <= 0 or T_10 <= 0:
        return 1e10

    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where(tp == 5, T_5, T_10)
    B = np.where(fr_is_12, B_hf_12, B_hf_14)

    return _compute_nll_vec(df["theta"].values, T, B, df["is_lf"].values, sigma, delta, lapse)


def nll_minimal(params, df):
    """Minimal model: shared B_HF and shared T (6 params)."""
    sigma, B_hf, delta, lapse, T_5, T_10 = params

    if sigma <= 0.05 or sigma > 2.0 or delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5 or T_5 <= 0 or T_10 <= 0:
        return 1e10

    tp = df["time_pressure"].values
    T = np.where(tp == 5, T_5, T_10)
    B = np.full(len(df), B_hf)

    return _compute_nll_vec(df["theta"].values, T, B, df["is_lf"].values, sigma, delta, lapse)


def nll_interaction(params, df):
    """Interaction model: T with interaction term (8 params)."""
    sigma, B_hf_12, B_hf_14, delta, lapse, T_5, T_10, gamma = params

    if sigma <= 0.05 or sigma > 2.0 or delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5 or T_5 <= 0 or T_10 <= 0:
        return 1e10

    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where(tp == 5, T_5, T_10) + np.where((tp == 5) & ~fr_is_12, gamma, 0)
    B = np.where(fr_is_12, B_hf_12, B_hf_14)

    return _compute_nll_vec(df["theta"].values, T, B, df["is_lf"].values, sigma, delta, lapse)


def fit_model_comparison(df, full_params=None):
    """Fit all model variants and compare using AIC/BIC."""
    n_obs = len(df)
    models = {}

    if full_params is None:
        x0_base = [0.2, 0.5, 0.9, 0.4, 0.15, 6.0, 9.0, 7.0, 8.0]
    else:
        x0_base = [
            full_params["sigma"], full_params["B_hf_12"], full_params["B_hf_14"],
            full_params["delta"], full_params["lapse"],
            full_params["T_12_5"], full_params["T_12_10"],
            full_params["T_14_5"], full_params["T_14_10"]
        ]

    def fit_variant(name, nll_fn, x0, bounds, k):
        print(f"  {name}...", end=" ", flush=True)
        result = minimize(nll_fn, x0, args=(df,), method='L-BFGS-B', bounds=bounds)
        print(f"NLL={result.fun:.1f}")
        return {"k": k, "nll": result.fun}

    bounds_9 = [(0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.01, 0.3),
                (0.1, 30.0), (0.1, 30.0), (0.1, 30.0), (0.1, 30.0)]
    bounds_8_T = [(0.1, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.01, 0.3),
                  (0.1, 30.0), (0.1, 30.0), (0.1, 30.0), (0.1, 30.0)]
    bounds_8_g = [(0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.01, 0.3),
                  (0.1, 30.0), (0.1, 30.0), (-10.0, 10.0)]
    bounds_7 = [(0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.01, 0.3),
                (0.1, 30.0), (0.1, 30.0)]
    bounds_6 = [(0.1, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.01, 0.3),
                (0.1, 30.0), (0.1, 30.0)]

    models["Full (B×FR, T×FR×TP)"] = fit_variant(
        "Full (9 params)", neg_log_likelihood, x0_base, bounds_9, 9)
    models["Shared B_HF (T×FR×TP)"] = fit_variant(
        "Shared B_HF (8 params)", nll_shared_B,
        [x0_base[0], (x0_base[1]+x0_base[2])/2, x0_base[3], x0_base[4]] + x0_base[5:],
        bounds_8_T, 8)
    models["Shared T (B×FR, T×TP)"] = fit_variant(
        "Shared T (7 params)", nll_shared_T,
        x0_base[:5] + [(x0_base[5]+x0_base[7])/2, (x0_base[6]+x0_base[8])/2],
        bounds_7, 7)
    models["Minimal (shared B, T×TP)"] = fit_variant(
        "Minimal (6 params)", nll_minimal,
        [x0_base[0], (x0_base[1]+x0_base[2])/2, x0_base[3], x0_base[4],
         (x0_base[5]+x0_base[7])/2, (x0_base[6]+x0_base[8])/2],
        bounds_6, 6)
    models["Interaction (B×FR, T+γ)"] = fit_variant(
        "Interaction (8 params)", nll_interaction,
        x0_base[:5] + [x0_base[5], x0_base[6], x0_base[7] - x0_base[5]],
        bounds_8_g, 8)

    for m in models.values():
        m["aic"] = 2 * m["k"] + 2 * m["nll"]
        m["bic"] = m["k"] * np.log(n_obs) + 2 * m["nll"]

    return models, n_obs


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RSA RACE MODEL FITTING (Trial-level NLL)")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = load_data()
    print(f"  N = {len(df):,} trials from {df['subject_id'].nunique()} participants")
    print()

    # Show condition breakdown
    for tp in [5, 10]:
        for fr in ["1:2", "1:4"]:
            n = len(df[(df["time_pressure"] == tp) & (df["frequency_ratio"] == fr)])
            print(f"  Condition T={tp}s, FR={fr}: {n:,} trials")
    print()

    print("-" * 70)
    params = fit_model(df)
    print()

    # Print results
    print("=" * 70)
    print("FITTED PARAMETERS")
    print("=" * 70)
    print()
    print("Shared across conditions:")
    print(f"  σ (semantic width) = {params['sigma']:.3f}")
    print(f"  δ (drift rate)     = {params['delta']:.3f}")
    print(f"  λ (lapse rate)     = {params['lapse']:.3f}")
    print()
    print("By frequency ratio:")
    print(f"  B_HF (1:2) = {params['B_hf_12']:.3f}")
    print(f"  B_HF (1:4) = {params['B_hf_14']:.3f}")
    print(f"  Δ B_HF     = {params['B_hf_14'] - params['B_hf_12']:.3f} (1:4 - 1:2)")
    print()
    print("Interaction model parameters:")
    print(f"  T_5 (base at 5s)  = {params['T_5']:.2f}")
    print(f"  T_10 (base at 10s) = {params['T_10']:.2f}")
    print(f"  γ (1:4 offset at 5s) = {params['gamma']:.2f}")
    print()
    print("Derived T values:")
    print(f"  1:2, 5s:  T = T_5 = {params['T_12_5']:.2f}")
    print(f"  1:2, 10s: T = T_10 = {params['T_12_10']:.2f}")
    print(f"  1:4, 5s:  T = T_5 + γ = {params['T_14_5']:.2f}")
    print(f"  1:4, 10s: T = T_10 = {params['T_14_10']:.2f}")
    print()

    print("Time pressure effect (ΔT = T_10 - T_5):")
    delta_T_12 = params['T_10'] - params['T_5']
    delta_T_14 = params['T_10'] - (params['T_5'] + params['gamma'])
    print(f"  1:2: ΔT = {delta_T_12:.2f}")
    print(f"  1:4: ΔT = {delta_T_14:.2f}")
    print(f"  Ratio: {delta_T_12 / delta_T_14:.2f}x")
    print()

    # Plot
    print("-" * 70)
    plot_fit(params, conditions)

    # Save parameters
    np.savez("fitted_params.npz", **{k: v for k, v in params.items()
                                      if k not in ["nll", "n_trials"]})
    print("Saved fitted_params.npz")
    print()

    # Model comparison
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print()
    print("Fitting model variants...")
    model_variants, n_obs = fit_model_comparison(df, full_params=params)
    print()

    # Sort by AIC
    sorted_models = sorted(model_variants.items(), key=lambda x: x[1]["aic"])

    print(f"{'Model':<30} {'k':>3} {'NLL':>12} {'AIC':>12} {'BIC':>12} {'ΔAIC':>8}")
    print("-" * 80)
    best_aic = sorted_models[0][1]["aic"]
    for name, m in sorted_models:
        delta_aic = m["aic"] - best_aic
        print(f"{name:<30} {m['k']:>3} {m['nll']:>12.1f} {m['aic']:>12.1f} {m['bic']:>12.1f} {delta_aic:>8.1f}")
    print()
    print(f"N trials: {n_obs:,}")
    print()

    # Likelihood ratio tests for nested models
    print("Likelihood Ratio Tests (nested models):")
    print("-" * 60)

    # Full vs Interaction: tests whether T_10 can be shared across FR
    nll_full = model_variants["Full (B×FR, T×FR×TP)"]["nll"]
    nll_interaction = model_variants["Interaction (B×FR, T+γ)"]["nll"]
    lr_stat = 2 * (nll_interaction - nll_full)
    df = 9 - 8  # 1 parameter difference
    p_val = 1 - chi2.cdf(lr_stat, df)
    print(f"  Full vs Interaction: LR = {lr_stat:.2f}, df = {df}, p = {p_val:.3f}")

    # Full vs Shared T: tests whether T can be shared across FR
    nll_shared_t = model_variants["Shared T (B×FR, T×TP)"]["nll"]
    lr_stat = 2 * (nll_shared_t - nll_full)
    df = 9 - 7  # 2 parameter difference
    p_val = 1 - chi2.cdf(lr_stat, df)
    print(f"  Full vs Shared T:    LR = {lr_stat:.2f}, df = {df}, p = {p_val:.3f}")

    # Full vs Shared B: tests whether B_HF varies by FR
    nll_shared_b = model_variants["Shared B_HF (T×FR×TP)"]["nll"]
    lr_stat = 2 * (nll_shared_b - nll_full)
    df = 9 - 8  # 1 parameter difference
    p_val = 1 - chi2.cdf(lr_stat, df)
    print(f"  Full vs Shared B:    LR = {lr_stat:.2f}, df = {df}, p = {p_val:.3f}")

    # Interaction vs Minimal: tests whether B_HF varies by FR (within interaction structure)
    nll_minimal = model_variants["Minimal (shared B, T×TP)"]["nll"]
    lr_stat = 2 * (nll_minimal - nll_interaction)
    df = 8 - 6  # 2 parameter difference
    p_val = 1 - chi2.cdf(lr_stat, df)
    print(f"  Interaction vs Min:  LR = {lr_stat:.2f}, df = {df}, p = {p_val:.3f}")
    print()

    return params


if __name__ == "__main__":
    params = main()
