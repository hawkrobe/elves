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
    """Full model (8 params, delta fixed to 1)."""
    sigma, B_hf_12, B_hf_14, lapse, T_12_5, T_12_10, T_14_5, T_14_10 = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in [T_12_5, T_12_10, T_14_5, T_14_10]):
        return 1e10
    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where((tp == 5) & fr_is_12, T_12_5,
            np.where((tp == 10) & fr_is_12, T_12_10,
                np.where(tp == 5, T_14_5, T_14_10)))
    B_hf = np.where(fr_is_12, B_hf_12, B_hf_14)
    return _compute_nll_vec(df["theta"].values, T, B_hf, df["is_lf"].values, sigma, lapse)


# =============================================================================
# Fitting
# =============================================================================

def fit_model(df):
    """Fit Interaction model (delta fixed to 1)."""
    x0 = [0.2, 0.5, 0.9, 0.15, 2.0, 3.5, 1.0]

    print(f"Fitting Interaction model ({len(df):,} trials)...")
    result = minimize(nll_interaction, x0, args=(df,), method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})

    param_names = ["sigma", "B_hf_12", "B_hf_14", "lapse", "T_5", "T_10", "gamma"]
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

def _compute_nll_vec(theta, T, B_hf, y, sigma, lapse):
    """Vectorized NLL computation (delta fixed to 1)."""
    try:
        probs = _race_model_vec(
            jnp.array(theta), sigma, jnp.array(B_hf), 1.0, jnp.array(T), lapse)
        p_lf = np.array(probs[:, 1])
    except:
        return 1e10

    eps = 1e-10
    p_lf = np.clip(p_lf, eps, 1 - eps)
    return -np.sum(y * np.log(p_lf) + (1 - y) * np.log(1 - p_lf))


def nll_shared_B(params, df):
    """Shared B_HF across frequency ratios (7 params)."""
    sigma, B_hf, lapse, T_12_5, T_12_10, T_14_5, T_14_10 = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in [T_12_5, T_12_10, T_14_5, T_14_10]):
        return 1e10
    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where((tp == 5) & fr_is_12, T_12_5,
            np.where((tp == 10) & fr_is_12, T_12_10,
                np.where(tp == 5, T_14_5, T_14_10)))
    return _compute_nll_vec(df["theta"].values, T, np.full(len(df), B_hf), df["is_lf"].values, sigma, lapse)


def nll_shared_T(params, df):
    """Shared T across frequency ratios (6 params)."""
    sigma, B_hf_12, B_hf_14, lapse, T_5, T_10 = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10
    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where(tp == 5, T_5, T_10)
    B = np.where(fr_is_12, B_hf_12, B_hf_14)
    return _compute_nll_vec(df["theta"].values, T, B, df["is_lf"].values, sigma, lapse)


def nll_minimal(params, df):
    """Minimal: shared B_HF and shared T (5 params)."""
    sigma, B_hf, lapse, T_5, T_10 = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10
    tp = df["time_pressure"].values
    T = np.where(tp == 5, T_5, T_10)
    return _compute_nll_vec(df["theta"].values, T, np.full(len(df), B_hf), df["is_lf"].values, sigma, lapse)


def nll_interaction(params, df):
    """Interaction model (7 params, delta fixed to 1)."""
    sigma, B_hf_12, B_hf_14, lapse, T_5, T_10, gamma = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10
    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where(tp == 5, T_5, T_10) + np.where((tp == 5) & ~fr_is_12, gamma, 0)
    B = np.where(fr_is_12, B_hf_12, B_hf_14)
    return _compute_nll_vec(df["theta"].values, T, B, df["is_lf"].values, sigma, lapse)


def nll_shared_B_interaction(params, df):
    """Shared B_HF with T interaction (6 params, delta fixed to 1)."""
    sigma, B_hf, lapse, T_5, T_10, gamma = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10
    tp = df["time_pressure"].values
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where(tp == 5, T_5, T_10) + np.where((tp == 5) & ~fr_is_12, gamma, 0)
    return _compute_nll_vec(df["theta"].values, T, np.full(len(df), B_hf), df["is_lf"].values, sigma, lapse)


def nll_no_TP_effect(params, df):
    """B_HF varies by FR, but T is shared across time pressure (5 params)."""
    sigma, B_hf_12, B_hf_14, lapse, T = params
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if T <= 0:
        return 1e10
    fr_is_12 = (df["frequency_ratio"] == "1:2").values
    B = np.where(fr_is_12, B_hf_12, B_hf_14)
    return _compute_nll_vec(df["theta"].values, np.full(len(df), T), B, df["is_lf"].values, sigma, lapse)


def fit_model_comparison(df, full_params=None):
    """Fit all model variants and compare using AIC/BIC. Delta fixed to 1."""
    n_obs = len(df)
    models = {}

    # Base starting values: sigma, B_12, B_14, lapse, T_12_5, T_12_10, T_14_5, T_14_10
    if full_params is None:
        x0 = [0.2, 0.5, 0.9, 0.15, 2.0, 3.5, 3.0, 3.5]
    else:
        x0 = [full_params["sigma"], full_params["B_hf_12"], full_params["B_hf_14"],
              full_params["lapse"], full_params["T_12_5"], full_params["T_12_10"],
              full_params["T_14_5"], full_params["T_14_10"]]

    def fit_variant(name, nll_fn, x0, bounds, k):
        print(f"  {name}...", end=" ", flush=True)
        result = minimize(nll_fn, x0, args=(df,), method='L-BFGS-B', bounds=bounds)
        print(f"NLL={result.fun:.1f}")
        return {"k": k, "nll": result.fun}

    # Bounds for each model variant
    b_sig = (0.1, 2.0)
    b_B = (-2.0, 2.0)
    b_lapse = (0.01, 0.3)
    b_T = (0.1, 30.0)
    b_gamma = (-10.0, 10.0)

    models["Full (B×FR, T×FR×TP)"] = fit_variant("Full (8)", neg_log_likelihood,
        x0, [b_sig, b_B, b_B, b_lapse, b_T, b_T, b_T, b_T], 8)
    models["Shared B_HF (T×FR×TP)"] = fit_variant("Shared B (7)", nll_shared_B,
        [x0[0], (x0[1]+x0[2])/2, x0[3], x0[4], x0[5], x0[6], x0[7]],
        [b_sig, b_B, b_lapse, b_T, b_T, b_T, b_T], 7)
    models["Shared T (B×FR, T×TP)"] = fit_variant("Shared T (6)", nll_shared_T,
        [x0[0], x0[1], x0[2], x0[3], (x0[4]+x0[6])/2, (x0[5]+x0[7])/2],
        [b_sig, b_B, b_B, b_lapse, b_T, b_T], 6)
    models["Minimal (shared B, T×TP)"] = fit_variant("Minimal (5)", nll_minimal,
        [x0[0], (x0[1]+x0[2])/2, x0[3], (x0[4]+x0[6])/2, (x0[5]+x0[7])/2],
        [b_sig, b_B, b_lapse, b_T, b_T], 5)
    models["Interaction (B×FR, T+γ)"] = fit_variant("Interaction (7)", nll_interaction,
        [x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6] - x0[4]],
        [b_sig, b_B, b_B, b_lapse, b_T, b_T, b_gamma], 7)
    models["Shared B + Interaction (T+γ)"] = fit_variant("Shared B + Int (6)", nll_shared_B_interaction,
        [x0[0], (x0[1]+x0[2])/2, x0[3], x0[4], x0[5], x0[6] - x0[4]],
        [b_sig, b_B, b_lapse, b_T, b_T, b_gamma], 6)
    models["No TP effect (B×FR, T shared)"] = fit_variant("No TP effect (5)", nll_no_TP_effect,
        [x0[0], x0[1], x0[2], x0[3], (x0[4]+x0[5]+x0[6]+x0[7])/4],
        [b_sig, b_B, b_B, b_lapse, b_T], 5)

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
    print("Shared across conditions (δ fixed to 1):")
    print(f"  σ (semantic width) = {params['sigma']:.3f}")
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
    print("Likelihood Ratio Tests (against Interaction model):")
    print("-" * 60)

    nll_int = model_variants["Interaction (B×FR, T+γ)"]["nll"]
    nll_shared_t = model_variants["Shared T (B×FR, T×TP)"]["nll"]
    nll_shared_b_int = model_variants["Shared B + Interaction (T+γ)"]["nll"]
    nll_no_tp = model_variants["No TP effect (B×FR, T shared)"]["nll"]

    # Test B_HF variation: Interaction vs Shared B + Interaction
    lr = 2 * (nll_shared_b_int - nll_int)
    print(f"  Effect of B_HF × FR:       LR = {lr:.2f}, df = 1, p = {1 - chi2.cdf(lr, 1):.3f}")
    print(f"    (Interaction vs Shared B + Interaction)")

    # Test T main effect of time pressure: Interaction vs No TP effect
    lr = 2 * (nll_no_tp - nll_int)
    print(f"  Effect of T × TP:          LR = {lr:.2f}, df = 2, p = {1 - chi2.cdf(lr, 2):.3f}")
    print(f"    (Interaction vs No TP effect)")

    # Test T interaction: Interaction vs Shared T
    lr = 2 * (nll_shared_t - nll_int)
    print(f"  Effect of T × FR (γ):      LR = {lr:.2f}, df = 1, p = {1 - chi2.cdf(lr, 1):.3f}")
    print(f"    (Interaction vs Shared T)")
    print()

    return params


if __name__ == "__main__":
    params = main()
