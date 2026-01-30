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
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path

from rsa_race import race_model


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


def get_binned_data(df, n_bins=9):
    """Bin data by condition for model fitting."""
    conditions = {}

    for tp in [5, 10]:
        for fr in ["1:2", "1:4"]:
            subset = df[(df["time_pressure"] == tp) & (df["frequency_ratio"] == fr)]

            binned = subset.groupby(
                pd.cut(subset["dist_to_lf_deg"], bins=n_bins),
                observed=True
            ).agg({
                "is_lf": ["mean", "count", "sem"],
                "theta": "mean",
                "dist_to_lf_deg": "mean",
            })
            binned.columns = ["p_lf", "n", "se", "theta_mean", "dist_mean"]
            binned = binned.reset_index(drop=True)

            conditions[(tp, fr)] = binned

    return conditions


# =============================================================================
# Model Prediction
# =============================================================================

def predict_p_lf(theta, sigma, B_hf, delta, T, lapse):
    """Predict P(LF) at a given position."""
    probs = race_model(theta, sigma, B_hf, delta, T, lapse)
    return float(probs[1])  # Index 1 = LF


# =============================================================================
# Loss Function
# =============================================================================

def loss_fn(params, conditions):
    """
    Compute weighted MSE loss across all 4 conditions.

    Parameters (9 total):
        - sigma: semantic width (shared)
        - B_hf_12: HF baseline advantage for 1:2 frequency ratio
        - B_hf_14: HF baseline advantage for 1:4 frequency ratio
        - delta: drift rate (shared)
        - lapse: lapse rate (shared)
        - T_12_5: effective processing time for 1:2, 5s deadline
        - T_12_10: effective processing time for 1:2, 10s deadline
        - T_14_5: effective processing time for 1:4, 5s deadline
        - T_14_10: effective processing time for 1:4, 10s deadline
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

    # Parameter mappings
    T_map = {
        (5, "1:2"): T_12_5,
        (10, "1:2"): T_12_10,
        (5, "1:4"): T_14_5,
        (10, "1:4"): T_14_10,
    }
    B_map = {"1:2": B_hf_12, "1:4": B_hf_14}

    total_loss = 0
    total_n = 0

    for (tp, fr), data in conditions.items():
        T = T_map[(tp, fr)]
        B_hf = B_map[fr]

        for _, row in data.iterrows():
            try:
                pred = predict_p_lf(row["theta_mean"], sigma, B_hf, delta, T, lapse)
            except (ValueError, RuntimeError):
                return 1e10

            total_loss += row["n"] * (pred - row["p_lf"]) ** 2
            total_n += row["n"]

    return total_loss / total_n


def nll_trial_level(params, df):
    """
    Compute negative log-likelihood on TRIAL-LEVEL binary outcomes.
    
    Parameters: same as loss_fn
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

    # Parameter mappings
    T_map = {
        (5, "1:2"): T_12_5,
        (10, "1:2"): T_12_10,
        (5, "1:4"): T_14_5,
        (10, "1:4"): T_14_10,
    }
    B_map = {"1:2": B_hf_12, "1:4": B_hf_14}

    EPS = 1e-10  # for log(0)
    total_nll = 0.0

    for _, row in df.iterrows():
        tp = row["time_pressure"]
        fr = row["frequency_ratio"]
        theta = row["theta"]
        y = row["is_lf"]  # Binary: 1 if chose LF, 0 otherwise

        T = T_map[(tp, fr)]
        B_hf = B_map[fr]

        try:
            p_lf = predict_p_lf(theta, sigma, B_hf, delta, T, lapse)
            p_lf = np.clip(p_lf, EPS, 1 - EPS)  # no log(0)
        except (ValueError, RuntimeError):
            return 1e10

        # Binary cross-entropy for this trial
        total_nll -= y * np.log(p_lf) + (1 - y) * np.log(1 - p_lf)

    return total_nll


# =============================================================================
# Fitting
# =============================================================================

def fit_model(conditions, verbose=True):
    """
    Fit race model to all 4 conditions simultaneously.

    Returns dict of fitted parameters.
    """
    # Parameter bounds: sigma, B_hf_12, B_hf_14, delta, lapse, T_12_5, T_12_10, T_14_5, T_14_10
    bounds = [
        (0.1, 2.0),    # sigma
        (-2.0, 2.0),   # B_hf_12
        (-2.0, 2.0),   # B_hf_14
        (0.1, 10.0),   # delta (capped at 10 so T values stay interpretable as ~RTs)
        (0.0, 0.3),    # lapse
        (0.1, 30.0),   # T_12_5
        (0.1, 30.0),   # T_12_10
        (0.1, 30.0),   # T_14_5
        (0.1, 30.0),   # T_14_10
    ]

    if verbose:
        print("Fitting race model to all 4 conditions...")
        print("  Shared params: σ, δ, λ")
        print("  By frequency ratio: B_HF(1:2), B_HF(1:4)")
        print("  By condition: T_12_5, T_12_10, T_14_5, T_14_10")
        print()

    result = differential_evolution(
        loss_fn,
        bounds,
        args=(conditions,),
        maxiter=300,
        seed=42,
        disp=verbose,
        polish=True,
        workers=1,
        tol=1e-7,
    )

    param_names = ["sigma", "B_hf_12", "B_hf_14", "delta", "lapse",
                   "T_12_5", "T_12_10", "T_14_5", "T_14_10"]
    params = {name: val for name, val in zip(param_names, result.x)}
    params["loss"] = result.fun

    return params


def fit_model_trial_level(df, verbose=True):
    """
    Fit race model using trial-level log-likelihood.
    """
    bounds = [
        (0.1, 2.0),    # sigma
        (-2.0, 2.0),   # B_hf_12
        (-2.0, 2.0),   # B_hf_14
        (0.1, 10.0),   # delta
        (0.0, 0.3),    # lapse
        (0.1, 30.0),   # T_12_5
        (0.1, 30.0),   # T_12_10
        (0.1, 30.0),   # T_14_5
        (0.1, 30.0),   # T_14_10
    ]

    if verbose:
        print("Fitting race model using trial-level log-likelihood...")
        print("  Loss: Negative log-likelihood (binary cross-entropy)")
        print("  Shared params: σ, δ, λ")
        print("  By frequency ratio: B_HF(1:2), B_HF(1:4)")
        print("  By condition: T_12_5, T_12_10, T_14_5, T_14_10")
        print(f"  N trials: {len(df):,}")
        print()

    result = differential_evolution(
        nll_trial_level,
        bounds,
        args=(df,),
        maxiter=300,
        seed=42,
        disp=verbose,
        polish=True,
        workers=1,
        tol=1e-7,
    )

    param_names = ["sigma", "B_hf_12", "B_hf_14", "delta", "lapse",
                   "T_12_5", "T_12_10", "T_14_5", "T_14_10"]
    params = {name: val for name, val in zip(param_names, result.x)}
    params["nll"] = result.fun
    params["ll"] = -result.fun

    return params


# =============================================================================
# Fit Statistics
# =============================================================================

def compute_fit_stats(params, conditions):
    """Compute R² for each condition and overall."""
    T_map = {
        (5, "1:2"): params["T_12_5"],
        (10, "1:2"): params["T_12_10"],
        (5, "1:4"): params["T_14_5"],
        (10, "1:4"): params["T_14_10"],
    }
    B_map = {"1:2": params["B_hf_12"], "1:4": params["B_hf_14"]}

    results = {}
    all_true, all_pred = [], []

    for (tp, fr), data in conditions.items():
        T = T_map[(tp, fr)]
        B_hf = B_map[fr]

        y_true = data["p_lf"].values
        y_pred = np.array([
            predict_p_lf(th, params["sigma"], B_hf, params["delta"], T, params["lapse"])
            for th in data["theta_mean"].values
        ])

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results[(tp, fr)] = {"r2": r2, "y_true": y_true, "y_pred": y_pred}
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    # Overall R²
    all_true, all_pred = np.array(all_true), np.array(all_pred)
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2)
    results["overall"] = {"r2": 1 - ss_res / ss_tot}

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_fit(params, conditions, save_path="race_model_fit.png"):
    """Create 2-panel visualization of model fit."""
    T_map = {
        (5, "1:2"): params["T_12_5"],
        (10, "1:2"): params["T_12_10"],
        (5, "1:4"): params["T_14_5"],
        (10, "1:4"): params["T_14_10"],
    }
    B_map = {"1:2": params["B_hf_12"], "1:4": params["B_hf_14"]}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    colors = {5: "#E94F37", 10: "#2E86AB"}
    labels = {5: "Strict (5s)", 10: "Lenient (10s)"}

    for idx, fr in enumerate(["1:2", "1:4"]):
        ax = axes[idx]
        B_hf = B_map[fr]

        for tp in [5, 10]:
            data = conditions[(tp, fr)]
            T = T_map[(tp, fr)]

            # Plot data points
            ax.errorbar(
                data["dist_mean"], data["p_lf"], yerr=data["se"],
                fmt="o", color=colors[tp], markersize=7, capsize=2,
                label=f"{labels[tp]} (data)", linewidth=0, alpha=0.8
            )

            # Plot model predictions
            dist_smooth = np.linspace(0.5, 44.5, 100)
            theta_smooth = 0.5 - dist_smooth / 90
            pred = [
                predict_p_lf(th, params["sigma"], B_hf, params["delta"], T, params["lapse"])
                for th in theta_smooth
            ]
            ax.plot(dist_smooth, pred, color=colors[tp], linewidth=2.5,
                    label=f"{labels[tp]} (model)")

        ax.axvline(22.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Distance from LF (deg)", fontsize=11)
        ax.set_ylabel("P(LF)", fontsize=11)
        ax.set_title(f"Frequency Ratio {fr}", fontsize=12)
        ax.set_xlim(0, 45)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")

    return fig


# =============================================================================
# Model Comparison
# =============================================================================

def loss_fn_shared_B(params, conditions):
    """Model with shared B_HF across frequency ratios (8 params)."""
    sigma, B_hf, delta, lapse, T_12_5, T_12_10, T_14_5, T_14_10 = params

    if sigma <= 0.05 or sigma > 2.0:
        return 1e10
    if delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in [T_12_5, T_12_10, T_14_5, T_14_10]):
        return 1e10

    T_map = {
        (5, "1:2"): T_12_5, (10, "1:2"): T_12_10,
        (5, "1:4"): T_14_5, (10, "1:4"): T_14_10,
    }

    total_loss, total_n = 0, 0
    for (tp, fr), data in conditions.items():
        T = T_map[(tp, fr)]
        for _, row in data.iterrows():
            try:
                pred = predict_p_lf(row["theta_mean"], sigma, B_hf, delta, T, lapse)
            except (ValueError, RuntimeError):
                return 1e10
            total_loss += row["n"] * (pred - row["p_lf"]) ** 2
            total_n += row["n"]
    return total_loss / total_n


def loss_fn_shared_T(params, conditions):
    """Model with shared T across frequency ratios (7 params)."""
    sigma, B_hf_12, B_hf_14, delta, lapse, T_5, T_10 = params

    if sigma <= 0.05 or sigma > 2.0:
        return 1e10
    if delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10

    T_map = {(5, "1:2"): T_5, (10, "1:2"): T_10, (5, "1:4"): T_5, (10, "1:4"): T_10}
    B_map = {"1:2": B_hf_12, "1:4": B_hf_14}

    total_loss, total_n = 0, 0
    for (tp, fr), data in conditions.items():
        T, B_hf = T_map[(tp, fr)], B_map[fr]
        for _, row in data.iterrows():
            try:
                pred = predict_p_lf(row["theta_mean"], sigma, B_hf, delta, T, lapse)
            except (ValueError, RuntimeError):
                return 1e10
            total_loss += row["n"] * (pred - row["p_lf"]) ** 2
            total_n += row["n"]
    return total_loss / total_n


def loss_fn_minimal(params, conditions):
    """Minimal model: shared B_HF and shared T (6 params)."""
    sigma, B_hf, delta, lapse, T_5, T_10 = params

    if sigma <= 0.05 or sigma > 2.0:
        return 1e10
    if delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10

    T_map = {(5, "1:2"): T_5, (10, "1:2"): T_10, (5, "1:4"): T_5, (10, "1:4"): T_10}

    total_loss, total_n = 0, 0
    for (tp, fr), data in conditions.items():
        T = T_map[(tp, fr)]
        for _, row in data.iterrows():
            try:
                pred = predict_p_lf(row["theta_mean"], sigma, B_hf, delta, T, lapse)
            except (ValueError, RuntimeError):
                return 1e10
            total_loss += row["n"] * (pred - row["p_lf"]) ** 2
            total_n += row["n"]
    return total_loss / total_n


def loss_fn_interaction(params, conditions):
    """
    Interaction model: T parameterized with interaction term (8 params).

    T(tp, fr) structure:
      - T_5:  base processing time at 5s deadline (for 1:2)
      - T_10: base processing time at 10s deadline (for both)
      - γ:    offset for 1:4 at 5s (captures early saturation)

    So: T(5, 1:2) = T_5
        T(5, 1:4) = T_5 + γ
        T(10, 1:2) = T_10
        T(10, 1:4) = T_10

    This captures the freq × time pressure interaction with one parameter.
    """
    sigma, B_hf_12, B_hf_14, delta, lapse, T_5, T_10, gamma = params

    if sigma <= 0.05 or sigma > 2.0:
        return 1e10
    if delta <= 0 or delta > 50:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10
    if T_5 <= 0 or T_10 <= 0:
        return 1e10

    # T structure with interaction
    T_map = {
        (5, "1:2"): T_5,
        (5, "1:4"): T_5 + gamma,
        (10, "1:2"): T_10,
        (10, "1:4"): T_10,
    }
    B_map = {"1:2": B_hf_12, "1:4": B_hf_14}

    total_loss, total_n = 0, 0
    for (tp, fr), data in conditions.items():
        T, B_hf = T_map[(tp, fr)], B_map[fr]
        for _, row in data.iterrows():
            try:
                pred = predict_p_lf(row["theta_mean"], sigma, B_hf, delta, T, lapse)
            except (ValueError, RuntimeError):
                return 1e10
            total_loss += row["n"] * (pred - row["p_lf"]) ** 2
            total_n += row["n"]
    return total_loss / total_n


def fit_model_comparison(conditions):
    """Fit all model variants and compare using AIC/BIC."""
    # Count total observations
    n_obs = sum(data["n"].sum() for data in conditions.values())

    models = {}

    # Model 1: Full (9 params) - already have this
    bounds_full = [
        (0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.0, 0.3),
        (0.1, 30.0), (0.1, 30.0), (0.1, 30.0), (0.1, 30.0),
    ]
    result = differential_evolution(loss_fn, bounds_full, args=(conditions,),
                                    maxiter=200, seed=42, disp=False, polish=True)
    models["Full (B×FR, T×FR×TP)"] = {"k": 9, "mse": result.fun, "params": result.x}

    # Model 2: Shared B_HF (8 params)
    bounds_shared_B = [
        (0.1, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.0, 0.3),
        (0.1, 30.0), (0.1, 30.0), (0.1, 30.0), (0.1, 30.0),
    ]
    result = differential_evolution(loss_fn_shared_B, bounds_shared_B, args=(conditions,),
                                    maxiter=200, seed=42, disp=False, polish=True)
    models["Shared B_HF (T×FR×TP)"] = {"k": 8, "mse": result.fun, "params": result.x}

    # Model 3: Shared T across FR (7 params)
    bounds_shared_T = [
        (0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.0, 0.3),
        (0.1, 30.0), (0.1, 30.0),
    ]
    result = differential_evolution(loss_fn_shared_T, bounds_shared_T, args=(conditions,),
                                    maxiter=200, seed=42, disp=False, polish=True)
    models["Shared T (B×FR, T×TP)"] = {"k": 7, "mse": result.fun, "params": result.x}

    # Model 4: Minimal (6 params)
    bounds_minimal = [
        (0.1, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.0, 0.3),
        (0.1, 30.0), (0.1, 30.0),
    ]
    result = differential_evolution(loss_fn_minimal, bounds_minimal, args=(conditions,),
                                    maxiter=200, seed=42, disp=False, polish=True)
    models["Minimal (shared B, T×TP)"] = {"k": 6, "mse": result.fun, "params": result.x}

    # Model 5: Interaction model (8 params) - reduced from full
    bounds_interaction = [
        (0.1, 2.0), (-2.0, 2.0), (-2.0, 2.0), (0.1, 10.0), (0.0, 0.3),
        (0.1, 30.0), (0.1, 30.0), (-10.0, 10.0),  # gamma can be negative
    ]
    result = differential_evolution(loss_fn_interaction, bounds_interaction, args=(conditions,),
                                    maxiter=200, seed=42, disp=False, polish=True)
    models["Interaction (B×FR, T+γ)"] = {"k": 8, "mse": result.fun, "params": result.x}

    # Compute AIC and BIC for each model
    for name, m in models.items():
        # Log-likelihood approximation from MSE (assuming Gaussian errors)
        # LL ≈ -n/2 * log(2π) - n/2 * log(MSE) - n/2
        # For comparison, we just need: -n * log(MSE)
        ll_approx = -n_obs * np.log(m["mse"])
        m["aic"] = 2 * m["k"] - 2 * ll_approx
        m["bic"] = m["k"] * np.log(n_obs) - 2 * ll_approx

    return models, n_obs


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RSA RACE MODEL FITTING")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = load_data()
    print(f"  N = {len(df):,} trials from {df['subject_id'].nunique()} participants")
    print()

    # Bin data
    conditions = get_binned_data(df)
    for (tp, fr), data in conditions.items():
        n = data["n"].sum()
        print(f"  Condition T={tp}s, FR={fr}: {n:,} trials")
    print()

    # Fit model
    print("-" * 70)
    params = fit_model_trial_level(df, verbose=True)
    print()

    # Compute fit statistics
    #fit_stats = compute_fit_stats(params, conditions)

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
    print("By condition (T = effective processing time):")
    print(f"  1:2, 5s:  T = {params['T_12_5']:.2f}")
    print(f"  1:2, 10s: T = {params['T_12_10']:.2f}")
    print(f"  1:4, 5s:  T = {params['T_14_5']:.2f}")
    print(f"  1:4, 10s: T = {params['T_14_10']:.2f}")
    print()

    print("Time pressure effect (ΔT = T_10s - T_5s):")
    delta_T_12 = params['T_12_10'] - params['T_12_5']
    delta_T_14 = params['T_14_10'] - params['T_14_5']
    print(f"  1:2: ΔT = {delta_T_12:.2f}")
    print(f"  1:4: ΔT = {delta_T_14:.2f}")
    print(f"  Ratio: {delta_T_12 / delta_T_14:.2f}x")
    print()

    print("=" * 70)
    print("FIT STATISTICS")
    print("=" * 70)
    print()
    #for (tp, fr) in [(5, "1:2"), (5, "1:4"), (10, "1:2"), (10, "1:4")]:
    #    r2 = fit_stats[(tp, fr)]["r2"]
    #    print(f"  T={tp}s, FR={fr}: R² = {r2:.4f}")
    print()
    #print(f"  Overall: R² = {fit_stats['overall']['r2']:.4f}")
    print()

    # Plot
    print("-" * 70)
    plot_fit(params, conditions)

    # Save parameters
    np.savez(
        "fitted_params.npz",
        sigma=params["sigma"],
        B_hf_12=params["B_hf_12"],
        B_hf_14=params["B_hf_14"],
        delta=params["delta"],
        lapse=params["lapse"],
        T_12_5=params["T_12_5"],
        T_12_10=params["T_12_10"],
        T_14_5=params["T_14_5"],
        T_14_10=params["T_14_10"],
    )
    print("Saved fitted_params.npz")
    print()

    # Model comparison
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print()
    print("Fitting 4 model variants...")
    model_variants, n_obs = fit_model_comparison(conditions)
    print()

    # Sort by AIC
    sorted_models = sorted(model_variants.items(), key=lambda x: x[1]["aic"])

    print(f"{'Model':<30} {'k':>3} {'MSE':>10} {'AIC':>12} {'BIC':>12} {'ΔAIC':>8}")
    print("-" * 77)
    best_aic = sorted_models[0][1]["aic"]
    for name, m in sorted_models:
        delta_aic = m["aic"] - best_aic
        print(f"{name:<30} {m['k']:>3} {m['mse']:>10.6f} {m['aic']:>12.1f} {m['bic']:>12.1f} {delta_aic:>8.1f}")
    print()
    print(f"N observations: {n_obs:,}")
    print()

    return params, fit_stats


if __name__ == "__main__":
    params, fit_stats = main()
