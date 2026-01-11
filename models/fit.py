"""
Parameter fitting for the RSA Race Model (reduced parameterization).

Fits 4 free parameters:
  - σ (sigma): semantic width
  - ΔB (delta_B): HF baseline advantage (B_hf - B_lf)
  - δ (delta): drift rate
  - lapse: motor errors / incomplete learning

Fixed parameters:
  - α = 1 (absorbed into other params)
  - B_lf = 0 (reference baseline)
"""

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from scipy import stats
import matplotlib.pyplot as plt
from rsa_race import race_model, Word
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and preprocess experimental data."""
    production = pd.read_csv('../data/full-production/processed_data/production.csv')
    participants = pd.read_csv('../data/full-production/processed_data/participants.csv')

    df = production.merge(
        participants[['subject_id', 'time_pressure', 'frequency_ratio']],
        on='subject_id'
    )

    df['dist_to_lf_deg'] = np.where(
        df['nearestFreq'] == 'LF',
        np.abs(df['distance']),
        np.abs(df['targetAngle'] - df['nextNearestAngle'])
    )
    df['dist_to_lf_deg'] = np.where(
        df['dist_to_lf_deg'] > 180,
        360 - df['dist_to_lf_deg'],
        df['dist_to_lf_deg']
    )

    # Convert to model theta space (0.5 = LF prototype, 0/1 = HF prototypes)
    df['theta'] = 0.5 - df['dist_to_lf_deg'] / 90

    df = df[df['between_freqs'] == True]
    df = df[df['timed_out'] == 0]
    df = df[(df['rt'] > 200) & (df['rt'] < df['time_pressure'] * 1000)]
    df['is_lf'] = (df['responseFreq'] == 'LF').astype(int)

    return df


def get_binned_data(df, n_bins=9):
    """Compute binned summaries for fitting."""
    summaries = {}

    for tp in [5, 10]:
        subset = df[df['time_pressure'] == tp]
        binned = subset.groupby(
            pd.cut(subset['dist_to_lf_deg'], bins=n_bins),
            observed=True
        ).agg({
            'is_lf': ['mean', 'std', 'count'],
            'rt': ['mean', 'std'],
            'dist_to_lf_deg': 'mean',
            'theta': 'mean',
        })
        binned.columns = ['p_lf', 'p_lf_std', 'n', 'rt_mean', 'rt_std', 'dist_mean', 'theta_mean']
        binned['p_lf_se'] = np.sqrt(binned['p_lf'] * (1 - binned['p_lf']) / binned['n'])
        binned['rt_se'] = binned['rt_std'] / np.sqrt(binned['n'])
        summaries[tp] = binned.reset_index(drop=True)

    return summaries


# =============================================================================
# Model Prediction (reduced parameterization)
# =============================================================================

# Fixed parameters
ALPHA = 1.0  # absorbed into other params
B_LF = 0.0   # reference baseline


def predict_p_lf(theta, sigma, delta_B, delta, T, lapse):
    """Predict P(LF) using reduced parameterization."""
    probs = race_model(theta, sigma, delta_B, delta, T, lapse)
    return float(probs[Word.LF])


def predict_all(thetas, sigma, delta_B, delta, T, lapse):
    """Predict P(LF) for array of thetas."""
    return np.array([predict_p_lf(th, sigma, delta_B, delta, T, lapse)
                     for th in thetas])


# =============================================================================
# Loss Function
# =============================================================================

def loss_fn(params, binned_data):
    """Weighted MSE loss for lexical choice data."""
    sigma, delta_B, delta, lapse = params

    # Bounds check
    if sigma <= 0.05 or sigma > 1.0:
        return 1e10
    if delta <= 0:
        return 1e10
    if lapse < 0 or lapse > 0.5:
        return 1e10

    total_loss = 0
    total_weight = 0

    for tp in [5, 10]:
        T = float(tp)
        data = binned_data[tp]

        for _, row in data.iterrows():
            theta = row['theta_mean']
            p_lf_emp = row['p_lf']
            n = row['n']

            try:
                p_lf_pred = predict_p_lf(theta, sigma, delta_B, delta, T, lapse)
            except:
                return 1e10

            total_loss += n * (p_lf_pred - p_lf_emp) ** 2
            total_weight += n

    return total_loss / total_weight


# =============================================================================
# Fitting
# =============================================================================

def fit_model(binned_data, verbose=True):
    """Fit reduced race model to lexical choice data."""

    bounds = [
        (0.1, 1.0),    # sigma
        (-2.0, 2.0),   # delta_B (can be negative)
        (0.1, 10.0),   # delta
        (0.0, 0.3),    # lapse
    ]

    if verbose:
        print("Fitting reduced race model (4 free params)...")
        print("  Fixed: α = 1, B_lf = 0")
        print("  Free: σ, ΔB, δ, lapse")
        print("\nRunning differential evolution...")

    result = differential_evolution(
        loss_fn,
        bounds,
        args=(binned_data,),
        maxiter=200,
        seed=42,
        disp=verbose,
        polish=True,
        workers=1,
        tol=1e-6,
    )

    param_names = ['sigma', 'delta_B', 'delta', 'lapse']
    params = {name: val for name, val in zip(param_names, result.x)}

    if verbose:
        print(f"\nOptimization {'succeeded' if result.success else 'failed'}")
        print(f"Final loss: {result.fun:.6f}")
        print("\nFitted parameters:")
        for name, val in params.items():
            print(f"  {name}: {val:.4f}")

    return params


def compute_fit_statistics(params, binned_data):
    """Compute R², RMSE for each condition."""
    results = {}

    for tp in [5, 10]:
        T = float(tp)
        data = binned_data[tp]

        y_true = data['p_lf'].values
        y_pred = predict_all(data['theta_mean'].values,
                             params['sigma'], params['delta_B'],
                             params['delta'], T, params['lapse'])

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        results[tp] = {'r2': r2, 'rmse': rmse, 'y_true': y_true, 'y_pred': y_pred}

    # Overall
    all_true = np.concatenate([results[5]['y_true'], results[10]['y_true']])
    all_pred = np.concatenate([results[5]['y_pred'], results[10]['y_pred']])
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2)

    results['overall'] = {
        'r2': 1 - ss_res / ss_tot,
        'rmse': np.sqrt(np.mean((all_true - all_pred) ** 2)),
    }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_fit(params, binned_data, save_path='race_model_fit.png'):
    """Create fit visualization."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {5: '#E94F37', 10: '#2E86AB'}
    labels = {5: 'Strict (5s)', 10: 'Lenient (10s)'}

    dist_smooth = np.linspace(0.5, 44.5, 100)
    theta_smooth = 0.5 - dist_smooth / 90

    # === Panel A: P(LF) vs distance ===
    ax = axes[0]

    for tp in [5, 10]:
        data = binned_data[tp]
        T = float(tp)

        ax.errorbar(data['dist_mean'], data['p_lf'], yerr=data['p_lf_se'],
                   marker='o', color=colors[tp], linewidth=0, markersize=8,
                   capsize=3, label=f'{labels[tp]} (data)')

        pred = predict_all(theta_smooth, params['sigma'], params['delta_B'],
                          params['delta'], T, params['lapse'])
        ax.plot(dist_smooth, pred, color=colors[tp], linewidth=2.5,
               label=f'{labels[tp]} (model)')

    ax.axvline(22.5, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Distance from LF (°)', fontsize=11)
    ax.set_ylabel('P(LF)', fontsize=11)
    ax.set_title('Lexical Choice', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1)

    # === Panel B: Parameters & Stats ===
    ax = axes[1]
    fit_stats = compute_fit_statistics(params, binned_data)

    param_text = f"""
Reduced Race Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A(w, t) = B(w) + δ · M(θ, w) · t

Free parameters (4):
  σ (semantic width)  = {params['sigma']:.3f}
  ΔB (HF advantage)   = {params['delta_B']:.3f}
  δ (drift rate)      = {params['delta']:.3f}
  lapse               = {params['lapse']:.3f}

Fixed parameters:
  α = 1 (absorbed)
  B_lf = 0 (reference)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fit Statistics

Strict (5s):   R² = {fit_stats[5]['r2']:.4f}
Lenient (10s): R² = {fit_stats[10]['r2']:.4f}
Overall:       R² = {fit_stats['overall']['r2']:.4f}
"""
    ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {save_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("RSA RACE MODEL FITTING (Reduced Parameterization)")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data()
    print(f"N = {len(df):,} trials from {df['subject_id'].nunique()} participants")

    print("\nBinning data...")
    binned = get_binned_data(df, n_bins=9)

    print("\n" + "-" * 60)
    params = fit_model(binned, verbose=True)

    fit_stats = compute_fit_statistics(params, binned)

    print("\n" + "=" * 60)
    print("FIT STATISTICS")
    print("=" * 60)
    print(f"Strict (5s):   R² = {fit_stats[5]['r2']:.4f}")
    print(f"Lenient (10s): R² = {fit_stats[10]['r2']:.4f}")
    print(f"Overall:       R² = {fit_stats['overall']['r2']:.4f}")

    print("\n" + "-" * 60)
    plot_fit(params, binned)

    # Output for rsa_race.py
    print("\n" + "=" * 60)
    print("UPDATE rsa_race.py DEFAULT_PARAMS:")
    print("=" * 60)
    print(f"""
DEFAULT_PARAMS = {{
    'sigma': {params['sigma']:.2f},
    'alpha': 1.0,       # fixed
    'B_hf': {params['delta_B']:.2f},
    'B_lf': 0.0,        # reference
    'delta': {params['delta']:.2f},
    'lapse': {params['lapse']:.2f},
}}
""")

    np.savez('fitted_params.npz', **params)
    print("Saved fitted_params.npz")

    return params, fit_stats


if __name__ == "__main__":
    params, fit_stats = main()
