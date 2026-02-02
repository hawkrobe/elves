"""
RSA Race Model Fitting — Experiment 2 (payoff × time pressure × frequency ratio).

Design: 2 (time pressure: 5s vs 10s) × 2 (payoff: linear vs binary) × 2 (frequency ratio: 1:2 vs 1:4).
Fits B_HF per frequency ratio and T per (time_pressure, payoff_structure, frequency_ratio).

Usage:
    From repo root:  python models/fit_exp2.py

Data:
    Loads from data/full-production/raw_data/exp2-0, exp2-1, and exp2-2.
    exp2-0/exp2-1: 1:4; exp2-2: 1:2. Expects payoff_structure and frequency_ratio in raw CSVs.
    Next-nearest angle is reconstructed from geometry (nearest ±45°; whichever is closer to target),
    not from the jsPsych next_nearest_angle column.

Outputs (to models/):
    race_model_data_exp2.csv (trial-level data used for fit; includes datasource for comparison),
    race_model_datasource_summary_exp2.csv (per-datasource and per-condition counts and means),
    fitted_params_exp2.csv, race_model_predictions_exp2.csv,
    race_model_fit_exp2.png, race_model_fit_exp2.pdf,
    race_model_schematic_exp2.png/pdf, race_model_schematic_exp2_boundary.png/pdf,
    race_model_schematic_exp2_data.csv, race_model_schematic_exp2_params.csv,
    race_model_P_LF_P_HF_vs_time_exp2_boundary.png/pdf,
    race_model_P_LF_P_HF_vs_time_exp2_LF_prototype.png/pdf,
    race_model_P_LF_P_HF_vs_time_exp2_boundary.csv,
    race_model_P_LF_P_HF_vs_time_exp2_LF_prototype.csv
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as scipy_norm
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax.numpy as jnp
from fit import _compute_nll_vec, _race_model_vec


# =============================================================================
# Data loading (Exp 2: raw from exp2-0, exp2-1, exp2-2)
# =============================================================================

def _circ_dist(a, b):
    """Circular distance in [0, 360)."""
    d = np.abs(a - b)
    return np.minimum(d, 360 - d)


def _next_nearest_angle_from_geometry(nearest_deg, target_deg):
    """
    Next-nearest trained angle = the closer of the two ±45° neighbors of nearest (mod 360).
    Does not use jsPsych next_nearest_angle; reconstructs from geometry.
    """
    nearest = np.asarray(nearest_deg, dtype=float)
    target = np.asarray(target_deg, dtype=float)
    n_plus = (nearest + 45) % 360
    n_minus = (nearest - 45 + 360) % 360
    d_plus = _circ_dist(target, n_plus)
    d_minus = _circ_dist(target, n_minus)
    return np.where(d_plus < d_minus, n_plus, n_minus)


def load_data_exp2():
    """Load and preprocess Experiment 2 data from raw CSVs (exp2-0, exp2-1, exp2-2)."""
    base = Path(__file__).parent.parent / "data" / "full-production" / "raw_data"
    frames = []
    for folder in ["exp2-0", "exp2-1", "exp2-2"]:
        raw_dir = base / folder
        if not raw_dir.exists():
            continue
        files = [f for f in raw_dir.glob("*.csv") if "training_complete" not in f.name and "inprog" not in f.name]
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            if df.empty or "nearest_trained_angle" not in df.columns:
                continue
            df = df.copy()
            df["datasource"] = folder
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No Exp 2 raw CSVs found under {base}/exp2-0, exp2-1, exp2-2. "
            "Expect columns: subject_id, time_pressure, payoff_structure, frequency_ratio, nearest_trained_angle, etc."
        )
    raw = pd.concat(frames, ignore_index=True)

    # Normalize column names (exp2 runs may use different headers)
    def _norm(s):
        return str(s).strip().lower().replace(" ", "_")
    raw_col_norm = {c: _norm(c) for c in raw.columns}
    rename = {}
    for c in raw.columns:
        n = raw_col_norm[c]
        if n == "time_pressure" and c != "time_pressure":
            rename[c] = "time_pressure"
        elif n == "payoff_structure" and c != "payoff_structure":
            rename[c] = "payoff_structure"
        elif n == "frequency_ratio" and c != "frequency_ratio":
            rename[c] = "frequency_ratio"
    if rename:
        raw = raw.rename(columns=rename)

    # Participant-level: one row per subject. Prefer any row that has time_pressure (often
    # repeated on every trial in the raw CSV); fall back to survey rows if needed.
    part_cols = ["subject_id", "time_pressure", "payoff_structure"]
    if "frequency_ratio" in raw.columns:
        part_cols.append("frequency_ratio")
    part_cols = [c for c in part_cols if c in raw.columns]
    if "time_pressure" not in part_cols:
        raise ValueError(
            "Raw Exp 2 CSVs must have a time_pressure column (or 'time pressure' / 'timePressure'). "
            f"Found columns: {list(raw.columns)}"
        )
    # Build from any row that has time_pressure (production rows usually have it)
    part = raw.loc[raw["time_pressure"].notna(), part_cols].drop_duplicates("subject_id")
    if part.empty:
        part = raw.loc[raw["survey_difficulty"].notna(), part_cols].drop_duplicates("subject_id")
    part["time_pressure"] = pd.to_numeric(part["time_pressure"], errors="coerce")
    part = part.dropna(subset=["time_pressure"])
    if "payoff_structure" not in part.columns:
        part["payoff_structure"] = "linear"
    else:
        part["payoff_structure"] = part["payoff_structure"].astype(str).str.strip().str.lower()
        part.loc[~part["payoff_structure"].isin(["linear", "binary"]), "payoff_structure"] = "linear"
    if "frequency_ratio" not in part.columns:
        part["frequency_ratio"] = "1:4"  # exp2-0/1 only
    else:
        part["frequency_ratio"] = part["frequency_ratio"].astype(str).str.strip()
        part.loc[~part["frequency_ratio"].isin(["1:2", "1:4"]), "frequency_ratio"] = "1:4"

    # Exposure: angle -> word, freq (correct trials only so mapping is the learned angle->word)
    resp = raw["response"].astype(str)
    is_correct_val = (
        (pd.to_numeric(raw["is_correct"], errors="coerce") == 1)
        | (raw["is_correct"].astype(str).str.strip().str.lower().isin(("1", "true", "yes")))
    )
    exp_mask = is_correct_val & resp.str.contains("exposure", case=False, na=False)
    exposure = raw.loc[exp_mask, ["subject_id", "angle", "target", "targetFreq"]].drop_duplicates()
    exposure["angle"] = pd.to_numeric(exposure["angle"], errors="coerce")
    exposure = exposure.dropna(subset=["angle"])

    # Production trials
    prod = raw[raw["nearest_trained_angle"].notna()].copy()
    prod = prod.rename(columns={"angle": "targetAngle"})
    for c in ["targetAngle", "nearest_trained_angle", "rt", "timed_out"]:
        if c in prod.columns:
            prod[c] = pd.to_numeric(prod[c], errors="coerce")
    prod = prod.dropna(subset=["nearest_trained_angle", "targetAngle"])
    prod["nearestAngle"] = prod["nearest_trained_angle"]
    # Reconstruct next-nearest from geometry (nearest ±45°; whichever is closer to target). Do not use jsPsych next_nearest_angle.
    prod["nextNearestAngle"] = _next_nearest_angle_from_geometry(
        prod["nearest_trained_angle"].values, prod["targetAngle"].values
    )

    # Join exposure: nearestFreq, nextNearestFreq, responseFreq
    exp_angle = exposure.rename(columns={"angle": "nearestAngle", "targetFreq": "nearestFreq", "target": "nearestLabel"})
    prod = prod.merge(exp_angle[["subject_id", "nearestAngle", "nearestFreq"]], on=["subject_id", "nearestAngle"], how="left")
    exp_next = exposure.rename(columns={"angle": "nextNearestAngle", "targetFreq": "nextNearestFreq", "target": "nextNearestLabel"})
    prod = prod.merge(exp_next[["subject_id", "nextNearestAngle", "nextNearestFreq"]], on=["subject_id", "nextNearestAngle"], how="left")
    prod = prod.merge(exposure.rename(columns={"target": "response_text", "targetFreq": "responseFreq"})[["subject_id", "response_text", "responseFreq"]], on=["subject_id", "response_text"], how="left")
    prod = prod.merge(part, on="subject_id", how="inner", suffixes=("", "_part"))
    # Prefer participant-level values (canonical); drop _part suffix columns after copying
    for col in ["time_pressure", "payoff_structure", "frequency_ratio"]:
        part_col = f"{col}_part"
        if part_col in prod.columns:
            prod[col] = prod[part_col].values
            prod = prod.drop(columns=[part_col])
    if "payoff_structure" in prod.columns:
        prod["payoff_structure"] = prod["payoff_structure"].astype(str).str.strip().str.lower()
        prod.loc[~prod["payoff_structure"].isin(["linear", "binary"]), "payoff_structure"] = "linear"
    if "frequency_ratio" not in prod.columns and "frequency_ratio" in part.columns:
        prod["frequency_ratio"] = prod["subject_id"].map(part.set_index("subject_id")["frequency_ratio"])

    # Distance and theta (mirror fit.py / experiment2.qmd)
    prod["distance"] = _circ_dist(prod["targetAngle"].values, prod["nearestAngle"].values)
    prod["dist_to_lf_deg"] = np.where(
        prod["nearestFreq"] == "LF",
        np.abs(prod["distance"]),
        _circ_dist(prod["targetAngle"].values, prod["nextNearestAngle"].values),
    )
    prod["theta"] = 0.5 - prod["dist_to_lf_deg"] / 90
    prod["between_freqs"] = prod["nearestFreq"] != prod["nextNearestFreq"]
    prod["is_lf"] = (prod["responseFreq"] == "LF").astype(int)

    # Critical trials only
    df = prod[prod["between_freqs"]].copy()
    df = df[df["timed_out"] == 0]
    if "time_pressure" not in df.columns:
        raise ValueError("time_pressure missing after merge with participants. Check that participant table has time_pressure.")
    df = df[(df["rt"] > 200) & (df["rt"] < df["time_pressure"] * 1000)]
    df = df.dropna(subset=["theta", "is_lf", "payoff_structure"])
    if "frequency_ratio" not in df.columns:
        df["frequency_ratio"] = "1:4"
    # Exclude exp2-0, 1:4, linear (replaced in design; match experiment2.qmd). Keep datasource for export/summary.
    if "datasource" in df.columns:
        drop_mask = (df["datasource"] == "exp2-0") & (df["frequency_ratio"] == "1:4") & (df["payoff_structure"] == "linear")
        df = df.loc[~drop_mask]
    return df


# =============================================================================
# NLL and fitting (up to 8 conditions: time_pressure × payoff_structure × frequency_ratio)
# =============================================================================

def nll_exp2(params, df):
    """12 params: sigma, B_hf_12, B_hf_14, lapse, then 8 T values (linear/binary × 5/10 × 1:2/1:4)."""
    sigma, B_12, B_14, lapse = params[0], params[1], params[2], params[3]
    T_lin_5_12, T_lin_10_12, T_bin_5_12, T_bin_10_12 = params[4], params[5], params[6], params[7]
    T_lin_5_14, T_lin_10_14, T_bin_5_14, T_bin_10_14 = params[8], params[9], params[10], params[11]
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in params[4:12]):
        return 1e10
    tp = df["time_pressure"].values
    is_linear = (df["payoff_structure"] == "linear").values
    fr_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where((tp == 5) & is_linear & fr_12, T_lin_5_12,
          np.where((tp == 10) & is_linear & fr_12, T_lin_10_12,
            np.where((tp == 5) & ~is_linear & fr_12, T_bin_5_12,
              np.where((tp == 10) & ~is_linear & fr_12, T_bin_10_12,
                np.where((tp == 5) & is_linear & ~fr_12, T_lin_5_14,
                  np.where((tp == 10) & is_linear & ~fr_12, T_lin_10_14,
                    np.where((tp == 5) & ~is_linear & ~fr_12, T_bin_5_14, T_bin_10_14)))))))
    B_hf = np.where(fr_12, B_12, B_14)
    return _compute_nll_vec(df["theta"].values, T, B_hf, df["is_lf"].values, sigma, lapse)


def fit_model_exp2(df):
    """Fit race model with B per frequency ratio and T per (time_pressure, payoff_structure, frequency_ratio)."""
    # Starting values: sigma, B_12, B_14, lapse, then 8 T's (order: lin5_12, lin10_12, bin5_12, bin10_12, lin5_14, lin10_14, bin5_14, bin10_14)
    x0 = [0.2, 0.5, 0.9, 0.15, 2.0, 3.5, 2.0, 3.5, 2.0, 3.5, 2.0, 3.5]
    print(f"Fitting Exp 2 model ({len(df):,} trials)...")
    result = minimize(nll_exp2, x0, args=(df,), method="Nelder-Mead",
                      options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-4})
    names = [
        "sigma", "B_hf_12", "B_hf_14", "lapse",
        "T_linear_5_12", "T_linear_10_12", "T_binary_5_12", "T_binary_10_12",
        "T_linear_5_14", "T_linear_10_14", "T_binary_5_14", "T_binary_10_14",
    ]
    params = dict(zip(names, result.x))
    params["nll"] = result.fun
    params["n_trials"] = len(df)
    return params


# =============================================================================
# Model comparison (nested variants)
# =============================================================================

def nll_exp2_shared_B(params, df):
    """11 params: sigma, B_hf (single), lapse, 8 T's. B same across frequency ratio."""
    sigma, B_hf, lapse = params[0], params[1], params[2]
    T_lin_5_12, T_lin_10_12, T_bin_5_12, T_bin_10_12 = params[3], params[4], params[5], params[6]
    T_lin_5_14, T_lin_10_14, T_bin_5_14, T_bin_10_14 = params[7], params[8], params[9], params[10]
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in params[3:11]):
        return 1e10
    tp = df["time_pressure"].values
    is_linear = (df["payoff_structure"] == "linear").values
    fr_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where((tp == 5) & is_linear & fr_12, T_lin_5_12,
          np.where((tp == 10) & is_linear & fr_12, T_lin_10_12,
            np.where((tp == 5) & ~is_linear & fr_12, T_bin_5_12,
              np.where((tp == 10) & ~is_linear & fr_12, T_bin_10_12,
                np.where((tp == 5) & is_linear & ~fr_12, T_lin_5_14,
                  np.where((tp == 10) & is_linear & ~fr_12, T_lin_10_14,
                    np.where((tp == 5) & ~is_linear & ~fr_12, T_bin_5_14, T_bin_10_14)))))))
    return _compute_nll_vec(df["theta"].values, T, np.full(len(df), B_hf), df["is_lf"].values, sigma, lapse)


def nll_exp2_T_no_payoff(params, df):
    """8 params: sigma, B_12, B_14, lapse, T_5_12, T_10_12, T_5_14, T_10_14. T same for linear/binary within (freq, time)."""
    sigma, B_12, B_14, lapse = params[0], params[1], params[2], params[3]
    T_5_12, T_10_12, T_5_14, T_10_14 = params[4], params[5], params[6], params[7]
    if sigma <= 0.05 or sigma > 2.0 or lapse < 0 or lapse > 0.5:
        return 1e10
    if any(t <= 0 for t in [T_5_12, T_10_12, T_5_14, T_10_14]):
        return 1e10
    tp = df["time_pressure"].values
    fr_12 = (df["frequency_ratio"] == "1:2").values
    T = np.where((tp == 5) & fr_12, T_5_12,
          np.where((tp == 10) & fr_12, T_10_12,
            np.where((tp == 5) & ~fr_12, T_5_14, T_10_14)))
    B_hf = np.where(fr_12, B_12, B_14)
    return _compute_nll_vec(df["theta"].values, T, B_hf, df["is_lf"].values, sigma, lapse)


def fit_model_comparison_exp2(df, full_params=None):
    """Fit full and nested variants; return dict of models and n_obs."""
    from scipy.stats import chi2
    n_obs = len(df)
    models = {}

    if full_params is None:
        x0_full = [0.2, 0.5, 0.9, 0.15, 2.0, 3.5, 2.0, 3.5, 2.0, 3.5, 2.0, 3.5]
    else:
        x0_full = [
            full_params["sigma"], full_params["B_hf_12"], full_params["B_hf_14"], full_params["lapse"],
            full_params["T_linear_5_12"], full_params["T_linear_10_12"], full_params["T_binary_5_12"], full_params["T_binary_10_12"],
            full_params["T_linear_5_14"], full_params["T_linear_10_14"], full_params["T_binary_5_14"], full_params["T_binary_10_14"],
        ]
    b_sig = (0.05, 2.0)
    b_B = (-2.0, 2.0)
    b_lapse = (0.001, 0.5)
    b_T = (0.1, 30.0)

    def fit_variant(name, nll_fn, x0, bounds, k):
        print(f"  {name}...", end=" ", flush=True)
        result = minimize(nll_fn, x0, args=(df,), method="L-BFGS-B", bounds=bounds)
        print(f"NLL={result.fun:.1f}")
        return {"k": k, "nll": result.fun}

    # Full (12)
    models["Full (B×FR, T×FR×payoff×TP)"] = fit_variant(
        "Full (12)", nll_exp2, x0_full,
        [b_sig, b_B, b_B, b_lapse] + [b_T] * 8, 12)

    # Shared B (11)
    x0_shared_B = [x0_full[0], (x0_full[1] + x0_full[2]) / 2, x0_full[3]] + x0_full[4:12]
    models["Shared B_HF (T×FR×payoff×TP)"] = fit_variant(
        "Shared B (11)", nll_exp2_shared_B, x0_shared_B,
        [b_sig, b_B, b_lapse] + [b_T] * 8, 11)

    # T no payoff (8)
    x0_no_payoff = [
        x0_full[0], x0_full[1], x0_full[2], x0_full[3],
        (x0_full[4] + x0_full[6]) / 2, (x0_full[5] + x0_full[7]) / 2,
        (x0_full[8] + x0_full[10]) / 2, (x0_full[9] + x0_full[11]) / 2,
    ]
    models["T×FR×TP only (no payoff)"] = fit_variant(
        "T no payoff (8)", nll_exp2_T_no_payoff, x0_no_payoff,
        [b_sig, b_B, b_B, b_lapse] + [b_T] * 4, 8)

    for m in models.values():
        m["aic"] = 2 * m["k"] + 2 * m["nll"]
        m["bic"] = m["k"] * np.log(n_obs) + 2 * m["nll"]

    return models, n_obs


# =============================================================================
# Plot: 4 panels (frequency_ratio × time_pressure); payoff = solid (linear) vs dashed (binary)
# =============================================================================

def plot_fit_exp2(df, params, out_dir=None):
    """2×2 facets: columns = frequency ratio (1:2, 1:4), rows = payoff (linear, binary). Strict (5s)=red, Lenient (10s)=blue."""
    if out_dir is None:
        out_dir = Path(__file__).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 9
    dist_lo, dist_hi = 0.0, 45.0
    dist_grid = np.linspace(0.5, 44.5, 200)
    theta_grid = 0.5 - dist_grid / 90.0

    group_cols = ["frequency_ratio", "time_pressure", "payoff_structure"]
    if "frequency_ratio" not in df.columns:
        df = df.copy()
        df["frequency_ratio"] = "1:4"
    df_plot = df.copy()
    df_plot["dist_bin"] = pd.cut(
        df_plot["dist_to_lf_deg"],
        bins=np.linspace(dist_lo, dist_hi, n_bins + 1),
        include_lowest=True,
    )
    binned = (
        df_plot.groupby(group_cols + ["dist_bin"], observed=True)
        .agg(
            dist_mean=("dist_to_lf_deg", "mean"),
            p_lf=("is_lf", "mean"),
            n=("is_lf", "count"),
        )
        .reset_index()
    )
    binned["p_lf_se"] = np.sqrt(binned["p_lf"] * (1 - binned["p_lf"]) / binned["n"])

    sigma = params["sigma"]
    lapse = params["lapse"]

    def curve_p_lf(B_hf, T):
        probs = _race_model_vec(
            jnp.array(theta_grid),
            sigma,
            jnp.full(len(theta_grid), B_hf),
            1.0,
            jnp.full(len(theta_grid), T),
            lapse,
        )
        return np.array(probs[:, 1])

    curves = {}
    T_keys = {
        ("1:2", 5, "linear"): "T_linear_5_12", ("1:2", 10, "linear"): "T_linear_10_12",
        ("1:2", 5, "binary"): "T_binary_5_12", ("1:2", 10, "binary"): "T_binary_10_12",
        ("1:4", 5, "linear"): "T_linear_5_14", ("1:4", 10, "linear"): "T_linear_10_14",
        ("1:4", 5, "binary"): "T_binary_5_14", ("1:4", 10, "binary"): "T_binary_10_14",
    }
    for (fr, tp, pay), key in T_keys.items():
        if key not in params:
            continue
        B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
        curves[(fr, tp, pay)] = curve_p_lf(B_hf, params[key])

    # Strict (5s) = red, Lenient (10s) = blue; payoff faceted so only 2 lines per panel
    colors_tp = {5: "#C0392B", 10: "#2980B9"}  # red, blue
    tp_labels = {5: "Strict (5s)", 10: "Lenient (10s)"}
    pay_labels = {"linear": "Linear", "binary": "Binary"}

    # 2×2: rows = payoff (linear, binary), columns = frequency ratio (1:2, 1:4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for i_row, pay in enumerate(["linear", "binary"]):
        for i_col, fr in enumerate(["1:2", "1:4"]):
            ax = axes[i_row, i_col]
            for tp in [5, 10]:
                key = (fr, tp, pay)
                if key not in curves:
                    continue
                sub = binned[(binned["frequency_ratio"] == fr) & (binned["time_pressure"] == tp) & (binned["payoff_structure"] == pay)]
                if not sub.empty:
                    ax.errorbar(
                        sub["dist_mean"], sub["p_lf"], yerr=sub["p_lf_se"],
                        fmt="o", color=colors_tp[tp], capsize=3,
                    )
                ax.plot(
                    dist_grid, curves[key],
                    color=colors_tp[tp], linewidth=1.5, label=tp_labels[tp],
                )
            ax.axvline(22.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xlim(0, 45)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Distance from LF (deg)")
            ax.set_ylabel("P(LF)")
            ax.set_title(f"{pay_labels[pay]} — FR {fr}")
            ax.legend(loc="upper right", framealpha=0.9)
            ax.set_xticks(np.arange(0, 46, 10))
            ax.set_yticks(np.arange(0, 1.25, 0.25))
            ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out_path = out_dir / f"race_model_fit_exp2.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"Saved {out_path}")
    plt.close(fig)

    pred_rows = []
    for (fr, tp, pay), p_lf in curves.items():
        for d, p in zip(dist_grid, p_lf):
            pred_rows.append({"dist_to_lf": d, "frequency_ratio": fr, "time_pressure": tp, "payoff_structure": pay, "p_lf": p})
    pd.DataFrame(pred_rows).to_csv(out_dir / "race_model_predictions_exp2.csv", index=False)
    print(f"Saved {out_dir / 'race_model_predictions_exp2.csv'}")


# =============================================================================
# Schematic: activation vs time (2×2 facets: columns = FR, rows = payoff)
# =============================================================================

def plot_race_schematic_exp2(params, out_dir=None, theta=0.5, suffix=""):
    """
    Plot activation vs time at a given θ: HF vs LF baseline at t=0 and slopes (δ·M(θ,w)).
    2×2 facets: columns = frequency ratio (1:2, 1:4), rows = payoff (linear, binary).
    Each panel shows HF/LF activation and vertical lines at T for Strict (5s) and Lenient (10s).
    """
    if out_dir is None:
        out_dir = Path(__file__).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma = params["sigma"]
    delta = 1.0
    word_locs = [0.0, 0.5, 1.0]  # HF1, LF, HF2
    M_hf = float(scipy_norm.pdf(theta, word_locs[0], sigma))
    M_lf = float(scipy_norm.pdf(theta, word_locs[1], sigma))
    slope_hf = delta * M_hf
    slope_lf = delta * M_lf

    t_max = 6.0
    t_vals = np.linspace(0, t_max, 200)

    color_hf = "#7B7B7B"
    color_lf = "#2E86AB"
    color_t5 = "#C0392B"   # Strict (5s) red
    color_t10 = "#2980B9"  # Lenient (10s) blue

    T_keys = {
        ("1:2", "linear"): ("T_linear_5_12", "T_linear_10_12"),
        ("1:2", "binary"): ("T_binary_5_12", "T_binary_10_12"),
        ("1:4", "linear"): ("T_linear_5_14", "T_linear_10_14"),
        ("1:4", "binary"): ("T_binary_5_14", "T_binary_10_14"),
    }
    pay_labels = {"linear": "Linear", "binary": "Binary"}

    # 2×2: rows = payoff (linear, binary), columns = frequency ratio (1:2, 1:4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for i_row, pay in enumerate(["linear", "binary"]):
        for i_col, fr in enumerate(["1:2", "1:4"]):
            ax = axes[i_row, i_col]
            B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
            k5, k10 = T_keys[(fr, pay)]
            T_5 = params.get(k5, np.nan)
            T_10 = params.get(k10, np.nan)

            A_hf = B_hf + slope_hf * t_vals
            A_lf = 0 + slope_lf * t_vals
            ax.plot(t_vals, A_hf, color=color_hf, lw=2, label=f"HF (B = {B_hf:.2f})")
            ax.plot(t_vals, A_lf, color=color_lf, lw=2, label="LF (B = 0)")
            ax.axhline(B_hf, color=color_hf, ls=":", alpha=0.6)
            ax.axhline(0, color=color_lf, ls=":", alpha=0.6)
            if not np.isnan(T_5):
                ax.axvline(T_5, color=color_t5, ls="--", lw=1.5, alpha=0.8, label="Strict (5s)")
            if not np.isnan(T_10):
                ax.axvline(T_10, color=color_t10, ls="--", lw=1.5, alpha=0.8, label="Lenient (10s)")
            ax.set_xlim(0, t_max)
            ax.set_xlabel("Time (effective units)")
            ax.set_ylabel("Activation" if i_col == 0 else "")
            ax.set_title(f"{pay_labels[pay]} — FR {fr}")
            ax.legend(loc="upper left", frameon=True, fontsize=8)
            ax.grid(True, alpha=0.3)
            theta_label = "θ = 0.5 (LF prototype)" if theta == 0.5 else "θ = 0.25 (boundary)"
            ax.text(0.98, 0.02, f"{theta_label}\nslope HF = {slope_hf:.3f}\nslope LF = {slope_lf:.3f}",
                    transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()

    base = f"race_model_schematic_exp2{suffix}"
    for ext in ["png", "pdf"]:
        out_path = out_dir / f"{base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"Saved {out_path}")
    plt.close(fig)

    # Export schematic data for R/ggplot
    if suffix == "":
        schematic_data = []
        for fr in ["1:2", "1:4"]:
            B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
            for pay in ["linear", "binary"]:
                k5, k10 = T_keys[(fr, pay)]
                T_5, T_10 = params.get(k5, np.nan), params.get(k10, np.nan)
                for word, baseline, slope in [("HF", B_hf, slope_hf), ("LF", 0.0, slope_lf)]:
                    for t in np.linspace(0, t_max, 50):
                        schematic_data.append({
                            "frequency_ratio": fr,
                            "payoff_structure": pay,
                            "word": word,
                            "baseline": baseline,
                            "slope": slope,
                            "t": t,
                            "activation": baseline + slope * t,
                        })
        pd.DataFrame(schematic_data).to_csv(out_dir / "race_model_schematic_exp2_data.csv", index=False)
        summary = []
        for fr in ["1:2", "1:4"]:
            B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
            for pay in ["linear", "binary"]:
                k5, k10 = T_keys[(fr, pay)]
                summary.append({
                    "frequency_ratio": fr,
                    "payoff_structure": pay,
                    "B_hf": B_hf,
                    "T_5": params.get(k5, np.nan),
                    "T_10": params.get(k10, np.nan),
                })
        pd.DataFrame(summary).to_csv(out_dir / "race_model_schematic_exp2_params.csv", index=False)
        print(f"Saved {out_dir / 'race_model_schematic_exp2_data.csv'} and race_model_schematic_exp2_params.csv")


# =============================================================================
# P(LF) and P(HF) as a function of decision time (T) — shift from one to the other
# =============================================================================

def plot_P_LF_P_HF_vs_time_exp2(params, out_dir=None, theta=0.25, theta_label=None, suffix=""):
    """
    Plot P(LF) and P(HF) on the same axes as a function of effective decision time T.

    Shows how producers shift from HF to LF (or vice versa) as T increases.
    Uses fitted σ, B_HF per FR; at a fixed θ. 2×2 facets: columns = FR, rows = payoff.
    Marks fitted T for Strict (5s) and Lenient (10s) in each panel.

    Note: At θ=0.25 (boundary), M(θ, HF) = M(θ, LF) so drift slopes are equal; HF always
    leads due to baseline B_HF and LF never overtakes. At θ=0.5 (LF prototype), M(θ, LF)
    is maximal and LF can beat HF as T increases — use theta=0.5 to see that crossover.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if theta_label is None:
        theta_label = f"θ = {theta}" + (" (boundary)" if theta == 0.25 else " (LF prototype)" if theta == 0.5 else "")

    sigma = params["sigma"]
    lapse = params["lapse"]
    T_grid = np.linspace(0.2, 5.0, 300)
    theta_vec = jnp.full(len(T_grid), theta)

    T_keys = {
        ("1:2", "linear"): ("T_linear_5_12", "T_linear_10_12"),
        ("1:2", "binary"): ("T_binary_5_12", "T_binary_10_12"),
        ("1:4", "linear"): ("T_linear_5_14", "T_linear_10_14"),
        ("1:4", "binary"): ("T_binary_5_14", "T_binary_10_14"),
    }
    pay_labels = {"linear": "Linear", "binary": "Binary"}
    color_lf = "#2980B9"   # blue for LF
    color_hf = "#7B7B7B"   # gray for HF
    color_t5 = "#C0392B"    # red Strict (5s)
    color_t10 = "#1A5276"   # darker blue Lenient (10s), distinct from P(LF) curve

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    for i_row, pay in enumerate(["linear", "binary"]):
        for i_col, fr in enumerate(["1:2", "1:4"]):
            ax = axes[i_row, i_col]
            B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
            probs = _race_model_vec(
                theta_vec,
                sigma,
                jnp.full(len(T_grid), B_hf),
                1.0,
                jnp.array(T_grid),
                lapse,
            )
            probs = np.array(probs)
            p_lf = probs[:, 1]
            p_hf = probs[:, 0] + probs[:, 2]

            ax.plot(T_grid, p_lf, color=color_lf, linewidth=1.5, label="P(LF)")
            ax.plot(T_grid, p_hf, color=color_hf, linewidth=1.5, label="P(HF)")

            k5, k10 = T_keys[(fr, pay)]
            T_5, T_10 = params.get(k5, np.nan), params.get(k10, np.nan)
            if not np.isnan(T_5) and T_5 <= T_grid.max():
                idx5 = np.argmin(np.abs(T_grid - T_5))
                ax.axvline(T_5, color=color_t5, ls="--", lw=1.2, alpha=0.8, label="Strict (5s)")
                ax.plot(T_5, p_lf[idx5], "o", color=color_lf, markersize=6)
                ax.plot(T_5, p_hf[idx5], "o", color=color_hf, markersize=6)
            if not np.isnan(T_10) and T_10 <= T_grid.max():
                idx10 = np.argmin(np.abs(T_grid - T_10))
                ax.axvline(T_10, color=color_t10, ls="--", lw=1.2, alpha=0.8, label="Lenient (10s)")
                ax.plot(T_10, p_lf[idx10], "o", color=color_lf, markersize=6)
                ax.plot(T_10, p_hf[idx10], "o", color=color_hf, markersize=6)

            ax.set_xlim(T_grid.min(), T_grid.max())
            ax.set_ylim(0, 1)
            ax.set_xlabel("Effective decision time T")
            ax.set_ylabel("P(LF) / P(HF)" if i_col == 0 else "")
            ax.set_title(f"{pay_labels[pay]} — FR {fr}")
            ax.legend(loc="center right", frameon=True, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.98, theta_label, transform=ax.transAxes, fontsize=7, va="top", ha="left")

    plt.tight_layout()
    base = f"race_model_P_LF_P_HF_vs_time_exp2{suffix}"
    for ext in ["png", "pdf"]:
        out_path = out_dir / f"{base}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"Saved {out_path}")
    plt.close(fig)

    # Export curve grid for R/ggplot: T, p_lf, p_hf per (FR, payoff)
    curve_rows = []
    for pay in ["linear", "binary"]:
        for fr in ["1:2", "1:4"]:
            B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
            probs = _race_model_vec(
                theta_vec,
                sigma,
                jnp.full(len(T_grid), B_hf),
                1.0,
                jnp.array(T_grid),
                lapse,
            )
            probs = np.array(probs)
            p_lf = probs[:, 1]
            p_hf = probs[:, 0] + probs[:, 2]
            for t, plf, phf in zip(T_grid, p_lf, p_hf):
                curve_rows.append({
                    "frequency_ratio": fr, "payoff_structure": pay, "T": t,
                    "p_lf": plf, "p_hf": phf, "theta": theta,
                })
    out_csv = out_dir / f"race_model_P_LF_P_HF_vs_time_exp2{suffix}.csv"
    pd.DataFrame(curve_rows).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


# =============================================================================
# Fit quality (binned observed vs predicted)
# =============================================================================

def _print_binned_fit_quality(df, params, n_bins=9):
    """Print correlation of binned observed P(LF) vs model predicted P(LF)."""
    group_cols = ["frequency_ratio", "time_pressure", "payoff_structure"]
    if "frequency_ratio" not in df.columns:
        df = df.copy()
        df["frequency_ratio"] = "1:4"
    dist_lo, dist_hi = 0.0, 45.0
    df_plot = df.copy()
    df_plot["dist_bin"] = pd.cut(
        df_plot["dist_to_lf_deg"],
        bins=np.linspace(dist_lo, dist_hi, n_bins + 1),
        include_lowest=True,
    )
    binned = (
        df_plot.groupby(group_cols + ["dist_bin"], observed=True)
        .agg(dist_mean=("dist_to_lf_deg", "mean"), p_lf_obs=("is_lf", "mean"))
        .reset_index()
    )
    sigma = params["sigma"]
    lapse = params["lapse"]
    T_keys = {
        ("1:2", 5, "linear"): "T_linear_5_12", ("1:2", 10, "linear"): "T_linear_10_12",
        ("1:2", 5, "binary"): "T_binary_5_12", ("1:2", 10, "binary"): "T_binary_10_12",
        ("1:4", 5, "linear"): "T_linear_5_14", ("1:4", 10, "linear"): "T_linear_10_14",
        ("1:4", 5, "binary"): "T_binary_5_14", ("1:4", 10, "binary"): "T_binary_10_14",
    }
    binned["p_lf_pred"] = np.nan
    for (fr, tp, pay), grp in binned.groupby(group_cols):
        key = (fr, int(tp), pay)
        if key not in T_keys:
            continue
        B_hf = params["B_hf_12"] if fr == "1:2" else params["B_hf_14"]
        T = params[T_keys[key]]
        theta = 0.5 - grp["dist_mean"].values / 90.0
        probs = _race_model_vec(
            jnp.array(theta), sigma, jnp.full(len(theta), B_hf), 1.0, jnp.full(len(theta), T), lapse,
        )
        binned.loc[grp.index, "p_lf_pred"] = np.array(probs[:, 1])
    valid = binned["p_lf_pred"].notna()
    if valid.sum() < 2:
        print("  (Binned fit quality: too few bins)")
        return
    r = np.corrcoef(binned.loc[valid, "p_lf_obs"], binned.loc[valid, "p_lf_pred"])[0, 1]
    rmse = np.sqrt(np.mean((binned.loc[valid, "p_lf_obs"] - binned.loc[valid, "p_lf_pred"]) ** 2))
    print(f"  Binned observed vs predicted: r = {r:.3f}, RMSE = {rmse:.3f}  (n_bins = {valid.sum()})")


def _print_and_save_datasource_summary(df, out_dir):
    """Print and save per-datasource and per-condition summary to identify why exp2-0/1/2 differ."""
    if "datasource" not in df.columns:
        return
    # Per (datasource, frequency_ratio, payoff_structure, time_pressure)
    grp = df.groupby(["datasource", "frequency_ratio", "payoff_structure", "time_pressure"], dropna=False)
    agg = grp.agg(
        n_participants=("subject_id", "nunique"),
        n_trials=("subject_id", "count"),
        mean_responded_lf=("is_lf", "mean"),
    ).reset_index()
    if "rt" in df.columns:
        rt_agg = grp["rt"].mean().reset_index().rename(columns={"rt": "mean_rt_ms"})
        agg = agg.merge(rt_agg, on=["datasource", "frequency_ratio", "payoff_structure", "time_pressure"])
    out_path = out_dir / "race_model_datasource_summary_exp2.csv"
    agg.to_csv(out_path, index=False)
    print()
    print("DATASOURCE SUMMARY (exp2-0 vs exp2-1 vs exp2-2)")
    print("-" * 60)
    for ds in agg["datasource"].unique():
        sub = agg[agg["datasource"] == ds]
        n_part = df.loc[df["datasource"] == ds, "subject_id"].nunique()
        n_trials = len(df[df["datasource"] == ds])
        p_lf = df.loc[df["datasource"] == ds, "is_lf"].mean()
        rt_str = f", mean RT = {df.loc[df['datasource'] == ds, 'rt'].mean():.0f} ms" if "rt" in df.columns else ""
        print(f"  {ds}: {n_part} participants, {n_trials:,} trials, mean P(LF) = {p_lf:.3f}{rt_str}")
        for _, row in sub.iterrows():
            print(f"    -> {row['frequency_ratio']} {row['payoff_structure']} {int(row['time_pressure'])}s: "
                  f"{int(row['n_trials'])} trials, P(LF) = {row['mean_responded_lf']:.3f}")
    print(f"  Saved {out_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RSA RACE MODEL — EXPERIMENT 2 (payoff × time pressure × frequency ratio)")
    print("=" * 70)
    print()
    out_dir = Path(__file__).parent

    print("Loading Exp 2 data (raw exp2-0, exp2-1, exp2-2)...")
    df = load_data_exp2()
    print(f"  N = {len(df):,} trials from {df['subject_id'].nunique()} participants")
    for fr in ["1:2", "1:4"]:
        sub = df[df["frequency_ratio"] == fr]
        if sub.empty:
            continue
        for tp in [5, 10]:
            for pay in ["linear", "binary"]:
                n = len(sub[(sub["time_pressure"] == tp) & (sub["payoff_structure"] == pay)])
                if n > 0:
                    print(f"  FR {fr}, {tp}s, {pay}: {n:,} trials")
    print()

    # Export exact trial-level data used for fitting (so R can plot same data and align with model)
    export_cols = ["dist_to_lf_deg", "is_lf", "time_pressure", "payoff_structure", "frequency_ratio"]
    if "datasource" in df.columns:
        export_cols.append("datasource")
    df_export = df[export_cols].copy()
    df_export = df_export.rename(columns={"dist_to_lf_deg": "dist_to_lf", "is_lf": "responded_lf"})
    df_export.to_csv(out_dir / "race_model_data_exp2.csv", index=False)
    print("Saved race_model_data_exp2.csv (trial-level data used for fit)")
    # Datasource summary (identify why exp2-0 / exp2-1 / exp2-2 differ)
    _print_and_save_datasource_summary(df, out_dir)

    params = fit_model_exp2(df)
    n_obs = params["n_trials"]
    k = 12
    aic = 2 * k + 2 * params["nll"]
    bic = k * np.log(n_obs) + 2 * params["nll"]
    print()
    print("FITTED PARAMETERS (Exp 2, δ = 1)")
    print("-" * 50)
    print(f"  σ = {params['sigma']:.3f}, λ = {params['lapse']:.3f}")
    print(f"  B_HF 1:2 = {params['B_hf_12']:.3f}, B_HF 1:4 = {params['B_hf_14']:.3f}")
    print("  T by (freq_ratio, payoff, time):")
    for fr in ["1:2", "1:4"]:
        for pay in ["linear", "binary"]:
            t5 = params.get(f"T_{pay}_5_{fr.replace(':', '')}", np.nan)
            t10 = params.get(f"T_{pay}_10_{fr.replace(':', '')}", np.nan)
            print(f"    {fr} {pay:6s}  5s={t5:.2f}  10s={t10:.2f}")
    print()
    print("FIT (trial-level NLL)")
    print("-" * 50)
    print(f"  NLL = {params['nll']:.1f}   AIC = {aic:.1f}   BIC = {bic:.1f}   N = {n_obs:,} trials")
    # Binned fit quality: correlation of observed vs predicted P(LF)
    _print_binned_fit_quality(df, params)
    print()

    save = {k: v for k, v in params.items() if k not in ["nll", "n_trials"]}
    pd.DataFrame([save]).to_csv(out_dir / "fitted_params_exp2.csv", index=False)
    print("Saved fitted_params_exp2.csv")
    print()

    # Model comparison
    print("MODEL COMPARISON")
    print("-" * 50)
    print("Fitting nested variants...")
    model_variants, n_obs = fit_model_comparison_exp2(df, full_params=params)
    print()
    sorted_models = sorted(model_variants.items(), key=lambda x: x[1]["aic"])
    print(f"{'Model':<32} {'k':>3} {'NLL':>12} {'AIC':>12} {'BIC':>12} {'ΔAIC':>8}")
    print("-" * 82)
    best_aic = sorted_models[0][1]["aic"]
    for name, m in sorted_models:
        delta_aic = m["aic"] - best_aic
        print(f"{name:<32} {m['k']:>3} {m['nll']:>12.1f} {m['aic']:>12.1f} {m['bic']:>12.1f} {delta_aic:>8.1f}")
    print()
    nll_full = model_variants["Full (B×FR, T×FR×payoff×TP)"]["nll"]
    nll_shared_B = model_variants["Shared B_HF (T×FR×payoff×TP)"]["nll"]
    nll_no_payoff = model_variants["T×FR×TP only (no payoff)"]["nll"]
    from scipy.stats import chi2
    print("Likelihood ratio tests (nested):")
    lr = 2 * (nll_shared_B - nll_full)
    print(f"  Full vs Shared B:     LR = {lr:.2f}, df = 1, p = {1 - chi2.cdf(lr, 1):.3f}")
    lr = 2 * (nll_no_payoff - nll_full)
    print(f"  Full vs T no payoff:  LR = {lr:.2f}, df = 4, p = {1 - chi2.cdf(lr, 4):.3f}")
    print()

    print("Plotting (2×2 facets: FR × payoff; Strict/Lenient = red/blue)...")
    plot_fit_exp2(df, params, out_dir=out_dir)
    print("Plotting race model schematic Exp 2 (θ = 0.5, LF prototype)...")
    plot_race_schematic_exp2(params, out_dir=out_dir, theta=0.5)
    print("Plotting race model schematic Exp 2 (θ = 0.25, boundary)...")
    plot_race_schematic_exp2(params, out_dir=out_dir, theta=0.25, suffix="_boundary")
    print("Plotting P(LF) and P(HF) vs decision time (θ = 0.25, boundary)...")
    plot_P_LF_P_HF_vs_time_exp2(params, out_dir=out_dir, theta=0.25, suffix="_boundary")
    print("Plotting P(LF) and P(HF) vs decision time (θ = 0.5, LF prototype)...")
    plot_P_LF_P_HF_vs_time_exp2(params, out_dir=out_dir, theta=0.5, suffix="_LF_prototype")
    print("Done.")
    return params


if __name__ == "__main__":
    main()
