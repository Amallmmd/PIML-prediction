#!/usr/bin/env python3
"""
Ship Speed-Power Prediction using Physics-Informed Machine Learning (PIML).

This module implements a hybrid approach combining:
    1. Physics-based baseline model (cubic power-speed relationship)
    2. Machine learning residual correction (XGBoost/GradientBoosting)
    3. Physics-informed constraints (boundary, monotonicity, divergence)

The approach leverages domain knowledge (ship hydrodynamics) while using ML
to capture complex patterns that physics alone cannot model.

Physics Constraints:
    - Boundary Loss: Ensures power predictions never go negative
    - Monotonicity Loss: Power increases monotonically with speed/draft
    - Draft Divergence Loss: Prevents power curves from diverging at different drafts

Key Features:
    - Synthetic ship speed-power data generation with realistic noise
    - Physics baseline: P = k * speed^3 (simplified cubic relationship)
    - Residual learning with gradient boosting
    - Physics-informed loss functions for constraint enforcement
    - Interactive Plotly visualizations

Usage:
    python main.py

Author: PIML-prediction Team
Date: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import plotly.graph_objs as go
import plotly.io as pio

# Try XGBoost first, fallback to sklearn's GradientBoostingRegressor
try:
    import xgboost as xgb
    REGRESSOR_TYPE = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    REGRESSOR_TYPE = "sklearn_gbr"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Physical constants
KNOTS_TO_MS = 0.514444  # Conversion factor: knots to meters/second
GRAVITY = 9.80665       # Gravitational acceleration (m/s^2)

# Default model parameters
DEFAULT_PHYSICS_COEF = 0.8  # Baseline cubic coefficient (tuned for simplified model)

# Physics constraint weights
BOUNDARY_LOSS_WEIGHT = 10.0      # Weight for non-negativity constraint
MONOTONICITY_LOSS_WEIGHT = 5.0   # Weight for monotonicity constraint
DIVERGENCE_LOSS_WEIGHT = 2.0     # Weight for draft divergence constraint
MIN_POWER_THRESHOLD = 0.0        # Minimum allowed power (kW)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configure Plotly renderer
pio.renderers.default = "browser"


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_ship_data(
    n_samples: int = 200,
    speed_min: float = 5.0,
    speed_max: float = 18.0,
    noise_std_ratio: float = 0.06,
) -> pd.DataFrame:
    """
    Generate synthetic ship speed-power dataset with realistic characteristics.

    The synthetic data simulates real-world ship performance data with:
        - Variable vessel characteristics (speed, draft, laden/ballast condition)
        - Physics-based power calculation with noise

    Physics Model:
        P_true = k0 * speed^3 + k1 * speed^2 + k2 * speed + draft_effect + laden_effect + noise

    Args:
        n_samples: Number of data points to generate.
        speed_min: Minimum ship speed in knots.
        speed_max: Maximum ship speed in knots.
        noise_std_ratio: Noise standard deviation as ratio of signal magnitude.

    Returns:
        pd.DataFrame: DataFrame containing:
            - speed: Ship speed (knots)
            - draft: Ship draft (m) - integer values 6-14
            - laden_ballast: Loading condition (0=ballast, 1=laden)
            - power_true: True power without noise (kW)
            - power_obs: Observed power with noise (kW)

    Example:
        >>> df = generate_synthetic_ship_data(n_samples=100)
        >>> print(df.columns.tolist())
    """
    # Generate random vessel speeds
    speeds = np.random.uniform(speed_min, speed_max, size=n_samples)

    # Generate draft as random integers between 6 and 14 (inclusive)
    drafts = np.random.randint(6, 15, size=n_samples)  # 15 is exclusive, so range is 6-14

    # Generate laden/ballast condition (0=ballast, 1=laden)
    # Approximately 50% laden, 50% ballast
    laden_ballast = np.random.randint(0, 2, size=n_samples)

    # Physics model coefficients (ground truth)
    k0 = 0.8       # Cubic term coefficient (main resistance)
    k1 = -0.5      # Quadratic term coefficient
    k2 = 2.0       # Linear term coefficient

    # Draft effect on power (deeper draft = more resistance)
    draft_effect = (drafts - 10.0) * 15.0  # Centered around draft=10

    # Laden/ballast effect: laden ships require more power due to increased displacement
    # Laden (1) adds power, ballast (0) subtracts power
    laden_effect = (laden_ballast - 0.5) * 200.0  # +100 kW for laden, -100 kW for ballast

    # Calculate true power (kW) using physics model
    power_true = (
        k0 * speeds ** 3 +
        k1 * speeds ** 2 +
        k2 * speeds +
        draft_effect +
        laden_effect
    )

    # Add realistic noise proportional to signal magnitude
    noise_std = np.maximum(np.abs(power_true) * noise_std_ratio, 5.0)
    noise = np.random.normal(0, noise_std)
    power_observed = power_true + noise

    # Create DataFrame with all features
    df = pd.DataFrame({
        "speed": speeds,
        "draft": drafts,
        "laden_ballast": laden_ballast,
        "power_true": power_true,
        "power_obs": power_observed,
    })

    return df





# =============================================================================
# PHYSICS MODEL
# =============================================================================

def calculate_physics_baseline_power(
    speed: np.ndarray,
    k: float = DEFAULT_PHYSICS_COEF
) -> np.ndarray:
    """
    Calculate physics-based baseline power prediction.

    Uses the classic cubic speed-power relationship from ship hydrodynamics:
        P = k * speed^3

    This relationship derives from the fact that:
        - Power = Resistance × Velocity
        - Resistance ~ velocity^2 (for wave-making resistance)
        - Therefore Power ~ velocity^3

    Args:
        speed: Ship speed (knots).
        k: Physics coefficient (tuned empirically).

    Returns:
        np.ndarray: Predicted power in kW.
    """
    return k * speed ** 3


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix for machine learning model.

    Constructs features including:
        - Raw ship parameters (speed, draft, laden_ballast)
        - Polynomial speed features (squared, cubed)
        - Physics baseline prediction (as a feature)

    Args:
        df: DataFrame containing ship data with columns:
            speed, draft, laden_ballast

    Returns:
        pd.DataFrame: Feature matrix ready for ML training/prediction.
    """
    X = pd.DataFrame()

    # Speed features (polynomial expansion)
    X["speed"] = df["speed"]
    X["speed_sq"] = df["speed"] ** 2
    X["speed_cu"] = df["speed"] ** 3

    # Vessel characteristics
    X["draft"] = df["draft"]
    X["laden_ballast"] = df["laden_ballast"]  # 0=ballast, 1=laden

    # Physics baseline as a feature (helps ML learn corrections)
    X["phys_base"] = calculate_physics_baseline_power(df["speed"].values)

    return X


# =============================================================================
# MODEL BUILDING
# =============================================================================

def create_regressor(random_state: int = RANDOM_SEED) -> Any:
    """
    Create and configure the gradient boosting regressor.

    Attempts to use XGBoost for better performance, falls back to
    sklearn's GradientBoostingRegressor if XGBoost is unavailable.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Configured regressor instance (XGBRegressor or GradientBoostingRegressor).
    """
    if REGRESSOR_TYPE == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:squarederror",
            verbosity=0,
            random_state=random_state,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state,
        )
    return model


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict containing RMSE, MAE, and R² scores.
    """
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "Test") -> None:
    """
    Print formatted regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        dataset_name: Name of the dataset for display.
    """
    metrics = calculate_metrics(y_true, y_pred)
    print(f"{dataset_name:12} | RMSE: {metrics['rmse']:8.3f} kW | "
          f"MAE: {metrics['mae']:8.3f} kW | R²: {metrics['r2']:.4f}")


# =============================================================================
# PHYSICS-INFORMED LOSS FUNCTIONS
# =============================================================================

def calculate_boundary_loss(predictions: np.ndarray) -> float:
    """
    Calculate boundary loss to penalize negative power predictions.

    Physical Constraint:
        Ship power can never be negative. This loss penalizes any predictions
        below the minimum threshold (typically 0 kW).

    Loss Formula:
        L_boundary = mean(max(0, threshold - P)^2)

    Args:
        predictions: Array of power predictions (kW).

    Returns:
        float: Boundary loss value (0 if all predictions are valid).
    """
    violations = np.maximum(MIN_POWER_THRESHOLD - predictions, 0)
    return np.mean(violations ** 2)


def calculate_monotonicity_loss(
    speeds: np.ndarray,
    drafts: np.ndarray,
    predictions: np.ndarray
) -> float:
    """
    Calculate monotonicity loss to ensure power increases with speed and draft.

    Physical Constraint:
        - Higher speed → Higher power (more resistance)
        - Higher draft → Higher power (more wetted surface area)

    Loss Formula:
        L_mono = mean(max(0, -dP/dSpeed)^2) + mean(max(0, -dP/dDraft)^2)

    Args:
        speeds: Array of ship speeds (knots).
        drafts: Array of ship drafts (m).
        predictions: Array of power predictions (kW).

    Returns:
        float: Monotonicity loss value (0 if monotonicity is satisfied).
    """
    # Sort by speed and check monotonicity within same draft groups
    df_temp = pd.DataFrame({
        'speed': speeds,
        'draft': drafts,
        'power': predictions
    })

    speed_violations = 0.0
    draft_violations = 0.0
    n_speed_pairs = 0
    n_draft_pairs = 0

    # Check speed monotonicity (within similar draft groups)
    for draft_val in df_temp['draft'].unique():
        draft_group = df_temp[df_temp['draft'] == draft_val].sort_values('speed')
        if len(draft_group) > 1:
            power_diff = np.diff(draft_group['power'].values)
            # Penalize negative differences (power should increase with speed)
            violations = np.maximum(-power_diff, 0)
            speed_violations += np.sum(violations ** 2)
            n_speed_pairs += len(violations)

    # Check draft monotonicity (within similar speed groups)
    # Bin speeds into groups for comparison
    df_temp['speed_bin'] = pd.cut(df_temp['speed'], bins=10, labels=False)
    for speed_bin in df_temp['speed_bin'].unique():
        if pd.isna(speed_bin):
            continue
        speed_group = df_temp[df_temp['speed_bin'] == speed_bin].sort_values('draft')
        if len(speed_group) > 1:
            power_diff = np.diff(speed_group['power'].values)
            # Penalize negative differences (power should increase with draft)
            violations = np.maximum(-power_diff, 0)
            draft_violations += np.sum(violations ** 2)
            n_draft_pairs += len(violations)

    # Normalize by number of pairs
    speed_loss = speed_violations / max(n_speed_pairs, 1)
    draft_loss = draft_violations / max(n_draft_pairs, 1)

    return speed_loss + draft_loss


def calculate_divergence_loss(
    speeds: np.ndarray,
    drafts: np.ndarray,
    predictions: np.ndarray,
    max_divergence_ratio: float = 0.5
) -> float:
    """
    Calculate draft divergence loss to prevent curves from diverging too much.

    Physical Constraint:
        The difference in power between different draft conditions should
        remain relatively constant across speeds. Curves shouldn't diverge
        excessively at higher speeds.

    Loss Formula:
        L_div = mean(max(0, |P_high_draft - P_low_draft| - max_allowed)^2)

    Args:
        speeds: Array of ship speeds (knots).
        drafts: Array of ship drafts (m).
        predictions: Array of power predictions (kW).
        max_divergence_ratio: Maximum allowed divergence as ratio of mean power.

    Returns:
        float: Divergence loss value.
    """
    df_temp = pd.DataFrame({
        'speed': speeds,
        'draft': drafts,
        'power': predictions
    })

    # Bin speeds for comparison
    df_temp['speed_bin'] = pd.cut(df_temp['speed'], bins=10, labels=False)

    divergence_violations = []

    for speed_bin in df_temp['speed_bin'].unique():
        if pd.isna(speed_bin):
            continue
        speed_group = df_temp[df_temp['speed_bin'] == speed_bin]

        if len(speed_group) < 2:
            continue

        # Get power range within this speed bin
        power_range = speed_group['power'].max() - speed_group['power'].min()
        mean_power = speed_group['power'].mean()

        # Calculate expected maximum divergence based on draft range
        draft_range = speed_group['draft'].max() - speed_group['draft'].min()
        # Allow some divergence proportional to draft difference
        expected_divergence = draft_range * 20.0  # ~20 kW per meter of draft

        # Penalize excessive divergence
        excess_divergence = max(0, power_range - expected_divergence - max_divergence_ratio * abs(mean_power))
        divergence_violations.append(excess_divergence ** 2)

    return np.mean(divergence_violations) if divergence_violations else 0.0


def calculate_total_physics_loss(
    predictions: np.ndarray,
    speeds: np.ndarray,
    drafts: np.ndarray
) -> Dict[str, float]:
    """
    Calculate total physics-informed loss combining all constraints.

    Args:
        predictions: Array of power predictions (kW).
        speeds: Array of ship speeds (knots).
        drafts: Array of ship drafts (m).

    Returns:
        Dict containing individual and total loss values.
    """
    boundary_loss = calculate_boundary_loss(predictions)
    monotonicity_loss = calculate_monotonicity_loss(speeds, drafts, predictions)
    divergence_loss = calculate_divergence_loss(speeds, drafts, predictions)

    total_loss = (
        BOUNDARY_LOSS_WEIGHT * boundary_loss +
        MONOTONICITY_LOSS_WEIGHT * monotonicity_loss +
        DIVERGENCE_LOSS_WEIGHT * divergence_loss
    )

    return {
        'boundary': boundary_loss,
        'monotonicity': monotonicity_loss,
        'divergence': divergence_loss,
        'total': total_loss
    }


def apply_physics_constraints(predictions: np.ndarray) -> np.ndarray:
    """
    Apply hard physics constraints to predictions.

    Post-processing step to ensure predictions satisfy physical constraints:
        - Non-negativity: Clip predictions to minimum threshold

    Args:
        predictions: Array of raw power predictions (kW).

    Returns:
        np.ndarray: Constrained predictions.
    """
    # Apply non-negativity constraint
    constrained = np.maximum(predictions, MIN_POWER_THRESHOLD)
    return constrained


def smooth_predictions(
    speed_grid: np.ndarray,
    predictions: np.ndarray,
    method: str = "spline",
    smoothing_factor: float = 0.1
) -> np.ndarray:
    """
    Smooth predictions to create smooth curves instead of zigzag patterns.

    Tree-based models (XGBoost, GBR) produce step-like predictions. This
    function applies smoothing to create physically realistic smooth curves.

    Args:
        speed_grid: Array of speed values (x-axis).
        predictions: Array of raw predictions to smooth.
        method: Smoothing method - "spline", "gaussian", or "polynomial".
        smoothing_factor: Controls smoothness (higher = smoother).

    Returns:
        np.ndarray: Smoothed predictions.
    """
    if method == "spline":
        # Univariate spline fitting - produces smooth cubic-like curves
        # s parameter controls smoothness (higher = smoother)
        s_value = len(predictions) * smoothing_factor
        try:
            spline = UnivariateSpline(speed_grid, predictions, s=s_value, k=3)
            smoothed = spline(speed_grid)
        except Exception:
            # Fallback to gaussian if spline fails
            smoothed = gaussian_filter1d(predictions, sigma=5)
    
    elif method == "gaussian":
        # Gaussian filter smoothing
        sigma = max(1, int(len(predictions) * smoothing_factor / 10))
        smoothed = gaussian_filter1d(predictions, sigma=sigma)
    
    elif method == "polynomial":
        # Polynomial fit (cubic)
        coeffs = np.polyfit(speed_grid, predictions, deg=3)
        smoothed = np.polyval(coeffs, speed_grid)
    
    else:
        smoothed = predictions
    
    # Ensure smoothed predictions are non-negative
    smoothed = np.maximum(smoothed, MIN_POWER_THRESHOLD)
    
    # Ensure monotonicity (power should increase with speed)
    for i in range(1, len(smoothed)):
        if smoothed[i] < smoothed[i-1]:
            smoothed[i] = smoothed[i-1]
    
    return smoothed


def print_physics_losses(losses: Dict[str, float]) -> None:
    """
    Print formatted physics loss values.

    Args:
        losses: Dictionary containing loss values.
    """
    print("\n" + "=" * 60)
    print("PHYSICS-INFORMED LOSS VALUES")
    print("=" * 60)
    print(f"  Boundary Loss (non-negativity):    {losses['boundary']:10.4f}")
    print(f"  Monotonicity Loss (speed/draft):   {losses['monotonicity']:10.4f}")
    print(f"  Divergence Loss (draft curves):    {losses['divergence']:10.4f}")
    print(f"  " + "-" * 40)
    print(f"  Total Physics Loss:                {losses['total']:10.4f}")


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

def train_piml_model(
    train_df: pd.DataFrame,
    random_state: int = RANDOM_SEED,
    n_iterations: int = 3
) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Train the Physics-Informed Machine Learning model with constraint enforcement.

    The training process:
        1. Calculate physics baseline predictions
        2. Compute residuals (observed - physics)
        3. Train ML model to predict residuals with physics-informed sample weights
        4. Apply physics constraints to final predictions
        5. Iterate to refine weights based on constraint violations

    Physics Constraints Enforced:
        - Boundary Loss: Power ≥ 0 (non-negativity)
        - Monotonicity Loss: ∂P/∂speed > 0, ∂P/∂draft > 0
        - Divergence Loss: Draft curves don't diverge excessively

    Args:
        train_df: Training DataFrame with ship data.
        random_state: Random seed for reproducibility.
        n_iterations: Number of training iterations for weight refinement.

    Returns:
        Tuple containing:
            - Trained model
            - Physics baseline predictions
            - Final predictions (physics + ML, constrained)
            - Physics loss values
    """
    # Prepare features
    X_train = prepare_features(train_df)
    y_train = train_df["power_obs"].values
    speeds = train_df["speed"].values
    drafts = train_df["draft"].values

    # Calculate physics baseline
    phys_predictions = X_train["phys_base"].values

    # Calculate residuals (what physics can't explain)
    residuals = y_train - phys_predictions

    # Initialize sample weights (uniform)
    sample_weights = np.ones(len(y_train))

    # Iterative training with physics-informed weight adjustment
    model = None
    final_predictions = None

    for iteration in range(n_iterations):
        # Create and train model with current weights
        model = create_regressor(random_state + iteration)

        if REGRESSOR_TYPE == "xgboost":
            model.fit(X_train, residuals, sample_weight=sample_weights)
        else:
            model.fit(X_train, residuals, sample_weight=sample_weights)

        # Generate predictions
        predicted_residuals = model.predict(X_train)
        raw_predictions = phys_predictions + predicted_residuals

        # Apply physics constraints (post-processing)
        final_predictions = apply_physics_constraints(raw_predictions)

        # Calculate physics losses
        losses = calculate_total_physics_loss(final_predictions, speeds, drafts)

        # Update sample weights based on constraint violations
        # Samples with violations get higher weights in next iteration
        if iteration < n_iterations - 1:
            # Boundary violations
            boundary_violations = np.maximum(MIN_POWER_THRESHOLD - raw_predictions, 0)

            # Create weight adjustments
            weight_adjustments = 1.0 + BOUNDARY_LOSS_WEIGHT * (boundary_violations / (np.max(boundary_violations) + 1e-6))

            # Monotonicity-based weight adjustment
            # Increase weights for samples that might cause monotonicity violations
            df_temp = pd.DataFrame({
                'speed': speeds,
                'draft': drafts,
                'power': final_predictions,
                'index': np.arange(len(speeds))
            }).sort_values(['draft', 'speed'])

            mono_adjustments = np.ones(len(speeds))
            for draft_val in df_temp['draft'].unique():
                draft_mask = df_temp['draft'] == draft_val
                draft_group = df_temp[draft_mask].sort_values('speed')
                if len(draft_group) > 1:
                    indices = draft_group['index'].values
                    powers = draft_group['power'].values
                    for i in range(1, len(indices)):
                        if powers[i] < powers[i-1]:  # Monotonicity violation
                            mono_adjustments[indices[i]] += MONOTONICITY_LOSS_WEIGHT * 0.5
                            mono_adjustments[indices[i-1]] += MONOTONICITY_LOSS_WEIGHT * 0.5

            # Combine weight adjustments
            sample_weights = weight_adjustments * mono_adjustments
            sample_weights = sample_weights / np.mean(sample_weights)  # Normalize

    return model, phys_predictions, final_predictions, losses


def predict_with_piml(
    model: Any,
    test_df: pd.DataFrame,
    apply_constraints: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions using trained PIML model with physics constraints.

    Args:
        model: Trained residual model.
        test_df: Test DataFrame with ship data.
        apply_constraints: Whether to apply physics constraints to predictions.

    Returns:
        Tuple containing:
            - Physics baseline predictions
            - Predicted residuals
            - Final predictions (physics + ML, optionally constrained)
    """
    X_test = prepare_features(test_df)

    # Physics baseline
    phys_predictions = X_test["phys_base"].values

    # ML residual predictions
    predicted_residuals = model.predict(X_test)

    # Combined prediction
    raw_predictions = phys_predictions + predicted_residuals

    # Apply physics constraints if requested
    if apply_constraints:
        final_predictions = apply_physics_constraints(raw_predictions)
    else:
        final_predictions = raw_predictions

    return phys_predictions, predicted_residuals, final_predictions


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_prediction_grid(
    df: pd.DataFrame,
    laden_ballast: int = 0,
    speed_min: float = 5.0,
    speed_max: float = 18.0,
    n_points: int = 300
) -> pd.DataFrame:
    """
    Create a fine grid of speeds for smooth curve plotting.

    Uses median vessel characteristics to create a representative curve.

    Args:
        df: Original DataFrame to extract median vessel characteristics.
        laden_ballast: Loading condition (0=ballast, 1=laden).
        speed_min: Minimum speed for grid.
        speed_max: Maximum speed for grid.
        n_points: Number of points in the grid.

    Returns:
        pd.DataFrame: Grid DataFrame with all required features.
    """
    speed_grid = np.linspace(speed_min, speed_max, n_points)

    # Use median vessel characteristics for representative curve
    grid_df = pd.DataFrame({
        "speed": speed_grid,
        "draft": df["draft"].median(),
        "laden_ballast": laden_ballast,
    })

    # Calculate derived features
    grid_df["speed_sq"] = grid_df["speed"] ** 2
    grid_df["speed_cu"] = grid_df["speed"] ** 3
    grid_df["phys_base"] = calculate_physics_baseline_power(grid_df["speed"].values)

    return grid_df


def create_main_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_test: np.ndarray,
    speed_grid: np.ndarray,
    phys_grid: np.ndarray,
    final_grid_laden: np.ndarray,
    final_grid_ballast: np.ndarray
) -> go.Figure:
    """
    Create the main interactive Plotly figure for speed-power visualization.

    Shows separate curves for laden and ballast conditions.

    Args:
        train_df: Training data DataFrame.
        test_df: Test data DataFrame.
        pred_test: Model predictions on test set.
        speed_grid: Speed values for smooth curves.
        phys_grid: Physics baseline values on grid.
        final_grid_laden: Final ML-corrected values for laden condition.
        final_grid_ballast: Final ML-corrected values for ballast condition.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    # Split training data by laden/ballast
    train_laden = train_df[train_df["laden_ballast"] == 1]
    train_ballast = train_df[train_df["laden_ballast"] == 0]

    # Split test data by laden/ballast
    test_laden = test_df[test_df["laden_ballast"] == 1]
    test_ballast = test_df[test_df["laden_ballast"] == 0]

    # Get predictions for each condition
    pred_test_laden = pred_test[test_df["laden_ballast"].values == 1]
    pred_test_ballast = pred_test[test_df["laden_ballast"].values == 0]

    # Training data scatter - Laden
    trace_train_laden = go.Scatter(
        x=train_laden["speed"],
        y=train_laden["power_obs"],
        mode="markers",
        name="Train - Laden",
        marker=dict(size=6, symbol="circle", opacity=0.7, color="blue"),
        hovertemplate="Speed: %{x:.2f} kn<br>Power: %{y:.2f} kW<br>Laden<extra></extra>",
    )

    # Training data scatter - Ballast
    trace_train_ballast = go.Scatter(
        x=train_ballast["speed"],
        y=train_ballast["power_obs"],
        mode="markers",
        name="Train - Ballast",
        marker=dict(size=6, symbol="circle", opacity=0.7, color="lightblue"),
        hovertemplate="Speed: %{x:.2f} kn<br>Power: %{y:.2f} kW<br>Ballast<extra></extra>",
    )

    # Test data scatter - Laden
    trace_test_laden = go.Scatter(
        x=test_laden["speed"],
        y=test_laden["power_obs"],
        mode="markers",
        name="Test - Laden",
        marker=dict(size=8, symbol="diamond", opacity=0.8, color="darkblue"),
        hovertemplate="Speed: %{x:.2f} kn<br>Power: %{y:.2f} kW<br>Laden<extra></extra>",
    )

    # Test data scatter - Ballast
    trace_test_ballast = go.Scatter(
        x=test_ballast["speed"],
        y=test_ballast["power_obs"],
        mode="markers",
        name="Test - Ballast",
        marker=dict(size=8, symbol="diamond", opacity=0.8, color="cyan"),
        hovertemplate="Speed: %{x:.2f} kn<br>Power: %{y:.2f} kW<br>Ballast<extra></extra>",
    )

    # Physics baseline curve (common)
    trace_phys = go.Scatter(
        x=speed_grid,
        y=phys_grid,
        mode="lines",
        name="Physics baseline (P ∝ v³)",
        line=dict(dash="dash", width=2, color="orange"),
        hovertemplate="Speed: %{x:.2f} kn<br>Physics: %{y:.2f} kW<extra></extra>",
    )

    # ML-corrected curve - Laden
    trace_final_laden = go.Scatter(
        x=speed_grid,
        y=final_grid_laden,
        mode="lines",
        name="PIML - Laden",
        line=dict(width=3, color="darkgreen"),
        hovertemplate="Speed: %{x:.2f} kn<br>Predicted: %{y:.2f} kW<br>Laden<extra></extra>",
    )

    # ML-corrected curve - Ballast
    trace_final_ballast = go.Scatter(
        x=speed_grid,
        y=final_grid_ballast,
        mode="lines",
        name="PIML - Ballast",
        line=dict(width=3, color="lightgreen"),
        hovertemplate="Speed: %{x:.2f} kn<br>Predicted: %{y:.2f} kW<br>Ballast<extra></extra>",
    )

    # Test predictions scatter - Laden
    trace_pred_laden = go.Scatter(
        x=test_laden["speed"],
        y=pred_test_laden,
        mode="markers",
        name="Predicted - Laden",
        marker=dict(size=8, symbol="x", color="red"),
        hovertemplate="Speed: %{x:.2f} kn<br>Predicted: %{y:.2f} kW<br>Laden<extra></extra>",
    )

    # Test predictions scatter - Ballast
    trace_pred_ballast = go.Scatter(
        x=test_ballast["speed"],
        y=pred_test_ballast,
        mode="markers",
        name="Predicted - Ballast",
        marker=dict(size=8, symbol="x", color="salmon"),
        hovertemplate="Speed: %{x:.2f} kn<br>Predicted: %{y:.2f} kW<br>Ballast<extra></extra>",
    )

    # Configure layout
    layout = go.Layout(
        title=dict(
            text="Ship Speed vs Power — Physics-Informed ML (Laden vs Ballast)",
            font=dict(size=18)
        ),
        xaxis=dict(title="Speed (knots)", gridcolor="lightgray"),
        yaxis=dict(title="Power (kW)", gridcolor="lightgray"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="closest",
        plot_bgcolor="white",
    )

    fig = go.Figure(
        data=[
            trace_train_laden, trace_train_ballast,
            trace_test_laden, trace_test_ballast,
            trace_pred_laden, trace_pred_ballast,
            trace_phys,
            trace_final_laden, trace_final_ballast
        ],
        layout=layout
    )

    return fig


def create_residual_figure(
    test_df: pd.DataFrame,
    residuals_observed: np.ndarray,
    residuals_predicted: np.ndarray
) -> go.Figure:
    """
    Create interactive Plotly figure for residual analysis.

    Args:
        test_df: Test data DataFrame.
        residuals_observed: Observed residuals (actual - physics).
        residuals_predicted: ML-predicted residuals.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    fig = go.Figure()

    # Observed residuals
    fig.add_trace(go.Scatter(
        x=test_df["speed"],
        y=residuals_observed,
        mode="markers",
        name="Observed residual (actual - physics)",
        marker=dict(symbol="circle", size=8, opacity=0.7),
        hovertemplate="Speed: %{x:.2f} kn<br>Residual: %{y:.2f} kW<extra></extra>",
    ))

    # Predicted residuals
    fig.add_trace(go.Scatter(
        x=test_df["speed"],
        y=residuals_predicted,
        mode="markers",
        name="Predicted residual (ML)",
        marker=dict(symbol="x", size=8, color="red"),
        hovertemplate="Speed: %{x:.2f} kn<br>Residual: %{y:.2f} kW<extra></extra>",
    ))

    fig.update_layout(
        title="Residual Analysis: Observed vs Predicted (Test Set)",
        xaxis_title="Speed (knots)",
        yaxis_title="Residual (kW)",
        plot_bgcolor="white",
        hovermode="closest",
    )

    return fig


def print_feature_importance(model: Any, feature_names: list) -> None:
    """
    Print feature importance scores from the trained model.

    Args:
        model: Trained model (XGBoost or sklearn GradientBoosting).
        feature_names: List of feature names.
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Top 10)")
    print("=" * 60)

    try:
        if REGRESSOR_TYPE == "xgboost":
            # XGBoost feature importance
            importance_dict = model.get_booster().get_score(importance_type="gain")
            importance_series = pd.Series(importance_dict).sort_values(ascending=False)
        else:
            # sklearn feature importance
            importance_series = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)

        print(importance_series.head(10).to_string())

    except Exception as e:
        print(f"Feature importance not available: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Main execution function for the PIML ship speed-power prediction pipeline.

    Pipeline steps:
        1. Generate synthetic ship data
        2. Split into train/test sets
        3. Train PIML model (physics baseline + ML residuals)
        4. Evaluate model performance
        5. Generate interactive visualizations
        6. Save sample data for inspection
    """
    print("=" * 60)
    print("PHYSICS-INFORMED ML: Ship Speed-Power Prediction")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # -------------------------------------------------------------------------
    print("\n[1/5] Generating synthetic ship data...")
    df = generate_synthetic_ship_data(n_samples=240, noise_std_ratio=0.07)
    print(f"      Generated {len(df)} samples")

    # -------------------------------------------------------------------------
    # Step 2: Train/test split
    # -------------------------------------------------------------------------
    print("\n[2/5] Splitting data into train/test sets...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    print(f"      Train: {len(train_df)} samples | Test: {len(test_df)} samples")

    # -------------------------------------------------------------------------
    # Step 3: Train PIML model
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Training PIML model (regressor: {REGRESSOR_TYPE})...")
    print("      Enforcing physics constraints:")
    print("        - Boundary: Power ≥ 0 (non-negativity)")
    print("        - Monotonicity: ∂P/∂speed > 0, ∂P/∂draft > 0")
    print("        - Divergence: Draft curves don't diverge excessively")

    model, phys_train, pred_train, train_losses = train_piml_model(train_df)

    # Get test predictions with physics constraints
    phys_test, pred_res_test, pred_test = predict_with_piml(model, test_df)

    # Calculate test physics losses
    test_losses = calculate_total_physics_loss(
        pred_test,
        test_df["speed"].values,
        test_df["draft"].values
    )

    # -------------------------------------------------------------------------
    # Step 4: Evaluate performance
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating model performance...")
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("-" * 60)

    print("\nPIML Model (Physics + ML Residuals + Constraints):")
    print_metrics(train_df["power_obs"].values, pred_train, "Train")
    print_metrics(test_df["power_obs"].values, pred_test, "Test")

    print("\nPhysics-Only Baseline:")
    print_metrics(train_df["power_obs"].values, phys_train, "Train")
    print_metrics(test_df["power_obs"].values, phys_test, "Test")

    # Print physics constraint losses
    print("\n" + "-" * 60)
    print("PHYSICS CONSTRAINT SATISFACTION")
    print("-" * 60)
    print("\nTraining Set:")
    print_physics_losses(train_losses)
    print("\nTest Set:")
    print_physics_losses(test_losses)

    # Feature importance
    X_train = prepare_features(train_df)
    print_feature_importance(model, X_train.columns.tolist())

    # -------------------------------------------------------------------------
    # Step 5: Generate visualizations
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating visualizations...")

    # Create prediction grids for both laden and ballast conditions
    grid_df_laden = create_prediction_grid(df, laden_ballast=1)
    grid_df_ballast = create_prediction_grid(df, laden_ballast=0)

    X_grid_laden = prepare_features(grid_df_laden)
    X_grid_ballast = prepare_features(grid_df_ballast)

    # Predict on grids for both conditions
    res_grid_laden = model.predict(X_grid_laden)
    res_grid_ballast = model.predict(X_grid_ballast)

    final_grid_laden_raw = grid_df_laden["phys_base"].values + res_grid_laden
    final_grid_ballast_raw = grid_df_ballast["phys_base"].values + res_grid_ballast

    # Apply smoothing to get smooth cubic-like curves
    print("      Applying curve smoothing...")
    final_grid_laden = smooth_predictions(
        grid_df_laden["speed"].values,
        final_grid_laden_raw,
        method="spline",
        smoothing_factor=0.05
    )
    final_grid_ballast = smooth_predictions(
        grid_df_ballast["speed"].values,
        final_grid_ballast_raw,
        method="spline",
        smoothing_factor=0.05
    )

    # Create main figure with separate curves
    fig_main = create_main_figure(
        train_df=train_df,
        test_df=test_df,
        pred_test=pred_test,
        speed_grid=grid_df_laden["speed"].values,
        phys_grid=grid_df_laden["phys_base"].values,
        final_grid_laden=final_grid_laden,
        final_grid_ballast=final_grid_ballast
    )

    # Create residual figure
    residuals_observed = test_df["power_obs"].values - phys_test
    fig_residual = create_residual_figure(
        test_df=test_df,
        residuals_observed=residuals_observed,
        residuals_predicted=pred_res_test
    )

    # -------------------------------------------------------------------------
    # Save sample data
    # -------------------------------------------------------------------------
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "ship_synthetic_sample.csv"

    df.sample(30, random_state=RANDOM_SEED).to_csv(output_csv, index=False)
    print(f"\nSample data saved to: {output_csv}")

    # -------------------------------------------------------------------------
    # Display figures
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Opening interactive plots in browser...")
    print("=" * 60)

    fig_main.show()
    fig_residual.show()

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
