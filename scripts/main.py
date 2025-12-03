#!/usr/bin/env python3
"""
Ship Speed-Power Prediction using Physics-Informed Machine Learning (PIML).

This module implements a hybrid approach combining:
    1. Physics-based baseline model (cubic power-speed relationship)
    2. Machine learning residual correction (XGBoost/GradientBoosting)

The approach leverages domain knowledge (ship hydrodynamics) while using ML
to capture complex patterns that physics alone cannot model.

Key Features:
    - Synthetic ship speed-power data generation with realistic noise
    - Physics baseline: P = k * speed^3 (simplified cubic relationship)
    - Residual learning with gradient boosting
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
        - Variable vessel characteristics (speed, draft)
        - Physics-based power calculation with noise

    Physics Model:
        P_true = k0 * speed^3 + k1 * speed^2 + k2 * speed + draft_effect + noise

    Args:
        n_samples: Number of data points to generate.
        speed_min: Minimum ship speed in knots.
        speed_max: Maximum ship speed in knots.
        noise_std_ratio: Noise standard deviation as ratio of signal magnitude.

    Returns:
        pd.DataFrame: DataFrame containing:
            - speed: Ship speed (knots)
            - draft: Ship draft (m) - integer values 6-14
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

    # Physics model coefficients (ground truth)
    k0 = 0.8       # Cubic term coefficient (main resistance)
    k1 = -0.5      # Quadratic term coefficient
    k2 = 2.0       # Linear term coefficient

    # Draft effect on power (deeper draft = more resistance)
    draft_effect = (drafts - 10.0) * 15.0  # Centered around draft=10

    # Calculate true power (kW) using physics model
    power_true = (
        k0 * speeds ** 3 +
        k1 * speeds ** 2 +
        k2 * speeds +
        draft_effect
    )

    # Add realistic noise proportional to signal magnitude
    noise_std = np.maximum(np.abs(power_true) * noise_std_ratio, 5.0)
    noise = np.random.normal(0, noise_std)
    power_observed = power_true + noise

    # Create DataFrame with all features
    df = pd.DataFrame({
        "speed": speeds,
        "draft": drafts,
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
        - Raw ship parameters (speed, draft)
        - Polynomial speed features (squared, cubed)
        - Physics baseline prediction (as a feature)

    Args:
        df: DataFrame containing ship data with columns:
            speed, draft

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
# TRAINING PIPELINE
# =============================================================================

def train_piml_model(
    train_df: pd.DataFrame,
    random_state: int = RANDOM_SEED
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train the Physics-Informed Machine Learning model.

    The training process:
        1. Calculate physics baseline predictions
        2. Compute residuals (observed - physics)
        3. Train ML model to predict residuals
        4. Final prediction = physics + ML residual

    Args:
        train_df: Training DataFrame with ship data.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing:
            - Trained model
            - Physics baseline predictions
            - Final predictions (physics + ML)
    """
    # Prepare features
    X_train = prepare_features(train_df)
    y_train = train_df["power_obs"].values

    # Calculate physics baseline
    phys_predictions = X_train["phys_base"].values

    # Calculate residuals (what physics can't explain)
    residuals = y_train - phys_predictions

    # Train ML model on residuals
    model = create_regressor(random_state)
    model.fit(X_train, residuals)

    # Generate final predictions
    predicted_residuals = model.predict(X_train)
    final_predictions = phys_predictions + predicted_residuals

    return model, phys_predictions, final_predictions


def predict_with_piml(
    model: Any,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions using trained PIML model.

    Args:
        model: Trained residual model.
        test_df: Test DataFrame with ship data.

    Returns:
        Tuple containing:
            - Physics baseline predictions
            - Predicted residuals
            - Final predictions (physics + ML)
    """
    X_test = prepare_features(test_df)

    # Physics baseline
    phys_predictions = X_test["phys_base"].values

    # ML residual predictions
    predicted_residuals = model.predict(X_test)

    # Combined prediction
    final_predictions = phys_predictions + predicted_residuals

    return phys_predictions, predicted_residuals, final_predictions


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_prediction_grid(
    df: pd.DataFrame,
    speed_min: float = 5.0,
    speed_max: float = 18.0,
    n_points: int = 300
) -> pd.DataFrame:
    """
    Create a fine grid of speeds for smooth curve plotting.

    Uses median vessel characteristics to create a representative curve.

    Args:
        df: Original DataFrame to extract median vessel characteristics.
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
    final_grid: np.ndarray
) -> go.Figure:
    """
    Create the main interactive Plotly figure for speed-power visualization.

    Args:
        train_df: Training data DataFrame.
        test_df: Test data DataFrame.
        pred_test: Model predictions on test set.
        speed_grid: Speed values for smooth curves.
        phys_grid: Physics baseline values on grid.
        final_grid: Final ML-corrected values on grid.

    Returns:
        go.Figure: Configured Plotly figure.
    """
    # Training data scatter
    trace_train = go.Scatter(
        x=train_df["speed"],
        y=train_df["power_obs"],
        mode="markers",
        name="Train (observed)",
        marker=dict(size=6, symbol="circle", opacity=0.7),
        hovertemplate="Speed: %{x:.2f} kn<br>Power: %{y:.2f} kW<extra></extra>",
    )

    # Test data scatter
    trace_test = go.Scatter(
        x=test_df["speed"],
        y=test_df["power_obs"],
        mode="markers",
        name="Test (observed)",
        marker=dict(size=8, symbol="diamond", opacity=0.8),
        hovertemplate="Speed: %{x:.2f} kn<br>Power: %{y:.2f} kW<extra></extra>",
    )

    # Physics baseline curve
    trace_phys = go.Scatter(
        x=speed_grid,
        y=phys_grid,
        mode="lines",
        name="Physics baseline (P ∝ v³)",
        line=dict(dash="dash", width=2, color="orange"),
        hovertemplate="Speed: %{x:.2f} kn<br>Physics: %{y:.2f} kW<extra></extra>",
    )

    # ML-corrected final curve
    trace_final = go.Scatter(
        x=speed_grid,
        y=final_grid,
        mode="lines",
        name="PIML prediction",
        line=dict(width=3, color="green"),
        hovertemplate="Speed: %{x:.2f} kn<br>Predicted: %{y:.2f} kW<extra></extra>",
    )

    # Test predictions scatter
    trace_test_pred = go.Scatter(
        x=test_df["speed"],
        y=pred_test,
        mode="markers",
        name="Test (predicted)",
        marker=dict(size=8, symbol="x", color="red"),
        hovertemplate="Speed: %{x:.2f} kn<br>Predicted: %{y:.2f} kW<extra></extra>",
    )

    # Configure layout
    layout = go.Layout(
        title=dict(
            text="Ship Speed vs Power — Physics-Informed Machine Learning",
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
        data=[trace_train, trace_test, trace_test_pred, trace_phys, trace_final],
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
    model, phys_train, pred_train = train_piml_model(train_df)

    # Get test predictions
    phys_test, pred_res_test, pred_test = predict_with_piml(model, test_df)

    # -------------------------------------------------------------------------
    # Step 4: Evaluate performance
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating model performance...")
    print("\n" + "-" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("-" * 60)

    print("\nPIML Model (Physics + ML Residuals):")
    print_metrics(train_df["power_obs"].values, pred_train, "Train")
    print_metrics(test_df["power_obs"].values, pred_test, "Test")

    print("\nPhysics-Only Baseline:")
    print_metrics(train_df["power_obs"].values, phys_train, "Train")
    print_metrics(test_df["power_obs"].values, phys_test, "Test")

    # Feature importance
    X_train = prepare_features(train_df)
    print_feature_importance(model, X_train.columns.tolist())

    # -------------------------------------------------------------------------
    # Step 5: Generate visualizations
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating visualizations...")

    # Create prediction grid for smooth curves
    grid_df = create_prediction_grid(df)
    X_grid = prepare_features(grid_df)

    # Predict on grid
    res_grid = model.predict(X_grid)
    final_grid = grid_df["phys_base"].values + res_grid

    # Create main figure
    fig_main = create_main_figure(
        train_df=train_df,
        test_df=test_df,
        pred_test=pred_test,
        speed_grid=grid_df["speed"].values,
        phys_grid=grid_df["phys_base"].values,
        final_grid=final_grid
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
