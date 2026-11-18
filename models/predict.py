"""
PathSnap - Model 5 (MLP) Inference and Probability Prediction (Member 3)
========================================================================
Provides reusable functions to compute candidate arc probabilities for GPS observations.

Output Schema (Team Convention):
- trip_id: str
- obs_idx: int
- arc_id: int
- prob: float (calibrated probability in [0.0, 1.0])
- model_name: str ("mlp")

Note on HMM Integration:
These probabilities are pointwise binary classification scores P(arc_i | features).
In the upcoming Phase 6 HMM/Viterbi route decoder, emission probabilities B_i(z_t)
can be directly derived from these scores, optionally normalized over each observation's
candidate set such that sum_i P(a_i | z_t) = 1.0.
"""

from typing import Union, List, Optional
import os
import numpy as np
import pandas as pd
import torch
import joblib

from .mlp_model import BinaryMLPClassifier
from .candidate_adapter import METADATA_COLUMNS, get_feature_columns


def predict_candidate_probabilities(
    model: BinaryMLPClassifier,
    preprocessor: dict,
    candidates_df: pd.DataFrame,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 1024
) -> pd.DataFrame:
    """
    Computes candidate probabilities using the trained MLP model and preprocessor.

    Args:
        model: Trained BinaryMLPClassifier instance.
        preprocessor: Dict containing "scaler", "imputer", and "feature_columns".
        candidates_df: DataFrame with candidate features.
        device: Torch device ("cpu", "mps", or torch.device).
        batch_size: Batch size for model inference.

    Returns:
        DataFrame with columns: ['trip_id', 'obs_idx', 'arc_id', 'prob', 'model_name']
    """
    if len(candidates_df) == 0:
        return pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id", "prob", "model_name"])

    # Determine device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()

    # Extract required feature columns
    feature_cols = preprocessor["feature_columns"]
    missing_cols = [c for c in feature_cols if c not in candidates_df.columns]
    if missing_cols:
        raise ValueError(f"candidates_df is missing expected feature columns: {missing_cols}")

    X_raw = candidates_df[feature_cols].values

    # Preprocessing (imputation & scaling fitted on training split)
    imputer = preprocessor.get("imputer")
    scaler = preprocessor.get("scaler")

    if imputer is not None:
        X_imp = imputer.transform(X_raw)
    else:
        X_imp = X_raw

    if scaler is not None:
        X_scaled = scaler.transform(X_imp)
    else:
        X_scaled = X_imp

    # Batch model evaluation
    all_probs = []
    n_samples = len(X_scaled)

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = torch.tensor(X_scaled[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_probs = model.predict_proba(batch_x).cpu().numpy().flatten()
            all_probs.extend(batch_probs)

    all_probs = np.asarray(all_probs, dtype=np.float64)

    # Sanity checks on probabilities
    all_probs = np.clip(all_probs, 0.0, 1.0)
    if np.isnan(all_probs).any():
        raise ValueError("Model produced NaN probabilities during inference.")

    # Determine arc ID column name
    arc_col = "candidate_arc_id" if "candidate_arc_id" in candidates_df.columns else "arc_id"
    obs_col = "obs_idx" if "obs_idx" in candidates_df.columns else "observation_index"

    res_df = pd.DataFrame({
        "trip_id": candidates_df["trip_id"].astype(str).values,
        "obs_idx": candidates_df[obs_col].astype(int).values,
        "arc_id": candidates_df[arc_col].astype(int).values,
        "prob": all_probs,
        "model_name": "mlp"
    })

    return res_df


def load_model_and_preprocessor(artifact_dir: str, device: str = "cpu"):
    """
    Loads saved Model 5 artifacts from artifact_dir:
    - best_model.pt
    - preprocessing.joblib
    - training_config.json
    """
    model_path = os.path.join(artifact_dir, "best_model.pt")
    prep_path = os.path.join(artifact_dir, "preprocessing.joblib")
    config_path = os.path.join(artifact_dir, "training_config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(prep_path):
        raise FileNotFoundError(f"Preprocessing artifact not found: {prep_path}")

    preprocessor = joblib.load(prep_path)

    # Initialize model using saved config or preprocessor feature count
    input_dim = len(preprocessor["feature_columns"])
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct architecture from checkpoint if saved with config
    model = BinaryMLPClassifier(
        input_dim=input_dim,
        hidden_dims=checkpoint.get("hidden_dims", [128, 64, 32]),
        dropout_rate=checkpoint.get("dropout_rate", 0.2)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, preprocessor
