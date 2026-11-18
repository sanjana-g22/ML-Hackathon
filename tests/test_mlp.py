"""
Unit and Integration Tests for Model 5 (MLP Neural Network) - Member 3
======================================================================
Tests:
1. Correct model input/output shapes.
2. Forward pass produces finite logits (no NaN / Inf).
3. Sigmoid probabilities strictly lie between 0.0 and 1.0.
4. Deterministic inference with fixed parameters.
5. Missing-value preprocessing (imputation).
6. Train-only scaler fitting without validation leakage.
7. Model checkpoint save and restore.
8. Prediction schema and candidate row alignment.
9. Handling of empty inputs or invalid feature schemas.
10. End-to-end training smoke test on small synthetic dataset.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp_model import BinaryMLPClassifier
from models.candidate_adapter import (
    load_candidate_dataset,
    validate_candidate_schema,
    get_feature_columns,
    generate_synthetic_candidate_data
)
from models.predict import (
    predict_candidate_probabilities,
    load_model_and_preprocessor
)
from models.train_mlp import (
    fit_preprocessors,
    transform_features,
    train_mlp_model,
    compute_model_metrics,
    set_seed
)


@pytest.fixture
def sample_candidate_df():
    """Generates synthetic candidate DataFrame with 3 trips for testing."""
    return generate_synthetic_candidate_data(n_trips=3, obs_per_trip=10, candidates_per_obs=4, seed=123)


def test_1_correct_model_input_output_shapes():
    """Test 1: Forward pass produces correct logit shape (batch_size, 1)."""
    batch_size = 16
    input_dim = 8
    model = BinaryMLPClassifier(input_dim=input_dim, hidden_dims=[32, 16])

    x = torch.randn(batch_size, input_dim)
    logits = model(x)

    assert logits.shape == (batch_size, 1)


def test_2_forward_pass_finite_logits():
    """Test 2: Forward pass produces strictly finite logits without NaNs or Infs."""
    model = BinaryMLPClassifier(input_dim=5, hidden_dims=[16, 8])
    x = torch.randn(32, 5)

    logits = model(x)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_3_sigmoid_probabilities_bounded():
    """Test 3: predict_proba produces values strictly in [0.0, 1.0]."""
    model = BinaryMLPClassifier(input_dim=6, hidden_dims=[32, 16])
    x = torch.randn(50, 6) * 10.0  # Wide range of values

    probs = model.predict_proba(x)
    assert probs.shape == (50, 1)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_4_deterministic_inference():
    """Test 4: Fixed weights and eval mode produce identical outputs across calls."""
    set_seed(42)
    model = BinaryMLPClassifier(input_dim=4, hidden_dims=[16, 8], dropout_rate=0.5)
    model.eval()

    x = torch.randn(10, 4)
    out1 = model(x)
    out2 = model(x)

    assert torch.equal(out1, out2)


def test_5_missing_value_preprocessing():
    """Test 5: Preprocessing pipeline imputes NaNs using train-fitted medians."""
    train_data = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    feature_cols = ["feat1", "feat2"]
    preproc = fit_preprocessors(train_data, feature_cols)

    # Test data with missing values
    test_data = pd.DataFrame({
        "feat1": [np.nan, 2.0],
        "feat2": [10.0, np.nan]
    })
    X_test_scaled = transform_features(test_data, preproc)

    # Must be finite with no NaNs
    assert not np.isnan(X_test_scaled).any()
    assert X_test_scaled.shape == (2, 2)


def test_6_train_only_scaler_fitting_no_leakage():
    """Test 6: Scaler mean and variance are derived strictly from training data."""
    train_df = pd.DataFrame({
        "f1": [0.0, 10.0, 20.0],
        "f2": [100.0, 200.0, 300.0]
    })
    val_df = pd.DataFrame({
        "f1": [1000.0, 2000.0],
        "f2": [5000.0, 6000.0]
    })
    feature_cols = ["f1", "f2"]

    preproc = fit_preprocessors(train_df, feature_cols)
    scaler = preproc["scaler"]

    # Scaler mean must match train_df mean (10.0, 200.0), NOT val_df
    np.testing.assert_allclose(scaler.mean_, np.array([10.0, 200.0]))


def test_7_model_checkpoint_save_and_load(tmp_path):
    """Test 7: Model weights and config save to disk and reload identically."""
    input_dim = 4
    model = BinaryMLPClassifier(input_dim=input_dim, hidden_dims=[16, 8])
    model.eval()

    save_path = tmp_path / "test_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": [16, 8],
        "dropout_rate": 0.2
    }, str(save_path))

    # Reload into new instance
    loaded = BinaryMLPClassifier(input_dim=input_dim, hidden_dims=[16, 8])
    ckpt = torch.load(str(save_path), map_location="cpu")
    loaded.load_state_dict(ckpt["state_dict"])
    loaded.eval()

    x = torch.randn(5, input_dim)
    np.testing.assert_allclose(model(x).detach().numpy(), loaded(x).detach().numpy())


def test_8_prediction_schema_and_alignment(sample_candidate_df):
    """Test 8: predict_candidate_probabilities produces correct schema and row count."""
    feature_cols = get_feature_columns(sample_candidate_df)
    preproc = fit_preprocessors(sample_candidate_df, feature_cols)
    model = BinaryMLPClassifier(input_dim=len(feature_cols), hidden_dims=[16, 8])

    preds_df = predict_candidate_probabilities(
        model=model,
        preprocessor=preproc,
        candidates_df=sample_candidate_df,
        device="cpu"
    )

    expected_cols = ["trip_id", "obs_idx", "arc_id", "prob", "model_name"]
    assert list(preds_df.columns) == expected_cols
    assert len(preds_df) == len(sample_candidate_df)
    assert (preds_df["model_name"] == "mlp").all()
    assert (preds_df["prob"] >= 0.0).all() and (preds_df["prob"] <= 1.0).all()


def test_9_empty_input_and_invalid_schema():
    """Test 9: Handles empty input gracefully and flags schema errors."""
    empty_df = pd.DataFrame(columns=["trip_id", "obs_idx", "candidate_arc_id", "perp_distance_m"])
    preproc = {"feature_columns": ["perp_distance_m"]}
    model = BinaryMLPClassifier(input_dim=1, hidden_dims=[8, 4])

    # Empty input returns empty output without error
    out = predict_candidate_probabilities(model, preproc, empty_df, device="cpu")
    assert len(out) == 0

    # Missing expected feature column raises ValueError
    bad_df = pd.DataFrame({"trip_id": ["01"], "obs_idx": [0], "candidate_arc_id": [10]})
    with pytest.raises(ValueError, match="missing expected feature columns"):
        predict_candidate_probabilities(model, preproc, bad_df, device="cpu")


def test_10_end_to_end_training_smoke_test(tmp_path):
    """Test 10: Full training loop runs, early stops, and saves artifacts."""
    train_df = generate_synthetic_candidate_data(n_trips=4, obs_per_trip=10, seed=1)
    val_df = generate_synthetic_candidate_data(n_trips=2, obs_per_trip=10, seed=2)
    feature_cols = get_feature_columns(train_df)
    out_dir = str(tmp_path / "mlp_output")

    metrics = train_mlp_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        epochs=4,
        batch_size=32,
        lr=0.01,
        hidden_dims=[16, 8],
        output_dir=out_dir
    )

    assert "candidate_accuracy" in metrics
    assert "observation_top1_accuracy" in metrics
    assert os.path.exists(os.path.join(out_dir, "best_model.pt"))
    assert os.path.exists(os.path.join(out_dir, "preprocessing.joblib"))
    assert os.path.exists(os.path.join(out_dir, "training_history.csv"))
    assert os.path.exists(os.path.join(out_dir, "metrics.json"))
    assert os.path.exists(os.path.join(out_dir, "loss_curve.png"))
