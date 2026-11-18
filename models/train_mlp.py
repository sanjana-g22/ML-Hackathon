"""
PathSnap - Model 5 (MLP) Training & Evaluation Pipeline (Member 3)
==================================================================
Trains a Multi-Layer Perceptron binary classifier for candidate road segment scoring.

Features & Capabilities:
- Group-aware cross-validation / splitting by trip_id (prevents intra-trip leakage).
- Missing value imputation & StandardScaler fitted strictly on training split.
- Class-imbalance compensation via pos_weight in BCEWithLogitsLoss.
- Early stopping with best validation loss checkpoint restoration.
- Comprehensive metric suite: Binary Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
  Log Loss, Confusion Matrix, and Observation-level Top-1 Selection Accuracy.
- Generates publication-quality loss, accuracy, and confusion matrix plots.
- Exports all model artifacts to results/model5_mlp/.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import sys
import json
import argparse
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .mlp_model import BinaryMLPClassifier
from .candidate_adapter import (
    load_candidate_dataset,
    validate_candidate_schema,
    get_feature_columns,
    generate_synthetic_candidate_data
)
from .predict import predict_candidate_probabilities


# ==============================================================================
# PyTorch Dataset
# ==============================================================================

class CandidateDataset(Dataset):
    """PyTorch Dataset wrapping candidate feature matrices and binary targets."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==============================================================================
# Device and Reproducibility Helpers
# ==============================================================================

def set_seed(seed: int = 42):
    """Sets random seed across all libraries for deterministic execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects best available PyTorch device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        try:
            # Test small tensor allocation on MPS to verify stability
            test_t = torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


# ==============================================================================
# Preprocessing Pipeline (Leak-Free)
# ==============================================================================

def fit_preprocessors(
    train_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Fits missing value imputer and standard scaler ONLY on training data.
    """
    X_train_raw = train_df[feature_cols].values

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_raw)
    X_train_imp = imputer.transform(X_train_raw)

    scaler = StandardScaler()
    scaler.fit(X_train_imp)

    return {
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": feature_cols
    }


def transform_features(
    df: pd.DataFrame,
    preprocessor: Dict[str, Any]
) -> np.ndarray:
    """Transforms raw features using fitted preprocessor artifacts."""
    feature_cols = preprocessor["feature_columns"]
    X_raw = df[feature_cols].values
    X_imp = preprocessor["imputer"].transform(X_raw)
    X_scaled = preprocessor["scaler"].transform(X_imp)
    return X_scaled


# ==============================================================================
# Evaluation Function
# ==============================================================================

def compute_model_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eval_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Computes candidate-level classification metrics and observation-level Top-1 accuracy.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        loss_val = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1.0 - 1e-7)))
    except Exception:
        loss_val = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Observation-level Top-1 Selection Accuracy
    # Group by (trip_id, obs_idx): does candidate with highest prob have y_true == 1?
    top1_acc = 0.0
    total_obs = 0
    correct_obs = 0

    eval_copy = eval_df.copy()
    eval_copy["_pred_prob"] = y_prob
    eval_copy["_ground_truth"] = y_true

    grouped = eval_copy.groupby(["trip_id", "obs_idx"])
    for _, group in grouped:
        best_cand = group.sort_values(by="_pred_prob", ascending=False).iloc[0]
        if best_cand["_ground_truth"] == 1:
            correct_obs += 1
        total_obs += 1

    if total_obs > 0:
        top1_acc = float(correct_obs / total_obs)

    return {
        "candidate_accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": loss_val,
        "confusion_matrix": cm,
        "observation_top1_accuracy": top1_acc,
        "total_evaluated_observations": total_obs,
        "total_evaluated_candidates": len(y_true),
        "positive_candidate_count": int(np.sum(y_true == 1)),
        "negative_candidate_count": int(np.sum(y_true == 0))
    }


# ==============================================================================
# Training Loop
# ==============================================================================

def train_mlp_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    patience: int = 7,
    seed: int = 42,
    output_dir: str = "results/model5_mlp"
) -> Dict[str, Any]:
    """
    Executes end-to-end MLP training with early stopping and checkpoint restoration.
    """
    set_seed(seed)
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    # 1. Fit Preprocessor on Train split only
    preprocessor = fit_preprocessors(train_df, feature_cols)
    X_train = transform_features(train_df, preprocessor)
    y_train = train_df["is_ground_truth"].values.astype(np.float32)

    X_val = transform_features(val_df, preprocessor)
    y_val = val_df["is_ground_truth"].values.astype(np.float32)

    # 2. Class Imbalance pos_weight
    n_pos = float(np.sum(y_train == 1.0))
    n_neg = float(np.sum(y_train == 0.0))
    pos_weight_val = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    # 3. DataLoaders
    train_dataset = CandidateDataset(X_train, y_train)
    val_dataset = CandidateDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model, Criterion, Optimizer
    model = BinaryMLPClassifier(
        input_dim=len(feature_cols),
        hidden_dims=hidden_dims,
        dropout_rate=dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    history = []

    print(f"\nTraining Model 5 (MLP) on {device}...")
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Imbalance ratio (neg/pos): {pos_weight_val:.2f}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(batch_x)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_samples += len(batch_x)

        train_loss = train_loss_total / train_samples
        train_acc = train_correct / train_samples

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_samples = 0
        val_probs_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss_total += loss.item() * len(batch_x)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_samples += len(batch_x)
                val_probs_list.extend(probs.cpu().numpy().flatten())

        val_loss = val_loss_total / val_samples
        val_acc = val_correct / val_samples
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": curr_lr
        }
        history.append(epoch_stats)

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            is_best = False

        if epoch % 5 == 0 or is_best or epoch == epochs:
            best_tag = " [BEST]" if is_best else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{best_tag}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with val_loss {best_val_loss:.4f}).")
            break

    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f} seconds.")

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 5. Final Evaluation on Validation Split using Best Model
    model.eval()
    val_probs_arr = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            probs = model.predict_proba(batch_x).cpu().numpy().flatten()
            val_probs_arr.extend(probs)
    val_probs_arr = np.asarray(val_probs_arr)

    metrics = compute_model_metrics(y_val, val_probs_arr, val_df)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["training_time_sec"] = round(train_duration, 3)

    # 6. Save Artifacts
    # Save Model Weights
    best_model_path = os.path.join(output_dir, "best_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": len(feature_cols),
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }, best_model_path)

    # Save Preprocessing Pipeline
    prep_path = os.path.join(output_dir, "preprocessing.joblib")
    joblib.dump(preprocessor, prep_path)

    # Save Feature Columns
    cols_path = os.path.join(output_dir, "feature_columns.json")
    with open(cols_path, "w") as fp:
        json.dump(feature_cols, fp, indent=2)

    # Save Training History CSV
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, "training_history.csv")
    history_df.to_csv(history_path, index=False)

    # Save Training Config
    config_dict = {
        "model_family": "Neural Network (MLP)",
        "model_name": "Model 5 - BinaryMLPClassifier",
        "input_dim": len(feature_cols),
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "patience": patience,
        "seed": seed,
        "device": str(device),
        "features": feature_cols
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as fp:
        json.dump(config_dict, fp, indent=2)

    # Save Summary & Metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    summary_dict = {
        "model": "Model 5 (MLP)",
        "status": "TRAINED",
        "best_epoch": best_epoch,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "candidate_accuracy": metrics["candidate_accuracy"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "observation_top1_accuracy": metrics["observation_top1_accuracy"]
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as fp:
        json.dump(summary_dict, fp, indent=2)

    # Save Validation Predictions
    val_preds_df = predict_candidate_probabilities(model, preprocessor, val_df, device=device)
    val_preds_df["ground_truth"] = y_val.astype(int)
    val_preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # 7. Generate Visual Evaluation Plots
    generate_training_plots(history_df, metrics, output_dir)

    print(f"\nAll Model 5 artifacts successfully saved to: {output_dir}/")
    return metrics


# ==============================================================================
# Visualization / Plotting
# ==============================================================================

def generate_training_plots(
    history_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str
):
    """
    Generates training loss curve, accuracy curve, and confusion matrix.
    """
    sns.set_theme(style="whitegrid")

    # Plot 1: Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss", color="#2980b9", lw=2)
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss", color="#e74c3c", lw=2)
    best_ep = metrics.get("best_epoch", 1)
    best_loss = metrics.get("best_val_loss", 0.0)
    plt.axvline(best_ep, color="green", linestyle="--", alpha=0.8, label=f"Best Checkpoint (Ep {best_ep}: {best_loss:.4f})")
    plt.title("Model 5 (MLP): Training and Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Plot 2: Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Training Accuracy", color="#2980b9", lw=2)
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy", color="#27ae60", lw=2)
    plt.title("Model 5 (MLP): Candidate Classification Accuracy", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    # Plot 3: Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative (0)", "Ground Truth (1)"],
        yticklabels=["Negative (0)", "Ground Truth (1)"]
    )
    plt.title(f"Model 5 (MLP) Confusion Matrix\n(F1: {metrics['f1_score']:.3f} | Top-1: {metrics['observation_top1_accuracy']:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Candidate Label")
    plt.ylabel("True Candidate Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PathSnap Model 5: Multi-Layer Perceptron (MLP) Training Pipeline"
    )
    parser.add_argument("--data", type=str, default=None, help="Path to real candidate dataset (.parquet or .csv)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden-dims", type=str, default="128,64,32", help="Comma-separated hidden layer dimensions (default: 128,64,32)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability (default: 0.2)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--output-dir", type=str, default="results/model5_mlp", help="Directory to save model artifacts")
    parser.add_argument("--synthetic-smoke-test", action="store_true", help="Run end-to-end smoke test on synthetic fixtures (testing only)")

    args = parser.parse_args()

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]

    # Case 1: Synthetic smoke test requested
    if args.synthetic_smoke_test:
        print("=== RUNNING MODEL 5 PIPELINE SMOKE TEST ON SYNTHETIC FIXTURE ===")
        print("NOTE: Synthetic fixtures are used strictly for pipeline and test verification.")
        df_candidates = generate_synthetic_candidate_data(n_trips=8, obs_per_trip=25, seed=args.seed)
    # Case 2: Real data supplied via --data
    elif args.data is not None:
        if not os.path.exists(args.data):
            print(f"ERROR: Specified candidate dataset not found at: {args.data}")
            sys.exit(1)
        print(f"Loading candidate dataset from: {args.data}")
        df_candidates = load_candidate_dataset(args.data)
    # Case 3: No data supplied
    else:
        print("\n================================================================================")
        print("MODEL 5 (MLP) - CANDIDATE DATASET REQUIREMENT")
        print("================================================================================")
        print("Model 5 requires a preprocessed candidate feature dataset from Member 1.")
        print("Expected columns:")
        print("  - Metadata: trip_id (str), obs_idx (int), candidate_arc_id (int)")
        print("  - Target:   is_ground_truth (int: 0 or 1)")
        print("  - Features: perp_distance_m, bearing_difference_deg, segment_length_m, speed_mps")
        print("\nUsage:")
        print("  python -m models.train_mlp --data <path-to-candidates.parquet> [options]")
        print("\nTo verify the pipeline execution on synthetic test data without claiming real accuracy:")
        print("  python -m models.train_mlp --synthetic-smoke-test --epochs 5 --output-dir /tmp/mlp_smoke")
        print("================================================================================\n")
        sys.exit(0)

    # Validate Schema
    is_valid, issues = validate_candidate_schema(df_candidates, require_labels=True)
    if not is_valid:
        print(f"ERROR: Candidate dataset schema validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    feature_cols = get_feature_columns(df_candidates)
    print(f"Detected {len(feature_cols)} feature columns: {feature_cols}")

    # Trip-Grouped Splitting (prevents intra-trip data leakage)
    unique_trips = sorted(df_candidates["trip_id"].unique())
    print(f"Dataset contains {len(df_candidates)} total candidate rows across {len(unique_trips)} unique trips.")

    if len(unique_trips) < 2:
        print("ERROR: Dataset must contain at least 2 distinct trips to form train and validation splits.")
        sys.exit(1)

    gkf = GroupKFold(n_splits=min(5, len(unique_trips)))
    train_idx, val_idx = next(gkf.split(df_candidates, groups=df_candidates["trip_id"]))

    train_split = df_candidates.iloc[train_idx].reset_index(drop=True)
    val_split = df_candidates.iloc[val_idx].reset_index(drop=True)

    train_trips = set(train_split["trip_id"].unique())
    val_trips = set(val_split["trip_id"].unique())
    assert len(train_trips & val_trips) == 0, "CRITICAL ERROR: Trip leakage between train and validation splits!"

    print(f"Train partition: {len(train_split)} candidates across {len(train_trips)} trips: {sorted(list(train_trips))[:5]}...")
    print(f"Validation partition: {len(val_split)} candidates across {len(val_trips)} trips: {sorted(list(val_trips))[:5]}...")

    # Execute Training
    metrics = train_mlp_model(
        train_df=train_split,
        val_df=val_split,
        feature_cols=feature_cols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        output_dir=args.output_dir
    )

    print("\n=== FINAL EVALUATION METRICS ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:30s}: {v}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
