import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier

from src.metrics import FIGURES, run_experiment

CONFIG = dict(n_estimators=300, random_state=42, n_jobs=-1, verbose=-1)


def scale_pos_weight(y):
    y = np.asarray(y)
    return (y == 0).sum() / max((y == 1).sum(), 1)


def make_model(y):
    return LGBMClassifier(**CONFIG, scale_pos_weight=scale_pos_weight(y))


def fit_with_curves(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_names=["train", "val"],
              eval_metric="binary_logloss")
    history = model.evals_result_
    return {"train_logloss": history["train"]["binary_logloss"], "val_logloss": history["val"]["binary_logloss"]}


def plot_loss_curve(curves, path):
    train_loss = np.array([c["train_logloss"] for c in curves])
    val_loss = np.array([c["val_logloss"] for c in curves])
    iters = np.arange(1, train_loss.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    for loss, label, color in [(train_loss, "train", "#4C72B0"), (val_loss, "validation", "#DD8452")]:
        mean, std = loss.mean(axis=0), loss.std(axis=0)
        ax.plot(iters, mean, label=f"{label} (mean of {len(loss)} folds)", color=color)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.2)
    best = val_loss.mean(axis=0).argmin() + 1
    ax.axvline(best, ls="--", color="grey", lw=1, label=f"min val logloss @ {best}")
    ax.set_xlabel("boosting iteration")
    ax.set_ylabel("binary logloss")
    ax.set_title("LightGBM train vs validation loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    config = {**CONFIG, "scale_pos_weight": "n_neg/n_pos (per training set)"}
    result, _ = run_experiment("lgbm", "Gradient boosting (trees)", config, make_model,
                               fit_fn=fit_with_curves, artifact_name="lgbm_untuned.joblib")
    plot_loss_curve(result["fold_extras"], FIGURES / "lgbm_loss_curve.png")
