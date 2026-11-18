"""Model: Logistic Regression (linear / probabilistic family). Run: python models/train_logistic.py"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.member1_runner import run

# Initial (untuned) configuration. Imputer/scaler live inside the pipeline so they are fit on
# training folds only (no leakage into validation folds or student_test).
CONFIG = dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=500, class_weight="balanced",
              imputer="median", scaler="StandardScaler", random_state=42)


def make_model(y=None):
    return make_pipeline(
        SimpleImputer(strategy=CONFIG["imputer"]), StandardScaler(),
        LogisticRegression(C=CONFIG["C"], solver=CONFIG["solver"], max_iter=CONFIG["max_iter"],
                           class_weight=CONFIG["class_weight"], random_state=CONFIG["random_state"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-eval-trips", type=int, default=10, help="trips used for train-side metrics per fold")
    a = ap.parse_args()
    run("logistic", "Logistic Regression", "Linear / probabilistic", CONFIG, make_model,
        "logistic_pipeline.joblib", cv_eval_trips=None, train_eval_trips=a.train_eval_trips)
