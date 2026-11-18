import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier

from src.metrics import run_experiment

CONFIG = dict(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)


def make_model(y):
    return RandomForestClassifier(**CONFIG)


if __name__ == "__main__":
    run_experiment("rf", "Bagging ensemble (trees)", CONFIG, make_model, artifact_name="rf_model.joblib")
