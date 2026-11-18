"""Tune LightGBM with Optuna (TPE) on mean 5-fold CV Macro F1, then compare against the untuned model."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

from models.train_lgbm import CONFIG as UNTUNED, scale_pos_weight
from src.metrics import RESULTS, feature_columns, get_cv_splits, load_candidates, run_experiment

STUDY_NAME = "lgbm_macro_f1"
FIXED = dict(random_state=42, n_jobs=-1, verbose=-1, subsample_freq=1)  # subsample is ignored unless subsample_freq > 0


def suggest_params(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def comparison_row(label, result):
    cv, cv_std = result["cv"]["val_mean"], result["cv"]["val_std"]
    st = result["student_test"] or {}
    return {
        "Model": label,
        "CV Macro F1 (mean)": cv["macro_f1"],
        "CV Macro F1 (std)": cv_std["macro_f1"],
        "CV Train Macro F1": result["cv"]["train_mean"]["macro_f1"],
        "CV Top-1 Accuracy": cv["top1_accuracy"],
        "CV Log Loss": cv["log_loss"],
        "Student-Test Macro F1": st.get("macro_f1"),
        "Student-Test Top-1 Accuracy": st.get("top1_accuracy"),
        "Student-Test Log Loss": st.get("log_loss"),
        "Route Continuity": st.get("route_continuity"),
        "Inference ms/trip": st.get("inference_ms_per_trip"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--refit-only", action="store_true",
                        help="skip new trials and the baseline re-run; refit the best saved trial and evaluate it")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    df = load_candidates()
    train = df[df["split"] == "train"].reset_index(drop=True)
    n_trips = train["trip_id"].nunique()
    if n_trips != 80:
        raise ValueError(f"Expected the 80 official training trips, found {n_trips}. Never tune on student_test.")
    feats = feature_columns(train)
    X, y = train[feats], train["label"].to_numpy()
    folds = get_cv_splits(train)

    def objective(trial):
        params = suggest_params(trial)
        scores = []
        for tr_idx, va_idx in folds:
            model = LGBMClassifier(**params, **FIXED, scale_pos_weight=scale_pos_weight(y[tr_idx]))
            model.fit(X.iloc[tr_idx], y[tr_idx])
            scores.append(f1_score(y[va_idx], model.predict(X.iloc[va_idx]), average="macro", zero_division=0))
        trial.set_user_attr("fold_macro_f1", [float(s) for s in scores])
        trial.set_user_attr("std_macro_f1", float(np.std(scores)))
        return float(np.mean(scores))

    # load_if_exists keeps earlier trials, so re-running adds to the same study
    storage = f"sqlite:///{(RESULTS / 'optuna_study.db').as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42), load_if_exists=True)

    if args.refit_only:
        untuned = json.loads((RESULTS / "metrics_lgbm.json").read_text())
        del df, train, X
    else:
        # untuned baseline, using the same folds and test split as the tuned model
        untuned, _ = run_experiment(
            "lgbm", "Gradient boosting (trees)",
            {**UNTUNED, "scale_pos_weight": "n_neg/n_pos (per training set)"},
            lambda yy: LGBMClassifier(**UNTUNED, scale_pos_weight=scale_pos_weight(yy)),
            artifact_name="lgbm_untuned.joblib",
        )
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    study.trials_dataframe().to_csv(RESULTS / "optuna_trials.csv", index=False)
    best = study.best_params
    print(f"Best CV Macro F1={study.best_value:.4f} params={best}")

    config = {**best, **FIXED, "scale_pos_weight": "n_neg/n_pos (per training set)"}
    tuned, _ = run_experiment(
        "lgbm_tuned", "Gradient boosting (trees)", config,
        lambda yy: LGBMClassifier(**best, **FIXED, scale_pos_weight=scale_pos_weight(yy)),
        artifact_name="best_model_tuned.joblib",
    )
    tuned["optuna"] = {
        "n_trials": len(study.trials),
        "completed_trials": sum(t.value is not None for t in study.trials),
        "new_trials": 0 if args.refit_only else args.n_trials,
        "sampler": "TPESampler(seed=42)",
        "objective": "mean 5-fold CV Macro F1",
        "best_value": study.best_value,
        "best_params": best,
        "best_trial": study.best_trial.number,
    }
    (RESULTS / "metrics_lgbm_tuned.json").write_text(json.dumps(tuned, indent=2, default=str))

    table = pd.DataFrame([comparison_row("LightGBM (untuned)", untuned),
                          comparison_row("LightGBM (Optuna-tuned)", tuned)])
    table.to_csv(RESULTS / "untuned_vs_tuned_comparison.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
