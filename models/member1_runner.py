"""Member 1 experiment runner (Logistic Regression, KNN).

Same protocol and output files as src.metrics.run_experiment (5-fold GroupKFold CV on the
80 train trips, refit on all train, evaluate on the 10 student_test trips), with two
optional compute caps that matter for instance-based models:

  cv_eval_trips   - validation-fold and train-fold *metrics* are computed on a seeded
                    subset of whole trips (never a row subset, so Top-1 stays valid).
                    Fitting still uses every training row of the fold.
  train_eval_trips- same, for the train-side metrics used for the overfitting check.

student_test metrics are always computed on all 10 trips.
Outputs are compatible with `python src/metrics.py` (comparison table and bar charts).
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

from src.metrics import (ARTIFACTS, FIGURES, PRED_COLS, RESULTS, evaluate_predictions,
                         feature_columns, get_cv_splits, load_candidates, plot_confusion,
                         predict_df, summarize_folds, to_python)

SEED = 42
M1 = RESULTS / "member1"  # Member 1 outputs; copies are mirrored to results/ for src/metrics.py


def pick_trips(df, n, seed):
    """Seeded subset of whole trips; n=None keeps everything."""
    if not n:
        return df
    trips = np.array(sorted(df.trip_id.unique()))
    keep = np.random.default_rng(seed).permutation(trips)[:n]
    return df[df.trip_id.isin(keep)]


def run(model_name, display_name, family, config, make_model, artifact_name,
        cv_eval_trips=None, train_eval_trips=None):
    M1.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); ARTIFACTS.mkdir(exist_ok=True)
    df = load_candidates()
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "student_test"].reset_index(drop=True)
    feats = feature_columns(train)
    print(f"[{model_name}] {len(feats)} features | train {len(train):,} rows/{train.trip_id.nunique()} trips"
          f" | student_test {len(test):,} rows/{test.trip_id.nunique()} trips", flush=True)

    folds = []
    for k, (tr_idx, va_idx) in enumerate(get_cv_splits(train)):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        t0 = time.perf_counter()
        model = make_model(tr["label"]); model.fit(tr[feats], tr["label"])
        fit_s = time.perf_counter() - t0
        tr_e = pick_trips(tr, train_eval_trips, SEED + k).reset_index(drop=True)
        va_e = pick_trips(va, cv_eval_trips, SEED + k).reset_index(drop=True)
        m_tr = evaluate_predictions(predict_df(model, tr_e, feats, model_name), tr_e)
        m_va = evaluate_predictions(predict_df(model, va_e, feats, model_name), va_e)
        m_va["fit_seconds"] = fit_s
        folds.append({"fold": k, "train": m_tr, "val": m_va})
        print(f"  fold {k}: fit {fit_s:.0f}s | train F1={m_tr['macro_f1']:.4f} top1={m_tr['top1_accuracy']:.4f}"
              f" | val F1={m_va['macro_f1']:.4f} LL={m_va['log_loss']:.4f} top1={m_va['top1_accuracy']:.4f}"
              f" ({m_va['n_trips']:.0f} val trips)", flush=True)

    tr_mean, tr_std = summarize_folds([f["train"] for f in folds])
    va_mean, va_std = summarize_folds([f["val"] for f in folds])
    print(f"  CV mean: train F1={tr_mean['macro_f1']:.4f} | val F1={va_mean['macro_f1']:.4f}"
          f" +/- {va_std['macro_f1']:.4f}", flush=True)

    final = make_model(train["label"])
    t0 = time.perf_counter(); final.fit(train[feats], train["label"])
    fit_seconds = time.perf_counter() - t0
    joblib.dump({"model": final, "features": feats}, ARTIFACTS / artifact_name)

    preds = predict_df(final, test, feats, model_name, per_trip_timing=True)
    preds[PRED_COLS].to_parquet(M1 / f"predictions_{model_name}.parquet", index=False)
    st = evaluate_predictions(preds, test, verbose=True)
    plot_confusion(st["confusion_matrix"], f"{display_name} (student_test)", M1 / f"cm_{model_name}.png")
    print(f"  student_test: F1={st['macro_f1']:.4f} top1={st['top1_accuracy']:.4f} LL={st['log_loss']:.4f}"
          f" continuity={st['route_continuity']:.4f} seqF1={st['sequence_f1']:.4f}", flush=True)

    result = {"model": model_name, "display_name": display_name, "family": family, "config": config,
              "features": feats, "n_train_rows": len(train), "n_train_trips": train.trip_id.nunique(),
              "n_test_rows": len(test), "n_test_trips": test.trip_id.nunique(),
              "pos_rate_train": train["label"].mean(), "final_fit_seconds": fit_seconds,
              "eval_caps": {"cv_eval_trips_per_fold": cv_eval_trips, "train_eval_trips_per_fold": train_eval_trips,
                            "note": "metrics on whole-trip subsets; fitting always uses all fold rows"},
              "cv": {"folds": folds, "train_mean": tr_mean, "train_std": tr_std,
                     "val_mean": va_mean, "val_std": va_std},
              "student_test": st}
    (M1 / f"metrics_{model_name}.json").write_text(json.dumps(result, indent=2, default=to_python))
    (ARTIFACTS / "feature_config.json").write_text(json.dumps({"features": feats, "model": model_name,
                                                               "config": config}, indent=2, default=to_python))
    for name in (f"predictions_{model_name}.parquet", f"metrics_{model_name}.json"):
        shutil.copy(M1 / name, RESULTS / name)  # so `python src/metrics.py` includes this model
    shutil.copy(M1 / f"cm_{model_name}.png", FIGURES / f"cm_{model_name}.png")
    return result, final
