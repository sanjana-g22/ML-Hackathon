"""
Model: k-Nearest Neighbours candidate classifier (instance-based family).

Run from the repo root (needs data/candidates.parquet from `python -m src.member1_pipeline --output data`):
    python models/train_knn.py
    python models/train_knn.py --sample-size-study      # also measure training-sample size vs score/time

Uses the shared harness in src/metrics.py: the same GroupKFold(5) folds, the same metric
definitions and the same output files (results/metrics_knn.json, results/predictions_knn.parquet),
so `python src/metrics.py` puts KNN in the model comparison table next to the other models.

Why it is fast
--------------
1. KNN stores every training row and searches them at prediction time, so its cost grows with
   the training-set size. Each fold is fitted on a fixed-size uniform random sample of the
   training candidates (seeded), which keeps the class balance unchanged.
2. Neighbours are queried ONCE per fold at the largest k in the sweep. The prediction for every
   smaller k (uniform and distance-weighted) is derived from that single query with cumulative
   sums, so the whole k validation curve costs about the same as one model.
3. Brute-force search (BLAS matrix maths) on all CPU cores, in chunks. Benchmarked on 300k stored
   rows x 14 features: ~140 us/query vs ~930-1080 us for a KD-tree and 3,900-15,000 us for a
   ball tree; trees cannot prune well in 14 dimensions.
4. The k sweep is scored on a random 25% of each validation fold's GPS points (all candidates
   of a sampled point are kept, so Top-1 stays well defined). The chosen model is then scored on
   every candidate of all 10 student-test trips.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import (ARTIFACTS, FIGURES, KEYS, PRED_COLS, RESULTS, evaluate_predictions,
                         feature_columns, get_cv_splits, load_candidates, plot_confusion,
                         predict_df, summarize_folds, to_python)

MODEL_NAME = "knn"
FAMILY = "Instance-based"
SEED = 42
TRAIN_SAMPLE = 300_000                      # training candidates stored per fit
K_VALUES = [1, 3, 5, 9, 15, 25, 35, 51]     # k values compared in the validation curve
WEIGHTS = ["uniform", "distance"]
CHUNK = 50_000                              # validation rows per neighbour query
VAL_OBS_FRACTION = 0.25                     # share of validation GPS points used in the k sweep
ALGORITHM = "brute"                         # fastest in 14-D (see module docstring)
TRAIN_EVAL_ROWS = 50_000                    # rows used to measure training-set (fitted) F1


# --------------------------------------------------------------------------- #
# Preprocessing + neighbour search
# --------------------------------------------------------------------------- #
def sample_rows(df, n, seed=SEED):
    return df.sample(n=min(n, len(df)), random_state=seed)


def fit_preprocessor(X):
    """Median-impute missing motion features (first point, GPS gaps), then standardise.

    KNN uses Euclidean distance, so features must be on comparable scales, and it cannot
    handle NaN (tree models can).
    """
    imputer = SimpleImputer(strategy="median").fit(X)
    scaler = StandardScaler().fit(imputer.transform(X))
    return lambda Z: scaler.transform(imputer.transform(Z)).astype(np.float32)


def sweep_probabilities(nn, y_train, Xq, k_values):
    """P(label=1) for every k in k_values and both weightings, from one neighbour query.

    Matches sklearn's KNeighborsClassifier: uniform = mean neighbour label; distance =
    1/d-weighted mean, and if any of the k neighbours is at distance 0 only those count.
    """
    k_max = max(k_values)
    out = {(w, k): np.empty(len(Xq), dtype=np.float32) for w in WEIGHTS for k in k_values}
    kk = np.array(k_values) - 1
    for start in range(0, len(Xq), CHUNK):
        sl = slice(start, start + CHUNK)
        dist, idx = nn.kneighbors(Xq[sl], n_neighbors=k_max)
        lab = y_train[idx].astype(np.float64)

        uni = np.cumsum(lab, axis=1)[:, kk] / (kk + 1)

        zero = dist == 0
        with np.errstate(divide="ignore"):
            w = np.where(zero, 0.0, 1.0 / dist)
        wsum, wy = np.cumsum(w, axis=1)[:, kk], np.cumsum(w * lab, axis=1)[:, kk]
        zsum, zy = np.cumsum(zero, axis=1)[:, kk], np.cumsum(zero * lab, axis=1)[:, kk]
        with np.errstate(invalid="ignore", divide="ignore"):
            dw = np.where(zsum > 0, zy / np.maximum(zsum, 1), wy / np.maximum(wsum, 1e-300))

        for j, k in enumerate(k_values):
            out[("uniform", k)][sl] = uni[:, j]
            out[("distance", k)][sl] = dw[:, j]
    return out


def quick_metrics(frame, prob):
    """Candidate-level macro F1 / log loss and observation-level Top-1 (same definitions as
    src/metrics.evaluate_predictions; ties go to the nearest candidate, as rows are distance-sorted)."""
    y = frame["label"].to_numpy()
    f1 = f1_score(y, prob >= 0.5, average="macro", zero_division=0)
    ll = log_loss(y, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1])
    tmp = frame[KEYS + ["label"]].assign(prob=prob)
    top = tmp.loc[tmp.groupby(KEYS, sort=False)["prob"].idxmax()]
    return f1, (top["label"] == 1).mean(), ll


def make_final_model(k, weights):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=ALGORITHM, n_jobs=-1))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_validation_curve(sweep, best, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"uniform": "#4C72B0", "distance": "#DD8452"}
    for w in WEIGHTS:
        s = sweep[sweep.weights == w].sort_values("k")
        c = colors[w]
        axes[0].errorbar(s.k, s.cv_macro_f1, yerr=s.cv_macro_f1_std, marker="o", color=c, capsize=3,
                         label=f"{w}: validation")
        axes[0].plot(s.k, s.train_macro_f1, ls="--", marker=".", color=c, alpha=.7, label=f"{w}: training (fitted rows)")
        axes[1].errorbar(s.k, s.cv_top1, yerr=s.cv_top1_std, marker="o", color=c, capsize=3, label=w)
        axes[2].plot(s.k, s.cv_log_loss, marker="o", color=c, label=w)
    for ax, title, ylab in [(axes[0], "Macro F1 vs k (5-fold CV)", "Macro F1"),
                            (axes[1], "Top-1 accuracy vs k (5-fold CV)", "Top-1 accuracy"),
                            (axes[2], "Log loss vs k (5-fold CV, lower is better)", "Log loss")]:
        ax.axvline(best["k"], color="grey", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
        ax.set_xlabel("k (number of neighbours)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=.3)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="lower left", ncol=2)
        else:
            ax.legend(fontsize=8)
    axes[0].set_ylim(0.62, 1.03)
    fig.suptitle(f"KNN validation curve: chosen k={best['k']}, weights={best['weights']} "
                 f"(train sample {TRAIN_SAMPLE:,} candidates/fold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_summary(cv_mean, cv_std, test, path):
    names = ["Macro F1", "Top-1 accuracy", "Route continuity", "Sequence F1"]
    keys = ["macro_f1", "top1_accuracy", "route_continuity", "sequence_f1"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    cv_vals = np.array([cv_mean[k] for k in keys], dtype=float)
    cv_err = np.nan_to_num(np.array([cv_std[k] for k in keys], dtype=float))
    b1 = ax.bar(x - .2, np.nan_to_num(cv_vals), .4, yerr=cv_err, capsize=3,
                label="5-fold CV (25% of validation points)", color="#4C72B0")
    b2 = ax.bar(x + .2, [test[k] for k in keys], .4, label="Student-test (all 10 trips)", color="#DD8452")
    for bars, vals, errs in ((b1, cv_vals, cv_err), (b2, [test[k] for k in keys], [0.0] * len(keys))):
        for b, v, e in zip(bars, vals, errs):
            text = "n/a*" if not np.isfinite(v) else f"{v:.3f}"
            ax.annotate(text, (b.get_x() + b.get_width() / 2, 0 if not np.isfinite(v) else v + e),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.18)
    ax.set_title("KNN: cross-validation vs student-test\n"
                 "*route metrics need whole trips, so they are reported on student-test only", fontsize=10)
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_per_trip(preds, test, path):
    m = preds.merge(test[KEYS + ["arc_id", "label"]], on=KEYS + ["arc_id"])
    top = m.loc[m.groupby(KEYS, sort=False)["prob"].idxmax()]
    acc = top.groupby("trip_id")["label"].mean().sort_values()
    n = top.groupby("trip_id").size()[acc.index]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    bars = ax.bar(acc.index, acc.values, color="#4C72B0")
    for b, v, cnt in zip(bars, acc.values, n.values):
        ax.annotate(f"{v:.2f}\n{cnt:,} pts", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=6.5, xytext=(0, 2), textcoords="offset points")
    ax.axhline(acc.mean(), color="grey", ls="--", lw=1, label=f"mean of trips = {acc.mean():.3f}")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlabel("student-test trip")
    ax.set_title("KNN Top-1 accuracy per student-test trip")
    ax.legend(loc="upper left", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def sample_size_study(train, feats, folds, best, sizes=(50_000, 100_000, 300_000, 600_000)):
    """Fold 0 only: how much does storing more training rows help, and what does it cost?"""
    tr_idx, va_idx = folds[0]
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    rows = []
    for n in sizes:
        fit = sample_rows(tr, n)
        prep = fit_preprocessor(fit[feats])
        t0 = time.perf_counter()
        nn = NearestNeighbors(algorithm=ALGORITHM, n_jobs=-1).fit(prep(fit[feats]))
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        probs = sweep_probabilities(nn, fit["label"].to_numpy(), prep(va[feats]), [best["k"]])
        query_s = time.perf_counter() - t0
        f1, top1, ll = quick_metrics(va, probs[(best["weights"], best["k"])])
        rows.append({"train_sample": min(n, len(tr)), "val_macro_f1": f1, "val_top1": top1, "val_log_loss": ll,
                     "fit_s": fit_s, "query_s": query_s, "val_rows": len(va)})
        print(f"  sample {n:>7,}: F1={f1:.4f} top1={top1:.4f} fit={fit_s:.1f}s query={query_s:.1f}s", flush=True)
    study = pd.DataFrame(rows)
    study.to_csv(RESULTS / "knn_sample_size_study.csv", index=False, float_format="%.4f")

    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(study.train_sample, study.val_macro_f1, marker="o", color="#4C72B0", label="validation Macro F1")
    ax1.plot(study.train_sample, study.val_top1, marker="s", color="#55A868", label="validation Top-1")
    ax1.set_xscale("log")
    ax1.set_xlabel("training candidates stored by KNN (fold 0)")
    ax1.set_ylabel("score")
    ax2 = ax1.twinx()
    ax2.bar(study.train_sample, study.query_s, width=study.train_sample * .25, alpha=.25, color="grey",
            label="prediction time (s)")
    ax2.set_ylabel("prediction time for the fold (s)")
    lines = ax1.get_legend_handles_labels()
    bars = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + bars[0], lines[1] + bars[1], fontsize=8, loc="lower right")
    ax1.set_title(f"KNN training-sample size vs accuracy and time (k={best['k']}, {best['weights']})")
    fig.tight_layout()
    fig.savefig(FIGURES / "knn_sample_size.png", dpi=140)
    plt.close(fig)


def redraw_plots():
    """Regenerate all KNN figures from results/ without retraining."""
    sweep = pd.read_csv(RESULTS / "knn_k_sweep.csv")
    result = json.loads((RESULTS / f"metrics_{MODEL_NAME}.json").read_text())
    best = {"k": result["config"]["n_neighbors"], "weights": result["config"]["weights"]}
    nan_if_none = lambda d: {k: (float("nan") if v is None else v) for k, v in d.items()}
    plot_validation_curve(sweep, best, FIGURES / "knn_validation_curve.png")
    plot_summary(nan_if_none(result["cv"]["val_mean"]), nan_if_none(result["cv"]["val_std"]),
                 result["student_test"], FIGURES / "knn_cv_vs_test.png")
    test = load_candidates().query("split == 'student_test' and label >= 0").reset_index(drop=True)
    preds = pd.read_parquet(RESULTS / f"predictions_{MODEL_NAME}.parquet")
    plot_per_trip(preds, test, FIGURES / "knn_per_trip_top1.png")
    plot_confusion(result["student_test"]["confusion_matrix"], f"KNN k={best['k']} (student_test)",
                   FIGURES / "cm_knn.png")
    print("figures redrawn in", FIGURES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size-study", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="redraw figures from saved results")
    args = parser.parse_args()
    if args.plots_only:
        return redraw_plots()
    t_start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    df = load_candidates()
    df = df[df["label"] >= 0]                        # -1 = no reference (never in dev splits)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "student_test"].reset_index(drop=True)
    feats = feature_columns(train)
    print(f"[knn] {len(feats)} features: {feats}")
    print(f"[knn] train {len(train):,} rows / {train.trip_id.nunique()} trips | "
          f"test {len(test):,} rows / {test.trip_id.nunique()} trips | positive rate {train.label.mean():.4f}")

    # ---- 1. 5-fold grouped CV: one neighbour query per fold gives every (k, weights) ----
    folds = get_cv_splits(train)
    sweep_rows, fold_probs, timings = [], [], []
    for f, (tr_idx, va_idx) in enumerate(folds):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        points = va[KEYS].drop_duplicates().sample(frac=VAL_OBS_FRACTION, random_state=SEED + f)
        va = va.merge(points, on=KEYS)          # keeps every candidate of each sampled point, in order
        fit = sample_rows(tr, TRAIN_SAMPLE, SEED + f)
        prep = fit_preprocessor(fit[feats])
        X_fit, y_fit = prep(fit[feats]), fit["label"].to_numpy()

        t0 = time.perf_counter()
        nn = NearestNeighbors(algorithm=ALGORITHM, n_jobs=-1).fit(X_fit)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        val_probs = sweep_probabilities(nn, y_fit, prep(va[feats]), K_VALUES)
        query_s = time.perf_counter() - t0
        held = sample_rows(fit, TRAIN_EVAL_ROWS, SEED)
        train_probs = sweep_probabilities(nn, y_fit, prep(held[feats]), K_VALUES)
        timings.append({"fold": f, "fit_s": fit_s, "val_query_s": query_s, "val_rows": len(va)})

        for (w, k), p in val_probs.items():
            f1, top1, ll = quick_metrics(va, p)
            train_f1 = f1_score(held["label"], train_probs[(w, k)] >= .5, average="macro", zero_division=0)
            sweep_rows.append({"fold": f, "weights": w, "k": k, "val_macro_f1": f1, "val_top1": top1,
                               "val_log_loss": ll, "train_macro_f1": train_f1})
        fold_probs.append((va, val_probs))
        print(f"  fold {f}: fit {fit_s:.1f}s, query {len(va):,} rows {query_s:.1f}s", flush=True)

    per_fold = pd.DataFrame(sweep_rows)
    sweep = per_fold.groupby(["weights", "k"]).agg(
        cv_macro_f1=("val_macro_f1", "mean"), cv_macro_f1_std=("val_macro_f1", "std"),
        cv_top1=("val_top1", "mean"), cv_top1_std=("val_top1", "std"),
        cv_log_loss=("val_log_loss", "mean"), train_macro_f1=("train_macro_f1", "mean")).reset_index()
    sweep.to_csv(RESULTS / "knn_k_sweep.csv", index=False, float_format="%.4f")
    per_fold.to_csv(RESULTS / "knn_k_sweep_per_fold.csv", index=False, float_format="%.4f")

    # Selection: team primary metric (CV Macro F1), log loss as tie-breaker.
    best = sweep.sort_values(["cv_macro_f1", "cv_log_loss"], ascending=[False, True]).iloc[0]
    best = {"k": int(best.k), "weights": best.weights}
    print("\n" + sweep.sort_values(["weights", "k"]).to_string(index=False, float_format="%.4f"))
    print(f"\n[knn] selected k={best['k']} weights={best['weights']} (highest mean CV Macro F1)")
    plot_validation_curve(sweep, best, FIGURES / "knn_validation_curve.png")

    # ---- 2. Full shared-harness metrics for the selected configuration on each CV fold ----
    fold_results = []
    for f, (va, probs) in enumerate(fold_probs):
        pred = va[["trip_id", "split", "obs_idx", "arc_id"]].assign(prob=probs[(best["weights"], best["k"])])
        m_va = evaluate_predictions(pred, va)
        m_va["route_continuity"] = m_va["sequence_f1"] = float("nan")   # sampled points: not a route
        tr_f1 = per_fold.query("fold == @f and weights == @best['weights'] and k == @best['k']").train_macro_f1.item()
        fold_results.append({"fold": f, "train": {"macro_f1": tr_f1}, "val": m_va})
    va_mean, va_std = summarize_folds([r["val"] for r in fold_results])
    tr_mean = {"macro_f1": float(np.mean([r["train"]["macro_f1"] for r in fold_results]))}

    # ---- 3. Refit on all 80 training trips (same sample size) and score the 10 student-test trips ----
    final = make_final_model(best["k"], best["weights"])
    fit = sample_rows(train, TRAIN_SAMPLE, SEED)
    t0 = time.perf_counter()
    final.fit(fit[feats], fit["label"])
    final_fit_s = time.perf_counter() - t0
    joblib.dump({"model": final, "features": feats}, ARTIFACTS / "knn_model.joblib")

    preds = predict_df(final, test, feats, MODEL_NAME, per_trip_timing=True)
    preds[PRED_COLS].to_parquet(RESULTS / f"predictions_{MODEL_NAME}.parquet", index=False)
    test_metrics = evaluate_predictions(preds, test, verbose=True)
    plot_confusion(test_metrics["confusion_matrix"], f"KNN k={best['k']} (student_test)", FIGURES / "cm_knn.png")
    plot_summary(va_mean, va_std, test_metrics, FIGURES / "knn_cv_vs_test.png")
    plot_per_trip(preds, test, FIGURES / "knn_per_trip_top1.png")

    config = {"n_neighbors": best["k"], "weights": best["weights"], "algorithm": ALGORITHM,
              "preprocessing": "median impute + StandardScaler", "train_sample_per_fit": TRAIN_SAMPLE,
              "k_grid": K_VALUES, "weights_grid": WEIGHTS, "cv_val_obs_fraction": VAL_OBS_FRACTION, "seed": SEED}
    result = {
        "model": MODEL_NAME,
        "display_name": f"KNN (k={best['k']}, {best['weights']})",
        "family": FAMILY,
        "config": config,
        "features": feats,
        "n_train_rows": len(train), "n_train_trips": train.trip_id.nunique(),
        "n_test_rows": len(test), "n_test_trips": test.trip_id.nunique(),
        "pos_rate_train": train["label"].mean(),
        "final_fit_seconds": final_fit_s,
        "cv": {"folds": fold_results, "train_mean": tr_mean, "train_std": {}, "val_mean": va_mean, "val_std": va_std},
        "student_test": test_metrics,
        "fold_timings": timings,
        "total_runtime_s": None,
    }

    if args.sample_size_study:
        print("\n[knn] sample-size study (fold 0)")
        sample_size_study(train, feats, folds, best)

    result["total_runtime_s"] = time.perf_counter() - t_start
    (RESULTS / f"metrics_{MODEL_NAME}.json").write_text(json.dumps(result, indent=2, default=to_python))

    print("\n================ KNN summary ================")
    print(f"config            : k={best['k']}, weights={best['weights']}, {TRAIN_SAMPLE:,} stored training candidates")
    print(f"CV Macro F1       : {va_mean['macro_f1']:.4f} +/- {va_std['macro_f1']:.4f}   (training rows: {tr_mean['macro_f1']:.4f})")
    print(f"CV Top-1          : {va_mean['top1_accuracy']:.4f} +/- {va_std['top1_accuracy']:.4f}")
    for key, label in [("macro_f1", "Test Macro F1"), ("top1_accuracy", "Test Top-1"), ("precision", "Test precision"),
                       ("recall", "Test recall"), ("log_loss", "Test log loss"), ("route_continuity", "Test continuity"),
                       ("sequence_f1", "Test sequence F1"), ("candidate_miss_rate", "Candidate miss"),
                       ("inference_ms_per_trip", "Inference ms/trip")]:
        print(f"{label:<18}: {test_metrics[key]:.4f}")
    print(f"total runtime     : {result['total_runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
