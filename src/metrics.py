"""Evaluation metrics, CV splits and the shared train/evaluate loop.

Run directly to build the model comparison table:  python src/metrics.py
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "candidates.parquet"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
ARTIFACTS = ROOT / "artifacts"
RAW_DIRS = [ROOT, ROOT.parent / "Student folder"]

KEYS = ["trip_id", "split", "obs_idx"]
EXCLUDE = ["trip_id", "split", "obs_idx", "t", "arc_id", "label", "true_arc_id", "is_missed"]
PRED_COLS = ["trip_id", "split", "obs_idx", "arc_id", "prob", "pred_label", "model_name", "inference_ms"]

MODEL_NAMES = {
    "rf": ("Random Forest", "Bagging ensemble (trees)"),
    "lgbm": ("LightGBM (untuned)", "Gradient boosting (trees)"),
    "lgbm_tuned": ("LightGBM (Optuna-tuned)", "Gradient boosting (trees)"),
}


def load_candidates():
    df = pd.read_parquet(CANDIDATES)
    missing = set(KEYS + ["arc_id", "label"]) - set(df.columns)
    if missing:
        raise ValueError(f"candidates.parquet is missing columns: {sorted(missing)}")
    df["trip_id"] = df["trip_id"].astype(str)
    return df


def feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


def get_cv_splits(df_train, n_splits=5):
    # grouped by trip so no trip ends up in both train and validation
    return list(GroupKFold(n_splits=n_splits).split(df_train, groups=df_train["trip_id"]))


def read_ints(path):
    with open(path) as f:
        return np.array(f.read().split(), dtype=np.int64)


def load_trip_files(split, trip_id):
    """Return (arcs, route) for a trip, or None for whichever file is missing."""
    for base in RAW_DIRS:
        for tid in [trip_id, trip_id.zfill(8)]:
            folder = base / split / tid
            if folder.is_dir():
                arcs_files = list(folder.glob("*.arcs"))
                route_files = list(folder.glob("*.route"))
                arcs = read_ints(arcs_files[0]).reshape(-1, 2) if arcs_files else None
                route = read_ints(route_files[0]) if route_files else None
                return arcs, route
    return None, None


def collapse_repeats(seq):
    # several GPS points on the same arc count as one visit; later revisits are kept
    seq = np.asarray(seq)
    if len(seq) < 2:
        return seq
    keep = np.r_[True, seq[1:] != seq[:-1]]
    return seq[keep]


def lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def route_continuity(seq, arcs):
    ok = 0
    for a, b in zip(seq[:-1], seq[1:]):
        # consecutive arcs are connected if they are the same arc or to_node(a) == from_node(b)
        if a == b or (a < len(arcs) and b < len(arcs) and arcs[a, 1] == arcs[b, 0]):
            ok += 1
    return ok / (len(seq) - 1)


def sequence_f1(seq, route):
    pred, true = collapse_repeats(seq), collapse_repeats(route)
    lcs = lcs_length(pred, true)
    p = lcs / len(pred) if len(pred) else 0.0
    r = lcs / len(true) if len(true) else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate_predictions(pred_df, candidates_df, verbose=False):
    pred = pred_df.copy()
    pred["trip_id"] = pred["trip_id"].astype(str)
    cand = candidates_df.copy()
    cand["trip_id"] = cand["trip_id"].astype(str)
    cand = cand.merge(pred[["trip_id", "split"]].drop_duplicates(), on=["trip_id", "split"])

    merged = pred.drop(columns="label", errors="ignore").merge(
        cand[KEYS + ["arc_id", "label"]], on=KEYS + ["arc_id"]).reset_index(drop=True)
    y = merged["label"].to_numpy()
    p = merged["prob"].to_numpy()
    y_hat = (p >= 0.5).astype(int)

    res = {
        "n_candidates": len(merged),
        "macro_f1": f1_score(y, y_hat, average="macro", zero_division=0),
        "precision": precision_score(y, y_hat, zero_division=0),
        "recall": recall_score(y, y_hat, zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y, y_hat, labels=[0, 1]).tolist(),
    }

    # Top-1: every observation stays in the denominator. If the true arc was never
    # generated as a candidate, that observation counts as wrong.
    has_true_arc = cand.groupby(KEYS)["label"].max()
    predicted = pd.MultiIndex.from_frame(merged[KEYS].drop_duplicates())
    n_missing = len(has_true_arc.index.difference(predicted))
    if n_missing:
        raise ValueError(f"Predictions are missing {n_missing} observations")

    top = merged.loc[merged.groupby(KEYS, sort=False)["prob"].idxmax()].sort_values(KEYS)
    n_points = len(has_true_arc)
    res["n_points"] = n_points
    res["candidate_miss_rate"] = 1 - has_true_arc.mean() if n_points else np.nan
    res["top1_accuracy"] = (top["label"] == 1).sum() / n_points if n_points else np.nan
    if verbose:
        print(f"  candidate miss rate: {res['candidate_miss_rate']:.4f}")

    conts, seq_f1s = [], []
    for (trip_id, split), g in top.groupby(["trip_id", "split"]):
        seq = g.sort_values("obs_idx")["arc_id"].to_numpy()
        arcs, route = load_trip_files(split, trip_id)
        if arcs is not None and len(seq) > 1:
            conts.append(route_continuity(seq, arcs))
        if route is not None:
            seq_f1s.append(sequence_f1(seq, route))
    res["route_continuity"] = np.mean(conts) if conts else np.nan
    res["sequence_f1"] = np.mean(seq_f1s) if seq_f1s else np.nan
    res["n_trips"] = top.groupby(["trip_id", "split"]).ngroups

    # inference_ms holds the total time for the trip, repeated on each of its rows
    if "inference_ms" in pred:
        res["inference_ms_per_trip"] = pred.groupby(["trip_id", "split"])["inference_ms"].first().mean()

    return {k: (v if isinstance(v, (list, int)) else float(v)) for k, v in res.items()}


def summarize_folds(fold_metrics):
    keys = [k for k, v in fold_metrics[0].items() if not isinstance(v, list)]
    table = pd.DataFrame(fold_metrics)[keys]
    return table.mean().to_dict(), table.std(ddof=0).to_dict()


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def predict_df(model, df, feats, model_name, per_trip_timing=False):
    probs = np.zeros(len(df))
    ms = np.zeros(len(df))
    if per_trip_timing:
        for _, idx in df.groupby(["trip_id", "split"]).indices.items():
            start = time.perf_counter()
            probs[idx] = model.predict_proba(df.iloc[idx][feats])[:, 1]
            ms[idx] = (time.perf_counter() - start) * 1000
    else:
        start = time.perf_counter()
        probs = model.predict_proba(df[feats])[:, 1]
        total_ms = (time.perf_counter() - start) * 1000
        ms[:] = total_ms / df.groupby(["trip_id", "split"]).ngroups

    out = df[["trip_id", "split", "obs_idx", "arc_id"]].reset_index(drop=True)
    out["prob"] = probs
    out["pred_label"] = (probs >= 0.5).astype(int)
    out["model_name"] = model_name
    out["inference_ms"] = ms
    return out


def to_python(obj):
    return obj.item() if hasattr(obj, "item") else str(obj)


def run_experiment(model_name, family, config, make_model, fit_fn=None, artifact_name=None):
    """5-fold CV on train, refit on all of train, then predict student_test and save everything."""
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    df = load_candidates()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "student_test"].reset_index(drop=True)
    if train.empty:
        raise ValueError("No train rows in candidates.parquet")
    feats = feature_columns(train)
    print(f"[{model_name}] {len(feats)} features: {feats}")
    print(f"[{model_name}] train: {len(train)} rows, {train.trip_id.nunique()} trips | "
          f"test: {len(test)} rows, {test.trip_id.nunique()} trips")

    folds, curves = [], []
    for k, (tr_idx, va_idx) in enumerate(get_cv_splits(train)):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        model = make_model(tr["label"])
        if fit_fn:
            curves.append(fit_fn(model, tr[feats], tr["label"], va[feats], va["label"]))
        else:
            model.fit(tr[feats], tr["label"])
        m_tr = evaluate_predictions(predict_df(model, tr, feats, model_name), tr)
        m_va = evaluate_predictions(predict_df(model, va, feats, model_name), va)
        folds.append({"fold": k, "train": m_tr, "val": m_va})
        print(f"  fold {k}: train F1={m_tr['macro_f1']:.4f} LL={m_tr['log_loss']:.4f} top1={m_tr['top1_accuracy']:.4f}"
              f" | val F1={m_va['macro_f1']:.4f} LL={m_va['log_loss']:.4f} top1={m_va['top1_accuracy']:.4f}")

    tr_mean, tr_std = summarize_folds([f["train"] for f in folds])
    va_mean, va_std = summarize_folds([f["val"] for f in folds])
    print(f"  CV mean: train F1={tr_mean['macro_f1']:.4f} | val F1={va_mean['macro_f1']:.4f} "
          f"+/- {va_std['macro_f1']:.4f}")

    final = make_model(train["label"])
    start = time.perf_counter()
    final.fit(train[feats], train["label"])
    fit_seconds = time.perf_counter() - start
    joblib.dump({"model": final, "features": feats}, ARTIFACTS / (artifact_name or f"{model_name}_model.joblib"))

    test_metrics = None
    if len(test):
        preds = predict_df(final, test, feats, model_name, per_trip_timing=True)
        preds[PRED_COLS].to_parquet(RESULTS / f"predictions_{model_name}.parquet", index=False)
        test_metrics = evaluate_predictions(preds, test, verbose=True)
        plot_confusion(test_metrics["confusion_matrix"], f"{model_name} (student_test)",
                       FIGURES / f"cm_{model_name}.png")
        print(f"  student_test: F1={test_metrics['macro_f1']:.4f} top1={test_metrics['top1_accuracy']:.4f} "
              f"LL={test_metrics['log_loss']:.4f} continuity={test_metrics['route_continuity']:.4f}")

    result = {
        "model": model_name,
        "display_name": MODEL_NAMES.get(model_name, (model_name,))[0],
        "family": family,
        "config": config,
        "features": feats,
        "n_train_rows": len(train),
        "n_train_trips": train.trip_id.nunique(),
        "n_test_rows": len(test),
        "n_test_trips": test.trip_id.nunique(),
        "pos_rate_train": train["label"].mean(),
        "final_fit_seconds": fit_seconds,
        "cv": {"folds": folds, "train_mean": tr_mean, "train_std": tr_std, "val_mean": va_mean, "val_std": va_std},
        "student_test": test_metrics,
        "fold_extras": curves or None,
    }
    with open(RESULTS / f"metrics_{model_name}.json", "w") as f:
        json.dump(result, f, indent=2, default=to_python)
    return result, final


def bar_chart(names, values, title, path):
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 3.5))
    ax.bar(names, values, color="#4C72B0")
    ax.set_ylabel(title)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    cand = load_candidates()
    rows = []
    for path in sorted(RESULTS.glob("predictions_*.parquet")):
        name = path.stem.replace("predictions_", "")
        metrics_path = RESULTS / f"metrics_{name}.json"
        meta = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        display, family = MODEL_NAMES.get(name, (meta.get("display_name", name), meta.get("family", "")))

        st = evaluate_predictions(pd.read_parquet(path), cand)
        plot_confusion(st["confusion_matrix"], f"{display} (student_test)", FIGURES / f"cm_{name}.png")

        config = meta.get("config") or {}
        config_str = "; ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in config.items())
        rows.append({
            "Model": display,
            "Algorithm Family": meta.get("family", family),
            "Key Configuration": config_str,
            "CV Macro F1-score": meta.get("cv", {}).get("val_mean", {}).get("macro_f1", np.nan),
            "Top-1 Accuracy": st["top1_accuracy"],
            "Route Continuity": st["route_continuity"],
            "Student-Test Macro F1-score": st["macro_f1"],
            "Log Loss": st["log_loss"],
            "Inference Time": st.get("inference_ms_per_trip", np.nan),
        })

    if not rows:
        print("No results/predictions_*.parquet found.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "model_comparison_summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False))

    bar_chart(summary["Model"], summary["CV Macro F1-score"], "CV Macro F1-score", FIGURES / "bar_cv_macro_f1.png")
    bar_chart(summary["Model"], summary["Log Loss"], "Log Loss", FIGURES / "bar_log_loss.png")

    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(1.8 * len(summary) + 2, 3.5))
    ax.bar(x - 0.19, summary["Top-1 Accuracy"], 0.38, label="Top-1 Accuracy", color="#4C72B0")
    ax.bar(x + 0.19, summary["Student-Test Macro F1-score"], 0.38, label="Macro F1", color="#DD8452")
    ax.set_xticks(x, summary["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Student-test: Top-1 Accuracy & Macro F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "bar_student_test_top1_f1.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
