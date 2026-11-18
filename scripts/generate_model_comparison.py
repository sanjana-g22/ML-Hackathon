"""
Generate comprehensive model comparison tables and charts for all 5 models:
1. Logistic Regression
2. KNN
3. Random Forest
4. LightGBM
5. MLP (Model 5)

Explicitly generates:
- Fair comparison on the IDENTICAL 3-trip common subset (00000000, 00000007, 00000009)
- Full student-test comparison (10 trips for Models 1-4, 3 trips for MLP)
- Pointwise Top-1 Accuracy vs Sequence F1
- Route Continuity (%) and Unresolved Discontinuities (Gaps)
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COMMON_TRIPS = ["00000000", "00000007", "00000009"]

models_info = [
    {"id": "logistic", "name": "Logistic Regression", "family": "Linear Classifier", "train_trips": 80, "eval_file": "hmm_evaluation_logistic.csv", "summary_file": "hmm_summary_logistic.json"},
    {"id": "knn", "name": "K-Nearest Neighbors", "family": "Instance-based", "train_trips": 80, "eval_file": "hmm_evaluation_knn.csv", "summary_file": "hmm_summary_knn.json"},
    {"id": "rf", "name": "Random Forest", "family": "Bagging Trees", "train_trips": 80, "eval_file": "hmm_evaluation_rf.csv", "summary_file": "hmm_summary_rf.json"},
    {"id": "lgbm", "name": "LightGBM", "family": "Gradient Boosting", "train_trips": 80, "eval_file": "hmm_evaluation_lgbm.csv", "summary_file": "hmm_summary_lgbm.json"},
    {"id": "mlp", "name": "Model 5 MLP", "family": "Neural Network", "train_trips": 3, "eval_file": "hmm_evaluation_mlp.csv", "summary_file": "hmm_summary_mlp.json"},
]

# --------------------------------------------------------------------------- #
# 1. Fair Comparison on Common 3-Trip Subset
# --------------------------------------------------------------------------- #
common_rows = []
for m in models_info:
    path = os.path.join(RESULTS_DIR, m["eval_file"])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["trip_id"] = df["trip_id"].astype(str).str.zfill(8)
        sub = df[df["trip_id"].isin(COMMON_TRIPS)]
        b_seq = sub["baseline_sequence_f1"].mean()
        h_seq = sub["hmm_sequence_f1"].mean()
        b_cont = sub["baseline_route_continuity"].mean() * 100
        h_cont = sub["hmm_route_continuity"].mean() * 100
        b_gaps = int(sub["baseline_gaps"].sum())
        h_gaps = int(sub["hmm_gaps"].sum())
        t_ms = sub["execution_time_ms"].mean()
    else:
        b_seq, h_seq, b_cont, h_cont, b_gaps, h_gaps, t_ms = 0, 0, 0, 0, 0, 0, 0

    common_rows.append({
        "Model": m["name"],
        "Algorithm Family": m["family"],
        "Training Trips": m["train_trips"],
        "Evaluated Trips": 3,
        "Trip IDs": "00000000, 00000007, 00000009",
        "Baseline Sequence F1": round(b_seq, 4),
        "HMM Sequence F1": round(h_seq, 4),
        "Sequence F1 Gain (x)": round(h_seq / max(1e-4, b_seq), 2),
        "Baseline Route Continuity (%)": round(b_cont, 2),
        "HMM Route Continuity (%)": round(h_cont, 2),
        "Baseline Gaps": b_gaps,
        "HMM Gaps": h_gaps,
        "Gap Reduction (%)": round(100.0 * (b_gaps - h_gaps) / max(1, b_gaps), 1),
        "HMM Latency (ms/trip)": round(t_ms, 1)
    })

common_df = pd.DataFrame(common_rows)
common_csv = os.path.join(RESULTS_DIR, "hmm_model_comparison_common_subset.csv")
common_df.to_csv(common_csv, index=False)
print(f"Saved {common_csv}")

# --------------------------------------------------------------------------- #
# 2. Comprehensive Comparison Table (Explicit Evaluation Scope)
# --------------------------------------------------------------------------- #
all_rows = []
for m in models_info:
    path = os.path.join(RESULTS_DIR, m["summary_file"])
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        n_eval = data.get("n_trips", 0)
        n_cands = data.get("total_observations", 0)
        b_seq = data.get("baseline_mean_sequence_f1", 0.0)
        h_seq = data.get("hmm_mean_sequence_f1", 0.0)
        b_cont = data.get("baseline_mean_route_continuity", 0.0) * 100
        h_cont = data.get("hmm_mean_route_continuity", 0.0) * 100
        b_gaps = int(data.get("baseline_total_gaps", 0))
        h_gaps = int(data.get("hmm_total_gaps", 0))
        t_ms = data.get("average_execution_time_ms_per_trip", 0.0)
    else:
        n_eval, n_cands, b_seq, h_seq, b_cont, h_cont, b_gaps, h_gaps, t_ms = 0, 0, 0, 0, 0, 0, 0, 0, 0

    all_rows.append({
        "Model": m["name"],
        "Algorithm Family": m["family"],
        "Training Trips": m["train_trips"],
        "Test Trips Evaluated": n_eval,
        "Evaluation Scope Note": f"{n_eval} of 10 student trips" if n_eval < 10 else "Full 10 student trips",
        "Baseline Sequence F1": round(b_seq, 4),
        "HMM Sequence F1": round(h_seq, 4),
        "Sequence F1 Gain (x)": round(h_seq / max(1e-4, b_seq), 2),
        "Baseline Route Continuity (%)": round(b_cont, 2),
        "HMM Route Continuity (%)": round(h_cont, 2),
        "Baseline Gaps": b_gaps,
        "HMM Gaps": h_gaps,
        "Gap Reduction (%)": round(100.0 * (b_gaps - h_gaps) / max(1, b_gaps), 1),
        "HMM Latency (ms/trip)": round(t_ms, 1)
    })

all_df = pd.DataFrame(all_rows)
all_csv = os.path.join(RESULTS_DIR, "hmm_model_comparison_summary.csv")
all_df.to_csv(all_csv, index=False)
print(f"Saved {all_csv}")

# --------------------------------------------------------------------------- #
# 3. Figure: Sequence F1 on Common 3-Trip Subset
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(common_df))
w = 0.35
ax.bar(x - w/2, common_df["Baseline Sequence F1"], w, label="Baseline Top-1 (No HMM)", color="#e41a1c", alpha=0.85)
ax.bar(x + w/2, common_df["HMM Sequence F1"], w, label="HMM + Viterbi (Reconstructed)", color="#2ca02c", alpha=0.85)
ax.set_ylabel("Sequence F1 Score vs Reference Route", fontsize=11, fontweight="bold")
ax.set_title("Fair Comparison on Identical 3-Trip Common Subset\n(Trips 00000000, 00000007, 00000009)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(common_df["Model"], fontsize=10, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.set_ylim(0, 1.05)
for i in range(len(common_df)):
    ax.text(i - w/2, common_df["Baseline Sequence F1"].iloc[i] + 0.02, f"{common_df['Baseline Sequence F1'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, common_df["HMM Sequence F1"].iloc[i] + 0.02, f"{common_df['HMM Sequence F1'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
p1 = os.path.join(FIGURES_DIR, "hmm_all_models_pointwise_vs_sequence_f1.png")
plt.savefig(p1, dpi=300)
plt.close()
print(f"Saved {p1}")
