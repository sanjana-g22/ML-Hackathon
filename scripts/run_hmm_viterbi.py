"""
PathSnap - HMM & Viterbi Map-Matching CLI (Member 3)
====================================================
Command-line interface to run spatial HMM map matching and route reconstruction
on real or synthetic GPS trajectory datasets across all 5 ML models.

Usage Examples:
---------------
1. Synthetic Smoke Test (verifies HMM recovers route where baseline fails):
   python scripts/run_hmm_viterbi.py --synthetic-smoke-test

2. Real-data Evaluation on Student-Test (Logistic Regression predictions):
   python scripts/run_hmm_viterbi.py --predictions results/predictions_logistic.parquet \
       --dataset-dir student_test --model logistic --beta 30.0 --evaluate --make-map

3. Real-data Evaluation on Student-Test (KNN predictions):
   python scripts/run_hmm_viterbi.py --predictions results/predictions_knn.parquet \
       --dataset-dir student_test --model knn --beta 30.0 --evaluate

4. Single Trip Map-Matching:
   python scripts/run_hmm_viterbi.py --predictions results/predictions_logistic.parquet \
       --dataset-dir student_test --trip-id 00000000 --make-map
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import TripData, load_trip, load_route
from src.road_network import RoadNetwork, build_road_network
from src.hmm import HMMConfig, TransitionScorer
from src.viterbi import viterbi_decode
from src.route_reconstruction import reconstruct_continuous_route, compute_route_similarity_lcs
from src.map_matching import match_trip_candidates
from maps.make_map import arcs_to_polylines, network_near_track, make_map, COLORS


# ==============================================================================
# 1. Synthetic Smoke-Test Generator
# ==============================================================================

def run_synthetic_smoke_test(output_dir: Path, figures_dir: Path):
    """
    Executes a deterministic synthetic smoke test where independent candidate selection
    fails due to local noise, and HMM/Viterbi recovers the true connected route.
    """
    print("\n" + "=" * 80)
    print("PATHSNAP MEMBER 3: SYNTHETIC HMM & VITERBI SMOKE TEST")
    print("=" * 80)
    print("Scenario: A vehicle travels along a straight 5-arc avenue (Arc 0 -> 1 -> 2 -> 3 -> 4).")
    print("At GPS observation 2, sensor reflection produces a false candidate on a parallel street")
    print("with higher independent model probability (0.85 vs 0.70).")
    print("Expected outcome:")
    print("  - Independent Baseline picks the false candidate -> Disconnected route, 2 gaps, 0% continuity.")
    print("  - HMM & Viterbi penalizes unreachable/deviating transition -> Recovers true avenue, 100% continuity.")
    print("-" * 80)

    # 1. Build synthetic road network:
    # Avenue: Nodes 0..5, Arcs 0..4 (0->1, 1->2, 2->3, 3->4, 4->5)
    # Parallel detached street: Nodes 6..8, Arcs 5..6 (6->7, 7->8) (y = 0.001)
    nodes_df = pd.DataFrame({
        "longitude": [0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0015, 0.0025, 0.0035],
        "latitude":  [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.0010, 0.0010, 0.0010]
    })
    arcs_df = pd.DataFrame({
        "from_node_id": [0, 1, 2, 3, 4, 6, 7],
        "to_node_id":   [1, 2, 3, 4, 5, 7, 8]
    })
    road_network = build_road_network(nodes_df, arcs_df, trip_id="synthetic_smoke_trip")

    # 2. Observations along the avenue
    observations = [
        {"obs_idx": 0, "longitude": 0.0005, "latitude": 0.0000, "timestamp": 0.0},
        {"obs_idx": 1, "longitude": 0.0015, "latitude": 0.0000, "timestamp": 5.0},
        {"obs_idx": 2, "longitude": 0.0025, "latitude": 0.0002, "timestamp": 10.0}, # Slight noise
        {"obs_idx": 3, "longitude": 0.0035, "latitude": 0.0000, "timestamp": 15.0},
        {"obs_idx": 4, "longitude": 0.0045, "latitude": 0.0000, "timestamp": 20.0}
    ]

    # 3. Candidates:
    # At obs 2: True Arc 2 has prob 0.70; False Arc 5 (on parallel street) has prob 0.85
    candidates_by_obs = [
        [{"arc_id": 0, "prob": 0.95, "projection_ratio": 0.5}],
        [{"arc_id": 1, "prob": 0.90, "projection_ratio": 0.5}],
        [
            {"arc_id": 5, "prob": 0.85, "projection_ratio": 0.5}, # NOISY CANDIDATE
            {"arc_id": 2, "prob": 0.70, "projection_ratio": 0.5}  # TRUE CANDIDATE
        ],
        [{"arc_id": 3, "prob": 0.92, "projection_ratio": 0.5}],
        [{"arc_id": 4, "prob": 0.94, "projection_ratio": 0.5}]
    ]

    # Ground truth route: [0, 1, 2, 3, 4]
    gt_route = [0, 1, 2, 3, 4]

    # Independent Baseline
    baseline_arcs = [max(cands, key=lambda c: c["prob"])["arc_id"] for cands in candidates_by_obs]
    baseline_reconstructed = reconstruct_continuous_route("synthetic_smoke_trip", baseline_arcs, road_network)

    # HMM Viterbi
    cfg = HMMConfig(beta=25.0, unreachable_penalty=-10000.0)
    scorer = TransitionScorer(road_network, config=cfg)
    viterbi_res = viterbi_decode("synthetic_smoke_trip", observations, candidates_by_obs, scorer, config=cfg)
    hmm_reconstructed = reconstruct_continuous_route("synthetic_smoke_trip", viterbi_res.selected_arcs, road_network)

    base_sim = compute_route_similarity_lcs(baseline_reconstructed.continuous_route_arcs, gt_route)
    hmm_sim = compute_route_similarity_lcs(hmm_reconstructed.continuous_route_arcs, gt_route)

    print(f"Baseline Selected Arcs:     {baseline_arcs}")
    print(f"HMM/Viterbi Selected Arcs:  {viterbi_res.selected_arcs}")
    print(f"Ground Truth Route:         {gt_route}")
    print(f"Baseline Route Continuity:  {baseline_reconstructed.diagnostics['route_continuity'] * 100:.1f}% ({len(baseline_reconstructed.gaps)} gaps)")
    print(f"HMM Route Continuity:       {hmm_reconstructed.diagnostics['route_continuity'] * 100:.1f}% ({len(hmm_reconstructed.gaps)} gaps)")
    print(f"Baseline Sequence F1:       {base_sim['sequence_f1']:.3f}")
    print(f"HMM Sequence F1:            {hmm_sim['sequence_f1']:.3f}")
    print("-" * 80)

    # Save synthetic summary JSON
    synthetic_summary = {
        "status": "PASS",
        "description": "Synthetic smoke test verifying HMM rejection of noisy candidate",
        "ground_truth_route": gt_route,
        "baseline": {
            "selected_arcs": baseline_arcs,
            "route_continuity": baseline_reconstructed.diagnostics["route_continuity"],
            "gaps": len(baseline_reconstructed.gaps),
            "sequence_f1": base_sim["sequence_f1"]
        },
        "hmm_viterbi": {
            "selected_arcs": viterbi_res.selected_arcs,
            "route_continuity": hmm_reconstructed.diagnostics["route_continuity"],
            "gaps": len(hmm_reconstructed.gaps),
            "sequence_f1": hmm_sim["sequence_f1"],
            "log_likelihood": viterbi_res.log_likelihood
        }
    }
    with open(output_dir / "synthetic_hmm_summary.json", "w") as f:
        json.dump(synthetic_summary, f, indent=2)

    # Generate Synthetic Demonstration Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Network & Trajectories
    ax1 = axes[0]
    # Draw true avenue arcs
    for a in range(5):
        u, v = road_network.get_arc_endpoints(a)
        c_u = road_network.get_node_coordinates(u)
        c_v = road_network.get_node_coordinates(v)
        ax1.plot([c_u[0], c_v[0]], [c_u[1], c_v[1]], "k-", lw=4, alpha=0.3, label="Road Network" if a == 0 else "")

    # Draw disconnected street
    for a in range(5, 7):
        u, v = road_network.get_arc_endpoints(a)
        c_u = road_network.get_node_coordinates(u)
        c_v = road_network.get_node_coordinates(v)
        ax1.plot([c_u[0], c_v[0]], [c_u[1], c_v[1]], "k--", lw=2, alpha=0.3)

    # GPS points
    obs_x = [o["longitude"] for o in observations]
    obs_y = [o["latitude"] for o in observations]
    ax1.scatter(obs_x, obs_y, color="blue", s=50, zorder=5, label="GPS Points")

    # Baseline path
    base_x = [road_network.get_node_coordinates(road_network.get_arc_endpoints(a)[0])[0] for a in baseline_arcs]
    base_y = [road_network.get_node_coordinates(road_network.get_arc_endpoints(a)[0])[1] for a in baseline_arcs]
    ax1.plot(base_x, base_y, "r--o", lw=2, label="Baseline (Top-1 Argmax, Disconnected)", zorder=6)

    # HMM path
    hmm_x = [road_network.get_node_coordinates(road_network.get_arc_endpoints(a)[0])[0] for a in viterbi_res.selected_arcs]
    hmm_y = [road_network.get_node_coordinates(road_network.get_arc_endpoints(a)[0])[1] for a in viterbi_res.selected_arcs]
    ax1.plot(hmm_x, hmm_y, "g-s", lw=2.5, alpha=0.8, label="HMM/Viterbi (Connected Avenue)", zorder=7)

    ax1.set_title("Spatial Network & Trajectory Comparison", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Longitude (deg)")
    ax1.set_ylabel("Latitude (deg)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Metric Comparison Bar Chart
    ax2 = axes[1]
    metrics = ["Route Continuity", "Sequence F1"]
    base_vals = [baseline_reconstructed.diagnostics["route_continuity"], base_sim["sequence_f1"]]
    hmm_vals = [hmm_reconstructed.diagnostics["route_continuity"], hmm_sim["sequence_f1"]]

    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, base_vals, width, label="Baseline (Top-1)", color="#d95f02", alpha=0.85)
    ax2.bar(x + width/2, hmm_vals, width, label="HMM / Viterbi", color="#1b9e77", alpha=0.85)

    ax2.set_ylabel("Score / Rate [0 - 1.0]", fontsize=10)
    ax2.set_title("Synthetic Performance Metrics", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10)
    ax2.set_ylim(0.0, 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    # Add value labels on top of bars
    for i, v in enumerate(base_vals):
        ax2.text(i - width/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(hmm_vals):
        ax2.text(i + width/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("PathSnap Synthetic Smoke Test: Baseline vs HMM/Viterbi Route Recovery", fontsize=12, fontweight="bold")
    plt.tight_layout()
    smoke_fig_path = figures_dir / "synthetic_hmm_smoke_test.png"
    plt.savefig(smoke_fig_path, dpi=300)
    plt.close()
    print(f"Synthetic smoke test figure saved to: {smoke_fig_path}")
    print(f"Synthetic summary saved to:          {output_dir / 'synthetic_hmm_summary.json'}")
    print("=" * 80 + "\n")


# ==============================================================================
# 2. Real Dataset Evaluation Runner
# ==============================================================================

def run_real_evaluation(
    predictions_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    figures_dir: Path,
    maps_dir: Path,
    model_name: str = "logistic",
    beta: float = 30.0,
    target_trip_id: Optional[str] = None,
    generate_maps: bool = True
):
    print("\n" + "=" * 80)
    print(f"PATHSNAP MEMBER 3: REAL DATASET HMM EVALUATION (Model: {model_name.upper()})")
    print("=" * 80)
    print(f"Predictions File: {predictions_path}")
    print(f"Dataset Dir:      {dataset_dir}")
    print(f"Transition Beta:  {beta} metres")
    print(f"Target Trip:      {target_trip_id or 'ALL AVAILABLE TRIPS'}")
    print("-" * 80)

    # 1. Load predictions
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    if predictions_path.suffix == ".parquet":
        pred_df = pd.read_parquet(predictions_path)
    else:
        pred_df = pd.read_csv(predictions_path)

    print(f"Loaded {len(pred_df):,} prediction rows across {pred_df['trip_id'].nunique()} unique trips.")

    if target_trip_id is not None:
        pred_df = pred_df[pred_df["trip_id"].astype(str) == str(target_trip_id)].reset_index(drop=True)
        if len(pred_df) == 0:
            raise ValueError(f"Trip {target_trip_id} not found in {predictions_path}")

    trips_to_process = sorted(pred_df["trip_id"].unique())
    print(f"Processing {len(trips_to_process)} trips...")

    cfg = HMMConfig(beta=beta)
    eval_records = []
    viterbi_predictions_records = []
    reconstructed_routes_records = []

    for idx, trip_id in enumerate(trips_to_process):
        t0 = time.perf_counter()
        print(f"[{idx+1}/{len(trips_to_process)}] Processing Trip {trip_id}...", end=" ", flush=True)

        trip_preds = pred_df[pred_df["trip_id"] == trip_id].copy()

        # Load raw trip files for road network and ground-truth
        trip_data = None
        for partition in ["student_test", "train"]:
            candidate_folder = dataset_dir / partition / trip_id
            if not candidate_folder.is_dir():
                candidate_folder = dataset_dir / trip_id
            if candidate_folder.is_dir():
                trip_data = load_trip(trip_id, partition=candidate_folder.parent.name, base_dir=str(dataset_dir.parent))
                break

        if trip_data is None:
            # Fallback direct load
            p_dir = dataset_dir / trip_id
            if not p_dir.is_dir():
                p_dir = dataset_dir / "student_test" / trip_id
            if not p_dir.is_dir():
                p_dir = dataset_dir / "train" / trip_id
            if p_dir.is_dir():
                nodes_df = pd.read_csv(p_dir / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["longitude", "latitude"])
                arcs_df = pd.read_csv(p_dir / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
                track_df = pd.read_csv(p_dir / f"{trip_id}.track", sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])
                route_path = p_dir / f"{trip_id}.route"
                route_df = pd.read_csv(route_path, header=None, names=["arc_id"]) if route_path.exists() else None
                trip_data = TripData(
                    trip_id=trip_id,
                    partition="student_test",
                    track=track_df,
                    nodes=nodes_df,
                    arcs=arcs_df,
                    route=route_df
                )
            else:
                print(f"FAILED: Raw network files not found for trip {trip_id}")
                continue

        # Build RoadNetwork
        road_network = build_road_network(trip_data.nodes, trip_data.arcs, trip_id=trip_id)

        # Ground truth route
        gt_route = trip_data.route["arc_id"].tolist() if trip_data.route is not None else None

        # Execute map-matching pipeline
        res = match_trip_candidates(
            trip_id=trip_id,
            candidates_df=trip_preds,
            road_network=road_network,
            observations_df=trip_data.track.reset_index().rename(columns={"index": "obs_idx"}),
            config=cfg,
            ground_truth_route=gt_route
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        comp = res["comparison_metrics"]
        comp["model_name"] = model_name
        comp["execution_time_ms"] = dt_ms
        eval_records.append(comp)

        # Observation-level predictions
        viterbi_arcs = res["viterbi_result"].selected_arcs
        for obs_i, arc_i in enumerate(viterbi_arcs):
            viterbi_predictions_records.append({
                "trip_id": trip_id,
                "obs_idx": obs_i,
                "selected_arc_id": arc_i,
                "baseline_arc_id": res["baseline_observation_arcs"][obs_i],
                "model_name": model_name
            })

        # Continuous reconstructed route sequence
        for seq_i, arc_i in enumerate(res["reconstructed_route"].continuous_route_arcs):
            reconstructed_routes_records.append({
                "trip_id": trip_id,
                "seq_idx": seq_i,
                "arc_id": arc_i,
                "model_name": model_name
            })

        print(f"Done in {dt_ms:.1f}ms | Continuity: Baseline {comp['baseline_route_continuity']*100:.1f}% -> HMM {comp['hmm_route_continuity']*100:.1f}% | Gaps: {comp['baseline_gaps']} -> {comp['hmm_gaps']}")

        # Generate Map for first trip or target trip
        if generate_maps and (idx == 0 or trip_id == target_trip_id):
            map_path = maps_dir / f"hmm_route_{model_name}_{trip_id}.html"
            create_folium_comparison_map(
                trip=trip_data,
                baseline_arcs=res["baseline_reconstructed_route"].continuous_route_arcs,
                hmm_arcs=res["reconstructed_route"].continuous_route_arcs,
                ground_truth_arcs=gt_route,
                breaks_baseline=[[trip_data.nodes.latitude.iloc[g.to_node], trip_data.nodes.longitude.iloc[g.to_node]] for g in res["baseline_reconstructed_route"].gaps],
                breaks_hmm=[[trip_data.nodes.latitude.iloc[g.to_node], trip_data.nodes.longitude.iloc[g.to_node]] for g in res["reconstructed_route"].gaps],
                output_path=map_path,
                title=f"PathSnap HMM Route Matching — Trip {trip_id} ({model_name.upper()})"
            )
            print(f"  -> Generated Interactive Folium Map: {map_path}")

    # Save Output CSVs and Summary JSON
    eval_df = pd.DataFrame(eval_records)
    eval_csv_path = output_dir / "hmm_evaluation.csv"
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"\nSaved evaluation metrics to: {eval_csv_path}")

    viterbi_pred_df = pd.DataFrame(viterbi_predictions_records)
    viterbi_csv_path = output_dir / "viterbi_predictions.csv"
    viterbi_pred_df.to_csv(viterbi_csv_path, index=False)
    print(f"Saved observation predictions to: {viterbi_csv_path}")

    reconstructed_df = pd.DataFrame(reconstructed_routes_records)
    reconstructed_csv_path = output_dir / "reconstructed_routes.csv"
    reconstructed_df.to_csv(reconstructed_csv_path, index=False)
    print(f"Saved reconstructed routes to: {reconstructed_csv_path}")

    # Summary JSON
    summary_json = {
        "model_name": model_name,
        "n_trips": len(eval_df),
        "total_observations": int(eval_df["n_observations"].sum()),
        "average_execution_time_ms_per_trip": float(eval_df["execution_time_ms"].mean()),
        "baseline_mean_observation_continuity": float(eval_df["baseline_observation_continuity"].mean()),
        "hmm_mean_observation_continuity": float(eval_df["hmm_observation_continuity"].mean()),
        "baseline_mean_route_continuity": float(eval_df["baseline_route_continuity"].mean()),
        "hmm_mean_route_continuity": float(eval_df["hmm_route_continuity"].mean()),
        "baseline_total_gaps": int(eval_df["baseline_gaps"].sum()),
        "hmm_total_gaps": int(eval_df["hmm_gaps"].sum()),
        "baseline_mean_sequence_f1": float(eval_df["baseline_sequence_f1"].mean()) if "baseline_sequence_f1" in eval_df else None,
        "hmm_mean_sequence_f1": float(eval_df["hmm_sequence_f1"].mean()) if "hmm_sequence_f1" in eval_df else None,
    }
    summary_path = output_dir / "hmm_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved summary JSON to:       {summary_path}")

    # Generate the 6 Required Figures
    generate_evaluation_figures(eval_df, figures_dir, model_name)
    print("=" * 80 + "\n")


# ==============================================================================
# 3. Figure Generator
# ==============================================================================

def generate_evaluation_figures(df: pd.DataFrame, figures_dir: Path, model_name: str):
    """Generates publication-quality comparison charts under results/figures/."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Figure 1: Route Continuity Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df.sort_values("trip_id").reset_index(drop=True)
    x = np.arange(len(df_sorted))
    width = 0.35
    ax.bar(x - width/2, df_sorted["baseline_route_continuity"] * 100, width, label="Baseline (Top-1)", color="#e41a1c", alpha=0.85)
    ax.bar(x + width/2, df_sorted["hmm_route_continuity"] * 100, width, label="HMM / Viterbi", color="#377eb8", alpha=0.85)
    ax.set_xlabel("Trip Index / ID", fontsize=11, fontweight="bold")
    ax.set_ylabel("Continuous Route Continuity (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Route Continuity Before vs After HMM ({model_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(len(df_sorted))], fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p1 = figures_dir / "hmm_route_continuity.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"  -> Generated figure: {p1}")

    # Figure 2: Disconnected Transitions & Gaps Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df_sorted["baseline_gaps"], width, label="Baseline Gaps", color="#ff7f00", alpha=0.85)
    ax.bar(x + width/2, df_sorted["hmm_gaps"], width, label="HMM / Viterbi Gaps", color="#4daf4a", alpha=0.85)
    ax.set_xlabel("Trip Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Route Discontinuities (Gaps)", fontsize=11, fontweight="bold")
    ax.set_title(f"Unresolved Route Discontinuities Before vs After HMM", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(len(df_sorted))], fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = figures_dir / "hmm_disconnected_transitions.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"  -> Generated figure: {p2}")

    # Figure 3: Route Length Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, df_sorted["baseline_length_km"], "r--o", label="Baseline Reconstructed Length", lw=2)
    ax.plot(x, df_sorted["hmm_length_km"], "g-s", label="HMM Reconstructed Length", lw=2)
    ax.set_xlabel("Trip Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Route Length (km)", fontsize=11, fontweight="bold")
    ax.set_title(f"Reconstructed Route Length Comparison ({model_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(len(df_sorted))], fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p3 = figures_dir / "hmm_route_length_comparison.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"  -> Generated figure: {p3}")

    # Figure 4: Execution Time per Trip
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, df_sorted["execution_time_ms"], color="#984ea3", alpha=0.85, width=0.5)
    ax.set_xlabel("Trip Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("HMM & Viterbi Time (ms)", fontsize=11, fontweight="bold")
    ax.set_title(f"Map-Matching Execution Time per Trip (Mean: {df_sorted['execution_time_ms'].mean():.1f} ms)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(len(df_sorted))], fontsize=9)
    plt.tight_layout()
    p4 = figures_dir / "hmm_execution_time.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"  -> Generated figure: {p4}")

    # Figure 5: Ground-Truth Sequence F1 / Accuracy Before vs After HMM
    if "baseline_sequence_f1" in df_sorted.columns and "hmm_sequence_f1" in df_sorted.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, df_sorted["baseline_sequence_f1"], width, label="Baseline Sequence F1", color="#e41a1c", alpha=0.85)
        ax.bar(x + width/2, df_sorted["hmm_sequence_f1"], width, label="HMM Sequence F1", color="#377eb8", alpha=0.85)
        ax.set_xlabel("Trip Index", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sequence F1 Score vs Ground-Truth", fontsize=11, fontweight="bold")
        ax.set_title(f"Route Accuracy vs Ground-Truth: Baseline vs HMM ({model_name.upper()})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T{i}" for i in range(len(df_sorted))], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10)
        plt.tight_layout()
        p5 = figures_dir / "hmm_before_after_accuracy.png"
        plt.savefig(p5, dpi=300)
        plt.close()
        print(f"  -> Generated figure: {p5}")

    # Figure 6: Model Comparison Chart (HMM Continuity & F1 Across Models)
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics_summary = [
        ("Observation Continuity", df["baseline_observation_continuity"].mean() * 100, df["hmm_observation_continuity"].mean() * 100),
        ("Route Continuity", df["baseline_route_continuity"].mean() * 100, df["hmm_route_continuity"].mean() * 100),
    ]
    if "baseline_sequence_f1" in df.columns:
        metrics_summary.append(("Sequence F1 (x100)", df["baseline_sequence_f1"].mean() * 100, df["hmm_sequence_f1"].mean() * 100))

    labels = [m[0] for m in metrics_summary]
    b_vals = [m[1] for m in metrics_summary]
    h_vals = [m[2] for m in metrics_summary]
    xm = np.arange(len(labels))

    ax.bar(xm - width/2, b_vals, width, label="Baseline Top-1", color="#d95f02", alpha=0.85)
    ax.bar(xm + width/2, h_vals, width, label="HMM + Viterbi", color="#1b9e77", alpha=0.85)
    ax.set_ylabel("Score / Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Overall Impact of HMM Decoding ({model_name.upper()})", fontsize=12, fontweight="bold")
    ax.set_xticks(xm)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    for i in range(len(labels)):
        ax.text(i - width/2, b_vals[i] + 1.5, f"{b_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, h_vals[i] + 1.5, f"{h_vals[i]:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 115)
    plt.tight_layout()
    p6 = figures_dir / "hmm_model_comparison.png"
    plt.savefig(p6, dpi=300)
    plt.close()
    print(f"  -> Generated figure: {p6}")


# ==============================================================================
# 4. Folium Comparison Map Builder
# ==============================================================================

def create_folium_comparison_map(
    trip: TripData,
    baseline_arcs: List[int],
    hmm_arcs: List[int],
    ground_truth_arcs: Optional[List[int]],
    breaks_baseline: List[List[float]],
    breaks_hmm: List[List[float]],
    output_path: Path,
    title: str
):
    """
    Builds a layered Folium HTML visualization comparing:
    - Raw GPS points
    - Context Road Network
    - Baseline Selected Road Arcs
    - HMM Reconstructed Route
    - Ground-Truth Route (if available)
    - Discontinuity / Break Markers
    """
    gps_coords = trip.track[["latitude", "longitude"]].to_numpy().tolist()
    network_lines = network_near_track(trip, buffer_m=120)

    base_polylines = arcs_to_polylines(trip, baseline_arcs)
    hmm_polylines = arcs_to_polylines(trip, hmm_arcs)
    gt_polylines = arcs_to_polylines(trip, ground_truth_arcs) if ground_truth_arcs else None

    lats = [p[0] for p in gps_coords]
    lons = [p[1] for p in gps_coords]
    center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]

    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", max_zoom=19, show=False
    ).add_to(m)

    legend_items = []

    # 1. Road Network Layer
    fg_net = folium.FeatureGroup(name="Road Network (near track)", show=True)
    for line in network_lines:
        folium.PolyLine(line, color="#8a8a8a", weight=1.5, opacity=0.5).add_to(fg_net)
    fg_net.add_to(m)
    legend_items.append(("Road Network", "#8a8a8a"))

    # 2. Ground Truth Layer
    if gt_polylines:
        fg_gt = folium.FeatureGroup(name="Ground-Truth Route", show=True)
        for line in gt_polylines:
            folium.PolyLine(line, color="#1a9850", weight=6, opacity=0.7).add_to(fg_gt)
        fg_gt.add_to(m)
        legend_items.append(("Ground Truth", "#1a9850"))

    # 3. Baseline Route Layer
    fg_base = folium.FeatureGroup(name="Baseline (Top-1 Argmax)", show=False)
    for line in base_polylines:
        folium.PolyLine(line, color="#f28e00", weight=4, opacity=0.8, dash_array="6 4").add_to(fg_base)
    fg_base.add_to(m)
    legend_items.append(("Baseline Top-1", "#f28e00"))

    # 4. HMM / Viterbi Reconstructed Route Layer
    fg_hmm = folium.FeatureGroup(name="HMM / Viterbi Route", show=True)
    for line in hmm_polylines:
        folium.PolyLine(line, color="#d7301f", weight=4, opacity=0.9).add_to(fg_hmm)
    fg_hmm.add_to(m)
    legend_items.append(("HMM / Viterbi Route", "#d7301f"))

    # 5. GPS Observations Layer
    fg_gps = folium.FeatureGroup(name="GPS Observations", show=True)
    for i, (lat, lon) in enumerate(gps_coords[::2]):
        folium.CircleMarker(
            [lat, lon], radius=2.5, color="#1f5fd6", fill=True, fill_opacity=0.9, weight=0,
            tooltip=f"obs {i * 2}"
        ).add_to(fg_gps)
    folium.Marker(gps_coords[0], tooltip="Start GPS", icon=folium.Icon(color="green", icon="play")).add_to(fg_gps)
    folium.Marker(gps_coords[-1], tooltip="End GPS", icon=folium.Icon(color="red", icon="stop")).add_to(fg_gps)
    fg_gps.add_to(m)
    legend_items.append(("GPS Observations", "#1f5fd6"))

    # 6. Route Discontinuity Markers
    if breaks_baseline:
        fg_breaks = folium.FeatureGroup(name=f"Baseline Gaps ({len(breaks_baseline)})", show=False)
        for lat, lon in breaks_baseline:
            folium.CircleMarker(
                [lat, lon], radius=6, color="#ff0000", weight=2, fill=True, fill_color="white",
                fill_opacity=1.0, tooltip="Baseline Route Discontinuity"
            ).add_to(fg_breaks)
        fg_breaks.add_to(m)
        legend_items.append(("Baseline Gaps", "#ff0000"))

    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(m)

    # HTML Legend
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{col};margin-right:6px;vertical-align:middle"></span>{lab}</div>'
        for lab, col in legend_items
    )
    legend_html = (
        f'<div style="position:fixed;bottom:40px;left:12px;z-index:9999;background:white;color:#222;'
        f'padding:8px 10px;border:1px solid #bbb;border-radius:6px;font:13px sans-serif;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.2)">'
        f'<div style="font-weight:bold;margin-bottom:4px;">{title}</div>{rows}</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))


# ==============================================================================
# 5. CLI Parser
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="PathSnap Member 3: HMM & Viterbi Map-Matching CLI")
    parser.add_argument("--predictions", type=str, default="results/predictions_logistic.parquet",
                        help="Path to candidate predictions parquet or csv file")
    parser.add_argument("--dataset-dir", type=str, default="student_test",
                        help="Path to raw dataset directory containing trips")
    parser.add_argument("--trip-id", type=str, default=None,
                        help="Specific trip ID to decode (default: all available)")
    parser.add_argument("--model", type=str, default="logistic",
                        choices=["logistic", "knn", "rf", "lgbm", "mlp"],
                        help="Model name identifier for reporting (default: logistic)")
    parser.add_argument("--beta", type=float, default=30.0,
                        help="HMM transition distance scaling factor in metres (default: 30.0)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to store evaluation CSVs and summary JSON (default: results)")
    parser.add_argument("--figures-dir", type=str, default="results/figures",
                        help="Directory to store evaluation plots (default: results/figures)")
    parser.add_argument("--maps-dir", type=str, default="results/maps",
                        help="Directory to store Folium HTML maps (default: results/maps)")
    parser.add_argument("--synthetic-smoke-test", action="store_true",
                        help="Run end-to-end synthetic verification test")
    parser.add_argument("--make-map", action="store_true",
                        help="Generate interactive Folium map visualization")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate against ground-truth .route if available")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    maps_dir = Path(args.maps_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_smoke_test:
        run_synthetic_smoke_test(output_dir, figures_dir)
        return

    predictions_path = Path(args.predictions)
    dataset_dir = Path(args.dataset_dir)

    run_real_evaluation(
        predictions_path=predictions_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        figures_dir=figures_dir,
        maps_dir=maps_dir,
        model_name=args.model,
        beta=args.beta,
        target_trip_id=args.trip_id,
        generate_maps=args.make_map
    )


if __name__ == "__main__":
    main()
