"""
PathSnap - Candidate Dataset Adapter (Member 3)
===============================================
Provides standardized loading, schema validation, and feature extraction
for candidate road segment datasets produced by Member 1's preprocessing.

Schema Conventions (Agreed in Team README):
- Grouping: trip_id (str), obs_idx (int)
- Road Network: candidate_arc_id (int) or arc_id (int)
- Target: is_ground_truth (int: 0 or 1)
- Geometric / Kinematic Features:
  * perp_distance_m: float (perpendicular distance in metres)
  * bearing_difference_deg: float (angular difference in degrees)
  * segment_length_m: float (arc length in metres)
  * speed_mps: float (vehicle speed in m/s)
"""

import os
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd


METADATA_COLUMNS = {
    "trip_id",
    "split",
    "obs_idx",
    "observation_index",
    "candidate_arc_id",
    "arc_id",
    "from_node_id",
    "to_node_id",
    "timestamp",
    "t",
    "gps_lon",
    "gps_lat",
    "longitude",
    "latitude",
    "is_ground_truth",
    "label",
    "target"
}


def load_candidate_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads candidate features from Parquet or CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Candidate dataset not found at: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
    elif ext in [".tsv"]:
        df = pd.read_csv(file_path, sep="\t")
    else:
        # Attempt parquet first, fallback to csv
        try:
            df = pd.read_parquet(file_path)
        except Exception:
            df = pd.read_csv(file_path)

    # Standardize column naming if variations exist
    rename_map = {}
    if "observation_index" in df.columns and "obs_idx" not in df.columns:
        rename_map["observation_index"] = "obs_idx"
    if "arc_id" in df.columns and "candidate_arc_id" not in df.columns:
        rename_map["arc_id"] = "candidate_arc_id"
    if "label" in df.columns and "is_ground_truth" not in df.columns:
        rename_map["label"] = "is_ground_truth"
    if "target" in df.columns and "is_ground_truth" not in df.columns:
        rename_map["target"] = "is_ground_truth"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def validate_candidate_schema(
    df: pd.DataFrame,
    require_labels: bool = True
) -> Tuple[bool, List[str]]:
    """
    Verifies that the candidate DataFrame contains required identifiers,
    target labels, and numeric feature columns.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check grouping IDs
    if "trip_id" not in df.columns:
        issues.append("Missing required column 'trip_id'")
    if "obs_idx" not in df.columns:
        issues.append("Missing required column 'obs_idx'")
    if "candidate_arc_id" not in df.columns:
        issues.append("Missing required column 'candidate_arc_id'")

    # Check labels
    if require_labels:
        if "is_ground_truth" not in df.columns:
            issues.append("Missing target label column 'is_ground_truth'")
        else:
            unique_vals = set(df["is_ground_truth"].dropna().unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                issues.append(f"Target column 'is_ground_truth' must be binary (0 or 1), got {unique_vals}")

    # Check for feature columns
    features = get_feature_columns(df)
    if not features:
        issues.append("No numeric feature columns found (expected candidate attributes like perp_distance_m, etc.)")

    is_valid = len(issues) == 0
    return is_valid, issues


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extracts numerical feature columns from the candidate DataFrame,
    excluding metadata identifiers and target labels.
    """
    feature_cols = [
        col for col in df.columns
        if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    return sorted(feature_cols)


def generate_synthetic_candidate_data(
    n_trips: int = 6,
    obs_per_trip: int = 20,
    candidates_per_obs: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic synthetic candidate dataset strictly for unit testing,
    pipeline integration verification, and test coverage.

    DO NOT use this for reporting real model accuracy!
    """
    rng = np.random.RandomState(seed)
    rows = []

    for t_idx in range(n_trips):
        trip_id = f"{t_idx + 1:08d}"
        for obs_idx in range(obs_per_trip):
            ts = float(obs_idx * 1.0)
            gps_lon = 8.58 + rng.normal(0, 0.005)
            gps_lat = 49.85 + rng.normal(0, 0.005)
            speed = float(np.clip(rng.normal(12.0, 4.0), 0.5, 45.0))

            # Exactly one ground-truth candidate per observation
            gt_cand_pos = rng.randint(0, candidates_per_obs)

            for c_idx in range(candidates_per_obs):
                arc_id = 1000 + c_idx * 10 + rng.randint(0, 5)
                is_gt = 1 if c_idx == gt_cand_pos else 0

                # Ground truth arc typically closer and better aligned
                if is_gt:
                    perp_dist = float(np.abs(rng.normal(4.0, 3.0)))
                    bearing_diff = float(np.abs(rng.normal(8.0, 6.0)))
                else:
                    perp_dist = float(np.abs(rng.normal(35.0, 18.0)) + 8.0)
                    bearing_diff = float(np.abs(rng.normal(45.0, 25.0)) + 15.0)

                seg_len = float(np.clip(rng.normal(40.0, 15.0), 5.0, 200.0))

                rows.append({
                    "trip_id": trip_id,
                    "obs_idx": obs_idx,
                    "timestamp": ts,
                    "gps_lon": gps_lon,
                    "gps_lat": gps_lat,
                    "candidate_arc_id": arc_id,
                    "from_node_id": arc_id * 2,
                    "to_node_id": arc_id * 2 + 1,
                    "perp_distance_m": perp_dist,
                    "bearing_difference_deg": min(180.0, bearing_diff),
                    "segment_length_m": seg_len,
                    "speed_mps": speed,
                    "is_ground_truth": is_gt
                })

    return pd.DataFrame(rows)
