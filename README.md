# PathSnap: Distributed Map-Matching and Route Reconstruction

[![Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/sanjana-g22/ML-Hackathon)
[![Course](https://img.shields.io/badge/PES%20University-UE24CS352A%20Machine%20Learning-orange.svg)](#hackathon-context)
[![Phase](https://img.shields.io/badge/Status-Phase%201%20Common%20Setup%20Only-yellow.svg)](#implementation-status)

---

## 1. Project Purpose & Overview

GPS tracking units generate noisy, irregular observations subject to urban multipath effects, signal interference, building obstruction, and sensor dropouts. When plotted naively, a vehicle appears to jump across buildings, swerve onto parallel streets, or traverse non-existent connections.

**PathSnap** is a distributed machine learning and sequential optimization system designed to reconstruct realistic vehicle trajectories by accurately matching noisy GPS observations against an underlying geographical road network. 

The complete solution spans:
1. Exploratory data analysis, validation, and geographic preprocessing.
2. Spatial candidate road-segment generation and geometric feature engineering.
3. Multi-family machine learning classifiers to predict candidate arc relevance.
4. Bayesian hyperparameter optimization on group-aware cross-validation folds.
5. Hidden Markov Model (HMM) and Viterbi dynamic programming for topological route decoding.
6. A distributed two-laptop deployment featuring a FastAPI backend and an interactive evaluator UI.

---

## 2. Hackathon Context

This project is developed for the **PES University UE24CS352A Machine Learning Hackathon**.

### Authoritative Requirements & Policies (from `docs/`):
- **Independent Experimentation:** All models, features, code, and tuning must be developed independently. Model weights, hyperparameter configs, and code are compared across all teams.
- **Evaluation Integrity & Data Leakage Prevention:**
  - `train` (80 trips): Exclusively used for preprocessing, training, feature engineering, and cross-validation.
  - `student_test` (10 trips): Exclusively used for development-stage evaluation, model comparison, and tuning validation.
  - `evaluator_test` (10 trips): Reserved strictly for final evaluation. Ground-truth route labels remain server-side and must never enter the feature extraction or prediction pipelines.
  - Hardcoding predictions or bypassing API evaluation logic is strictly prohibited.
- **Model Diversity Requirement:** Teams must train, evaluate, and compare at least **five distinct models** across different algorithm families before selecting the final model.

---

## 3. Four-Member Collaboration Structure

This project is maintained and executed by a four-member engineering team hosted at:  
👉 **Repository:** [https://github.com/sanjana-g22/ML-Hackathon](https://github.com/sanjana-g22/ML-Hackathon)

### Module Ownership & Responsibilities

```
+-------------------------------------------------------------------------------------------------------+
|                                           PATHSNAP TEAM                                               |
+-----------------------------------+-----------------------------------+-------------------------------+
| Member                            | Core Technical Modules            | Deliverable Sections          |
+-----------------------------------+-----------------------------------+-------------------------------+
| Member 1                          | • Dataset loading & validation    | • Dataset EDA Report          |
|                                   | • Preprocessing & anomalies       | • Preprocessing documentation |
|                                   | • Candidate generation & features | • Linear / KNN Model Sections |
|                                   | • Logistic Regression & KNN       |                               |
+-----------------------------------+-----------------------------------+-------------------------------+
| Member 2                          | • Unified evaluation harness      | • Model Comparison Table      |
|                                   | • Random Forest Classifier        | • Hyperparameter Search Logs  |
|                                   | • Gradient Boosting Classifier    | • Model 3 & 4 Report Sections |
|                                   | • Bayesian Optimization (Optuna)  |                               |
+-----------------------------------+-----------------------------------+-------------------------------+
| Member 3                          | • Directed road network graph     | • Graph Topology Analysis     |
|                                   | • MLP Neural Network Classifier   | • Route Continuity Metrics    |
|                                   | • HMM & Viterbi Route Decoder     | • HMM Formulation & Viterbi   |
|                                   | • Transition / emission modeling  | • Model 5 Report Section      |
+-----------------------------------+-----------------------------------+-------------------------------+
| Member 4                          | • Folium geospatial visualization | • Geographical Map Figures    |
|                                   | • FastAPI inference service       | • Two-Laptop Architecture     |
|                                   | • Evaluator frontend interface    | • API & UI Demonstration      |
|                                   | • Two-laptop network deployment   | • Deployment Runbook          |
+-----------------------------------+-----------------------------------+-------------------------------+
```

---

## 4. Project Folder Structure

The repository is structured to isolate responsibilities while providing shared contracts:

```
PathSnap/
├── docs/                                # Official hackathon requirements & guidelines
│   ├── Assessment and Plagiarism Policy.pdf
│   ├── Dataset Exploration.pdf
│   ├── Deliverables.pdf
│   ├── Main Guidelines.pdf
│   └── Models and Training Guidelines.pdf
├── data/                                # Dataset root (git-ignored raw data, preserved via .gitkeep)
│   ├── .gitkeep
│   ├── train/                          # 80 training trips
│   └── student_test/                   # 10 development-testing trips
├── notebooks/                           # Collaborative research and EDA notebooks
│   └── .gitkeep
├── src/                                 # Shared Python package source code
│   └── __init__.py                     # Package entry point
├── tests/                               # Automated unit and integration tests
│   └── .gitkeep
├── results/                             # Evaluation outputs, metrics tables, and EDA figures
│   └── .gitkeep
├── maps/                                # Saved Folium interactive HTML map artifacts
│   └── .gitkeep
├── models/                              # Model metadata and saved checkpoint configurations
│   └── .gitkeep
├── backend/                             # FastAPI application (Laptop 1: Model & Evaluation Service)
│   └── .gitkeep
├── frontend/                            # Evaluator Web UI (Laptop 2: Client Interface)
│   └── .gitkeep
├── .gitignore                           # Git exclusion rules for environments, caches, and raw data
├── requirements.txt                     # Shared Python dependencies
└── README.md                            # Common project architecture and collaboration guide
```

---

## 5. Expected Dataset Formats (Official Specification)

Each trip partition directory (`train/`, `student_test/`) contains individual subdirectories identified by an 8-digit zero-padded integer (e.g., `00000001/`).

Each trip directory consists of four whitespace-separated, headerless plain text files:

| File Extension | Content Description | Columns / Format | Indexing / Conventions |
| :--- | :--- | :--- | :--- |
| **`.track`** | Raw GPS trajectory observations | `longitude` (float), `latitude` (float), `timestamp` (float seconds) | Nominally 1.0 Hz (~97% of consecutive timestamps are 1 s apart). May contain GPS dropouts ($\Delta t > 5\text{s}$) or multipath noise. |
| **`.nodes`** | Geographical road-network nodes | `longitude` (float), `latitude` (float) | **Zero-based row index** represents the global `node_id` $\in [0, N_{nodes}-1]$. |
| **`.arcs`** | Directed road network segments | `from_node_id` (int), `to_node_id` (int) | **Zero-based row index** represents the global `arc_id` $\in [0, N_{arcs}-1]$. Directed connection from origin to destination node. |
| **`.route`** | Reference ground truth route | `arc_id` (int) | Single-column chronological sequence of ground-truth `arc_id`s traversed by the vehicle. Supplied on `train` and `student_test`. |

> [!IMPORTANT]
> **Cardinality Rule:** Do NOT assume a 1-to-1 correspondence between rows in `.track` and `.route`. A vehicle typically remains on a single road arc across multiple seconds (multiple GPS observations), or may traverse short segments between observations.

---

## 6. Proposed Team Integration Conventions & Interfaces

To ensure friction-free collaboration across all four team members, all modules adhere to the following shared data schemas and interface conventions:

### 6.1 Identifiers and Indexing Standard
- `trip_id`: String (e.g., `"00000001"`), zero-padded 8-digit format.
- `observation_index`: Integer ($0$-indexed) denoting the observation row inside `.track`.
- `node_id`: Integer ($0$-indexed) matching row index in `.nodes`.
- `arc_id`: Integer ($0$-indexed) matching row index in `.arcs`.
- Coordinate System: WGS84 angular degrees (`longitude`, `latitude`). Longitude is first in raw files, latitude is second. Distances are computed strictly in metres using Geodesic or Haversine formulations.

### 6.2 Proposed Module Contracts

```
[Member 1: Data Loader & Preprocessing]
                  │
                  ▼
         Cleaned Observations
                  │
                  ▼
[Member 1: Candidate Generation] ──────► Candidate Set per Observation:
                  │                      {trip_id, obs_idx, candidate_arc_id,
                  │                       perp_dist_m, bearing_diff_deg, ...}
                  ▼
[Member 1 & 2: ML Classifiers]   ──────► Candidate Probability Vector:
                  │                      P(arc_id | observation_features)
                  ▼
[Member 3: HMM / Viterbi Decoder]◄────── [Member 3: Directed Graph Topology]
                  │
                  ▼
         Decoded Route Output:
         [arc_id_0, arc_id_1, ..., arc_id_K]
                  │
         ┌────────┴────────┐
         ▼                 ▼
[Member 4: FastAPI]   [Member 4: Folium Visualization]
         │                 │
         ▼                 ▼
Laptop 2 Evaluator UI  maps/sample_trip_map.html
```

#### A. Candidate Features Contract (`CandidateFeatureRecord`)
For each GPS observation $i$, candidate generation produces $K$ candidate road segments within spatial radius $R$ (e.g., $R = 100\text{m}$):
```python
candidate_schema = {
    "trip_id": str,               # e.g. "00000001"
    "obs_idx": int,               # 0-indexed GPS point index
    "timestamp": float,           # Observation timestamp in seconds
    "gps_lon": float,             # Raw GPS longitude
    "gps_lat": float,             # Raw GPS latitude
    "candidate_arc_id": int,      # Road arc under evaluation
    "from_node_id": int,          # Arc start node
    "to_node_id": int,            # Arc end node
    "perp_distance_m": float,     # Perpendicular distance from GPS to segment in metres
    "bearing_difference_deg": float, # Difference between trajectory heading and road azimuth
    "segment_length_m": float,    # Physical length of candidate arc
    "speed_mps": float,           # Vehicle speed at observation
    "is_ground_truth": int        # 1 if candidate_arc_id is in reference route, else 0 (training only)
}
```

#### B. Model Probability Output Contract
Every classification model produces emission candidate probabilities per observation:
```python
# Dict mapping observation index to list of (arc_id, probability) pairs
emission_probabilities = {
    obs_idx: [
        (arc_id_1, prob_1),
        (arc_id_2, prob_2),
        ...
    ]
}
```

#### C. Predicted Route Contract
The HMM/Viterbi sequential decoder outputs a valid sequence of traversed arcs:
```python
route_prediction_schema = {
    "trip_id": str,
    "matched_arc_sequence": [arc_id_1, arc_id_2, ...], # Chronological sequence
    "pointwise_matches": [                              # Per-observation assignments
        {"obs_idx": 0, "matched_arc_id": 1042},
        {"obs_idx": 1, "matched_arc_id": 1042},
        ...
    ],
    "route_continuity_pct": float,                      # Continuity score (0 - 100%)
    "inference_time_ms": float
}
```

---

## 7. Environment Setup & Dependency Installation

### Prerequisites
- Python 3.11+ (Python 3.11 to 3.13 recommended; tested on macOS / Linux / Windows)
- Git 2.30+

### Step-by-Step Installation:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sanjana-g22/ML-Hackathon.git
   cd ML-Hackathon
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (cmd.exe):
   python -m venv .venv
   .venv\Scripts\activate.bat

   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install shared project dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify environment and core packages:**
   ```bash
   python -c "import numpy, pandas, scipy, sklearn, folium, fastapi; print('Environment verified successfully!')"
   ```

---

## 8. GitHub Collaboration Workflow

To ensure smooth multi-member collaboration without merge conflicts or regressions:

1. **Branch Naming Conventions:**
   - Member 1: `feature/member1-preprocessing-candidates`
   - Member 2: `feature/member2-evaluation-boosters`
   - Member 3: `feature/member3-graph-hmm-viterbi`
   - Member 4: `feature/member4-folium-api-frontend`

2. **Workflow Rules:**
   - **Never push directly to `main`.** All code lands via Pull Requests (PRs).
   - Before opening a PR, rebase against current `main`:
     ```bash
     git checkout main
     git pull origin main
     git checkout feature/<your-branch>
     git rebase main
     ```
   - Run unit tests locally before submitting PR:
     ```bash
     pytest tests/
     ```
   - Require at least one peer team review before merging.
   - Do NOT commit large binaries, raw datasets (`.track`, `.nodes`, etc.), or sensitive `.env` files.

---

## 9. Implementation Status & Roadmap

> [!NOTE]
> This section clearly delineates the current setup status from upcoming phases.

| Phase | Milestone | Owner | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Common Team Setup:** Structure, configs, requirements, interfaces | All Members | **COMPLETED** |
| **Phase 2** | **Dataset EDA & Preprocessing:** Quality audit, data loaders, sanity checks | Member 1 | Planned |
| **Phase 3** | **Candidate Generation & Feature Engineering:** Spatial indexing, features | Member 1 | Planned |
| **Phase 4** | **Model Experimentation:** 5 distinct algorithms across linear, tree, ensemble, neural | Members 1, 2, 3 | Planned |
| **Phase 5** | **Bayesian Optimization:** Optuna group-aware tuning on training set | Member 2 | Planned |
| **Phase 6** | **HMM & Viterbi Route Reconstruction:** Transition modeling & route decoding | Member 3 | Planned |
| **Phase 7** | **Interactive Visualization:** Folium map generation with layer controls | Member 4 | Planned |
| **Phase 8** | **Two-Laptop Deployment:** FastAPI server (Laptop 1) and Evaluator UI (Laptop 2) | Member 4 | Planned |
| **Phase 9** | **Technical Report & Demonstration:** Consolidated PDF report and viva rehearsal | All Members | Planned |

---

## 10. Planned Module Specifications (Placeholders)

### 10.1 Exploratory Data Analysis & Preprocessing (Planned — Member 1)
- Statistical profiling of trajectory lengths, sampling frequencies, and bounding extents.
- Identification of GPS dropouts ($\Delta t > 5.0\text{s}$) and stationary duplicate readings.
- Preprocessing transformations applied strictly to permitted training data.

### 10.2 Model Experimentation & Comparison (Planned — Members 1, 2, 3)
- Minimum of 5 candidate classification models across diverse families:
  1. *Linear / Probabilistic:* Logistic Regression
  2. *Instance-Based:* Distance-Weighted KNN
  3. *Ensemble / Tree:* Random Forest Classifier
  4. *Boosting:* Gradient Boosting / HistGradientBoosting
  5. *Neural Network:* Multi-Layer Perceptron (MLP)
- Standardized cross-validation using GroupKFold grouped by `trip_id` to prevent intra-trip leakage.
- Comparison metrics: Observation-level Top-1 Accuracy, Macro F1, Log Loss, and Candidate Miss Rate.

### 10.3 Bayesian Hyperparameter Optimization (Planned — Member 2)
- Automated search using Optuna with group-aware cross-validation on 80 training trips.
- Objective: Maximize validation Macro F1 score while constraining inference latency.

### 10.4 HMM & Viterbi Route Reconstruction (Planned — Member 3)
- Hidden states: Candidate road segments $A_i$.
- Emission probabilities: Derived from classifier calibrated posterior probabilities $P(\text{arc} \mid \text{features})$.
- Transition probabilities: Network topology reachability matrix and shortest-path distance vs trajectory displacement.
- Viterbi recurrence for global optimal path decoding ensuring high route continuity.

### 10.5 Folium Geographical Visualization (Planned — Member 4)
- Interactive maps rendering:
  - Raw GPS points (blue markers)
  - Road network segments (slate gray lines)
  - Reference route (green polyline)
  - Start/End markers and layer control toggles.
- Exported as `maps/sample_trip_map.html`.

### 10.6 Two-Laptop Deployment Architecture (Planned — Member 4)
- **Laptop 1 (Model Server & API):**
  - Runs FastAPI service listening on local network interface.
  - Hosts trained models, road network graphs, and HMM decoder.
  - Calculates prediction metrics against internal server-side reference data.
- **Laptop 2 (Evaluator Client & UI):**
  - Displays the 10 evaluator trip options.
  - Sends selected `trip_id` via REST payload to Laptop 1 endpoint `/evaluate/{trip_id}`.
  - Renders the predicted route, map visualization, and performance metrics.
