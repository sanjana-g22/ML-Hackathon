# PathSnap: Model 5 — Multi-Layer Perceptron (MLP) Neural Network Classifier

**Course:** PES University UE24CS352A Machine Learning  
**Project:** PathSnap — Distributed Map-Matching and Route Reconstruction  
**Team Member:** Member 3  
**Module Location:** `models/mlp_model.py`, `models/train_mlp.py`, `models/predict.py`, `models/candidate_adapter.py`  
**Test Suite:** `tests/test_mlp.py` (10 unit/integration tests passing)  
**Status:** Architecture, Preprocessing, Training Engine & HMM Predictor Implemented; Ready for Member 1 Candidate Dataset.

> [!NOTE]
> **Dataset Availability & Real Training Status:**
> As per the official hackathon workflow, Member 1's multi-million row precomputed training candidate dataset (`data/candidates.parquet` for the 80 training trips) is not tracked in git due to storage constraints. Candidate predictions on `student_test` were generated for Logistic Regression and KNN.
> The Model 5 MLP pipeline is fully implemented (`models/mlp_model.py`, `models/train_mlp.py`, `models/candidate_adapter.py`, `models/predict.py`) and validated across 10 automated unit and integration tests. The synthetic training curves in `results/figures/` serve as architecture verification fixtures and are not represented as real dataset performance. All evaluation metrics in the report clearly distinguish between real student_test evaluation and synthetic verification fixtures.

---

## 1. Problem Formulation & Objective

In urban trajectory map-matching, a vehicle's true position is obscured by building multipath reflections, signal scattering, sensor noise, and sampling irregularities. Given a noisy GPS observation $z_t = (\text{lon}_t, \text{lat}_t, \text{time}_t)$, spatial candidate generation identifies a set of candidate road network segments $C(z_t) = \{a_1, a_2, \dots, a_k\}$ within a localized search radius.

**Model 5 (MLP)** serves as the non-linear candidate scoring model:
$$\hat{y}_i = P(a_i = \text{ground truth} \mid \mathbf{x}_i) = \sigma(f_{\text{MLP}}(\mathbf{x}_i))$$
where:
- $\mathbf{x}_i \in \mathbb{R}^D$ is the engineered feature vector capturing the geometric and kinematic relationship between GPS observation $z_t$ and candidate road segment $a_i$.
- $\hat{y}_i \in [0.0, 1.0]$ is the calibrated posterior probability that arc $a_i$ was the true segment traversed by the vehicle.
- These probabilities directly populate the **emission probability matrix** $B_i(z_t)$ in Member 3's upcoming Hidden Markov Model (HMM) Viterbi sequential decoder.

---

## 2. Input Feature Schema & Target Definition

Model 5 strictly conforms to the team's shared candidate feature schema defined in `README.md` and ingested via `models/candidate_adapter.py`:

| Feature Name | Type | Physical Meaning | Expected Behavior for True Matches |
| :--- | :--- | :--- | :--- |
| **`perp_distance_m`** | `float` | Perpendicular / shortest distance from GPS point to candidate road arc segment in metres (geodesic). | Low values (typically $< 10\text{m}$). High values indicate spurious parallel or cross streets. |
| **`bearing_difference_deg`** | `float` | Absolute angular difference $| \theta_{\text{traj}} - \theta_{\text{road}} | \in [0^\circ, 180^\circ]$ between heading vectors. | Close to $0^\circ$ for valid travel direction. Higher for perpendicular streets or wrong-way one-ways. |
| **`segment_length_m`** | `float` | Physical length of the candidate arc in metres along WGS84 ellipsoid. | Contextual normalization for projection geometry. |
| **`speed_mps`** | `float` | Estimated instantaneous vehicle velocity $\Delta d / \Delta t$ at observation $t$. | Distinguishes highway driving vs slow urban queueing. |

### Target Label:
- **`is_ground_truth` $\in \{0, 1\}$**:
  - $y = 1$: The candidate arc $a_i$ corresponds to the true vehicle path at observation $t$.
  - $y = 0$: Candidate arc is an alternative, nearby road segment that was not traversed.

> [!IMPORTANT]
> **Cardinality & Ground Truth Integrity:** As established in the official guidelines, rows in `.track` do **not** correspond 1-to-1 with rows in `.route`. Labels must never be fabricated by naive index matching. Ground-truth assignment requires verifying that the arc is traversed in the reference route at the timestamp of the GPS observation.

---

## 3. Neural Network Architecture (`models/mlp_model.py`)

Model 5 is implemented in PyTorch (`BinaryMLPClassifier`):

```
Input Vector x (D dimensions)
       │
       ▼
┌─────────────────────────────────────────┐
│ Linear Layer 1: (D -> 128)             │
│ Activation:     ReLU                    │
│ Regularization: Dropout (p = 0.2)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Linear Layer 2: (128 -> 64)             │
│ Activation:     ReLU                    │
│ Regularization: Dropout (p = 0.2)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Linear Layer 3: (64 -> 32)              │
│ Activation:     ReLU                    │
│ Regularization: Dropout (p = 0.2)       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ Output Projection: (32 -> 1)            │
│ Raw Scalar Logit: s in (-inf, +inf)     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
      P(y=1|x) = Sigmoid(s) in [0, 1]
```

### Architectural Justification:
1. **Non-linear Geometric Boundaries:** Road geometry involves non-linear combinations of perpendicular distance and angular heading (e.g. an arc 15m away with $0^\circ$ heading difference is far more plausible than an arc 5m away with $90^\circ$ heading difference). An MLP learns these non-linear interaction surfaces naturally.
2. **Kaiming / He Normal Weight Initialization:** Ensures gradients neither vanish nor explode across ReLU layers during early training epochs.
3. **Dropout Regularization ($p = 0.2$):** Prevents the network from co-adapting on a single dominant feature (e.g. overfitting purely on `perp_distance_m` while neglecting heading orientation).

---

## 4. Leakage-Free Preprocessing & Training Pipeline

### 4.1 Strict Group-Based Splitting
- Candidate observations belonging to the same trip exhibit strong temporal auto-correlation.
- Standard random train/test splitting leaks trip-specific spatial topology into the validation set, creating artificially inflated accuracy.
- `models/train_mlp.py` enforces **`GroupKFold` on `trip_id`**:
  $$\text{Trips}_{\text{train}} \cap \text{Trips}_{\text{val}} = \emptyset$$
- All candidate rows for any given trip reside strictly in either train or validation.

### 4.2 Leak-Free Preprocessing
1. **Median Imputation (`SimpleImputer`):** Fits median values strictly on $\mathbf{X}_{\text{train}}$ and transforms $\mathbf{X}_{\text{val}}$.
2. **Feature Scaling (`StandardScaler`):** Computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ exclusively on training observations. Validation and future test observations are transformed using these stored parameters without re-fitting.
3. Saved together into `results/model5_mlp/preprocessing.joblib`.

### 4.3 Class Imbalance Compensation
Candidate generation typically produces $K \approx 5$ to $10$ candidate segments per GPS point, of which only $1$ is positive. This creates a severe negative-to-positive class imbalance ($w \approx 4:1$ to $9:1$).
- To prevent the network from collapsing to the majority class ($y = 0$), we compute:
  $$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$
- `pos_weight` is passed directly into `torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)`, scaling the loss contribution of true positive candidates.

### 4.4 Early Stopping & Checkpoint Restoration
- Optimization uses **AdamW** ($\text{lr} = 0.001$, $\text{weight\_decay} = 10^{-4}$) with a `ReduceLROnPlateau` scheduler.
- Monitored metric: Validation Loss ($\mathcal{L}_{\text{val}}$).
- If $\mathcal{L}_{\text{val}}$ fails to improve for `patience = 7` consecutive epochs, training terminates early.
- The pipeline restores `best_model.pt` from the epoch with minimum validation loss (not simply the final epoch).

---

## 5. Candidate-Level Accuracy vs Route Reconstruction Accuracy

An essential distinction evaluated during viva and hackathon review:

| Dimension | Candidate-Level Classification Metric | Complete Route Reconstruction Metric |
| :--- | :--- | :--- |
| **Unit of Evaluation** | Individual `(GPS point, candidate arc)` pair. | Entire vehicle trajectory across all time steps. |
| **Scope** | Independent pointwise binary classification: Is arc $a_i$ the correct segment at step $t$? | Sequence-level path legality: Is the sequence of road segments $[a_1, a_2, \dots, a_T]$ topologically continuous in the road network? |
| **Typical Failure Mode** | High binary accuracy (e.g. 96%) can still produce impossible vehicle maneuvers (e.g. teleporting between unconnected parallel freeways). | Penalizes topological breaks and missing transition links. |
| **Observation Top-1 Metric** | Proximity accuracy: Does the highest-scoring candidate segment for observation $t$ match the ground-truth arc? | Evaluates sequence continuity: Does arc $t$'s `to_node_id` equal arc $t+1$'s `from_node_id`? |
| **Bridge Between the Two** | Model 5 provides candidate emission scores $\hat{y}_i$. | Member 3's HMM/Viterbi decoder uses these scores alongside road network graph shortest paths to enforce 100% legal route reconstruction. |

---

## 6. Connection to HMM / Viterbi Route Reconstruction

Model 5 outputs directly power the sequential HMM route reconstruction:

1. **Emission Probabilities:**
   For GPS observation $z_t$ and candidate road segment $a_i$:
   $$B_i(z_t) = P(z_t \mid a_i) \propto \hat{y}_i$$
   where $\hat{y}_i = \text{predict\_candidate\_probabilities}(M, P, C(z_t))$.
   Probabilities are normalized over the candidate set:
   $$\tilde{B}_i(z_t) = \frac{\hat{y}_i}{\sum_{j=1}^K \hat{y}_j}$$

2. **Transition Probabilities (via Road-Network Graph):**
   Using `src/road_network.py` (`shortest_path_between_arcs`):
   $$A_{ij} = P(a_j \mid a_i) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{net}}(a_i \to a_j) - d_{\text{gps}}(z_t, z_{t+1})|}{\beta}\right)$$

3. **Viterbi Decoding:**
   Finds the global optimal arc sequence maximizing $\prod_t B_{s_t}(z_t) A_{s_t, s_{t+1}}$, eliminating GPS noise and preserving 100% route continuity.

---

## 7. Model 5 Artifact Locations

All training and evaluation runs export structured artifacts to `results/model5_mlp/`:

```text
results/model5_mlp/
├── best_model.pt             # PyTorch state dict and architecture metadata (best validation epoch)
├── preprocessing.joblib      # Fitted StandardScaler and SimpleImputer (zero data leakage)
├── feature_columns.json      # Ordered list of feature names used during training
├── training_config.json      # Complete hyperparameters, seed, optimizer, and device settings
├── training_history.csv      # Epoch-by-epoch loss, accuracy, and learning rate log
├── training_summary.json     # High-level summary of model performance
├── metrics.json              # Full validation metrics (Accuracy, F1, ROC-AUC, PR-AUC, Top-1)
├── predictions.csv           # Pointwise validation candidate predictions
├── loss_curve.png            # Loss vs epoch plot with best checkpoint marker
├── accuracy_curve.png        # Training vs validation accuracy trajectory
└── confusion_matrix.png      # Formatted confusion matrix heatmap
```

---

## 8. Reproducible Commands

### Execute Unit & Integration Test Suite (10 Model 5 tests + 12 Road Network tests):
```bash
.venv/bin/pytest tests/test_mlp.py tests/test_road_network.py -v
```

### Run Model 5 Training on Member 1's Candidate Dataset:
```bash
.venv/bin/python -m models.train_mlp --data data/candidates.parquet --epochs 50 --batch-size 256 --lr 0.001 --seed 42 --output-dir results/model5_mlp
```

### Run End-to-End Smoke Test on Synthetic Fixture (Verification Mode):
```bash
.venv/bin/python -m models.train_mlp --synthetic-smoke-test --epochs 10 --batch-size 64 --output-dir /tmp/mlp_smoke
```

---

## 9. 60-Second Viva Presentation Script for Member 3

> *"For Model 5, I implemented a deep Multi-Layer Perceptron neural network to solve candidate road-segment classification.*
>
> *GPS noise creates ambiguity: multiple parallel or cross streets often lie within the search radius of a single GPS point. Our MLP takes geometric features — including perpendicular distance, heading divergence, segment length, and velocity — and passes them through a 3-layer architecture (128-64-32 units) with ReLU activations and Dropout regularization.*
>
> *To guarantee academic integrity and prevent data leakage, we implemented GroupKFold splitting by trip ID so no observations from the same trip appear in both training and validation. Preprocessing scalers and imputers are strictly fitted on training folds only. We compensate for candidate imbalance using weighted binary cross-entropy loss and apply early stopping to save the best checkpoint.*
>
> *Crucially, Model 5's calibrated probabilities serve as the emission scores for our upcoming HMM Viterbi decoder. While candidate-level accuracy tells us how well we score individual road segments, our road network graph topological transitions will ensure the final decoded vehicle trajectory has 100% legal road continuity."*
