## 3.3 Model 2 KNN

### A Model understanding

PathSnap is framed as a candidate classification problem: every (GPS observation, candidate arc) pair is a row, labelled 1 if the candidate is the reference arc and 0 otherwise, and the highest-probability candidate per observation is the prediction. k-Nearest Neighbours (KNN) suits this framing because candidates that look alike in feature space tend to share the same label. A candidate that is close to its GPS point, aligned with the vehicle's heading and well inside its segment resembles the true arcs seen in training, while a distant candidate pointing the opposite way resembles the false ones. KNN uses this similarity directly: it stores labelled training candidates and scores a new candidate by looking at its most similar stored examples. It makes no assumption about the shape of the decision boundary, so it can capture non-linear combinations such as "close but wrong direction" without hand-made interaction terms, which the linear Logistic Regression could not.

Strengths for this task: it is non-parametric and needs no iterative training (fitting only stores the sample, 0.6 s); its predictions are easy to explain ("these nine similar candidates were mostly true arcs"); and the only real hyperparameters, k and the neighbour weighting, can be compared cheaply on one validation curve.

Limitations: KNN measures similarity with Euclidean distance, so all 14 features count equally after scaling, including weak ones such as the node degrees, which dilutes the informative distance and heading features. It needs scaled features and cannot accept missing values, so the motion features must be imputed. Prediction cost grows with the number of stored rows, which forced us to store a 300,000-row sample of the 5.17 million training candidates and still made it the slowest model at inference (about 5.9 s per trip). Like the other classifiers, it scores each candidate independently and cannot enforce route connectivity, and its probabilities come from only k = 9 neighbours, so many candidates receive exactly 0.

### B Training and configuration

Hyperparameters: n_neighbors (k) = 9 and weights = distance, selected from the grid k ∈ {1, 3, 5, 9, 15, 25, 35, 51} × weights ∈ {uniform, distance} (scikit-learn's default starting point is k = 5, uniform); distance metric = Euclidean (Minkowski, p = 2) on standardised features; neighbour search = brute force; stored training sample = 300,000 candidate rows per fit, drawn uniformly at random so the class balance (3.9% positive) is unchanged; random seed = 42 (42 + fold number for the fold samples).

Objective and prediction rule: KNN has no loss function or iterative optimisation. Fitting stores the preprocessed training sample. For a new candidate, the model finds the 9 stored candidates with the smallest Euclidean distance and predicts P(true arc) = Σ wᵢ·yᵢ / Σ wᵢ, where yᵢ is the neighbour's label and wᵢ = 1 / distanceᵢ, so closer neighbours count more (if a neighbour is at distance 0, only zero-distance neighbours are used). A candidate is labelled positive if this probability is at least 0.5. k and the weighting were chosen by the highest mean 5-fold CV Macro F1, with CV Log Loss as the tie-breaker.

Preprocessing: the 14 shared features are median-imputed (speed_mps, dt_s, heading_difference_rad and heading_cosine are missing for 0.05–0.22% of rows, mainly first observations and stationary points) and standardised with StandardScaler. The motion_valid and time_valid flags are kept, so the model can still tell imputed values apart. Both transformations are fitted on the stored training sample of each fold only. Identifier columns (trip_id, split, obs_idx, t, arc_id) and the label are excluded.

Training partition and validation: 5-fold GroupKFold grouped by trip_id on the 80 training trips (5,167,312 candidate rows), using the shared folds from src/metrics.py. To keep KNN practical on a CPU, three measures were used. Each fold stores a 300,000-row random sample of its training candidates. Neighbours are queried once per fold at k = 51, and the probabilities for every smaller k and both weightings are derived from that single query with cumulative sums, so the whole validation curve costs about the same as one model. Brute-force search was used because, measured on 300,000 stored rows with 14 features, it took about 140 µs per query against 930–1,080 µs for a KD-tree and 3,900–15,000 µs for a ball tree. Validation metrics were computed on a random 25% of each validation fold's GPS observations, keeping every candidate of each sampled observation so that Top-1 Accuracy stays well defined (about 256,000 rows per fold). Route Continuity and Sequence F1 need complete trips, so they are reported for the student-test partition only. The final model was refitted on a 300,000-row sample of all 80 training trips and evaluated once on all 468,861 candidate rows of the 10 student-test trips, with no sampling. The full experiment took 319 s.

### C Experimental results

| Metric | 5-fold CV (80 train trips) | Student test (10 trips) |
| :-- | :-: | :-: |
| Macro F1 | 0.801 ± 0.030 | 0.828 |
| Top-1 Accuracy | 0.622 ± 0.123 | 0.713 |
| Log Loss | 0.147 | 0.182 |
| Precision (true-arc class) | 0.747 | 0.755 |
| Recall (true-arc class) | 0.526 | 0.599 |
| Route Continuity | – (sampled observations) | 0.798 |
| Sequence F1 | – (sampled observations) | 0.513 |
| Candidate Miss Rate | – | 0.0% |
| Inference time | – | 5,925 ms per trip |

*CV values are the mean ± standard deviation over the five validation folds. Student-test confusion matrix at p ≥ 0.5: 445,973 true negatives, 3,728 false positives, 7,691 false negatives, 11,469 true positives.*

| k | Uniform: CV Macro F1 | Distance: CV Macro F1 | Uniform: training-row Macro F1 | CV Log Loss (distance) |
| :-: | :-: | :-: | :-: | :-: |
| 1 | 0.780 | 0.780 | 1.000 | 0.489 |
| 5 | 0.800 | 0.800 | 0.878 | 0.184 |
| **9** | 0.799 | **0.801** | 0.853 | 0.147 |
| 25 | 0.787 | 0.792 | 0.819 | 0.115 |
| 51 | 0.773 | 0.779 | 0.792 | 0.107 |

*Selected subset of the k sweep; the full sweep is in results/knn_k_sweep.csv.*

**Overfitting and underfitting:** The validation curve shows both. With k = 1, uniform KNN scores a perfect 1.000 Macro F1 on its stored training rows (each row is its own nearest neighbour) but only 0.780 on validation trips, a clear case of overfitting: the model copies the label of a single, often noisy, example. As k increases, the training-row score falls (0.878 at k = 5, 0.853 at k = 9) and validation Macro F1 rises to a plateau of about 0.800 at k = 5–9. Beyond k ≈ 15, validation Macro F1 declines (0.779 at k = 51) as the model averages over increasingly dissimilar candidates, which indicates underfitting. Log Loss keeps improving with larger k (0.489 at k = 1 to 0.107 at k = 51) because more neighbours give smoother, less extreme probabilities, so k = 9 trades some probability quality for the best Macro F1. With distance weighting, the training-row score is 1.000 for every k, because a stored row is always at distance 0 from itself and receives all the weight. That number is therefore not a measure of overfitting, and the uniform curve above is the meaningful comparison. The selected model generalised well to unseen trips: its student-test Macro F1 (0.828) is consistent with, and slightly above, its CV mean (0.801). Uniform k = 5 (CV Macro F1 0.800, CV Top-1 0.635) is within the fold-to-fold standard deviation of the selected configuration, so the choice between them is not significant.

**Validation:** Top-1 Accuracy varies strongly between folds (0.415 on fold 0 to 0.748 on fold 2) and between student-test trips (0.53 on trip 00000077 to 0.88 on trip 00000007). The hardest fold is the same as for every other model, so the variation reflects trip difficulty (GPS noise, parallel and divided roads, off-network dwelling and uncertain reference alignment) rather than instability of KNN. Unlike the class-weighted models, KNN has higher precision than recall (0.755 vs 0.599): with only 3.9% positive rows and no class weighting, a candidate is labelled positive only when most of its nine neighbours are true arcs, so the model is conservative. Its Macro F1 is close to Random Forest's (0.828 vs 0.829), but it ranks candidates within an observation worse (Top-1 0.713 vs 0.818) and produces less continuous routes (0.798 vs 0.898). A likely cause is that many candidates of the same observation receive a probability of exactly 0, so the ranking among them carries no information. KNN is also about 90 times slower at inference than Random Forest (5.9 s vs 66 ms per trip) because every candidate is compared with all 300,000 stored rows. For these reasons KNN was not selected as the final model, but it provides a useful instance-based baseline showing that local similarity in the shared features is already highly informative.

Plots: k validation curve results/figures/knn_validation_curve.png; CV versus student-test comparison results/figures/knn_cv_vs_test.png; Top-1 Accuracy per student-test trip results/figures/knn_per_trip_top1.png; confusion matrix results/figures/cm_knn.png; combined six-panel summary results/figures/knn_overview.png; model comparison charts in Section 3.7. Configuration, per-fold metrics and student-test metrics: results/metrics_knn.json. k sweep: results/knn_k_sweep.csv and results/knn_k_sweep_per_fold.csv. Predictions: results/predictions_knn.parquet. Training script: models/train_knn.py (python models/train_knn.py; --plots-only redraws the figures). Metric and plotting code: src/metrics.py. The trained model (artifacts/knn_model.joblib, 19 MB) is not committed because of its size; the training script recreates it.
