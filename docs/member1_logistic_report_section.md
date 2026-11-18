# Model: Logistic Regression (Member 1)

Fill the numbers from `results/member1/metrics_logistic.json` and `results/model_comparison_summary.csv`
after running. Do not report any number you did not produce.

## A. Model understanding
1. **Why suitable:** each candidate (GPS point, nearby arc) is a binary decision, "is this the true arc?". Features are geometric (distance, heading, speed, arc degree). LR is a simple linear baseline that outputs probabilities, which the HMM can use as emission scores.
2. **Strengths:** fast, convex, interpretable coefficients, probabilities usable by Viterbi. **Limitations:** linear boundary, so it cannot model interactions such as "close but wrong heading". It has no sequence awareness. It is sensitive to scaling, and `class_weight=balanced` inflates probabilities, which hurts log loss.

## B. Training and configuration
- Initial hyperparameters: C=1.0, L2, lbfgs, max_iter=500, class_weight=balanced; median imputer + StandardScaler inside a Pipeline, fit on training folds only.
- Objective: L2-regularised binary cross-entropy (log loss).
- Protocol: 5-fold GroupKFold by trip on the 80 train trips; refit on all train; evaluate on the 10 student_test trips.

## C. Results (fill in)
- Student-test Top-1 / Macro F1 / Log Loss / Route continuity / Sequence F1: ___
- CV Macro F1 (mean ± std): ___ ; train-side F1: ___
- Overfitting/underfitting: compare train-side vs CV vs student_test F1 and Top-1 in the JSON (per-fold values are in `cv.folds`). A weak score on both train and validation indicates underfitting, which is expected for a linear model. Large fold-to-fold swings show sensitivity to trip type. State what YOU observe.
- Note: train-side metrics use 10 whole trips per fold (`--train-eval-trips`) for speed. Validation and student_test use all rows.
- Figures: `results/member1/cm_logistic.png`; comparison charts via `python src/metrics.py`.
