# PathSnap

Map-matching GPS tracks to road arcs. Each GPS observation has a set of candidate arcs, and we train classifiers to pick the true one.

## Structure

```
src/
  data_loader.py      load raw trip files (.track, .nodes, .arcs, .route)
  metrics.py          metrics, trip-grouped CV, shared train/eval loop, comparison table
models/
  train_rf.py         Random Forest
  train_lgbm.py       LightGBM (untuned) + loss curve
tuning/
  optuna_tuning.py    Optuna TPE search for LightGBM
train/, student_test/ raw trip folders
docs/                 assignment documents
```

Outputs go to `results/` (predictions, metrics JSON, CSV tables, `figures/`) and `artifacts/` (saved models).

## Setup

```
pip install -r requirements.txt
```

## Input data

`data/candidates.parquet`, one row per (observation, candidate arc):

| column | description |
|---|---|
| `trip_id` | trip folder name |
| `split` | `train` or `student_test` |
| `obs_idx` | index of the GPS point in the trip |
| `t` | timestamp |
| `arc_id` | candidate arc (row number in the trip's `.arcs` file, local to the trip) |
| `label` | 1 if this is the true arc, else 0 |
| other numeric columns | features |

Columns `trip_id, split, obs_idx, t, arc_id, label, true_arc_id, is_missed` are never used as features.

Route continuity and sequence F1 read `.arcs` and `.route` from `train/<trip_id>/` / `student_test/<trip_id>/` (or `../Student folder/`).

## Running

```
python models/train_rf.py
python models/train_lgbm.py
python tuning/optuna_tuning.py --n-trials 50
python src/metrics.py
```

`optuna_tuning.py` also re-trains the untuned LightGBM baseline so both models use the same folds, and writes `results/untuned_vs_tuned_comparison.csv`. `src/metrics.py` builds `results/model_comparison_summary.csv` and the comparison charts from every `results/predictions_*.parquet`.

All CV is 5-fold `GroupKFold` grouped by `trip_id`, on `train` only. `student_test` is used once for final evaluation.
