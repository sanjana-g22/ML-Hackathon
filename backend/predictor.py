"""
Prediction pipeline used by the API.

IMPORTANT: this module must never read the .route file. It receives a trip whose
`route` has been removed (see backend.main) and returns one predicted arc_id per GPS
observation (-1 where it has no prediction).

Pipeline
    1. Candidates + features   src.member1_pipeline.inference_features (M1), the same
                               code and settings used to build the training data
    2. Candidate classifier    artifacts/<model>.joblib saved by src.metrics.run_experiment
                               (default: Random Forest; override with PATHSNAP_MODEL)
    3. Decoder                 hmm.decoder.decode_trip (M3) global Viterbi decoder;
                               fallback to classifier-only: highest-probability arc per point
"""
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import TripData
from src.member1_pipeline import inference_features

log = logging.getLogger("pathsnap")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", REPO_ROOT / "artifacts" / "rf_model.joblib"))
CANDIDATE_RADIUS_M = 250.0     # must match the training data (src.member1_pipeline defaults)
MAX_CANDIDATES = 30
HMM_TOP_K = int(os.environ.get("HMM_TOP_K", 10))   # candidates per point passed to the HMM (0 = all)

try:                            # M3's Viterbi decoder
    from hmm.decoder import decode_trip as _hmm_decode
except ImportError:
    try:
        from backend.hmm_decoder import viterbi_map_match_trip
        def _hmm_decode(trip, network, candidates, xy=None, valid=None):
            return viterbi_map_match_trip(trip, candidates)["observation_arcs"]
    except ImportError:
        _hmm_decode = None

DECODER_NAME = "HMM/Viterbi (Member 3)" if _hmm_decode else "classifier-only (argmax per point)"
_MODEL_NAMES = {
    "rf_model": "Random Forest",
    "best_model_tuned": "Tuned model",
    "lgbm_untuned": "LightGBM",
    "logistic_pipeline": "Logistic Regression",
    "knn_model": "KNN",
    "mlp_model": "MLP"
}
PIPELINE_NAME = f"{_MODEL_NAMES.get(MODEL_PATH.stem, MODEL_PATH.stem)} + {DECODER_NAME}"

_model = None


def load_model():
    """Load the classifier once. The saved file is {'model': estimator, 'features': [...]}."""
    global _model
    if _model is None:
        with warnings.catch_warnings():
            # artifacts may come from a slightly different scikit-learn version; predictions were
            # checked to match the recorded student-test metrics exactly
            warnings.simplefilter("ignore")
            artifact = joblib.load(MODEL_PATH)
        _model = (artifact["model"], list(artifact["features"]))
    return _model


@dataclass
class Prediction:
    arcs: np.ndarray                 # one arc_id per GPS observation, -1 = none
    candidates: pd.DataFrame         # candidates with features and classifier probability
    network: object                  # src.network.SpatialNetwork (reference-free)
    xy: np.ndarray                   # projected GPS points (metres)
    valid: np.ndarray                # usable GPS points
    timings_ms: dict = field(default_factory=dict)


def argmax_decode(candidates: pd.DataFrame, n_obs: int) -> np.ndarray:
    """Classifier-only route: the most probable candidate for each GPS point.
    Candidates are sorted by distance within a point, so ties go to the nearest road."""
    arcs = np.full(n_obs, -1, dtype=np.int64)
    if len(candidates):
        best = candidates.loc[candidates.groupby("obs_idx", sort=False)["prob"].idxmax()]
        arcs[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return arcs


def predict(trip: TripData) -> Prediction:
    assert trip.route is None, "predictor must not see the reference route"
    model, features = load_model()
    timings = {}

    t0 = time.perf_counter()
    candidates, network, xy, valid = inference_features(trip, CANDIDATE_RADIUS_M, MAX_CANDIDATES)
    timings["features_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if len(candidates) and hasattr(model, "predict_proba"):
        candidates["prob"] = model.predict_proba(candidates[features])[:, 1]
    elif len(candidates) and "distance_m" in candidates.columns:
        candidates["prob"] = np.exp(-candidates["distance_m"].to_numpy() / 50.0)
    else:
        candidates["prob"] = []
    timings["classifier_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    decoder_used, hmm_undecided = "classifier-only", 0
    if _hmm_decode is not None and len(candidates) > 0:
        # The HMM's cost is dominated by shortest-path transitions between every pair of candidates
        # of consecutive points; keeping each point's HMM_TOP_K most probable candidates gave
        # identical student-test results at top-10 while running 4-5x faster.
        hmm_candidates = (candidates.sort_values(["obs_idx", "prob"], ascending=[True, False])
                          .groupby("obs_idx", sort=False).head(HMM_TOP_K)
                          .sort_values(["obs_idx", "distance_m"])) if HMM_TOP_K else candidates
        try:
            arcs = np.asarray(_hmm_decode(trip, network, hmm_candidates, xy, valid), dtype=np.int64)
            if len(arcs) == trip.n_points:
                decoder_used = "hmm"
                unassigned = arcs < 0      # points the HMM left undecided: use the classifier's choice
                hmm_undecided = int(unassigned.sum())
                if hmm_undecided:
                    arcs[unassigned] = argmax_decode(candidates, trip.n_points)[unassigned]
            else:
                log.warning(f"trip {trip.trip_id}: HMM returned {len(arcs)} arcs for {trip.n_points} points; "
                            "using classifier-only")
                arcs = argmax_decode(candidates, trip.n_points)
        except Exception as e:             # never fail the request, but never hide the fallback either
            log.warning(f"trip {trip.trip_id}: HMM failed ({type(e).__name__}: {e}); using classifier-only")
            arcs = argmax_decode(candidates, trip.n_points)
    else:
        arcs = argmax_decode(candidates, trip.n_points)
    timings["decoder_ms"] = (time.perf_counter() - t0) * 1000

    info = {k: round(v, 1) for k, v in timings.items()}
    info.update({"decoder_used": decoder_used, "hmm_undecided_points": hmm_undecided})
    return Prediction(arcs, candidates, network, xy, valid, info)


def predict_arcs(trip: TripData) -> np.ndarray:
    return predict(trip).arcs
