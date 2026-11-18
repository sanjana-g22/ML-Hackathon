"""
PathSnap - Viterbi Decoding Algorithm (Member 3)
================================================
Implements dynamic programming in log-space to find the most probable
sequence of candidate road arcs for a GPS trajectory under the spatial HMM.

Mathematical Formulation:
-------------------------
For observations t = 0 ... T-1 and candidate states i = 0 ... K_t - 1:
Initialization (t = 0):
    V_0(i) = log P(z_0 | a_{0,i})
    B_0(i) = -1

Recursion (t = 1 ... T-1):
    V_t(i) = max_{j} [ V_{t-1}(j) + log P(a_{t,i} | a_{t-1,j}) ] + log P(z_t | a_{t,i})
    B_t(i) = argmax_{j} [ V_{t-1}(j) + log P(a_{t,i} | a_{t-1,j}) ]

Termination:
    i*_T-1 = argmax_{i} V_{T-1}(i)
    log_likelihood = max_{i} V_{T-1}(i)

Backtracking:
    i*_t = B_{t+1}(i*_{t+1}) for t = T-2 ... 0

Features:
- Numerically stable log-space calculations (avoids underflow).
- Variable candidate counts per observation (K_t >= 0).
- Support for single-observation trips (T = 1).
- Graceful handling of missing candidates (K_t = 0) via state propagation.
- Graceful handling of fully disconnected transition matrices.
- Deterministic tie-breaking for reproducible outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import numpy as np
import pandas as pd

from src.hmm import HMMConfig, TransitionScorer, compute_emission_log_scores
from src.road_network import RoadNetwork, calculate_geodesic_distance


@dataclass
class ViterbiResult:
    """Encapsulates the output of Viterbi map-matching for a single trip."""
    trip_id: str
    selected_arcs: List[int]                     # Optimal arc_id per observation (len == T)
    selected_candidates: List[Dict[str, Any]]    # Metadata/features for each selected candidate
    log_likelihood: float                        # Cumulative log score of optimal path
    step_scores: List[float]                     # Log scores at each observation step
    diagnostics: Dict[str, Any]                  # Timing, continuity, and gap metrics


def viterbi_decode(
    trip_id: str,
    observations: List[Dict[str, Any]],
    candidates_by_obs: List[List[Dict[str, Any]]],
    transition_scorer: TransitionScorer,
    config: Optional[HMMConfig] = None
) -> ViterbiResult:
    """
    Decodes the most likely candidate road arc sequence for a GPS trip.

    Args:
        trip_id: Unique trip identifier string.
        observations: List of observation dicts with keys:
                      ['obs_idx', 'longitude', 'latitude', 'timestamp']
        candidates_by_obs: List of candidate lists, where candidates_by_obs[t] contains
                           candidates for observation t with keys:
                           ['arc_id', 'prob', 'projection_ratio', ...]
        transition_scorer: Initialized TransitionScorer for network distance.
        config: HMMConfig hyperparameters (or uses transition_scorer.config).

    Returns:
        ViterbiResult with selected arcs, log-likelihood, and diagnostics.
    """
    t_start = time.perf_counter()
    cfg = config or transition_scorer.config

    T = len(observations)
    if T == 0:
        return ViterbiResult(
            trip_id=trip_id,
            selected_arcs=[],
            selected_candidates=[],
            log_likelihood=float("-inf"),
            step_scores=[],
            diagnostics={
                "n_observations": 0,
                "n_candidates_total": 0,
                "disconnected_transitions": 0,
                "missing_candidate_steps": 0,
                "execution_time_ms": 0.0
            }
        )

    # 1. Pre-extract Coordinates and Timestamps
    coords = [
        (float(obs["longitude"]), float(obs["latitude"]))
        for obs in observations
    ]
    timestamps = [
        float(obs["timestamp"]) if "timestamp" in obs and obs["timestamp"] is not None else None
        for obs in observations
    ]

    # Handle missing candidates or empty steps
    # Store indices of valid observation steps
    n_missing_candidates = 0
    total_candidates = sum(len(c) for c in candidates_by_obs)

    # Compute emission scores for each observation step
    emissions_by_step: List[np.ndarray] = []
    for t in range(T):
        cand_list = candidates_by_obs[t]
        if len(cand_list) == 0:
            n_missing_candidates += 1
            emissions_by_step.append(np.empty(0, dtype=np.float64))
        else:
            probs = np.array([float(c.get("prob", 0.5)) for c in cand_list], dtype=np.float64)
            emissions = compute_emission_log_scores(probs, normalize=cfg.normalize_emissions, eps=cfg.eps)
            emissions_by_step.append(emissions)

    # If all observations have 0 candidates
    if total_candidates == 0:
        return ViterbiResult(
            trip_id=trip_id,
            selected_arcs=[-1] * T,
            selected_candidates=[{}] * T,
            log_likelihood=float("-inf"),
            step_scores=[float("-inf")] * T,
            diagnostics={
                "n_observations": T,
                "n_candidates_total": 0,
                "disconnected_transitions": 0,
                "missing_candidate_steps": T,
                "execution_time_ms": (time.perf_counter() - t_start) * 1000.0
            }
        )

    # 2. Dynamic Programming Tables
    # V[t] will be a 1D array of shape (K_t,)
    # B[t] will be a 1D array of shape (K_t,) storing predecessor state index j in V[prev_valid_t]
    V: List[Optional[np.ndarray]] = [None] * T
    B: List[Optional[np.ndarray]] = [None] * T
    prev_step_map: List[Optional[int]] = [None] * T  # maps t to the previous valid step index

    disconnected_transitions_count = 0
    last_valid_step: Optional[int] = None

    # Step 0: Find first step with non-empty candidates
    for t in range(T):
        k_t = len(candidates_by_obs[t])
        if k_t == 0:
            continue

        if last_valid_step is None:
            # Initialization
            V[t] = emissions_by_step[t].copy()
            B[t] = np.full(k_t, -1, dtype=np.int64)
            prev_step_map[t] = None
            last_valid_step = t
        else:
            # Recurrence from last_valid_step to t
            prev_t = last_valid_step
            prev_V = V[prev_t]
            k_prev = len(candidates_by_obs[prev_t])

            # Elapsed time and GPS displacement
            dt_s = None
            if timestamps[t] is not None and timestamps[prev_t] is not None:
                dt_s = max(0.0, timestamps[t] - timestamps[prev_t])

            d_gps = calculate_geodesic_distance(
                coords[prev_t][0], coords[prev_t][1],
                coords[t][0], coords[t][1]
            )

            # Compute transition score matrix of shape (k_prev, k_t)
            trans_matrix = transition_scorer.compute_transition_matrix(
                prev_candidates=candidates_by_obs[prev_t],
                curr_candidates=candidates_by_obs[t],
                prev_coords=coords[prev_t],
                curr_coords=coords[t],
                dt_s=dt_s
            )

            # V_t(i) = max_j [ prev_V(j) + trans_matrix(j, i) ] + emission(i)
            # Shapes: prev_V[:, None] is (k_prev, 1), trans_matrix is (k_prev, k_t)
            candidate_step_scores = prev_V[:, None] + trans_matrix  # (k_prev, k_t)

            curr_V = np.empty(k_t, dtype=np.float64)
            curr_B = np.empty(k_t, dtype=np.int64)

            for i in range(k_t):
                incoming_scores = candidate_step_scores[:, i]
                max_score = np.max(incoming_scores)

                # Check if all transitions into state i are unreachable
                if max_score <= cfg.unreachable_penalty:
                    disconnected_transitions_count += 1
                    # Fallback: tie-break using previous V alone + emission
                    best_j = int(np.argmax(prev_V))
                    curr_B[i] = best_j
                    curr_V[i] = prev_V[best_j] + cfg.unreachable_penalty + emissions_by_step[t][i]
                else:
                    # Deterministic tie-breaking:
                    # Find all indices matching max_score within numerical tolerance
                    best_indices = np.where(np.isclose(incoming_scores, max_score, atol=1e-12, rtol=1e-10))[0]
                    # Select the smallest index deterministically
                    best_j = int(best_indices[0])
                    curr_B[i] = best_j
                    curr_V[i] = incoming_scores[best_j] + emissions_by_step[t][i]

            V[t] = curr_V
            B[t] = curr_B
            prev_step_map[t] = prev_t
            last_valid_step = t

    # 3. Termination and Backtracking
    if last_valid_step is None:
        # No valid steps
        selected_arcs = [-1] * T
        selected_cands = [{}] * T
        total_score = float("-inf")
    else:
        # Optimal terminal state at last_valid_step
        last_V = V[last_valid_step]
        max_term_score = np.max(last_V)
        # Deterministic tie-breaking on termination
        term_best_indices = np.where(np.isclose(last_V, max_term_score, atol=1e-12, rtol=1e-10))[0]
        curr_state_idx = int(term_best_indices[0])
        total_score = float(max_term_score)

        # Backtrack along the valid steps
        selected_arcs_map: Dict[int, int] = {}
        selected_cands_map: Dict[int, Dict[str, Any]] = {}
        step_scores_map: Dict[int, float] = {}

        curr_t = last_valid_step
        while curr_t is not None:
            cand = candidates_by_obs[curr_t][curr_state_idx]
            arc_id = int(cand.get("arc_id", cand.get("candidate_arc_id", -1)))
            selected_arcs_map[curr_t] = arc_id
            selected_cands_map[curr_t] = cand
            step_scores_map[curr_t] = float(V[curr_t][curr_state_idx])

            pred_j = int(B[curr_t][curr_state_idx])
            curr_state_idx = pred_j
            curr_t = prev_step_map[curr_t]

        # Reconstruct full length-T sequences, filling any missing steps
        selected_arcs = []
        selected_cands = []
        step_scores = []

        last_known_arc = -1
        last_known_cand = {}
        for t in range(T):
            if t in selected_arcs_map:
                arc_id = selected_arcs_map[t]
                cand = selected_cands_map[t]
                score = step_scores_map[t]
                last_known_arc = arc_id
                last_known_cand = cand
            else:
                # Interpolate or carry forward if step was missing candidates
                arc_id = last_known_arc
                cand = last_known_cand
                score = float("-inf")

            selected_arcs.append(arc_id)
            selected_cands.append(cand)
            step_scores.append(score)

    execution_time_ms = (time.perf_counter() - t_start) * 1000.0

    return ViterbiResult(
        trip_id=trip_id,
        selected_arcs=selected_arcs,
        selected_candidates=selected_cands,
        log_likelihood=total_score,
        step_scores=step_scores,
        diagnostics={
            "n_observations": T,
            "n_candidates_total": total_candidates,
            "disconnected_transitions": disconnected_transitions_count,
            "missing_candidate_steps": n_missing_candidates,
            "execution_time_ms": execution_time_ms
        }
    )
