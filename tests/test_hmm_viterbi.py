"""
PathSnap - Unit and Integration Tests for HMM & Viterbi (Member 3)
==================================================================
Comprehensive test suite validating:
- Numerical stability and error handling of emission probabilities
- Directed transition scoring and one-way road restrictions
- Dijkstra shortest-path calculations and distance caching
- Manually verifiable Viterbi optimal sequences
- Variable candidate counts per observation
- Single-observation trips
- Multiple independent trips
- Missing candidate observations
- Intermediate road arc reconstruction
- Parallel directed edges
- Legitimate route loops (no global deduplication)
- Deterministic tie-breaking
- Disconnected transitions and no feasible route scenarios
- Observation-level and continuous route continuity metrics
- Model prediction DataFrame compatibility
- Prevention of ground-truth leakage into inference
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.road_network import RoadNetwork, build_road_network
from src.hmm import (
    HMMConfig,
    TransitionScorer,
    compute_emission_log_scores,
)
from src.viterbi import (
    ViterbiResult,
    viterbi_decode,
)
from src.route_reconstruction import (
    ReconstructedRoute,
    reconstruct_continuous_route,
    compute_route_similarity_lcs,
)
from src.map_matching import match_trip_candidates


# ==============================================================================
# Helper Fixtures: Toy Road Networks
# ==============================================================================

def create_linear_toy_network() -> RoadNetwork:
    """
    Creates a simple 4-node, 3-arc directed linear network:
    Node 0 (0.0, 0.0) -> Arc 0 -> Node 1 (0.001, 0.0) -> Arc 1 -> Node 2 (0.002, 0.0) -> Arc 2 -> Node 3 (0.003, 0.0)
    All arcs are one-way directed from left to right.
    """
    nodes_df = pd.DataFrame({
        "longitude": [0.0, 0.001, 0.002, 0.003],
        "latitude": [0.0, 0.0, 0.0, 0.0]
    })
    arcs_df = pd.DataFrame({
        "from_node_id": [0, 1, 2],
        "to_node_id": [1, 2, 3]
    })
    return build_road_network(nodes_df, arcs_df, trip_id="toy_linear")


def create_network_with_parallel_and_loop() -> RoadNetwork:
    """
    Creates a 4-node network:
    - Node 0 -> Node 1 (Arc 0: direct)
    - Node 1 -> Node 2 (Arc 1: short, Arc 2: parallel longer via intermediate geometry/weight)
    - Node 2 -> Node 3 (Arc 3: direct)
    - Node 3 -> Node 0 (Arc 4: loop back)
    """
    nodes_df = pd.DataFrame({
        "longitude": [0.0, 0.001, 0.002, 0.003],
        "latitude": [0.0, 0.0, 0.0, 0.0]
    })
    # Arcs: 0: 0->1, 1: 1->2, 2: 1->2 (parallel), 3: 2->3, 4: 3->0 (loop)
    arcs_df = pd.DataFrame({
        "from_node_id": [0, 1, 1, 2, 3],
        "to_node_id": [1, 2, 2, 3, 0]
    })
    net = build_road_network(nodes_df, arcs_df, trip_id="toy_parallel_loop")
    # Manually assign length_m for parallel arc 2 to be longer than arc 1
    edge_data_1 = net.graph.get_edge_data(1, 2, key=1)
    edge_data_2 = net.graph.get_edge_data(1, 2, key=2)
    edge_data_1["length_m"] = 100.0
    edge_data_2["length_m"] = 250.0
    return net


# ==============================================================================
# 1. Emission Probability Tests
# ==============================================================================

def test_emission_probabilities_basic_and_normalization():
    probs = np.array([0.7, 0.2, 0.1])
    emissions = compute_emission_log_scores(probs, normalize=True)
    assert len(emissions) == 3
    # Sum of exp(emissions) should equal 1.0
    assert np.isclose(np.sum(np.exp(emissions)), 1.0, atol=1e-6)
    # Highest probability candidate must have highest log score
    assert np.argmax(emissions) == 0


def test_emission_zero_and_negative_stability():
    probs = np.array([0.0, -0.5, np.nan, np.inf, 1.0])
    emissions = compute_emission_log_scores(probs, normalize=False, eps=1e-7)
    # No NaNs or infinities allowed
    assert not np.isnan(emissions).any()
    assert not np.isneginf(emissions).any()
    assert not np.isposinf(emissions).any()
    # All log values should be <= 0.0
    assert (emissions <= 0.0).all()


# ==============================================================================
# 2. Transition Scoring & Directed Constraints Tests
# ==============================================================================

def test_transition_directed_edges_and_oneway():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net, config=HMMConfig(beta=30.0, unreachable_penalty=-9999.0))

    # Arc 0 (0->1) to Arc 1 (1->2): Forward valid transition
    score_fwd = scorer.transition_score(from_arc_id=0, to_arc_id=1, d_gps=111.0)
    assert score_fwd > -500.0, "Forward transition should be reachable"

    # Arc 1 (1->2) to Arc 0 (0->1): Backward traversal on directed one-way road
    score_bwd = scorer.transition_score(from_arc_id=1, to_arc_id=0, d_gps=111.0)
    assert score_bwd <= -9000.0, "Backward transition on one-way must be penalized as unreachable"


def test_transition_shortest_path_caching():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net, config=HMMConfig(beta=30.0))

    assert len(scorer._distance_cache) == 0
    d1 = scorer.shortest_path_distance(0, 2)
    assert d1 is not None
    assert len(scorer._distance_cache) == 1

    # Second call should hit cache
    d2 = scorer.shortest_path_distance(0, 2)
    assert d1 == d2
    assert len(scorer._distance_cache) == 1


# ==============================================================================
# 3. Viterbi Algorithm Tests
# ==============================================================================

def test_manually_verifiable_viterbi_sequence():
    """
    Scenario:
    2 observations.
    Obs 0 has candidates:
      - Arc 1 (1->2), prob = 0.9 (Local argmax!)
      - Arc 0 (0->1), prob = 0.4
    Obs 1 has candidates:
      - Arc 1 (1->2), prob = 0.9

    If the vehicle moved from Node 0 to Node 2:
    Arc 1 cannot transition to Arc 1 with positive GPS displacement without exiting and looping around!
    Arc 0 -> Arc 1 is a direct connected transition with matching GPS distance.
    The HMM should penalize Arc 1 -> Arc 1 backward/stall and choose Arc 0 -> Arc 1!
    """
    net = create_linear_toy_network()
    scorer = TransitionScorer(net, config=HMMConfig(beta=20.0))

    observations = [
        {"obs_idx": 0, "longitude": 0.0005, "latitude": 0.0, "timestamp": 0.0},
        {"obs_idx": 1, "longitude": 0.0015, "latitude": 0.0, "timestamp": 10.0}
    ]
    candidates_by_obs = [
        [
            {"arc_id": 1, "prob": 0.9, "projection_ratio": 0.5},
            {"arc_id": 0, "prob": 0.4, "projection_ratio": 0.5}
        ],
        [
            {"arc_id": 1, "prob": 0.9, "projection_ratio": 0.5}
        ]
    ]

    res = viterbi_decode("toy_trip", observations, candidates_by_obs, scorer)
    assert res.selected_arcs == [0, 1], f"Expected [0, 1] but got {res.selected_arcs}"


def test_viterbi_single_observation_trip():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net)

    observations = [{"obs_idx": 0, "longitude": 0.0, "latitude": 0.0, "timestamp": 0.0}]
    candidates_by_obs = [[
        {"arc_id": 0, "prob": 0.3},
        {"arc_id": 1, "prob": 0.8},
        {"arc_id": 2, "prob": 0.1}
    ]]

    res = viterbi_decode("single_obs", observations, candidates_by_obs, scorer)
    assert res.selected_arcs == [1]
    assert len(res.selected_arcs) == 1
    assert res.diagnostics["n_observations"] == 1


def test_viterbi_variable_candidate_counts():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net)

    observations = [
        {"obs_idx": 0, "longitude": 0.0, "latitude": 0.0},
        {"obs_idx": 1, "longitude": 0.001, "latitude": 0.0},
        {"obs_idx": 2, "longitude": 0.002, "latitude": 0.0}
    ]
    # Obs 0 has 3 cands, Obs 1 has 1 cand, Obs 2 has 2 cands
    candidates_by_obs = [
        [{"arc_id": 0, "prob": 0.5}, {"arc_id": 1, "prob": 0.3}, {"arc_id": 2, "prob": 0.2}],
        [{"arc_id": 1, "prob": 0.9}],
        [{"arc_id": 2, "prob": 0.8}, {"arc_id": 0, "prob": 0.2}]
    ]

    res = viterbi_decode("var_cands", observations, candidates_by_obs, scorer)
    assert len(res.selected_arcs) == 3
    assert res.selected_arcs == [0, 1, 2]


def test_viterbi_missing_candidate_observations():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net)

    observations = [
        {"obs_idx": 0, "longitude": 0.0, "latitude": 0.0},
        {"obs_idx": 1, "longitude": 0.001, "latitude": 0.0},
        {"obs_idx": 2, "longitude": 0.002, "latitude": 0.0}
    ]
    # Obs 1 has 0 candidates
    candidates_by_obs = [
        [{"arc_id": 0, "prob": 0.8}],
        [],
        [{"arc_id": 2, "prob": 0.8}]
    ]

    res = viterbi_decode("missing_obs", observations, candidates_by_obs, scorer)
    assert len(res.selected_arcs) == 3
    assert res.diagnostics["missing_candidate_steps"] == 1
    assert res.selected_arcs[0] == 0
    assert res.selected_arcs[2] == 2


def test_viterbi_deterministic_tie_breaking():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net)

    observations = [
        {"obs_idx": 0, "longitude": 0.0, "latitude": 0.0},
        {"obs_idx": 1, "longitude": 0.001, "latitude": 0.0}
    ]
    # Two identical candidates with identical probabilities
    candidates_by_obs = [
        [{"arc_id": 0, "prob": 0.5}, {"arc_id": 1, "prob": 0.5}],
        [{"arc_id": 2, "prob": 0.5}]
    ]

    res1 = viterbi_decode("tie_1", observations, candidates_by_obs, scorer)
    res2 = viterbi_decode("tie_2", observations, candidates_by_obs, scorer)
    assert res1.selected_arcs == res2.selected_arcs, "Tie breaking must be completely deterministic"


def test_viterbi_all_unreachable_disconnected_transitions():
    net = create_linear_toy_network()
    scorer = TransitionScorer(net, config=HMMConfig(unreachable_penalty=-10000.0))

    observations = [
        {"obs_idx": 0, "longitude": 0.002, "latitude": 0.0},
        {"obs_idx": 1, "longitude": 0.0, "latitude": 0.0}
    ]
    # Arc 2 (2->3) cannot reach Arc 0 (0->1)
    candidates_by_obs = [
        [{"arc_id": 2, "prob": 0.9}],
        [{"arc_id": 0, "prob": 0.9}]
    ]

    res = viterbi_decode("disconnected", observations, candidates_by_obs, scorer)
    assert len(res.selected_arcs) == 2
    assert res.diagnostics["disconnected_transitions"] > 0


# ==============================================================================
# 4. Continuous Route Reconstruction Tests
# ==============================================================================

def test_continuous_route_reconstruction_intermediate_arcs():
    net = create_linear_toy_network()
    # Selected arcs only at endpoint observations: Arc 0 (0->1) and Arc 2 (2->3)
    # Intermediate Arc 1 (1->2) was skipped by GPS observation
    selected_arcs = [0, 2]
    reconstructed = reconstruct_continuous_route("route_inter", selected_arcs, net)

    assert reconstructed.continuous_route_arcs == [0, 1, 2], (
        f"Intermediate arc 1 should be inserted: got {reconstructed.continuous_route_arcs}"
    )
    assert reconstructed.intermediate_arcs_added == 1
    assert reconstructed.diagnostics["route_continuity"] == 1.0


def test_continuous_route_reconstruction_parallel_arcs():
    net = create_network_with_parallel_and_loop()
    # Node 0 -> Node 1 (Arc 0), then Node 2 -> Node 3 (Arc 3)
    # Between Node 1 and Node 2, there are parallel arcs 1 (100m) and 2 (250m)
    selected_arcs = [0, 3]
    reconstructed = reconstruct_continuous_route("route_parallel", selected_arcs, net)

    assert 1 in reconstructed.continuous_route_arcs
    assert 2 not in reconstructed.continuous_route_arcs, "Should pick shorter parallel arc 1"


def test_continuous_route_reconstruction_preserves_legitimate_loops():
    net = create_network_with_parallel_and_loop()
    # Vehicle traverses: 0->1 (Arc 0), 1->2 (Arc 1), 2->3 (Arc 3), 3->0 (Arc 4), 0->1 (Arc 0)
    selected_arcs = [0, 1, 3, 4, 0]
    reconstructed = reconstruct_continuous_route("route_loop", selected_arcs, net)

    # Arc 0 must appear twice! No global deduplication!
    assert reconstructed.continuous_route_arcs.count(0) == 2
    assert reconstructed.continuous_route_arcs == [0, 1, 3, 4, 0]


def test_route_continuity_and_similarity_metrics():
    # Ground truth route: [0, 1, 2]
    gt_route = [0, 1, 2]
    pred_perfect = [0, 1, 2]
    pred_partial = [0, 1]

    sim_perfect = compute_route_similarity_lcs(pred_perfect, gt_route)
    assert sim_perfect["route_precision"] == 1.0
    assert sim_perfect["route_recall"] == 1.0
    assert sim_perfect["sequence_f1"] == 1.0

    sim_partial = compute_route_similarity_lcs(pred_partial, gt_route)
    assert sim_partial["route_precision"] == 1.0
    assert sim_partial["route_recall"] == 2.0 / 3.0
    assert np.isclose(sim_partial["sequence_f1"], 2 * 1.0 * (2/3) / (1.0 + 2/3))


# ==============================================================================
# 5. Integration and Model Compatibility Tests
# ==============================================================================

def test_compatibility_with_model_predictions_dataframe():
    net = create_linear_toy_network()
    # Mock candidate predictions DataFrame matching team schema
    cand_df = pd.DataFrame({
        "trip_id": ["toy_trip", "toy_trip", "toy_trip", "toy_trip"],
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [0, 1, 1, 2],
        "prob": [0.8, 0.2, 0.9, 0.1],
        "projection_ratio": [0.5, 0.5, 0.5, 0.5],
        "model_name": ["mlp", "mlp", "mlp", "mlp"]
    })

    result = match_trip_candidates(
        trip_id="toy_trip",
        candidates_df=cand_df,
        road_network=net,
        ground_truth_route=[0, 1]
    )

    assert result["trip_id"] == "toy_trip"
    assert result["viterbi_result"].selected_arcs == [0, 1]
    assert result["reconstructed_route"].continuous_route_arcs == [0, 1]
    assert "comparison_metrics" in result
    assert result["comparison_metrics"]["hmm_sequence_f1"] == 1.0


def test_no_ground_truth_leakage():
    """Verifies map-matching and reconstruction operate completely without ground-truth .route."""
    net = create_linear_toy_network()
    cand_df = pd.DataFrame({
        "trip_id": ["blind_trip", "blind_trip"],
        "obs_idx": [0, 1],
        "arc_id": [0, 1],
        "prob": [0.9, 0.9]
    })
    # ground_truth_route=None
    result = match_trip_candidates(
        trip_id="blind_trip",
        candidates_df=cand_df,
        road_network=net,
        ground_truth_route=None
    )
    assert result["viterbi_result"].selected_arcs == [0, 1]
    assert "ground_truth_arcs" not in result["comparison_metrics"]
