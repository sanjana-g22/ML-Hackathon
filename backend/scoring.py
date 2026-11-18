"""
Server-side evaluation. The ONLY place the API reads ground truth.

Nothing from this module is sent to the UI except aggregate metrics.

Per-point reference labels come from M1's monotone route alignment
(src.labels.align_reference), the same labels used to train and evaluate the models,
so the API reports the same Top-1 definition as the report.
"""
import dataclasses
import logging

import numpy as np

from src.data_loader import TripData, load_route
from src.labels import align_reference
from src.metrics import sequence_f1

log = logging.getLogger("pathsnap")


def route_continuity(trip: TripData, pred: np.ndarray) -> float:
    """Share of consecutive predicted arcs that are identical or directly connected."""
    p = pred[pred >= 0]
    if len(p) < 2:
        return float("nan")
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    same = p[:-1] == p[1:]
    connected = arcs[p[:-1], 1] == arcs[p[1:], 0]
    return float(np.mean(same | connected))


def score(trip: TripData, route_path: str, prediction) -> dict:
    pred = prediction.arcs
    metrics = {"route_continuity": route_continuity(trip, pred),
               "top1_accuracy": None, "top1_reliable_points": None, "unreliable_point_share": None,
               "route_precision": None, "route_recall": None, "sequence_f1": None}
    route = load_route(route_path)
    if route is None:
        return metrics

    route_arcs = route["arc_id"].to_numpy()
    pred_set, ref_set = set(pred[pred >= 0].tolist()), set(route_arcs.tolist())
    hit = len(pred_set & ref_set)
    metrics["route_precision"] = hit / max(len(pred_set), 1)
    metrics["route_recall"] = hit / max(len(ref_set), 1)
    metrics["sequence_f1"] = float(sequence_f1(pred[pred >= 0], route_arcs))

    try:
        labelled = dataclasses.replace(trip, route=route)
        ref = align_reference(labelled, prediction.network, prediction.xy, prediction.valid)
    except ValueError as e:      # e.g. reference uses zero-length arcs that the network drops
        log.warning(f"trip {trip.trip_id}: per-point labels unavailable ({e})")
        return metrics

    known = ref["reference_available"].to_numpy()
    correct = pred == ref["reference_arc_id"].to_numpy()
    reliable = known & ~ref["unreliable"].to_numpy()
    metrics["top1_accuracy"] = float(correct[known].mean()) if known.any() else None
    metrics["top1_reliable_points"] = float(correct[reliable].mean()) if reliable.any() else None
    metrics["unreliable_point_share"] = float(1 - reliable.sum() / max(known.sum(), 1))
    return metrics
