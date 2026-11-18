"""
Server-side evaluation. The ONLY place the API reads ground truth.

Nothing from this module is sent to the UI except aggregate metrics.

Per-point labels: PLACEHOLDER (nearest reference-route arc to each point).
Swap `point_labels` for M1's src.labels alignment once it is ready, so the API
scores exactly the way the report does.
"""
import numpy as np
from scipy.spatial import cKDTree

from src.data_loader import TripData, load_route
from backend.predictor import _xy


def point_labels(trip: TripData, route_arcs: np.ndarray) -> np.ndarray:
    lon0, lat0 = trip.track["longitude"].mean(), trip.track["latitude"].mean()
    nodes = _xy(trip.nodes["longitude"].to_numpy(), trip.nodes["latitude"].to_numpy(), lon0, lat0)
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()[route_arcs]
    mid = (nodes[arcs[:, 0]] + nodes[arcs[:, 1]]) / 2
    gps = _xy(trip.track["longitude"].to_numpy(), trip.track["latitude"].to_numpy(), lon0, lat0)
    _, idx = cKDTree(mid).query(gps)
    return route_arcs[idx]


def route_continuity(trip: TripData, pred: np.ndarray) -> float:
    """Share of consecutive predicted arcs that are identical or directly connected."""
    p = pred[pred >= 0]
    if len(p) < 2:
        return float("nan")
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    same = p[:-1] == p[1:]
    connected = arcs[p[:-1], 1] == arcs[p[1:], 0]
    return float(np.mean(same | connected))


def score(trip: TripData, route_path: str, pred: np.ndarray) -> dict:
    route = load_route(route_path)
    metrics = {"route_continuity": route_continuity(trip, pred)}
    if route is None:
        metrics.update({"top1_accuracy": None, "route_precision": None, "route_recall": None})
        return metrics

    route_arcs = route["arc_id"].to_numpy()
    labels = point_labels(trip, route_arcs)
    metrics["top1_accuracy"] = float(np.mean(pred == labels))

    pred_set, ref_set = set(pred[pred >= 0].tolist()), set(route_arcs.tolist())
    hit = len(pred_set & ref_set)
    metrics["route_precision"] = hit / max(len(pred_set), 1)
    metrics["route_recall"] = hit / max(len(ref_set), 1)
    return metrics
