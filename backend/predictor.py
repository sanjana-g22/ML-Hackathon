"""
Prediction pipeline used by the API.

IMPORTANT: this module must never read the .route file. It receives a trip
whose `route` has been removed (see backend.main) and returns one predicted arc_id
per GPS observation (-1 where it has no prediction).

Current status: PLACEHOLDER baseline (nearest road arc to each GPS point) so
the API/UI can be built and tested before the real pipeline is ready.
Replace the body of `predict_arcs` with:
    candidates/features (M1)  ->  tuned classifier (M2)  ->  Viterbi decode (M3)
"""
import numpy as np
from scipy.spatial import cKDTree

from src.data_loader import TripData

EARTH_RADIUS_M = 6_371_000.0
PIPELINE_NAME = "placeholder: nearest-arc baseline"


def _xy(lon, lat, lon0, lat0):
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def predict_arcs(trip: TripData) -> np.ndarray:
    assert trip.route is None, "predictor must not see the reference route"

    lon0, lat0 = trip.track["longitude"].mean(), trip.track["latitude"].mean()
    nodes = _xy(trip.nodes["longitude"].to_numpy(), trip.nodes["latitude"].to_numpy(), lon0, lat0)
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    gps = _xy(trip.track["longitude"].to_numpy(), trip.track["latitude"].to_numpy(), lon0, lat0)

    # nearest arc midpoint - crude, but needs no trained model
    mid = (nodes[arcs[:, 0]] + nodes[arcs[:, 1]]) / 2
    _, idx = cKDTree(mid).query(gps)
    return idx.astype(int)
