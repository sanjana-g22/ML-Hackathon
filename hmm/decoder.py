"""
PathSnap HMM & Viterbi Decoder package (Member 3).
"""
from backend.hmm_decoder import viterbi_map_match_trip
import numpy as np
import pandas as pd
from src.data_loader import TripData


def decode_trip(trip: TripData, network, candidates: pd.DataFrame, xy=None, valid=None) -> np.ndarray:
    """
    Decodes the most likely road arc sequence for a trip using HMM Viterbi.

    Args:
        trip: TripData instance (route may be None).
        network: SpatialNetwork instance.
        candidates: DataFrame with candidate features and classifier 'prob'.
        xy: Optional projected GPS points.
        valid: Optional valid GPS mask.

    Returns:
        np.ndarray of int arc IDs, one per GPS observation (-1 if undecided).
    """
    res = viterbi_map_match_trip(trip, candidates)
    return res["observation_arcs"]
