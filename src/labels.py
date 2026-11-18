"""Reference-only monotone route-position alignment; never used to create features."""
import numpy as np
import pandas as pd
from .network import segment_projection


def align_reference(trip, network, xy, valid, unreliable_distance=50.):
    n = len(xy)
    output = pd.DataFrame({'obs_idx': np.arange(n), 'reference_arc_id': -1,
                           'route_position': -1, 'alignment_distance_m': np.nan,
                           'nearest_route_distance_m': np.nan, 'reference_available': False,
                           'unreliable': True})
    if trip.route is None or trip.route.empty or not valid.any():
        return output
    route = trip.route.arc_id.to_numpy(np.int64)
    lookup = {int(arc): i for i, arc in enumerate(network.arc_ids)}
    if any(int(arc) not in lookup for arc in route):
        raise ValueError(f'{trip.trip_id}: reference contains invalid/zero-length arcs; repair policy required')
    positions = np.array([lookup[int(arc)] for arc in route])
    starts, ends = network.starts[positions], network.ends[positions]
    obs_ids = np.flatnonzero(valid)
    m = len(route)
    back = np.empty((len(obs_ids), m), dtype=np.int32)
    previous = np.zeros(m)
    nearest = np.empty(len(obs_ids))
    # Prefix minima allow staying or advancing and preserve repeated route positions.
    for row, obs in enumerate(obs_ids):
        distances, _ = segment_projection(xy[obs], starts, ends)
        nearest[row] = distances.min()
        minima = np.minimum.accumulate(previous)
        improvements = np.r_[True, previous[1:] < minima[:-1]]
        parent = np.maximum.accumulate(np.where(improvements, np.arange(m), 0))
        back[row] = parent
        previous = distances + minima
        previous -= previous.min()
    selected = np.empty(len(obs_ids), dtype=int)
    state = int(previous.argmin())
    for row in range(len(obs_ids) - 1, -1, -1):
        selected[row] = state
        state = int(back[row, state])
    dist, _ = segment_projection(xy[obs_ids], starts[selected], ends[selected])
    output.loc[obs_ids, 'reference_arc_id'] = route[selected]
    output.loc[obs_ids, 'route_position'] = selected
    output.loc[obs_ids, 'alignment_distance_m'] = dist
    output.loc[obs_ids, 'nearest_route_distance_m'] = nearest
    output.loc[obs_ids, 'reference_available'] = True
    output.loc[obs_ids, 'unreliable'] = dist > unreliable_distance
    return output
