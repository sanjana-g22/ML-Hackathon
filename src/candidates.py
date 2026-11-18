"""Reference-free nearby-segment candidates."""
import numpy as np
import pandas as pd
from shapely import points
from .network import segment_projection


def generate_candidates(trip, network, radius=250.0, k=30):
    if radius <= 0 or k < 1:
        raise ValueError('radius must be positive and k must be >= 1')
    raw = trip.track[['longitude', 'latitude']].to_numpy(float)
    valid = np.isfinite(raw).all(1) & (abs(raw[:, 0]) <= 180) & (abs(raw[:, 1]) <= 90)
    xy = np.full_like(raw, np.nan)
    xy[valid] = network.project(raw[valid, 0], raw[valid, 1])
    valid &= np.isfinite(xy).all(1)
    indices = np.flatnonzero(valid)
    frames = []
    for start in range(0, len(indices), 4096):
        batch = indices[start:start + 4096]
        pairs = network.tree.query(points(xy[batch]), predicate='dwithin', distance=radius)
        if not pairs.shape[1]:
            continue
        obs, pos = batch[pairs[0]], pairs[1]
        distance, ratio = segment_projection(xy[obs], network.starts[pos], network.ends[pos])
        frame = pd.DataFrame({'obs_idx': obs, 'arc_id': network.arc_ids[pos], 'segment_pos': pos,
                              'distance_m': distance, 'projection_ratio': ratio})
        frame = frame.sort_values(['obs_idx', 'distance_m', 'arc_id'], kind='stable')
        frames.append(frame.groupby('obs_idx', sort=False).head(k))
    columns = ['obs_idx', 'arc_id', 'segment_pos', 'distance_m', 'projection_ratio']
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    for col in ['obs_idx', 'arc_id', 'segment_pos']:
        frame[col] = frame[col].astype('int64')
    return frame, xy, valid
