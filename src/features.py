"""Features available identically during training and inference."""
import numpy as np
from .candidates import generate_candidates

FEATURE_COLUMNS = ['distance_m', 'projection_ratio', 'projection_distance_m', 'arc_length_m',
                   'heading_difference_rad', 'heading_cosine', 'speed_mps', 'dt_s',
                   'from_in_degree', 'from_out_degree', 'to_in_degree', 'to_out_degree',
                   'motion_valid', 'time_valid']


def make_features(trip, network, radius=250., k=30):
    frame, xy, valid = generate_candidates(trip, network, radius, k)
    n = len(xy)
    t = trip.track.timestamp.to_numpy(float)
    dt = np.r_[np.nan, np.diff(t)]
    step = np.vstack([np.full((1, 2), np.nan), np.diff(xy, axis=0)]) if n else np.empty((0, 2))
    distance = np.linalg.norm(step, axis=1)
    time_valid = np.isfinite(dt) & (dt > 0)
    motion_valid = time_valid & np.isfinite(distance) & (distance > .5)
    speed = np.divide(distance, dt, out=np.full(n, np.nan), where=time_valid)
    heading = np.arctan2(step[:, 0], step[:, 1])
    heading[~motion_valid] = np.nan
    obs = frame.obs_idx.to_numpy(int)
    pos = frame.segment_pos.to_numpy(int)
    difference = np.abs((heading[obs] - network.bearings[pos] + np.pi) % (2 * np.pi) - np.pi)
    frame['projection_distance_m'] = frame.projection_ratio * network.lengths[pos]
    frame['arc_length_m'] = network.lengths[pos]
    frame['heading_difference_rad'] = difference
    frame['heading_cosine'] = np.cos(difference)
    frame['speed_mps'] = speed[obs]
    frame['dt_s'] = np.where(time_valid[obs], dt[obs], np.nan)
    for name, values, nodes in [('from_in_degree', network.in_degree, network.from_nodes),
                               ('from_out_degree', network.out_degree, network.from_nodes),
                               ('to_in_degree', network.in_degree, network.to_nodes),
                               ('to_out_degree', network.out_degree, network.to_nodes)]:
        frame[name] = values[nodes[pos]]
    frame['motion_valid'] = motion_valid[obs].astype(float)
    frame['time_valid'] = time_valid[obs].astype(float)
    frame.insert(0, 'trip_id', trip.trip_id)
    frame.insert(1, 'split', trip.partition)
    frame['t'] = t[obs]
    frame = frame.drop(columns='segment_pos')
    for column in FEATURE_COLUMNS:
        frame[column] = frame[column].astype('float32')
    return frame, xy, valid
