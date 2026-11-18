"""Metric road geometry, segment indexing and directed graph construction."""
from dataclasses import dataclass
import numpy as np
from pyproj import CRS, Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree, linestrings


@dataclass
class SpatialNetwork:
    xy: np.ndarray
    arc_ids: np.ndarray
    from_nodes: np.ndarray
    to_nodes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    lengths: np.ndarray
    bearings: np.ndarray
    in_degree: np.ndarray
    out_degree: np.ndarray
    crs: str
    transformer: Transformer
    tree: STRtree

    def project(self, longitude, latitude):
        x, y = self.transformer.transform(longitude, latitude)
        return np.column_stack([x, y])


def build_network(trip):
    nodes = trip.nodes[['longitude', 'latitude']].to_numpy(float)
    good = np.isfinite(nodes).all(1) & (abs(nodes[:, 0]) <= 180) & (abs(nodes[:, 1]) <= 90)
    if not good.any():
        raise ValueError(f'{trip.trip_id}: no valid road nodes')
    lon, lat = np.median(nodes[good], axis=0)
    if -80 <= lat <= 84:
        zone = int(np.clip(np.floor((lon + 180) / 6) + 1, 1, 60))
        crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    else:
        crs = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    xy = np.column_stack(transformer.transform(nodes[:, 0], nodes[:, 1]))
    uv = trip.arcs[['from_node_id', 'to_node_id']].to_numpy(np.int64)
    valid = (uv >= 0).all(1) & (uv < len(nodes)).all(1)
    valid_idx = np.flatnonzero(valid)
    valid[valid_idx] &= good[uv[valid_idx]].all(1)
    ids = np.flatnonzero(valid)
    uv = uv[ids]
    starts, ends = xy[uv[:, 0]], xy[uv[:, 1]]
    lengths = np.linalg.norm(ends - starts, axis=1)
    keep = np.isfinite(lengths) & (lengths > 1e-6)
    ids, uv, starts, ends, lengths = ids[keep], uv[keep], starts[keep], ends[keep], lengths[keep]
    if not len(ids):
        raise ValueError(f'{trip.trip_id}: no usable road segments')
    delta = ends - starts
    bearings = np.arctan2(delta[:, 0], delta[:, 1])
    return SpatialNetwork(xy, ids, uv[:, 0], uv[:, 1], starts, ends, lengths, bearings,
                          np.bincount(uv[:, 1], minlength=len(nodes)),
                          np.bincount(uv[:, 0], minlength=len(nodes)),
                          crs.to_string(), transformer, STRtree(linestrings(np.stack([starts, ends], axis=1))))


def build_index(trip):
    return build_network(trip)


def build_graph(network):
    # Shortest paths use the shortest parallel edge; arc identity stays in the network arrays.
    import pandas as pd
    edges = pd.DataFrame({'u': network.from_nodes, 'v': network.to_nodes, 'length': network.lengths})
    edges = edges.groupby(['u', 'v'], as_index=False)['length'].min()
    return coo_matrix((edges.length, (edges.u, edges.v)), shape=(len(network.xy), len(network.xy))).tocsr()


def network_summary(network):
    graph = build_graph(network)
    reverse = graph.transpose().tocsr()
    bidirectional = graph.multiply(reverse.sign()).nnz
    return {'usable_arcs': len(network.arc_ids), 'nodes': len(network.xy),
            'isolated_nodes': int(((network.in_degree + network.out_degree) == 0).sum()),
            'weak_components': int(connected_components(graph, directed=True, connection='weak')[0]),
            'strong_components': int(connected_components(graph, directed=True, connection='strong')[0]),
            'reverse_edge_fraction': bidirectional / max(graph.nnz, 1), 'crs': network.crs}


def segment_projection(points, starts, ends):
    delta = ends - starts
    denominator = np.maximum(np.sum(delta * delta, axis=-1), 1e-12)
    ratio = np.clip(np.sum((points - starts) * delta, axis=-1) / denominator, 0, 1)
    projected = starts + ratio[..., None] * delta
    return np.linalg.norm(points - projected, axis=-1), ratio
