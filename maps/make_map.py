"""
PathSnap - reusable Folium map builder.

Layers (each toggleable in the layer control):
  1. Raw GPS observations       (blue dots)
  2. Road-network geometry      (grey lines, only arcs near the track)
  3. Reference route            (green)  - ONLY for train / student-test analysis
  4. Predicted route            (red)

Usage from code:
    from maps.make_map import make_map, trip_layers
    m = make_map(**trip_layers(trip, predicted_arcs=pred))
    m.save("results/maps/trip.html")

Usage from the command line:
    python -m maps.make_map --partition train --trip 00000001
"""

import argparse
import os
import sys
from typing import List, Optional, Sequence

import folium
import numpy as np
from branca.element import Element
from scipy.spatial import cKDTree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import TripData, load_trip  # noqa: E402

EARTH_RADIUS_M = 6_371_000.0

COLORS = {
    "gps": "#1f5fd6",
    "network": "#8a8a8a",
    "reference": "#1a9850",
    "predicted": "#d7301f",
}

LatLon = List[float]          # [lat, lon]
Polyline = List[LatLon]       # [[lat, lon], [lat, lon], ...]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _local_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Equirectangular projection to metres around (lon0, lat0).

    Good to well under 1% over a single trip's extent; only used here to pick
    which road arcs are close enough to the track to be worth drawing.
    """
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def arcs_to_polylines(trip: TripData, arc_ids: Sequence[int]) -> List[Polyline]:
    """Turn an ordered arc sequence into polylines.

    Consecutive duplicates are collapsed. A new polyline starts whenever two
    consecutive arcs are not connected (to_node != next from_node), so breaks in
    a route stay visible on the map instead of being bridged by a straight line.
    """
    nodes = trip.nodes[["latitude", "longitude"]].to_numpy()
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()

    lines: List[Polyline] = []
    prev_arc, prev_to = None, None
    for a in arc_ids:
        if a is None or a < 0 or a == prev_arc:
            continue
        u, v = arcs[a]
        if lines and prev_to == u:
            lines[-1].append(nodes[v].tolist())
        else:
            lines.append([nodes[u].tolist(), nodes[v].tolist()])
        prev_arc, prev_to = a, v
    return lines


def network_near_track(trip: TripData, buffer_m: float = 150.0, max_arcs: int = 8000) -> List[Polyline]:
    """Arcs with at least one end node within `buffer_m` of any GPS point.

    Full networks can have 150k+ arcs, which Folium cannot draw usefully, so
    only the neighbourhood of the trajectory is returned.
    """
    lon0 = trip.track["longitude"].mean()
    lat0 = trip.track["latitude"].mean()
    track_xy = _local_xy(trip.track["longitude"].to_numpy(), trip.track["latitude"].to_numpy(), lon0, lat0)
    node_xy = _local_xy(trip.nodes["longitude"].to_numpy(), trip.nodes["latitude"].to_numpy(), lon0, lat0)

    dist, _ = cKDTree(track_xy).query(node_xy, distance_upper_bound=buffer_m)
    near_node = np.isfinite(dist)

    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    keep = near_node[arcs[:, 0]] | near_node[arcs[:, 1]]

    # Draw each undirected road once (A->B and B->A are usually both present).
    pairs = np.sort(arcs[keep], axis=1)
    pairs = np.unique(pairs, axis=0)[:max_arcs]

    nodes = trip.nodes[["latitude", "longitude"]].to_numpy()
    return [[nodes[u].tolist(), nodes[v].tolist()] for u, v in pairs]


# --------------------------------------------------------------------------- #
# Map construction
# --------------------------------------------------------------------------- #
def _legend(entries: List[tuple]) -> Element:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;'
        f'background:{color};margin-right:6px;vertical-align:middle"></span>{label}</div>'
        for label, color in entries
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;color:#222;'
        'padding:8px 10px;border:1px solid #bbb;border-radius:6px;font:13px sans-serif;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.2)">{rows}</div>'
    )
    return Element(html)


def make_map(
    gps: Polyline,
    network: Optional[List[Polyline]] = None,
    reference: Optional[List[Polyline]] = None,
    predicted: Optional[List[Polyline]] = None,
    title: Optional[str] = None,
    gps_every: int = 1,
) -> folium.Map:
    """Build a layered Folium map. All coordinates are [lat, lon]."""
    lats = [p[0] for p in gps]
    lons = [p[1] for p in gps]
    center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]
    span = max(max(lats) - min(lats), (max(lons) - min(lons)) * np.cos(np.radians(center[0])), 1e-4)
    # a ~600 px map at zoom z shows about 843 / 2**z degrees; leave some margin
    zoom = int(np.clip(np.floor(np.log2(650 / span)), 3, 18))   # explicit start view; fit_bounds refines it
    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)
    legend = []

    if network:
        fg = folium.FeatureGroup(name="Road network (near track)", show=True)
        for line in network:
            folium.PolyLine(line, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", COLORS["network"]))

    if reference:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        for line in reference:
            folium.PolyLine(line, color=COLORS["reference"], weight=6, opacity=0.7).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", COLORS["reference"]))

    if predicted:
        fg = folium.FeatureGroup(name="Predicted route", show=True)
        for line in predicted:
            folium.PolyLine(line, color=COLORS["predicted"], weight=3.5, opacity=0.9, dash_array="6 4").add_to(fg)
        fg.add_to(m)
        legend.append(("Predicted route", COLORS["predicted"]))

    fg = folium.FeatureGroup(name="GPS observations", show=True)
    for i, (lat, lon) in enumerate(gps[::gps_every]):
        folium.CircleMarker(
            [lat, lon], radius=2.5, color=COLORS["gps"], fill=True, fill_opacity=0.9, weight=0,
            tooltip=f"obs {i * gps_every}",
        ).add_to(fg)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fg)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fg)
    fg.add_to(m)
    legend.append(("GPS observations", COLORS["gps"]))

    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(20, 20))

    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(legend))
    if title:
        m.get_root().html.add_child(Element(
            '<div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;color:#222;'
            'padding:6px 10px;border:1px solid #bbb;border-radius:6px;font:bold 14px sans-serif">'
            f"{title}</div>"
        ))
    return m


def trip_layers(
    trip: TripData,
    predicted_arcs: Optional[Sequence[int]] = None,
    show_reference: bool = True,
    show_network: bool = True,
) -> dict:
    """Convenience: build make_map() keyword arguments straight from a TripData."""
    gps = trip.track[["latitude", "longitude"]].to_numpy().tolist()
    layers = {
        "gps": gps,
        "network": network_near_track(trip) if show_network else None,
        "reference": None,
        "predicted": arcs_to_polylines(trip, predicted_arcs) if predicted_arcs is not None else None,
        "title": f"Trip {trip.trip_id} ({trip.partition})",
        # thin out very long tracks so the HTML stays responsive
        "gps_every": max(1, len(gps) // 3000),
    }
    if show_reference and trip.route is not None:
        layers["reference"] = arcs_to_polylines(trip, trip.route["arc_id"].tolist())
    return layers


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a Folium map for one trip.")
    ap.add_argument("--partition", default="train", choices=["train", "student_test"])
    ap.add_argument("--trip", required=True, help="trip id, e.g. 00000001")
    ap.add_argument("--out", default=None, help="output HTML path")
    args = ap.parse_args()

    trip = load_trip(args.trip, args.partition, base_dir=REPO_ROOT)
    out = args.out or os.path.join(REPO_ROOT, "results", "maps", f"{args.partition}_{args.trip}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    make_map(**trip_layers(trip)).save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
