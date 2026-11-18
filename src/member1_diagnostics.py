"""EDA and reference-alignment maps from real pipeline outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from .member1_io import load_trip


def save_plot(path, title, ylabel):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def alignment_map(trip, observations, destination):
    coords = trip.track[['latitude', 'longitude']].to_numpy(float)
    valid = np.isfinite(coords).all(1) & (abs(coords[:, 0]) <= 90) & (abs(coords[:, 1]) <= 180)
    if not valid.any():
        return
    map_view = folium.Map(location=np.median(coords[valid], axis=0).tolist(), tiles='OpenStreetMap')
    layers = {name: folium.FeatureGroup(name=name, show=show).add_to(map_view)
              for name, show in [('GPS observations', True), ('Road network', False), ('Reference route', True), ('Aligned reference labels', True)]}
    nodes = trip.nodes[['latitude', 'longitude']].to_numpy(float)
    arcs = trip.arcs[['from_node_id', 'to_node_id']].to_numpy(int)
    lower, upper = coords[valid].min(0)-.005, coords[valid].max(0)+.005
    # Keep the interactive map manageable by clipping its road layer to the trip bounds.
    usable = (arcs >= 0).all(1) & (arcs < len(nodes)).all(1)
    ids = np.flatnonzero(usable)
    ends = nodes[arcs[ids]]
    intersects = ((ends.max(1) >= lower) & (ends.min(1) <= upper)).all(1) & np.isfinite(ends).all((1, 2))
    features = [{'type': 'Feature', 'properties': {}, 'geometry': {'type': 'LineString', 'coordinates': pair[:, ::-1].tolist()}}
                for pair in ends[intersects]]
    folium.GeoJson({'type': 'FeatureCollection', 'features': features},
                   style_function=lambda _: {'color': '#999999', 'weight': 1}).add_to(layers['Road network'])
    if trip.route is not None:
        for arc in trip.route.arc_id:
            folium.PolyLine(nodes[arcs[int(arc)]].tolist(), color='green', weight=4, opacity=.7).add_to(layers['Reference route'])
    for index in np.flatnonzero(valid):
        folium.CircleMarker(coords[index].tolist(), radius=2, color='blue', fill=True, tooltip=f'GPS {index}').add_to(layers['GPS observations'])
    for row in observations.itertuples():
        if not row.reference_available:
            continue
        color = '#dc2626' if row.unreliable else '#7c3aed'
        folium.CircleMarker([row.latitude, row.longitude], radius=3, color=color,
                            tooltip=f'obs={row.obs_idx}, arc={row.reference_arc_id}, route_position={row.route_position}, distance={row.alignment_distance_m:.1f}m').add_to(layers['Aligned reference labels'])
    folium.LayerControl().add_to(map_view)
    map_view.fit_bounds([coords[valid].min(0).tolist(), coords[valid].max(0).tolist()])
    map_view.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:10px">Blue: GPS | Green: reference roads<br>Purple: aligned point | Red: alignment &gt;50 m<br>Reference alignment diagnostic — not a predicted route</div>'))
    map_view.save(str(destination))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='results/member1/data')
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--output', default='results/member1/figures')
    parser.add_argument('--maps', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(Path(args.data) / 'inventory.csv', dtype={'trip_id': str})
    obs = pd.read_parquet(Path(args.data) / 'observations.parquet')
    inventory.groupby('split')[['n_observations', 'raw_arcs', 'nodes', 'missing_track_values', 'gaps_over_30s', 'no_candidates', 'unreliable_alignment']].sum().to_csv(out / 'eda_summary.csv')
    plt.figure(figsize=(7,4))
    for split, group in inventory.groupby('split'):
        plt.hist(group.n_observations, bins=20, alpha=.6, label=split)
    plt.legend(); plt.xlabel('GPS observations per trip')
    save_plot(out / 'trip_sizes.png', 'Trajectory sizes', 'Trips')
    dt = obs.sort_values(['split','trip_id','obs_idx']).groupby(['split','trip_id']).t.diff()
    plt.figure(figsize=(7,4)); plt.hist(np.log10(dt[dt > 0]), bins=40); plt.xlabel('log10(time gap in seconds)')
    save_plot(out / 'time_gaps.png', 'Positive GPS sampling gaps', 'Steps')
    plt.figure(figsize=(7,4)); plt.hist(obs.n_candidates, bins=np.arange(obs.n_candidates.max()+2)-.5); plt.xlabel('Candidates per observation')
    save_plot(out / 'candidate_counts.png', 'Candidate availability, including misses', 'Observations')
    plt.figure(figsize=(7,4)); plt.scatter(inventory.nodes, inventory.weak_components, s=15); plt.xlabel('Road nodes')
    save_plot(out / 'connectivity.png', 'Road-network connectivity', 'Weak components')
    plt.figure(figsize=(7,4)); plt.hist(obs.alignment_distance_m.dropna().clip(upper=200), bins=40); plt.xlabel('Aligned distance in metres (clipped at 200)')
    save_plot(out / 'alignment_distances.png', 'Derived reference label quality', 'Observations')
    selected = list(obs.loc[obs.split == 'train','trip_id'].drop_duplicates()[:max(args.maps, 0)])
    for trip_id in selected:
        part = obs[(obs.split == 'train') & (obs.trip_id == trip_id)]
        plt.figure(figsize=(7,5)); plt.plot(part.longitude, part.latitude, '.', markersize=2); plt.xlabel('Longitude')
        save_plot(out / f'trajectory_{trip_id}.png', f'GPS trajectory {trip_id}', 'Latitude')
        alignment_map(load_trip(trip_id, 'train', args.data_root, load_reference=True), part, out / f'alignment_{trip_id}.html')
    print(f'Generated outputs in {out.resolve()}')


if __name__ == '__main__':
    main()
