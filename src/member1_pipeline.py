"""Cached development datasets and reference-free inference entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from .member1_io import load_trip, list_trips, resolve_partition_dir
from .network import build_network, network_summary
from .features import make_features, FEATURE_COLUMNS
from .labels import align_reference


def inference_features(trip, radius=250., k=30):
    network = build_network(trip)
    candidates, xy, valid = make_features(trip, network, radius, k)
    return candidates, network, xy, valid


def fingerprint(folder, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for p in sorted(folder.iterdir()):
        if p.suffix in {'.track', '.nodes', '.arcs', '.route'}:
            digest.update(p.name.encode())
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    digest.update(chunk)
    for name in ['data_loader.py', 'member1_io.py', 'network.py', 'candidates.py', 'features.py', 'labels.py', 'member1_pipeline.py']:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def prepare_trip(trip, radius, k):
    network = build_network(trip)
    candidates, xy, valid = make_features(trip, network, radius, k)
    observations = align_reference(trip, network, xy, valid)
    observations.insert(0, 'trip_id', trip.trip_id)
    observations.insert(1, 'split', trip.partition)
    observations['t'] = trip.track.timestamp.to_numpy()
    observations['longitude'] = trip.track.longitude.to_numpy()
    observations['latitude'] = trip.track.latitude.to_numpy()
    observations['valid_coordinate'] = valid
    observations['n_candidates'] = observations.obs_idx.map(candidates.groupby('obs_idx').size()).fillna(0).astype(int)
    references = observations.set_index('obs_idx').reference_arc_id
    candidates['label'] = (candidates.arc_id.to_numpy() == candidates.obs_idx.map(references).to_numpy()).astype('int8')
    unknown = ~candidates.obs_idx.map(observations.set_index('obs_idx').reference_available)
    candidates.loc[unknown, 'label'] = -1
    hit = candidates.loc[candidates.label == 1, 'obs_idx']
    observations['candidate_hit'] = observations.obs_idx.isin(hit)
    dt = np.diff(trip.track.timestamp.to_numpy(float))
    known = observations.reference_available
    info = {'trip_id': trip.trip_id, 'split': trip.partition, 'n_observations': len(observations),
            'raw_arcs': len(trip.arcs), 'route_arcs': len(trip.route) if trip.route is not None else 0,
            'missing_track_values': int(trip.track.isna().sum().sum()),
            'invalid_coordinates': int((~valid).sum()), 'nonfinite_timestamps': int((~np.isfinite(trip.track.timestamp)).sum()),
            'duplicate_timestamp_steps': int((dt == 0).sum()), 'negative_timestamp_steps': int((dt < 0).sum()),
            'gaps_over_30s': int((dt > 30).sum()), 'irregular_steps': int((~np.isclose(dt, 1)).sum()),
            'max_gap_s': float(np.nanmax(dt)) if len(dt) else 0.,
            'n_candidates': len(candidates), 'no_candidates': int((observations.n_candidates == 0).sum()),
            'n_reference_observations': int(known.sum()),
            'unreliable_alignment': int((known & observations.unreliable).sum()),
            'far_from_every_route_arc': int((observations.nearest_route_distance_m > 50).sum()),
            'candidate_miss_rate': float(1 - observations.loc[known, 'candidate_hit'].mean()) if known.any() else None,
            **network_summary(network)}
    return candidates, observations, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--output', default='results/member1/data')
    parser.add_argument('--radius', type=float, default=250.)
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--limit', type=int, help='Explicit smoke run: first N trips per development split')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.radius <= 0 or args.k < 1 or (args.limit is not None and args.limit < 1):
        parser.error('radius, k and limit must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {'radius_m': args.radius, 'max_candidates': args.k, 'features': FEATURE_COLUMNS,
              'label_method': 'monotone minimum-distance route-position alignment',
              'unreliable_distance_m': 50., 'limit': args.limit}
    candidates_all, observations_all, inventory = [], [], []
    for split in ['train', 'student_test']:
        ids = list_trips(split, args.data_root)
        if args.limit:
            ids = ids[:args.limit]
        for i, trip_id in enumerate(ids):
            folder = Path(resolve_partition_dir(split, args.data_root)) / trip_id
            signature = fingerprint(folder, config)
            cache = output / 'cache' / split / trip_id
            cache.mkdir(parents=True, exist_ok=True)
            meta = cache / 'metadata.json'
            if not args.force and meta.exists() and json.loads(meta.read_text()).get('signature') == signature:
                c = pd.read_parquet(cache / 'candidates.parquet')
                o = pd.read_parquet(cache / 'observations.parquet')
                info = json.loads(meta.read_text())['inventory']
            else:
                trip = load_trip(trip_id, split, args.data_root, load_reference=True)
                if trip.route is None or trip.route.empty:
                    raise ValueError(f'{split}/{trip_id}: missing development reference')
                c, o, info = prepare_trip(trip, args.radius, args.k)
                c.to_parquet(cache / 'candidates.parquet', index=False)
                o.to_parquet(cache / 'observations.parquet', index=False)
                meta.write_text(json.dumps({'signature': signature, 'inventory': info}, indent=2))
            candidates_all.append(c)
            observations_all.append(o)
            inventory.append(info)
            print(f'{split} {i+1}/{len(ids)} {trip_id}: {len(c)} candidates, miss={info["candidate_miss_rate"]:.3%}', flush=True)
    candidates = pd.concat(candidates_all, ignore_index=True)
    observations = pd.concat(observations_all, ignore_index=True)
    candidates.to_parquet(output / 'candidates.parquet', index=False)
    observations.to_parquet(output / 'observations.parquet', index=False)
    pd.DataFrame(inventory).to_csv(output / 'inventory.csv', index=False)
    train_ids = sorted(observations.loc[observations.split == 'train', 'trip_id'].unique())
    if len(train_ids) >= 5:
        train_rows = candidates[(candidates.split == 'train') & (candidates.label >= 0)]
        folds = {}
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_, validation) in enumerate(splitter.split(train_rows, groups=train_rows.trip_id)):
            for trip_id in train_rows.iloc[validation].trip_id.unique():
                folds[trip_id] = fold
        pd.DataFrame({'trip_id': sorted(folds), 'fold': [folds[x] for x in sorted(folds)]}).to_csv(output / 'folds.csv', index=False)
    elif (output / 'folds.csv').exists():
        (output / 'folds.csv').unlink()
    (output / 'feature_config.json').write_text(json.dumps(config, indent=2))
    print(f'Saved development dataset to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
