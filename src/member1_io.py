"""Inference-safe loader built on the existing repository parsers."""
from pathlib import Path
from .data_loader import (TripData, load_track, load_nodes, load_arcs, load_route,
                          list_trips, resolve_partition_dir)


def load_trip(trip_id, partition='train', base_dir='.', load_reference=False):
    folder = Path(resolve_partition_dir(partition, base_dir)) / trip_id
    if not folder.is_dir():
        raise FileNotFoundError(f'Trip directory not found: {folder}')
    prefix = folder / trip_id
    return TripData(trip_id, partition, load_track(str(prefix)+'.track'),
                    load_nodes(str(prefix)+'.nodes'), load_arcs(str(prefix)+'.arcs'),
                    load_route(str(prefix)+'.route') if load_reference else None)
