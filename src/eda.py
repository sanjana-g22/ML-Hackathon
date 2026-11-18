"""
PathSnap - Exploratory Data Analysis (Report Section 2.2).

Partitions analysed: train (80 trips) and student_test (10 trips).
The evaluator-test partition is never read here.

Run from the repo root:
    python -m src.eda

Outputs (results/eda/):
    trip_inventory.csv      one row per trip: sizes, gaps, extent, network stats
    partition_summary.csv   aggregated per partition
    missing_values.csv      NaN / invalid counts per file type and partition
    data_quality_notes.md   anomalies found automatically
    fig_*.png               plots for the report
    ../maps/eda_*.html      Folium maps of selected trips
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_loader import list_trips, load_trip, verify_trip_integrity  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "results", "eda")
MAP_DIR = os.path.join(REPO_ROOT, "results", "maps")
PARTITIONS = ["train", "student_test"]
EARTH_RADIUS_M = 6_371_000.0
GAP_THRESHOLD_S = 5.0          # a "gap" = more than this many seconds between fixes
SPEED_LIMIT_MS = 70.0          # ~250 km/h: anything faster is treated as a GPS jump


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def utm_zone(lon: float) -> int:
    return int((lon + 180) // 6) + 1


def network_stats(trip) -> dict:
    """Connectivity of the trip's road graph."""
    n = trip.n_nodes
    u = trip.arcs["from_node_id"].to_numpy()
    v = trip.arcs["to_node_id"].to_numpy()
    adj = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n)).tocsr()

    n_weak, weak_labels = connected_components(adj, directed=True, connection="weak")
    n_strong, _ = connected_components(adj, directed=True, connection="strong")
    largest_weak = np.bincount(weak_labels).max()

    degree = np.bincount(u, minlength=n) + np.bincount(v, minlength=n)

    # how many arcs have their reverse arc present (two-way roads)
    fwd = set(zip(u.tolist(), v.tolist()))
    two_way = sum((b, a) in fwd for a, b in fwd)

    # arc lengths
    lon = trip.nodes["longitude"].to_numpy()
    lat = trip.nodes["latitude"].to_numpy()
    arc_len = haversine_m(lon[u], lat[u], lon[v], lat[v])

    return {
        "n_weak_components": n_weak,
        "n_strong_components": n_strong,
        "largest_component_share": largest_weak / n,
        "isolated_nodes": int((degree == 0).sum()),
        "self_loops": int((u == v).sum()),
        "duplicate_arcs": len(u) - len(fwd),
        "two_way_arc_share": two_way / max(len(fwd), 1),
        "arc_len_median_m": float(np.median(arc_len)),
        "arc_len_max_m": float(arc_len.max()),
    }


def trip_row(trip, trip_dir: str) -> dict:
    tr = trip.track
    t = tr["timestamp"].to_numpy()
    dt = np.diff(t)
    step = haversine_m(tr["longitude"].to_numpy()[:-1], tr["latitude"].to_numpy()[:-1],
                       tr["longitude"].to_numpy()[1:], tr["latitude"].to_numpy()[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)

    integrity = verify_trip_integrity(trip)
    route = trip.route["arc_id"].to_numpy() if trip.route is not None else np.array([], dtype=int)

    size_kb = {
        ext: os.path.getsize(os.path.join(trip_dir, f"{trip.trip_id}.{ext}")) / 1024
        for ext in ("track", "nodes", "arcs", "route")
        if os.path.exists(os.path.join(trip_dir, f"{trip.trip_id}.{ext}"))
    }

    lon_c, lat_c = tr["longitude"].mean(), tr["latitude"].mean()
    row = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_points": trip.n_points,
        "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "share_dt_1s": float(np.mean(dt == 1.0)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "n_gaps_gt5s": int((dt > GAP_THRESHOLD_S).sum()),
        "n_zero_or_negative_dt": int((dt <= 0).sum()),
        "track_length_km": float(np.nansum(step) / 1000),
        "speed_median_ms": float(np.nanmedian(speed)),
        "speed_p95_ms": float(np.nanpercentile(speed, 95)),
        "n_speed_jumps": int(np.nansum(speed > SPEED_LIMIT_MS)),
        "nan_track_values": int(tr.isna().sum().sum()),
        "center_lon": lon_c,
        "center_lat": lat_c,
        "utm_zone": utm_zone(lon_c),
        "extent_ns_km": float(haversine_m(lon_c, tr["latitude"].min(), lon_c, tr["latitude"].max()) / 1000),
        "extent_ew_km": float(haversine_m(tr["longitude"].min(), lat_c, tr["longitude"].max(), lat_c) / 1000),
        "n_nodes": trip.n_nodes,
        "n_arcs": trip.n_arcs,
        "n_route_arcs": len(route),
        "route_unique_arcs": int(len(np.unique(route))),
        "points_per_route_arc": trip.n_points / max(len(route), 1),
        "route_continuity_pct": integrity["route_continuity_pct"],
        "integrity_issues": "; ".join(integrity["issues"]),
        **{f"size_kb_{k}": round(v, 1) for k, v in size_kb.items()},
    }
    row.update(network_stats(trip))
    return row, dt


def missing_value_table(partition: str, trips) -> list:
    rows = []
    for name, getter in [("track", lambda t: t.track), ("nodes", lambda t: t.nodes),
                         ("arcs", lambda t: t.arcs), ("route", lambda t: t.route)]:
        frames = [getter(t) for t in trips if getter(t) is not None]
        n_missing_files = sum(getter(t) is None for t in trips)
        total_rows = sum(len(f) for f in frames)
        nan_cells = sum(int(f.isna().sum().sum()) for f in frames)
        rows.append({
            "partition": partition, "file": name, "trips_with_file": len(frames),
            "trips_missing_file": n_missing_files, "total_rows": total_rows, "nan_cells": nan_cells,
        })
    return rows


def plots(inv: pd.DataFrame, all_dt: dict) -> None:
    colors = {"train": "#1f5fd6", "student_test": "#e08214"}

    # 1. GPS points per trip
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in inv.groupby("partition"):
        ax.hist(g["n_points"], bins=30, alpha=0.7, label=f"{p} (n={len(g)})", color=colors[p])
    ax.set(xlabel="GPS observations per trip", ylabel="trips", title="Trajectory length")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_points_per_trip.png"), dpi=150); plt.close(fig)

    # 2. sampling interval distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, dt in all_dt.items():
        ax.hist(np.clip(dt, 0, 60), bins=60, alpha=0.7, label=p, color=colors[p], log=True)
    ax.set(xlabel="seconds between consecutive fixes (clipped at 60)", ylabel="count (log)",
           title="Sampling interval distribution")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_sampling_interval.png"), dpi=150); plt.close(fig)

    # 3. largest gap per trip
    fig, ax = plt.subplots(figsize=(9, 4))
    s = inv.sort_values("dt_max_s", ascending=False)
    ax.bar(range(len(s)), s["dt_max_s"], color=[colors[p] for p in s["partition"]])
    ax.set_yscale("log")
    ax.set(xlabel="trips (sorted)", ylabel="largest gap (s, log)", title="Largest GPS gap per trip")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_max_gap_per_trip.png"), dpi=150); plt.close(fig)

    # 4. where the trips are
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p, g in inv.groupby("partition"):
        ax.scatter(g["center_lon"], g["center_lat"], s=25, label=p, color=colors[p], alpha=0.8)
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip locations (centroids)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_trip_locations.png"), dpi=150); plt.close(fig)

    # 5. network size vs track length
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in inv.groupby("partition"):
        ax.scatter(g["n_points"], g["n_arcs"], s=25, label=p, color=colors[p], alpha=0.8)
    ax.set(xscale="log", yscale="log", xlabel="GPS observations", ylabel="road arcs in network",
           title="Road-network size vs trajectory length")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_network_size.png"), dpi=150); plt.close(fig)

    # 6. points per route arc (why labels must be derived)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(inv["points_per_route_arc"], bins=30, color="#555")
    ax.set(xlabel="GPS points / reference route arcs", ylabel="trips",
           title="GPS points per reference arc (≠ 1, so per-point labels must be derived)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_points_per_route_arc.png"), dpi=150); plt.close(fig)

    # 7. a few raw trajectories in local metres
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    sample = inv.sort_values("n_points").iloc[np.linspace(0, len(inv) - 1, 6).astype(int)]
    for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
        trip = load_trip(r["trip_id"], r["partition"], base_dir=REPO_ROOT)
        lon, lat = trip.track["longitude"].to_numpy(), trip.track["latitude"].to_numpy()
        x = np.radians(lon - lon.mean()) * EARTH_RADIUS_M * np.cos(np.radians(lat.mean())) / 1000
        y = np.radians(lat - lat.mean()) * EARTH_RADIUS_M / 1000
        ax.plot(x, y, lw=0.8, color="#1f5fd6")
        ax.scatter(x[0], y[0], color="green", s=20, zorder=3)
        ax.scatter(x[-1], y[-1], color="red", s=20, zorder=3)
        ax.set_title(f"{r['partition']} {r['trip_id']} ({r['n_points']} pts)", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.supxlabel("east (km)"); fig.supylabel("north (km)")
    fig.suptitle("Sample GPS trajectories (local equirectangular projection)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_sample_trajectories.png"), dpi=150); plt.close(fig)


def quality_notes(inv: pd.DataFrame) -> str:
    lines = ["# Data-quality notes (auto-generated by src/eda.py)", ""]
    lines.append(f"- Trips: {inv.groupby('partition').size().to_dict()}")
    lines.append(f"- Distinct UTM zones across trips: {inv['utm_zone'].nunique()} "
                 f"-> projection must be chosen per trip")
    lines.append(f"- Overall share of 1 s sampling steps: {inv['share_dt_1s'].mean():.1%} (mean over trips)")
    big = inv[inv["dt_max_s"] > 30].sort_values("dt_max_s", ascending=False)
    lines.append(f"- Trips with a gap > 30 s: {len(big)} "
                 f"(worst: {', '.join(f'{r.trip_id} ({r.dt_max_s:.0f}s)' for r in big.head(5).itertuples())})")
    lines.append(f"- Trips with non-increasing timestamps: {(inv['n_zero_or_negative_dt'] > 0).sum()}")
    lines.append(f"- Trips with implied speed > {SPEED_LIMIT_MS:.0f} m/s (GPS jumps): {(inv['n_speed_jumps'] > 0).sum()}")
    lines.append(f"- NaN cells in tracks: {int(inv['nan_track_values'].sum())}")
    lines.append(f"- Trips whose reference route is not fully connected arc-to-arc: "
                 f"{(inv['route_continuity_pct'] < 100).sum()} "
                 f"(min continuity {inv['route_continuity_pct'].min():.1f}%)")
    lines.append(f"- Trips whose road graph has >1 weak component: {(inv['n_weak_components'] > 1).sum()}")
    lines.append(f"- Median share of two-way arcs: {inv['two_way_arc_share'].median():.1%}")
    issues = inv[inv["integrity_issues"].astype(str) != ""]
    lines.append(f"- Trips with integrity issues: {len(issues)}")
    for r in issues.itertuples():
        lines.append(f"  - {r.partition}/{r.trip_id}: {r.integrity_issues}")
    return "\n".join(lines) + "\n"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MAP_DIR, exist_ok=True)

    rows, missing, all_dt = [], [], {}
    for partition in PARTITIONS:
        trips, dts = [], []
        for trip_id in list_trips(partition, base_dir=REPO_ROOT):
            trip = load_trip(trip_id, partition, base_dir=REPO_ROOT)
            trip_dir = os.path.join(REPO_ROOT, partition, trip_id)
            row, dt = trip_row(trip, trip_dir)
            rows.append(row)
            dts.append(dt)
            trips.append(trip)
            print(f"{partition}/{trip_id}: {trip.n_points} pts, {trip.n_arcs} arcs")
        all_dt[partition] = np.concatenate(dts)
        missing += missing_value_table(partition, trips)

    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT_DIR, "trip_inventory.csv"), index=False)
    pd.DataFrame(missing).to_csv(os.path.join(OUT_DIR, "missing_values.csv"), index=False)

    summary = inv.groupby("partition").agg(
        trips=("trip_id", "count"),
        total_points=("n_points", "sum"),
        points_median=("n_points", "median"),
        points_min=("n_points", "min"),
        points_max=("n_points", "max"),
        duration_median_s=("duration_s", "median"),
        track_km_median=("track_length_km", "median"),
        share_dt_1s_mean=("share_dt_1s", "mean"),
        trips_with_gap_gt30s=("dt_max_s", lambda s: int((s > 30).sum())),
        arcs_median=("n_arcs", "median"),
        arcs_max=("n_arcs", "max"),
        route_arcs_median=("n_route_arcs", "median"),
        route_continuity_mean=("route_continuity_pct", "mean"),
    )
    summary.to_csv(os.path.join(OUT_DIR, "partition_summary.csv"))
    print("\n", summary.T.to_string())

    with open(os.path.join(OUT_DIR, "data_quality_notes.md"), "w", encoding="utf-8") as f:
        f.write(quality_notes(inv))

    plots(inv, all_dt)

    # Folium maps: one typical train trip, the trip with the worst gap, one student-test trip
    from maps.make_map import make_map, trip_layers
    train = inv[inv["partition"] == "train"]
    picks = {
        "typical": train.iloc[(train["n_points"] - train["n_points"].median()).abs().argsort().iloc[0]],
        "largest_gap": train.sort_values("dt_max_s").iloc[-1],
        "student_test": inv[inv["partition"] == "student_test"].iloc[0],
    }
    for tag, r in picks.items():
        trip = load_trip(r["trip_id"], r["partition"], base_dir=REPO_ROOT)
        out = os.path.join(MAP_DIR, f"eda_{tag}_{r['partition']}_{r['trip_id']}.html")
        make_map(**trip_layers(trip)).save(out)
        print(f"map: {out}")

    print(f"\nEDA outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
