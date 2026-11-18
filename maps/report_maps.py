"""
Report maps (Section 6, and Section 5 once the HMM exists).

For each student-test trip this runs the same pipeline as the API (backend.predictor) and draws
the four layers the guidelines require, plus diagnostics:
    GPS observations · road network near the track · reference route · predicted route
    route breaks (consecutive predicted arcs not connected) · GPS points matched to the wrong road
When hmm/decoder.py exists, the classifier-only route is added as an extra layer, so the same
maps show classifier-only vs classifier + HMM.

Two HTML views per trip: the whole trip, and a close-up of the stretch with the most errors.
PNG screenshots (for the PDF) are rendered with headless Microsoft Edge for two trips: the one
closest to the overall Top-1 ("representative") and the one with the lowest Top-1 ("hardest").

Run from the repo root:
    python -m maps.report_maps                       # all 10 student-test trips + PNGs
    python -m maps.report_maps --trips 00000076 --no-png

Outputs: results/maps/report/<trip>_{full,detail}.html, report_maps_summary.csv, *.png

Only train / student_test trips are allowed: evaluator trips must not be inspected.
"""
import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import predictor, scoring  # noqa: E402
from maps.make_map import COLORS, arcs_to_polylines, make_map, network_near_track, route_breaks  # noqa: E402
from src.data_loader import list_trips, load_trip  # noqa: E402
from src.labels import align_reference  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "maps" / "report"
ALLOWED_PARTITIONS = ("train", "student_test")
DETAIL_WINDOW = 120          # GPS points in the close-up window
DETAIL_ZOOM = 16
EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def worst_window(trip, wrong):
    """Centre of the DETAIL_WINDOW consecutive GPS points with the most wrong matches."""
    lat, lon = trip.track["latitude"].to_numpy(), trip.track["longitude"].to_numpy()
    w = min(DETAIL_WINDOW, len(wrong))
    counts = np.convolve(wrong.astype(int), np.ones(w, dtype=int), mode="valid")
    start = int(counts.argmax()) if counts.max() > 0 else max(len(wrong) // 2 - w // 2, 0)
    sl = slice(start, start + w)
    return float(lat[sl].mean()), float(lon[sl].mean()), DETAIL_ZOOM


def build_trip(trip_id, partition):
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"report maps use {ALLOWED_PARTITIONS} only, never evaluator trips")
    full = load_trip(trip_id, partition, base_dir=str(REPO_ROOT))
    blind = dataclasses.replace(full, route=None)          # the predictor never sees the route

    pred = predictor.predict(blind)
    metrics = scoring.score(blind, str(REPO_ROOT / partition / trip_id / f"{trip_id}.route"), pred)
    ref = align_reference(full, pred.network, pred.xy, pred.valid)
    known = ref["reference_available"].to_numpy()
    wrong = known & (pred.arcs != ref["reference_arc_id"].to_numpy())

    gps = full.track[["latitude", "longitude"]].to_numpy().tolist()
    predicted = arcs_to_polylines(full, pred.arcs.tolist())
    layers = {
        "gps": gps,
        "network": network_near_track(full),
        "reference": arcs_to_polylines(full, full.route["arc_id"].tolist()),
        "predicted": predicted,
        "breaks": route_breaks(predicted),
        "wrong_points": [gps[i] for i in np.flatnonzero(wrong)],
        "predicted_label": "Predicted route (HMM)" if predictor._hmm_decode else "Predicted route (classifier-only)",
        "gps_every": max(1, len(gps) // 3000),
    }
    if predictor._hmm_decode is not None:                 # show what the HMM changed
        classifier_only = predictor.argmax_decode(pred.candidates, full.n_points)
        layers["extra_routes"] = [("Classifier-only route", arcs_to_polylines(full, classifier_only.tolist()),
                                   COLORS["alternative"])]

    summary = {"trip_id": trip_id, "partition": partition, "points": full.n_points,
               "top1": metrics["top1_accuracy"], "top1_reliable": metrics["top1_reliable_points"],
               "continuity": metrics["route_continuity"], "sequence_f1": metrics["sequence_f1"],
               "route_breaks": len(layers["breaks"]), "wrong_points": int(wrong.sum())}
    return layers, summary, worst_window(full, wrong)


def render_png(html_path, png_path, size=(1600, 1000)):
    edge = next((p for p in EDGE_PATHS if os.path.exists(p)), shutil.which("msedge"))
    if not edge:
        print("  (Microsoft Edge not found: open the HTML and take a screenshot instead)")
        return False
    Path(png_path).unlink(missing_ok=True)      # so a failed render is never reported as saved
    # No pipes: on Windows a lingering Edge helper process would keep a captured pipe open and
    # hang the script after the screenshot is already written.
    proc = subprocess.Popen([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                             f"--window-size={size[0]},{size[1]}", "--virtual-time-budget=15000",
                             f"--screenshot={png_path}", Path(html_path).resolve().as_uri()],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
    # Edge's launcher can exit a moment before its renderer writes the file: wait until the
    # screenshot exists and its size has stopped changing
    png, last_size = Path(png_path), -1
    for _ in range(60):
        if png.exists() and png.stat().st_size == last_size > 0:
            return True
        last_size = png.stat().st_size if png.exists() else -1
        time.sleep(0.5)
    return png.exists()


def main():
    parser = argparse.ArgumentParser(description="Report maps: GPS, network, reference and predicted routes.")
    parser.add_argument("--partition", default="student_test", choices=ALLOWED_PARTITIONS)
    parser.add_argument("--trips", nargs="*", help="trip ids (default: every trip in the partition)")
    parser.add_argument("--no-png", action="store_true", help="skip the headless-Edge screenshots")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trip_ids = args.trips or list_trips(args.partition, base_dir=str(REPO_ROOT))
    rows, focus_of = [], {}
    for trip_id in trip_ids:
        layers, summary, focus = build_trip(trip_id, args.partition)
        title = (f"Trip {trip_id} ({args.partition}) · {predictor.PIPELINE_NAME} · "
                 f"Top-1 {summary['top1']:.1%} · continuity {summary['continuity']:.1%}")
        # full view: the four required layers only; break and wrong-point markers would hide the routes
        # at this zoom, so those layers start switched off (one click away in the layer control)
        # GPS dots are thinned to ~200 so they do not cover the route lines at this zoom
        full_view = {**layers, "gps_every": max(1, len(layers["gps"]) // 200)}
        make_map(**full_view, title=title, wrong_points_visible=False, breaks_visible=False) \
            .save(OUT_DIR / f"{trip_id}_full.html")
        make_map(**layers, title=title + " · close-up of the most errors", focus=focus) \
            .save(OUT_DIR / f"{trip_id}_detail.html")
        rows.append(summary)
        focus_of[trip_id] = focus
        print(f"{trip_id}: Top-1 {summary['top1']:.3f}  continuity {summary['continuity']:.3f}  "
              f"breaks {summary['route_breaks']}  wrong points {summary['wrong_points']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "report_maps_summary.csv", index=False, float_format="%.4f")

    if not args.no_png and len(df):
        pooled = (df.top1 * df.points).sum() / df.points.sum()
        picks = {"representative": df.loc[(df.top1 - pooled).abs().idxmin(), "trip_id"],
                 "hardest": df.loc[df.top1.idxmin(), "trip_id"]}
        for tag, trip_id in picks.items():
            for view in ("full", "detail"):
                png = OUT_DIR / f"{tag}_{trip_id}_{view}.png"
                ok = render_png(OUT_DIR / f"{trip_id}_{view}.html", png)
                print(f"  {'saved' if ok else 'FAILED'} {png.relative_to(REPO_ROOT)}", flush=True)
    print(f"maps in {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
