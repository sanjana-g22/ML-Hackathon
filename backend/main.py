"""
PathSnap API (runs on Laptop 1).

Start (from the repo root):
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health           liveness + which pipeline is loaded
    GET  /trips            trip ids found in EVAL_DIR (never hardcoded)
    POST /predict          {"trip_id": "..."} -> GPS, predicted route, metrics
    POST /evaluate/{id}    same as /predict, path-style (name used in the team README)

Ground truth (.route) is read only by backend.scoring, on this machine. The
response contains GPS points, the predicted route and aggregate metrics only.
"""
import dataclasses
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from backend import predictor, scoring
from backend.config import EVAL_DIR
from maps.make_map import arcs_to_polylines, network_near_track
from src.data_loader import TripData, list_trips, load_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("pathsnap")

BASE_DIR, PARTITION = os.path.dirname(EVAL_DIR), os.path.basename(EVAL_DIR)
TRIPS: Dict[str, TripData] = {}   # preloaded trips, with route stripped


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    for trip_id in list_trips(PARTITION, base_dir=BASE_DIR):
        trip = load_trip(trip_id, PARTITION, base_dir=BASE_DIR)
        TRIPS[trip_id] = dataclasses.replace(trip, route=None)   # model side never holds the route
    log.info(f"loaded {len(TRIPS)} trips from {EVAL_DIR} in {time.perf_counter() - t0:.1f}s")
    log.info(f"pipeline: {predictor.PIPELINE_NAME}")
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)


class PredictRequest(BaseModel):
    trip_id: str


class PredictResponse(BaseModel):
    trip_id: str
    pipeline: str
    gps: List[List[float]]                     # [[lat, lon], ...]
    predicted_route: List[List[List[float]]]   # polylines, split where arcs are not connected
    network: List[List[List[float]]]           # road segments near the track (context layer)
    metrics: dict


@app.get("/health")
def health():
    return {"status": "ok", "pipeline": predictor.PIPELINE_NAME, "n_trips": len(TRIPS)}


@app.get("/trips")
def trips(request: Request):
    log.info(f"GET /trips from {request.client.host}")
    return {"trips": [
        {
            "trip_id": k,
            "n_points": v.n_points,
            "duration_s": float(v.track["timestamp"].iloc[-1] - v.track["timestamp"].iloc[0]),
        }
        for k, v in sorted(TRIPS.items())
    ]}


@app.post("/evaluate/{trip_id}", response_model=PredictResponse)
def evaluate(trip_id: str, request: Request):
    return predict(PredictRequest(trip_id=trip_id), request)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    trip = TRIPS.get(req.trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail=f"unknown trip_id {req.trip_id}")

    predictor.load_model()             # loads once, on the first request; not counted as inference
    t0 = time.perf_counter()
    prediction = predictor.predict(trip)
    inference_ms = (time.perf_counter() - t0) * 1000
    pred = prediction.arcs

    route_path = os.path.join(EVAL_DIR, req.trip_id, f"{req.trip_id}.route")
    metrics = scoring.score(trip, route_path, prediction)
    metrics.update({"n_obs": trip.n_points, "inference_ms": round(inference_ms, 1), **prediction.timings_ms})

    top1 = metrics["top1_accuracy"]
    log.info(
        f"POST /predict from {request.client.host} | trip={req.trip_id} | n_obs={trip.n_points} | "
        f"inference={inference_ms:.0f} ms | top1={'n/a' if top1 is None else f'{top1:.3f}'} | "
        f"continuity={metrics['route_continuity']:.3f}"
    )

    return PredictResponse(
        trip_id=req.trip_id,
        pipeline=predictor.PIPELINE_NAME,
        gps=trip.track[["latitude", "longitude"]].to_numpy().tolist(),
        predicted_route=arcs_to_polylines(trip, pred.tolist()),
        network=network_near_track(trip, buffer_m=100),
        metrics=metrics,
    )
