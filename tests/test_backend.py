"""
Backend tests. Run from the repo root:  pytest tests/

Uses the student_test partition as the trip folder (the EVAL_DIR default).
"""
import dataclasses
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import predictor
from backend.main import app
from src.data_loader import load_trip


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # the context manager runs the startup that loads trips
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["n_trips"] > 0


def test_trips_lists_folder_contents(client):
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 10
    assert all(set(t) == {"trip_id", "n_points"} for t in trips)


def test_predict_shape_and_metrics(client):
    trip_id = client.get("/trips").json()["trips"][0]["trip_id"]
    body = client.post("/predict", json={"trip_id": trip_id}).json()

    assert set(body) == {"trip_id", "pipeline", "gps", "predicted_route", "network", "metrics"}
    assert len(body["gps"]) == body["metrics"]["n_obs"]
    assert 0.0 <= body["metrics"]["top1_accuracy"] <= 1.0
    assert 0.0 <= body["metrics"]["route_continuity"] <= 1.0


def test_evaluate_alias_matches_predict(client):
    trip_id = client.get("/trips").json()["trips"][0]["trip_id"]
    a = client.post("/predict", json={"trip_id": trip_id}).json()
    b = client.post(f"/evaluate/{trip_id}").json()
    assert a["predicted_route"] == b["predicted_route"]


def test_unknown_trip_is_404(client):
    assert client.post("/predict", json={"trip_id": "does-not-exist"}).status_code == 404


def test_response_contains_no_ground_truth(client):
    """The reference route must never leave the server."""
    trip_id = client.get("/trips").json()["trips"][0]["trip_id"]
    text = json.dumps(client.post("/predict", json={"trip_id": trip_id}).json()).lower()
    for word in ("route_arc", "reference", "label", "ground_truth", "\"route\""):
        assert word not in text


def test_predictor_refuses_trips_with_route():
    trip = load_trip("00000052", "student_test", base_dir=".")
    with pytest.raises(AssertionError):
        predictor.predict_arcs(trip)
    pred = predictor.predict_arcs(dataclasses.replace(trip, route=None))
    assert len(pred) == trip.n_points
    assert np.all((pred >= 0) & (pred < trip.n_arcs))
