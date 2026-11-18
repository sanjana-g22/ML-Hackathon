# PathSnap two-laptop deployment (Member 4)

## Components

| Path | Runs on | What it does |
|---|---|---|
| `backend/main.py` | Laptop 1 | FastAPI service: loads the trips, runs the prediction pipeline, scores it server-side |
| `backend/predictor.py` | Laptop 1 | Prediction pipeline. **Never sees `.route`.** Placeholder nearest-arc baseline until M1–M3's pipeline is plugged in |
| `backend/scoring.py` | Laptop 1 | The only code that reads ground truth; returns aggregate metrics only |
| `backend/config.py` | Laptop 1 | `EVAL_DIR`: folder of trip sub-folders (default `student_test/`) |
| `frontend/app.py` | Laptop 2 | Streamlit evaluator UI |
| `maps/make_map.py` | both | Folium map builder shared by the UI, EDA and report maps |

## Setup (both laptops)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Laptop 1: API

```bash
set EVAL_DIR=C:\path\to\evaluator_trips
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- `EVAL_DIR` must contain one sub-folder per trip (`<id>/<id>.track`, `.nodes`, `.arcs`, `.route`). If unset, `student_test/` is used.
- Connect both laptops to the same **phone hotspot**; college Wi-Fi usually blocks laptop-to-laptop traffic.
- Allow the port through Windows Defender Firewall (admin PowerShell, once):
  `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
- Find this laptop's address with `ipconfig` (IPv4 address of the Wi-Fi adapter).
- Check from Laptop 2's browser: `http://<laptop-1-ip>:8000/health`

Each request is logged in this terminal (time, client IP, trip, observations, inference ms, accuracy). Show this window to the evaluator.

## Laptop 2: UI

```bash
set API_URL=http://<laptop-1-ip>:8000
streamlit run frontend/app.py
```

The API URL can also be edited in the sidebar.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, pipeline name, number of trips |
| GET | `/trips` | – | `{"trips": [{"trip_id", "n_points"}]}` |
| POST | `/predict` | `{"trip_id": "..."}` | see below |
| POST | `/evaluate/{trip_id}` | – | same as `/predict` |

Response: `gps` `[[lat, lon], ...]`, `predicted_route` (list of polylines, split wherever consecutive predicted arcs are not connected), `network` (road segments near the track, for map context), `metrics` `{top1_accuracy, route_continuity, route_precision, route_recall, n_obs, inference_ms}`.

## Plugging in the real pipeline

1. `backend/predictor.py`: replace the body of `predict_arcs(trip)` with candidates/features (M1) → tuned classifier (M2) → Viterbi (M3). It must return one `arc_id` per GPS observation (`-1` if none). Update `PIPELINE_NAME`.
2. `backend/scoring.py`: replace `point_labels` with M1's label alignment so the API scores exactly like the report.

## Tests

```bash
pytest tests/
```

Includes a check that no ground truth appears in API responses and that the predictor refuses trips that still carry a route.

## EDA and maps

```bash
python -m src.eda                                         # results/eda/ + results/maps/
python -m maps.make_map --partition train --trip 00000001  # single-trip map
```
