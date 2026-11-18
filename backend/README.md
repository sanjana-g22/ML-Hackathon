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

The API URL can also be edited in the sidebar. When testing on one machine, use
`http://127.0.0.1:8000` rather than `localhost`: on Windows `localhost` tries IPv6
first and adds about 2 s to every request.

What the evaluator sees:

- **Sidebar:** connection status with round-trip time and the loaded pipeline;
  map options (Streets/Satellite basemap, road network, GPS points, route-break markers).
- **Trip picker:** every trip the API reports, with GPS point count and duration;
  trips already run in this session are ticked.
- **Result cards:** Top-1 accuracy, route continuity, route precision/recall,
  route breaks and inference time, each with a hover explanation.
- **Map tab:** GPS points, predicted route, nearby roads and orange rings where the
  predicted route jumps between unconnected segments (what HMM decoding should remove).
- **Session history tab:** one row per run with a CSV download, useful for going
  through all ten evaluator trips.
- **Details tab:** trip facts and the raw metrics returned by the API.
- If the API cannot be reached, the page shows a troubleshooting checklist instead.

Theme colours live in `.streamlit/config.toml`; the UI follows the viewer's light/dark setting.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | status, pipeline name, number of trips |
| GET | `/trips` | – | `{"trips": [{"trip_id", "n_points"}]}` |
| POST | `/predict` | `{"trip_id": "..."}` | see below |
| POST | `/evaluate/{trip_id}` | – | same as `/predict` |

Response: `gps` `[[lat, lon], ...]`, `predicted_route` (list of polylines, split wherever consecutive predicted arcs are not connected), `network` (road segments near the track, for map context), `metrics` `{top1_accuracy, route_continuity, route_precision, route_recall, n_obs, inference_ms}`.

## Prediction pipeline

`backend/predictor.py` runs, for each requested trip:

1. **Candidates + features**: `src.member1_pipeline.inference_features` (M1), the same code and settings
   (250 m radius, 30 candidates per point) used to build the training data.
2. **Classifier**: the saved model in `artifacts/` (default `rf_model.joblib`). Switch model with
   `set PATHSNAP_MODEL=artifacts\best_model_tuned.joblib` before starting uvicorn.
3. **Decoder**: if `hmm/decoder.py` exists, its `decode_trip(trip, network, candidates, xy, valid)` is used
   (M3's Viterbi); otherwise classifier-only (highest-probability arc per GPS point).

`backend/scoring.py` labels each GPS point with M1's `align_reference` (the same labels used for training and
the report) and returns Top-1 (all points and reliable points only), route continuity, route precision/recall
and sequence F1. Only these aggregate numbers leave Laptop 1.

Checked on the 10 student-test trips with the Random Forest: the API's Top-1 equals the evaluation harness
for every trip (pooled 0.8178, as in `results/metrics_rf.json`). End-to-end time per trip is 0.1–1.2 s,
mostly feature building.

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
