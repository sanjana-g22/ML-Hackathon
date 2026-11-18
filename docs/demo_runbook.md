# PathSnap final demo runbook

Laptop 1 (Windows) runs the API with the evaluator trips. Laptop 2 runs the UI.
Pipeline in the demo: M1 candidates and features → Random Forest → HMM/Viterbi (Member 3).

## Before the evaluation

- [ ] Both laptops: `git pull` on `main`; `pip install -r requirements.txt` in the venv.
- [ ] Laptop 1: evaluator trips extracted to `data/evaluator_test/` (gitignored, never committed).
- [ ] Both laptops on the **same phone hotspot** with internet (map backgrounds load from OpenStreetMap).
- [ ] Laptop 1 firewall rule for port 8000 exists (admin PowerShell, once):
      `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
- [ ] Mac only: System Settings → Privacy & Security → **Local Network** → allow Terminal.

## Laptop 1: API (PowerShell, one line at a time)

```powershell
cd C:\Users\rohan\ML-Hackathon
$env:EVAL_DIR = "C:\Users\rohan\ML-Hackathon\data\evaluator_test"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --no-use-colors
```

Expected startup lines:

```
loaded 10 trips from ...\data\evaluator_test
pipeline: Random Forest + HMM/Viterbi (Member 3) ...
Uvicorn running on http://0.0.0.0:8000
```

Then `ipconfig` → note the Wi-Fi IPv4 address (changes on every hotspot reconnect).

## Laptop 2: UI

Browser check first: `http://<laptop-1-ip>:8000/health` must show `"status":"ok"` and `"n_trips":10`.

macOS:

```bash
export API_URL=http://<laptop-1-ip>:8000
streamlit run frontend/app.py
```

Windows: `set API_URL=http://<laptop-1-ip>:8000` then `streamlit run frontend/app.py`.

## During the evaluation

1. Sidebar shows **Connected · 10 trips** and pipeline **Random Forest + HMM/Viterbi**.
2. The evaluator picks any trip → **Run prediction**. Each takes **2–21 s** (largest trip 00000015 ≈ 21 s);
   the status line says "Evaluating trip …" meanwhile.
3. Point at Laptop 1's terminal: one line per request with the client IP, trip, inference time,
   `decoder=hmm`, Top-1 and continuity.
4. Map tab: GPS (blue), predicted route (red), nearby roads (grey), route breaks (orange).
   Session history tab lists every trip run.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `/health` times out from Laptop 2 | Same hotspot? Firewall rule? Re-check Laptop 1 IP with `ipconfig` |
| Mac: "No route to host" | Local Network permission for Terminal/browser |
| "Connection refused" | uvicorn not started with `--host 0.0.0.0` |
| Log shows `decoder=classifier-only` | HMM failed for that trip; the warning line above it says why. The route is still returned |
| Grey map background | No internet on the hotspot; routes still draw |
| Port 8000 already in use | Another API window is still open: close it or press Ctrl+C there |

## Facts to have ready

- Student-test, scored like the report (`results/hmm_vs_classifier_api_scoring.csv`):
  Top-1 0.818 → 0.825, route continuity 0.898 → 0.965, sequence F1 0.609 → 0.729, route breaks 1,876 → 628.
- The HMM receives each point's 10 most probable candidates: identical student-test results to using all
  candidates, 4–5× faster.
- Ground truth stays on Laptop 1 (`backend/scoring.py`); the UI receives GPS, the predicted route and
  aggregate metrics only.
