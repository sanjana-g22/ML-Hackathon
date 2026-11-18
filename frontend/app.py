"""
PathSnap evaluator UI (runs on Laptop 2).

Start (from the repo root):
    streamlit run frontend/app.py

The API address comes from the API_URL environment variable, or can be edited
in the sidebar, e.g. http://192.168.43.10:8000 (Laptop 1's hotspot IP).
The UI never sees ground truth: it only shows what the API returns.
"""
import os
import sys

import requests
import streamlit as st
from streamlit_folium import st_folium

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maps.make_map import make_map  # noqa: E402

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching evaluator")

api_url = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_trips(url: str):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    st.sidebar.success(f"API online · {health['n_trips']} trips")
    st.sidebar.caption(f"Pipeline: {health['pipeline']}")
    trips = fetch_trips(api_url)
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"Cannot reach API at {api_url}")
    st.error(f"API not reachable: {e}")
    st.stop()

labels = {f"{t['trip_id']}  ({t['n_points']} points)": t["trip_id"] for t in trips}
choice = st.selectbox("Evaluation trip", list(labels))

if st.button("Run prediction", type="primary"):
    with st.spinner("Sending trip to the API..."):
        try:
            r = requests.post(f"{api_url}/predict", json={"trip_id": labels[choice]}, timeout=300)
            r.raise_for_status()
            st.session_state["result"] = r.json()
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

res = st.session_state.get("result")
if res:
    m = res["metrics"]

    def pct(v):
        return "n/a" if v is None else f"{v:.1%}"

    c = st.columns(3)
    c[0].metric("Top-1 accuracy", pct(m.get("top1_accuracy")))
    c[1].metric("Route continuity", pct(m.get("route_continuity")))
    c[2].metric("Inference time", f"{m['inference_ms']:.0f} ms")
    c = st.columns(3)
    c[0].metric("Route precision", pct(m.get("route_precision")))
    c[1].metric("Route recall", pct(m.get("route_recall")))
    c[2].metric("GPS observations", m["n_obs"])
    st.caption(f"Trip {res['trip_id']} · pipeline: {res['pipeline']} · metrics computed server-side")

    fmap = make_map(
        gps=res["gps"],
        network=res["network"],
        predicted=res["predicted_route"],
        title=f"Trip {res['trip_id']}",
        gps_every=max(1, len(res["gps"]) // 3000),
    )
    st_folium(
        fmap, height=620, use_container_width=True, returned_objects=[], key=f"map_{res['trip_id']}",
        center=fmap.location, zoom=fmap.options.get("zoom"),   # st_folium ignores fit_bounds, so pass the view
    )
