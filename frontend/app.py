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
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from maps.make_map import make_map, route_breaks  # noqa: E402

st.set_page_config(
    page_title="PathSnap Evaluator",
    page_icon=":material/route:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1400px;}
      .ps-sub {opacity: .65; margin-top: -.6rem; margin-bottom: 1.2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.9rem;}
      .ps-note {font-size: .85rem; opacity: .7;}
    </style>
    """,
    unsafe_allow_html=True,
)

METRIC_HELP = {
    "top1": "Share of GPS points whose predicted road segment matches the reference segment. "
            "Computed on Laptop 1; the reference route never leaves the server.",
    "continuity": "Share of consecutive predicted segments that are the same or directly connected "
                  "in the road network. Higher means a more drivable route.",
    "precision": "Of the road segments the route uses, the share that are on the reference route.",
    "recall": "Of the reference route's road segments, the share the predicted route covers.",
    "breaks": "Number of places where the predicted route jumps between unconnected road segments.",
    "latency": "Time spent in the prediction pipeline on Laptop 1 (excludes network and map drawing).",
}


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_health(url: str):
    t0 = time.perf_counter()
    r = requests.get(f"{url}/health", timeout=5)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000


@st.cache_data(ttl=60, show_spinner=False)
def api_trips(url: str):
    r = requests.get(f"{url}/trips", timeout=10)
    r.raise_for_status()
    return r.json()["trips"]


def api_predict(url: str, trip_id: str):
    t0 = time.perf_counter()
    r = requests.post(f"{url}/predict", json={"trip_id": trip_id}, timeout=300)
    r.raise_for_status()
    return r.json(), (time.perf_counter() - t0) * 1000


def pct(v):
    return "n/a" if v is None or v != v else f"{v:.1%}"


def fmt_duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes} min" if minutes < 120 else f"{minutes / 60:.1f} h"


# --------------------------------------------------------------------------- #
# Sidebar: connection + display options
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Connection")
    api_url = st.text_input(
        "API address",
        os.environ.get("API_URL", "http://127.0.0.1:8000"),
        help="Laptop 1's address, e.g. http://192.168.43.10:8000",
    ).rstrip("/")

    try:
        health, ping_ms = api_health(api_url)
        st.success(f"Connected · {health['n_trips']} trips · {ping_ms:.0f} ms", icon=":material/check_circle:")
        st.caption(f"Pipeline: **{health['pipeline']}**")
        connected, err = True, None
    except Exception as e:  # noqa: BLE001
        st.error("API not reachable", icon=":material/error:")
        connected, err = False, e

    st.divider()
    st.subheader("Map display")
    basemap = st.segmented_control("Basemap", ["Streets", "Satellite"], default="Streets") or "Streets"
    show_network = st.toggle("Road network near the track", value=True)
    show_gps = st.toggle("GPS observations", value=True)
    show_breaks = st.toggle("Highlight route breaks", value=True,
                            help="Orange rings mark where consecutive predicted segments are not connected.")

    st.divider()
    with st.expander("How this works"):
        st.markdown(
            "1. This page asks **Laptop 1** for the available trips.\n"
            "2. **Run prediction** sends only the trip id to `/predict`.\n"
            "3. Laptop 1 runs the pipeline (candidates → classifier → HMM/Viterbi) and "
            "scores it against its own reference data.\n"
            "4. This page receives GPS points, the predicted route and the metrics. "
            "Ground-truth routes are never sent here."
        )

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("PathSnap")
st.markdown('<div class="ps-sub">Map-matching evaluator · noisy GPS → road-network route</div>',
            unsafe_allow_html=True)

if not connected:
    with st.container(border=True):
        st.markdown(f"#### Can't reach the API at `{api_url}`")
        st.markdown(
            "- Is `uvicorn backend.main:app --host 0.0.0.0 --port 8000` running on Laptop 1?\n"
            "- Are both laptops on the same hotspot?\n"
            "- Is port 8000 allowed through Laptop 1's firewall?\n"
            "- Is the address in the sidebar Laptop 1's current IP (`ipconfig`)?"
        )
        st.caption(f"Error: {err}")
    st.stop()

# --------------------------------------------------------------------------- #
# Trip selection
# --------------------------------------------------------------------------- #
trips = api_trips(api_url)
trip_by_id = {t["trip_id"]: t for t in trips}
history = st.session_state.setdefault("history", [])
evaluated = {h["Trip"] for h in history}

with st.container(border=True):
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    trip_id = c1.selectbox(
        f"Evaluation trip ({len(trips)} available)",
        list(trip_by_id),
        format_func=lambda t: (
            f"Trip {t}  ·  {trip_by_id[t]['n_points']:,} GPS points  ·  "
            f"{fmt_duration(trip_by_id[t].get('duration_s', 0))}"
            + ("  ·  ✓ evaluated" if t in evaluated else "")
        ),
        key="trip_select",   # stable identity: the labels change after a run, the selection must not reset
    )
    run = c2.button("Run prediction", type="primary", icon=":material/play_arrow:", width="stretch")

if run:
    with st.status(f"Evaluating trip {trip_id}…", expanded=False) as status:
        try:
            status.write("Sending trip id to Laptop 1")
            res, roundtrip_ms = api_predict(api_url, trip_id)
            res["roundtrip_ms"] = roundtrip_ms
            st.session_state["result"] = res
            m = res["metrics"]
            history.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Trip": trip_id,
                "GPS points": m["n_obs"],
                "Top-1": m.get("top1_accuracy"),
                "Continuity": m.get("route_continuity"),
                "Precision": m.get("route_precision"),
                "Recall": m.get("route_recall"),
                "Route breaks": max(len(res["predicted_route"]) - 1, 0),
                "Inference (ms)": m["inference_ms"],
            })
            status.update(label=f"Trip {trip_id} evaluated in {roundtrip_ms / 1000:.1f} s", state="complete")
        except requests.HTTPError as e:
            status.update(label="Prediction failed", state="error")
            st.error(f"The API returned an error: {e.response.status_code} {e.response.text}")
        except Exception as e:  # noqa: BLE001
            status.update(label="Prediction failed", state="error")
            st.error(f"Could not complete the request: {e}")

res = st.session_state.get("result")

if not res:
    with st.container(border=True):
        st.markdown("#### Select a trip and press **Run prediction**")
        st.markdown(
            '<span class="ps-note">The predicted route, accuracy metrics and a map of the trip will appear here. '
            "Every request is logged on Laptop 1.</span>",
            unsafe_allow_html=True,
        )
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
m = res["metrics"]
n_breaks = max(len(res["predicted_route"]) - 1, 0)

st.subheader(f"Trip {res['trip_id']}")
st.caption(f"Pipeline: {res['pipeline']} · metrics computed on Laptop 1 · "
           f"round trip {res.get('roundtrip_ms', 0) / 1000:.1f} s")


def bar(v):
    if v is not None and v == v:   # skip None / NaN
        st.progress(min(max(float(v), 0.0), 1.0))


k = st.columns(5)
with k[0].container(border=True):
    st.metric("Top-1 accuracy", pct(m.get("top1_accuracy")), help=METRIC_HELP["top1"])
    bar(m.get("top1_accuracy"))
with k[1].container(border=True):
    st.metric("Route continuity", pct(m.get("route_continuity")), help=METRIC_HELP["continuity"])
    bar(m.get("route_continuity"))
with k[2].container(border=True):
    st.metric("Route precision", pct(m.get("route_precision")), help=METRIC_HELP["precision"])
    st.caption(f"Recall {pct(m.get('route_recall'))}", help=METRIC_HELP["recall"])
with k[3].container(border=True):
    st.metric("Route breaks", f"{n_breaks:,}", help=METRIC_HELP["breaks"])
    st.caption(f"{m['n_obs']:,} GPS points")
with k[4].container(border=True):
    st.metric("Inference time", f"{m['inference_ms']:.0f} ms", help=METRIC_HELP["latency"])
    st.caption(f"{m['n_obs'] / max(m['inference_ms'], 1e-3) * 1000:,.0f} points / s")

tab_map, tab_history, tab_details = st.tabs(
    [":material/map: Map", f":material/history: Session history ({len(history)})", ":material/info: Details"]
)

with tab_map:
    gps = res["gps"]
    fmap = make_map(
        gps=gps,
        network=res["network"] if show_network else None,
        predicted=res["predicted_route"],
        breaks=route_breaks(res["predicted_route"]) if show_breaks else None,
        title=f"Trip {res['trip_id']}",
        gps_every=max(1, len(gps) // 3000),
        basemap=basemap,
        collapse_layers=True,
        show_gps=show_gps,
    )
    st_folium(
        fmap, height=640, use_container_width=True, returned_objects=[],
        key=f"map_{res['trip_id']}_{basemap}_{show_network}_{show_gps}_{show_breaks}",
        center=fmap.location, zoom=fmap.options.get("zoom"),
    )
    st.caption("Blue: GPS observations · red dashed: predicted route · grey: nearby roads · "
               "orange rings: route breaks. Toggle layers in the sidebar or with the layer button (top right).")

with tab_history:
    df = pd.DataFrame(history)
    st.dataframe(
        df, hide_index=True, width="stretch",
        column_config={
            "Top-1": st.column_config.ProgressColumn("Top-1", format="percent", min_value=0, max_value=1),
            "Continuity": st.column_config.ProgressColumn("Continuity", format="percent", min_value=0, max_value=1),
            "Precision": st.column_config.NumberColumn("Precision", format="percent"),
            "Recall": st.column_config.NumberColumn("Recall", format="percent"),
            "Inference (ms)": st.column_config.NumberColumn("Inference (ms)", format="%.0f"),
        },
    )
    latest = df.drop_duplicates("Trip")
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
    c1.caption(f"{len(latest)} of {len(trips)} trips evaluated this session · "
               f"mean Top-1 {pct(latest['Top-1'].mean())} · mean continuity {pct(latest['Continuity'].mean())}")
    c2.download_button("Download CSV", df.to_csv(index=False), "pathsnap_session.csv", "text/csv",
                       icon=":material/download:", width="stretch")
    if c3.button("Clear history", icon=":material/delete:", width="stretch"):
        st.session_state["history"] = []
        st.session_state.pop("result", None)
        st.rerun()

with tab_details:
    trip_meta = trip_by_id.get(res["trip_id"], {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Trip**")
        st.table(pd.DataFrame({
            "Field": ["Trip id", "GPS points", "Duration", "Predicted route pieces", "Nearby road segments drawn"],
            "Value": [res["trip_id"], f"{m['n_obs']:,}", fmt_duration(trip_meta.get("duration_s", 0)),
                      f"{len(res['predicted_route']):,}", f"{len(res['network']):,}"],
        }).set_index("Field"))
    with c2:
        st.markdown("**Metrics returned by the API**")
        st.json(m, expanded=True)
