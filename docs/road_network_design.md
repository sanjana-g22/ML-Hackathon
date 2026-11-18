# PathSnap: Directed Road-Network Graph Architecture (Member 3)

**Author:** Member 3  
**Module:** `src/road_network.py`  
**Test Suite:** `tests/test_road_network.py`  
**Validation Artifact:** `results/road_network_validation.csv`  

---

## 1. Overview and Core Purpose

In map-matching and route reconstruction, GPS observations cannot be mapped onto an arbitrary collection of detached line segments. Vehicles operate under strict physical and legal topological constraints: one-way streets, divided highways, turn restrictions, and continuous pavement connectivity.

The **Road-Network Graph Module** provides the foundational spatial and topological backbone for the entire PathSnap pipeline:
1. **Candidate Spatial Evaluation:** Serves road segment geometries and endpoints to Member 1 for spatial candidate generation.
2. **Interactive Mapping:** Serves continuous arc polylines and metadata to Member 4 for Folium rendering.
3. **Sequential Route Decoding:** Models edge reachability, topological successor sets, and shortest-path distances to compute HMM transition probabilities ($A_{ij}$) for Member 3's upcoming Viterbi decoder.

---

## 2. Technical Design & Data Assumptions

### 2.1 Data Ingestion & Indexing Preservation
- **Nodes (`.nodes`):** 2 whitespace-separated columns (`longitude`, `latitude`).  
  *Convention:* The **0-based row index** serves as the immutable `node_id` $\in [0, N_{\text{nodes}} - 1]$. Coordinates are stored as node attributes in WGS84 angular degrees.
- **Arcs (`.arcs`):** 2 whitespace-separated columns (`from_node_id`, `to_node_id`).  
  *Convention:* The **0-based row index** serves as the immutable `arc_id` $\in [0, N_{\text{arcs}} - 1]$. Arcs represent directed road segments originating at `from_node_id` and terminating at `to_node_id`.
- **Integrity Validation:** Every graph construction automatically verifies that $0 \le u, v < N_{\text{nodes}}$ and checks for duplicate IDs, self-loops, and zero-length segments.

### 2.2 NetworkX MultiDiGraph Choice & Parallel Edges
Real road networks frequently contain **parallel directed edges** connecting the exact same pair of junction nodes $(u, v)$ — for example, dual carriageways, express vs local lanes, bus-only lanes, or parallel ramps. 
- In Trip `00000001`, empirical audit reveals **292 unique parallel node pairs** (584 total parallel arcs).
- A standard `nx.DiGraph` silently collapses multiple edges between $(u, v)$ into a single edge, corrupting the arc index and dropping valid road segments.
- `RoadNetwork` utilizes `nx.MultiDiGraph`, storing `key=arc_id` for every edge. This guarantees that:
  - Every single `arc_id` is preserved distinctly.
  - Parallel edges coexist without collisions.
  - Edge lookup by `(u, v, key=arc_id)` is $O(1)$.
  - Fast global lookup via internal `_arc_lookup[arc_id] = (u, v, arc_id)` provides instant $O(1)$ access from any `arc_id`.

### 2.3 Geographic Geodesic Distance (WGS84 Ellipsoid)
Ordinary Euclidean distance in degrees $(\Delta \text{lon}^2 + \Delta \text{lat}^2)^{1/2}$ is physically invalid because longitudinal degree distance shrinks by $\cos(\text{latitude})$.
- We calculate physical arc length using **`pyproj.Geod(ellps="WGS84")`**, solving the inverse geodesic problem along the standard reference ellipsoid.
- Arc lengths are vectorized across entire networks in NumPy, computing >165,000 edge lengths in ~30 milliseconds.
- Stored as the primary edge attribute `length_m` (in metres), which serves as edge weight during shortest-path calculations.

### 2.4 Graph Connectivity & Disconnected Handling
Road networks extracted around vehicle corridors are subgraphs that often contain minor disconnected fragments (e.g. isolated pedestrian paths, parking aisles, or railway tracks).
- Shortest-path calculations (`shortest_path_distance` and `shortest_path`) catch `nx.NetworkXNoPath` and `nx.NodeNotFound` internally.
- If a destination node is unreachable or belongs to a disconnected component, the function returns `None` gracefully without throwing an unhandled exception or halting the pipeline.

---

## 3. Public API & Function Signatures

### 3.1 Class: `RoadNetwork`

```python
class RoadNetwork:
    """Encapsulates a directed road-network MultiDiGraph with domain helpers."""

    graph: nx.MultiDiGraph
    trip_id: Optional[str]
    validation_report: Dict[str, Any]

    # Properties
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_arcs(self) -> int: ...

    # Node & Arc Geometry Queries
    def get_node_coordinates(self, node_id: int) -> Tuple[float, float]:
        """Returns (longitude, latitude) for a given node_id."""

    def get_arc_endpoints(self, arc_id: int) -> Tuple[int, int]:
        """Returns (from_node_id, to_node_id) in O(1) time."""

    def get_arc_geometry(self, arc_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Returns line geometry ((from_lon, from_lat), (to_lon, to_lat))."""

    def get_arc_length(self, arc_id: int) -> float:
        """Returns physical geodesic length in metres."""

    # Topological Successor & Predecessor Queries
    def get_successor_arcs(self, arc_id: int) -> List[int]:
        """Returns all arc IDs originating from this arc's destination node."""

    def get_predecessor_arcs(self, arc_id: int) -> List[int]:
        """Returns all arc IDs terminating at this arc's origin node."""

    # Shortest-Path & Reachability
    def shortest_path_distance(
        self, source_node: int, target_node: int, weight: str = "length_m"
    ) -> Optional[float]:
        """Returns shortest path network distance in metres (or None if unreachable)."""

    def shortest_path(
        self, source_node: int, target_node: int, weight: str = "length_m"
    ) -> Optional[List[int]]:
        """Returns node sequence [source, ..., target] (or None if unreachable)."""

    def shortest_path_between_arcs(
        self, from_arc_id: int, to_arc_id: int, weight: str = "length_m"
    ) -> Optional[float]:
        """
        Computes network distance from destination of from_arc to origin of to_arc.
        Returns 0.0 if directly continuous; None if unreachable.
        """

    # Graph Profiling & Statistics
    def compute_degree_statistics(self) -> Dict[str, Any]: ...
    def compute_connectivity_statistics(self) -> Dict[str, Any]: ...
    def compute_arc_length_statistics(self) -> Dict[str, Any]: ...
    def summary(self) -> Dict[str, Any]: ...
```

### 3.2 Constructor Functions

```python
def build_road_network(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    trip_id: Optional[str] = None,
    ellipsoid: str = "WGS84"
) -> RoadNetwork:
    """Builds a validated RoadNetwork instance from nodes and arcs DataFrames."""

def load_and_build_road_network(
    nodes_source: Union[str, pd.DataFrame],
    arcs_source: Optional[Union[str, pd.DataFrame]] = None,
    trip_id: Optional[str] = None,
    partition: str = "train",
    base_dir: str = "."
) -> RoadNetwork:
    """Convenience factory: loads via Member 1 loader or raw filepaths and returns RoadNetwork."""
```

---

## 4. Team Integration Guide

### 4.1 How Member 1 Uses the Graph for Candidate Generation
Member 1 is responsible for generating candidate road segments for each noisy GPS observation $z_t$:
```python
from src.road_network import load_and_build_road_network

net = load_and_build_road_network(trip_id="00000001", partition="train")

# For any candidate arc_id:
u, v = net.get_arc_endpoints(arc_id)
(from_lon, from_lat), (to_lon, to_lat) = net.get_arc_geometry(arc_id)
length_metres = net.get_arc_length(arc_id)

# Calculate perpendicular distance from GPS (lon, lat) to segment ((from_lon, from_lat), (to_lon, to_lat))
# Calculate heading / bearing alignment between trajectory vector and road arc
```

### 4.2 How Member 4 Uses the Graph for Folium Maps
Member 4 renders the interactive Folium map (`maps/sample_trip_map.html`):
```python
# To render the road network layer or reference route:
for arc_id in route_arcs:
    ((lon1, lat1), (lon2, lat2)) = net.get_arc_geometry(arc_id)
    # Folium expects [latitude, longitude]
    folium.PolyLine(
        locations=[[lat1, lon1], [lat2, lon2]],
        color="#27ae60",
        weight=4
    ).add_to(route_group)
```

### 4.3 How Member 3 Uses the Graph for HMM Transitions (Upcoming Phase)
In Hidden Markov Model route reconstruction, the transition probability between candidate state $c_{t, i}$ (on arc $a_i$) and candidate state $c_{t+1, j}$ (on arc $a_j$) depends on the discrepancy between the network shortest-path distance $d_{\text{net}}$ and the great-circle displacement $d_{\text{gps}}$:

$$P(a_j \mid a_i) = \frac{1}{\beta} \exp\left(-\frac{|d_{\text{net}}(a_i \to a_j) - d_{\text{gps}}(z_t, z_{t+1})|}{\beta}\right)$$

Member 3's HMM transition matrix builder will compute:
```python
# Topological reachability check
if a_j in net.get_successor_arcs(a_i):
    # Direct legal transition
    d_net = 0.0
else:
    # Shortest path network distance from end of a_i to start of a_j
    d_net = net.shortest_path_between_arcs(a_i, a_j)

if d_net is None:
    # Topologically unreachable
    prob = 1e-9  # Transition penalty floor
```

---

## 5. Real-Data Validation Results (Trip `00000001`)

```
================================================================================
Metric                           Value / Finding
================================================================================
Trip ID                          00000001 (train partition)
Nodes Count                      88,809
Arcs Count                       165,790
Valid Edges                      165,790 (100.0%)
Invalid Node References          0
Duplicate Arc IDs                0
Self-loops (u == v)              0
Parallel Directed Arcs           584 edges (292 unique parallel node pairs)
Zero-length Arcs (<1cm)          8 segments (split junction nodes sharing coordinates)
Isolated Nodes Count             0
Weakly Connected Components      175 (Largest component: 86,854 nodes, 97.80%)
Strongly Connected Components    1,942 (Largest component: 84,999 nodes, 95.71%)
In-Degree Statistics             Min: 0, Max: 6, Mean: 1.87, Median: 2.0
Out-Degree Statistics            Min: 0, Max: 6, Mean: 1.87, Median: 2.0
Arc Physical Length (metres)     Min: 0.0m, Max: 1,107.17m, Mean: 39.70m, Median: 20.23m
Total Network Pavement Length    6,581.32 km
Graph Construction Time          0.4930 seconds
================================================================================
```
