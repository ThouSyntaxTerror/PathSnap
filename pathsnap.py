"""Raw trip loading, candidate road features, and route decoding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx

FEATURE_NAMES = ["log1p_distance_m", "direction_cos", "log1p_speed_mps", "log1p_segment_length_m", "log1p_time_gap_s", "log1p_movement_m"]


def read_array(path: Path, columns: int, dtype=float) -> np.ndarray:
    array = np.loadtxt(path, dtype=dtype, ndmin=2)
    if array.shape[1] != columns:
        raise ValueError(f"{path}: expected {columns} columns, got {array.shape[1]}")
    return array


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None

    @classmethod
    def load(cls, folder: Path, include_route: bool = False) -> "Trip":
        trip_id = folder.name
        route_path = folder / f"{trip_id}.route"
        route = read_array(route_path, 1, int).ravel() if include_route and route_path.exists() else None
        track = read_array(folder / f"{trip_id}.track", 3)
        nodes = read_array(folder / f"{trip_id}.nodes", 2)
        arcs = read_array(folder / f"{trip_id}.arcs", 2, int)
        if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
            raise ValueError(f"{trip_id}: arc references invalid node")
        if route is not None and (np.any(route < 0) or np.any(route >= len(arcs))):
            raise ValueError(f"{trip_id}: route references invalid arc")
        return cls(trip_id, track, nodes, arcs, route)


class RoadIndex:
    def __init__(self, trip: Trip):
        self.trip = trip
        # Equirectangular projection centred on the trip. All geometric features use metres.
        center = np.mean(trip.track[:, :2], axis=0)
        self.center = center
        self.scale = np.array([111_320 * np.cos(np.deg2rad(center[1])), 111_320.0])
        self.track_xy = (trip.track[:, :2] - center) * self.scale
        self.node_xy = (trip.nodes - center) * self.scale
        self.a = self.node_xy[trip.arcs[:, 0]]
        self.b = self.node_xy[trip.arcs[:, 1]]
        self.v = self.b - self.a
        self.length = np.linalg.norm(self.v, axis=1)
        sample = np.concatenate((self.a, self.b, (self.a + self.b) / 2))
        self.tree = cKDTree(sample)
        self.n_arcs = len(trip.arcs)

    def exact_distances(self, points: np.ndarray, arc_ids: np.ndarray) -> np.ndarray:
        start, vector = self.a[arc_ids], self.v[arc_ids]
        projection = np.sum((points - start) * vector, axis=-1) / np.maximum(np.sum(vector * vector, axis=-1), 1e-9)
        projection = np.clip(projection, 0, 1)
        closest = start + projection[..., None] * vector
        return np.linalg.norm(points - closest, axis=-1)

    def candidate_ids(self, points: np.ndarray, count: int = 6, samples: int = 18) -> np.ndarray:
        _, sample_ids = self.tree.query(points, k=min(samples, self.n_arcs * 3))
        sample_ids = np.atleast_2d(sample_ids)
        if sample_ids.shape[0] != len(points):
            sample_ids = sample_ids.T
        arc_ids = sample_ids % self.n_arcs
        distance = self.exact_distances(points[:, None, :], arc_ids)
        # Duplicate sampled endpoints can occur; retain closest distinct arcs per point.
        order = np.argsort(distance, axis=1)
        output = np.empty((len(points), count), dtype=np.int64)
        for i, ranking in enumerate(order):
            unique = list(dict.fromkeys(int(arc_ids[i, j]) for j in ranking))
            output[i] = (unique + [unique[-1]] * count)[:count]
        return output

    def reference_arc_per_point(self, points: np.ndarray) -> np.ndarray:
        if self.trip.route is None:
            raise ValueError("Reference route unavailable")
        ref = np.unique(self.trip.route)
        # The route is small enough to search exactly in batches; this is used only for labels/metrics.
        out = np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), 128):
            batch = points[start:start + 128]
            distances = self.exact_distances(batch[:, None, :], ref[None, :])
            out[start:start + len(batch)] = ref[np.argmin(distances, axis=1)]
        return out

    def features(self, point_indices: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        points = self.track_xy[point_indices]
        prev_idx = np.maximum(point_indices - 1, 0)
        movement = points - self.track_xy[prev_idx]
        gap = self.trip.track[point_indices, 2] - self.trip.track[prev_idx, 2]
        move_len = np.linalg.norm(movement, axis=1)
        speed = move_len / np.maximum(gap, 1)
        road_vec = self.v[candidates]
        cosine = np.sum(movement[:, None, :] * road_vec, axis=2) / np.maximum(move_len[:, None] * self.length[candidates], 1e-6)
        dist = self.exact_distances(points[:, None, :], candidates)
        k = candidates.shape[1]
        return np.stack((
            np.log1p(dist), cosine, np.broadcast_to(np.log1p(speed[:, None]), (len(points), k)),
            np.log1p(self.length[candidates]), np.broadcast_to(np.log1p(gap[:, None]), (len(points), k)),
            np.broadcast_to(np.log1p(move_len[:, None]), (len(points), k)),
        ), axis=2)


def make_rows(trip: Trip, max_points: int | None = None, candidate_count: int = 6):
    road = RoadIndex(trip)
    if max_points and len(trip.track) > max_points:
        point_ids = np.unique(np.linspace(0, len(trip.track) - 1, max_points).astype(int))
    else:
        point_ids = np.arange(len(trip.track))
    points = road.track_xy[point_ids]
    candidates = road.candidate_ids(points, candidate_count)
    features = road.features(point_ids, candidates)
    labels = None
    if trip.route is not None:
        reference = road.reference_arc_per_point(points)
        labels = (candidates == reference[:, None]).astype(np.int8)
    return road, point_ids, candidates, features, labels


def top1_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    covered = labels.any(axis=1)
    return float(np.mean(labels[np.arange(len(scores)), np.argmax(scores, axis=1)])) if len(scores) else 0.0


def decode(trip: Trip, road: RoadIndex, point_ids: np.ndarray, candidates: np.ndarray, scores: np.ndarray,
           transition_weight: float = 1.0) -> np.ndarray:
    """Viterbi over candidate arcs; transition favours shared nodes and short road-to-road gaps."""
    n, k = candidates.shape
    log_emission = np.log(np.clip(scores, 1e-6, 1.0))
    dp = log_emission[0].copy()
    back = np.zeros((n, k), dtype=np.int8)
    for t in range(1, n):
        prev, curr = candidates[t - 1], candidates[t]
        prev_nodes, curr_nodes = trip.arcs[prev], trip.arcs[curr]
        same = prev[:, None] == curr[None, :]
        connected = prev_nodes[:, 1, None] == curr_nodes[None, :, 0]
        # Fallback for gaps/disconnected candidates: endpoint distance is a proxy, not graph shortest path.
        endpoint_gap = np.min(np.linalg.norm(road.node_xy[prev_nodes][:, None, :, None, :] -
                                             road.node_xy[curr_nodes][None, :, None, :, :], axis=-1), axis=(2, 3))
        elapsed = max(float(trip.track[point_ids[t], 2] - trip.track[point_ids[t - 1], 2]), 1.0)
        penalty = np.where(same, 0.0, np.where(connected, 0.25, 1.5 + endpoint_gap / (25 * elapsed)))
        possible = dp[:, None] - transition_weight * penalty
        back[t] = np.argmax(possible, axis=0)
        dp = log_emission[t] + np.max(possible, axis=0)
    chosen = np.empty(n, dtype=np.int64)
    state = int(np.argmax(dp))
    for t in range(n - 1, -1, -1):
        chosen[t] = candidates[t, state]
        if t: state = int(back[t, state])
    return chosen


def continuity(trip: Trip, arcs: np.ndarray) -> float:
    if len(arcs) < 2: return 1.0
    a, b = trip.arcs[arcs[:-1]], trip.arcs[arcs[1:]]
    same = arcs[:-1] == arcs[1:]
    touch = a[:, 1] == b[:, 0]
    return float(np.mean(same | touch))


def bridge_route(trip: Trip, road: RoadIndex, chosen: np.ndarray):
    """Insert directed shortest-path arcs between decoded observations where possible."""
    graph = nx.DiGraph()
    for arc_id, (start, end) in enumerate(trip.arcs):
        weight = max(float(road.length[arc_id]), 0.1)
        if not graph.has_edge(int(start), int(end)) or weight < graph[int(start)][int(end)]["weight"]:
            graph.add_edge(int(start), int(end), arc_id=arc_id, weight=weight)
    collapsed = [(int(chosen[0]), 0)]
    for index, arc in enumerate(chosen[1:], 1):
        if int(arc) != collapsed[-1][0]: collapsed.append((int(arc), index))
    route = [collapsed[0][0]]
    inserted = 0
    unresolved = 0
    for (next_arc, point_index), (_, prior_index) in zip(collapsed[1:], collapsed[:-1]):
        start = int(trip.arcs[route[-1], 1])
        end = int(trip.arcs[next_arc, 0])
        if start != end:
            try:
                path = nx.astar_path(graph, start, end,
                    heuristic=lambda a, b: float(np.linalg.norm(road.node_xy[a] - road.node_xy[b])), weight="weight")
                bridge = [graph[a][b]["arc_id"] for a, b in zip(path[:-1], path[1:])]
                bridge_m = sum(float(road.length[arc]) for arc in bridge)
                elapsed = max(float(trip.track[point_index, 2] - trip.track[prior_index, 2]), 1.0)
                movement = float(np.linalg.norm(road.track_xy[point_index] - road.track_xy[prior_index]))
                plausible_limit = max(100.0, min(3000.0, 40.0 * elapsed + 3.0 * movement))
                if bridge_m <= plausible_limit:
                    route.extend(bridge)
                    inserted += len(bridge)
                else:
                    unresolved += 1
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                unresolved += 1
        route.append(next_arc)
    return np.asarray(route, dtype=np.int64), inserted, unresolved
