"""Audit the supplied trip files and save reproducible EDA evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import Trip, read_array, clean_track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    rows = []
    for partition in ("train", "student_test"):
        for folder in sorted((args.data / partition).iterdir()):
            if not folder.is_dir(): continue
            raw_track = read_array(folder / f"{folder.name}.track", 3)
            _, cleaning = clean_track(raw_track)
            trip = Trip.load(folder, include_route=True)
            gap = np.diff(trip.track[:, 2])
            valid_gps = np.isfinite(trip.track).all(axis=1) & (np.abs(trip.track[:, 0]) <= 180) & (np.abs(trip.track[:, 1]) <= 90)
            node_degree = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
            graph = csr_matrix((np.ones(len(trip.arcs), dtype=np.int8),
                               (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(len(trip.nodes), len(trip.nodes)))
            components = connected_components(graph, directed=False, return_labels=False)
            rows.append({
                "partition": partition, "trip_id": trip.trip_id,
                "gps_points": len(trip.track), "nodes": len(trip.nodes), "arcs": len(trip.arcs),
                "raw_gps_points": cleaning["raw_points"],
                "duplicate_gps_rows_removed": cleaning["duplicate_rows"],
                "gps_outlier_rows_removed": cleaning["outlier_rows"],
                "timestamp_conflicts_removed": cleaning["timestamp_conflicts"],
                "timestamps_reordered": cleaning["timestamps_reordered"],
                "reference_route_arcs": len(trip.route) if trip.route is not None else 0,
                "invalid_gps_rows": cleaning["invalid_rows"],
                "nonincreasing_time_steps": int((gap <= 0).sum()),
                "one_second_steps": int((gap == 1).sum()),
                "time_steps": len(gap), "gaps_over_one_second": int((gap > 1).sum()),
                "max_gap_seconds": float(gap.max()) if len(gap) else 0,
                "isolated_nodes": int((node_degree == 0).sum()),
                "undirected_components": int(components),
                "longitude_min": float(trip.track[:, 0].min()), "longitude_max": float(trip.track[:, 0].max()),
                "latitude_min": float(trip.track[:, 1].min()), "latitude_max": float(trip.track[:, 1].max()),
                "file_bytes": sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),
            })
            print(f"{partition}/{trip.trip_id}: {len(trip.track)} GPS, {components} components", flush=True)
    summary = {"trips": rows, "totals": {
        "trip_count": len(rows), "gps_points": sum(r["gps_points"] for r in rows),
        "raw_gps_points": sum(r["raw_gps_points"] for r in rows),
        "duplicate_gps_rows_removed": sum(r["duplicate_gps_rows_removed"] for r in rows),
        "gps_outlier_rows_removed": sum(r["gps_outlier_rows_removed"] for r in rows),
        "timestamp_conflicts_removed": sum(r["timestamp_conflicts_removed"] for r in rows),
        "timestamps_reordered": sum(r["timestamps_reordered"] for r in rows),
        "invalid_gps_rows": sum(r["invalid_gps_rows"] for r in rows),
        "nonincreasing_time_steps": sum(r["nonincreasing_time_steps"] for r in rows),
        "one_second_steps": sum(r["one_second_steps"] for r in rows),
        "time_steps": sum(r["time_steps"] for r in rows),
        "trips_with_isolated_nodes": sum(r["isolated_nodes"] > 0 for r in rows),
        "trips_with_multiple_components": sum(r["undirected_components"] > 1 for r in rows),
        "max_gap_seconds": max(r["max_gap_seconds"] for r in rows),
    }}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset_inventory.json").write_text(json.dumps(summary, indent=2))
    print("Saved", args.output / "dataset_inventory.json")


if __name__ == "__main__": main()
