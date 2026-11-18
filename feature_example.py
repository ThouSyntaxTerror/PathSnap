"""Export actual candidate-feature rows for a training trip as a viva example."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from pathsnap import Trip, make_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_sample.csv"))
    parser.add_argument("--trip", default="00000001")
    args = parser.parse_args()
    trip = Trip.load(args.data / "train" / args.trip, include_route=True)
    _, point_ids, candidates, features, labels = make_rows(trip, max_points=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["trip_id", "gps_row", "timestamp_s", "gps_longitude", "gps_latitude", "candidate_arc_id",
               "from_node_id", "to_node_id", "distance_m", "direction_cos", "speed_mps",
               "segment_length_m", "time_gap_s", "movement_m", "proxy_positive_label"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(point_ids):
            for j, arc in enumerate(candidates[i]):
                x = features[i, j]
                writer.writerow({
                    "trip_id": trip.trip_id, "gps_row": int(row), "timestamp_s": float(trip.track[row, 2]),
                    "gps_longitude": float(trip.track[row, 0]), "gps_latitude": float(trip.track[row, 1]),
                    "candidate_arc_id": int(arc), "from_node_id": int(trip.arcs[arc, 0]),
                    "to_node_id": int(trip.arcs[arc, 1]), "distance_m": round(float(np.expm1(x[0])), 3),
                    "direction_cos": round(float(x[1]), 4), "speed_mps": round(float(np.expm1(x[2])), 3),
                    "segment_length_m": round(float(np.expm1(x[3])), 3),
                    "time_gap_s": round(float(np.expm1(x[4])), 3),
                    "movement_m": round(float(np.expm1(x[5])), 3),
                    "proxy_positive_label": int(labels[i, j]),
                })
    print(f"Saved {len(point_ids) * candidates.shape[1]} rows to {args.output}")


if __name__ == "__main__": main()
