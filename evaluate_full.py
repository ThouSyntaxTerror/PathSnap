"""Run the same server pipeline on every student-test trip for report evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    os.environ["PATHSNAP_DATA_ROOT"] = str(args.data.resolve())
    os.environ.pop("PATHSNAP_EVAL_ROOT", None)
    os.environ["PATHSNAP_MODEL"] = str((args.output / "final_model.joblib").resolve())
    import server
    server.startup()
    rows = []
    for trip_id in server.trips()["trip_ids"]:
        m = server.predict(server.Request(trip_id=trip_id))["metrics"]
        rows.append({"trip_id": trip_id, "gps_points": m["gps_points"],
            "approx_accuracy_before": m.get("approx_pointwise_accuracy_classifier"),
            "approx_accuracy_after": m.get("approx_pointwise_accuracy_viterbi"),
            "continuity_before": m["continuity_classifier"], "continuity_after": m["continuity_viterbi"],
            "reference_arc_recall": m.get("reference_arc_recall"),
            "bridge_arcs_inserted": m["bridge_arcs_inserted"],
            "unresolved_connections": m["unresolved_connections"]})
        print(f"{trip_id}: {m['gps_points']} GPS points")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "full_student_test_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
