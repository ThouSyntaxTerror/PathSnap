"""Data-quality audit on the RAW trip files (before cleaning) with a time-gap figure.

Writes artifacts/data_quality.json and artifacts/data_time_gaps.png so the cleaning
decisions in pathsnap.clean_track are backed by measured counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def audit_trip(folder: Path) -> dict:
    tid = folder.name
    issues = {"missing_files": [f for f in (".track", ".nodes", ".arcs") if not (folder / f"{tid}{f}").exists()]}
    if issues["missing_files"]:
        return {"trip_id": tid, "issues": issues, "gaps": np.array([])}
    track = np.loadtxt(folder / f"{tid}.track", ndmin=2)
    nodes = np.loadtxt(folder / f"{tid}.nodes", ndmin=2)
    arcs = np.loadtxt(folder / f"{tid}.arcs", ndmin=2, dtype=int)
    t = track[:, 2]
    gaps = np.diff(t)
    issues["nan_rows"] = int(np.sum(~np.all(np.isfinite(track), axis=1)))
    issues["out_of_range_coordinates"] = int(np.sum((np.abs(track[:, 0]) > 180) | (np.abs(track[:, 1]) > 90)))
    issues["not_time_sorted"] = int(np.sum(gaps < 0))
    issues["duplicate_timestamps"] = int(np.sum(gaps == 0))
    issues["arcs_with_invalid_node"] = int(np.sum((arcs < 0) | (arcs >= len(nodes))))
    issues["self_loop_arcs"] = int(np.sum(arcs[:, 0] == arcs[:, 1])) if len(arcs) else 0
    route = folder / f"{tid}.route"
    if route.exists():
        r = np.loadtxt(route, ndmin=1, dtype=int)
        issues["route_invalid_arc"] = int(np.sum((r < 0) | (r >= len(arcs))))
        issues["route_length"] = int(len(r))
    issues["track_length"] = int(len(track))
    return {"trip_id": tid, "issues": issues, "gaps": gaps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary, all_gaps = {}, []
    for partition in ("train", "student_test"):
        rows = [audit_trip(f) for f in sorted((args.data / partition).iterdir()) if f.is_dir()]
        totals = {}
        for r in rows:
            for k, v in r["issues"].items():
                if isinstance(v, int):
                    totals[k] = totals.get(k, 0) + v
                elif v:
                    totals.setdefault("trips_with_missing_files", []).append(r["trip_id"])
            all_gaps.append(r["gaps"])
        gaps = np.concatenate([r["gaps"] for r in rows]) if rows else np.array([])
        summary[partition] = {"trips": len(rows), "totals": totals,
                              "gap_share_1s": float(np.mean(gaps == 1)) if len(gaps) else None,
                              "gap_p50_s": float(np.median(gaps)) if len(gaps) else None,
                              "gap_p99_s": float(np.percentile(gaps, 99)) if len(gaps) else None,
                              "gap_max_s": float(gaps.max()) if len(gaps) else None,
                              "per_trip": [{"trip_id": r["trip_id"], **r["issues"]} for r in rows]}
    (args.output / "data_quality.json").write_text(json.dumps(summary, indent=2))

    gaps = np.concatenate(all_gaps)
    gaps = gaps[(gaps > 0) & (gaps <= 60)]
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.hist(gaps, bins=np.arange(0.5, 61.5, 1), color="#5388ba", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Seconds between consecutive GPS observations (gaps up to 60 s)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("GPS sampling gaps across all raw trips")
    fig.savefig(args.output / "data_time_gaps.png", dpi=160)
    print("Saved data_quality.json and data_time_gaps.png")


if __name__ == "__main__":
    main()
