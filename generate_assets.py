"""Create model-comparison plots (including route continuity) and an interactive Folium map.

The comparison plots read only experiment_results.json, so they work on saved results.
The Folium map needs the raw student-test data folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pathsnap import Trip, make_rows, decode, bridge_route

# Old result files used these key names; new runs use the route_continuity_* names.
CONTINUITY_KEYS = {
    "before": ("route_continuity_before_decoding", "mean_continuity_classifier"),
    "after": ("route_continuity_after_decoding", "mean_continuity_viterbi"),
}


DISPLAY = {"logistic_regression": "Logistic Regression", "gaussian_nb": "Gaussian Naive Bayes",
           "decision_tree": "Decision Tree", "random_forest": "Random Forest",
           "hist_gradient_boosting": "Histogram Gradient Boosting"}


def continuity(record: dict, phase: str) -> float:
    for key in CONTINUITY_KEYS[phase]:
        if key in record:
            return float(record[key])
    raise KeyError(f"no {phase}-decoding continuity recorded")


def comparison_plots(results: dict, output: Path):
    names = list(results["models"])
    labels = [DISPLAY.get(s, s) for s in names]
    phase1 = results.get("selection_phase1", {})
    selection = results.get("selection", {})
    criterion = selection.get("criterion", "")
    stale_test_selection = not phase1 and "student-test" in criterion.lower()
    selected = phase1.get("family", selection.get("family"))
    colours = ["#147d64" if n == selected and not stale_test_selection else "#5388ba" for n in names]

    metrics = [("cv_macro_f1_mean", "Grouped training-CV macro F1"),
               ("candidate_macro_f1", "Student-test candidate macro F1"),
               ("mean_trip_top1_accuracy", "Student-test Top-1 (proxy)"),
               ("mean_trip_accuracy_viterbi", "Student-test after Viterbi (proxy)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, (key, title) in zip(axes.flat, metrics):
        values = [results["models"][n][key] for n in names]
        bars = ax.barh(labels, values, color=colours)
        ax.set_xlim(0, 1); ax.set_title(title); ax.invert_yaxis()
        for bar, val in zip(bars, values):
            ax.text(val + .01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    if stale_test_selection:
        title = "Saved results use test-informed selection; rerun train.py before submission"
    else:
        title = f"Green = selected by training CV ({labels[names.index(selected)]})"
    fig.suptitle(title, fontsize=9, color="#42566e")
    fig.savefig(output / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Route continuity: the five models before and after Viterbi decoding, on student-test trips.
    before = [continuity(results["models"][n], "before") for n in names]
    after = [continuity(results["models"][n], "after") for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.barh(y - 0.2, before, height=0.38, color="#b8c8d9", label="Before decoding (classifier argmax)")
    ax.barh(y + 0.2, after, height=0.38, color="#147d64", label="After Viterbi decoding")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b + .005, i - 0.2, f"{b:.3f}", va="center", fontsize=8)
        ax.text(a + .005, i + 0.2, f"{a:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Route continuity (share of consecutive steps that are valid; higher is better)")
    ax.set_title("Route continuity by model, student-test trips")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output / "route_continuity_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.6), layout="constrained")
    losses = [results["models"][n]["log_loss"] for n in names]
    bars = ax.barh(labels, losses, color="#6385ad")
    ax.invert_yaxis(); ax.set_title("Student-test candidate log loss (lower is better)")
    ax.set_xlim(0, max(losses) * 1.25)
    for bar, val in zip(bars, losses):
        ax.text(val + .008, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    fig.savefig(output / "log_loss_comparison.png", dpi=180)
    plt.close(fig)

    # Show each model's student-test confusion matrix as row-normalized percentages.
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), layout="constrained")
    for ax, name in zip(axes, names):
        matrix = np.asarray(results["models"][name]["confusion_matrix_tn_fp_fn_tp"], dtype=float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,.0f}\n({normalized[row, col]:.0%})",
                        ha="center", va="center", fontsize=8,
                        color="white" if normalized[row, col] > .55 else "#18324b")
        ax.set_title(DISPLAY.get(name, name), fontsize=9)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
    fig.colorbar(image, ax=list(axes), label="Share within true class", shrink=.82)
    fig.suptitle("Student-test candidate confusion matrices (counts and row percentages)")
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def folium_map(data: Path, output: Path, trip_id: str):
    folder = data / "student_test" / trip_id
    trip = Trip.load(folder, include_route=True)
    road, ids, candidates, x, _ = make_rows(trip)
    model = joblib.load(output / "final_model.joblib")
    scores = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(candidates.shape)
    predicted = decode(trip, road, ids, candidates, scores)
    routed, _, _ = bridge_route(trip, road, predicted)
    inventory = json.loads((output / "dataset_inventory.json").read_text())
    records = inventory["trips"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for partition, color in (("train", "#4784b8"), ("student_test", "#e58b33")):
        values = [r["gps_points"] for r in records if r["partition"] == partition]
        axes[0].hist(values, bins=16, alpha=.65, color=color, label=partition.replace("_", " "))
    axes[0].set_title("GPS observations per trip"); axes[0].set_xlabel("Observations"); axes[0].set_ylabel("Trips")
    axes[0].legend(fontsize=7)
    gap_rate = [100 * r["one_second_steps"] / max(r["time_steps"], 1) for r in records]
    axes[1].hist(gap_rate, bins=15, color="#639b86")
    axes[1].set_title("One-second sampling share"); axes[1].set_xlabel("Percent of time steps"); axes[1].set_ylabel("Trips")
    relevant_arcs = np.unique(candidates[::max(1, len(candidates)//250)].ravel())
    segments = np.stack((road.a[relevant_arcs], road.b[relevant_arcs]), axis=1)
    axes[2].add_collection(LineCollection(segments, colors="#aab7c6", linewidths=.5, alpha=.55))
    ref_ids = np.unique(trip.route)
    axes[2].add_collection(LineCollection(np.stack((road.a[ref_ids], road.b[ref_ids]), axis=1),
                                      colors="#e58b33", linewidths=1.7, alpha=.85))
    axes[2].scatter(road.track_xy[:, 0], road.track_xy[:, 1], s=2, color="#2779d1", alpha=.7)
    axes[2].autoscale(); axes[2].set_aspect("equal", adjustable="box")
    axes[2].set_title(f"Trip {trip_id}: GPS and roads"); axes[2].set_xlabel("Local metres"); axes[2].set_ylabel("Local metres")
    fig.savefig(output / "eda_summary.png", dpi=180)
    plt.close(fig)
    center = [float(np.mean(trip.track[:, 1])), float(np.mean(trip.track[:, 0]))]
    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)
    roads = folium.FeatureGroup(name="Nearby road network", show=True)
    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    reference = folium.FeatureGroup(name="Reference route (student test only)", show=True)
    prediction = folium.FeatureGroup(name="Predicted route", show=True)
    relevant = np.unique(candidates[::max(1, len(candidates) // 350)].ravel())
    for arc in relevant:
        n1, n2 = trip.arcs[arc]
        folium.PolyLine([trip.nodes[n1, ::-1].tolist(), trip.nodes[n2, ::-1].tolist()], color="#9baabe", weight=1, opacity=.45).add_to(roads)
    for lon, lat, _ in trip.track[::max(1, len(trip.track) // 700)]:
        folium.CircleMarker([lat, lon], radius=1, color="#267dd1", fill=True, fill_opacity=.5, opacity=.5).add_to(gps)
    for arc in dict.fromkeys(trip.route.tolist()):
        n1, n2 = trip.arcs[arc]
        folium.PolyLine([trip.nodes[n1, ::-1].tolist(), trip.nodes[n2, ::-1].tolist()], color="#e58b33", weight=5, opacity=.8).add_to(reference)
    for arc in dict.fromkeys(routed.tolist()):
        n1, n2 = trip.arcs[arc]
        folium.PolyLine([trip.nodes[n1, ::-1].tolist(), trip.nodes[n2, ::-1].tolist()], color="#11976b", weight=3, opacity=.85).add_to(prediction)
    for layer in (roads, reference, prediction, gps):
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    bounds = [[float(trip.track[:, 1].min()), float(trip.track[:, 0].min())],
              [float(trip.track[:, 1].max()), float(trip.track[:, 0].max())]]
    m.fit_bounds(bounds)
    m.save(output / f"folium_trip_{trip_id}.html")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=None, help="Student folder; needed only for the Folium map")
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    p.add_argument("--trip", default="00000052")
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    results = json.loads((args.output / "experiment_results.json").read_text())
    comparison_plots(results, args.output)
    print("Saved model_comparison.png, route_continuity_comparison.png, log_loss_comparison.png, confusion_matrices.png")
    if args.data is None:
        print("Skipped Folium map (pass --data to generate it).")
    else:
        folium_map(args.data, args.output, args.trip)
        print(f"Saved Folium map for student-test trip {args.trip}")


if __name__ == "__main__":
    main()
