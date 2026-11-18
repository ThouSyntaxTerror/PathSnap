"""Create real experiment plots and an interactive Folium comparison map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np

from pathsnap import Trip, make_rows, decode, bridge_route


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    p.add_argument("--trip", default="00000052")
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    results = json.loads((args.output / "experiment_results.json").read_text())
    names = list(results["models"])
    labels = [s.replace("_", " ").title() for s in names]
    metrics = ["cv_macro_f1_mean", "candidate_macro_f1", "mean_trip_top1_accuracy", "mean_trip_accuracy_viterbi"]
    titles = ["Grouped CV macro F1", "Student-test macro F1", "Student-test Top-1", "Student-test after Viterbi"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, key, title in zip(axes.flat, metrics, titles):
        values = [results["models"][n][key] for n in names]
        bars = ax.barh(labels, values, color=["#147d64" if n == results["selection"]["family"] else "#5388ba" for n in names])
        ax.set_xlim(0, 1); ax.set_title(title); ax.invert_yaxis()
        for bar, val in zip(bars, values): ax.text(val + .01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    fig.savefig(args.output / "model_comparison.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 2.6), layout="constrained")
    losses = [results["models"][n]["log_loss"] for n in names]
    bars = ax.barh(labels, losses, color="#6385ad")
    ax.invert_yaxis(); ax.set_title("Student-test candidate log loss (lower is better)")
    ax.set_xlim(0, max(losses) * 1.25)
    for bar, val in zip(bars, losses): ax.text(val + .008, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    fig.savefig(args.output / "log_loss_comparison.png", dpi=180)
    plt.close(fig)

    folder = args.data / "student_test" / args.trip
    trip = Trip.load(folder, include_route=True)
    road, ids, candidates, x, _ = make_rows(trip)
    model = joblib.load(args.output / "final_model.joblib")
    scores = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(candidates.shape)
    predicted = decode(trip, road, ids, candidates, scores)
    routed, _, _ = bridge_route(trip, road, predicted)
    center = [float(np.mean(trip.track[:, 1])), float(np.mean(trip.track[:, 0]))]
    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)
    roads = folium.FeatureGroup(name="Nearby road network", show=True)
    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    reference = folium.FeatureGroup(name="Reference route (student test only)", show=True)
    prediction = folium.FeatureGroup(name="Predicted route", show=True)
    relevant = np.unique(candidates[::max(1, len(candidates)//350)].ravel())
    for arc in relevant:
        n1, n2 = trip.arcs[arc]
        folium.PolyLine([trip.nodes[n1, ::-1].tolist(), trip.nodes[n2, ::-1].tolist()], color="#9baabe", weight=1, opacity=.45).add_to(roads)
    for lon, lat, _ in trip.track[::max(1, len(trip.track)//700)]:
        folium.CircleMarker([lat, lon], radius=1, color="#267dd1", fill=True, fill_opacity=.5, opacity=.5).add_to(gps)
    for arc in dict.fromkeys(trip.route.tolist()):
        n1, n2 = trip.arcs[arc]
        folium.PolyLine([trip.nodes[n1, ::-1].tolist(), trip.nodes[n2, ::-1].tolist()], color="#e58b33", weight=5, opacity=.8).add_to(reference)
    for arc in dict.fromkeys(routed.tolist()):
        n1, n2 = trip.arcs[arc]
        folium.PolyLine([trip.nodes[n1, ::-1].tolist(), trip.nodes[n2, ::-1].tolist()], color="#11976b", weight=3, opacity=.85).add_to(prediction)
    for layer in (roads, reference, prediction, gps): layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    bounds = [[float(trip.track[:, 1].min()), float(trip.track[:, 0].min())],
              [float(trip.track[:, 1].max()), float(trip.track[:, 0].max())]]
    m.fit_bounds(bounds)
    m.save(args.output / f"folium_trip_{args.trip}.html")
    print(f"Saved model comparison and Folium map for student-test trip {args.trip}")


if __name__ == "__main__": main()
