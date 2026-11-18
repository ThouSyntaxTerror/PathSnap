"""Laptop 1: model, road graph, private reference routes, and evaluation API."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import Trip, make_rows, decode, continuity, read_array, bridge_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", "data")).resolve()
EVAL_ROOT = Path(os.environ.get("PATHSNAP_EVAL_ROOT", str(DATA_ROOT / "student_test"))).resolve()
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", "artifacts/final_model.joblib")).resolve()
MODE = "evaluator" if "PATHSNAP_EVAL_ROOT" in os.environ else "student-test rehearsal"
model = None
app = FastAPI(title="PathSnap evaluation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    global model
    if not MODEL_PATH.exists():
        log.error("Model missing: %s. Run train.py first.", MODEL_PATH)
    else:
        model = joblib.load(MODEL_PATH)
        log.info("Loaded model %s; mode=%s; trip folder=%s", MODEL_PATH, MODE, EVAL_ROOT)


def trip_folders():
    if not EVAL_ROOT.exists(): return []
    return sorted(p for p in EVAL_ROOT.iterdir() if p.is_dir() and p.name.isdigit() and (p / f"{p.name}.track").exists())


@app.get("/health")
def health():
    return {"ok": model is not None, "mode": MODE, "trip_count": len(trip_folders())}


@app.get("/trips")
def trips():
    return {"mode": MODE, "trip_ids": [p.name for p in trip_folders()]}


class Request(BaseModel):
    trip_id: str


@app.post("/predict")
def predict(request: Request):
    if model is None: raise HTTPException(503, "Model not loaded; run training first")
    choices = {p.name: p for p in trip_folders()}
    if request.trip_id not in choices: raise HTTPException(404, "Unknown trip ID")
    log.info("INFERENCE REQUEST trip=%s mode=%s", request.trip_id, MODE)
    # Inference reads no reference route. Open it only after predictions are fixed.
    trip = Trip.load(choices[request.trip_id], include_route=False)
    road, ids, candidates, x, _ = make_rows(trip)
    probability = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(candidates.shape)
    raw_arcs = candidates[np.arange(len(ids)), np.argmax(probability, axis=1)]
    decoded = decode(trip, road, ids, candidates, probability)
    routed, bridge_count, unresolved = bridge_route(trip, road, decoded)
    metrics = {
        "gps_points": int(len(ids)),
        "continuity_classifier": continuity(trip, raw_arcs),
        "continuity_viterbi": continuity(trip, decoded),
        "bridge_arcs_inserted": int(bridge_count),
        "unresolved_connections": int(unresolved),
    }
    route_path = choices[request.trip_id] / f"{request.trip_id}.route"
    if route_path.exists():
        trip.route = read_array(route_path, 1, int).ravel()
        # .route is an arc sequence, not a pointwise label. This proxy is labelled honestly.
        reference_per_point = road.reference_arc_per_point(road.track_xy)
        metrics["approx_pointwise_accuracy_classifier"] = float(np.mean(raw_arcs == reference_per_point))
        metrics["approx_pointwise_accuracy_viterbi"] = float(np.mean(decoded == reference_per_point))
        ref_set, pred_set = set(trip.route.tolist()), set(routed.tolist())
        metrics["reference_arc_recall"] = len(ref_set & pred_set) / max(len(ref_set), 1)
    # Send geometries only. Reference route IDs and geometry never leave the server.
    sample_idx = np.unique(np.linspace(0, len(ids) - 1, min(len(ids), 1200)).astype(int))
    gps = trip.track[sample_idx, :2].tolist()
    selected = list(dict.fromkeys(routed.tolist()))
    if len(selected) > 2000:
        selected = [selected[i] for i in np.linspace(0, len(selected) - 1, 2000).astype(int)]
    route_lines = [[trip.nodes[trip.arcs[a, 0]].tolist(), trip.nodes[trip.arcs[a, 1]].tolist()] for a in selected]
    road_sample = np.unique(candidates[sample_idx].ravel())
    if len(road_sample) > 2500: road_sample = road_sample[:2500]
    road_lines = [[trip.nodes[trip.arcs[a, 0]].tolist(), trip.nodes[trip.arcs[a, 1]].tolist()] for a in road_sample]
    log.info("INFERENCE DONE trip=%s gps=%d continuity=%.3f approx_accuracy=%s", request.trip_id, len(ids),
             metrics["continuity_viterbi"], metrics.get("approx_pointwise_accuracy_viterbi"))
    return {"trip_id": request.trip_id, "mode": MODE, "metrics": metrics, "gps": gps,
            "road_lines": road_lines, "predicted_lines": route_lines,
            "metric_note": "Pointwise accuracy uses the nearest reference-route arc as a proxy; the source does not provide a GPS-to-arc label for every point."}
