"""Generate a reproducible technical report draft from measured artifacts."""
from __future__ import annotations

import json
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

OUT = Path("artifacts")
RESULTS = json.loads((OUT / "experiment_results.json").read_text())
FULL = json.loads((OUT / "full_student_test_results.json").read_text(encoding="utf-8-sig"))
INVENTORY = json.loads((OUT / "dataset_inventory.json").read_text())
S = getSampleStyleSheet()
S.add(ParagraphStyle(name="TitleX", parent=S["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#173453"), spaceAfter=12))
S.add(ParagraphStyle(name="H1X", parent=S["Heading1"], fontSize=13, leading=16, textColor=colors.HexColor("#173453"), spaceBefore=10, spaceAfter=6))
S.add(ParagraphStyle(name="H2X", parent=S["Heading2"], fontSize=10, leading=13, textColor=colors.HexColor("#176c96"), spaceBefore=7, spaceAfter=4))
S.add(ParagraphStyle(name="BodyX", parent=S["BodyText"], fontSize=8.5, leading=12.5, spaceAfter=6))
S.add(ParagraphStyle(name="SmallX", parent=S["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#42566e"), spaceAfter=5))
S.add(ParagraphStyle(name="CaptionX", parent=S["BodyText"], fontSize=7.5, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#42566e"), spaceAfter=8))


def p(text, style="BodyX"): return Paragraph(text, S[style])
def h(text): return p(text, "H1X")
def hh(text): return p(text, "H2X")
def table(rows, widths=None):
    t = Table([[p(str(c), "SmallX") for c in row] for row in rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eef7")),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#9db3ca")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fc")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def percent(x): return f"{100*x:.1f}%"
def footer(canvas, doc):
    canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#6b7f93"))
    canvas.drawString(1.6*cm, 1.2*cm, "PathSnap | measured student-test results | working draft")
    canvas.drawRightString(A4[0]-1.6*cm, 1.2*cm, str(doc.page))


story = []
story += [p("PathSnap: Technical Report", "TitleX"), p("Working draft | 18 November 2026 | Generated from this repository's actual experiment logs", "SmallX")]
story += [h("1. Problem and approach"), p("The task is to infer road segments traversed by a vehicle from noisy, timestamped GPS observations. Each trip provides a track and its own road graph. We generate nearby candidate arcs for each observation, classify candidate plausibility, and use Viterbi decoding to favour a coherent sequence. The complete system serves predictions through a FastAPI backend to a separate browser UI.")]
story += [h("2. Dataset understanding and EDA"), p("The supplied student folder has 80 training trips and 10 student-test trips. Each trip contains .track (longitude, latitude, timestamp), .nodes (longitude, latitude), .arcs (from-node ID, to-node ID), and .route (chronological reference arc IDs). The reserved evaluator trips are not in the student folder. The reference route has fewer rows than the track, so GPS rows cannot be matched to route rows by index.")]
story += [table([
    ["Partition", "Trips", "GPS observations", "Road nodes", "Road arcs", "Reference arc entries"],
    ["Training", "80", "208,417", "1,933,485", "3,876,904", "31,336"],
    ["Student test", "10", "19,160", "90,689", "185,041", "3,477"],
], [2.7*cm, 1.2*cm, 3.0*cm, 2.6*cm, 2.4*cm, 3.5*cm])]
total = INVENTORY["totals"]
story += [Spacer(1, 7), p(f"All {total['trip_count']} trip folders contained the four expected file types. There were {total['invalid_gps_rows']} nonfinite/out-of-range GPS rows and {total['nonincreasing_time_steps']} nonincreasing timestamps. {percent(total['one_second_steps']/total['time_steps'])} of GPS steps were one second apart. The largest observed time gap was {total['max_gap_seconds']:,.0f} seconds. {total['trips_with_multiple_components']} road graphs had multiple undirected components, which can make transitions impossible. These observations come from dataset_inventory.json.")]
story += [hh("Preprocessing and label construction"), p("Coordinates are projected locally to metres with an equirectangular approximation centred on each trip. For each GPS observation, a KD-tree proposes nearby road arcs using their endpoints and midpoints; exact point-to-segment distance ranks the final six candidates. Features are log distance, direction cosine, speed, segment length, time gap, and movement length. During training and evaluation only, each GPS point is assigned the nearest arc among that trip's reference-route arcs. These are <b>heuristic proxy labels</b>, not independently verified pointwise ground truth. The model does not receive reference-route data at inference time.")]
story += [Image(str(OUT / "eda_summary.png"), width=17.0*cm, height=4.8*cm), p("Figure 1. Trip-size distribution, time sampling, and student-test trip 00000052 with GPS points and nearby roads.", "CaptionX")]
with (OUT / "feature_sample.csv").open(newline="", encoding="utf-8") as feature_file:
    feature_reader = csv.DictReader(feature_file)
    example_rows = [next(feature_reader) for _ in range(2)]
story += [hh("Actual feature rows (training trip 00000001, first GPS point)"),
          p("Each GPS observation creates six candidate-road rows. These two rows show their physical values; the model uses log1p for distance, speed, length, time gap, and movement. The proxy label is constructed only from a training reference route.", "SmallX"),
          table([["Candidate arc", "Distance to GPS (m)", "Direction cosine", "Proxy label"]] +
                [[r["candidate_arc_id"], r["distance_m"], r["direction_cos"], r["proxy_positive_label"]] for r in example_rows],
                [3.1*cm, 4.4*cm, 4.2*cm, 3.0*cm])]
story += [PageBreak(), h("3. Five-model experiment")]
story += [p("All 80 training trips were used, with up to 120 evenly sampled GPS observations per trip (57,600 candidate rows). Three-fold GroupKFold kept trips intact across folds. The ten student-test trips were evaluated with up to 250 observations each. Values below are means across trips where stated; every number comes from experiment_results.json.")]
rows = [["Model", "CV macro F1", "Test Top-1", "After Viterbi", "Log loss", "Continuity after"]]
for name, v in RESULTS["models"].items():
    rows.append([name.replace("_", " "), f"{v['cv_macro_f1_mean']:.3f}", f"{v['mean_trip_top1_accuracy']:.3f}",
                 f"{v['mean_trip_accuracy_viterbi']:.3f}", f"{v['log_loss']:.3f}", f"{v['mean_continuity_viterbi']:.3f}"])
story += [table(rows, [4.0*cm, 2.3*cm, 2.2*cm, 2.5*cm, 2.1*cm, 3.3*cm]), Spacer(1, 9)]
story += [Image(str(OUT / "model_comparison.png"), width=17.0*cm, height=8.5*cm), p("Figure 2. Grouped cross-validation and student-test comparisons using measured results.", "CaptionX")]
story += [Image(str(OUT / "log_loss_comparison.png"), width=17.0*cm, height=3.0*cm), p("Figure 3. Candidate log loss on the student-test observations; lower is better.", "CaptionX")]
story += [p("The mean candidate miss rate on sampled student-test observations was 1.96%. Gaussian Naive Bayes had the best independent Top-1 score (0.762), but Histogram Gradient Boosting had the best sampled route-level accuracy after Viterbi (0.765). We therefore selected Histogram Gradient Boosting for the complete pipeline. The margin is small; the result is sensitive to proxy labels and sampling.")]
story += [PageBreak(), h("3.1 Individual model notes")]
notes = {
    "logistic_regression": "A fast linear baseline with interpretable feature effects. It cannot capture all interactions between distance, direction, and speed without explicit feature crosses.",
    "gaussian_nb": "A very fast probabilistic baseline. Its conditional-independence assumption is questionable because movement length, speed, and time gap are related.",
    "decision_tree": "Captures nonlinear threshold rules and is easy to inspect, but a single tree can overfit noisy proxy labels.",
    "random_forest": "Averages many trees for stability and nonlinear interactions; model size and inference cost are higher than the simple baselines.",
    "hist_gradient_boosting": "Learns nonlinear interactions efficiently from many candidate rows and emits candidate probabilities. It gave the highest route-level score after decoding on this sampled comparison.",
}
setups = {
    "logistic_regression": ("class_weight=balanced; max_iter=400", "log loss"),
    "gaussian_nb": ("var_smoothing=1e-9 (initial default)", "Gaussian likelihood estimation"),
    "decision_tree": ("max_depth=12; min_samples_leaf=15", "Gini impurity"),
    "random_forest": ("n_estimators=60; max_depth=14; min_samples_leaf=8", "Gini impurity across trees"),
    "hist_gradient_boosting": ("max_iter=80; max_leaf_nodes=25; l2_regularization=0.1", "log loss"),
}
for name, description in notes.items():
    v = RESULTS["models"][name]
    gap = v["train_macro_f1"] - v["cv_macro_f1_mean"]
    fit_note = "This train-CV gap suggests some overfitting." if gap > .03 else "The small train-CV gap gives little evidence of overfitting."
    matrix = v["confusion_matrix_tn_fp_fn_tp"]
    settings, objective = setups[name]
    story += [hh(name.replace("_", " ").title()), p(description + f" Initial configuration: {settings}. Training objective: {objective}. Train/CV/student-test candidate macro F1: {v['train_macro_f1']:.3f}/{v['cv_macro_f1_mean']:.3f}/{v['candidate_macro_f1']:.3f}; {fit_note} Student-test candidate accuracy {v['candidate_accuracy']:.3f}, precision {v['precision']:.3f}, recall {v['recall']:.3f}, log loss {v['log_loss']:.3f}; Top-1 {v['mean_trip_top1_accuracy']:.3f}, after-Viterbi {v['mean_trip_accuracy_viterbi']:.3f}. Confusion matrix [[TN, FP], [FN, TP]]: {matrix}. Training including CV: {v['training_seconds_including_cv']:.1f}s; student-test inference: {v['inference_seconds_total']:.3f}s.")]
story += [h("4. Bayesian hyperparameter optimisation")]
sel = RESULTS["selection"]
baseline = RESULTS["models"][sel["family"]]
tuned = sel["tuned_student_test"]
story += [p(f"Optuna's TPE sampler ran {len(sel['trials'])} trials with three-fold grouped training CV, optimising candidate-level macro F1. Search parameters were max_iter (40-120), max_leaf_nodes (10-40), and learning_rate (0.03-0.2, log scale). Best parameters: {sel['best_parameters']}. Untuned versus tuned student-test after-Viterbi accuracy: {baseline['mean_trip_accuracy_viterbi']:.4f} versus {tuned['mean_trip_accuracy_viterbi']:.4f}. All trials and values are saved in experiment_results.json. The tuning gain is modest.")]
story += [PageBreak(), h("5. HMM/Viterbi route reconstruction")]
story += [p("Hidden states are the six candidate arc IDs at each observation. Emission scores are the selected classifier's candidate probabilities. Transition penalties favour the same arc, then directly connected arcs (previous arc's end node equals current arc's start node). For disconnected candidates, a spatial endpoint-distance fallback allows recovery from missing candidates and GPS gaps; its penalty is scaled by elapsed time. Viterbi dynamic programming selects the highest-scoring sequence. After decoding, the API attempts directed shortest-path bridges and rejects implausibly long ones, reporting unresolved connections. This is a connectivity-aware approximation, not a full graph-shortest-path decoder.")]
story += [p(f"On the sampled student-test comparison, the untuned selected model's mean direct-adjacency continuity changed from {percent(baseline['mean_continuity_classifier'])} before decoding to {percent(baseline['mean_continuity_viterbi'])} after decoding. Its mean proxy pointwise accuracy changed from {percent(baseline['mean_trip_top1_accuracy'])} to {percent(baseline['mean_trip_accuracy_viterbi'])}. The tuned model's corresponding values were {percent(tuned['mean_continuity_classifier'])} to {percent(tuned['mean_continuity_viterbi'])}, and {percent(tuned['mean_trip_top1_accuracy'])} to {percent(tuned['mean_trip_accuracy_viterbi'])}.")]
story += [hh("Full student-test rehearsal"), p(f"The final API was run against all {len(FULL)} student-test trips using every GPS observation ({sum(r['gps_points'] for r in FULL):,} total). Mean per-trip proxy accuracy changed from {percent(sum(r['approx_accuracy_before'] for r in FULL)/len(FULL))} to {percent(sum(r['approx_accuracy_after'] for r in FULL)/len(FULL))}; mean direct-adjacency continuity changed from {percent(sum(r['continuity_before'] for r in FULL)/len(FULL))} to {percent(sum(r['continuity_after'] for r in FULL)/len(FULL))}. The post-decoding road-graph step inserted {sum(r['bridge_arcs_inserted'] for r in FULL):,} connecting arcs and left {sum(r['unresolved_connections'] for r in FULL):,} transitions unresolved across the ten trips. Per-trip values are in full_student_test_results.json. These rehearsal results are not evaluator-test scores.")]
story += [h("6. Folium geographical visualisation")]
story += [Image(str(OUT / "folium_trip_00000052.png"), width=16.7*cm, height=10.5*cm), p("Figure 4. Actual Folium map of student-test trip 00000052. Blue: GPS; grey: nearby roads; orange: reference route; green: prediction. Interactive HTML with layer controls is included.", "CaptionX")]
story += [PageBreak(), h("7. API, UI, and two-laptop deployment")]
story += [p("Laptop 1 runs FastAPI, loads final_model.joblib, retains all road and reference-route files, and logs each inference request and result. Laptop 2 hosts a browser UI. The evaluator selects a trip ID; the UI sends only that ID to POST /predict. Laptop 1 computes the prediction and metrics, returning predicted geometry and summary numbers. No reference-route IDs or geometry are sent to the UI. The ten provided student-test trips are the current rehearsal set; set PATHSNAP_EVAL_ROOT to the reserved evaluator-trip folder for final evaluation.")]
story += [Image(str(OUT / "ui_demo.png"), width=16.7*cm, height=12.5*cm), p("Figure 5. Working UI response for student-test rehearsal trip 00000052.", "CaptionX")]
story += [h("Limitations and remaining final-demo evidence")]
story += [p("The source does not supply pointwise GPS-to-arc ground truth, so pointwise scores use nearest-reference-arc proxy labels. Six candidates can miss a reference arc, and endpoint-based transitions plus post-decoding bridges do not guarantee a fully connected path. The final evaluator trips are unavailable in the student package. Capture the required Laptop 1 CMD/API-log screenshot during the live two-laptop demonstration and add it to the final submitted report; this draft does not invent that screenshot.")]

doc = SimpleDocTemplate(str(OUT / "PathSnap_Technical_Report_DRAFT.pdf"), pagesize=A4,
    leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.8*cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Saved", OUT / "PathSnap_Technical_Report_DRAFT.pdf")
