"""Build a short first-evaluation code walkthrough from recorded PathSnap results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
RESULTS = json.loads((ART / "experiment_results.json").read_text())
INVENTORY = json.loads((ART / "dataset_inventory.json").read_text())
FULL = json.loads((ART / "full_student_test_results.json").read_text())
with (ART / "feature_sample.csv").open(newline="", encoding="utf-8") as stream:
    sample = list(csv.DictReader(stream))[:2]

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                          textColor=colors.HexColor("#15324f"), spaceAfter=11))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontSize=13, leading=16,
                          textColor=colors.HexColor("#15324f"), spaceBefore=9, spaceAfter=5))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontSize=10, leading=13,
                          textColor=colors.HexColor("#176d9a"), spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=8.5, leading=12.5, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=7.6, leading=10.5, spaceAfter=4))
styles.add(ParagraphStyle(name="CodeX", parent=styles["BodyText"], fontName="Courier", fontSize=7.2, leading=10,
                          backColor=colors.HexColor("#eff4f8"), borderPadding=5, spaceAfter=7))
styles.add(ParagraphStyle(name="CaptionX", parent=styles["BodyText"], fontSize=7.5, leading=10,
                          alignment=TA_CENTER, textColor=colors.HexColor("#586c80"), spaceAfter=6))


def p(text, style="BodyX"):
    return Paragraph(text, styles[style])


def h(text): return p(text, "H1X")
def hh(text): return p(text, "H2X")
def pct(value): return f"{100 * value:.1f}%"


def table(rows, widths):
    obj = Table([[p(str(value), "SmallX") for value in row] for row in rows], colWidths=widths, repeatRows=1)
    obj.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3edf7")),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#96adc3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f9fc")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return obj


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#687e92"))
    canvas.drawString(1.6 * cm, 1.2 * cm, "PathSnap first evaluation | recorded student-test evidence")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.2 * cm, str(doc.page))


story = [p("PathSnap: First Evaluation Guide", "TitleX"),
         p("Code walkthrough and speaking notes | 18 November 2026 | Repository: github.com/ThouSyntaxTerror/PathSnap", "SmallX")]

story += [h("The 30-second explanation"),
          p("GPS dots can drift away from the road a vehicle actually used. PathSnap reads each trip's GPS track and road graph, generates six nearby road-segment candidates for every GPS point, scores those candidates with a trained classifier, and uses Viterbi decoding to choose a more coherent sequence. The model and reference routes stay on Laptop 1; Laptop 2 displays predictions through an API.")]
story += [h("What to show for the three required components")]
story += [table([
    ["Component", "Code to open", "Evidence to show"],
    ["Data processing", "pathsnap.py: Trip.load, RoadIndex; inventory.py", "dataset_inventory.json and EDA plot"],
    ["Feature engineering", "pathsnap.py: candidate_ids, features, make_rows; feature_example.py", "feature_sample.csv: six candidate rows per GPS point"],
    ["Model training", "train.py: models, collect, grouped_cv, score_model", "experiment_results.json, comparison plot, five saved models"],
], [3.0 * cm, 7.0 * cm, 6.4 * cm])]

totals = INVENTORY["totals"]
story += [h("Dataset facts from the actual files"),
          p(f"The supplied folder has 80 training trips and 10 student-test trips, with {totals['gps_points']:,} GPS observations in total. Each trip has .track (longitude, latitude, time), .nodes (road coordinates), .arcs (from-node ID, to-node ID), and .route (reference arc IDs). We found {totals['invalid_gps_rows']} invalid GPS rows and {totals['nonincreasing_time_steps']} nonincreasing timestamps. {pct(totals['one_second_steps']/totals['time_steps'])} of time steps were one second. {totals['trips_with_multiple_components']} road graphs have multiple undirected components. Reserved evaluator trips were not supplied or used.")]
story += [Image(str(ART / "eda_summary.png"), width=16.7 * cm, height=5.0 * cm),
          p("Figure 1. Real trip-size, sampling, and geographic exploration from the student folder.", "CaptionX")]

story += [PageBreak(), h("How the feature extraction code works")]
story += [table([
    ["Step", "Function", "What it does"],
    ["1", "Trip.load", "Reads the four whitespace-separated files; validates node and arc IDs."],
    ["2", "RoadIndex", "Projects longitude/latitude locally into approximate metres for spatial calculations."],
    ["3", "candidate_ids", "Uses a KD-tree on arc endpoints/midpoints, then exact point-to-segment distance, to retain six road candidates."],
    ["4", "features", "Builds six numeric clues for every GPS point + candidate arc pair."],
    ["5", "make_rows", "Flattens candidate rows for ML and constructs proxy training labels from reference-route arcs."],
], [1.0 * cm, 3.6 * cm, 12.1 * cm])]
story += [hh("The six features")]
story += [table([
    ["Feature", "Why it helps"],
    ["Point-to-road distance", "A plausible road is usually near the GPS observation, allowing for noise."],
    ["Direction cosine", "The road's orientation should agree with movement between GPS observations."],
    ["GPS speed", "Helps distinguish plausible movement from a large GPS jump."],
    ["Road-segment length", "Provides context for short junction arcs versus long road arcs."],
    ["Time gap", "A large gap allows more travel than a one-second step."],
    ["Movement distance", "Measures how far the GPS position changed since the preceding observation."],
], [5.0 * cm, 11.7 * cm])]
story += [p("The model input applies log1p to nonnegative distance, speed, segment length, time gap, and movement. Direction cosine is unchanged. This reduces the influence of very large values while retaining their order.")]
story += [hh("Show real feature rows")]
story += [table([["GPS row", "Candidate arc", "Distance (m)", "Direction", "Proxy label"]] +
                [[r["gps_row"], r["candidate_arc_id"], r["distance_m"], r["direction_cos"], r["proxy_positive_label"]] for r in sample],
                [2.5 * cm, 3.3 * cm, 3.5 * cm, 3.2 * cm, 3.2 * cm])]
story += [p("Both rows refer to the same first GPS point of training trip 00000001. The label is derived by finding the nearest arc <b>within the supplied reference route</b>; the dataset does not provide a direct label for every GPS dot. At the first point, direction and speed are zero because there is no previous observation.", "SmallX")]
story += [p("PowerShell demonstration:", "SmallX"),
          p("Import-Csv artifacts/feature_sample.csv | Select-Object -First 6 | Format-Table gps_row,candidate_arc_id,distance_m,direction_cos,proxy_positive_label", "CodeX")]

story += [PageBreak(), h("Five-model experiment and measured results")]
story += [p("train.py trains Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and Histogram Gradient Boosting. It uses all 80 training trips, sampling at most 120 GPS points per trip: 57,600 candidate rows. Three-fold GroupKFold keeps trips intact. The ten student-test trips support development comparison, using at most 250 GPS points each for this table.")]
rows = [["Model", "Acc.", "Prec.", "Recall", "Macro F1", "Top-1", "After V", "Log loss"]]
for name, v in RESULTS["models"].items():
    rows.append([name.replace("_", " "), pct(v["candidate_accuracy"]), pct(v["precision"]),
                 pct(v["recall"]), f"{v['candidate_macro_f1']:.3f}", pct(v["mean_trip_top1_accuracy"]),
                 pct(v["mean_trip_accuracy_viterbi"]), f"{v['log_loss']:.3f}"])
story += [table(rows, [3.3 * cm, 1.65 * cm, 1.65 * cm, 1.65 * cm, 2.1 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm])]
story += [p("Acc./precision/recall/F1/log loss are binary candidate-classification metrics. Top-1 chooses the highest-scoring road candidate for each GPS observation. After V is proxy pointwise accuracy after Viterbi. Because candidate negatives are numerous, candidate accuracy alone is not a sufficient selection metric.", "SmallX")]
story += [Image(str(ART / "model_comparison.png"), width=16.7 * cm, height=8.3 * cm),
          p("Figure 2. Grouped CV macro F1, student-test candidate F1, pointwise Top-1, and after-Viterbi accuracy.", "CaptionX")]
story += [hh("Why Histogram Gradient Boosting was selected")]
hgb = RESULTS["models"]["hist_gradient_boosting"]
gnb = RESULTS["models"]["gaussian_nb"]
rf = RESULTS["models"]["random_forest"]
story += [p(f"The task depends on nonlinear interactions: a nearby segment may be wrong if its direction and neighboring road connections disagree. Tree boosting learns such combinations without manually listing every rule. Its probabilities feed the Viterbi emission score, and it had the lowest student-test log loss ({hgb['log_loss']:.3f}) among the five initial models. Its train/CV macro F1 values ({hgb['train_macro_f1']:.3f}/{hgb['cv_macro_f1_mean']:.3f}) were close, while a single tree and random forest had larger gaps. Crucially, Histogram Gradient Boosting gave the highest sampled <b>after-Viterbi</b> result ({pct(hgb['mean_trip_accuracy_viterbi'])}), versus {pct(gnb['mean_trip_accuracy_viterbi'])} for Gaussian Naive Bayes. Gaussian Naive Bayes did have a better classifier-only Top-1 result ({pct(gnb['mean_trip_top1_accuracy'])} versus {pct(hgb['mean_trip_top1_accuracy'])}); we selected for the final route task, not one isolated metric.")]
story += [p(f"The saved tuned model is {round((ART / 'final_model.joblib').stat().st_size / 1024)} KB, compared with {round((ART / 'random_forest.joblib').stat().st_size / (1024*1024), 1)} MB for the saved random forest. This makes model loading and deployment lightweight. Do not claim it trained fastest in our latest run: measured times vary and are in experiment_results.json.", "SmallX")]

story += [PageBreak(), h("Tuning, route decoding, and live explanation")]
selected = RESULTS["selection"]
tuned = selected["tuned_student_test"]
story += [hh("Bayesian optimisation in train.py"),
          p(f"After comparing the five initial models, Optuna TPE ran {len(selected['trials'])} Bayesian trials using three-fold grouped CV on the 80 training trips. It searched max_iter (40-120), max_leaf_nodes (10-40), and learning_rate (0.03-0.2), optimising candidate macro F1. The selected configuration was max_iter={selected['best_parameters']['max_iter']}, max_leaf_nodes={selected['best_parameters']['max_leaf_nodes']}, learning_rate={selected['best_parameters']['learning_rate']:.4f}. Sampled after-Viterbi proxy accuracy changed from {pct(hgb['mean_trip_accuracy_viterbi'])} to {pct(tuned['mean_trip_accuracy_viterbi'])}. The gain is modest.")]
story += [hh("HMM/Viterbi in pathsnap.py"),
          p("Each GPS point has six candidate arc states. Emission scores come from the classifier's probabilities. Transition penalties favour staying on the same arc, then moving to an arc whose start node matches the previous arc's end node. GPS gaps can permit a penalised fallback. Viterbi chooses the best sequence across the trip; bridge_route then tries to connect gaps with directed graph paths. The full ten-trip rehearsal still had 163 unresolved connections, so do not claim perfect continuity.")]
story += [hh("What to demonstrate in front of the teacher")]
story += [table([
    ["Show", "Say"],
    ["dataset_inventory.json + EDA plot", "We inspected all 90 supplied trips and checked GPS validity, timestamps, road-graph connectivity, and sizes."],
    ["pathsnap.py + feature_sample.csv", "Each GPS point becomes six candidate rows; these six measured features describe how plausible each road is."],
    ["train.py + experiment_results.json", "We trained five distinct models with trip-grouped CV and compared real student-test metrics."],
    ["Comparison chart + this guide", "We chose Histogram Gradient Boosting for route-level performance, probability quality, stability, and a compact artifact."],
    ["Laptop 1 API console + Laptop 2 UI", "Selecting a trip sends its ID to the backend; it performs live inference and logs the request."],
], [5.1 * cm, 11.6 * cm])]
story += [hh("Be explicit about these limits"),
          p("Pointwise labels are a nearest-reference-arc proxy, not verified GPS-to-road truth. The candidate generator can miss the reference arc (1.96% on the sampled student test). The reserved evaluator trips were not used during development. The final two-laptop demo and its Laptop 1 console screenshot must be added to the submission evidence. Do not invent an evaluator score or claim the route is always connected.")]
story += [p("Source files and full logs: github.com/ThouSyntaxTerror/PathSnap | Technical report: artifacts/PathSnap_Technical_Report_DRAFT.pdf", "SmallX")]

output = ART / "PathSnap_First_Evaluation_Code_Guide.pdf"
doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.5 * cm, bottomMargin=1.8 * cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Saved", output)
