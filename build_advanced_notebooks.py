"""Build and execute the Bayesian tuning and Viterbi explanation notebooks."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).parent


def M(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def C(source: str):
    return nbf.v4.new_code_cell(source.strip())


def bayesian_notebook():
    cells = [
        M("""# Bayesian hyperparameter optimisation: chosen model

This notebook reproduces the **five Optuna TPE trials** for Histogram Gradient Boosting (HGB), then compares the untuned baseline and tuned model on exactly the same training trips, grouped CV folds, student-test trips, six candidate arcs per GPS point, and sampled observations. Saved outputs are from a real run. No reserved evaluator trips are used.

The target is **mean three-fold grouped training-CV macro F1**. Model family selection was also based on grouped training CV. Student-test metrics below are reporting-only, not the optimisation target."""),
        M("""## 1. Data and evaluation setup

Training: 80 trips, at most 120 GPS observations per trip; six candidate road arcs per observation. Student-test: 10 trips, at most 250 GPS observations per trip. The labels are **proxy pointwise labels** made by choosing the nearest arc on the provided reference route; the source does not contain a verified arc ID for each GPS observation. GroupKFold keeps whole trips together and prevents observations from the same trip appearing in both training and validation folds."""),
        C("""import os
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from IPython.display import display
from sklearn.ensemble import HistGradientBoostingClassifier
from train import collect, grouped_cv, score_model

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", "data"))
assert (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "student_test").is_dir(), DATA_ROOT
with redirect_stdout(StringIO()):
    X, y, groups, _ = collect(DATA_ROOT / "train", 120)
    _, _, _, test_trips = collect(DATA_ROOT / "student_test", 250)
print(f"Training rows: {len(y):,}; student-test trips: {len(test_trips)}; positive-label share: {y.mean():.2%}")
print("The same X, y, groups, and test_trips objects are used for both model versions.")"""),
        M("""## 2. Search design: questions 1–4

Optuna's **TPE (Tree-structured Parzen Estimator)** sampler is a Bayesian/sequential search method. It uses results from earlier trials to suggest later settings; it is **not a Gaussian-process search**.

| Hyperparameter | Meaning | Range/distribution |
|---|---|---|
| max_iter | Maximum boosting rounds / trees | Integer, 40–120 |
| max_leaf_nodes | Maximum leaf nodes in each tree; controls complexity | Integer, 10–40 |
| learning_rate | Size of each boosting update | Float, 0.03–0.20 on a log scale |

We ran **five trials**, seed 42. Each trial constructs HGB with those three values and random_state=42, then maximises mean macro F1 over three GroupKFold training folds. Other HGB settings use library defaults during the search, including l2_regularization=0.0. The untuned baseline uses max_iter=80, max_leaf_nodes=25, learning_rate=0.1, and l2_regularization=0.1. The final tuned model is constructed with the same fixed settings as the search candidate, so the search score matches the model configuration."""),
        C("""optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    candidate = HistGradientBoostingClassifier(
        max_iter=trial.suggest_int("max_iter", 40, 120),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 40),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
        random_state=42,
    )
    return float(np.mean(grouped_cv(candidate, X, y, groups, folds=3)))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=5)
trials = pd.DataFrame([{"trial": t.number + 1, "CV macro F1": t.value, **t.params} for t in study.trials])
display(trials.style.format({"CV macro F1": "{:.4f}", "learning_rate": "{:.4f}"}))
print("Best settings:", study.best_params)
print(f"Best search objective: {study.best_value:.4f}")"""),
        M("""## 3. Fair untuned-versus-tuned comparison: questions 5–6

Both versions use the **same training matrix, grouped folds, student-test rows, score_model function, and Viterbi decoder**. The untuned settings are the original fixed HGB baseline. The tuned settings are the best Optuna trial. Candidate macro F1 is the average of the negative- and positive-class F1 scores; Top-1 and after-Viterbi accuracy compare against the proxy reference arc per GPS observation. Lower log loss is better. Inference timing is noisy on a single short run, so it is supporting evidence rather than a decisive ranking."""),
        C("""baseline = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=25,
    learning_rate=0.1, l2_regularization=0.1, random_state=42)
tuned = HistGradientBoostingClassifier(**study.best_params, random_state=42)
records = {}
for name, model in [("Untuned HGB", baseline), ("Tuned HGB", tuned)]:
    cv = grouped_cv(model, X, y, groups, folds=3)
    model.fit(X, y)
    student = score_model(model, test_trips)
    records[name] = {
        "CV macro F1": float(np.mean(cv)),
        "Student Top-1 proxy accuracy": student["mean_trip_top1_accuracy"],
        "Student candidate macro F1": student["candidate_macro_f1"],
        "Student candidate log loss": student["log_loss"],
        "Inference seconds (sampled trips)": student["inference_seconds_total"],
        "Student after-Viterbi proxy accuracy": student["mean_trip_accuracy_viterbi"],
        "Route continuity before Viterbi": student["mean_continuity_classifier"],
        "Route continuity after Viterbi": student["mean_continuity_viterbi"],
        "Continuity change (percentage points)": 100*(student["mean_continuity_viterbi"]-student["mean_continuity_classifier"]),
    }
comparison = pd.DataFrame(records).T
comparison.loc["Tuned − untuned"] = comparison.loc["Tuned HGB"] - comparison.loc["Untuned HGB"]
display(comparison.style.format("{:.4f}"))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
comparison.loc[["Untuned HGB", "Tuned HGB"],
    ["CV macro F1", "Student Top-1 proxy accuracy", "Student candidate macro F1",
     "Student after-Viterbi proxy accuracy"]].T.plot.bar(ax=axes[0], ylim=(0, 1), rot=28)
axes[0].set_title("Higher is better"); axes[0].legend(fontsize=8)
comparison.loc[["Untuned HGB", "Tuned HGB"],
    ["Student candidate log loss", "Inference seconds (sampled trips)"]].T.plot.bar(ax=axes[1], rot=20)
axes[1].set_title("Lower is better; timing varies"); axes[1].legend(fontsize=8)
plt.show()"""),
        C("""delta = comparison.loc["Tuned − untuned"]
print(f"Grouped CV macro F1: {comparison.loc['Untuned HGB','CV macro F1']:.4f} → {comparison.loc['Tuned HGB','CV macro F1']:.4f} ({100*delta['CV macro F1']:+.2f} percentage points).")
print(f"Student Top-1 proxy accuracy: {100*comparison.loc['Untuned HGB','Student Top-1 proxy accuracy']:.2f}% → {100*comparison.loc['Tuned HGB','Student Top-1 proxy accuracy']:.2f}%.")
print(f"Student candidate macro F1: {comparison.loc['Untuned HGB','Student candidate macro F1']:.4f} → {comparison.loc['Tuned HGB','Student candidate macro F1']:.4f}.")
print(f"Student candidate log loss: {comparison.loc['Untuned HGB','Student candidate log loss']:.4f} → {comparison.loc['Tuned HGB','Student candidate log loss']:.4f} (lower is better).")
print(f"Student after-Viterbi proxy accuracy: {100*comparison.loc['Untuned HGB','Student after-Viterbi proxy accuracy']:.2f}% → {100*comparison.loc['Tuned HGB','Student after-Viterbi proxy accuracy']:.2f}%.")
print("Conclusion: tuning improved the CV objective and some route metrics, but it did not improve every metric. Do not claim universal improvement.")"""),
        M("""## Viva answers and limitations

1. **Why HGB?** It had the highest grouped training-CV macro F1 among the five initial families. Its nonlinear trees can combine distance, direction, movement, and road length clues. Student-test outcomes were not used to choose the family.
2. **Why Bayesian optimisation?** TPE uses previous trial scores to guide later suggestions. Five trials were a small, time-limited search; they do not prove a global optimum.
3. **What was optimised?** Mean three-fold GroupKFold macro F1 on training trips. The student-test set was evaluated afterward.
4. **What were the best values?** Read the executed trial table above. The selected trial used max_iter=97, max_leaf_nodes=10, and learning_rate≈0.1889 in the recorded run.
5. **Did accuracy improve?** Report each metric separately from the comparison table. CV macro F1 and student Top-1/after-Viterbi proxy accuracy improved slightly; candidate macro F1 and log loss did not. Inference time varies by run.
6. **What are hidden layers?** There are **no neural-network hidden layers** in Histogram Gradient Boosting. It is an ensemble of decision trees. The later route decoder has **hidden states** (candidate road arcs), which are a different concept.
7. **Why can test metrics disagree?** Optuna targets one metric on training CV. Candidate classification, probability calibration, and full route decoding measure different behavior. Five trials and proxy labels add uncertainty.
8. **What would improve the evidence?** More grouped-CV folds or repeated seeds, a larger training-only search, calibration checks, and independent verified pointwise labels. Do not tune on the student-test or reserved evaluator trips."""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def hmm_notebook():
    cells = [
        M("""# HMM-style Viterbi route reconstruction

This notebook explains the code in pathsnap.py and compares **classifier-only** arc choices with **classifier plus Viterbi** choices. It uses the saved final tuned HGB model and the same 10 student-test trips. Saved outputs come from a real run. The separate reserved evaluator set is not used.

Our decoder is **HMM-style**: it uses hidden road-arc states, model-based emission scores, heuristic transition penalties, and Viterbi dynamic programming. It does **not** fit a fully normalized HMM transition matrix. This wording matters in a viva."""),
        M("""## 1. Hidden states, observations, and emissions

At GPS time t, the hidden state is **one of six candidate directed road arcs** near that observation. It is called hidden because the GPS receiver tells us longitude, latitude, and time, but not the exact road arc. These are **hidden states, not neural-network hidden layers**. HGB and this decoder have no neural layers.

For each GPS point–candidate arc pair, HGB sees six engineered features: point-to-arc distance, movement-versus-road direction cosine, speed, road-segment length, time gap, and movement length. HGB predict_proba gives the candidate positive-class score. The decoder clips it to [1e-6, 1] and uses its log as the **emission score**. It is a classifier score, not a separately estimated generative GPS-noise distribution."""),
        C("""import os
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from IPython.display import display
from matplotlib.collections import LineCollection
from train import collect
from pathsnap import decode, route_continuity

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", "data"))
assert (DATA_ROOT / "student_test").is_dir(), DATA_ROOT
model = joblib.load("artifacts/final_model.joblib")
with redirect_stdout(StringIO()):
    _, _, _, trips = collect(DATA_ROOT / "student_test", 250)
print(type(model).__name__, "| student-test trips:", len(trips))
print("Candidate states per observation:", trips[0]["candidates"].shape[1])"""),
        M("""## 2. Transition scores and Viterbi recursion

For the previous arc i and current arc j, the code uses a **penalty**:

| Relationship | Penalty |
|---|---:|
| Same arc | 0 |
| Previous arc ends where next arc starts | 0.25 |
| Otherwise disconnected | 1.5 + endpoint_gap_metres / (25 × elapsed_seconds) |

The transition score is **negative penalty times transition_weight** (default 1.0). These are *unnormalized heuristic scores*, not a learned transition-probability matrix. Viterbi keeps, for each current candidate, the best preceding candidate and cumulative score: previous score + log emission − transition penalty. Backtracking from the best final state recovers one arc per GPS observation.

A larger GPS time gap reduces the distance part of the disconnected penalty, so the decoder can recover after missing samples. If the correct road is absent from all six candidates, decoding cannot select it at that observation. For disconnected decoded arcs, bridge_route later attempts a directed A* road-graph bridge; it rejects implausibly long bridges and reports unresolved links."""),
        C("""rows = []
decoded_by_trip = {}
for item in trips:
    trip, road, ids, candidates, features = (item[k] for k in
        ("trip", "road", "point_ids", "candidates", "features"))
    scores = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(candidates.shape)
    classifier = candidates[np.arange(len(ids)), np.argmax(scores, axis=1)]
    decoded = decode(trip, road, ids, candidates, scores)
    truth_proxy = road.reference_arc_per_point(road.track_xy[ids])
    before = float(np.mean(classifier == truth_proxy))
    after = float(np.mean(decoded == truth_proxy))
    cb = route_continuity(trip, classifier)
    ca = route_continuity(trip, decoded)
    rows.append({"trip": trip.trip_id, "GPS sample": len(ids),
        "Classifier-only proxy accuracy (%)": 100*before,
        "After-Viterbi proxy accuracy (%)": 100*after,
        "Accuracy change (pp)": 100*(after-before),
        "Continuity before (%)": 100*cb,
        "Continuity after (%)": 100*ca,
        "Continuity change (pp)": 100*(ca-cb),
        "Candidate miss (%)": 100*(1-item["labels"].any(axis=1).mean())})
    decoded_by_trip[trip.trip_id] = (item, classifier, decoded)
per_trip = pd.DataFrame(rows).set_index("trip")
display(per_trip.style.format("{:.2f}"))
mean_row = per_trip.mean(numeric_only=True)
print(f"Mean per-trip proxy accuracy: {mean_row['Classifier-only proxy accuracy (%)']:.2f}% → {mean_row['After-Viterbi proxy accuracy (%)']:.2f}%")
print(f"Mean per-trip route continuity: {mean_row['Continuity before (%)']:.2f}% → {mean_row['Continuity after (%)']:.2f}%")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
per_trip[["Classifier-only proxy accuracy (%)", "After-Viterbi proxy accuracy (%)"]].plot.bar(ax=axes[0], ylim=(0,100))
axes[0].set_title("Pointwise proxy accuracy by trip"); axes[0].set_ylabel("%")
per_trip[["Continuity before (%)", "Continuity after (%)"]].plot.bar(ax=axes[1], ylim=(0,100))
axes[1].set_title("Directed route continuity by trip"); axes[1].set_ylabel("%")
plt.show()"""),
        M("""## 3. Where decoding helped less

The table below sorts trips by **accuracy change**, including any negative result if one occurs. The ten full-trip API rehearsal results are also in artifacts/full_student_test_results.json. A gain on these student trips does not guarantee a gain on the reserved evaluator set. Even if continuity rises, the path may still be wrong because a smooth but incorrect road sequence can score well. Report unresolved graph links instead of claiming a perfectly connected path."""),
        C("""display(per_trip.sort_values("Accuracy change (pp)").head(5).style.format("{:.2f}"))
full = pd.read_json("artifacts/full_student_test_results.json").set_index("trip_id")
full["Accuracy change (pp)"] = 100*(full["approx_accuracy_after"]-full["approx_accuracy_before"])
full["Continuity change (pp)"] = 100*(full["route_continuity_after"]-full["route_continuity_before"])
display(full[["gps_points", "Accuracy change (pp)", "Continuity change (pp)", "unresolved_connections"]]
    .sort_values("Accuracy change (pp)").style.format({"Accuracy change (pp)":"{:.2f}","Continuity change (pp)":"{:.2f}"}))
print("Full-trip mean accuracy gain (pp):", round(full["Accuracy change (pp)"].mean(), 2))
print("Full-trip mean continuity gain (pp):", round(full["Continuity change (pp)"].mean(), 2))
print("Full-trip unresolved connections:", int(full["unresolved_connections"].sum()))"""),
        M("""### Highest accuracy: name the metric

The highest candidate-classification accuracy in the five-model student-test table was **90.48%** for the untuned Histogram Gradient Boosting classifier. It was achieved by classifying each GPS–candidate-arc row at the 0.5 probability threshold; it is not the accuracy of the final decoded route. After tuning, candidate accuracy was 90.46%. The best full-trip decoded pointwise proxy accuracy average was 78.30%. Route continuity reached 97.10% after Viterbi, but continuity measures connected neighboring arcs, not route accuracy. Do not report the 97.10% continuity as accuracy."""),
        M("""## 4. See the route change on one trip

Blue dots are the observed GPS samples, orange is the provided reference route (evaluation only), and green shows the model's chosen road arcs. The left panel uses independent classifier choices; the right panel uses Viterbi decoding. The reference route is never passed into the prediction model or decoder."""),
        C("""example_id = "00000052"
item, classifier, decoded = decoded_by_trip[example_id]
trip, road, ids = item["trip"], item["road"], item["point_ids"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
for ax, chosen, title in zip(axes, [classifier, decoded], ["Classifier only", "Classifier + Viterbi"]):
    ref = np.unique(trip.route)
    ax.add_collection(LineCollection(np.stack([road.a[ref], road.b[ref]], axis=1),
        colors="#e38b32", linewidths=3, alpha=.8, label="Reference route"))
    selected = np.unique(chosen)
    ax.add_collection(LineCollection(np.stack([road.a[selected], road.b[selected]], axis=1),
        colors="#16866b", linewidths=1.5, alpha=.9, label="Predicted arcs"))
    ax.scatter(road.track_xy[ids,0], road.track_xy[ids,1], s=3, color="#2779d1",
        alpha=.7, label="Observed GPS")
    ax.autoscale(); ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{example_id}: {title}"); ax.set_xlabel("Local metres"); ax.set_ylabel("Local metres")
    ax.legend(fontsize=8)
plt.show()"""),
        M("""## Viva answers and boundaries

1. **Hidden states?** Six candidate directed road arcs at each GPS observation. Their count changes by observation, but the candidate array has six positions per point. There are no neural hidden layers.
2. **Emissions?** HGB positive-class predict_proba for each candidate, clipped then logged. The inputs are the six engineered features. These probabilities are not a fitted GPS-error density.
3. **Transitions?** Fixed, unnormalized penalties: 0 same arc; 0.25 directed node connection; otherwise 1.5 plus endpoint distance divided by 25 times elapsed seconds. transition_weight defaults to 1.0.
4. **Large gaps or missing candidates?** Elapsed time softens the disconnected penalty. The decoder may jump if necessary. A missing true arc cannot be recovered at that point; bridge_route may insert a plausible directed path afterward.
5. **Disconnected segments?** They receive a larger penalty, not a hard ban. A* bridging is attempted after Viterbi; unresolved links remain visible in the results.
6. **Did decoding help?** Quote both proxy pointwise accuracy and route continuity from the executed per-trip and full-trip tables. Distinguish the 250-point sampled comparison from the full-trip API rehearsal. Do not claim that smoother always means more correct.
7. **Is this a fully trained HMM?** No. Emissions come from a discriminative classifier, transition terms are hand-set scores, and no normalized initial/transition distributions are learned. Call it HMM-style Viterbi map matching.
8. **How could it fail?** Wrong candidate generation, noisy/un-calibrated emissions, disconnected road graphs, unusually long GPS gaps, or a transition penalty that prefers a smooth wrong road. The full student-test rehearsal still has unresolved links."""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.environ["PATHSNAP_DATA_ROOT"] = str(args.data.resolve())
    notebooks = {
        "04_PathSnap_Bayesian_Hyperparameter_Optimisation.ipynb": bayesian_notebook(),
        "05_PathSnap_HMM_Viterbi_Route_Reconstruction.ipynb": hmm_notebook(),
    }
    for name, notebook in notebooks.items():
        path = ROOT / name
        nbf.write(notebook, path)
        if args.execute:
            NotebookClient(notebook, timeout=900, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}}).execute()
            nbf.write(notebook, path)
        print(("Executed" if args.execute else "Created"), path, flush=True)


if __name__ == "__main__":
    main()
