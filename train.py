"""Run five-model comparison, grouped CV, Bayesian tuning, and route evaluation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from pathsnap import Trip, make_rows, decode, route_continuity as continuity, top1_accuracy, FEATURE_NAMES


def models():
    return {
        "logistic_regression": LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
        "gaussian_nb": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=15, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=8,
                                                 class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=25,
                                                                  l2_regularization=0.1, random_state=42),
    }


def collect(partition: Path, max_points: int):
    features, labels, groups, trips = [], [], [], []
    for folder in sorted(partition.iterdir()):
        if not folder.is_dir(): continue
        trip = Trip.load(folder, include_route=True)
        road, ids, candidates, x, y = make_rows(trip, max_points=max_points)
        if y is None: continue
        features.append(x.reshape(-1, x.shape[-1]))
        labels.append(y.ravel())
        groups.extend([trip.trip_id] * x.shape[0] * x.shape[1])
        trips.append({"trip": trip, "road": road, "point_ids": ids, "candidates": candidates, "features": x,
                      "labels": y})
        print(f"loaded {partition.name}/{trip.trip_id}: {len(ids)} points, candidate coverage {y.any(axis=1).mean():.3f}", flush=True)
    return np.concatenate(features), np.concatenate(labels), np.asarray(groups), trips


def score_model(model, trips):
    all_y, all_p = [], []
    top1, decoded_accuracy, continuity_before, continuity_after, miss, seconds = [], [], [], [], [], []
    for item in trips:
        x = item["features"]
        tic = time.perf_counter()
        probability = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(x.shape[:2])
        seconds.append(time.perf_counter() - tic)
        y = item["labels"]
        all_y.append(y.ravel()); all_p.append(probability.ravel())
        top1.append(top1_accuracy(probability, y))
        miss.append(1 - float(y.any(axis=1).mean()))
        baseline = item["candidates"][np.arange(len(probability)), np.argmax(probability, axis=1)]
        decoded = decode(item["trip"], item["road"], item["point_ids"], item["candidates"], probability)
        decoded_accuracy.append(float(np.mean(decoded == item["road"].reference_arc_per_point(item["road"].track_xy[item["point_ids"]]))))
        continuity_before.append(continuity(item["trip"], baseline))
        continuity_after.append(continuity(item["trip"], decoded))
    y, p = np.concatenate(all_y), np.concatenate(all_p)
    hard = (p >= 0.5).astype(int)
    return {
        "candidate_accuracy": float(accuracy_score(y, hard)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, hard, labels=[0, 1]).tolist(),
        "candidate_macro_f1": float(f1_score(y, hard, average="macro", zero_division=0)),
        "precision": float(precision_score(y, hard, zero_division=0)),
        "recall": float(recall_score(y, hard, zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "mean_trip_top1_accuracy": float(np.mean(top1)),
        "mean_trip_accuracy_viterbi": float(np.mean(decoded_accuracy)),
        "mean_candidate_miss_rate": float(np.mean(miss)),
        "mean_continuity_classifier": float(np.mean(continuity_before)),
        "mean_continuity_viterbi": float(np.mean(continuity_after)),
        "inference_seconds_total": float(sum(seconds)),
    }


def grouped_cv(model, x, y, groups, folds=3):
    scores = []
    for fit, valid in GroupKFold(n_splits=folds).split(x, y, groups):
        candidate = clone(model)
        candidate.fit(x[fit], y[fit])
        pred = candidate.predict(x[valid])
        scores.append(float(f1_score(y[valid], pred, average="macro", zero_division=0)))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to extracted Student folder")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-points-per-trip", type=int, default=180)
    parser.add_argument("--test-points-per-trip", type=int, default=400)
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Loading training trips", flush=True)
    x, y, groups, train_trips = collect(args.data / "train", args.train_points_per_trip)
    print("Loading student-test trips", flush=True)
    _, _, _, test_trips = collect(args.data / "student_test", args.test_points_per_trip)
    print(f"Training candidate rows: {len(y)}, positive rate: {y.mean():.4f}", flush=True)
    results = {"dataset": {"training_trips": len(train_trips), "student_test_trips": len(test_trips),
                           "training_candidate_rows": len(y), "positive_rate": float(y.mean()),
                           "label_method": "nearest reference-route arc to each GPS observation (heuristic)"},
               "features": FEATURE_NAMES, "models": {}}
    fitted = {}
    for name, model in models().items():
        print(f"Training {name}", flush=True)
        start = time.perf_counter()
        cv = grouped_cv(model, x, y, groups, folds=3)
        model.fit(x, y)
        train_seconds = time.perf_counter() - start
        train_macro_f1 = float(f1_score(y, model.predict(x), average="macro", zero_division=0))
        results["models"][name] = {"cv_macro_f1": cv, "cv_macro_f1_mean": float(np.mean(cv)),
                                    "train_macro_f1": train_macro_f1,
                                    "training_seconds_including_cv": train_seconds,
                                    "parameters": model.get_params()}
        fitted[name] = model
        joblib.dump(model, args.output / f"{name}.joblib")
        print(f"{name}: grouped training-CV macro F1={np.mean(cv):.4f}", flush=True)
        (args.output / "experiment_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Select the family using grouped training CV only; student-test results are reporting-only.
    best_name = max(results["models"], key=lambda name: results["models"][name]["cv_macro_f1_mean"])
    results["selection_phase1"] = {"family": best_name, "criterion": "highest mean grouped training-CV macro F1"}
    for name, model in fitted.items():
        results["models"][name].update(score_model(model, test_trips))
    (args.output / "experiment_results.json").write_text(json.dumps(results, indent=2, default=str))

    # Tune the selected family with grouped training CV only.
    def objective(trial):
        if best_name == "hist_gradient_boosting":
            candidate = HistGradientBoostingClassifier(max_iter=trial.suggest_int("max_iter", 40, 120),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 40),
                learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True), random_state=42)
        elif best_name == "random_forest":
            candidate = RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 30, 100),
                max_depth=trial.suggest_int("max_depth", 6, 20), min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 20),
                n_jobs=-1, random_state=42)
        elif best_name == "decision_tree":
            candidate = DecisionTreeClassifier(max_depth=trial.suggest_int("max_depth", 4, 24),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 30), random_state=42)
        elif best_name == "logistic_regression":
            candidate = LogisticRegression(C=trial.suggest_float("C", 0.01, 10, log=True),
                max_iter=400, class_weight="balanced", random_state=42)
        else:
            candidate = GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True))
        return float(np.mean(grouped_cv(candidate, x, y, groups, folds=3)))
    search_spaces = {
        "hist_gradient_boosting": {"max_iter": [40, 120], "max_leaf_nodes": [10, 40], "learning_rate": [0.03, 0.2], "learning_rate_log": True},
        "random_forest": {"n_estimators": [30, 100], "max_depth": [6, 20], "min_samples_leaf": [2, 20]},
        "decision_tree": {"max_depth": [4, 24], "min_samples_leaf": [2, 30]},
        "logistic_regression": {"C": [0.01, 10], "C_log": True},
        "gaussian_nb": {"var_smoothing": [1e-12, 1e-6], "var_smoothing_log": True},
    }
    search_space = search_spaces[best_name]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.trials)
    tuned = models()[best_name].set_params(**study.best_params)
    tuned.fit(x, y)
    joblib.dump(tuned, args.output / "final_model.joblib")
    results["selection"] = {"family": best_name, "criterion": "highest mean grouped training-CV macro F1; student-test used for reporting only",
                            "search_space": search_space,
                            "best_parameters": study.best_params, "best_cv_macro_f1": study.best_value,
                            "tuned_student_test": score_model(tuned, test_trips),
                            "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials]}
    (args.output / "experiment_results.json").write_text(json.dumps(results, indent=2, default=str))
    print("Saved model and experiment_results.json", flush=True)


if __name__ == "__main__": main()
