# GitHub repository deliverable status

This maps the hackathon checklist to the current repository. The source and example artifacts are committed; a fresh run needs the dataset folder, which is intentionally kept outside GitHub.

The configured GitHub repository is public.

| Checklist item | Repository evidence | Status |
|---|---|---|
| Data processing | `pathsnap.py`, `inventory.py`, `validate_data.py` | Included: trip readers, raw-data audit, GPS cleaning, and preprocessing. |
| Feature engineering | `pathsnap.py`, `feature_example.py`, `artifacts/feature_sample.csv` | Included: candidate generation, six model features, and a sample feature export. |
| Model training | `train.py`, five model `.joblib` files, `artifacts/final_model.joblib` | Included: five classifiers and saved models. Model-family selection now uses grouped training CV; test-set measurements are for reporting. |
| Model evaluation | `train.py`, `evaluate_full.py`, `artifacts/experiment_results.json`, `artifacts/full_student_test_results.json`, `generate_assets.py`, `artifacts/confusion_matrices.png` | Included: metrics, confusion-matrix counts and plot, model comparison, and full student-test rehearsal outputs. |
| Hyperparameter optimisation | Optuna TPE search in `train.py`; trial details in `artifacts/experiment_results.json` | Included: search code, parameter spaces, trial values, and best parameters. |
| HMM / Viterbi | `pathsnap.py` (`decode`, `bridge_route`) | Included: Viterbi candidate-arc decoding and graph-based route bridging. |
| API | `server.py` | Included: FastAPI health, trip-list, and prediction endpoints with model loading. |
| UI | `ui/index.html` | Included: browser evaluator interface. |
| Deployment | `requirements.txt`, deployment steps in `README.md` | Included: dependencies and two-laptop run instructions. No platform-specific cloud deployment is configured. |
| Visualisations | `generate_assets.py`; plots and Folium HTML under `artifacts/` | Included: comparison, continuity, log-loss, confusion-matrix plots, and a Folium map. The map/EDA output requires the local student dataset. |
| Supporting results | `artifacts/*.json`, `artifacts/feature_sample.csv`, model files, report PDFs, `VIVA_GUIDE.md` | Included: experiment outputs, data audit, metrics, feature example, models, and guides. |

## Remaining before final submission

- Training, student-test evaluation, inventory, feature example, and data-backed map/EDA were refreshed from the local `Student folder` dataset using the current training-only selection rule.
- Capture the real two-laptop API console log/screenshot and add it to the final report. The included report is explicitly a draft and does not fabricate this evidence.
The dataset and reserved evaluator data are not included. Never add reserved evaluator routes to training or use their scores to select a model.
