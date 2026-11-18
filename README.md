# PathSnap: two-laptop hackathon prototype

This repository contains a real, runnable pipeline for the supplied student dataset: GPS/road loading, six road candidates per observation, five classifier experiments, grouped cross-validation, Bayesian tuning, a Viterbi decoder, a FastAPI evaluator endpoint, and a browser UI. The included model and results were produced from the 80 training trips and 10 student-test trips. The separate evaluator trips were **not** supplied or used.

## Quick demo with the student-test trips

Use Python 3.12 on Laptop 1. From this folder in PowerShell:

```powershell
py -3.12 -m pip install -r requirements.txt
$env:PATHSNAP_DATA_ROOT='C:\path\to\Student folder'
py -3.12 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

The saved model is `artifacts/final_model.joblib`. The API reads the ten `student_test` trips by default and labels its response **student-test rehearsal**. It does not need to be retrained for this demo.

On Laptop 2, copy the `ui` folder, open PowerShell inside it, and run:

```powershell
py -3.12 -m http.server 8080
```

Open `http://localhost:8080` in Laptop 2's browser. Enter `http://LAPTOP1_IP:8000`, click **Connect**, choose any trip, then click **Predict route**. Find Laptop 1's IPv4 address with `ipconfig`; both laptops must be on the same network. Allow the Windows firewall prompt for Python on the private network if it appears. Laptop 1's console prints `INFERENCE REQUEST` and `INFERENCE DONE` for every selection. Show those logs to the evaluator.

To switch to the final reserved trips, set `PATHSNAP_EVAL_ROOT` **on Laptop 1 only** to their folder before starting the API:

```powershell
$env:PATHSNAP_EVAL_ROOT='C:\path\to\reserved_evaluator_trips'
py -3.12 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

The folder must contain trip subfolders with matching `.track`, `.nodes`, `.arcs`, and (for server-side accuracy and the orange map layer) `.route` files. Prediction runs without the reference route; after prediction, the API may send its geometry for visual comparison. The UI shows the recorded GPS trace in blue, the provided reference route in orange, and the predicted route in green. There is no separate pre-trip planned-route file. Do **not** put evaluator data in training or use its results to change model choices.

## Retrain and regenerate results

## Ordered evaluation notebooks

The five executed notebooks in the repository root are:

1. `01_PathSnap_EDA_and_Preprocessing.ipynb` — raw-data audit, cleaning, GPS/road plots, and an interactive Folium map.
2. `02_PathSnap_Feature_Engineering.ipynb` — six candidate features, real values, distributions, and road-candidate geometry.
3. `03_PathSnap_Model_Training_and_Evaluation.ipynb` — five models, grouped CV, metrics, confusion matrices, and comparison plots.
4. `04_PathSnap_Bayesian_Hyperparameter_Optimisation.ipynb` — five Optuna TPE trials and fair untuned/tuned comparison.
5. `05_PathSnap_HMM_Viterbi_Route_Reconstruction.ipynb` — hidden arc states, emission/transition scores, before/after results, and route plots.

They include saved outputs. To rerun them, place them in this repository root, set `PATHSNAP_DATA_ROOT` to the extracted student folder, install `requirements.txt`, and open Jupyter Notebook. The standalone report draft is `artifacts/PathSnap_Technical_Report_DRAFT.pdf`; add a real API-log screenshot from the final two-laptop demonstration before submission.

```powershell
py -3.12 train.py --data 'C:\path\to\Student folder' --output artifacts --train-points-per-trip 120 --test-points-per-trip 250 --trials 5
py -3.12 inventory.py --data 'C:\path\to\Student folder' --output artifacts
py -3.12 feature_example.py --data 'C:\path\to\Student folder' --output artifacts/feature_sample.csv
py -3.12 evaluate_full.py --data 'C:\path\to\Student folder' --output artifacts
py -3.12 generate_assets.py --data 'C:\path\to\Student folder' --output artifacts
py -3.12 make_report.py
py -3.12 make_first_eval_guide.py
```

To open the executed five-model notebook, set `PATHSNAP_DATA_ROOT` to the extracted student folder, then run `py -3.12 -m notebook PathSnap_Five_Model_Training.ipynb` (install `notebook` if needed). Each model has its own training/evaluation cell and confusion matrix; the last cell compares all five, including route continuity. The EDA cells show raw trip sizes, actual cleaning counts, and raw-versus-cleaned GPS positions.

`train.py` uses all 80 training trips, sampling at most 120 GPS observations per trip to fit within hackathon time. The five models are Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and Histogram Gradient Boosting. Three-fold GroupKFold separates trips. Model selection and Optuna TPE tuning use grouped training CV; the ten student-test trips are for reporting only. All model settings, timings, metrics, and tuning trials are in `artifacts/experiment_results.json`.

`generate_assets.py` saves comparison charts and an interactive Folium map of student-test trip `00000052`. The report generator uses actual logged data and screenshots. The included PDF is a **draft**: capture and add the required Laptop 1 CMD/API-log screenshot during the actual two-laptop demonstration. Do not submit invented measurements or a fabricated screenshot.

## Data and method

Each trip contains:

| File | Columns | Meaning |
|---|---|
| `.track` | longitude, latitude, time | GPS observations |
| `.nodes` | longitude, latitude | road graph coordinates; row number is node ID |
| `.arcs` | from-node ID, to-node ID | directed road arcs; row number is arc ID |
| `.route` | arc ID | chronological reference arc sequence |

`Trip.load` applies the same cleaning to training and API inference: remove invalid/nonfinite GPS rows, sort timestamps, remove exact duplicate rows and conflicting same-time observations, then reject points requiring speed above 70 m/s (252 km/h) from the last retained point. Repeated coordinates at different times are kept because the vehicle may be stopped. The student data has 227,577 raw observations; this cleaning removes 177 high-speed observations and no invalid, duplicate, or conflicting-timestamp rows. The 70 m/s threshold is a conservative heuristic, not a verified physical label; inspect it if the evaluator data has different vehicle speeds.

There is no one-to-one mapping from GPS rows to route rows. To construct approximate pointwise training labels, the code matches each GPS point to its nearest arc **among the reference-route arcs**. Evaluation pointwise accuracy uses the same proxy. This is not verified ground truth. Candidate generation searches arc endpoints and midpoints with a KD-tree, then measures exact point-to-segment distance in a local metre projection. A six-candidate search missed the proxy reference arc for 1.96% of sampled student-test observations. Some training trips have much larger miss rates; inspect those before claiming strong generalisation.

The Viterbi decoder favours staying on the same arc, then moving to a directly connected arc. When candidates are disconnected, an endpoint-distance fallback permits a jump with a penalty adjusted by GPS time gap. After decoding, the API attempts a directed shortest-path bridge through actual road arcs and rejects implausibly long bridges. It reports any unresolved connections. The decoder itself does **not** guarantee a fully connected graph route. A full road-network shortest-path transition model inside Viterbi is the main improvement to pursue if time permits.

## Files

- `pathsnap.py`: trip parsing, geometry, features, labels, Viterbi decoding
- `inventory.py`: reproducible dataset inventory and quality checks
- `feature_example.py`: export an actual, readable candidate-feature table for the viva
- `train.py`: five-model comparison, grouped CV, Optuna search, model artifacts
- `server.py`: FastAPI evaluator endpoint and server-side metrics
- `evaluate_full.py`: complete student-test rehearsal through the same prediction path
- `ui/index.html`: Laptop 2 interface, no JavaScript dependencies or online map tiles
- `generate_assets.py`: real comparison plot and interactive Folium map
- `make_report.py`: generated technical report draft
- `make_first_eval_guide.py`: short first-evaluation PDF tied to the source code and recorded metrics
- `VIVA_GUIDE.md`: short explanations of the required metrics and likely evaluator questions
- `artifacts/`: trained models, actual results, maps, screenshots, report draft

The source dataset is intentionally not copied into this project. Keep your downloaded dataset in its own folder and set `PATHSNAP_DATA_ROOT` to that path.
