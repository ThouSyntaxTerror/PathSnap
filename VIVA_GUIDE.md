# PathSnap viva quick reference

These values are from the recorded **student-test rehearsal**, not the hidden evaluator trips. Pointwise labels are a proxy: the nearest arc on the supplied reference route for each GPS point. Be ready to explain that limitation.

## Explain the pipeline in 30 seconds

1. `.track` gives noisy longitude, latitude, and time. `.nodes` and `.arcs` form the road graph. `.route` is a chronological reference arc sequence, not a row-by-row GPS label.
2. For each GPS point, we find six nearby road-arc candidates and compute distance, direction, speed, segment length, time gap, and movement features.
3. A tuned Histogram Gradient Boosting classifier gives each candidate a probability.
4. Viterbi chooses a sequence using classifier probabilities and road adjacency. A separate graph step tries to bridge remaining gaps; unresolved gaps are reported.
5. Laptop 1 runs the model/API and keeps reference routes. Laptop 2 sends a trip ID and displays the returned prediction and metrics.

## What each metric means

| Metric | Meaning | Tuned model, sampled student test |
|---|---|---:|
| Candidate accuracy | Fraction of candidate road rows classified yes/no correctly. Negatives are common, so this can look high even for a weak model. | 90.36% |
| Precision | When the classifier says a candidate is correct, how often is it correct? | 75.34% |
| Recall | Of the correct candidates present, how many did the classifier find? | 60.95% |
| Macro F1 | Average F1 of positive and negative candidate classes, giving each class equal weight. | 0.809 student test; 0.788 grouped CV for the untuned model |
| Confusion matrix | Counts of true negatives, false positives, false negatives, true positives. | `[[12060, 489], [957, 1494]]` |
| Log loss | Tests quality of predicted probabilities; lower is better. | 0.228 |
| Observation Top-1 | For each GPS point, choose the highest-scoring candidate and check its proxy label. | 75.16% |
| Candidate miss rate | Fraction of GPS points whose proxy reference arc was absent from all six candidates. The classifier cannot recover a missing candidate. | 1.96% |
| Route continuity | Fraction of successive predicted arcs that are the same or directly connected in the road graph. | 60.52% before, 71.89% after Viterbi on sampled points |
| Route-level result | Proxy pointwise accuracy after Viterbi; report alongside continuity, not as a true whole-route match score. | 76.76% |
| Training time | Recorded fitting plus three-fold CV time for the initial Histogram Gradient Boosting model on sampled training rows. | About 2.4 seconds on the development machine |
| Inference time | Model scoring only over the sampled student-test trips; candidate generation and Viterbi add time. | About 0.055 seconds total |

The **full** ten-trip rehearsal used all 19,160 GPS observations and achieved 78.2% mean per-trip proxy accuracy after Viterbi and 97.1% mean direct-adjacency continuity. These full-trip numbers are not directly comparable with the evenly sampled experiment above. The post-decoding bridge step still left 163 transitions unresolved across all ten trips.

## Why this model?

We compared Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and Histogram Gradient Boosting on trip-grouped CV and ten student-test trips. Gaussian Naive Bayes had the best classifier-only Top-1 score (76.24%), but Histogram Gradient Boosting had the best proxy pointwise score **after Viterbi** (76.52%) among the initial five. Route reconstruction is the final task, so we selected that model family. Bayesian optimisation with five Optuna TPE trials selected `max_iter=97`, `max_leaf_nodes=10`, and `learning_rate=0.1889`. The tuned after-Viterbi score was 76.76%, a small improvement. Never claim a large tuning gain.

## Questions to expect

- **Why not split GPS rows randomly?** Rows from the same trip are correlated. GroupKFold keeps whole trips together to avoid leakage.
- **Where do pointwise labels come from?** We infer them by nearest distance to arcs on the training/reference route. The `.route` file does not label each GPS row directly.
- **What do HMM and Viterbi add?** The classifier scores each point independently; Viterbi uses neighboring points and road adjacency to favour a coherent sequence.
- **Is the route always connected?** No. Viterbi has a spatial fallback for gaps; graph bridging is attempted afterward, and unresolved links are shown. Do not claim perfect connectivity.
- **What data did you use to tune?** Grouped CV on 80 training trips. The ten student-test trips supported development comparisons. The reserved evaluator trips were not available or used.
- **What are the biggest limitations?** Heuristic pointwise labels, missed candidates, disconnected road graphs, a very long observed GPS time gap, and unresolved route transitions.
- **What does the live demo show?** Laptop 2 selects any of ten trip IDs; Laptop 1 logs the request, runs inference, and returns a predicted route and server-side metrics. Capture the Laptop 1 log screenshot during the actual demo.

The exact recorded settings, confusion matrices, and timings are in `artifacts/experiment_results.json`; full-trip results are in `artifacts/full_student_test_results.json`.
