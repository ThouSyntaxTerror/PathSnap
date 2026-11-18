"""Create executed EDA and feature-engineering notebooks for report sections 2.2."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).parent
M = lambda s: nbf.v4.new_markdown_cell(s.strip())
C = lambda s: nbf.v4.new_code_cell(s.strip())


def eda():
    existing = nbf.read(ROOT / "PathSnap_Five_Model_Training.ipynb", as_version=4)
    cells = [
        M("""# 01 — Dataset understanding, EDA, and preprocessing

This is report section 2.2 and the repository's data-processing evidence. It uses the **80 train and 10 student-test trips** only. Raw files are inspected before cleaning, while training and prediction both use pathsnap.clean_track. Saved outputs are from the supplied files, not simulated data."""),
        C("""import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import joblib
from IPython.display import display
from matplotlib.collections import LineCollection
from pathsnap import Trip, RoadIndex, make_rows, decode, bridge_route, clean_track

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT", "data"))
assert (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "student_test").is_dir(), DATA_ROOT
print("Train folders:", len(list((DATA_ROOT/"train").glob("*/*.track"))))
print("Student-test folders:", len(list((DATA_ROOT/"student_test").glob("*/*.track"))))
print("Each trip has .track (lon,lat,time), .nodes (lon,lat), .arcs (node IDs), .route (reference arc IDs).")"""),
        M("""## Raw-data quality and cleaning

The cleaning function drops nonfinite or impossible coordinates/times, sorts timestamps, removes exact duplicate rows and conflicting same-time observations, then rejects observations requiring speed above 70 m/s from the last retained point. Repeated coordinates at different times remain because a car can stop. Zero removals for a category mean the raw supplied files did not contain that issue; the code still handles it."""),
        existing.cells[4],
        M("""## GPS trajectory and road network — student-test trip 00000052

Blue = recorded GPS points; grey = nearby road arcs; orange = supplied reference route. The reference is shown here for analysis only and never passed into the prediction model."""),
        C("""trip = Trip.load(DATA_ROOT/"student_test"/"00000052", include_route=True)
road, ids, candidates, x, labels = make_rows(trip, max_points=300)
nearby = np.unique(candidates.ravel())
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
ax.add_collection(LineCollection(np.stack((road.a[nearby],road.b[nearby]),axis=1),colors="#aab7c6",linewidths=.6,alpha=.7))
ref = np.unique(trip.route)
ax.add_collection(LineCollection(np.stack((road.a[ref],road.b[ref]),axis=1),colors="#e58b33",linewidths=2,alpha=.8))
ax.plot(road.track_xy[:,0],road.track_xy[:,1],".",markersize=2,color="#2779d1",alpha=.8)
ax.autoscale(); ax.set_aspect("equal",adjustable="box")
ax.set(title="Student-test trip 00000052: GPS, roads, reference route",xlabel="Local metres",ylabel="Local metres")
plt.show()
print("GPS observations:",len(trip.track),"nodes:",len(trip.nodes),"arcs:",len(trip.arcs),"reference arc entries:",len(trip.route))"""),
        M("""## Required interactive Folium geographical map

This real latitude/longitude map uses distinct layers for original GPS observations, nearby roads, the reference route, and the predicted route, with layer controls and extent fitted to the trip. It is generated from trip 00000052 in the **student-test** partition. The same code saves a standalone HTML map in the artifacts folder."""),
        C("""model = joblib.load("artifacts/final_model.joblib")
scores = model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(candidates.shape)
chosen = decode(trip,road,ids,candidates,scores)
routed, inserted, unresolved = bridge_route(trip,road,chosen)
center = [float(trip.track[:,1].mean()),float(trip.track[:,0].mean())]
m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)
roads = folium.FeatureGroup(name="Nearby road network",show=True)
gps = folium.FeatureGroup(name="Original GPS observations",show=True)
reference = folium.FeatureGroup(name="Reference route",show=True)
predicted = folium.FeatureGroup(name="Predicted route",show=True)
for arc in nearby:
    a,b=trip.arcs[arc]
    folium.PolyLine([trip.nodes[a,::-1].tolist(),trip.nodes[b,::-1].tolist()],color="#aab7c6",weight=1,opacity=.45).add_to(roads)
for lon,lat,_ in trip.track:
    folium.CircleMarker([lat,lon],radius=2,color="#2779d1",fill=True,fill_opacity=.7).add_to(gps)
for arc in dict.fromkeys(trip.route.tolist()):
    a,b=trip.arcs[arc]
    folium.PolyLine([trip.nodes[a,::-1].tolist(),trip.nodes[b,::-1].tolist()],color="#e58b33",weight=5,opacity=.8).add_to(reference)
for arc in dict.fromkeys(routed.tolist()):
    a,b=trip.arcs[arc]
    folium.PolyLine([trip.nodes[a,::-1].tolist(),trip.nodes[b,::-1].tolist()],color="#11976b",weight=3,opacity=.9).add_to(predicted)
for layer in (roads,reference,predicted,gps): layer.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[float(trip.track[:,1].min()),float(trip.track[:,0].min())],
              [float(trip.track[:,1].max()),float(trip.track[:,0].max())]])
m.save("artifacts/folium_trip_00000052.html")
display(m)
print("Saved interactive Folium HTML; bridges inserted:",inserted,"unresolved:",unresolved)"""),
        M("""## EDA conclusion

The raw student folder has 90 trips and 227,577 GPS observations. The cleaning audit found no invalid coordinates, exact duplicate rows, or timestamp conflicts; it removed 177 implausible speed observations and retained 227,400. Trip sizes and time gaps vary substantially, and many road graphs have disconnected components. The road and GPS plots demonstrate why nearest-road distance alone is insufficient and why temporal route decoding is needed. The reference route is a whole-path arc sequence, not a verified arc label for every GPS point."""),
    ]
    return nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})


def features():
    cells = [
        M("""# 02 — Feature engineering and candidate road preparation

This notebook shows the actual source-code path used in both training and inference: Trip.load → RoadIndex → candidate_ids → features → make_rows. The demonstration uses **training trip 00000001**. Six candidate arcs are prepared per sampled GPS observation. The reference route is used only to construct proxy training/evaluation labels, not inference features."""),
        C("""import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.collections import LineCollection
from pathsnap import Trip, RoadIndex, make_rows, FEATURE_NAMES

DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA_ROOT","data"))
trip = Trip.load(DATA_ROOT/"train"/"00000001",include_route=True)
road, ids, candidates, features, labels = make_rows(trip,max_points=120,candidate_count=6)
print("GPS observations sampled:",len(ids),"candidate arcs per point:",candidates.shape[1])
print("Model feature tensor:",features.shape,"candidate rows:",features.shape[0]*features.shape[1])
print("Proxy reference candidate coverage:",round(100*labels.any(axis=1).mean(),2),"%")"""),
        M("""## Six features: physical meaning and reason

| Feature | Meaning | Why it helps |
|---|---|---|
| log1p_distance_m | GPS point to candidate road segment | Nearby roads are more plausible |
| direction_cos | Agreement between GPS movement and directed road arc | Distinguishes parallel/reverse roads |
| log1p_speed_mps | Movement per elapsed second | Interprets plausible motion and GPS jumps |
| log1p_segment_length_m | Length of candidate arc | Captures road geometry scale |
| log1p_time_gap_s | Time since previous GPS observation | Larger gaps permit farther travel |
| log1p_movement_m | Straight-line movement since prior observation | Gives context to direction and speed |

The code projects longitude/latitude locally to approximate metres, uses a KD-tree on arc endpoints/midpoints for fast candidate proposals, then ranks candidates by exact point-to-segment distance. log1p compresses large nonnegative values; direction cosine remains in [-1,1]. Speed, movement, and time gap are related but tree boosting can learn their interactions."""),
        C("""frame = pd.DataFrame(features.reshape(-1,6),columns=FEATURE_NAMES)
frame["gps_index"] = np.repeat(ids,6)
frame["candidate_arc_id"] = candidates.ravel()
frame["proxy_positive_label"] = labels.ravel()
for field in ("distance_m","speed_mps","segment_length_m","time_gap_s","movement_m"):
    frame[field] = np.expm1(frame["log1p_"+field])
display(frame[["gps_index","candidate_arc_id","distance_m","direction_cos","speed_mps",
               "segment_length_m","time_gap_s","movement_m","proxy_positive_label"]].head(12))
display(frame.groupby("proxy_positive_label")[["distance_m","direction_cos","segment_length_m"]].median().round(2))"""),
        M("""## Actual feature distributions

These plots use the 120 sampled observations × six candidate arcs from training trip 00000001. Orange denotes the proxy reference candidate and blue the other candidates. Distributions overlap, which is why the model combines several features instead of using only distance."""),
        C("""fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout="constrained")
for label,color,name in ((0,"#6385ad","Other candidate"),(1,"#e58b33","Proxy route candidate")):
    s=frame[frame.proxy_positive_label==label]
    axes[0].hist(s["log1p_distance_m"],bins=25,density=True,alpha=.5,color=color,label=name)
    axes[1].hist(s["direction_cos"],bins=25,density=True,alpha=.5,color=color,label=name)
    axes[2].hist(s["log1p_segment_length_m"],bins=25,density=True,alpha=.5,color=color,label=name)
for ax,title in zip(axes,["Log GPS-to-road distance","Movement/road direction cosine","Log road-segment length"]):
    ax.set_title(title);ax.set_ylabel("Density")
axes[0].legend(fontsize=7)
plt.show()"""),
        M("""## Candidate geometry and proxy-label limit

The grey lines are six nearby candidate arcs for the first few GPS points; orange is the reference-route arc geometry, and blue points are recorded GPS samples. Candidate generation can miss the correct arc. The .route file does not give one label per GPS point, so labels are made by finding each point's nearest reference-route arc; this is a heuristic, not verified ground truth."""),
        C("""sample_ids=np.arange(min(25,len(ids)))
arc_ids=np.unique(candidates[sample_ids].ravel())
fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
ax.add_collection(LineCollection(np.stack((road.a[arc_ids],road.b[arc_ids]),axis=1),colors="#aab7c6",linewidths=1,alpha=.8))
ref=np.unique(trip.route)
visible=np.intersect1d(arc_ids,ref)
if len(visible): ax.add_collection(LineCollection(np.stack((road.a[visible],road.b[visible]),axis=1),colors="#e58b33",linewidths=3,alpha=.9))
ax.scatter(road.track_xy[ids[sample_ids],0],road.track_xy[ids[sample_ids],1],s=12,color="#2779d1")
ax.autoscale();ax.set_aspect("equal",adjustable="box")
ax.set(title="Training trip 00000001: candidate roads near GPS observations",xlabel="Local metres",ylabel="Local metres")
plt.show()"""),
        M("""## Source-code links for evaluators

The complete implementation is in pathsnap.py: Trip.load reads and cleans the track; RoadIndex.candidate_ids proposes and ranks directed arcs; RoadIndex.features computes six numbers; make_rows constructs the feature matrix and proxy labels. feature_example.py exports real physical-value examples to artifacts/feature_sample.csv. The same make_rows function is used by train.py and server.py, preventing training/inference feature mismatch."""),
    ]
    return nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    os.environ["PATHSNAP_DATA_ROOT"]=str(args.data.resolve())
    for name,book in [("01_PathSnap_EDA_and_Preprocessing.ipynb",eda()),
                      ("02_PathSnap_Feature_Engineering.ipynb",features())]:
        path=ROOT/name
        nbf.write(book,path)
        if args.execute:
            NotebookClient(book,timeout=900,kernel_name="python3",
                resources={"metadata":{"path":str(ROOT)}}).execute()
            nbf.write(book,path)
        print(("Executed" if args.execute else "Created"),path,flush=True)

if __name__=="__main__": main()
