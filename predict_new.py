
import argparse
import common as C, pipeline as P
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--models", default="results/models")
ap.add_argument("--interval-min", type=float, default=30.0)
a = ap.parse_args()
exp = P.load_models(a.models)
print(P.predict_batches(exp, C.split_batches(C.load_data(a.data, a.interval_min))).round(1).to_string())
