"""CLI: bandingkan ML tabular vs Deep Learning sequence (+clustering), simpan hasil.

  python run_compare.py --simulate 60
  python run_compare.py --data fermentor_samples_simulated.json          # 1 sesi -> TEST MODE
  python run_compare.py --data real.json --scores cupping.csv            # data asli
Untuk antarmuka interaktif:  streamlit run streamlit_app.py
"""
import argparse, os
import numpy as np, pandas as pd
import common as C, sim_data, pipeline as P, plots as PL

ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--scores"); ap.add_argument("--simulate", type=int, default=0)
ap.add_argument("--interval-min", type=float, default=30.0)
ap.add_argument("--epochs", type=int, default=120); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--out", default="results")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

if a.simulate or not a.data:
    df, scores = sim_data.simulate_batches(a.simulate or 60, seed=0, interval_min=a.interval_min)
    df.to_json(f"{a.out}/simulated_batches.json", orient="records")
    scores.rename_axis("session").reset_index().to_csv(f"{a.out}/simulated_scores.csv", index=False)
    mode = "SINTETIS"
else:
    df = C.load_data(a.data, a.interval_min)
    if df["session"].nunique() < 2:
        df = C.make_pseudo_batches(df, 5)
        scores = pd.Series(np.random.default_rng(0).uniform(70, 88, df["session"].nunique()).round(1),
                           index=sorted(df["session"].unique()))
        mode = "TEST MODE (skor acak)"
    else:
        scores, mode = C.load_scores(a.scores), "DATA ASLI"

batches, ids, y = P.prepare(df, scores)
print(f"Mode: {mode} | batch berlabel: {len(ids)}")
exp = P.run_experiment(batches, ids, y, {"epochs": a.epochs, "seeds": a.seeds, "repeats": a.repeats},
                       progress=lambda f, m: print(f"  [{f:4.0%}] {m}", flush=True))
res, thr = exp["results"], exp["cfg"]["threshold"]
cols = ["Pendekatan", "Model", "MAE", "MAE_std", "RMSE", "R2", "Acc_enak", "F1_enak"]
res[cols].to_csv(f"{a.out}/comparison.csv", index=False)
print(f"\n=== HASIL CV ({exp['cv_desc']}) ===\n{res[cols].round(3).to_string(index=False)}")
print(f"\nTerbaik tabular: {exp['best_tab']} | terbaik sequence: {exp['best_seq']}")
if exp["k_tab"]:
    for col in ["cluster_tabular", "cluster_sequence"]:
        print(f"\n{col}:\n{P.cluster_summary(exp['clusters'], col, thr).to_string()}")
    print(f"\nARI tabular vs sequence: {exp['ari']:.2f}")
exp["clusters"].to_csv(f"{a.out}/clusters.csv", index=False)
P.save_models(exp, f"{a.out}/models")

R = res.set_index("Model")
for nm, tag in [(exp["best_tab"], "tabular"), (exp["best_seq"], "sequence")]:
    PL.pred_vs_actual(y, exp["oof"][nm], nm, R.loc[nm, "MAE"], R.loc[nm, "R2"], thr).savefig(f"{a.out}/pred_{tag}.png", dpi=130)
PL.mae_bars(res).savefig(f"{a.out}/mae_bars.png", dpi=130)
print(f"\nSelesai. Output: {a.out}/")
