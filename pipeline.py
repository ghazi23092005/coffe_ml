import os
import numpy as np
import pandas as pd
import joblib, torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import common as C
import tabular_model as TM
import sequence_model as SM

DEFAULT_CFG = {
    "threshold": C.GOOD_THRESHOLD, "repeats": 3, "epochs": 120, "seeds": 2,
    "lr": 3e-3, "noise": 0.05, "k": None,
    "tab_models": ["Ridge", "Random Forest", "Gradient Boosting"],
    "archs": ["gru", "cnn"],
    "tab_params": dict(TM.DEFAULT_TAB_PARAMS),
}
ARCH_LABEL = {"gru": "GRU", "cnn": "1D-CNN"}


def prepare(df: pd.DataFrame, scores: pd.Series):
    batches = C.split_batches(df)
    ids = [s for s in batches if s in scores.index and not pd.isna(scores.loc[s])]
    batches = {s: batches[s] for s in ids}
    y = scores.loc[ids].values.astype(float)
    return batches, ids, y


def _pick_k(Z, k=None):
    if len(Z) < 4:
        return None, None
    if k is None:
        ks = range(2, min(6, len(Z) - 1) + 1)
        k = max(ks, key=lambda kk: silhouette_score(Z, KMeans(kk, n_init=10, random_state=42).fit_predict(Z)))
    k = min(k, len(Z) - 1)
    return k, KMeans(k, n_init=10, random_state=42).fit_predict(Z)


def run_experiment(batches: dict, ids: list, y: np.ndarray, cfg: dict, progress=None) -> dict:
    cfg = {**DEFAULT_CFG, **cfg}
    say = progress or (lambda frac, msg: None)
    thr, n = cfg["threshold"], len(ids)
    F = TM.build_feature_table(batches)
    X, S, _ = SM.batches_to_tensor(batches)
    splits, fpr = C.make_splits(n, n_repeats=cfg["repeats"])
    seeds = tuple(range(cfg["seeds"]))

    all_tab = TM.get_models(**cfg["tab_params"])
    tab_names = ["Baseline (mean)"] + [m for m in cfg["tab_models"] if m in all_tab]
    steps = len(tab_names) + len(cfg["archs"]) + 2
    done, rows, oof_store = 0, [], {}

    for name in tab_names:
        say(done / steps, f"Tabular cross-validation: {name}")
        r, oof = C.cv_evaluate(TM.make_pred_fn(all_tab[name], F, y), y, splits, fpr, thr)
        rows.append({"Approach": "Reference" if "Baseline" in name else "Tabular", "Model": name, **r})
        oof_store[name] = oof; done += 1
    for arch in cfg["archs"]:
        say(done / steps, f"Sequence cross-validation: {ARCH_LABEL[arch]} (longest step)")
        fn = SM.make_pred_fn(arch, X, S, y, cfg["epochs"], seeds)
        r, oof = C.cv_evaluate(fn, y, splits, fpr, thr)
        rows.append({"Approach": "Deep Learning (sequence)", "Model": ARCH_LABEL[arch], **r})
        oof_store[ARCH_LABEL[arch]] = oof; done += 1

    res = pd.DataFrame(rows)
    tab_rows = res[res["Approach"] == "Tabular"]
    seq_rows = res[res["Approach"].str.startswith("Deep")]
    best_tab = tab_rows.loc[tab_rows["MAE"].idxmin(), "Model"] if len(tab_rows) else None
    best_seq = seq_rows.loc[seq_rows["MAE"].idxmin(), "Model"] if len(seq_rows) else None

    say(done / steps, "Training final model on all batches"); done += 1
    out = {"cfg": cfg, "ids": ids, "y": y, "F": F, "X": X, "S": S, "results": res, "oof": oof_store,
           "best_tab": best_tab, "best_seq": best_seq, "n": n,
           "cv_desc": "Leave-One-Out" if n < 10 else f"5-fold × {cfg['repeats']} repeat"}
    out["tab_model"] = TM.get_models(**cfg["tab_params"])[best_tab].fit(F, y) if best_tab else None
    out["importance"] = TM.feature_importance(out["tab_model"]) if best_tab else pd.Series(dtype=float)
    label2arch = {v: k for k, v in ARCH_LABEL.items()}
    out["seq_bundle"] = (SM.fit_bundle(label2arch[best_seq], X, S, y, epochs=cfg["epochs"], seeds=seeds,
                                       lr=cfg["lr"], noise=cfg["noise"]) if best_seq else None)
    gru = out["seq_bundle"] if (best_seq == "GRU") else SM.fit_bundle(
        "gru", X, S, y, epochs=cfg["epochs"], seeds=seeds, lr=cfg["lr"], noise=cfg["noise"])

    say(done / steps, "Clustering batches"); done += 1
    Z_tab = StandardScaler().fit_transform(F)
    Z_seq = StandardScaler().fit_transform(SM.embed_bundle(gru, X, S))
    k_tab, lab_tab = _pick_k(Z_tab, cfg["k"]); k_seq, lab_seq = _pick_k(Z_seq, cfg["k"])
    cl = pd.DataFrame({"session": ids, "score": y})
    out.update({"k_tab": k_tab, "k_seq": k_seq, "clusters": cl, "ari": None, "pca_tab": None, "pca_seq": None})
    if lab_tab is not None:
        cl["cluster_tabular"], cl["cluster_sequence"] = lab_tab, lab_seq
        out["ari"] = adjusted_rand_score(lab_tab, lab_seq)
        out["pca_tab"] = PCA(2, random_state=0).fit_transform(Z_tab)
        out["pca_seq"] = PCA(2, random_state=0).fit_transform(Z_seq)
    say(1.0, "Completed")
    return out


def cluster_summary(clusters: pd.DataFrame, col: str, thr: float) -> pd.DataFrame:
    g = clusters.groupby(col)["score"].agg(["count", "mean", "std"]).rename(
        columns={"count": "Batch count", "mean": "Mean score", "std": "Std"})
    g["% good"] = clusters.groupby(col)["score"].apply(lambda s: 100 * (s >= thr).mean())
    g.index.name = "Cluster"
    return g.round(1)


def predict_batches(exp: dict, batches: dict) -> pd.DataFrame:
    thr = exp["cfg"]["threshold"]
    F = TM.build_feature_table(batches)
    X, S, ids = SM.batches_to_tensor(batches)
    out = pd.DataFrame(index=pd.Index(ids, name="session"))
    cols = []
    if exp["tab_model"] is not None:
        c = f"Tabular · {exp['best_tab']}"; out[c] = exp["tab_model"].predict(F); cols.append(c)
    if exp["seq_bundle"] is not None:
        c = f"Sequence · {exp['best_seq']}"; out[c] = SM.predict_bundle(exp["seq_bundle"], X, S); cols.append(c)
    out["Average"] = out[cols].mean(axis=1)
    out["Model difference"] = out[cols].max(axis=1) - out[cols].min(axis=1)
    out["Prediction"] = np.where(out["Average"] >= thr, "Good", "Poor")
    return out


def save_models(exp: dict, folder: str):
    os.makedirs(folder, exist_ok=True)
    joblib.dump({"model": exp["tab_model"], "name": exp["best_tab"], "features": TM.FEATURE_COLS,
                 "threshold": exp["cfg"]["threshold"]}, f"{folder}/tabular_best.joblib")
    torch.save(exp["seq_bundle"], f"{folder}/sequence_best.pt")


def load_models(folder: str) -> dict:
    tab = joblib.load(f"{folder}/tabular_best.joblib")
    seq = torch.load(f"{folder}/sequence_best.pt", weights_only=False)
    return {"tab_model": tab["model"], "best_tab": tab["name"], "seq_bundle": seq,
            "best_seq": ARCH_LABEL[seq["arch"]], "cfg": {**DEFAULT_CFG, "threshold": tab.get("threshold", C.GOOD_THRESHOLD)}}
