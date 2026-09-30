"""Fungsi umum: load data, test mode (pseudo-batch), skor, evaluasi CV bersama.

Dipakai oleh model tabular DAN model sequence supaya keduanya dievaluasi
dengan split CV yang persis sama (perbandingan yang adil).
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

SENSOR_COLS = ["o2_percent", "ph_value", "scd41_co2_ppm",
               "scd41_temp_c", "scd41_rh_percent"]
GOOD_THRESHOLD = 80.0   # skor cupping >= 80 dianggap "enak" (specialty SCA)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data(path_or_buffer, interval_min: float = 30.0) -> pd.DataFrame:
    """Baca JSON sensor. Kalau tidak ada kolom `elapsed_h`, dihitung dari
    nomor sampel x interval_min (menit)."""
    if hasattr(path_or_buffer, "read"):
        raw = json.load(path_or_buffer)
    else:
        with open(path_or_buffer, "r") as f:
            raw = json.load(f)
    df = pd.DataFrame(raw)
    df = df.drop(columns=[c for c in df.columns if c.startswith("pt100")],
                 errors="ignore")
    df = df.sort_values(["session", "sample"]).reset_index(drop=True)
    if "elapsed_h" not in df.columns:
        first = df.groupby("session")["sample"].transform("min")
        df["elapsed_h"] = (df["sample"] - first) * interval_min / 60.0
    return df


def make_pseudo_batches(df: pd.DataFrame, n_chunks: int = 5) -> pd.DataFrame:
    """TEST MODE: pecah 1 sesi jadi n_chunks 'batch semu' (hanya uji kode)."""
    df = df.sort_values("sample").copy()
    size = int(np.ceil(len(df) / n_chunks))
    df["session"] = (np.arange(len(df)) // size) + 1
    df["elapsed_h"] = df["elapsed_h"] - df.groupby("session")["elapsed_h"].transform("min")
    return df.reset_index(drop=True)


def split_batches(df: pd.DataFrame) -> dict:
    return {s: g.sort_values("elapsed_h").reset_index(drop=True)
            for s, g in df.groupby("session")}


def load_scores(path: str) -> pd.Series:
    """CSV dengan kolom: session,score (skor cupping dari Q-Grader)."""
    s = pd.read_csv(path)
    return pd.Series(s["score"].values, index=s["session"].values, name="score")


# ----------------------------------------------------------------------
# Evaluasi CV (dipakai bersama)
# ----------------------------------------------------------------------
def make_splits(n: int, n_splits: int = 5, n_repeats: int = 3, seed: int = 42):
    """Return (splits, folds_per_repeat). n kecil -> Leave-One-Out."""
    if n < 10:
        return list(LeaveOneOut().split(np.arange(n))), n
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rkf.split(np.arange(n))), n_splits


def _metrics(y, p, thr=GOOD_THRESHOLD):
    yc, pc = y >= thr, p >= thr
    out = {
        "MAE": mean_absolute_error(y, p),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": r2_score(y, p) if len(y) > 1 else np.nan,
        "Acc_enak": float((yc == pc).mean()),
    }
    out["F1_enak"] = f1_score(yc, pc, zero_division=0) if yc.any() else np.nan
    return out


def cv_evaluate(pred_fn, y: np.ndarray, splits, folds_per_repeat: int, threshold: float = GOOD_THRESHOLD):
    """pred_fn(train_idx, test_idx) -> prediksi utk test_idx.
    Return (dict metrik rata-rata & std antar-repeat, prediksi OOF rata-rata)."""
    n = len(y)
    n_rep = max(1, len(splits) // folds_per_repeat)
    oof = np.full((n_rep, n), np.nan)
    for i, (tr, te) in enumerate(splits):
        oof[i // folds_per_repeat, te] = pred_fn(tr, te)
    per_rep = [_metrics(y, oof[r], threshold) for r in range(n_rep)]
    res = {}
    for k in per_rep[0]:
        vals = np.array([m[k] for m in per_rep], dtype=float)
        res[k] = float(np.nanmean(vals))
        res[k + "_std"] = float(np.nanstd(vals))
    return res, np.nanmean(oof, axis=0)
