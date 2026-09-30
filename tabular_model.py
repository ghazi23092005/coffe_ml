"""Pendekatan 1 — ML TABULAR: ringkas tiap batch jadi fitur (berbasis JAM,
bukan 'per sampel'), lalu Ridge / Random Forest / Gradient Boosting.
Scaler ada di dalam Pipeline sehingga tidak bocor saat cross-validation."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

FEATURE_COLS = [
    "duration_h", "ph_start", "ph_end", "ph_drop", "ph_max_drop_rate",
    "t_ph_below_5_h", "h_ph_below_4",
    "co2_start", "co2_max", "co2_rise", "co2_mean",
    "o2_min", "o2_drop",
    "temp_mean", "temp_std", "temp_max", "rh_mean", "rh_std",
]


def batch_features(g: pd.DataFrame) -> dict:
    t = g["elapsed_h"].values.astype(float)
    sm = lambda c: g[c].rolling(5, center=True, min_periods=1).mean().values
    ph, co2, o2 = sm("ph_value"), sm("scd41_co2_ppm"), sm("o2_percent")
    temp, rh = g["scd41_temp_c"].values, g["scd41_rh_percent"].values
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    dph = np.gradient(ph, t) if len(t) > 2 and dur > 0 else np.zeros_like(ph)
    step = dur / max(len(t) - 1, 1)
    below5 = np.where(ph <= 5.0)[0]
    return {
        "duration_h": dur,
        "ph_start": ph[:3].mean(), "ph_end": ph[-3:].mean(),
        "ph_drop": ph[:3].mean() - ph[-3:].mean(),
        "ph_max_drop_rate": float(-dph.min()),                # pH turun tercepat (per jam)
        "t_ph_below_5_h": float(t[below5[0]] - t[0]) if len(below5) else dur,
        "h_ph_below_4": float(step * np.sum(ph < 4.0)),       # jam di zona over-fermentasi
        "co2_start": co2[:3].mean(), "co2_max": co2.max(),
        "co2_rise": co2.max() - co2[:3].mean(), "co2_mean": co2.mean(),
        "o2_min": o2.min(), "o2_drop": o2[:3].mean() - o2.min(),
        "temp_mean": temp.mean(), "temp_std": temp.std(), "temp_max": temp.max(),
        "rh_mean": rh.mean(), "rh_std": rh.std(),
    }


def build_feature_table(batches: dict) -> pd.DataFrame:
    return pd.DataFrame({s: batch_features(g) for s, g in batches.items()}).T[FEATURE_COLS]


DEFAULT_TAB_PARAMS = {"rf_trees": 300, "rf_depth": 6, "gb_trees": 150, "gb_depth": 2, "gb_lr": 0.05}


def get_models(rf_trees=300, rf_depth=6, gb_trees=150, gb_depth=2, gb_lr=0.05) -> dict:
    return {
        "Baseline (rata-rata)": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestRegressor(
            n_estimators=rf_trees, max_depth=rf_depth, min_samples_leaf=2, random_state=42, n_jobs=1)),
        "Gradient Boosting": make_pipeline(StandardScaler(), GradientBoostingRegressor(
            n_estimators=gb_trees, max_depth=gb_depth, learning_rate=gb_lr, subsample=0.8, random_state=42)),
    }


# Deskripsi fitur (dipakai dashboard)
FEATURE_INFO = {
    "duration_h": "Lama fermentasi (jam)",
    "ph_start": "pH awal (rata-rata 3 titik pertama)",
    "ph_end": "pH akhir (rata-rata 3 titik terakhir)",
    "ph_drop": "Total penurunan pH",
    "ph_max_drop_rate": "Laju penurunan pH tercepat (pH/jam)",
    "t_ph_below_5_h": "Jam sampai pH turun ≤ 5.0",
    "h_ph_below_4": "Jam pH < 4.0 (indikator over-fermentasi)",
    "co2_start": "CO₂ awal (ppm)", "co2_max": "CO₂ puncak (ppm)",
    "co2_rise": "Kenaikan CO₂ (ppm)", "co2_mean": "CO₂ rata-rata (ppm)",
    "o2_min": "O₂ minimum (%)", "o2_drop": "Penurunan O₂ (%)",
    "temp_mean": "Suhu rata-rata (°C)", "temp_std": "Variasi suhu (std)", "temp_max": "Suhu maksimum (°C)",
    "rh_mean": "Kelembapan rata-rata (%)", "rh_std": "Variasi kelembapan (std)",
}


def make_pred_fn(model_factory, X: pd.DataFrame, y: np.ndarray):
    from sklearn.base import clone
    def fn(tr, te):
        m = clone(model_factory)
        m.fit(X.iloc[tr], y[tr])
        return m.predict(X.iloc[te])
    return fn


def feature_importance(fitted_pipeline) -> pd.Series:
    est = fitted_pipeline[-1] if hasattr(fitted_pipeline, "steps") else fitted_pipeline
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    if hasattr(est, "coef_"):
        return pd.Series(np.abs(est.coef_), index=FEATURE_COLS).sort_values(ascending=False)
    return pd.Series(dtype=float)
