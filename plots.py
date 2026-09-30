"""Semua grafik bertema kopi (matplotlib). Dipakai CLI dan dashboard."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ESPRESSO, BROWN, MOCHA, LATTE, CREAM, BG = "#3B2314", "#6F4436", "#A9744F", "#C69C6D", "#EAD7C3", "#FFF8E7"
PALETTE = [BROWN, "#D08C3C", "#5B7F5B", "#7A6A9A", "#B5533C", "#2F6F8F"]
SENSORS = [("ph_value", "pH"), ("scd41_co2_ppm", "CO₂ (ppm)"), ("o2_percent", "O₂ (%)"),
           ("scd41_temp_c", "Suhu (°C)"), ("scd41_rh_percent", "Kelembapan (%)")]

plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=PALETTE), "axes.edgecolor": ESPRESSO, "axes.labelcolor": ESPRESSO,
    "xtick.color": ESPRESSO, "ytick.color": ESPRESSO, "text.color": ESPRESSO,
    "figure.facecolor": BG, "axes.facecolor": "#FFFDF8", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": CREAM, "grid.linewidth": .8, "font.size": 9.5,
})


def sensor_trends(batches: dict, sessions: list, x="elapsed_h"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (col, label) in zip(axes.flat, SENSORS):
        for i, s in enumerate(sessions):
            g = batches[s]
            ax.plot(g[x], g[col], lw=1.5, color=PALETTE[i % len(PALETTE)], label=f"Batch {s}")
        ax.set_title(label, fontweight="bold"); ax.set_xlabel("Jam fermentasi")
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False) if len(sessions) <= 10 else None
    fig.tight_layout(); return fig


def ph_by_quality(X, y, thr, T=None):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    grid = np.linspace(0, 1, X.shape[1])
    for i in range(len(y)):
        ax.plot(grid, X[i, :, 1], color=MOCHA if y[i] >= thr else "#9A9A9A", alpha=.45, lw=1)
    ax.plot([], [], color=MOCHA, label=f"Enak (≥{thr:.0f})"); ax.plot([], [], color="#9A9A9A", label="Kurang")
    ax.set_xlabel("Progres fermentasi (0–1)"); ax.set_ylabel("pH"); ax.set_title("Kurva pH: enak vs kurang", fontweight="bold")
    ax.legend(frameon=False); fig.tight_layout(); return fig


def mae_bars(res):
    r = res.sort_values("MAE", ascending=False)
    colors = [{"Pembanding": "#B8B8B8", "Tabular": MOCHA}.get(p, BROWN) for p in r["Pendekatan"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(r["Model"], r["MAE"], xerr=r["MAE_std"], color=colors, edgecolor=ESPRESSO, capsize=3)
    for i, v in enumerate(r["MAE"]): ax.text(v + .05, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("MAE (poin skor cupping, lebih kecil = lebih baik)"); ax.set_title("Perbandingan error (CV)", fontweight="bold")
    ax.grid(axis="y", visible=False)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#B8B8B8", MOCHA, BROWN]]
    ax.legend(h, ["Pembanding", "Tabular", "Sequence"], frameon=False, loc="lower right"); fig.tight_layout(); return fig


def pred_vs_actual(y, oof, title, mae, r2, thr):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ok = (y >= thr) == (oof >= thr)
    ax.scatter(y[ok], oof[ok], c=MOCHA, edgecolor=ESPRESSO, s=48, label="Kelas benar")
    ax.scatter(y[~ok], oof[~ok], c="#D9534F", edgecolor=ESPRESSO, s=48, marker="X", label="Kelas salah")
    lo, hi = min(y.min(), np.nanmin(oof)) - 2, max(y.max(), np.nanmax(oof)) + 2
    ax.plot([lo, hi], [lo, hi], "--", c="gray", lw=1); ax.axvline(thr, c="gray", lw=.6); ax.axhline(thr, c="gray", lw=.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Skor cupping asli"); ax.set_ylabel("Prediksi (out-of-fold)")
    ax.set_title(f"{title}\nMAE {mae:.2f} · R² {r2:.2f}", fontweight="bold"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); return fig


def importance_bar(imp, top=8):
    t = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(t.index, t.values, color=MOCHA, edgecolor=ESPRESSO); ax.grid(axis="y", visible=False)
    ax.set_title("Fitur sensor paling berpengaruh", fontweight="bold"); fig.tight_layout(); return fig


def loss_curve(history, arch):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(np.arange(1, len(history) + 1), history, color=BROWN, lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (Huber)"); ax.set_yscale("log")
    ax.set_title(f"Kurva training {arch} (model final)", fontweight="bold"); fig.tight_layout(); return fig


def cluster_scatter(P, labels, y, title):
    """Warna = cluster, ukuran titik = skor cupping (makin besar makin tinggi)."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    size = 35 + 170 * (y - y.min()) / max(np.ptp(y), 1e-9)
    for l in sorted(set(labels)):
        m = labels == l
        ax.scatter(P[m, 0], P[m, 1], s=size[m], color=PALETTE[l % len(PALETTE)], alpha=.75, edgecolor=ESPRESSO, linewidth=.8, label=f"Cluster {l}")
    ax.legend(frameon=False, fontsize=8, title="ukuran = skor cupping", title_fontsize=8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title, fontweight="bold"); fig.tight_layout(); return fig


def cluster_mean_curves(X, labels, channel=1, name="pH"):
    fig, ax = plt.subplots(figsize=(5.4, 4.4)); grid = np.linspace(0, 1, X.shape[1])
    for l in sorted(set(labels)):
        m = X[labels == l, :, channel]
        ax.plot(grid, m.mean(0), lw=2.4, color=PALETTE[l % len(PALETTE)], label=f"Cluster {l} (n={len(m)})")
        ax.fill_between(grid, m.mean(0) - m.std(0), m.mean(0) + m.std(0), color=PALETTE[l % len(PALETTE)], alpha=.15)
    ax.set_xlabel("Progres fermentasi (0–1)"); ax.set_ylabel(name); ax.set_title(f"Rata-rata kurva {name} per cluster", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); return fig


def predicted_curves(batches: dict, sessions: list):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (col, label) in zip(axes, [SENSORS[0], SENSORS[1], SENSORS[3]]):
        for i, s in enumerate(sessions):
            ax.plot(batches[s]["elapsed_h"], batches[s][col], lw=1.6, color=PALETTE[i % len(PALETTE)], label=f"Batch {s}")
        ax.set_title(label, fontweight="bold"); ax.set_xlabel("Jam fermentasi")
    axes[0].legend(frameon=False, fontsize=8); fig.tight_layout(); return fig
