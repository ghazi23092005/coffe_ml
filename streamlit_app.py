"""☕ Coffee Fermentation Lab — dashboard Streamlit
Jalankan:  streamlit run streamlit_app.py
"""
import io, os
import numpy as np
import pandas as pd
import joblib, torch
import streamlit as st
import matplotlib.pyplot as plt

import common as C
import sim_data
import pipeline as P
import plots as PL
import tabular_model as TM
import sequence_model as SM

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JSON = os.path.join(HERE, "fermentor_samples_simulated.json")
# override default berat (dipakai hanya untuk uji cepat)
DEF_EPOCHS = int(os.environ.get("COFFEE_EPOCHS", 80))
DEF_REPEATS = int(os.environ.get("COFFEE_REPEATS", 2))

st.set_page_config(page_title="Coffee Fermentation Lab", page_icon="☕", layout="wide")

# ======================================================================
# Tema & CSS
# ======================================================================
st.markdown("""
<style>
.block-container {padding-top: 3.6rem; max-width: 1350px;}
.hero {background: linear-gradient(135deg, #3B2314 0%, #6F4436 60%, #A9744F 100%);
       border-radius: 18px; padding: 26px 32px; margin-bottom: 18px; color: #FFF8E7;}
.hero h1 {color: #FFF8E7 !important; margin: 0; font-size: 2.1rem; padding: 0;}
.hero p {color: #EAD7C3 !important; margin: 6px 0 0 0; font-size: 1rem;}
.kpi {background: #FFFDF8; border: 1px solid #EAD7C3; border-left: 6px solid #A9744F;
      border-radius: 12px; padding: 14px 18px; height: 100%;}
.kpi-label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #8C5B3F;}
.kpi-value {font-size: 1.6rem; font-weight: 700; color: #3B2314; line-height: 1.25;}
.kpi-sub {font-size: .82rem; color: #6F4436;}
.kpi.win {border-left-color: #5B7F5B; background: #F6FAF2;}
.badge {display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 600; font-size: .85rem;}
.badge-good {background: #DDEBD3; color: #2F5D2F;}
.badge-bad {background: #F3D9D2; color: #8A3324;}
.note {background: #FFFDF8; border: 1px dashed #C69C6D; border-radius: 12px; padding: 12px 16px; color: #3B2314;}
h2, h3 {color: #3B2314 !important;}
[data-testid="stTabs"] button p {font-size: 1.02rem; font-weight: 600;}
[data-testid="stSidebar"] {border-right: 1px solid #D9B382;}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", win=False):
    return (f'<div class="kpi{" win" if win else ""}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def badge(pred):
    return f'<span class="badge {"badge-good" if pred == "Enak" else "badge-bad"}">{"☕ Enak" if pred == "Enak" else "Kurang"}</span>'


def empty_state(msg="Klik **▶ Jalankan training** di sidebar untuk melihat bagian ini."):
    st.markdown(f'<div class="note">🫘 {msg}</div>', unsafe_allow_html=True)


# ======================================================================
# Sidebar
# ======================================================================
sb = st.sidebar
sb.markdown("## ☕ Coffee Fermentation Lab")
sb.caption("Prediksi cita rasa dari sensor fermentasi — ML tabular vs deep learning sequence.")

sb.markdown("### 1 · Data")
source = sb.radio("Sumber data", ["Simulasi (sintetis)", "Upload data asli", "File contoh (test mode)"],
                  label_visibility="collapsed")
interval = sb.number_input("Interval sampling sensor (menit)", 1.0, 240.0, 30.0, 1.0,
                           help="Dipakai menghitung jam fermentasi bila data tidak punya kolom elapsed_h.")
n_sim, seed_sim, n_chunks, up_json, up_csv = 60, 0, 5, None, None
if source.startswith("Simulasi"):
    n_sim = sb.slider("Jumlah batch simulasi", 20, 120, 60, 5)
    seed_sim = sb.number_input("Seed simulasi", 0, 9999, 0)
elif source.startswith("Upload"):
    up_json = sb.file_uploader("Data sensor (.json)", type=["json"])
    up_csv = sb.file_uploader("Skor cupping (.csv: session,score) — opsional", type=["csv"])
else:
    n_chunks = sb.slider("Jumlah batch semu", 3, 10, 5)

sb.markdown("### 2 · Parameter training")
threshold = sb.slider("Batas skor 'enak'", 70.0, 90.0, 80.0, 0.5, help="Skor cupping ≥ batas ini dianggap enak (SCA specialty = 80).")
tab_models = sb.multiselect("Model tabular", ["Ridge", "Random Forest", "Gradient Boosting"],
                            default=["Ridge", "Random Forest", "Gradient Boosting"])
arch_labels = sb.multiselect("Model sequence (deep learning)", ["GRU", "1D-CNN"], default=["GRU", "1D-CNN"])
repeats = sb.slider("Ulangan cross-validation", 1, 5, DEF_REPEATS, help="5-fold × N ulangan. Batch < 10 otomatis Leave-One-Out.")
k_choice = sb.select_slider("Jumlah cluster", options=["Auto", 2, 3, 4, 5, 6], value="Auto",
                            help="Auto = dipilih via silhouette score.")

with sb.expander("Hyperparameter tabular"):
    rf_trees = st.slider("Random Forest · jumlah pohon", 50, 600, 300, 50)
    rf_depth = st.slider("Random Forest · kedalaman maks", 2, 12, 6)
    gb_trees = st.slider("Gradient Boosting · jumlah pohon", 50, 500, 150, 25)
    gb_depth = st.slider("Gradient Boosting · kedalaman", 1, 5, 2)
    gb_lr = st.select_slider("Gradient Boosting · learning rate", [0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
with sb.expander("Hyperparameter sequence"):
    epochs = st.slider("Epoch", 20, 300, DEF_EPOCHS, 10)
    seeds = st.slider("Ensemble seed", 1, 5, 2, help="Beberapa model dilatih dengan seed berbeda lalu dirata-rata.")
    lr = st.select_slider("Learning rate", [0.001, 0.003, 0.01], value=0.003)
    noise = st.slider("Augmentasi noise", 0.0, 0.3, 0.05, 0.01)

run_clicked = sb.button("▶ Jalankan training", type="primary", width="stretch")
sb.caption("Training di CPU biasa: ±1–5 menit tergantung jumlah batch dan epoch (GRU paling lama).")

# ======================================================================
# Muat data
# ======================================================================
@st.cache_data(show_spinner=False)
def _simulate(n, seed, interval):
    return sim_data.simulate_batches(n, seed=seed, interval_min=interval)


@st.cache_data(show_spinner=False)
def _load_json(raw: bytes, interval):
    return C.load_data(io.StringIO(raw.decode("utf-8")), interval)


df, scores, mode, mode_note, need_scores = None, None, "", "", False
if source.startswith("Simulasi"):
    df, scores = _simulate(n_sim, int(seed_sim), interval)
    mode = "Sintetis"
    mode_note = ("Data <b>sintetis</b> dari simulator (kurva pH/CO₂/O₂/suhu + skor cupping buatan). "
                 "Cocok untuk membangun & menguji model; hasilnya belum menggambarkan rasa kopi sungguhan.")
elif source.startswith("Upload"):
    if up_json is None:
        st.markdown('<div class="hero"><h1>☕ Coffee Fermentation Lab</h1><p>Unggah data sensor (.json) di sidebar, atau pilih sumber data lain.</p></div>', unsafe_allow_html=True)
        st.stop()
    df = _load_json(up_json.getvalue(), interval)
    if df["session"].nunique() < 2:
        df = C.make_pseudo_batches(df, 5)
        scores = pd.Series(np.random.default_rng(0).uniform(70, 88, df["session"].nunique()).round(1),
                           index=sorted(df["session"].unique()))
        mode, mode_note = "Test mode", "File hanya berisi 1 sesi → dipecah jadi batch semu, skor <b>ACAK</b>. Hanya untuk uji alur kode."
    elif up_csv is not None:
        scores = C.load_scores(io.StringIO(up_csv.getvalue().decode("utf-8")))
        mode, mode_note = "Data asli", "Data asli dengan skor cupping dari file CSV."
    else:
        mode, mode_note, need_scores = "Data asli", "Data asli — isi skor cupping tiap batch pada tabel di bawah (atau unggah CSV).", True
else:
    df = C.load_data(SAMPLE_JSON, interval)
    df = C.make_pseudo_batches(df, n_chunks)
    scores = pd.Series(np.random.default_rng(0).uniform(70, 88, n_chunks).round(1), index=range(1, n_chunks + 1))
    mode = "Test mode"
    mode_note = ("File contoh berisi <b>1 sesi</b> → dipecah jadi batch semu dengan skor <b>acak</b>. Ini hanya memastikan "
                 "alur kode berjalan; metrik model di sini tidak bermakna.")

st.markdown(f'<div class="hero"><h1>☕ Coffee Fermentation Lab</h1>'
            f'<p>Prediksi enak-tidaknya kopi dari sensor fermentasi &nbsp;·&nbsp; ML tabular vs deep learning sequence &nbsp;·&nbsp; '
            f'mode data: <b>{mode}</b></p></div>', unsafe_allow_html=True)
st.markdown(f'<div class="note">ℹ️ {mode_note}</div>', unsafe_allow_html=True)
st.write("")

if need_scores:
    with st.expander("✍️ Skor cupping per batch", expanded=True):
        sess = sorted(df["session"].unique())
        edited = st.data_editor(pd.DataFrame({"session": sess, "score": np.nan}), hide_index=True,
                                disabled=["session"], key="score_editor", width="stretch")
        scores = pd.Series(edited["score"].values, index=edited["session"].values)
        st.caption("Isi skor (0–100) untuk batch yang sudah di-cupping. Batch tanpa skor dilewati.")

batches, ids, y = P.prepare(df, scores)
n = len(ids)
if n < 2:
    st.warning("Butuh minimal 2 batch berlabel skor cupping untuk training.")
F_all = TM.build_feature_table(batches) if n else pd.DataFrame()
X_all, S_all, _ = SM.batches_to_tensor(batches) if n else (None, None, None)

# ======================================================================
# Training
# ======================================================================
cfg = {"threshold": threshold, "repeats": repeats, "epochs": epochs, "seeds": seeds, "lr": lr, "noise": noise,
       "k": None if k_choice == "Auto" else int(k_choice),
       "tab_models": tab_models, "archs": [{"GRU": "gru", "1D-CNN": "cnn"}[a] for a in arch_labels],
       "tab_params": {"rf_trees": rf_trees, "rf_depth": rf_depth, "gb_trees": gb_trees, "gb_depth": gb_depth, "gb_lr": gb_lr}}
sig = hash((tuple(ids), tuple(np.round(y, 3))))

if run_clicked and n >= 2:
    if not tab_models and not arch_labels:
        st.error("Pilih minimal satu model (tabular atau sequence).")
    else:
        bar, msg = st.progress(0.0), st.empty()
        def _cb(frac, text):
            bar.progress(min(frac, 1.0)); msg.markdown(f"⏳ **{text}**")
        st.session_state["exp"] = P.run_experiment(batches, ids, y, cfg, progress=_cb)
        st.session_state["exp"]["sig"] = sig
        bar.empty(); msg.empty()
        st.toast("Training selesai ☕", icon="✅")

exp = st.session_state.get("exp")
if exp is not None and exp.get("sig") != sig:
    st.warning("Data/skor berubah sejak training terakhir — hasil di bawah dari training sebelumnya. Jalankan training ulang.")

# ======================================================================
# Tabs
# ======================================================================
t_data, t_param, t_cmp, t_train, t_clu, t_pred = st.tabs(
    ["📊 Data & Tren", "⚙️ Parameter", "🏆 Perbandingan", "🧪 Hasil Training", "🧩 Cluster", "🔮 Prediksi"])

# ---------------------------------------------------------------- Data
with t_data:
    if n:
        tot = int(sum(len(g) for g in batches.values()))
        durs = [g["elapsed_h"].iloc[-1] - g["elapsed_h"].iloc[0] for g in batches.values()]
        good = int((y >= threshold).sum())
        c = st.columns(5)
        c[0].markdown(kpi("Batch berlabel", n, f"dari {df['session'].nunique()} sesi"), unsafe_allow_html=True)
        c[1].markdown(kpi("Total sampel sensor", f"{tot:,}", "titik pengukuran"), unsafe_allow_html=True)
        c[2].markdown(kpi("Durasi rata-rata", f"{np.mean(durs):.0f} jam", f"{min(durs):.0f}–{max(durs):.0f} jam"), unsafe_allow_html=True)
        c[3].markdown(kpi("Skor cupping", f"{y.mean():.1f}", f"rentang {y.min():.0f}–{y.max():.0f}"), unsafe_allow_html=True)
        c[4].markdown(kpi("Batch 'enak'", f"{good} / {n}", f"skor ≥ {threshold:.0f}"), unsafe_allow_html=True)
        st.write("")
        st.subheader("Tren sensor per batch")
        pick = st.multiselect("Pilih batch untuk ditampilkan", ids, default=ids[:4], key="trend_pick")
        if pick:
            show(PL.sensor_trends(batches, pick))
        a, b = st.columns(2)
        with a:
            show(PL.ph_by_quality(X_all, y, threshold))
        with b:
            st.markdown("**Sebaran skor cupping**")
            hist, edges = np.histogram(y, bins=np.arange(np.floor(y.min() / 2) * 2, y.max() + 2.1, 2))
            st.bar_chart(pd.DataFrame({"Jumlah batch": hist}, index=[f"{e:.0f}" for e in edges[:-1]]), height=330)
        with st.expander("Tabel fitur per batch (input model tabular)"):
            ft = F_all.copy(); ft.insert(0, "skor_cupping", y)
            st.dataframe(ft.round(3), width="stretch")
        with st.expander("Data sensor mentah"):
            st.dataframe(df.head(500), width="stretch")

# ---------------------------------------------------------------- Parameter
with t_param:
    st.subheader("Konfigurasi eksperimen")
    n_lab = n
    cv_txt = "Leave-One-Out" if n_lab < 10 else f"5-fold × {repeats} ulangan"
    c = st.columns(4)
    c[0].markdown(kpi("Validasi", cv_txt, "split sama untuk semua model"), unsafe_allow_html=True)
    c[1].markdown(kpi("Batas 'enak'", f"≥ {threshold:.0f}", "klasifikasi dari skor prediksi"), unsafe_allow_html=True)
    c[2].markdown(kpi("Fitur tabular", len(TM.FEATURE_COLS), "ringkasan per batch"), unsafe_allow_html=True)
    c[3].markdown(kpi("Input sequence", f"{SM.T_STEPS} × {SM.N_CHANNELS}", "langkah waktu × kanal"), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Model tabular")
        rows = [{"Model": "Baseline", "Hyperparameter": "prediksi = rata-rata skor training"},
                {"Model": "Ridge", "Hyperparameter": "alpha dipilih otomatis (RidgeCV, 0.01–1000), fitur distandarisasi"},
                {"Model": "Random Forest", "Hyperparameter": f"{rf_trees} pohon, depth ≤ {rf_depth}, min_samples_leaf 2"},
                {"Model": "Gradient Boosting", "Hyperparameter": f"{gb_trees} pohon, depth {gb_depth}, lr {gb_lr}, subsample 0.8"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Scaler ada di dalam Pipeline → tidak bocor antar fold.")
    with b:
        st.markdown("#### Model sequence")
        rows = [{"Komponen": "Input", "Nilai": f"pH, O₂, CO₂, suhu, RH + dpH/dt + dCO₂/dt · {SM.T_STEPS} langkah (waktu 0–1)"},
                {"Komponen": "Fitur statis", "Nilai": "durasi fermentasi (jam)"},
                {"Komponen": "GRU", "Nilai": "1 layer, hidden 32 → MLP 32 → skor"},
                {"Komponen": "1D-CNN", "Nilai": "Conv(16,k5) → Conv(32,k5,s2) → mean+max pool → MLP"},
                {"Komponen": "Training", "Nilai": f"{epochs} epoch · AdamW lr {lr} · Huber loss · ensemble {seeds} seed"},
                {"Komponen": "Augmentasi", "Nilai": f"noise {noise} + drift kalibrasi 0.1 (unit ter-normalisasi)"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("#### Kamus fitur tabular")
    st.dataframe(pd.DataFrame({"Fitur": list(TM.FEATURE_INFO), "Arti": list(TM.FEATURE_INFO.values())}),
                 hide_index=True, width="stretch")
    with st.expander("Cara membaca metrik"):
        st.markdown("""
- **MAE** — rata-rata selisih |prediksi − skor asli| (poin cupping). Makin kecil makin baik. ±std = variasi antar ulangan CV.
- **R²** — 1 = sempurna, 0 = sama dengan menebak rata-rata, negatif = lebih buruk dari menebak rata-rata.
- **Acc / F1 enak** — akurasi dan F1 saat skor prediksi diubah jadi label *enak/kurang* pada batas yang dipilih.
- **Baseline** — model "tebak rata-rata". Model yang layak harus jelas mengalahkannya.
""")

# ---------------------------------------------------------------- Perbandingan
with t_cmp:
    if exp is None:
        empty_state()
    else:
        res, R = exp["results"], exp["results"].set_index("Model")
        base = R.loc["Baseline (rata-rata)", "MAE"]
        thr_e = exp["cfg"]["threshold"]
        st.subheader("Siapa yang lebih baik?")
        c = st.columns(3)
        winner = None
        if exp["best_tab"] and exp["best_seq"]:
            dt, ds = R.loc[exp["best_tab"]], R.loc[exp["best_seq"]]
            winner = "tab" if dt.MAE <= ds.MAE else "seq"
            tol = max(dt.MAE_std, ds.MAE_std)
        for col, key, title, name in [(c[0], "tab", "Terbaik · Tabular", exp["best_tab"]), (c[1], "seq", "Terbaik · Sequence", exp["best_seq"])]:
            if name:
                r = R.loc[name]
                col.markdown(kpi(title, name, f"MAE {r.MAE:.2f} · R² {r.R2:.2f} · {100*(1-r.MAE/base):.0f}% lebih baik dari baseline",
                                 win=(winner == key)), unsafe_allow_html=True)
        c[2].markdown(kpi("Baseline (tebak rata-rata)", f"MAE {base:.2f}", "batas bawah yang harus dikalahkan"), unsafe_allow_html=True)
        if winner:
            gap = abs(dt.MAE - ds.MAE)
            who = exp["best_tab"] if winner == "tab" else exp["best_seq"]
            if tol == 0:
                st.markdown(f'<div class="note">📏 Selisih MAE {gap:.2f} poin untuk <b>{who}</b>. Variasi antar-ulangan CV belum bisa diukur '
                            f'(ulangan = 1) — naikkan "Ulangan cross-validation" agar tahu apakah selisih ini bermakna.</div>', unsafe_allow_html=True)
            elif gap <= tol:
                st.markdown(f'<div class="note">⚖️ Selisih MAE ({gap:.2f}) masih dalam variasi antar ulangan CV (±{tol:.2f}) → '
                            f'kedua pendekatan <b>setara secara statistik</b> pada data ini. Unggul tipis: <b>{who}</b>.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="note">🏆 <b>{who}</b> unggul dengan selisih MAE {gap:.2f} poin (lebih besar dari variasi CV ±{tol:.2f}).</div>', unsafe_allow_html=True)
        if exp["n"] < 15:
            st.warning(f"Baru {exp['n']} batch — hasil masih perkiraan awal; model kompleks (terutama deep learning) butuh lebih banyak batch.")
        if mode != "Data asli":
            st.caption(f"⚠️ Mode data **{mode}**: perbandingan ini menguji alur, belum membuktikan pendekatan mana yang lebih baik untuk kopi sungguhan.")

        st.write("")
        cols = ["Pendekatan", "Model", "MAE", "MAE_std", "RMSE", "R2", "Acc_enak", "F1_enak"]
        styled = (res[cols].style.format({c_: "{:.2f}" for c_ in cols[2:]})
                  .highlight_min(subset=["MAE", "RMSE"], color="#EAD7C3").highlight_max(subset=["R2", "Acc_enak", "F1_enak"], color="#EAD7C3"))
        st.dataframe(styled, hide_index=True, width="stretch")
        a, b = st.columns(2)
        with a:
            show(PL.mae_bars(res))
        with b:
            if len(exp["importance"]):
                show(PL.importance_bar(exp["importance"]))
        a, b = st.columns(2)
        for col, name in [(a, exp["best_tab"]), (b, exp["best_seq"])]:
            if name:
                with col:
                    show(PL.pred_vs_actual(exp["y"], exp["oof"][name], name, R.loc[name, "MAE"], R.loc[name, "R2"], thr_e))

# ---------------------------------------------------------------- Hasil training
with t_train:
    if exp is None:
        empty_state()
    else:
        thr_e = exp["cfg"]["threshold"]
        st.subheader("Ringkasan training")
        c = st.columns(4)
        c[0].markdown(kpi("Batch dipakai", exp["n"], "training model final"), unsafe_allow_html=True)
        c[1].markdown(kpi("Validasi", exp["cv_desc"], "out-of-fold"), unsafe_allow_html=True)
        c[2].markdown(kpi("Model tabular final", exp["best_tab"] or "—", "pemenang CV"), unsafe_allow_html=True)
        c[3].markdown(kpi("Model sequence final", exp["best_seq"] or "—", "pemenang CV"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            if exp["seq_bundle"] is not None and exp["seq_bundle"].get("history"):
                show(PL.loss_curve(exp["seq_bundle"]["history"], exp["best_seq"]))
        with b:
            st.markdown("**Galat per batch (out-of-fold) — batch yang paling sulit ditebak**")
            err = pd.DataFrame({"Batch": exp["ids"], "Skor asli": exp["y"]})
            for nm, tag in [(exp["best_tab"], "Tabular"), (exp["best_seq"], "Sequence")]:
                if nm:
                    err[f"Prediksi {tag}"] = exp["oof"][nm]; err[f"|Galat| {tag}"] = np.abs(exp["oof"][nm] - exp["y"])
            sort_col = [c_ for c_ in err.columns if c_.startswith("|Galat|")][0]
            st.dataframe(err.sort_values(sort_col, ascending=False).round(2), hide_index=True, height=330, width="stretch")
        st.markdown("#### Unduh hasil & model")
        d = st.columns(4)
        d[0].download_button("⬇️ comparison.csv", exp["results"].to_csv(index=False), "comparison.csv", "text/csv", width="stretch")
        d[1].download_button("⬇️ galat_per_batch.csv", err.to_csv(index=False), "galat_per_batch.csv", "text/csv", width="stretch")
        if exp["tab_model"] is not None:
            buf = io.BytesIO(); joblib.dump({"model": exp["tab_model"], "name": exp["best_tab"], "features": TM.FEATURE_COLS, "threshold": thr_e}, buf)
            d[2].download_button("⬇️ tabular_best.joblib", buf.getvalue(), "tabular_best.joblib", width="stretch")
        if exp["seq_bundle"] is not None:
            buf = io.BytesIO(); torch.save(exp["seq_bundle"], buf)
            d[3].download_button("⬇️ sequence_best.pt", buf.getvalue(), "sequence_best.pt", width="stretch")
        st.caption("Model tersimpan bisa dipakai lewat `python predict_new.py --data batch_baru.json --models <folder>`.")

# ---------------------------------------------------------------- Cluster
with t_clu:
    if exp is None:
        empty_state()
    elif exp["k_tab"] is None:
        st.info("Clustering butuh minimal 4 batch.")
    else:
        thr_e, cl = exp["cfg"]["threshold"], exp["clusters"]
        c = st.columns(3)
        c[0].markdown(kpi("Cluster tabular", f"k = {exp['k_tab']}", "fitur ringkasan sensor"), unsafe_allow_html=True)
        c[1].markdown(kpi("Cluster sequence", f"k = {exp['k_seq']}", "embedding GRU"), unsafe_allow_html=True)
        c[2].markdown(kpi("Kesamaan (ARI)", f"{exp['ari']:.2f}", "1 = identik, ≈0 = acak"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        for col, key, Pk, title, Xc in [(a, "cluster_tabular", "pca_tab", "Cluster dari fitur tabular", None),
                                        (b, "cluster_sequence", "pca_seq", "Cluster dari embedding GRU", None)]:
            with col:
                st.markdown(f"#### {title}")
                show(PL.cluster_scatter(exp[Pk], cl[key].values, exp["y"], title))
                st.dataframe(P.cluster_summary(cl, key, thr_e), width="stretch")
                show(PL.cluster_mean_curves(exp["X"], cl[key].values))
        st.markdown('<div class="note">ℹ️ Embedding GRU dilatih memakai skor cupping, jadi cluster-nya cenderung mengikuti skor '
                    '(semi-supervised). Cluster tabular murni dari pola sensor tanpa melihat skor — lebih jujur untuk menemukan '
                    '"jenis fermentasi", tapi belum tentu selaras dengan rasa.</div>', unsafe_allow_html=True)
        with st.expander("Tabel penugasan cluster per batch"):
            st.dataframe(cl, hide_index=True, width="stretch")
        st.download_button("⬇️ clusters.csv", cl.to_csv(index=False), "clusters.csv", "text/csv")

# ---------------------------------------------------------------- Prediksi
with t_pred:
    if exp is None:
        empty_state()
    else:
        st.subheader("Prediksi cita rasa batch")
        how = st.radio("Sumber batch", ["Upload batch baru (.json)", "Pilih dari data yang ada"], horizontal=True, index=1)
        new_batches, is_train = None, False
        if how.startswith("Upload"):
            a, b = st.columns([3, 1])
            up = a.file_uploader("Data sensor batch baru (format sama dengan data training)", type=["json"], key="new_up")
            b.write(""); b.write("")
            sample = pd.concat([batches[s] for s in list(batches)[:2]]).to_json(orient="records") if n else "[]"
            b.download_button("⬇️ Contoh format", sample, "contoh_batch_baru.json", "application/json", width="stretch")
            if up is not None:
                new_batches = C.split_batches(_load_json(up.getvalue(), interval))
        else:
            sel = st.multiselect("Pilih batch", ids, default=ids[:3])
            if sel:
                new_batches, is_train = {s: batches[s] for s in sel}, True
                st.caption("⚠️ Batch ini dipakai saat training, jadi prediksinya optimistis. Untuk uji jujur, gunakan batch baru.")
        if new_batches:
            pr = P.predict_batches(exp, new_batches)
            pred_cols = [c_ for c_ in pr.columns if c_.startswith(("Tabular", "Sequence"))]
            for s, r in pr.iterrows():
                c = st.columns([1, 1.2, 1.2, 1.2, 1.4])
                c[0].markdown(kpi("Batch", s, f"{len(new_batches[s])} sampel"), unsafe_allow_html=True)
                for i, pc in enumerate(pred_cols):
                    c[1 + i].markdown(kpi(pc.replace(" · ", " — "), f"{r[pc]:.1f}", "skor cupping prediksi"), unsafe_allow_html=True)
                c[3].markdown(kpi("Rata-rata kedua model", f"{r['Rata-rata']:.1f}", f"selisih {r['Selisih model']:.1f} poin"), unsafe_allow_html=True)
                c[4].markdown(kpi("Prediksi", badge(r["Prediksi"]), f"batas ≥ {exp['cfg']['threshold']:.0f}"), unsafe_allow_html=True)
                if r["Selisih model"] > 3:
                    st.caption(f"⚠️ Batch {s}: kedua model berbeda > 3 poin — perlakukan prediksi ini dengan hati-hati.")
                st.write("")
            show(PL.predicted_curves(new_batches, list(new_batches)[:8]))
            with st.expander("Tabel prediksi"):
                st.dataframe(pr.round(2), width="stretch")
            st.download_button("⬇️ prediksi.csv", pr.round(2).to_csv(), "prediksi.csv", "text/csv")
        elif how.startswith("Upload"):
            st.caption("Unggah file JSON batch baru untuk mendapat prediksi dari kedua model.")

st.markdown("---")
st.caption("☕ Coffee Fermentation Lab · ML tabular (scikit-learn) + deep learning sequence (PyTorch) · "
           "Data sintetis/test mode hanya untuk membangun & menguji model.")
