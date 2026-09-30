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
DEF_EPOCHS = int(os.environ.get("COFFEE_EPOCHS", 80))
DEF_REPEATS = int(os.environ.get("COFFEE_REPEATS", 2))

st.set_page_config(page_title="Coffee Fermentation Lab", page_icon="☕", layout="wide")

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
    return f'<span class="badge {"badge-good" if pred == "Enak" else "badge-bad"}">{"Good" if pred == "Enak" else "Not Good"}</span>'


def empty_state(msg="Click **▶ Run training** in the sidebar to see this section."):
    st.markdown(f'<div class="note">{msg}</div>', unsafe_allow_html=True)


sb = st.sidebar
sb.markdown("## ☕ Coffee Fermentation Lab")
sb.caption("Taste prediction from fermentation sensors : Tabular ML vs Deep Learning Sequence.")

sb.markdown("### 1 · Data")
source = sb.radio("Data source", ["Simulated (synthetic)", "Upload real data", "Sample file (test mode)"],
                  label_visibility="collapsed")
interval = sb.number_input("Sensor sampling interval (minutes)", 1.0, 240.0, 30.0, 1.0,
                           help="Used to compute fermentation hours when the data has no elapsed_h column.")
n_sim, seed_sim, n_chunks, up_json, up_csv = 60, 0, 5, None, None
if source.startswith("Simulated"):
    n_sim = sb.slider("Number of simulated batches", 20, 120, 60, 5)
    seed_sim = sb.number_input("Simulation seed", 0, 9999, 0)
elif source.startswith("Upload"):
    up_json = sb.file_uploader("Sensor data (.json)", type=["json"])
    up_csv = sb.file_uploader("Cupping scores (.csv: session,score) — optional", type=["csv"])
else:
    n_chunks = sb.slider("Number of pseudo batches", 3, 10, 5)

sb.markdown("### 2 · Training parameters")
threshold = sb.slider("'Good' score threshold", 70.0, 90.0, 80.0, 0.5, help="A cupping score ≥ this threshold is considered good (SCA specialty = 80).")
tab_models = sb.multiselect("Tabular models", ["Ridge", "Random Forest", "Gradient Boosting"],
                            default=["Ridge", "Random Forest", "Gradient Boosting"])
arch_labels = sb.multiselect("Sequence models (deep learning)", ["GRU", "1D-CNN"], default=["GRU", "1D-CNN"])
repeats = sb.slider("Cross-validation repeats", 1, 5, DEF_REPEATS, help="5-fold × N repeats. Fewer than 10 batches automatically uses Leave-One-Out.")
k_choice = sb.select_slider("Number of clusters", options=["Auto", 2, 3, 4, 5, 6], value="Auto",
                            help="Auto = selected via silhouette score.")

with sb.expander("Tabular hyperparameters"):
    rf_trees = st.slider("Random Forest · number of trees", 50, 600, 300, 50)
    rf_depth = st.slider("Random Forest · max depth", 2, 12, 6)
    gb_trees = st.slider("Gradient Boosting · number of trees", 50, 500, 150, 25)
    gb_depth = st.slider("Gradient Boosting · depth", 1, 5, 2)
    gb_lr = st.select_slider("Gradient Boosting · learning rate", [0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
with sb.expander("Sequence hyperparameters"):
    epochs = st.slider("Epochs", 20, 300, DEF_EPOCHS, 10)
    seeds = st.slider("Ensemble seeds", 1, 5, 2, help="Several models are trained with different seeds and then averaged.")
    lr = st.select_slider("Learning rate", [0.001, 0.003, 0.01], value=0.003)
    noise = st.slider("Noise augmentation", 0.0, 0.3, 0.05, 0.01)

run_clicked = sb.button("▶ Run training", type="primary", width="stretch")
sb.caption("Estimated time: 1–5 minutes depending on the number of batches and epochs.")

@st.cache_data(show_spinner=False)
def _simulate(n, seed, interval):
    return sim_data.simulate_batches(n, seed=seed, interval_min=interval)


@st.cache_data(show_spinner=False)
def _load_json(raw: bytes, interval):
    return C.load_data(io.StringIO(raw.decode("utf-8")), interval)


df, scores, mode, mode_note, need_scores = None, None, "", "", False
if source.startswith("Simulated"):
    df, scores = _simulate(n_sim, int(seed_sim), interval)
    mode = "Synthetic"
    mode_note = ("<b>Synthetic</b> data from the simulator (pH/CO₂/O₂/temperature curves with generated cupping scores). "
                 "Used to build and test the model.")
elif source.startswith("Upload"):
    if up_json is None:
        st.markdown('<div class="hero"><h1>☕ Coffee Fermentation Lab</h1><p>Upload sensor data (.json) in the sidebar, or select another data source.</p></div>', unsafe_allow_html=True)
        st.stop()
    df = _load_json(up_json.getvalue(), interval)
    if df["session"].nunique() < 2:
        df = C.make_pseudo_batches(df, 5)
        scores = pd.Series(np.random.default_rng(0).uniform(70, 88, df["session"].nunique()).round(1),
                           index=sorted(df["session"].unique()))
        mode, mode_note = "Test mode", "A single session is split into pseudo batches with <b>random</b> scores. For code-flow testing only."
    elif up_csv is not None:
        scores = C.load_scores(io.StringIO(up_csv.getvalue().decode("utf-8")))
        mode, mode_note = "Real data", "Real data with cupping scores from the CSV file."
    else:
        mode, mode_note, need_scores = "Real data", "Real data — enter the cupping score for each batch in the table below (or upload a CSV).", True
else:
    df = C.load_data(SAMPLE_JSON, interval)
    df = C.make_pseudo_batches(df, n_chunks)
    scores = pd.Series(np.random.default_rng(0).uniform(70, 88, n_chunks).round(1), index=range(1, n_chunks + 1))
    mode = "Test mode"
    mode_note = ("The sample file contains <b>one session</b> → split into pseudo batches with <b>random</b> scores. This only verifies "
                 "that the code flow runs; model metrics here are not meaningful.")

st.markdown(f'<div class="hero"><h1>☕ Coffee Fermentation Lab</h1>'
            f'<p>Coffee taste prediction from fermentation sensors &nbsp;·&nbsp; Tabular ML vs deep learning sequence &nbsp;·&nbsp; '
            f'data mode: <b>{mode}</b></p></div>', unsafe_allow_html=True)
st.markdown(f'<div class="note">{mode_note}</div>', unsafe_allow_html=True)
st.write("")

if need_scores:
    with st.expander("Cupping score per batch", expanded=True):
        sess = sorted(df["session"].unique())
        edited = st.data_editor(pd.DataFrame({"session": sess, "score": np.nan}), hide_index=True,
                                disabled=["session"], key="score_editor", width="stretch")
        scores = pd.Series(edited["score"].values, index=edited["session"].values)
        st.caption("Enter a score (0–100) for each batch that has been cupped. Batches without a score are skipped.")

batches, ids, y = P.prepare(df, scores)
n = len(ids)
if n < 2:
    st.warning("At least 2 labeled batches are required for training.")
F_all = TM.build_feature_table(batches) if n else pd.DataFrame()
X_all, S_all, _ = SM.batches_to_tensor(batches) if n else (None, None, None)

cfg = {"threshold": threshold, "repeats": repeats, "epochs": epochs, "seeds": seeds, "lr": lr, "noise": noise,
       "k": None if k_choice == "Auto" else int(k_choice),
       "tab_models": tab_models, "archs": [{"GRU": "gru", "1D-CNN": "cnn"}[a] for a in arch_labels],
       "tab_params": {"rf_trees": rf_trees, "rf_depth": rf_depth, "gb_trees": gb_trees, "gb_depth": gb_depth, "gb_lr": gb_lr}}
sig = hash((tuple(ids), tuple(np.round(y, 3))))

if run_clicked and n >= 2:
    if not tab_models and not arch_labels:
        st.error("Select at least one model (tabular or sequence).")
    else:
        bar, msg = st.progress(0.0), st.empty()
        def _cb(frac, text):
            bar.progress(min(frac, 1.0)); msg.markdown(f"**{text}**")
        st.session_state["exp"] = P.run_experiment(batches, ids, y, cfg, progress=_cb)
        st.session_state["exp"]["sig"] = sig
        bar.empty(); msg.empty()
        st.toast("Training complete")

exp = st.session_state.get("exp")
if exp is not None and exp.get("sig") != sig:
    st.warning("Data or scores have changed since the last training run. The results below are from the previous run. Please retrain.")

t_data, t_param, t_cmp, t_train, t_clu, t_pred = st.tabs(
    ["Data & Trends", "Parameters", "Comparison", "Training Results", "Clusters", "Prediction"])

with t_data:
    if n:
        tot = int(sum(len(g) for g in batches.values()))
        durs = [g["elapsed_h"].iloc[-1] - g["elapsed_h"].iloc[0] for g in batches.values()]
        good = int((y >= threshold).sum())
        c = st.columns(5)
        c[0].markdown(kpi("Labeled batches", n, f"of {df['session'].nunique()} sessions"), unsafe_allow_html=True)
        c[1].markdown(kpi("Total sensor samples", f"{tot:,}", "measurement points"), unsafe_allow_html=True)
        c[2].markdown(kpi("Average duration", f"{np.mean(durs):.0f} hours", f"{min(durs):.0f}–{max(durs):.0f} hours"), unsafe_allow_html=True)
        c[3].markdown(kpi("Cupping score", f"{y.mean():.1f}", f"range {y.min():.0f}–{y.max():.0f}"), unsafe_allow_html=True)
        c[4].markdown(kpi("'Good' batches", f"{good} / {n}", f"score ≥ {threshold:.0f}"), unsafe_allow_html=True)
        st.write("")
        st.subheader("Sensor trends per batch")
        pick = st.multiselect("Select batches to display", ids, default=ids[:4], key="trend_pick")
        if pick:
            show(PL.sensor_trends(batches, pick))
        a, b = st.columns(2)
        with a:
            show(PL.ph_by_quality(X_all, y, threshold))
        with b:
            st.markdown("**Cupping score distribution**")
            hist, edges = np.histogram(y, bins=np.arange(np.floor(y.min() / 2) * 2, y.max() + 2.1, 2))
            st.bar_chart(pd.DataFrame({"Number of batches": hist}, index=[f"{e:.0f}" for e in edges[:-1]]), height=330)
        with st.expander("Feature table per batch (tabular model input)"):
            ft = F_all.copy(); ft.insert(0, "cupping_score", y)
            st.dataframe(ft.round(3), width="stretch")
        with st.expander("Raw sensor data"):
            st.dataframe(df.head(500), width="stretch")

with t_param:
    st.subheader("Experiment configuration")
    n_lab = n
    cv_txt = "Leave-One-Out" if n_lab < 10 else f"5-fold × {repeats} repeats"
    c = st.columns(4)
    c[0].markdown(kpi("Validation", cv_txt, "same split across all models"), unsafe_allow_html=True)
    c[1].markdown(kpi("'Good' threshold", f"≥ {threshold:.0f}", "classification from predicted score"), unsafe_allow_html=True)
    c[2].markdown(kpi("Tabular features", len(TM.FEATURE_COLS), "per-batch summaries"), unsafe_allow_html=True)
    c[3].markdown(kpi("Sequence input", f"{SM.T_STEPS} × {SM.N_CHANNELS}", "time steps × channels"), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Tabular models")
        rows = [{"Model": "Baseline", "Hyperparameter": "prediction = mean training score"},
                {"Model": "Ridge", "Hyperparameter": "alpha selected automatically (RidgeCV, 0.01–1000), features standardized"},
                {"Model": "Random Forest", "Hyperparameter": f"{rf_trees} trees, depth ≤ {rf_depth}, min_samples_leaf 2"},
                {"Model": "Gradient Boosting", "Hyperparameter": f"{gb_trees} trees, depth {gb_depth}, lr {gb_lr}, subsample 0.8"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("The scaler lives inside the Pipeline, so there is no leakage between folds.")
    with b:
        st.markdown("#### Sequence models")
        rows = [{"Component": "Input", "Value": f"pH, O₂, CO₂, temperature, RH + dpH/dt + dCO₂/dt · {SM.T_STEPS} steps (time 0–1)"},
                {"Component": "Static features", "Value": "fermentation duration (hours)"},
                {"Component": "GRU", "Value": "1 layer, hidden 32 → MLP 32 → score"},
                {"Component": "1D-CNN", "Value": "Conv(16,k5) → Conv(32,k5,s2) → mean+max pool → MLP"},
                {"Component": "Training", "Value": f"{epochs} epochs · AdamW lr {lr} · Huber loss · ensemble of {seeds} seeds"},
                {"Component": "Augmentation", "Value": f"noise {noise} + calibration drift 0.1 (normalized units)"}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("#### Tabular feature dictionary")
    st.dataframe(pd.DataFrame({"Feature": list(TM.FEATURE_INFO), "Description": list(TM.FEATURE_INFO.values())}),
                 hide_index=True, width="stretch")
    with st.expander("How to read the metrics"):
        st.markdown("""
- **MAE** — average absolute difference |prediction − actual score| (cupping points). Lower is better. ±std = variation across CV repeats.
- **R²** — 1 = perfect, 0 = same as guessing the mean, negative = worse than guessing the mean.
- **Acc / F1 good** — accuracy and F1 when the predicted score is converted into a *good / not good* label at the selected threshold.
- **Baseline** — a "guess the mean" model. Any useful model must clearly beat it.
""")

with t_cmp:
    if exp is None:
        empty_state()
    else:
        res, R = exp["results"], exp["results"].set_index("Model")
        base = R.loc["Baseline (rata-rata)", "MAE"]
        thr_e = exp["cfg"]["threshold"]
        st.subheader("Which model performs better?")
        c = st.columns(3)
        winner = None
        if exp["best_tab"] and exp["best_seq"]:
            dt, ds = R.loc[exp["best_tab"]], R.loc[exp["best_seq"]]
            winner = "tab" if dt.MAE <= ds.MAE else "seq"
            tol = max(dt.MAE_std, ds.MAE_std)
        for col, key, title, name in [(c[0], "tab", "Best · Tabular", exp["best_tab"]), (c[1], "seq", "Best · Sequence", exp["best_seq"])]:
            if name:
                r = R.loc[name]
                col.markdown(kpi(title, name, f"MAE {r.MAE:.2f} · R² {r.R2:.2f} · {100*(1-r.MAE/base):.0f}% better than baseline",
                                 win=(winner == key)), unsafe_allow_html=True)
        c[2].markdown(kpi("Baseline (mean prediction)", f"MAE {base:.2f}", "lower bound that must be beaten"), unsafe_allow_html=True)
        if winner:
            gap = abs(dt.MAE - ds.MAE)
            who = exp["best_tab"] if winner == "tab" else exp["best_seq"]
            if tol == 0:
                st.markdown(f'<div class="note">MAE difference of {gap:.2f} points for <b>{who}</b>. Variation across CV repeats cannot yet be '
                            f'measured (repeats = 1) — increase "Cross-validation repeats" to see whether this difference is meaningful.</div>', unsafe_allow_html=True)
            elif gap <= tol:
                st.markdown(f'<div class="note">MAE difference ({gap:.2f}) is still within the variation across CV repeats (±{tol:.2f}) — '
                            f'both approaches are <b>statistically equivalent</b> on this data. Slight edge: <b>{who}</b>.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="note"><b>{who}</b> leads with an MAE difference of {gap:.2f} points (larger than the CV variation ±{tol:.2f}).</div>', unsafe_allow_html=True)
        if exp["n"] < 15:
            st.warning(f"Only {exp['n']} batches so far — results are preliminary; complex models (especially deep learning) need more batches.")
        if mode != "Real data":
            st.caption(f"Data mode **{mode}**: this comparison tests the pipeline, not which approach is better for real coffee.")

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

with t_train:
    if exp is None:
        empty_state()
    else:
        thr_e = exp["cfg"]["threshold"]
        st.subheader("Training summary")
        c = st.columns(4)
        c[0].markdown(kpi("Batches used", exp["n"], "final model training"), unsafe_allow_html=True)
        c[1].markdown(kpi("Validation", exp["cv_desc"], "out-of-fold"), unsafe_allow_html=True)
        c[2].markdown(kpi("Best tabular model", exp["best_tab"] or "—", "CV winner"), unsafe_allow_html=True)
        c[3].markdown(kpi("Best sequence model", exp["best_seq"] or "—", "CV winner"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            if exp["seq_bundle"] is not None and exp["seq_bundle"].get("history"):
                show(PL.loss_curve(exp["seq_bundle"]["history"], exp["best_seq"]))
        with b:
            st.markdown("**Per-batch error (out-of-fold) — hardest batches to predict**")
            err = pd.DataFrame({"Batch": exp["ids"], "Actual score": exp["y"]})
            for nm, tag in [(exp["best_tab"], "Tabular"), (exp["best_seq"], "Sequence")]:
                if nm:
                    err[f"Prediction {tag}"] = exp["oof"][nm]; err[f"|Error| {tag}"] = np.abs(exp["oof"][nm] - exp["y"])
            sort_col = [c_ for c_ in err.columns if c_.startswith("|Error|")][0]
            st.dataframe(err.sort_values(sort_col, ascending=False).round(2), hide_index=True, height=330, width="stretch")
        st.markdown("#### Download results & models")
        d = st.columns(4)
        d[0].download_button("Download comparison.csv", exp["results"].to_csv(index=False), "comparison.csv", "text/csv", width="stretch")
        d[1].download_button("Download per_batch_error.csv", err.to_csv(index=False), "per_batch_error.csv", "text/csv", width="stretch")
        if exp["tab_model"] is not None:
            buf = io.BytesIO(); joblib.dump({"model": exp["tab_model"], "name": exp["best_tab"], "features": TM.FEATURE_COLS, "threshold": thr_e}, buf)
            d[2].download_button("Download tabular_best.joblib", buf.getvalue(), "tabular_best.joblib", width="stretch")
        if exp["seq_bundle"] is not None:
            buf = io.BytesIO(); torch.save(exp["seq_bundle"], buf)
            d[3].download_button("Download sequence_best.pt", buf.getvalue(), "sequence_best.pt", width="stretch")
        st.caption("Saved models can be used via `python predict_new.py --data new_batch.json --models <folder>`.")

with t_clu:
    if exp is None:
        empty_state()
    elif exp["k_tab"] is None:
        st.info("Clustering requires at least 4 batches.")
    else:
        thr_e, cl = exp["cfg"]["threshold"], exp["clusters"]
        c = st.columns(3)
        c[0].markdown(kpi("Tabular cluster", f"k = {exp['k_tab']}", "sensor summary features"), unsafe_allow_html=True)
        c[1].markdown(kpi("Sequence cluster", f"k = {exp['k_seq']}", "GRU embedding"), unsafe_allow_html=True)
        c[2].markdown(kpi("Agreement (ARI)", f"{exp['ari']:.2f}", "1 = identical, ≈0 = random"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        for col, key, Pk, title, Xc in [(a, "cluster_tabular", "pca_tab", "Clusters from tabular features", None),
                                        (b, "cluster_sequence", "pca_seq", "Clusters from GRU embedding", None)]:
            with col:
                st.markdown(f"#### {title}")
                show(PL.cluster_scatter(exp[Pk], cl[key].values, exp["y"], title))
                st.dataframe(P.cluster_summary(cl, key, thr_e), width="stretch")
                show(PL.cluster_mean_curves(exp["X"], cl[key].values))
        st.markdown('<div class="note">The GRU embedding is trained using cupping scores, so its clusters tend to follow the score '
                    '(semi-supervised). Tabular clusters are based purely on sensor patterns without seeing the score more honest for '
                    'finding "fermentation types", but not necessarily aligned with taste.</div>', unsafe_allow_html=True)
        with st.expander("Cluster assignment per batch"):
            st.dataframe(cl, hide_index=True, width="stretch")
        st.download_button("Download clusters.csv", cl.to_csv(index=False), "clusters.csv", "text/csv")

with t_pred:
    if exp is None:
        empty_state()
    else:
        st.subheader("Batch taste prediction")
        how = st.radio("Batch source", ["Upload new batch (.json)", "Select from existing data"], horizontal=True, index=1)
        new_batches, is_train = None, False
        if how.startswith("Upload"):
            a, b = st.columns([3, 1])
            up = a.file_uploader("New batch sensor data (same format as training data)", type=["json"], key="new_up")
            b.write(""); b.write("")
            sample = pd.concat([batches[s] for s in list(batches)[:2]]).to_json(orient="records") if n else "[]"
            b.download_button("Download sample format", sample, "sample_new_batch.json", "application/json", width="stretch")
            if up is not None:
                new_batches = C.split_batches(_load_json(up.getvalue(), interval))
        else:
            sel = st.multiselect("Select batches", ids, default=ids[:3])
            if sel:
                new_batches, is_train = {s: batches[s] for s in sel}, True
                st.caption("These batches were used during training, so their predictions are optimistic. Use new batches for an honest test.")
        if new_batches:
            pr = P.predict_batches(exp, new_batches)
            pred_cols = [c_ for c_ in pr.columns if c_.startswith(("Tabular", "Sequence"))]
            for s, r in pr.iterrows():
                c = st.columns([1, 1.2, 1.2, 1.2, 1.4])
                c[0].markdown(kpi("Batch", s, f"{len(new_batches[s])} samples"), unsafe_allow_html=True)
                for i, pc in enumerate(pred_cols):
                    c[1 + i].markdown(kpi(pc.replace(" · ", " — "), f"{r[pc]:.1f}", "predicted cupping score"), unsafe_allow_html=True)
                c[3].markdown(kpi("Average of both models", f"{r['Rata-rata']:.1f}", f"difference {r['Selisih model']:.1f} points"), unsafe_allow_html=True)
                c[4].markdown(kpi("Prediction", badge(r["Prediksi"]), f"threshold ≥ {exp['cfg']['threshold']:.0f}"), unsafe_allow_html=True)
                if r["Selisih model"] > 3:
                    st.caption(f"Batch {s}: the two models differ by more than 3 points treat this prediction with caution.")
                st.write("")
            show(PL.predicted_curves(new_batches, list(new_batches)[:8]))
            with st.expander("Prediction table"):
                st.dataframe(pr.round(2), width="stretch")
            st.download_button("Download predictions.csv", pr.round(2).to_csv(), "predictions.csv", "text/csv")
        elif how.startswith("Upload"):
            st.caption("Upload a new batch JSON file to get predictions from both models.")

st.markdown("---")
st.caption("☕ Coffee Fermentation Lab : Tabular ML & Deep Learning Sequence · "
           "Synthetic / test-mode data is only for building and testing the model.")
