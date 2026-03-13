import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import io
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="StockML · Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #0d1726;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #64748b !important; font-size:12px; letter-spacing:0.1em; text-transform:uppercase; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-family: 'Space Mono', monospace; font-size: 1.6rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0d1220; border-bottom: 1px solid #1e2d45; gap:0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #475569; border:none; font-weight:600; padding: 10px 22px; }
.stTabs [aria-selected="true"] { background: #0f2744 !important; color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    padding: 10px 28px;
    letter-spacing: 0.05em;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #f0f9ff !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: #cbd5e1 !important; }

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; }

/* Selectbox / slider */
.stSelectbox label, .stSlider label { color: #64748b !important; font-size:12px; text-transform:uppercase; letter-spacing:0.07em; }

/* Expander */
.streamlit-expanderHeader { background: #0d1726 !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; }

/* Success / info boxes */
.stSuccess { background: #042f2e !important; border-left: 4px solid #10b981 !important; }
.stInfo    { background: #0c1a2e !important; border-left: 4px solid #38bdf8 !important; }
.stWarning { background: #1c1206 !important; border-left: 4px solid #f59e0b !important; }

/* Code block */
code { font-family: 'Space Mono', monospace !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette for charts ────────────────────────────────
PALETTE = ["#38bdf8", "#818cf8", "#fb7185", "#34d399", "#fbbf24"]
BG      = "#0d1726"
GRID    = "#1e2d45"
TEXT    = "#cbd5e1"

def apply_dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(GRID)

# ═══════════════════════════════════════════════════════════
#  PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def load_and_clean(df_raw):
    df = df_raw.copy()
    df.index = pd.to_datetime(df.index)
    before_null = df.isnull().sum().sum()
    before_dup  = df.duplicated().sum()
    df = df.drop_duplicates().ffill().bfill()
    return df, before_null, before_dup

def engineer_features(df, target_stock="Stock_1"):
    for col in df.columns:
        df[f"{col}_ret"] = df[col].pct_change()
    df[f"{target_stock}_ma5"]      = df[target_stock].rolling(5).mean()
    df[f"{target_stock}_ma10"]     = df[target_stock].rolling(10).mean()
    df[f"{target_stock}_std5"]     = df[target_stock].rolling(5).std()
    df[f"{target_stock}_momentum"] = df[target_stock] - df[target_stock].shift(5)
    others = [c for c in df.columns if c.startswith("Stock_") and "_" not in c[7:] and c != target_stock]
    for o in others:
        df[f"spread_{target_stock[-1]}_{o[-1]}"] = df[target_stock] - df[o]
    df["target"] = (df[target_stock].shift(-1) > df[target_stock]).astype(int)
    df = df.dropna()
    return df

def select_features(df, k=8):
    raw_stocks = [c for c in df.columns if c.startswith("Stock_") and "_" not in c[7:]]
    feature_cols = [c for c in df.columns if c != "target" and c not in raw_stocks]
    X, y = df[feature_cols], df["target"]
    selector = SelectKBest(f_classif, k=min(k, len(feature_cols)))
    selector.fit(X, y)
    selected = X.columns[selector.get_support()].tolist()
    return X[selected], y, feature_cols, selected

def split_and_scale(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train), scaler.transform(X_test),
            y_train, y_test, scaler)

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        cv = cross_val_score(m, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = {"model": m, "cv_mean": cv.mean(), "cv_std": cv.std()}
    return results

def evaluate_models(results, X_test, y_test):
    for name, info in results.items():
        m = info["model"]
        y_pred  = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1]
        info.update({
            "y_pred": y_pred, "y_proba": y_proba,
            "acc": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba),
            "cm":  confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred,
                          target_names=["DOWN", "UP"], output_dict=True),
        })
    return results

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload CSV", type=["csv"],
                                help="Expects date index + stock price columns")
    st.markdown("---")
    target_stock = st.selectbox("🎯 Target Stock", ["Stock_1","Stock_2","Stock_3","Stock_4","Stock_5"])
    top_k        = st.slider("🔢 Features to Select (K)", 4, 12, 8)
    test_pct     = st.slider("✂️ Test Split %", 10, 40, 20)
    st.markdown("---")
    run_btn = st.button("🚀  Run Full Pipeline")
    st.markdown("---")
    st.markdown("""
<small style='color:#334155'>
**Pipeline Steps**<br>
① Load Dataset<br>
② Data Cleaning<br>
③ Feature Engineering<br>
④ Feature Selection<br>
⑤ Train / Test Split<br>
⑥ Train 3 Models<br>
⑦ Evaluate & Compare<br>
⑧ Deploy & Predict
</small>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div style='padding:28px 0 8px 0'>
  <span style='font-family:Space Mono,monospace;font-size:11px;color:#38bdf8;letter-spacing:0.2em'>MACHINE LEARNING PIPELINE</span>
  <h1 style='margin:4px 0 0 0;font-size:2.4rem'>Stock Direction Predictor 📈</h1>
  <p style='color:#475569;margin-top:6px'>Upload your dataset · Configure · Train · Predict</p>
</div>
<hr style='border-color:#1e2d45;margin-bottom:24px'>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MAIN LOGIC
# ═══════════════════════════════════════════════════════════

if uploaded is None and not os.path.exists("stock_data.csv"):
    st.info("👈 Upload a stock CSV in the sidebar to get started, or place `stock_data.csv` in the same folder.")
    st.markdown("""
**Expected CSV format:**
```
date,Stock_1,Stock_2,Stock_3,...
2020-01-01,101.76,100.16,99.49,...
```
""")
    st.stop()

# Load raw data
if uploaded:
    df_raw = pd.read_csv(uploaded, index_col=0)
else:
    df_raw = pd.read_csv("stock_data.csv", index_col=0)

# ── Always show Step 1 preview ───────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "🧠 Model Training", "📈 Evaluation", "🔮 Predict", "💾 Export"
])

# ────────────────────────────────────────────────────────────
# TAB 1 — Data Explorer
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Step 1 · Raw Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Missing", int(df_raw.isnull().sum().sum()))
    c4.metric("Duplicates", int(df_raw.duplicated().sum()))

    with st.expander("🔍 Preview Data", expanded=True):
        st.dataframe(df_raw.head(20), use_container_width=True)

    st.markdown("### Price History")
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#080c14")
    apply_dark(ax)
    df_raw.index = pd.to_datetime(df_raw.index)
    for i, col in enumerate(df_raw.columns):
        ax.plot(df_raw.index, df_raw[col], label=col, color=PALETTE[i % len(PALETTE)], linewidth=1.5)
    ax.legend(labelcolor=TEXT, facecolor=BG, fontsize=9, framealpha=0.5)
    ax.set_title("Stock Prices Over Time", fontsize=13, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("### Descriptive Statistics")
    st.dataframe(df_raw.describe().round(4), use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 2 — Model Training
# ────────────────────────────────────────────────────────────
with tab2:
    if not run_btn and "results" not in st.session_state:
        st.info("⬅️ Configure options in the sidebar and click **Run Full Pipeline**.")
        st.stop()

    if run_btn:
        with st.status("⚙️ Running Pipeline…", expanded=True) as status:

            st.write("🧹 Step 2 · Cleaning data…")
            df_clean, nulls, dups = load_and_clean(df_raw)
            st.write(f"   Removed {nulls} nulls, {dups} duplicates")

            st.write("⚙️ Step 3 · Engineering features…")
            df_feat = engineer_features(df_clean.copy(), target_stock)

            st.write(f"✅ Step 4 · Selecting top {top_k} features…")
            X_sel, y, all_feats, sel_feats = select_features(df_feat, k=top_k)

            st.write(f"✂️ Step 5 · Splitting {100-test_pct}/{test_pct}…")
            X_tr, X_te, y_tr, y_te, scaler = split_and_scale(X_sel, y, test_size=test_pct/100)

            st.write("🤖 Step 6 · Training 3 models…")
            res = train_models(X_tr, y_tr)

            st.write("📊 Step 7 · Evaluating models…")
            res = evaluate_models(res, X_te, y_te)

            # Best model
            best_name = max(res, key=lambda k: res[k]["auc"])
            best_model = res[best_name]["model"]

            # Save to session state
            st.session_state.results      = res
            st.session_state.best_name    = best_name
            st.session_state.best_model   = best_model
            st.session_state.scaler       = scaler
            st.session_state.sel_feats    = sel_feats
            st.session_state.X_sel        = X_sel
            st.session_state.y            = y
            st.session_state.y_te         = y_te
            st.session_state.df_feat      = df_feat
            st.session_state.df_clean     = df_clean

            # Save .pkl in-memory
            buf_model  = io.BytesIO(); joblib.dump(best_model, buf_model); buf_model.seek(0)
            buf_scaler = io.BytesIO(); joblib.dump(scaler, buf_scaler);    buf_scaler.seek(0)
            st.session_state.buf_model  = buf_model
            st.session_state.buf_scaler = buf_scaler

            status.update(label="✅ Pipeline complete!", state="complete")

    if "results" not in st.session_state:
        st.stop()

    res       = st.session_state.results
    best_name = st.session_state.best_name
    sel_feats = st.session_state.sel_feats
    df_feat   = st.session_state.df_feat

    st.markdown("### Step 6 · Training Summary")
    cols = st.columns(3)
    for i, (name, info) in enumerate(res.items()):
        with cols[i]:
            st.metric(name, f"{info['cv_mean']:.3f}",
                      delta=f"±{info['cv_std']:.3f} CV std")

    st.markdown("### Selected Features")
    feat_df = pd.DataFrame({"Feature": sel_feats})
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Feature importance chart
    if "Random Forest" in res:
        st.markdown("### Feature Importances (Random Forest)")
        fi = pd.Series(
            res["Random Forest"]["model"].feature_importances_,
            index=sel_feats
        ).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#080c14")
        apply_dark(ax)
        bars = ax.barh(fi.index, fi.values,
                       color=[PALETTE[i % len(PALETTE)] for i in range(len(fi))],
                       edgecolor=GRID)
        ax.set_title("Feature Importances", fontsize=12)
        ax.set_xlabel("Importance", color=TEXT)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ────────────────────────────────────────────────────────────
# TAB 3 — Evaluation
# ────────────────────────────────────────────────────────────
with tab3:
    if "results" not in st.session_state:
        st.info("⬅️ Run the pipeline first.")
        st.stop()

    res       = st.session_state.results
    best_name = st.session_state.best_name
    y_te      = st.session_state.y_te

    st.markdown("### Step 7 · Model Evaluation")

    # Summary table
    summary = pd.DataFrame([{
        "Model":    n,
        "CV Acc":   f"{i['cv_mean']:.4f}",
        "Test Acc": f"{i['acc']:.4f}",
        "AUC-ROC":  f"{i['auc']:.4f}",
        "Best?":    "🏆" if n == best_name else ""
    } for n, i in res.items()])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Per-model detail
    selected_model = st.selectbox("Inspect model:", list(res.keys()))
    info = res[selected_model]

    col_a, col_b = st.columns(2)

    # ROC Curve
    with col_a:
        st.markdown("**ROC Curves**")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#080c14")
        apply_dark(ax)
        for i, (name, d) in enumerate(res.items()):
            fpr, tpr, _ = roc_curve(y_te, d["y_proba"])
            ax.plot(fpr, tpr, color=PALETTE[i], linewidth=2,
                    label=f"{['LR','RF','GBM'][i]} AUC={d['auc']:.2f}")
        ax.plot([0,1],[0,1], "--", color=GRID, linewidth=1)
        ax.legend(labelcolor=TEXT, facecolor=BG, fontsize=9)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curves", fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Confusion Matrix
    with col_b:
        st.markdown(f"**Confusion Matrix · {selected_model}**")
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#080c14")
        apply_dark(ax)
        cm = info["cm"]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["DOWN","UP"], color=TEXT)
        ax.set_yticklabels(["DOWN","UP"], color=TEXT)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha="center", va="center",
                        color="white", fontsize=16, fontweight="bold")
        ax.set_title(f"Confusion Matrix", fontsize=11)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Classification report
    st.markdown("**Classification Report**")
    rpt = info["report"]
    rpt_df = pd.DataFrame(rpt).T.round(3)
    st.dataframe(rpt_df, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 4 — Predict
# ────────────────────────────────────────────────────────────
with tab4:
    if "results" not in st.session_state:
        st.info("⬅️ Run the pipeline first.")
        st.stop()

    best_model = st.session_state.best_model
    scaler     = st.session_state.scaler
    sel_feats  = st.session_state.sel_feats
    X_sel      = st.session_state.X_sel
    best_name  = st.session_state.best_name

    st.markdown(f"### Step 8 · Predict with **{best_name}**")
    st.markdown("Adjust feature values below and click **Predict**.")

    last_row = X_sel.iloc[-1]
    inputs = {}
    col_pairs = [st.columns(2) for _ in range((len(sel_feats)+1)//2)]
    for i, feat in enumerate(sel_feats):
        col = col_pairs[i//2][i%2]
        mn, mx = float(X_sel[feat].min()), float(X_sel[feat].max())
        step = (mx - mn) / 200 if mx != mn else 0.01
        inputs[feat] = col.number_input(feat, value=float(last_row[feat]),
                                         min_value=mn - abs(mn),
                                         max_value=mx + abs(mx),
                                         step=step, format="%.5f")

    if st.button("🔮 Predict Next Day Direction"):
        row_df = pd.DataFrame([inputs])
        row_sc = scaler.transform(row_df[sel_feats])
        pred   = best_model.predict(row_sc)[0]
        proba  = best_model.predict_proba(row_sc)[0]
        conf   = max(proba) * 100

        if pred == 1:
            st.success(f"## 📈 Prediction: UP ▲   ({conf:.1f}% confidence)")
        else:
            st.error(f"## 📉 Prediction: DOWN ▼   ({conf:.1f}% confidence)")

        c1, c2 = st.columns(2)
        c1.metric("Prob UP ▲",   f"{proba[1]*100:.1f}%")
        c2.metric("Prob DOWN ▼", f"{proba[0]*100:.1f}%")

        # Probability bar
        fig, ax = plt.subplots(figsize=(6, 1.5), facecolor="#080c14")
        apply_dark(ax)
        ax.barh(["DOWN ▼"], [proba[0]], color="#fb7185", height=0.4)
        ax.barh(["UP ▲"],   [proba[1]], color="#34d399", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Probabilities", fontsize=10)
        for spine in ["top","right","left"]: ax.spines[spine].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### Predict on historical rows")
    n_rows = st.slider("Show last N predictions", 5, 50, 20)
    X_sample_sc  = scaler.transform(X_sel.iloc[-n_rows:])
    preds_hist   = best_model.predict(X_sample_sc)
    probas_hist  = best_model.predict_proba(X_sample_sc)[:, 1]
    hist_df = X_sel.iloc[-n_rows:].copy()
    hist_df["Predicted"]   = ["UP ▲" if p == 1 else "DOWN ▼" for p in preds_hist]
    hist_df["Prob UP"]     = (probas_hist * 100).round(1)
    hist_df["Actual"]      = st.session_state.y.iloc[-n_rows:].map({1:"UP ▲", 0:"DOWN ▼"}).values
    hist_df["Correct?"]    = (hist_df["Predicted"] == hist_df["Actual"]).map({True:"✅", False:"❌"})
    st.dataframe(hist_df[["Predicted","Prob UP","Actual","Correct?"]], use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 5 — Export
# ────────────────────────────────────────────────────────────
with tab5:
    if "results" not in st.session_state:
        st.info("⬅️ Run the pipeline first to unlock exports.")
        st.stop()

    st.markdown("### 💾 Download Pipeline Artifacts")
    best_name = st.session_state.best_name

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧠 Best Model**")
        st.caption(f"{best_name} — `.pkl` format")
        st.download_button(
            label="⬇️  Download Model",
            data=st.session_state.buf_model,
            file_name="best_model.pkl",
            mime="application/octet-stream",
        )

    with col2:
        st.markdown("**⚖️ Scaler**")
        st.caption("StandardScaler — `.pkl` format")
        st.download_button(
            label="⬇️  Download Scaler",
            data=st.session_state.buf_scaler,
            file_name="scaler.pkl",
            mime="application/octet-stream",
        )

    with col3:
        st.markdown("**📋 Selected Features**")
        st.caption("Feature list — `.csv` format")
        feat_csv = pd.Series(st.session_state.sel_feats, name="feature").to_csv(index=False)
        st.download_button(
            label="⬇️  Download Features",
            data=feat_csv,
            file_name="selected_features.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("### 📊 Download Results")

    res = st.session_state.results
    y_te = st.session_state.y_te

    summary_data = pd.DataFrame([{
        "Model": n, "CV Accuracy": round(i["cv_mean"], 4),
        "CV Std": round(i["cv_std"], 4),
        "Test Accuracy": round(i["acc"], 4),
        "AUC-ROC": round(i["auc"], 4),
    } for n, i in res.items()])

    st.download_button(
        label="⬇️  Download Evaluation Summary CSV",
        data=summary_data.to_csv(index=False),
        file_name="model_evaluation.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### 🔌 Inference Code Snippet")
    st.code(f'''
import joblib, pandas as pd

model   = joblib.load("best_model.pkl")
scaler  = joblib.load("scaler.pkl")
features = pd.read_csv("selected_features.csv").squeeze().tolist()

def predict(row: dict) -> dict:
    df   = pd.DataFrame([row])[features]
    sc   = scaler.transform(df)
    pred = model.predict(sc)[0]
    prob = model.predict_proba(sc)[0]
    return {{
        "direction":  "UP" if pred == 1 else "DOWN",
        "confidence": f"{{max(prob)*100:.1f}}%",
        "prob_up":    round(prob[1], 4),
    }}
''', language="python")
