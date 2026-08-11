"""
ML Assignment 2 - Streamlit front-end
Upload a test CSV, pick a model, see metrics + confusion matrix + classification report.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

st.set_page_config(page_title="Classifier Comparison Lab", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    meta = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in meta["model_files"].items()
    }
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    training_metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return meta, models, training_metrics


try:
    META, MODELS, TRAIN_METRICS = load_artifacts()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load model artifacts from `model/`. Run `python model/train_models.py` first.\n\n{exc}")
    st.stop()

TARGET = META["target_column"]
FEATURES = META["feature_columns"]
LE = META["label_encoder"]
CLASS_NAMES = META["class_names"]
IS_BINARY = META["task"] == "binary"


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score(model, X, y_true_enc):
    y_pred = model.predict(X)
    avg = "binary" if IS_BINARY else "weighted"
    try:
        proba = model.predict_proba(X)
        auc = (
            roc_auc_score(y_true_enc, proba[:, 1])
            if IS_BINARY
            else roc_auc_score(y_true_enc, proba, multi_class="ovr", average="weighted")
        )
    except Exception:  # noqa: BLE001
        auc = np.nan
    return y_pred, {
        "Accuracy": accuracy_score(y_true_enc, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true_enc, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true_enc, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true_enc, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true_enc, y_pred),
    }


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    st.caption(f"Must contain the target column `{TARGET}` plus the {len(FEATURES)} feature columns.")

    st.divider()
    chosen = st.selectbox("Model", list(MODELS.keys()))
    compare_all = st.checkbox("Compare all models", value=True)

    st.divider()
    st.caption(f"Task: **{META['task']}** · Classes: {', '.join(CLASS_NAMES)}")


st.title("📊 Classifier Comparison Lab")
st.write("Five classification models trained on one dataset, scored side by side on your uploaded test set.")

if uploaded is None:
    st.info("⬅️ Upload `test_data.csv` in the sidebar to begin.")
    if TRAIN_METRICS:
        st.subheader("Reference: metrics from training run")
        st.dataframe(pd.DataFrame(TRAIN_METRICS).T.round(4), use_container_width=True)
    st.stop()


# ----------------------------------------------------------------------------
# Validate upload
# ----------------------------------------------------------------------------
df = pd.read_csv(uploaded)
st.success(f"Loaded **{df.shape[0]}** rows × **{df.shape[1]}** columns.")

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"Missing target column `{TARGET}` — needed to compute evaluation metrics.")
    st.stop()

X = df[FEATURES]
try:
    y_true = LE.transform(df[TARGET].astype(LE.classes_.dtype))
except Exception as exc:  # noqa: BLE001
    st.error(f"Target values don't match the classes seen in training ({CLASS_NAMES}).\n\n{exc}")
    st.stop()

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
tab_single, tab_all = st.tabs(["🔍 Selected model", "📈 All models"])

with tab_single:
    y_pred, m = score(MODELS[chosen], X, y_true)
    st.subheader(chosen)

    cols = st.columns(6)
    for col, (label, val) in zip(cols, m.items()):
        col.metric(label, "n/a" if pd.isna(val) else f"{val:.4f}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    preds_out = df.copy()
    preds_out["predicted"] = LE.inverse_transform(y_pred)
    st.download_button("⬇️ Download predictions", preds_out.to_csv(index=False),
                       "predictions.csv", "text/csv")

with tab_all:
    if not compare_all:
        st.info("Enable *Compare all models* in the sidebar.")
    else:
        rows = {name: score(mdl, X, y_true)[1] for name, mdl in MODELS.items()}
        table = pd.DataFrame(rows).T
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#1f6f4a"),
            use_container_width=True,
        )

        metric_choice = st.selectbox("Chart metric", list(table.columns), index=4)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        vals = table[metric_choice]
        ax2.barh(vals.index, vals.values, color="#4c78a8")
        ax2.set_xlim(0, 1)
        ax2.set_xlabel(metric_choice)
        for i, v in enumerate(vals.values):
            ax2.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        st.pyplot(fig2)

        st.success(f"Best on this test set by F1: **{table['F1'].idxmax()}**")
