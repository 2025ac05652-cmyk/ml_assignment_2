"""
ML Assignment 2 - Model training script
Trains 5 classification models on a single dataset and exports:
  - model/*.joblib          (full sklearn Pipelines: preprocessing + classifier)
  - model/metadata.joblib   (feature list, label encoder, task type)
  - model/metrics.json      (evaluation metrics for the comparison table)
  - test_data.csv           (held-out test split, RAW columns, for the Streamlit app)

Run:  python model/train_models.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# 1. CONFIG  <-- the only block you need to edit for your own dataset
# ----------------------------------------------------------------------------
DATA_PATH = "data/dataset.csv"   # path to your downloaded Kaggle/UCI csv
TARGET_COLUMN = "Target"         # name of the label column
DROP_COLUMNS = []                # e.g. ["id", "customer_id"] - identifiers, not features
TEST_SIZE = 0.25
RANDOM_STATE = 42

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = HERE


# ----------------------------------------------------------------------------
# 2. LOAD + SANITY CHECK
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(os.path.join(ROOT, DATA_PATH))
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.dropna(subset=[TARGET_COLUMN])

    n_features = df.shape[1] - 1
    n_rows = df.shape[0]
    print(f"Loaded {n_rows} instances x {n_features} features")
    if n_features < 12:
        print(f"  !! WARNING: assignment requires >= 12 features, you have {n_features}")
    if n_rows < 500:
        print(f"  !! WARNING: assignment requires >= 500 instances, you have {n_rows}")
    return df


# ----------------------------------------------------------------------------
# 3. PREPROCESSOR - handles numeric + categorical columns automatically
# ----------------------------------------------------------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"  numeric: {len(numeric_cols)} | categorical: {len(categorical_cols)}")

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")


# ----------------------------------------------------------------------------
# 4. METRICS - one function, works for binary AND multi-class
# ----------------------------------------------------------------------------
def evaluate(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    avg = "binary" if n_classes == 2 else "weighted"

    try:
        proba = model.predict_proba(X_test)
        if n_classes == 2:
            auc = roc_auc_score(y_test, proba[:, 1])
        else:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
    except Exception as exc:                       # noqa: BLE001
        print(f"  AUC unavailable: {exc}")
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    df = load_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)
    task = "binary" if n_classes == 2 else "multiclass"
    print(f"Task: {task} ({n_classes} classes -> {list(label_encoder.classes_)})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    results = {}

    for name, clf in classifiers.items():
        print(f"Training: {name}")
        pipe = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)

        scores = evaluate(pipe, X_test, y_test, n_classes)
        results[name] = scores

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{slug}.joblib"))
        print("  " + " | ".join(f"{k}={v:.4f}" for k, v in scores.items()) + "\n")

    # shared metadata the Streamlit app needs
    joblib.dump(
        {
            "feature_columns": X.columns.tolist(),
            "target_column": TARGET_COLUMN,
            "label_encoder": label_encoder,
            "class_names": [str(c) for c in label_encoder.classes_],
            "task": task,
            "model_files": {
                n: n.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
                for n in classifiers
            },
        },
        os.path.join(MODEL_DIR, "metadata.joblib"),
    )

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # test split exported with ORIGINAL labels so the app can score an upload
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = label_encoder.inverse_transform(y_test)
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    print(f"Wrote test_data.csv ({len(test_df)} rows)")

    # markdown comparison table -> paste straight into README.md
    table = pd.DataFrame(results).T.round(4)
    print("\n=== Comparison table (copy into README.md) ===\n")
    print(table.to_markdown())
    print(f"\nBest by F1: {table['F1'].idxmax()}   |   Best by MCC: {table['MCC'].idxmax()}")


if __name__ == "__main__":
    main()
