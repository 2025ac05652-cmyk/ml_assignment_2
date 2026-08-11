# Machine Learning Assignment 2 — Classifier Comparison Lab

> **Fill in every `<...>` placeholder before submitting.** The observation table in
> section (e) is worth 3 marks and must be written in your own words based on
> *your* numbers — do not submit generic text.

---

## a. Problem Statement

<One short paragraph. What is being predicted, from what, and why it matters.
Example shape: "Given 16 clinical and demographic attributes recorded at
admission, predict whether a patient will be readmitted within 30 days. This is
a binary classification problem; accurate prediction lets hospitals target
follow-up resources at high-risk patients.">

**Type of problem:** <binary / multi-class> classification
**Number of classes:** <n> — <list class labels>

---

## b. Dataset Description  *(1 mark)*

| Item | Value |
|---|---|
| Dataset name | <name> |
| Source | <Kaggle / UCI> — <full URL> |
| Instances | <n> (requirement: ≥ 500) |
| Features | <n> (requirement: ≥ 12) |
| Target column | `<target>` |
| Class balance | <e.g. 68% / 32%, or counts per class> |
| Missing values | <yes — handled by median/mode imputation / none> |

### Feature dictionary

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `<name>` | numeric | <description> |
| 2 | `<name>` | categorical | <description> |
| … | | | |

### Preprocessing applied
- Numeric features: median imputation → `StandardScaler`
- Categorical features: mode imputation → `OneHotEncoder(handle_unknown="ignore")`
- Target: `LabelEncoder`
- Split: stratified <75>/<25> train/test, `random_state=42`

All preprocessing is wrapped inside each model's `sklearn.Pipeline`, so the saved
`.joblib` files accept raw CSV input directly — no leakage from test into train.

---

## c. GitHub Repository Link  *(1 mark)*

<https://github.com/USERNAME/REPO>

**Live Streamlit app:** <https://YOUR-APP.streamlit.app>

### Repository structure
```
project-folder/
├── app.py                  # Streamlit front-end
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test split for app upload
├── data/
│   └── dataset.csv         # full source dataset
└── model/
    ├── train_models.py     # training + evaluation script
    ├── metadata.joblib     # feature list, label encoder, class names
    ├── metrics.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest_ensemble.joblib
```

### How to reproduce
```bash
git clone <repo-url> && cd <repo>
pip install -r requirements.txt
python model/train_models.py     # retrains all 5 models, rewrites test_data.csv
streamlit run app.py
```

---

## d. Models Used  *(5 marks)*

| # | Model | Key hyperparameters |
|---|-------|---------------------|
| 1 | Logistic Regression | `max_iter=2000`, L2 penalty |
| 2 | Decision Tree | `max_depth=8`, `min_samples_leaf=5` |
| 3 | k-Nearest Neighbours | `n_neighbors=9`, `weights="distance"` |
| 4 | Naive Bayes | GaussianNB |
| 5 | Random Forest (Ensemble) | `n_estimators=300`, `min_samples_leaf=2` |

### Comparison Table — evaluation metrics on the held-out test set

<Paste the markdown table printed by `python model/train_models.py`.>

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| kNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |

*Precision / Recall / F1 are <binary — positive class `<label>` / weighted-averaged
across classes>. AUC is <binary ROC-AUC / one-vs-rest, weighted>.*

---

## e. Observations on Model Performance  *(3 marks)*

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <Did it under- or over-perform? What does that imply about linear separability of your feature space? Note the gap between accuracy and MCC if the classes are imbalanced.> |
| Decision Tree | <Compare against Random Forest — a large gap suggests the single tree is variance-limited. Mention depth/overfitting evidence if train accuracy ≫ test accuracy.> |
| kNN | <Sensitive to scaling and to the curse of dimensionality at <n> features. Comment on whether distance weighting helped and on prediction latency.> |
| Naive Bayes | <The feature-independence assumption is <plausible/violated> here because <reason>. Note that it often shows decent AUC but weaker calibrated accuracy.> |
| Random Forest (Ensemble) | <Usually strongest. Say by how much it beat the single tree and which features it ranked most important.> |
| **Overall winner for your dataset?** | <Name the model and justify with a specific metric. Prefer MCC or F1 over raw accuracy if your classes are imbalanced, and say why.> |

### Additional notes
- <Any class-imbalance handling, e.g. why you report MCC as the headline metric.>
- <Anything surprising in the confusion matrix — e.g. which two classes get confused.>

---

## f. Streamlit Application Features  *(4 marks)*

| Requirement | Implementation |
|---|---|
| Dataset upload option (CSV) | Sidebar file uploader; validates that all feature columns and the target column are present |
| Model selection dropdown | Sidebar `selectbox` across all 5 trained pipelines |
| Display of evaluation metrics | Six `st.metric` cards (Accuracy, AUC, Precision, Recall, F1, MCC) + an all-model comparison table with best-value highlighting and a bar chart |
| Confusion matrix / classification report | Seaborn heatmap confusion matrix **and** a per-class classification report table |
| Extra | Downloadable predictions CSV; metric-selectable comparison chart |

### Usage
1. Open the live app link.
2. Upload `test_data.csv` from this repository.
3. Choose a model from the dropdown, or open the **All models** tab to compare.

---

## g. BITS Virtual Lab Execution

Screenshot of the assignment running on BITS Virtual Lab is included in the
submitted PDF (Section 3 of the submission).

---

## Author

<Name> — <BITS ID>
M.Tech (AIML/DSE), Machine Learning — Assignment 2
