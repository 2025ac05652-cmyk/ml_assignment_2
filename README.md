# Machine Learning Assignment 2 — Classifier Comparison Lab

---

## a. Problem Statement

This project predicts whether a student will **Drop out**, stay **Enrolled**
past the normal course length, or **Graduate**, based on their academic and
personal details recorded at enrollment. Catching at-risk students early
means a college can step in with support before they actually leave.

**Type of problem:** Multi-class classification
**Number of classes:** 3 — `Dropout`, `Enrolled`, `Graduate`

---

## b. Dataset Description  *(1 mark)*

| Item | Value |
|---|---|
| Dataset name | Predict Students' Dropout and Academic Success |
| Source | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Citation | Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5MC89 (CC BY 4.0) |
| Instances | 4,424 (requirement: ≥ 500 ✓) |
| Features | 36 (requirement: ≥ 12 ✓) |
| Target column | `Target` |
| Class balance | Graduate: 2,209 (49.9%) · Dropout: 1,421 (32.1%) · Enrolled: 794 (17.9%) — moderately imbalanced towards Graduate |
| Missing values | None reported in the source data; the pipeline still imputes defensively (median for numeric, mode for categorical) |

### Feature dictionary (selected — see `get_data.py` output for the full 36)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `Marital status` | categorical (coded) | Student's marital status at enrollment |
| 2 | `Application mode` | categorical (coded) | Admission route used |
| 3 | `Application order` | numeric | Preference order in which the course was chosen |
| 4 | `Course` | categorical (coded) | Degree programme enrolled in |
| 5 | `Daytime/evening attendance` | categorical (coded) | Day or evening class schedule |
| 6 | `Previous qualification` | categorical (coded) | Qualification held before enrolling |
| 7 | `Previous qualification (grade)` | numeric | Grade of that prior qualification |
| 8 | `Nacionality` | categorical (coded) | Student nationality |
| 9 | `Mother's qualification` / `Father's qualification` | categorical (coded) | Parental education level |
| 10 | `Mother's occupation` / `Father's occupation` | categorical (coded) | Parental occupation |
| 11 | `Admission grade` | numeric | Grade at admission |
| 12 | `Displaced` | binary | Whether the student is displaced from home |
| 13 | `Educational special needs` | binary | Special-needs status |
| 14 | `Debtor` | binary | Outstanding tuition debt |
| 15 | `Tuition fees up to date` | binary | Payment status |
| 16 | `Gender` | binary | — |
| 17 | `Scholarship holder` | binary | — |
| 18 | `Age at enrollment` | numeric | — |
| 19–34 | `Curricular units 1st/2nd sem (credited/enrolled/evaluations/approved/grade/without evaluations)` | numeric | Per-semester academic-load and performance counts |
| 35 | `Unemployment rate` / `Inflation rate` / `GDP` | numeric | Macroeconomic indicators at time of enrollment |

Most categorical columns come pre-encoded as numbers from UCI (e.g.
`Marital status` is 1–6). Since pandas reads them as numeric, the pipeline
treats them that way too — they get scaled rather than one-hot encoded.
Worth knowing since it's not obvious just from opening the CSV.

### Preprocessing applied
- Numeric features: median imputation → `StandardScaler`
- Categorical features: mode imputation → `OneHotEncoder(handle_unknown="ignore")`
- Target: `LabelEncoder`
- Split: stratified 75/25 train/test, `random_state=42`

All preprocessing is wrapped inside each model's `sklearn.Pipeline`, so the saved
`.joblib` files accept raw CSV input directly — no leakage from test into train.

---

## c. GitHub Repository Link  *(1 mark)*

[Github repo link](https://github.com/2025ac05652-cmyk/ml_assignment_2)

**Live Streamlit app:** [App link](https://mlassignment2-ap5k79dxoxnwccbgrskhjh.streamlit.app/)

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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7694 | 0.8948 | 0.7555 | 0.7694 | 0.7560 | 0.6171 |
| Decision Tree | 0.7251 | 0.8538 | 0.7137 | 0.7251 | 0.7137 | 0.5433 |
| kNN | 0.6908 | 0.8156 | 0.6730 | 0.6908 | 0.6675 | 0.4817 |
| Naive Bayes | 0.6582 | 0.8080 | 0.6371 | 0.6582 | 0.6432 | 0.4281 |
| Random Forest (Ensemble) | **0.7722** | **0.9010** | **0.7597** | **0.7722** | **0.7575** | **0.6217** |

*Precision / Recall / F1 are weighted-averaged across the three classes
(Dropout, Enrolled, Graduate) to account for the class imbalance. AUC is
one-vs-rest, weighted by class support.*

---

## e. Observations on Model Performance  *(3 marks)*

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best model. Almost tied with Random Forest, so a simple linear model does nearly as well as the ensemble here. |
| Decision Tree | Clearly weaker than Random Forest (same features, much lower MCC). A single tree overfits; that's what the ensemble fixes. |
| kNN | Underperforms the others. Distance-based methods struggle a bit with 36 features. |
| Naive Bayes | Weakest model overall. Its independence assumption doesn't really hold here since a lot of the grade features are correlated. |
| Random Forest (Ensemble) | Best model on every metric, though only slightly ahead of Logistic Regression. |
| **Overall winner for your dataset?** | **Random Forest**, based on MCC rather than accuracy, since Enrolled is a small class and MCC accounts for that better. |

### Additional notes
- The dataset is imbalanced (Graduate 49.9% / Dropout 32.1% / Enrolled 17.9%), which is why MCC is used as the main metric instead of accuracy.
- The confusion matrix shows Enrolled is the hardest class to predict (recall only 0.37), and it gets confused with both Dropout and Graduate roughly equally. Makes sense since these students are still mid-way through and don't clearly look like either outcome yet.

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

<img width="1728" height="918" alt="Screenshot 2026-08-11 at 9 06 54 PM" src="https://github.com/user-attachments/assets/f4568a5e-7d28-4a68-9bab-5eddbcbc66a4" />
<img width="1726" height="921" alt="Screenshot 2026-08-11 at 9 06 22 PM" src="https://github.com/user-attachments/assets/59645cb8-5779-4766-b28f-74926c69cd29" />

---

## Author

Tarun Gupta — 2025ac05652
M.Tech (AIML/DSE), Machine Learning — Assignment 2
