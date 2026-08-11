# Machine Learning Assignment 2 — Classifier Comparison Lab

> **Fill in every `<...>` placeholder before submitting.** The observation table in
> section (e) is worth 3 marks and must be written in your own words based on
> *your* numbers — do not submit generic text.

---

## a. Problem Statement

Undergraduate students at a Portuguese higher education institution are known,
at enrollment and after their first two semesters, by a set of demographic,
socio-economic, and academic-performance attributes. This project predicts
each student's final academic outcome — **Dropout**, **Enrolled** (still
studying past the normal course duration), or **Graduate** — from those
attributes. Early identification of at-risk students lets an institution
target advising, financial aid, or academic support before a student
withdraws, rather than after the fact.

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
| Missing values | None reported in the source data; the training pipeline imputes defensively anyway (median for numeric, mode for categorical) in case your local copy differs |

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

Most categorical columns arrive **pre-encoded as integer codes** by UCI (e.g.
`Marital status` is 1–6), so `pandas` will read them as numeric dtype even
though they represent categories. The training pipeline in this repo treats
any non-numeric-dtype column as categorical automatically — since these
arrive numeric, they flow through the numeric branch (impute + scale) rather
than one-hot encoding. This is worth a line in your own write-up if you
noticed it while inspecting the data.

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

*Precision / Recall / F1 are weighted-averaged across the three classes
(Dropout, Enrolled, Graduate), which accounts for the class imbalance rather
than treating all classes as equally sized. AUC is one-vs-rest, weighted by
class support.*

---

## e. Observations on Model Performance  *(3 marks)*

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <Fill in from your run. Prompt: with 36 features and a scaled numeric pipeline, does linear separation do reasonably well, or does accuracy trail the tree-based models? A gap here suggests some non-linear interaction among the curricular-unit / grade features.> |
| Decision Tree | <Prompt: compare directly against Random Forest below — a noticeably lower F1/MCC on the single tree is evidence of overfitting to training splits, which the ensemble corrects for.> |
| kNN | <Prompt: with 36 features, distance-based methods often degrade (curse of dimensionality). Say whether your run confirms or contradicts that, and note that all features were standardized before distance computation.> |
| Naive Bayes | <Prompt: Naive Bayes assumes feature independence — this dataset has clearly correlated features (e.g. 1st/2nd semester grades, admission grade vs. previous-qualification grade), so expect it to be among the weaker performers despite often keeping a respectable AUC.> |
| Random Forest (Ensemble) | <Prompt: usually the strongest by MCC. State the actual margin over the single Decision Tree from your table.> |
| **Overall winner for your dataset?** | <Name the model using your MCC value, not accuracy — with the Enrolled class at only ~18% of the data, accuracy alone can look decent while badly under-serving the minority class. MCC is a fairer summary here.> |

### Additional notes
- The dataset is moderately imbalanced (Graduate 49.9% / Dropout 32.1% / Enrolled 17.9%), which is why MCC — rather than raw accuracy — is the more trustworthy headline metric in the table above.
- <Look at your confusion matrix and report which pair of classes is most confused. Across published work on this dataset, **Enrolled** is consistently the hardest class to separate — it sits between Dropout and Graduate on most academic-performance features, since these are students still mid-course rather than at a final outcome. State whether your matrix shows the same pattern.>

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
