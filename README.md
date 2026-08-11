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

[Github repo link](https://github.com/2025ac05652-cmyk/ml_assignment_2)

**Live Streamlit app:** <[App link](https://mlassignment2-3xvhres5dmxmgpfmuqpsng.streamlit.app/)>

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
(Dropout, Enrolled, Graduate), which accounts for the class imbalance rather
than treating all classes as equally sized. AUC is one-vs-rest, weighted by
class support.*

---

## e. Observations on Model Performance  *(3 marks)*

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best model overall (MCC 0.6171), trailing Random Forest by only 0.0046 MCC despite being a simple linear model. This closeness suggests the 36 features carry a largely linear, monotonic relationship with dropout risk — the ensemble's extra capacity to model non-linear interactions bought very little on top of what a weighted linear combination of features already captures. |
| Decision Tree | Clearly weaker than the Random Forest built from the same features (MCC 0.5433 vs 0.6217, a 0.078 gap — the largest tree-vs-ensemble gap in the table). A single tree overfits to specific splits in the training data and its predictions vary more across the training set than a forest of many trees averaged together; this is the classic bias-variance argument for ensembling, and this dataset demonstrates it directly. |
| kNN | The weakest of the non-Naive-Bayes models (MCC 0.4817, Accuracy 0.6908). With 36 standardized features, Euclidean distance becomes a less meaningful similarity measure — the curse of dimensionality dilutes the signal from a few informative features (e.g. semester grades) among many less-informative ones, so "nearest" neighbors in 36-D space are not necessarily the most academically similar students. |
| Naive Bayes | Lowest-scoring model on every metric (MCC 0.4281). This tracks with the model's feature-independence assumption, which this dataset clearly violates: 1st- and 2nd-semester curricular unit counts and grades are strongly correlated with each other and with admission/previous-qualification grades. Naive Bayes still keeps a respectable AUC (0.808) relative to its accuracy, meaning it ranks students by risk reasonably well even where its hard classifications are less accurate. |
| Random Forest (Ensemble) | Best model on all six metrics, though only marginally ahead of Logistic Regression (MCC 0.6217 vs 0.6171). It clearly outperforms the single Decision Tree it's built from (+0.078 MCC), confirming that averaging many trees reduces the variance/overfitting the standalone tree suffered from. |
| **Overall winner for your dataset?** | **Random Forest (Ensemble)**, by MCC (0.6217) rather than accuracy — with Enrolled at only ~18% of the data, accuracy can look reasonable while under-serving that minority class, so MCC is the fairer summary metric here. That said, Logistic Regression is a close second and is far cheaper to train and easier to interpret (coefficients map directly to feature effects), so it's a reasonable alternative if interpretability matters more than the last ~0.5 points of MCC. |

### Additional notes
- The dataset is moderately imbalanced (Graduate 49.9% / Dropout 32.1% / Enrolled 17.9%), which is why MCC — rather than raw accuracy — is the more trustworthy headline metric in the table above.
- The overall metric ordering (RF ≳ LogReg > Decision Tree > kNN > Naive Bayes) is consistent across every single metric column, not just MCC — a good sign that these results reflect genuine model quality differences rather than metric-specific artifacts.
- The confusion matrix confirms it: **Enrolled is by far the hardest class**, with recall of only 0.372 (74/199 correctly identified) versus 0.741 for Dropout and 0.937 for Graduate. Of the 125 misclassified Enrolled students, 74 were predicted as Graduate and 51 as Dropout — Enrolled genuinely sits *between* the other two classes rather than being confused with just one of them, which matches the intuition that these are students still mid-course, showing academic-performance signals partway between an eventual dropout and an eventual graduate. Dropout shows the same but weaker pattern (65 of its 92 errors go to Graduate, only 27 to Enrolled), while Graduate is rarely misclassified at all (35 errors out of 552). This is also why macro-averaged F1 (0.696) sits noticeably below the weighted average (0.7575) in the classification report — the model is doing well on the two larger classes and dragging down only on the minority Enrolled class, and a macro average weights all three classes equally so it exposes that gap.

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
