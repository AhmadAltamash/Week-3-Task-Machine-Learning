# Unsupervised Learning & Model Evaluation (Week 3)

Week 3 internship task: learn clustering (K-Means, hierarchical) and dimensionality reduction, plus model evaluation (cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, hyperparameter tuning). Deliverable: clustering on a dataset and a model evaluation report.

## Data
`data/employee_data.csv` – the same **synthetic** employee dataset used in Weeks 1 and 2 (500 rows after de-duplication, with some missing values that the pipelines fill).

## Part A – unsupervised (no labels used)
1. Standardize the 5 numeric features; show why scaling matters
2. PCA: scree plot, loadings, 2D projection
3. K-Means for K = 2–8: inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, stability (ARI over subsamples)
4. Silhouette profile of the chosen K
5. Hierarchical clustering: Ward / complete / average, cophenetic correlation, dendrogram, cluster sizes
6. Interpret clusters: profiles in original units, comparison with Job_Level (never used), attrition rate per cluster + chi-square test

## Part B – supervised evaluation (predict Attrition)
1. Stratified 80/20 split
2. Seven models compared with 5-fold CV × 3 repeats (accuracy, precision, recall, F1, ROC-AUC) vs a baseline
3. Check whether three engineered features help
4. GridSearchCV (ROC-AUC) for Logistic Regression, PCA + Logistic Regression, Random Forest, KNN
5. Choose threshold from out-of-fold predictions; evaluate once on the test set
6. Confusion matrices, ROC and precision-recall curves, bootstrap interval for the test ROC-AUC

## Main findings
- **Clusters:** a junior group vs a senior group (age, experience and income are strongly correlated). K = 2 is the most stable choice (silhouette 0.32, stability ARI 0.95), but the structure is weak: it is a continuum cut in two.
- **Average linkage** has the best silhouette (0.36) only because it isolates 16 outliers; Ward gives balanced clusters.
- The senior cluster has a **higher attrition rate (39.7% vs 27.0%, p = 0.004)**, found without using the label.
- **Classification** stays near ROC-AUC 0.55–0.59 whatever the model, tuning or features; the test ROC-AUC 0.54 has a 95% interval of 0.40–0.67. The synthetic Attrition label has very little signal.

## Run it
```bash
pip install -r requirements.txt
python week3_analysis.py
```
or open `week3_unsupervised_ml.ipynb` (outputs included).

## Files
| file / folder | what it is |
|---|---|
| `week3_analysis.py` | full workflow as one script |
| `week3_unsupervised_ml.ipynb` | same workflow with outputs |
| `data/employee_data.csv` | dataset |
| `results/` | all metric tables, cluster assignments (CSV) |
| `figures/` | 8 plots |
| `Week3_Report.docx` | written report |
