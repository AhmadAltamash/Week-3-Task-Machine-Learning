"""
Week 3 - Unsupervised Learning & Model Evaluation

Part A (no labels used): PCA, K-Means, hierarchical clustering, and how to
        judge clusters when there is no answer key.
Part B (labels available): cross-validation, hyperparameter tuning,
        confusion matrix, precision, recall, F1, ROC-AUC for predicting Attrition.

Run:  python week3_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from scipy.stats import chi2_contingency

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_curve)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, GridSearchCV,
                                     cross_validate, cross_val_predict)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
for folder in ("figures", "results"):
    os.makedirs(folder, exist_ok=True)

SEED = 42
NUM_COLS = ["Age", "Experience_Years", "Monthly_Income",
            "Satisfaction_Score", "Weekly_Work_Hours"]
CAT_COLS = ["Education", "Department", "City", "Job_Level"]


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
section("1. LOAD DATA")
df = pd.read_csv("data/employee_data.csv")
print("shape:", df.shape, "| duplicates:", df.duplicated().sum())
print("missing:", df.isnull().sum()[df.isnull().sum() > 0].to_dict())

# ===============================================================
#  PART A - UNSUPERVISED LEARNING (Attrition is NOT used here)
# ===============================================================

# ---------------------------------------------------------------
# 2. Prepare the clustering features
# ---------------------------------------------------------------
section("2. PREPARE CLUSTERING FEATURES")
prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler())])
Xs = prep.fit_transform(df[NUM_COLS])           # no labels involved, so using all rows is fine
print("clustering on", NUM_COLS)
print("\ncorrelation between the clustering features:")
corr = df[NUM_COLS].corr().round(2)
print(corr.to_string())

# why scaling matters: income is ~100,000x bigger than satisfaction in raw units
raw = SimpleImputer(strategy="median").fit_transform(df[NUM_COLS])
lab_raw = KMeans(3, n_init=20, random_state=SEED).fit_predict(raw)
lab_std = KMeans(3, n_init=20, random_state=SEED).fit_predict(Xs)
income_only = KMeans(3, n_init=20, random_state=SEED).fit_predict(raw[:, [2]])
print("\nK-Means (K=3) agreement, adjusted Rand index:")
print("  raw features vs income alone:   ", round(adjusted_rand_score(lab_raw, income_only), 3))
print("  scaled features vs raw features:", round(adjusted_rand_score(lab_std, lab_raw), 3))
print("  -> on raw data the clusters are basically just income bands")


# ---------------------------------------------------------------
# 3. PCA (dimensionality reduction)
# ---------------------------------------------------------------
section("3. PCA")
pca_full = PCA(random_state=SEED).fit(Xs)
evr = pca_full.explained_variance_ratio_
print("explained variance per component:", evr.round(3).tolist())
print("cumulative:", np.cumsum(evr).round(3).tolist())
loadings = pd.DataFrame(pca_full.components_[:2].T, index=NUM_COLS,
                        columns=["PC1", "PC2"]).round(2)
print("\nloadings (how each feature contributes):")
print(loadings.to_string())
pca2 = PCA(n_components=2, random_state=SEED)
Xp = pca2.fit_transform(Xs)
loadings.to_csv("results/pca_loadings.csv")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(range(1, 6), evr, color="#4C72B0", label="per component")
ax[0].plot(range(1, 6), np.cumsum(evr), "o-", color="#C44E52", label="cumulative")
ax[0].set_xlabel("principal component"); ax[0].set_ylabel("explained variance")
ax[0].set_title("PCA scree plot"); ax[0].legend()
im = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax[1].set_xticks(range(5)); ax[1].set_yticks(range(5))
ax[1].set_xticklabels(NUM_COLS, rotation=40, ha="right"); ax[1].set_yticklabels(NUM_COLS)
for i in range(5):
    for j in range(5):
        ax[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[1].set_title("Correlation of clustering features")
plt.tight_layout()
plt.savefig("figures/01_pca_and_correlation.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 4. K-Means: choosing K
# ---------------------------------------------------------------
section("4. K-MEANS: CHOOSING K")
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=20, random_state=SEED).fit(Xs)
    rows.append({"K": k, "Inertia": km.inertia_,
                 "Silhouette": silhouette_score(Xs, km.labels_),
                 "Calinski_Harabasz": calinski_harabasz_score(Xs, km.labels_),
                 "Davies_Bouldin": davies_bouldin_score(Xs, km.labels_)})
kres = pd.DataFrame(rows)

# stability: fit on random 80% subsets, label everyone, compare runs with ARI
rng = np.random.RandomState(SEED)
stab = []
for k in range(2, 9):
    runs = []
    for r in range(10):
        idx = rng.choice(len(Xs), int(0.8 * len(Xs)), replace=False)
        runs.append(KMeans(k, n_init=5, random_state=r).fit(Xs[idx]).predict(Xs))
    aris = [adjusted_rand_score(runs[i], runs[j])
            for i in range(10) for j in range(i + 1, 10)]
    stab.append(np.mean(aris))
kres["Stability_ARI"] = stab
print(kres.round(3).to_string(index=False))
kres.round(4).to_csv("results/kmeans_evaluation.csv", index=False)
best_k = int(kres.loc[kres.Silhouette.idxmax(), "K"])
print("\nK with the highest silhouette:", best_k)

fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for a, col, title in zip(ax.ravel(),
                         ["Inertia", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"],
                         ["Inertia (elbow plot, lower = tighter)", "Silhouette (higher = better)",
                          "Calinski-Harabasz (higher = better)", "Davies-Bouldin (lower = better)"]):
    a.plot(kres.K, kres[col], "o-")
    a.set_title(title, fontsize=10); a.set_xlabel("K")
plt.tight_layout()
plt.savefig("figures/02_kmeans_k_selection.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 5. Silhouette profile for the chosen K
# ---------------------------------------------------------------
section("5. SILHOUETTE PROFILE")
km = KMeans(best_k, n_init=20, random_state=SEED).fit(Xs)
km_labels = km.labels_
sil = silhouette_samples(Xs, km_labels)
for c in range(best_k):
    print(f"cluster {c}: size {np.sum(km_labels == c)}, mean silhouette {sil[km_labels == c].mean():.3f}, "
          f"share below 0: {np.mean(sil[km_labels == c] < 0):.1%}")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
y0 = 0
for c in range(best_k):
    vals = np.sort(sil[km_labels == c])
    ax.fill_betweenx(np.arange(y0, y0 + len(vals)), 0, vals, alpha=0.75, label=f"cluster {c}")
    y0 += len(vals) + 10
ax.axvline(sil.mean(), color="red", ls="--", label=f"mean = {sil.mean():.2f}")
ax.set_xlabel("silhouette value"); ax.set_yticks([])
ax.set_title(f"Silhouette profile, K = {best_k}"); ax.legend()
plt.tight_layout()
plt.savefig("figures/03_silhouette_profile.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 6. Hierarchical clustering
# ---------------------------------------------------------------
section("6. HIERARCHICAL CLUSTERING")
hrows = []
dists = pdist(Xs)
for method in ["ward", "complete", "average"]:
    coph = cophenet(linkage(Xs, method=method), dists)[0]
    for k in range(2, 7):
        lab = AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(Xs)
        hrows.append({"Linkage": method, "K": k,
                      "Silhouette": silhouette_score(Xs, lab),
                      "Cophenetic": coph,
                      "Smallest_cluster": int(np.bincount(lab).min())})
hres = pd.DataFrame(hrows)
print(hres.round(3).to_string(index=False))
hres.round(4).to_csv("results/hierarchical_evaluation.csv", index=False)

ward_lab = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(Xs)
print(f"\nagreement (ARI) between K-Means and Ward at K={best_k}:",
      round(adjusted_rand_score(km_labels, ward_lab), 3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for method, mk in zip(["ward", "complete", "average"], ["o", "s", "^"]):
    s = hres[hres.Linkage == method]
    ax[0].plot(s.K, s.Silhouette, marker=mk, label=method)
ax[0].plot(kres.K[kres.K <= 6], kres.Silhouette[kres.K <= 6], "k--", label="k-means")
ax[0].set_xlabel("number of clusters"); ax[0].set_ylabel("silhouette")
ax[0].set_title("Silhouette: hierarchical linkages vs K-Means"); ax[0].legend()
Z = linkage(Xs, method="ward")
dendrogram(Z, truncate_mode="lastp", p=20, ax=ax[1], leaf_rotation=90, leaf_font_size=8,
           color_threshold=Z[-best_k + 1, 2] - 1e-6 if best_k > 1 else None)
ax[1].set_title("Ward dendrogram (last 20 merges)")
ax[1].set_ylabel("merge distance"); ax[1].set_xlabel("(number of points in branch)")
plt.tight_layout()
plt.savefig("figures/04_hierarchical.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 7. Interpreting the clusters
# ---------------------------------------------------------------
section("7. WHAT ARE THE CLUSTERS?")
out = df.copy()
out["Cluster"] = km_labels
prof = out.groupby("Cluster")[NUM_COLS].mean().round(1)
prof["Size"] = out.groupby("Cluster").size()
print("average value per cluster (original units):")
print(prof.to_string())
prof.to_csv("results/cluster_profiles.csv")

print("\njob level inside each cluster (row %):")
jl = pd.crosstab(out["Cluster"], out["Job_Level"], normalize="index").round(2)
print(jl.to_string())

print("\nattrition rate per cluster (Attrition was NOT used to build clusters):")
att = out.groupby("Cluster")["Attrition"].agg(["mean", "sum", "count"]).round(3)
att.columns = ["Attrition_rate", "Left", "Size"]
print(att.to_string())
chi2, pval, _, _ = chi2_contingency(pd.crosstab(out["Cluster"], out["Attrition"]))
print(f"chi-square test, cluster vs attrition: chi2 = {chi2:.2f}, p = {pval:.4f}")
att.to_csv("results/cluster_attrition.csv")

# external check against a label that exists but was not used (Job_Level)
print("\nexternal validation against Job_Level (a label the clustering never saw):")
print("  adjusted Rand index:", round(adjusted_rand_score(out["Job_Level"], km_labels), 3))
print("  normalized mutual info:", round(normalized_mutual_info_score(out["Job_Level"], km_labels), 3))
print("  (0 = no relation, 1 = perfect match)")

out["PC1"], out["PC2"] = Xp[:, 0], Xp[:, 1]
out.to_csv("results/cluster_assignments.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for c in range(best_k):
    m = out.Cluster == c
    ax[0].scatter(out.PC1[m], out.PC2[m], s=14, alpha=0.7, label=f"cluster {c}")
ax[0].set_title("K-Means clusters in PCA space"); ax[0].legend()
colors = {"Junior": "#4C72B0", "Mid": "#55A868", "Senior": "#DD8452", "Lead": "#C44E52"}
for lvl, col in colors.items():
    m = out.Job_Level == lvl
    ax[1].scatter(out.PC1[m], out.PC2[m], s=14, alpha=0.7, color=col, label=lvl)
ax[1].set_title("Same points coloured by Job_Level"); ax[1].legend()
for v, col, nm in [(0, "#4C72B0", "stayed"), (1, "#C44E52", "left")]:
    m = out.Attrition == v
    ax[2].scatter(out.PC1[m], out.PC2[m], s=14, alpha=0.5, color=col, label=nm)
ax[2].set_title("Same points coloured by Attrition"); ax[2].legend()
for a in ax:
    a.set_xlabel(f"PC1 ({evr[0]:.0%})"); a.set_ylabel(f"PC2 ({evr[1]:.0%})")
plt.tight_layout()
plt.savefig("figures/05_pca_clusters.png", dpi=110)
plt.close()


# ===============================================================
#  PART B - SUPERVISED MODEL EVALUATION (predict Attrition)
# ===============================================================

def make_prep(num_cols=NUM_COLS):
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, CAT_COLS)])


def pipe(model, with_pca=False, num_cols=NUM_COLS):
    steps = [("prep", make_prep(num_cols))]
    if with_pca:
        steps.append(("pca", PCA(random_state=SEED)))
    steps.append(("model", model))
    return Pipeline(steps)


# ---------------------------------------------------------------
# 8. Split
# ---------------------------------------------------------------
section("8. SUPERVISED: SPLIT")
X = df.drop(columns=["Attrition"])
y = df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)
print("train:", X_train.shape, "test:", X_test.shape,
      "| leavers: train", int(y_train.sum()), "test", int(y_test.sum()))


# ---------------------------------------------------------------
# 9. Compare models with repeated cross-validation (train data only)
# ---------------------------------------------------------------
section("9. CROSS-VALIDATION OF SEVEN MODELS (5 folds x 3 repeats)")
cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
candidates = {
    "Baseline (always stayed)": pipe(DummyClassifier(strategy="most_frequent")),
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    "PCA + Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), with_pca=True),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED)),
    "Random Forest": pipe(RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                                 class_weight="balanced", random_state=SEED)),
    "KNN": pipe(KNeighborsClassifier(n_neighbors=21)),
    "Gradient Boosting": pipe(HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05,
                                                             max_iter=100, random_state=SEED)),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_rows = []
for name, model in candidates.items():
    r = cross_validate(model, X_train, y_train, cv=cv_rep, scoring=scoring)
    row = {"Model": name}
    for s in scoring:
        row[s] = r[f"test_{s}"].mean()
    row["roc_auc_std"] = r["test_roc_auc"].std()
    cv_rows.append(row)
cv_df = pd.DataFrame(cv_rows)
print(cv_df.round(3).to_string(index=False))
cv_df.round(4).to_csv("results/cross_validation_comparison.csv", index=False)


# ---------------------------------------------------------------
# 10. Can extra features rescue the classifiers?
#     Three engineered features, tested with the same repeated CV.
# ---------------------------------------------------------------
section("10. FEATURE ENGINEERING ATTEMPT")


def add_features(d):
    d = d.copy()
    d["Income_per_Exp_Year"] = d["Monthly_Income"] / (d["Experience_Years"] + 1)
    d["Overwork"] = (d["Weekly_Work_Hours"] > 45).astype(float)
    d["Low_Satisfaction"] = (d["Satisfaction_Score"] < 5).astype(float)
    return d


X_train_fe = add_features(X_train)
NUM_FE = NUM_COLS + ["Income_per_Exp_Year", "Overwork", "Low_Satisfaction"]
fe_rows = []
for name, mdl in [("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
                  ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                                           class_weight="balanced", random_state=SEED))]:
    before = cross_validate(pipe(mdl), X_train, y_train, cv=cv_rep, scoring="roc_auc")["test_score"]
    after = cross_validate(pipe(mdl, num_cols=NUM_FE), X_train_fe, y_train, cv=cv_rep, scoring="roc_auc")["test_score"]
    fe_rows.append({"Model": name, "AUC_original_features": before.mean(),
                    "AUC_with_new_features": after.mean(), "Std": after.std()})
fe_df = pd.DataFrame(fe_rows)
print(fe_df.round(3).to_string(index=False))
fe_df.round(4).to_csv("results/feature_engineering_check.csv", index=False)


# ---------------------------------------------------------------
# 11. Hyperparameter tuning (GridSearchCV, scored on ROC-AUC)
# ---------------------------------------------------------------
section("11. HYPERPARAMETER TUNING WITH GridSearchCV")
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grids = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                            {"model__C": [0.001, 0.01, 0.1, 1, 10]}),
    "PCA + Logistic Regression": (pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), with_pca=True),
                                  {"pca__n_components": [2, 4, 6, 8, 12],
                                   "model__C": [0.01, 0.1, 1]}),
    "Random Forest": (pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  random_state=SEED)),
                      {"model__max_depth": [2, 3, 4, 6],
                       "model__min_samples_leaf": [2, 5, 10, 20]}),
    "KNN": (pipe(KNeighborsClassifier()),
            {"model__n_neighbors": [5, 11, 21, 31, 41]}),
}
tuned, tune_rows = {}, []
for name, (model, grid) in grids.items():
    gs = GridSearchCV(model, grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    tuned[name] = gs.best_estimator_
    tune_rows.append({"Model": name, "Combinations": len(gs.cv_results_["params"]),
                      "Best_params": str(gs.best_params_), "Best_CV_ROC_AUC": gs.best_score_})
    print(f"{name:<28} best CV AUC {gs.best_score_:.3f}  {gs.best_params_}")
tune_df = pd.DataFrame(tune_rows)
tune_df.round(4).to_csv("results/hyperparameter_tuning.csv", index=False)

# how much does one hyperparameter matter? (random forest depth)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
Cs = [0.001, 0.01, 0.1, 1, 10]
cs = []
for C in Cs:
    r = cross_validate(pipe(LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
                       X_train, y_train, cv=cv5, scoring="roc_auc")
    cs.append(r["test_score"])
ax[0].errorbar(range(len(Cs)), [c.mean() for c in cs], yerr=[c.std() for c in cs], fmt="o-", capsize=4)
ax[0].set_xticks(range(len(Cs))); ax[0].set_xticklabels(Cs)
ax[0].axhline(0.5, color="gray", ls=":")
ax[0].set_xlabel("C (smaller = stronger regularization)"); ax[0].set_ylabel("CV ROC-AUC")
ax[0].set_title("Logistic Regression: effect of C (mean \u00b1 std over folds)")
depths = [2, 3, 4, 6, 10]
ds = []
for d in depths:
    r = cross_validate(pipe(RandomForestClassifier(n_estimators=200, max_depth=d, min_samples_leaf=5,
                                                   class_weight="balanced", random_state=SEED)),
                       X_train, y_train, cv=cv5, scoring="roc_auc")
    ds.append(r["test_score"])
ax[1].errorbar(range(len(depths)), [c.mean() for c in ds], yerr=[c.std() for c in ds], fmt="o-", capsize=4, color="#55A868")
ax[1].set_xticks(range(len(depths))); ax[1].set_xticklabels(depths)
ax[1].axhline(0.5, color="gray", ls=":")
ax[1].set_xlabel("max_depth"); ax[1].set_ylabel("CV ROC-AUC")
ax[1].set_title("Random Forest: effect of depth (mean \u00b1 std over folds)")
plt.tight_layout()
plt.savefig("figures/06_tuning_effect.png", dpi=110)
plt.close()


# ---------------------------------------------------------------
# 12. Pick the final model and set a decision threshold (train data only)
# ---------------------------------------------------------------
section("12. FINAL MODEL AND THRESHOLD")
best_name = tune_df.loc[tune_df.Best_CV_ROC_AUC.idxmax(), "Model"]
final = tuned[best_name]
print("selected model (highest CV ROC-AUC):", best_name)

oof = cross_val_predict(final, X_train, y_train, cv=cv5, method="predict_proba")[:, 1]
ths = np.linspace(0.2, 0.8, 61)
f1s = [f1_score(y_train, oof >= t) for t in ths]
best_t = float(ths[int(np.argmax(f1s))])
print(f"threshold with the best out-of-fold F1: {best_t:.2f} (F1 {max(f1s):.3f})")


# ---------------------------------------------------------------
# 13. Final evaluation on the untouched test set
# ---------------------------------------------------------------
section("13. TEST SET EVALUATION")
final.fit(X_train, y_train)
prob = final.predict_proba(X_test)[:, 1]


def report(y_true, prob, t):
    p = (prob >= t).astype(int)
    return {"Threshold": round(t, 2),
            "Accuracy": accuracy_score(y_true, p),
            "Precision": precision_score(y_true, p, zero_division=0),
            "Recall": recall_score(y_true, p, zero_division=0),
            "F1": f1_score(y_true, p, zero_division=0)}


test_rows = [{"Setting": "default threshold 0.50", **report(y_test, prob, 0.5)},
             {"Setting": "tuned threshold", **report(y_test, prob, best_t)}]
test_df = pd.DataFrame(test_rows)
auc = roc_auc_score(y_test, prob)
test_df["ROC_AUC"] = auc
print(test_df.round(3).to_string(index=False))
test_df.round(4).to_csv("results/test_metrics.csv", index=False)

# how sure can we be about that AUC? bootstrap the test set
rng = np.random.RandomState(SEED)
boots = []
yt = y_test.to_numpy()
for _ in range(2000):
    i = rng.randint(0, len(yt), len(yt))
    if yt[i].min() != yt[i].max():
        boots.append(roc_auc_score(yt[i], prob[i]))
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\ntest ROC-AUC = {auc:.3f}, 95% bootstrap interval [{lo:.3f}, {hi:.3f}]")
print("baseline test accuracy (always 'stayed'):", round(1 - yt.mean(), 3))

# confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, t, title in zip(ax, [0.5, best_t], ["default threshold 0.50", f"tuned threshold {best_t:.2f}"]):
    cm = confusion_matrix(y_test, (prob >= t).astype(int))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["stayed", "left"]); a.set_yticklabels(["stayed", "left"])
    a.set_xlabel("predicted"); a.set_ylabel("actual"); a.set_title(title)
    tn, fp, fn, tp = cm.ravel()
    print(f"{title:<26} TN={tn} FP={fp} FN={fn} TP={tp}")
    if t == 0.5:
        pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                     columns=["Predicted 0", "Predicted 1"]).to_csv("results/confusion_matrix.csv")
plt.tight_layout()
plt.savefig("figures/07_confusion_matrices.png", dpi=110)
plt.close()

# ROC and precision-recall curves for all tuned models
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for name, model in tuned.items():
    if name != best_name:
        model.fit(X_train, y_train)
    pr = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pr)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, pr):.2f})")
    p_, r_, _ = precision_recall_curve(y_test, pr)
    ax[1].plot(r_, p_, label=name)
ax[0].plot([0, 1], [0, 1], "k--", label="random guessing")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
ax[0].set_title("ROC curves (test set)"); ax[0].legend(fontsize=8)
ax[1].axhline(y_test.mean(), color="k", ls="--", label="share of leavers")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title("Precision-recall curves (test set)"); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/08_roc_pr_curves.png", dpi=110)
plt.close()

section("DONE - tables in results/, plots in figures/")
