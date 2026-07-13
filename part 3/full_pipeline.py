"""
Insurance Dataset — Part 2 (baseline) + Part 3 (Ensembles, Tuning, Full ML Pipeline)
=====================================================================================
Dataset: insurance.csv (medical insurance cost data)
Sampled to 500 rows as requested. All 7 original columns used; after one-hot
encoding the categorical columns (sex, smoker, region) the feature matrix has
8 columns, which is what "columns" means here since the raw dataset only has
7 fields to begin with.

Classification target (y_clf): "high_cost" = 1 if expenses > median(expenses), else 0.
This turns the regression dataset into a balanced binary classification problem,
which is what the Part 2 / Part 3 tasks require (ROC-AUC, classifiers, etc).
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

RANDOM_STATE = 42
results = {}

# ---------------------------------------------------------------------------
# PART 2 (recreated minimally, since Part 3 depends on its outputs)
# ---------------------------------------------------------------------------
df_full = pd.read_csv("/mnt/user-data/uploads/insurance.csv")
df = df_full.sample(n=500, random_state=RANDOM_STATE).reset_index(drop=True)

# Classification target: high-cost claim (median split)
median_cost = df["expenses"].median()
df["high_cost"] = (df["expenses"] > median_cost).astype(int)

# One-hot encode categoricals
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

feature_cols = [c for c in df_enc.columns if c not in ("expenses", "high_cost")]
X = df_enc[feature_cols].astype(float)
y_clf = df_enc["high_cost"]

results["n_rows_sampled"] = len(df)
results["n_features"] = len(feature_cols)
results["feature_cols"] = feature_cols
results["median_cost"] = float(median_cost)

X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# Part 2 baseline model: Logistic Regression
logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
logreg.fit(X_train_scaled, y_clf_train)
logreg_train_acc = accuracy_score(y_clf_train, logreg.predict(X_train_scaled))
logreg_test_acc = accuracy_score(y_clf_test, logreg.predict(X_test_scaled))
logreg_test_auc = roc_auc_score(y_clf_test, logreg.predict_proba(X_test_scaled)[:, 1])
results["logreg"] = dict(train_acc=logreg_train_acc, test_acc=logreg_test_acc, test_auc=logreg_test_auc)

# ---------------------------------------------------------------------------
# TASK 1: Decision Tree baseline (unconstrained)
# ---------------------------------------------------------------------------
dt_uncon = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_uncon.fit(X_train_scaled, y_clf_train)
dt_uncon_train_acc = accuracy_score(y_clf_train, dt_uncon.predict(X_train_scaled))
dt_uncon_test_acc = accuracy_score(y_clf_test, dt_uncon.predict(X_test_scaled))
results["dt_unconstrained"] = dict(train_acc=dt_uncon_train_acc, test_acc=dt_uncon_test_acc,
                                    gap=dt_uncon_train_acc - dt_uncon_test_acc)

# ---------------------------------------------------------------------------
# TASK 2: Controlled Decision Tree
# ---------------------------------------------------------------------------
dt_con = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE)
dt_con.fit(X_train_scaled, y_clf_train)
dt_con_train_acc = accuracy_score(y_clf_train, dt_con.predict(X_train_scaled))
dt_con_test_acc = accuracy_score(y_clf_test, dt_con.predict(X_test_scaled))
results["dt_controlled"] = dict(train_acc=dt_con_train_acc, test_acc=dt_con_test_acc,
                                 gap=dt_con_train_acc - dt_con_test_acc)

# ---------------------------------------------------------------------------
# TASK 3: Gini vs Entropy
# ---------------------------------------------------------------------------
dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_STATE)
dt_gini.fit(X_train_scaled, y_clf_train)
dt_gini_test_acc = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=RANDOM_STATE)
dt_entropy.fit(X_train_scaled, y_clf_train)
dt_entropy_test_acc = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

results["gini_vs_entropy"] = dict(gini_test_acc=dt_gini_test_acc, entropy_test_acc=dt_entropy_test_acc)

# ---------------------------------------------------------------------------
# TASK 4: Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
top5 = importances.head(5)
bottom5 = importances.tail(5)

results["rf"] = dict(train_acc=rf_train_acc, test_acc=rf_test_acc, test_auc=rf_test_auc,
                      top5=top5.to_dict(), bottom5=bottom5.to_dict(),
                      all_importances=importances.to_dict())

# ---------------------------------------------------------------------------
# TASK 4a: Gradient Boosting
# ---------------------------------------------------------------------------
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
results["gb"] = dict(train_acc=gb_train_acc, test_acc=gb_test_acc, test_auc=gb_test_auc)

# ---------------------------------------------------------------------------
# TASK 4b: Feature ablation study
# ---------------------------------------------------------------------------
lowest5_features = bottom5.index.tolist()
keep_features = [c for c in feature_cols if c not in lowest5_features]

X_train_reduced = X_train_scaled[keep_features]
X_test_reduced = X_test_scaled[keep_features]

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_reduced.fit(X_train_reduced, y_clf_train)
rf_reduced_test_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

results["ablation"] = dict(
    removed_features=lowest5_features,
    kept_features=keep_features,
    full_model_auc=rf_test_auc,
    reduced_model_auc=rf_reduced_test_auc,
    auc_delta=rf_test_auc - rf_reduced_test_auc,
)

# ---------------------------------------------------------------------------
# TASK 5: Cross-validated comparison
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_models = {
    "Logistic Regression": logreg,
    "Decision Tree (depth=5)": dt_con,
    "Random Forest": rf,
    "Gradient Boosting": gb,
}

cv_results = {}
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")
    cv_results[name] = dict(mean_auc=scores.mean(), std_auc=scores.std())

results["cv_comparison"] = cv_results

# ---------------------------------------------------------------------------
# TASK 6: GridSearchCV hyperparameter tuning
# ---------------------------------------------------------------------------
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=RANDOM_STATE),
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_clf_train)  # unscaled — pipeline handles scaling

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
total_fits = n_configs * 5

results["grid_search"] = dict(
    best_params=grid_search.best_params_,
    best_score=grid_search.best_score_,
    n_configs=n_configs,
    total_fits=total_fits,
)

best_pipeline = grid_search.best_estimator_
best_pipeline_test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
results["grid_search"]["best_pipeline_test_auc"] = best_pipeline_test_auc

# ---------------------------------------------------------------------------
# TASK 7: Manual learning curve
# ---------------------------------------------------------------------------
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_rows = []
n_train = len(X_train)

for f in fractions:
    n_sub = int(f * n_train)
    X_sub = X_train.iloc[:n_sub]
    y_sub = y_clf_train.iloc[:n_sub]

    lc_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(random_state=RANDOM_STATE, **{
            k.replace("randomforestclassifier__", ""): v
            for k, v in grid_search.best_params_.items()
        }),
    )
    lc_pipeline.fit(X_sub, y_sub)

    train_auc = roc_auc_score(y_sub, lc_pipeline.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, lc_pipeline.predict_proba(X_test)[:, 1])
    learning_curve_rows.append((f, n_sub, train_auc, test_auc))

results["learning_curve"] = learning_curve_rows

# ---------------------------------------------------------------------------
# TASK 8: Serialize best model
# ---------------------------------------------------------------------------
joblib.dump(best_pipeline, "/home/claude/best_model.pkl")

# Reload and predict on two hand-crafted rows
reloaded_model = joblib.load("/home/claude/best_model.pkl")
hand_crafted_rows = pd.DataFrame([
    {"age": 45, "bmi": 32.0, "children": 2, "sex_male": 1, "smoker_yes": 1,
     "region_northwest": 0, "region_southeast": 1, "region_southwest": 0},
    {"age": 22, "bmi": 21.5, "children": 0, "sex_male": 0, "smoker_yes": 0,
     "region_northwest": 1, "region_southeast": 0, "region_southwest": 0},
])[feature_cols]
reload_predictions = reloaded_model.predict(hand_crafted_rows)
results["reload_predictions"] = reload_predictions.tolist()

# ---------------------------------------------------------------------------
# Summary comparison table (Parts 2 + 3)
# ---------------------------------------------------------------------------
summary_rows = []
model_test_auc_map = {
    "Logistic Regression": logreg_test_auc,
    "Decision Tree (depth=5)": roc_auc_score(y_clf_test, dt_con.predict_proba(X_test_scaled)[:, 1]),
    "Random Forest": rf_test_auc,
    "Gradient Boosting": gb_test_auc,
}
for name, cvr in cv_results.items():
    summary_rows.append({
        "model": name,
        "cv_mean_auc": cvr["mean_auc"],
        "cv_std_auc": cvr["std_auc"],
        "test_auc": model_test_auc_map[name],
    })
results["summary_table"] = summary_rows

# ---------------------------------------------------------------------------
# Reload-and-predict block (also runnable standalone)
# ---------------------------------------------------------------------------
"""
import joblib
import pandas as pd

loaded_model = joblib.load("best_model.pkl")
new_rows = pd.DataFrame([
    {"age": 45, "bmi": 32.0, "children": 2, "sex_male": 1, "smoker_yes": 1,
     "region_northwest": 0, "region_southeast": 1, "region_southwest": 0},
    {"age": 22, "bmi": 21.5, "children": 0, "sex_male": 0, "smoker_yes": 0,
     "region_northwest": 1, "region_southeast": 0, "region_southwest": 0},
])
predictions = loaded_model.predict(new_rows)
print(predictions)  # -> array([1, 0])
"""

with open("/home/claude/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("DONE")
print(json.dumps(results, indent=2, default=str))
