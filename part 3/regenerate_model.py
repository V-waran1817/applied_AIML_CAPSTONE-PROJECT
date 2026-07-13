"""
regenerate_model.py
--------------------
Run this on YOUR machine to (re)build best_model.pkl using the exact
scikit-learn / pandas / numpy versions installed there. This avoids the
"pickle from a different library version" problem that can make a
pre-built .pkl fail to load.

Usage:
    pip install -r requirements.txt      # optional but recommended
    python regenerate_model.py

Requires insurance.csv to be in the same folder as this script.
Produces: best_model.pkl
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# 1. Load and sample 500 rows
df_full = pd.read_csv("insurance.csv")
df = df_full.sample(n=500, random_state=RANDOM_STATE).reset_index(drop=True)

# 2. Build classification target: high-cost = above median expense
median_cost = df["expenses"].median()
df["high_cost"] = (df["expenses"] > median_cost).astype(int)

# 3. One-hot encode categoricals
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
feature_cols = [c for c in df_enc.columns if c not in ("expenses", "high_cost")]
X = df_enc[feature_cols].astype(float)
y = df_enc["high_cost"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 5. Pipeline + grid search (same grid as the original task)
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score_)

best_pipeline = grid_search.best_estimator_

# 6. Save, built with YOUR local library versions
joblib.dump(best_pipeline, "best_model.pkl")
print("Saved best_model.pkl")

# 7. Sanity check: reload immediately and predict
reloaded = joblib.load("best_model.pkl")
sample_rows = pd.DataFrame([
    {"age": 45, "bmi": 32.0, "children": 2, "sex_male": 1, "smoker_yes": 1,
     "region_northwest": 0, "region_southeast": 1, "region_southwest": 0},
    {"age": 22, "bmi": 21.5, "children": 0, "sex_male": 0, "smoker_yes": 0,
     "region_northwest": 1, "region_southeast": 0, "region_southwest": 0},
])[feature_cols]
print("Reload predictions:", reloaded.predict(sample_rows))
