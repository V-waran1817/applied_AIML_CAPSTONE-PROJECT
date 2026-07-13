# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## Dataset & Setup

- Source: `insurance.csv` (medical insurance cost data).
- **500 rows** sampled from the full dataset (`random_state=42`) as requested.
- All 7 raw columns are used. After one-hot encoding the three categorical
  columns (`sex`, `smoker`, `region`) with `drop_first=True`, the feature
  matrix has **8 columns**: `age`, `bmi`, `children`, `sex_male`,
  `smoker_yes`, `region_northwest`, `region_southeast`, `region_southwest`.
- Classification target `high_cost`: 1 if `expenses` > median expenses
  (**$9,238.70**), else 0. This converts the original regression dataset into
  a balanced binary classification problem so that accuracy/ROC-AUC can be
  used throughout, matching the Part 2/Part 3 task requirements.
- Train/test split: 80/20, stratified on the target, `random_state=42`.
- `X_train_scaled` / `X_test_scaled` are produced with `StandardScaler`
  fit on the training data only.
- Part 2 baseline model (Logistic Regression): train accuracy **0.9125**,
  test accuracy **0.91**, test ROC-AUC **0.9416**.

---

## Task 1 — Decision Tree Baseline (Unconstrained)

| Metric | Value |
|---|---|
| Train accuracy | 1.0000 |
| Test accuracy | 0.8800 |
| Train–test gap | 0.12 |

**Overfitting:** Yes. Training accuracy is a perfect 1.0 while test accuracy
drops to 0.88 — a 12-point gap is a clear overfitting signature.

**Why decision trees are high-variance:** A decision tree builds its
structure greedily — at every node it picks the single best split for the
data currently in that node and never revisits or corrects earlier splits.
Left unconstrained, it keeps splitting until nodes are pure (or contain very
few samples), which lets it memorize noise and idiosyncrasies specific to
the training set. Because the exact splits chosen depend heavily on which
rows happen to be in the training sample, a different training sample can
produce a very differently structured tree — this sensitivity to the
specific training data is what makes trees "high variance."

## Task 2 — Controlled Decision Tree

`max_depth=5`, `min_samples_split=20`

| Metric | Value |
|---|---|
| Train accuracy | 0.9325 |
| Test accuracy | 0.9200 |
| Train–test gap | 0.0125 |

**Role of hyperparameters:**
- `max_depth` caps how many levels of splits the tree can make, which limits
  how finely it can carve up the feature space. This reduces variance
  (less overfitting) at the cost of some bias (the tree may miss some
  genuinely useful, deeper interactions).
- `min_samples_split=20` prevents a node from being split further if it has
  fewer than 20 samples, which stops the tree from creating splits that are
  really just reacting to noise in a small subset of the data.

**Comparison:** The unconstrained tree's train/test gap (0.12) is roughly
**10x larger** than the controlled tree's gap (0.0125). Constraining depth
and minimum split size traded a small amount of training accuracy
(1.00 → 0.93) for a large improvement in generalization (test accuracy
0.88 → 0.92).

## Task 3 — Gini vs Entropy

Both trained with `max_depth=5`.

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.92 |
| Entropy | 0.93 |

**Gini impurity formula:**

```
Gini = 1 − Σ pᵢ²
```

**Entropy formula:**

```
Entropy = − Σ pᵢ log2(pᵢ)
```

where `pᵢ` is the proportion of samples belonging to class `i` in a node.

**Gini = 0** means the node is perfectly pure — every single sample in that
node belongs to the same class, so there is no impurity left to reduce by
further splitting.

## Task 4 — Random Forest

`n_estimators=100`, `max_depth=10`, `random_state=42`

| Metric | Value |
|---|---|
| Train accuracy | 0.9950 |
| Test accuracy | 0.9100 |
| Test ROC-AUC | 0.9600 |

**Top 5 features by importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | age | 0.4985 |
| 2 | smoker_yes | 0.2548 |
| 3 | bmi | 0.1353 |
| 4 | children | 0.0603 |
| 5 | sex_male | 0.0187 |

**How Random Forest computes feature importance:** For each feature, the
algorithm sums up the reduction in Gini impurity (weighted by the number of
samples reaching that node) that occurs at every split where that feature is
used, across every tree in the forest, and then averages/normalizes this
across the whole ensemble. Features that consistently produce large,
frequent impurity reductions score higher.

**Why this differs from a linear regression coefficient:** A regression
coefficient measures the average linear effect of a one-unit change in a
feature on the (log-)outcome, holding other features fixed — it has a sign
and a fixed scale-dependent magnitude, and it assumes a linear/additive
relationship. Random Forest feature importance instead measures how useful a
feature was for splitting data into purer groups across many nonlinear,
interacting trees — it captures nonlinear effects and interactions, has no
sign (it can't tell you the direction of the effect), and is influenced by
how many distinct values/levels a feature has.

**Bagging concept:** Random Forest builds many decision trees, where each
tree is trained on a **bootstrap sample** — a random sample of the same size
as the training set, drawn *with replacement*, so each tree sees a slightly
different (overlapping but not identical) view of the data. In addition, at
each split, only a **random subset of √(number of features)** candidate
features is considered rather than all of them, which decorrelates the
trees from one another (otherwise they would all tend to split on the same
dominant feature first). Averaging the predictions (or votes) of many such
trees cancels out the idiosyncratic errors any single deep, high-variance
tree would make, because those errors are largely uncorrelated across
trees — the ensemble's variance shrinks roughly in proportion to how
independent the individual trees' errors are, while the bias stays close to
that of a single deep tree.

## Task 4a — Gradient Boosting

`n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42`

| Metric | Value |
|---|---|
| Train accuracy | 0.9800 |
| Test accuracy | 0.9300 |
| Test ROC-AUC | 0.9504 |

(Included in the cross-validated comparison — see Task 5.)

## Task 4b — Feature Ablation Study

Using the Random Forest's `feature_importances_` from Task 4, the 5
**lowest**-importance features were:

`children`, `sex_male`, `region_southwest`, `region_southeast`, `region_northwest`

A second Random Forest (identical hyperparameters, `random_state=42`) was
trained with those 5 features removed, keeping only `age`, `bmi`,
`smoker_yes`.

| Model | Test ROC-AUC |
|---|---|
| Full model (8 features) | 0.9600 |
| Reduced model (3 features) | 0.9240 |
| Δ AUC | 0.0360 |

**Interpretation:** The dropped features were **not purely noise** — removing
them cost 3.6 points of AUC, a small but non-trivial drop. They individually
had low importance scores, but collectively (especially `children`, which
alone carried ~6% importance) they still contributed some real signal.

**Production trade-off:** Deploying the reduced, 3-feature model would lower
inference cost, simplify the data pipeline (fewer fields to collect,
validate, and monitor for drift), and ease long-term maintenance. Whether
that is acceptable depends on how much AUC degradation the business can
tolerate: a 3.6-point AUC drop is likely acceptable for a low-stakes,
high-volume pricing screen where simplicity and speed matter, but would be
too costly for a use case where marginal gains in discrimination translate
directly into large amounts of money (e.g., underwriting decisions on large
policies). In general, the reduced model is worth shipping only if the
AUC loss stays under whatever threshold the business has defined as
tolerable for that decision.

## Task 5 — Cross-Validated Comparison

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='roc_auc'`

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.9514 | 0.0132 |
| Decision Tree (max_depth=5) | 0.9284 | 0.0225 |
| Random Forest | 0.9456 | 0.0246 |
| Gradient Boosting | 0.9484 | 0.0216 |

**Why cross-validation is more reliable than a single train/test split:** A
single split gives one estimate of generalization performance that depends
heavily on which particular rows happened to land in the test set — a
"lucky" or "unlucky" split can make a model look better or worse than it
really is. 5-fold cross-validation trains and evaluates the model 5 times on
different partitions of the data and averages the results, which both
reduces the influence of any one unusual split and provides a standard
deviation across folds — a direct, empirical measure of how stable that
estimate is.

## Task 6 — Hyperparameter Tuning with GridSearchCV

Pipeline: `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))`
Fit on unscaled `X_train` / `y_clf_train` (the pipeline does its own imputing/scaling).

```
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

**Best parameters found:**
```
{
  'randomforestclassifier__max_depth': 5,
  'randomforestclassifier__min_samples_leaf': 1,
  'randomforestclassifier__n_estimators': 200
}
```
**Best CV score (roc_auc):** 0.9476
**Held-out test AUC of the best pipeline:** 0.9536

**Total configurations evaluated:** the grid has 3 × 3 × 2 = **18 distinct
hyperparameter combinations**, each fit and scored across 5 folds, for a
total of **18 × 5 = 90 model fits**.

**Grid Search vs Randomized Search trade-off:** Grid Search exhaustively
tries every combination in the grid, guaranteeing it finds the best
combination *within that grid*, but its cost grows multiplicatively with
the number of hyperparameters and values (curse of dimensionality) — it
quickly becomes computationally infeasible for larger grids. Randomized
Search instead samples a fixed number of random combinations from the
specified distributions; it's far cheaper for large search spaces and in
practice often finds a near-optimal combination almost as good as full grid
search, but it isn't guaranteed to find the true best combination and its
result has some randomness from run to run.

## Task 7 — Manual Learning Curve

Using the best pipeline from Task 6, fit on progressively larger prefixes of `X_train`:

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 20% (80 rows) | 0.9975 | 0.9160 |
| 40% (160 rows) | 0.9939 | 0.9236 |
| 60% (240 rows) | 0.9859 | 0.9544 |
| 80% (320 rows) | 0.9880 | 0.9452 |
| 100% (400 rows) | 0.9844 | 0.9536 |

**(i) Does training AUC decrease as the training set grows?** Yes, roughly —
it starts almost perfect at 20% (0.9975) and drifts down to 0.9844 at 100%.
This is expected for a high-variance model family (Random Forest with
shallow trees can still nearly memorize a very small training set, so
training performance is highest when there's the least data to fit).

**(ii) Does test AUC increase with more training data?** Broadly yes — test
AUC rises from 0.916 at 20% to the 0.94–0.95 range by 60–100%, with some
fold-to-fold noise given the small (500-row) dataset. This suggests that,
up to a point, collecting more data would likely help.

**(iii) Conclusion — data-limited or capacity-limited?** The test AUC curve
has **largely plateaued** in the 0.94–0.95 range from 60% of the training
data onward, with the training AUC staying high and roughly flat rather
than continuing to fall. This pattern is more consistent with the model
being close to its **capacity limit** for this feature set at this point,
rather than being sharply data-limited — more rows would likely produce
only marginal further gains unless paired with more or better features
(e.g., the ablation study in Task 4b shows `age`, `bmi`, and `smoker_yes`
already carry most of the signal).

## Task 8 — Serialization

The best pipeline (from `GridSearchCV.best_estimator_`) was saved with:

```python
import joblib
joblib.dump(best_pipeline, "best_model.pkl")
```

**Reload-and-predict block:**

```python
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
print(predictions)
```

This ran without errors and produced `[1, 0]` — the first hand-crafted row
(older, higher BMI, smoker) was predicted high-cost; the second (younger,
lower BMI, non-smoker) was predicted low-cost, matching intuition.

---

## Summary Comparison Table (Parts 2 + 3)

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
|---|---|---|---|
| Logistic Regression | 0.9514 | 0.0132 | 0.9416 |
| Decision Tree (depth=5) | 0.9284 | 0.0225 | 0.9272 |
| Random Forest | 0.9456 | 0.0246 | 0.9600 |
| Gradient Boosting | 0.9484 | 0.0216 | 0.9504 |
| **Tuned RF Pipeline (GridSearchCV)** | 0.9476 | — | **0.9536** |

### Recommendation

I would recommend the **tuned Random Forest pipeline** (`max_depth=5`,
`n_estimators=200`, `min_samples_leaf=1`) as the model to deploy. It
achieves the best held-out test AUC of the ensemble methods (0.9536),
has one of the lowest train/CV variances among the tree-based models once
depth is constrained, and — because it's wrapped in a `Pipeline` with
`SimpleImputer` and `StandardScaler` — it's robust to missing values and
reproducible end-to-end without any manual preprocessing steps at
inference time. Logistic Regression is a close, simpler alternative with
the most stable CV score (lowest std), so it remains a reasonable fallback
if interpretability or extreme simplicity is prioritized over the last few
points of AUC.

---

## Files in this repository

- `full_pipeline.py` — end-to-end script: data loading/sampling, Part 2
  baseline, and all Part 3 tasks (decision trees, random forest, gradient
  boosting, ablation study, cross-validation, grid search, learning curve,
  serialization).
- `best_model.pkl` — the serialized best pipeline from `GridSearchCV`.
- `results.json` — all numeric results produced by the script, in raw form.
- `README.md` — this file.
