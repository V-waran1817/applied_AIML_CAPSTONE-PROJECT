# Health Care Insurance Premium Prediction — Part 2: Regression & Classification Models

This part builds on the cleaned dataset from Part 1 (`cleaned_data.csv`, 500 rows) and produces two
models: a regression model predicting continuous medical `expenses`, and a binary classification model
predicting whether a policyholder is a "high-cost" case. All code is in
`Insurance_Premium_Prediction_Part2.ipynb`.

## 1. Feature and Label Definitions

- **Feature matrix `X`**: every column except `expenses` — `age`, `sex`, `bmi`, `children`, `smoker`,
  `region`.
- **Regression label `y_reg`**: `expenses` (continuous — the medical cost billed by insurance).
- **Classification label `y_clf`**: a binary "high-cost policyholder" flag, defined as
  `y_clf = (y_reg > y_reg.median()).astype(int)`. `1` = above-median expenses (high-cost), `0` =
  at-or-below median (lower-cost). This gives a business-meaningful, naturally balanced binary target
  rather than an arbitrary cutoff.

## 2. Categorical Encoding

`sex`, `smoker`, and `region` are all categorical, and **none of them have a natural order** — there's
no meaningful sense in which "male" is greater or less than "female", or "northeast" ranks above
"southwest". Label-encoding any of these would introduce a **false ordinal relationship**: the model
would incorrectly treat the encoded integers as if they had a real numeric distance/order, which could
bias the learned coefficients toward a made-up ranking. So all three were **one-hot encoded** using
`pd.get_dummies(..., drop_first=True)`. Dropping the first dummy column per feature avoids the
"dummy-variable trap" — with `k` categories, only `k-1` binary columns are needed since the last category
is fully implied when all the others are 0 (keeping all `k` would create perfect multicollinearity).

Final encoded feature columns: `age`, `bmi`, `children`, `sex_male`, `smoker_yes`, `region_northwest`,
`region_southeast`, `region_southwest` (8 features total; `female` and `northeast` are the implied
reference categories).

## 3. Leak-Free Train-Test Split and Scaling

Data was split 80/20 (`test_size=0.2, random_state=42`) into 400 training rows and 100 test rows,
**before** any scaling. `StandardScaler` was fit **only on `X_train`**, then used to transform both
`X_train` and `X_test`.

**Why fitting the scaler on the full dataset would be data leakage:** the scaler's mean and standard
deviation are statistics learned from data. If it were fit on the combined train+test set, information
about the test set's distribution would leak into the preprocessing step used to train the model — even
though the model itself never "sees" the test labels, the scaling would already be calibrated using
knowledge of data it's supposed to be evaluated on as if unseen. In a real deployment, the scaler has to
be fixed once from historical data and then applied to genuinely new records whose distribution isn't
known in advance, so fitting on the full dataset would give an unrealistically optimistic estimate of how
the pipeline performs.

## 4. Regression Model: Linear Regression

| Metric | Value |
|---|---|
| MSE | 30,967,557.63 |
| R² | 0.7618 |

**Coefficients (sorted by absolute magnitude, on standardized features):**

| Feature | Coefficient |
|---|---|
| `smoker_yes` | **+9,837.68** |
| `age` | +3,536.93 |
| `bmi` | +1,962.30 |
| `region_southeast` | −620.44 |
| `children` | +511.91 |
| `sex_male` | −468.10 |
| `region_southwest` | −315.12 |
| `region_northwest` | +37.82 |

**Top 3 features by |coefficient|:** `smoker_yes`, `age`, `bmi`.

**Interpreting the coefficients:** because the features are standardized, each coefficient represents the
change in predicted `expenses` (in dollars) associated with a one-standard-deviation increase in that
feature, holding everything else fixed.
- A **large positive coefficient** (e.g. `smoker_yes` at +9,837.68) means that feature strongly *raises*
  predicted expenses — being a smoker is associated with a huge jump in predicted medical cost.
- A **large negative coefficient** (e.g. `region_southeast` at −620.44) means that feature *lowers*
  predicted expenses relative to the reference category — living in the southeast is associated with
  somewhat lower predicted costs than the reference region (northeast), all else equal.

### Ridge Regression Comparison

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 30,967,557.63 | 0.7618 |
| Ridge (alpha=1.0) | 30,949,810.38 | 0.7619 |

The two models perform almost identically here. **Why Ridge can produce a different coefficient profile
than OLS:** Ridge adds an L2 penalty term (`alpha * sum(coefficients²)`) to the loss function, which
shrinks all coefficients toward zero, more aggressively for less-informative or correlated features. The
`alpha` parameter controls the *strength* of that shrinkage — higher `alpha` means more shrinkage
(coefficients pulled closer to zero, more bias but less variance), while `alpha=0` reduces Ridge to plain
OLS. With `alpha=1.0` and features that aren't highly collinear in this dataset, the shrinkage is mild,
which is why Ridge's MSE and R² barely differ from OLS — there wasn't much multicollinearity or
overfitting for Ridge to correct in the first place.

## 5. Classification Model: Logistic Regression

### Class Balance Check

`y_clf_train` value counts: **197 (class 0) / 203 (class 1)** — minority class share ≈ **49.3%**, well
above the 35% threshold in the brief. Because `y_clf` was constructed by splitting `expenses` at its own
median, it is close to perfectly balanced by design.

**Imbalance handling chosen:** since no real imbalance exists here, strict resampling (SMOTE) wasn't
necessary. As a robustness default, the model still uses **`class_weight='balanced'`** in the
`LogisticRegression` constructor rather than leaving weights uniform — this costs nothing when the classes
are already balanced but protects against any imbalance that could show up if the label were redefined
later (e.g. a different, non-median-based high-cost threshold). Because `class_weight` re-weights the
loss function rather than resampling rows, the **class counts before and after are identical**
(197 / 203) — there's nothing to resample when the data is already this close to even.

### Results (threshold = 0.5, C = 1.0)

**Confusion matrix:**

| | Predicted: Low-cost (0) | Predicted: High-cost (1) |
|---|---|---|
| **Actual: Low-cost (0)** | 44 (TN) | 9 (FP) |
| **Actual: High-cost (1)** | 3 (FN) | 44 (TP) |

**Classification report:**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (low-cost) | 0.94 | 0.83 | 0.88 | 53 |
| 1 (high-cost) | 0.83 | 0.94 | 0.88 | 47 |
| **Accuracy** | | | **0.88** | 100 |

**AUC:** **0.9771**

**Precision and Recall formulas:**

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

**Which metric matters more here?** This model is meant to flag "high-cost" policyholders — presumably
so an insurer can plan reserves, review pricing, or intervene early. In that context, a **false negative**
(failing to flag a genuinely high-cost policyholder) is more expensive than a **false positive** (flagging
someone who turns out not to be high-cost, which just costs a bit of unnecessary review). So **recall is
the more important metric** for this task.

**What the AUC of 0.9771 means:** if you randomly pick one high-cost policyholder and one low-cost
policyholder from the test set, the model ranks the actual high-cost one higher about 97.7% of the time.
This is very strong separation between the two classes — `smoker`, `age`, and `bmi` together give the
model a lot of predictive signal.

## 5b. Decision-Threshold Sensitivity

Precision, Recall, and F1 at five thresholds using the same fitted model's predicted probabilities:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.703 | 0.957 | 0.811 |
| 0.40 | 0.763 | 0.957 | 0.849 |
| 0.50 | 0.830 | 0.936 | 0.880 |
| 0.60 | 0.917 | 0.936 | 0.926 |
| 0.70 | **0.977** | 0.915 | **0.945** |

**Precision:** Precision = TP / (TP + FP)
**Recall:** Recall = TP / (TP + FN)

**F1-maximizing threshold:** **0.70** (Precision ≈ 0.977, Recall ≈ 0.915, F1 ≈ 0.945).

**Which metric matters more for this task?** As above, **recall** — missing a real high-cost
policyholder is costlier than over-flagging a low-cost one.

**Would we raise or lower the threshold?** Since recall matters more here, we would **lower** the
threshold below the F1-optimal point of 0.70 — toward 0.30–0.40, where recall peaks at ~0.957. **The
cost:** precision drops substantially (down to ~0.70–0.76 at those thresholds), meaning more false
positives — more policyholders incorrectly flagged as high-cost, requiring unnecessary manual review. This
trade-off is acceptable given that under-flagging a genuinely high-cost policyholder is the more expensive
mistake for the insurer.

## 6. Regularization Experiment (C=0.01 vs. C=1.0)

| Model | Precision | Recall | AUC |
|---|---|---|---|
| Logistic Regression (C=1.0) | 0.830 | 0.936 | 0.9771 |
| Logistic Regression (C=0.01) | 0.815 | 0.936 | 0.9707 |

**What `C` controls:** in scikit-learn's `LogisticRegression`, `C` is the **inverse of the regularization
strength** — it balances fitting the training data closely against keeping the coefficients small. A
**smaller `C`** (like 0.01) applies **stronger L2 regularization**, shrinking coefficients more
aggressively toward zero, which produces a simpler, less flexible decision boundary — helpful against
overfitting, but risky if pushed too far (underfitting).

**Did reducing C help or hurt?** It **slightly hurt** performance here — AUC dropped from 0.9771 to
0.9707 and precision dropped a little, while recall was unchanged. This suggests the baseline model
(C=1.0) wasn't meaningfully overfitting to begin with — the relationship between `smoker`, `age`, `bmi`,
and the target is strong and fairly clean, so stronger regularization mostly removed useful signal rather
than noise.

## 7. Bootstrap Confidence Interval for the AUC Difference

500 bootstrap resamples of the test set (with replacement, `random_state`-seeded) were used to compute
the distribution of the AUC difference (C=1.0 model minus C=0.01 model):

| Statistic | Value |
|---|---|
| Mean AUC difference | **+0.0062** |
| 95% CI lower bound (2.5th percentile) | **−0.0020** |
| 95% CI upper bound (97.5th percentile) | **+0.0189** |

**Does the interval exclude zero?** **No** — the 95% confidence interval **[−0.0020, +0.0189] includes
zero**.

**Interpretation:** although the C=1.0 model has a slightly higher AUC on average across bootstrap
resamples, this advantage **is not statistically reliable** at the 95% confidence level — some resamples
of the test set actually favor the C=0.01 model. In practice, this means the small AUC gap observed in
Section 6 shouldn't be over-interpreted as proof that C=1.0 is definitively better; the two models perform
comparably, and either could reasonably be chosen depending on other priorities (e.g. favoring the less
regularized model since it isn't underperforming, or the more regularized one for extra robustness on
noisier future data).

## How to Run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Insurance_Premium_Prediction_Part2.ipynb
```

or open `Insurance_Premium_Prediction_Part2.ipynb` in Jupyter/Colab and run all cells top to bottom. This
notebook reads `cleaned_data.csv`, the output of Part 1.
