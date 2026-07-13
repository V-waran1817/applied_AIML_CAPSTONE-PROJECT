# Health Care Insurance Premium Prediction — Part 1: Data Cleaning & EDA

## 1. Dataset Description

The dataset (`insurance.csv`) used for this task is a **500-row random sample** (drawn with
`random_state=42` for reproducibility) from the original Kaggle source of 1,338 records:
https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction. This 500-row sample represents
the client's data as delivered for this phase.

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Insurance holder's age in years |
| `sex` | categorical | Gender (`male` / `female`) |
| `bmi` | numeric | Body Mass Index |
| `children` | numeric | Number of dependent children |
| `smoker` | categorical | Whether the holder smokes (`yes` / `no`) |
| `region` | categorical | Residential region (`northeast`, `northwest`, `southeast`, `southwest`) |
| `expenses` | numeric | Individual medical costs billed by insurance — the target variable |

This dataset was chosen because it mixes numeric and categorical features that need distinct cleaning
treatment, and the target (`expenses`) is realistically skewed — a good stress test for the
cleaning/EDA pipeline before modeling.

All code for this part lives in `Insurance_Premium_Prediction_EDA_Part1.ipynb`.

## 2. Null Value Analysis

`df.isnull().sum()` and the corresponding percentage table showed **0 missing values in every column
(0.0% across the board)** on this 500-row sample. No column exceeded the 20% threshold, so no column was
dropped.

Because the numeric-column median-fill step is still part of the pipeline (guarding against nulls that
may appear in future raw batches from the client), it is implemented generically: any numeric column
under the 20% threshold has its nulls filled with `df[col].median()`. **We use the median rather than
the mean** because `expenses` and `children` are both right-skewed (see Section 5) — the mean in a
skewed column is pulled toward the extreme values in the tail, while the median is robust to that and
better represents the "typical" record. Using the mean to impute a skewed column would bias every
imputed row toward values that are not actually representative of most people in that column.

## 3. Duplicate Detection and Removal

`df.duplicated().sum()` found **0 duplicate rows** in this 500-row sample (the one duplicate present in
the full 1,338-row dataset was not selected in this sample). The dataset remained at **500 rows** after
`df.drop_duplicates()`, and null percentages were unchanged (still 0.0% for every column) since no rows
were removed.

## 4. Data Type Correction

No numeric column was actually stored as `object`/string in this dataset — the raw file arrived already
reasonably typed. To still fulfil dtype correction meaningfully:

- **`sex`, `smoker`, `region`** are low-cardinality repetitive strings (2, 2, and 4 unique values
  respectively) that were stored as generic string dtype. These were converted to **`category`** dtype,
  which is the semantically correct type for repetitive labels and is far cheaper to store.
- **`age`, `children`** are small positive integers stored as the default 64-bit integer. These were
  downcast to a smaller integer type with `pd.to_numeric(..., downcast='integer')`, since their value
  ranges never come close to needing 64 bits.

**Memory usage before conversion:** 97,728 bytes
**Memory usage after conversion:** 11,075 bytes
**Reduction:** ~86,653 bytes, an **88.7% reduction** in memory footprint.

## 5. Descriptive Statistics and Skewness

`df.describe()` summary statistics are produced in the notebook. Skewness per numeric column:

| Column | Skew |
|---|---|
| `expenses` | **1.480** |
| `children` | 0.960 |
| `bmi` | 0.300 |
| `age` | 0.077 |

**`expenses` has the highest absolute skewness (1.480)** — a strong **positive (right) skew**. This
means the bulk of policyholders in this sample have relatively low medical expenses, but a long tail of
individuals (largely smokers and/or older patients) have very high expenses, pulling the mean well above
the median.

**Consequence for imputation:** because the distribution's tail is on the high side, the **mean is
inflated by the tail** and would over-state a "typical" value. Imputing missing `expenses` values with
the mean would systematically overestimate costs for the mass of ordinary policyholders. The **median is
the more representative central-tendency measure** for this column, and is what we use (see Section 8a).

## 6. Outlier Detection (IQR Method)

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outlier rows |
|---|---|---|---|---|---|---|
| `bmi` | 26.58 | 35.20 | 8.63 | 13.64 | 48.14 | **2** (0.4%) |
| `expenses` | 4,525.09 | 15,946.08 | 11,420.98 | -12,606.38 | 33,077.55 | **59** (11.8%) |

Outliers were **not dropped**. For Part 2 (modeling), the plan is to **retain** the `bmi` outliers (only
2 rows here, clinically plausible — severe obesity is real and predictive of higher costs) but
**investigate capping (winsorizing) the `expenses` upper tail** before training linear models, since
almost 12% of rows sit above the upper IQR fence and could dominate the loss function for models
sensitive to large residuals (e.g. plain Linear Regression). Tree-based models (Decision Tree / Random
Forest / Gradient Boosting) are far less sensitive to this and can likely use the outliers as-is, so any
capping will be tested as an A/B comparison against the un-capped version rather than applied
unconditionally.

## 7. Visualizations

All plots are produced in `Insurance_Premium_Prediction_EDA_Part1.ipynb`.

1. **Line plot** — `expenses`, with rows sorted by `age`. Shows expenses fluctuate noisily but the floor
   of the distribution trends upward with age, while a subset of very high-cost rows (smokers) appear at
   every age band, creating visible spikes in the line.
2. **Bar chart** — mean `expenses` by `region`. Southeast has the highest mean expenses of the four
   regions in this sample, though the differences between regions are modest compared to the differences
   driven by smoking status.
3. **Histogram** — `expenses` (the most skewed column, skew = 1.48). The distribution is unimodal but
   strongly right-skewed: most values cluster under ~$15,000, with a long thinning tail extending out
   past $50,000.
4. **Scatter plot** — `age` vs. `expenses`, colored by `smoker`. There's a clear positive relationship
   for both groups (expenses rise with age), but **smokers form a visibly separate, much higher band**
   than non-smokers at every age — smoking status looks like the dominant driver of the spread, not just
   age alone.
5. **Box plot** — `expenses` by `smoker`. Smokers have a dramatically higher median and a much wider
   spread than non-smokers, with almost no overlap between the two groups' interquartile ranges — this
   is the single clearest categorical signal in the dataset.
6. **Correlation heat map** — numeric features only (`age`, `bmi`, `children`, `expenses`).

## 8. Correlation Heat Map — Interpretation

Correlation matrix (Pearson) of numeric columns:

| | age | bmi | children | expenses |
|---|---|---|---|---|
| **age** | 1.00 | 0.10 | 0.02 | **0.31** |
| **bmi** | 0.10 | 1.00 | 0.00 | 0.19 |
| **children** | 0.02 | 0.00 | 1.00 | 0.06 |
| **expenses** | 0.31 | 0.19 | 0.06 | 1.00 |

The highest-correlation pair among the **numeric** columns is **`age` and `expenses`** (r ≈ 0.31) — a
moderate positive relationship. This is plausibly **partly causal** (older bodies genuinely cost more to
treat), but it is unlikely to be the whole story: the scatter plot in Section 7 shows `smoker` status
(a categorical variable, not part of this numeric correlation matrix) creates a much bigger split in
expenses than age does on its own. A reasonable **alternative explanation** is that `age` is acting
partly as a **proxy/confound** — older individuals in this sample may also happen to have a different
smoking mix or higher BMI, and it's the combination (or the omitted `smoker` variable) doing more of the
actual work than age in isolation. This is why Part 2 should model `smoker`, `age`, and `bmi` jointly
rather than relying on the numeric correlation matrix alone to judge feature importance.

## 8a. Imputation Strategy Comparison (Mean vs. Median)

Applied to the two highest-|skew| numeric columns, computed **before** any imputation:

| Column | Mean | Median | Skew |
|---|---|---|---|
| `expenses` | $13,179.46 | $9,238.70 | +1.480 (positive/right skew) |
| `children` | 1.096 | 1.0 | +0.960 (positive/right skew) |

**Chosen strategy: median**, for both columns. Both `expenses` and `children` are **positively skewed**
— their means are pulled *upward* by a minority of high-value records (high-cost patients / large
families), so the mean overstates what's typical. The median is unaffected by that tail and better
represents the central tendency for imputation purposes. After applying `fillna()` with the median,
`isnull().sum()` confirmed **zero remaining nulls** in both columns (there were none to begin with in
this sample, but the imputation logic is in place for any future batch that does contain gaps).

## 8b. Spearman Rank Correlation vs. Pearson

Spearman matrix:

| | age | bmi | children | expenses |
|---|---|---|---|---|
| **age** | 1.00 | 0.10 | 0.03 | **0.57** |
| **bmi** | 0.10 | 1.00 | 0.00 | 0.11 |
| **children** | 0.03 | 0.00 | 1.00 | 0.14 |
| **expenses** | 0.57 | 0.11 | 0.14 | 1.00 |

**Top 3 pairs by |Spearman − Pearson|:**

| Pair | Pearson | Spearman | \|diff\| |
|---|---|---|---|
| `age` – `expenses` | 0.311 | 0.570 | **0.259** |
| `bmi` – `expenses` | 0.195 | 0.107 | 0.087 |
| `children` – `expenses` | 0.059 | 0.141 | 0.082 |

- **`age` – `expenses` (diff = 0.259):** |Spearman| (0.57) is much larger than |Pearson| (0.31) — this
  indicates a **monotonic but non-linear** relationship. This matches the scatter plot: expenses rise
  with age consistently, but not proportionally — there are step-like jumps (largely driven by the
  smoker split) rather than a straight line, which Pearson under-credits and Spearman's rank-based
  approach captures better.
- **`bmi` – `expenses` (diff = 0.087):** Pearson exceeds Spearman here, suggesting the `bmi`–`expenses`
  relationship is **closer to linear** (or at least not more monotonic than linear) in this sample.
- **`children` – `expenses` (diff = 0.082):** Spearman modestly exceeds Pearson — a weak but slightly
  more monotonic-than-linear relationship, though both values are small enough that `children` is not a
  strong predictor of expenses on its own.

**For Part 2 feature-selection guidance, we will rely on Spearman for `age`**, since it exposes the
non-linear, step-like relationship with expenses that a linear model would otherwise under-value (or
that tree-based models can naturally capture). For `bmi` and `children`, the two measures agree closely
enough that either is fine, and we default to Pearson for simplicity there.

## 8c. Grouped Aggregation

Grouped `expenses` by `smoker`:

| smoker | mean | std | count |
|---|---|---|---|
| no | $8,350.44 | $5,992.57 | 402 |
| yes | $32,988.30 | $11,027.05 | 98 |

- **Highest mean group:** `yes` (smokers), at $32,988.30.
- **Highest std group:** `yes` (smokers) also has the highest standard deviation ($11,027.05 vs.
  $5,992.57 for non-smokers).
- **Within-group variance concern:** Yes, this matters for modeling — even within the "smoker" group,
  spend varies a lot (std of ~$11k against a mean of ~$33k), so `smoker` alone is not sufficient to pin
  down an individual's expenses precisely; it needs to be combined with `age` and `bmi` to narrow the
  prediction. It's still a strong first-order signal, just not a complete one.
- **Mean ratio:** $32,988.30 / $8,350.44 = **3.95x**. A ~4x gap between the highest and lowest group
  means is large — this confirms `smoker` carries strong predictive signal and should be one of the most
  important features in Part 2's models.

## 9. Output

The cleaned dataset is saved as **`cleaned_data.csv`** (500 rows × 7 columns, 0 nulls) for use in
Parts 2 and 3.

## How to Run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Insurance_Premium_Prediction_EDA_Part1.ipynb
```

or open `Insurance_Premium_Prediction_EDA_Part1.ipynb` directly in Jupyter/Colab and run all cells
top to bottom. This notebook reads from `insurance.csv`, which is the 500-row sample used for this
phase (not the full 1,338-row original dataset).
