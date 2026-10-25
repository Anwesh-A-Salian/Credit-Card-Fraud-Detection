# EDA Insights — Credit Card Fraud Detection

## 1. Dataset Overview

- **Original dataset size:** 284,807 rows and 31 columns.
- **Features:** 30 numerical columns and one target column, `Class`.
- **Target definition:**
  - `Class = 0`: Legitimate transaction.
  - `Class = 1`: Fraudulent transaction.
- Most feature columns are floating-point values; `Class` is an integer.
- The dataset contains anonymized PCA-transformed features (`V1`–`V28`), along with `Time` and `Amount`.

## 2. Data Quality

- No missing values were found in any column.
- **Duplicate rows found:** 1,081.
- Duplicate rows were removed.
- **Dataset size after deduplication:** 283,726 rows.
- **Class counts after deduplication:**
  - Legitimate transactions: 283,253.
  - Fraudulent transactions: 473.

> Note: Duplicate removal was applied to exact duplicate rows. In fraud-detection data, duplicates should be reviewed carefully because repeated transactions may be genuine; document the rationale for removing them.

## 3. Numerical Feature Distributions

### `Time`

- The histogram shows a non-uniform distribution.
- This may indicate temporal patterns in transaction activity.
- Further interpretation should account for how `Time` is encoded in the dataset.

### PCA Features (`V1`–`V28`)

- Most PCA-transformed features are concentrated around central values.
- Some show skewness, sharp peaks, or extreme values.
- Since these are anonymized PCA components, their individual real-world meanings are not directly interpretable from the dataset alone.

### `Amount`

- `Amount` is strongly right-skewed: most transactions have relatively small amounts, while a smaller number have much larger values.
- A log transformation could be tested during modeling if appropriate. Any transformation should be fitted or selected using training data only.

## 4. Target Class Distribution

After removing duplicates, the target distribution is:

| Class | Meaning | Transactions | Percentage |
|---|---|---:|---:|
| 0 | Legitimate | 283,253 | 99.8333% |
| 1 | Fraudulent | 473 | 0.1667% |
| **Total** | | **283,726** | **100%** |

- The dataset is **severely imbalanced**: fraudulent transactions represent approximately 0.17% of the data.
- A model that predicts every transaction as legitimate would achieve about 99.83% accuracy while detecting no fraud.
- Therefore, accuracy alone is not a suitable evaluation metric.
- Recommended metrics include **precision, recall, F1-score, PR-AUC / average precision, and the confusion matrix**. The choice of threshold should reflect the cost of false positives and false negatives.

## 5. Transaction Amount Analysis by Class

The observed class-wise statistics are:

| Statistic | Legitimate (`Class = 0`) | Fraudulent (`Class = 1`) |
|---|---:|---:|
| Mean amount | 88.41 | 123.87 |
| Median amount | 22.00 | 9.82 |
| Maximum amount | 25,691.16 | 2,125.87 |

- The mean amount for fraudulent transactions is higher than for legitimate transactions, while the median is lower.
- This indicates that the fraud amount distribution is influenced by its upper tail; the mean alone does not describe a typical fraudulent transaction.
- The maximum legitimate transaction amount is much larger than the maximum fraudulent transaction amount.
- Consequently, transaction amount alone is not a reliable rule for identifying fraud.
- The IQR method can flag unusually large values for investigation, but flagged observations should not be removed automatically. They may be valid and useful for fraud detection.

## 6. Correlation Analysis

- The correlation heatmap shows generally low pairwise linear correlations among most PCA features (`V1`–`V28`).
- Some relationships are visible between `Amount` and individual features.
- Some features show visible linear associations with `Class`.
- The diagonal of the correlation matrix equals 1 because each feature is perfectly correlated with itself.
- Low pairwise correlation does not rule out nonlinear relationships, interactions, or predictive value. Correlation with `Class` alone is not a complete measure of a feature's usefulness.

## 7. Feature Distributions by Class

Box plots were inspected for `V1`, `V3`, `V4`, `V10`, and `V14`.

- **`V1` and `V3`:** Fraudulent transactions show lower medians than legitimate transactions.
- **`V4`:** Fraudulent transactions show a higher median.
- **`V10` and `V14`:** Fraudulent transactions show substantially lower medians.
- These differences suggest that the selected features may help distinguish the classes.
- The distributions still show variation and potential overlap. Box plots alone cannot establish predictive performance; the features should be evaluated within a properly validated model.
- Because `V1`–`V28` are anonymized PCA components, their real-world interpretation is limited.

## 8. Key EDA Findings

1. The dataset has no observed missing values.
2. There were 1,081 exact duplicate rows, which were removed.
3. The target is extremely imbalanced, with only 473 fraudulent transactions after deduplication.
4. `Amount` is strongly right-skewed, and `Time` has a non-uniform distribution.
5. Several selected PCA components have visibly different distributions across legitimate and fraudulent transactions.
6. Transaction amount alone cannot reliably identify fraud.
7. Model evaluation must emphasize fraud detection quality rather than accuracy alone.

## 9. Planned Preprocessing and Modeling Considerations

- Split the data into training and test sets using stratification on `Class`, so both sets retain approximately similar class proportions.
- Fit all learned preprocessing steps on the training set only, then apply the fitted transformations to the test set.
- Consider scaling `Time` and `Amount` for models sensitive to feature scale. PCA components are already transformed, so assess whether further scaling is needed for the chosen model.
- Optionally evaluate a log transformation for `Amount`, comparing results rather than assuming it will improve performance.
- Establish a baseline model before trying class weights or resampling methods.
- If using oversampling methods such as SMOTE, apply them **only to the training data**, never before splitting the full dataset.
- Report precision, recall, F1-score, PR-AUC / average precision, and the confusion matrix. Choose the decision threshold based on the practical trade-off between missed fraud and false alarms.

## 10. Conclusion

The dataset is clean with respect to missing values, but its extreme class imbalance is the main modeling challenge. The distributions of `Time`, `Amount`, and selected PCA features provide useful directions for further investigation. The next step is to build a leakage-safe modeling pipeline, establish a baseline, and evaluate how effectively it detects fraudulent transactions using appropriate imbalanced-class metrics.
