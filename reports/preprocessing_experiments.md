# Preprocessing Experiments — Credit Card Fraud Detection

## 1. Objective

The objective of these experiments is to investigate whether feature scaling and a log transformation of transaction amounts improve Logistic Regression performance on a highly imbalanced credit card fraud detection dataset.

## 2. Dataset Preparation

- Duplicate rows were identified and removed.
- No missing values were found.
- The target variable is `Class`, where 0 represents legitimate transactions and 1 represents fraudulent transactions.
- The cleaned dataset contains 283,726 transactions, including 473 fraudulent transactions.
- The dataset was split into training and testing sets using stratification, with an 80:20 split and `random_state=42`.

## 3. Preprocessing Approaches

### Experiment A: StandardScaler

**Motivation:** `Time` and `Amount` have different numerical scales. Standardization allows scale-sensitive models such as Logistic Regression to train more effectively.

**Method:**
- Applied `StandardScaler` to `Time` and `Amount`.
- Fitted the scaler on training data only.
- Applied the fitted scaler to the test data.

### Experiment B: Log Transformation + StandardScaler

**Motivation:** EDA showed that `Amount` is strongly right-skewed, with a small number of high-value transactions.

**Method:**
- Applied `np.log1p()` to `Amount`.
- Standardized `Time` and the transformed `Amount`.
- Fitted the scaler on training data only.
- Applied the fitted scaler to the test data.

The PCA-transformed features `V1`–`V28` were left unchanged in these experiments.

## 4. Model

Logistic Regression was used as the initial classification model.

The model was evaluated using:
- Precision
- Recall
- F1-score
- Confusion matrix

The fraud class (Class 1) is the primary focus because it is the minority class.

## 5. Results

### Experiment A: StandardScaler

**Classification report — Class 1**

| Metric | Score |
|---|---:|
| Precision | 0.85 |
| Recall | 0.59 |
| F1-score | 0.70 |
| Support | 95 |

**Confusion matrix**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,641 | 10 |
| Actual Fraud | 39 | 56 |

### Experiment B: Log Transformation + StandardScaler

**Classification report — Class 1**

| Metric | Score |
|---|---:|
| Precision | 0.86 |
| Recall | 0.60 |
| F1-score | 0.71 |
| Support | 95 |

**Confusion matrix**

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,642 | 9 |
| Actual Fraud | 38 | 57 |

## 6. Interpretation

The log-transformed experiment produced the following changes on the evaluated split:

- Precision increased from 0.85 to 0.86.
- Recall increased from 0.59 to 0.60.
- F1-score increased from 0.70 to 0.71.
- One additional fraudulent transaction was detected.
- One fewer legitimate transaction was incorrectly flagged as fraud.

The improvement is small. These results suggest that log transformation is worth investigating further, but they do not establish that it will consistently outperform standard scaling alone.

## 7. Limitations

- The test set contains only 95 fraudulent transactions, so each correctly or incorrectly classified fraud case noticeably affects the metrics.
- The same test set was used to compare preprocessing approaches. It should not be treated as an untouched final evaluation set for subsequent model selection.
- The classification threshold was the default threshold used by Logistic Regression.
- No class weighting or resampling strategy has yet been evaluated in these experiments.
- The results reflect one train-test split and do not establish generalization across different splits or future transaction periods.

## 8. Next Steps

1. Establish a validation strategy for comparing model and preprocessing choices.
2. Evaluate Logistic Regression with `class_weight="balanced"`.
3. Investigate precision-recall trade-offs using predicted probabilities and validation data.
4. Compare additional classifiers, including Random Forest.
5. Evaluate average precision (PR-AUC), precision, recall, F1-score, and confusion matrices.
6. Select the final approach using validation results.
7. Perform final evaluation on a previously untouched test set.

## 9. Conclusion

Both preprocessing approaches provide a useful Logistic Regression baseline. The log-transformed approach showed a small improvement on the evaluated split, but further validation is needed.

The next priority is to investigate class imbalance and improve fraud recall while monitoring false positives.
