# Credit Card Fraud Detection

## Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning. It involves exploratory data analysis (EDA), data preprocessing, and classification modeling to identify potentially fraudulent transactions in a highly imbalanced dataset.

The objective is to build and evaluate models that can distinguish legitimate transactions from fraudulent ones while minimizing missed fraud cases and unnecessary false alarms.

## Problem Statement

Credit card fraud is a major challenge in financial security. Since fraudulent transactions represent only a small fraction of all transactions, a model must learn to identify rare fraud cases without relying on accuracy alone.

This project explores transaction patterns and prepares the dataset for developing and evaluating fraud detection models.

## Dataset

- **Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Original records:** 284,807 transactions
- **Features:** 30 numerical features
- **Target variable:** `Class`
- **Target values:**
  - `0` — Legitimate transaction
  - `1` — Fraudulent transaction

### Features

- `Time`: Seconds elapsed between a transaction and the first transaction in the dataset.
- `V1`–`V28`: Anonymized features obtained using Principal Component Analysis (PCA).
- `Amount`: Transaction amount.
- `Class`: Target variable indicating whether a transaction is fraudulent.

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset structure and data type inspection.
- Missing-value analysis.
- Duplicate-record detection and removal.
- Numerical feature distribution analysis using histograms.
- Transaction amount analysis by target class.
- Class imbalance analysis.
- Feature correlation analysis.
- Comparison of selected feature distributions across legitimate and fraudulent transactions.

### Key Findings

- No missing values were found.
- A total of 1,081 duplicate rows were identified and removed.
- The cleaned dataset contains 283,726 transactions.
- The dataset is highly imbalanced, with fraudulent transactions representing approximately 0.17% of the cleaned data.
- The `Amount` feature is strongly right-skewed.
- The `Time` feature exhibits a non-uniform distribution.
- Several PCA features, including `V1`, `V3`, `V4`, `V10`, and `V14`, show visible distributional differences between legitimate and fraudulent transactions.
- Transaction amount alone is insufficient to reliably identify fraudulent transactions.

For detailed observations, see [`eda_insights.md`](eda_insights.md).

## Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook
- **Version Control:** Git and GitHub

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_preprocessing.ipynb
├── src/
├── eda_insights.md
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Update the notebook and source filenames above to match your actual repository structure.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Credit-Card-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the CSV file in the `data/` directory.

The raw dataset is not included in this repository.

### 5. Run the notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory and execute the cells in the appropriate order.

Ensure that the dataset path used in the notebooks matches its actual location.

## Evaluation Metrics

Because the dataset is highly imbalanced, model evaluation should consider:

- **Precision:** The proportion of predicted fraudulent transactions that are actually fraudulent.
- **Recall:** The proportion of actual fraudulent transactions correctly detected.
- **F1-score:** The harmonic mean of precision and recall.
- **Precision-Recall AUC (PR-AUC):** Measures performance across classification thresholds, particularly useful for rare positive classes.
- **Confusion Matrix:** Summarizes true positives, true negatives, false positives, and false negatives.

Accuracy will also be reported where appropriate, but it will not be used as the sole measure of model performance.

## Current Progress

### Completed
- [x] Dataset inspection and data quality checks
- [x] Missing-value analysis
- [x] Duplicate detection and removal
- [x] Numerical feature distribution analysis
- [x] Transaction amount analysis by class
- [x] Class imbalance analysis
- [x] Correlation analysis
- [x] Feature distribution comparison by class
- [x] Train-test split using stratification
- [x] StandardScaler preprocessing experiment
- [x] Log transformation of `Amount` followed by StandardScaler
- [x] Initial Logistic Regression baseline

### In Progress
- [ ] Evaluate class weighting for imbalanced classification
- [ ] Compare classification thresholds
- [ ] Experiment with additional classification models
- [ ] Compare models using precision, recall, F1-score, and PR-AUC
- [ ] Final model selection and evaluation

## Initial Modeling Results

Logistic Regression was evaluated using two preprocessing approaches:

1. StandardScaler applied to `Time` and `Amount`.
2. Log transformation of `Amount`, followed by StandardScaler applied to `Time` and the transformed `Amount`.

| Metric (Fraud Class) | StandardScaler | Log + StandardScaler |
|---|---:|---:|
| Precision | 0.85 | 0.86 |
| Recall | 0.59 | 0.60 |
| F1-score | 0.70 | 0.71 |
| Fraud cases detected | 56 / 95 | 57 / 95 |
| Fraud cases missed | 39 | 38 |

The log-transformed experiment showed a small improvement in fraud-class precision, recall, and F1-score on the evaluated test split. Further validation is needed to determine whether this improvement is consistent.

Because the dataset is highly imbalanced, accuracy alone is insufficient for evaluating fraud detection performance.

Detailed observations and experiment results are documented in:
- [EDA Insights](eda_insights.md)
- [Preprocessing Experiments](reports/preprocessing_experiments.md)
