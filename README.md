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

## Current Status

- [x] Dataset inspection
- [x] Missing-value analysis
- [x] Duplicate detection and removal
- [x] Numerical feature distribution analysis
- [x] Class imbalance analysis
- [x] Correlation analysis
- [x] Selected feature comparisons by class
- [ ] Data preprocessing and feature transformation
- [ ] Baseline model development
- [ ] Model comparison and evaluation
- [ ] Final model selection

## Future Improvements

- Establish baseline classification models.
- Compare Logistic Regression, Random Forest, and other suitable classifiers.
- Evaluate class-weighting and training-only resampling techniques.
- Tune classification thresholds based on precision-recall trade-offs.
- Compare models using appropriate metrics for imbalanced classification.
- Develop a reusable prediction pipeline.

## Disclaimer

This project is intended for educational and portfolio purposes. Its models and results should not be considered a production-ready financial fraud detection system without further validation, testing, and monitoring.

## Author

**Anwesh A Salian**

B.E. — Artificial Intelligence and Machine Learning
