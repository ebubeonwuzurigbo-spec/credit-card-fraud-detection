# Credit Card Fraud Detection Pipeline

## 🚀 Project Overview
This project implements a production-ready machine learning pipeline to detect fraudulent credit card transactions. Utilizing a dataset of 284k European transactions, the system addresses extreme class imbalance (0.17% fraud) through strategic under-sampling and feature scaling.

## 🛠 Tech Stack
- **Language:** Python 3.11
- **Libraries:** Pandas, Scikit-Learn, Joblib, Matplotlib
- **Environment:** Isolated Virtual Environment (venv)

## 📊 Key Results
- **Model:** Logistic Regression (Baseline) vs. Random Forest
- **Precision:** 0.97
- **Recall:** 0.91
- **F1-Score:** 0.94

## 🏗 Directory Structure
- `data/`: Raw transaction data (Git-ignored for privacy)
- `src/`: Modular Python scripts for loading, preprocessing, and training
- `models/`: Serialized production models (.pkl)
- `notebooks/`: Exploratory data analysis

## 🧠 Engineering Decisions
- **Robustness:** Implemented defensive programming with `try/except` blocks and logging.
- **Scaling:** Standardized 'Amount' and 'Time' features to ensure distance-based model stability.
- **Sampling:** Utilized Random Under-sampling to establish a balanced baseline, preventing Accuracy Paradox.
