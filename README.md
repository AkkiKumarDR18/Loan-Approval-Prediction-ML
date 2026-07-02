# Loan Approval Prediction & Credit Risk Assessment

An end-to-end Machine Learning pipeline designed to evaluate loan applicants, calculate advanced financial risk ratios, and automate credit approvals. This project builds, benchmarks, and simulates live deployment for three distinct classification models to find the safest and most efficient system for a bank.

---

## Project Overview & Workflow

This project follows a complete data science lifecycle structured across 8 operational stages:
1. **Business Understanding:** Identifying banking risk thresholds and target metrics.
2. **Data Exploration (EDA):** Discovering trends between applicant profiles (Incomes, Loan Types) and final approval statuses.
3. **Data Cleaning:** Handling missing data, extreme outliers, and non-predictive tracking columns.
4. **Feature Engineering:** Creating smart, real-world banking ratios to expose hidden applicant financial burdens.
5. **Preprocessing:** One-Hot Encoding text data into numerical inputs and executing an 80/20 Stratified Train/Test split.
6. **Model Training:** Training **Decision Tree**, **Random Forest**, and **XGBoost** algorithms simultaneously.
7. **Model Evaluation:** Inspecting for overfitting and analyzing side-by-side visual Confusion Matrices.
8. **Deployment Simulation:** Testing how the models process a brand-new live customer profile in real-time.

---

## Smart Financial Features Created
To help our models spot risk like a real credit analyst, we engineered 5 high-impact financial ratios:
* **Debt-to-Income Ratio (Monthly):** `Monthly_Expenses / (Annual_Income / 12)`
* **Debt-to-Income Ratio (Annual):** `Outstanding_Debt / Annual_Income`
* **Expense-to-Income Ratio:** `(Monthly_Expenses * 12) / Annual_Income`
* **Loan-to-Income Ratio:** `Loan_Amount_Requested / Annual_Income`
* **Total Financial Burden:** `Outstanding_Debt + Total_Existing_Loan_Amount`

---

## Final Model Leaderboard

We evaluated our models on a hidden Test Set to choose the production champion for our lending platform:

| Model Name | Overall Accuracy <br>*(Right vs. Wrong)* | Precision <br>*(Reliability of Approvals)* | Recall <br>*(Ability to Catch Good Loans)* | F1-Score <br>*(The Balance Score)* |
| :--- | :---: | :---: | :---: | :---: |
| **Model 1: Decision Tree** <br>(Simple Baseline) | 84.57% | 84.81% | 92.52% | 88.50% |
| **Model 2: Random Forest** <br>(Team of 500 Trees) | **85.11%** | **85.10%** | **93.09%** | **88.91%** |
| **Model 3: XGBoost** <br>(Error-Fixer Champion) | **85.11%** | **85.10%** | **93.09%** | **88.91%** |

### Key Takeaways:
* **No Overfitting:** The gap between training and testing accuracy for all models stayed below 1%, proving our systems generalize perfectly to brand-new customer profiles.
* **Excellent Risk Control:** A peak **Recall of 93.09%** ensures the system safely catches and approves the vast majority of credit-worthy applicants, maximizing banking efficiency.

### Chosen Production Model: XGBoost
We chose **Model 3 (XGBoost)** as our final production engine. While it statistically tied with the Random Forest model across testing metrics, XGBoost is computationally lighter, runs faster, and uses less system memory—making it the ideal choice for live deployment behind an online banking app.

---

## How to Run Locally
1. Clone this repository or download the files.
2. Install the necessary data science libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
3. Open your terminal or Jupyter environment and run:
   ```bash
   jupyter notebook
