# Customer Churn Prediction Using Machine Learning

Predicting customer attrition in banking using demographic, transactional, and account activity data — enabling proactive retention strategies before customers leave.

---

## Business Context

Customer attrition leads to direct revenue loss and high replacement costs in banking. Retaining an existing customer is significantly cheaper than acquiring a new one, making early churn detection a high-value business problem.

**Objective:** Predict at-risk customers early using demographic, transactional, and account activity data, so the bank can intervene with targeted retention strategies before the customer leaves.

---

## Dataset

| Detail | Value |
|---|---|
| Records | 10,127 customers |
| Features | 21 |
| Target Variable | `Attrition_Flag` |
| Domain | Retail Banking |

Features span customer demographics (age, gender, income category, education), account details (card category, months on book), and transactional behavior (total transaction amount/count, inactivity period, credit utilization).

---

## Project Workflow

1. **Data Collection** — 10,127 records across banking operations.
2. **Exploratory Data Analysis (EDA)** — Analyzed feature distributions, category breakdowns (e.g. Blue Card dominance, income categories), and correlation with churn.
3. **Data Preprocessing** — Label encoding, ordinal/dummy encoding for categorical variables, `StandardScaler` feature scaling, and train-test splitting.
4. **Model Building** — Trained and compared three classifiers: Decision Tree, Random Forest, and Gradient Boosting.
5. **Evaluation & Tuning** — Used `GridSearchCV` to optimize hyperparameters, evaluating models on Accuracy, Precision, Recall, and F1-Score, with emphasis on Recall since missing an actual churner is the costliest error.
6. **Deployment & Action** — Outputs support proactive intervention workflows for customer retention.

---

## Models & Results

Test set performance across the three trained classifiers:

| Model | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|
| Decision Tree | 1.00 | 0.93 | 0.78 | 0.81 | 0.80 |
| Random Forest | 1.00 | 0.96 | 0.92 | 0.80 | 0.85 |
| **Gradient Boosting** | 0.98 | **0.96** | **0.93** | **0.81** | **0.86** |

### Final Model: Gradient Boosting

Decision Tree and Random Forest achieved perfect training scores but showed clear signs of **overfitting**, with performance dropping notably on unseen data. **Gradient Boosting** provided the best balance between training and test performance, generalizing better than the other two models. It was selected as the final model based on its superior precision, recall, and F1-score on the test set.

---

## Key Churn Signals Identified (EDA)

- Low transaction spend and transaction count
- Extended periods of account inactivity
- Certain income categories and card types showing elevated attrition risk

---

## Business Impact

- **Early identification** of at-risk customer profiles
- **Tailored retention incentives** and fee adjustments for flagged customers
- **Maximized Customer Lifetime Value (CLV)** through proactive, targeted engagement

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Techniques:** Label/Ordinal/Dummy Encoding, StandardScaler, GridSearchCV, Decision Tree, Random Forest, Gradient Boosting Classifier
