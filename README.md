# 📊 Fraud Detection in Mobile Money Transactions
*A Machine Learning Approach using the PaySim Dataset*

Fraud Cases are very difficult to find and practise on, as they remain highly confidential. There are some datasets on Kaggle, Paysim dataset is one of them. 

## 📌 Overview
This project compares **six machine learning classifiers** for detecting fraudulent mobile money transactions, using the **synthetic PaySim dataset**.  
The dataset simulates real transaction patterns from an African mobile money service to enable safe experimentation without revealing confidential data.

Fraud detection is challenging due to:
- **Extremely imbalanced data** (only 0.129% fraudulent transactions)
- The need to balance **accuracy** with **interpretability** in finance
- The high cost of **false negatives** (missed fraud cases)

---

## 🗂 Dataset
- **Source:** [PaySim on Kaggle](https://www.kaggle.com/datasets/ntnu-testimon/paysim1)  
- **Size:** 6.36 million rows, 11 columns  
- **Target Variable:** `isFraud` (1 = fraud, 0 = genuine)  
- **Preprocessing:**
  - Removed non-informative columns: `step`, `nameOrig`, `nameDest`, `isFlaggedFraud`
  - Label-encoded categorical `type`
  - Standardized numerical features
  - **Balanced dataset** via *Random Undersampling* (8,213 fraud + 8,213 non-fraud)

---

## 🧠 Models Compared
1. Logistic Regression (LR)
2. Naive Bayes (NB)
3. Decision Tree (DT)
4. Random Forest (RF)
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)

---

## ⚙️ Methodology

![Pipeline](images/ml_pipeline.png)

1. **Feature Selection & Encoding**
2. **Data Standardization**
3. **Class Balancing** (Random Undersampling)
4. **Train-Test Split** (70% / 30%)
5. **Model Training**
6. **Hyperparameter Tuning** (Grid Search for LR, SVM, KNN)
7. **Evaluation** using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

---

## 📈 Results

| Model              | Accuracy | Precision | Recall  | F1-score | AUC   |
|--------------------|----------|-----------|---------|----------|-------|
| Logistic Regression| 0.910    | 0.961     | 0.853   | 0.904    | —     |
| Naive Bayes        | 0.675    | 0.892     | 0.393   | 0.546    | —     |
| Decision Tree      | 0.992    | 0.989     | 0.995   | 0.992    | 0.992 |
| Random Forest      | 0.992    | 0.988     | 0.996   | 0.992    | 0.999 |
| KNN                | 0.956    | 0.946     | 0.965   | 0.956    | —     |
| SVM                | 0.914    | 0.959     | 0.863   | 0.909    | —     |

---

## 🔍 Key Findings
- **Random Forest** achieved the **highest recall** (99.6%) and AUC (0.999) — best at catching fraud.
- **Decision Tree** is almost as accurate (recall: 99.5%) but far more **interpretable**.
- **Naive Bayes** underperformed for this dataset, despite good results in credit card fraud literature.
- For **finance**, interpretability can outweigh minor performance gains. As a result, we use Decision Tree for explainability data science

---

## 📊 Visuals
- Feature Importance (Random Forest)
- Confusion Matrices for each classifier
- ROC Curves comparison

## ROC Curves Comparison
<img width="343" height="248" alt="image" src="https://github.com/user-attachments/assets/65173a61-3acd-49a2-b413-ae1b85778aa1" />


