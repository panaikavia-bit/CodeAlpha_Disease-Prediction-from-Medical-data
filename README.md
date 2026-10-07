# CodeAlpha_DiseasePrediction

Machine learning project that predicts the possibility of disease from structured medical data.
Built for the **CodeAlpha Machine Learning Internship – Task 4: Disease Prediction from Medical Data**.

## Objective
Predict whether a patient has a disease using classification techniques on patient features
such as age, blood pressure, glucose, cholesterol and clinical measurements.

## Datasets (UCI ML Repository)
| Disease | Source | Target |
|---|---|---|
| Breast Cancer | Wisconsin Diagnostic (bundled with scikit-learn) | Malignant / Benign |
| Diabetes | Pima Indians Diabetes (OpenML) | Tested positive / negative |
| Heart Disease | Statlog Heart (OpenML `heart-statlog`) | Present / Absent |

## Approach
1. **EDA** – class balance and feature correlation heatmap.
2. **Preprocessing** – median imputation (impossible zeros in Pima data treated as missing), standard scaling and one-hot encoding, all inside a scikit-learn `Pipeline` to prevent data leakage.
3. **Models** – Logistic Regression, SVM, Random Forest, XGBoost (class-imbalance handled with class weights).
4. **Tuning** – `RandomizedSearchCV` with stratified 5-fold cross-validation (ROC-AUC objective; F1 for SVM).
5. **Evaluation** – Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curves, permutation feature importance.
6. **Deployment-style inference** – `predict.py` loads the best saved model and predicts for new patients.

## Project structure
```
CodeAlpha_DiseasePrediction/
├── disease_prediction.py   # training, tuning, evaluation
├── predict.py              # disease detection on new patient data
├── requirements.txt
├── README.md
└── outputs/                # plots, results CSVs, saved models
```

## Installation
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

## Usage
**Train and evaluate**
```
python disease_prediction.py --dataset breast_cancer
python disease_prediction.py --dataset diabetes
python disease_prediction.py --dataset heart
```
**Predict for new patients**
```
python predict.py --dataset heart                                   # interactive
python predict.py --dataset diabetes --json "{\"plas\":148,\"mass\":33.6,\"age\":50}"
python predict.py --dataset heart --template                        # blank CSV template
python predict.py --dataset heart --csv heart_template.csv --out predictions.csv
```
Output: predicted class, risk score and risk level (LOW / MODERATE / HIGH).

## Results

### Heart Disease
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8333 | 0.7778 | 0.8750 | 0.8235 | 0.9083 |
| SVM | 0.8333 | 0.8000 | 0.8333 | 0.8163 | 0.9042 |
| Random Forest | 0.8333 | 0.8000 | 0.8333 | 0.8163 | 0.8819 |
| XGBoost | 0.7963 | 0.7407 | 0.8333 | 0.7843 | 0.8708 |

**Best model:** Logistic Regression (highest Recall and ROC-AUC).

### Diabetes
<!-- Paste the table from outputs/diabetes_results.csv -->

### Breast Cancer
<!-- Paste the table from outputs/breast_cancer_results.csv -->

### Plots
Saved in `outputs/` for every dataset: `*_eda.png`, `*_roc.png`, `*_best_model.png` (confusion matrix + feature importance).

## Key observations
- Recall is prioritised for medical data, because missing a diseased patient is costlier than a false alarm.
- Test sets are small (e.g. 54 patients for heart), so differences between models are small and should be read with caution.
- Glucose (`plas`), BMI (`mass`) and age are the strongest predictors for diabetes.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib.

## Disclaimer
Educational project only. It is not a medical diagnosis tool; consult a qualified doctor for health decisions.

## Author
Kavia – CodeAlpha ML Intern
