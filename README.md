# CodeAlpha_DiseasePrediction

Disease prediction from structured medical data (CodeAlpha ML Internship, Task 4).

## Datasets
- Breast Cancer Wisconsin (UCI, bundled with scikit-learn)
- Pima Indians Diabetes (UCI via OpenML)
- Heart Disease (UCI via OpenML `heart-statlog`)

## Approach
1. EDA: class balance, correlation heatmap
2. Preprocessing: median imputation (Pima zeros treated as missing), scaling, one-hot for categoricals, all inside a Pipeline (no data leakage)
3. Models: Logistic Regression, SVM, Random Forest, XGBoost
4. Tuning: RandomizedSearchCV, stratified 5-fold, ROC-AUC objective
5. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curves, permutation importance

## Run
```
pip install -r requirements.txt
python disease_prediction.py --dataset breast_cancer
python disease_prediction.py --dataset diabetes
python disease_prediction.py --dataset heart
```
Outputs (plots, results CSV, saved model) go to `outputs/`.

## Results
(Paste your results table here after running.)
