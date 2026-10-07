"""
CodeAlpha Task 4: Disease Prediction from Medical Data
Datasets : Breast Cancer (sklearn/UCI WDBC), Pima Diabetes (OpenML), Heart Disease (OpenML heart-statlog)
Models   : Logistic Regression, SVM, Random Forest, XGBoost
Usage    : python disease_prediction.py --dataset breast_cancer
           python disease_prediction.py --dataset diabetes
           python disease_prediction.py --dataset heart
           python disease_prediction.py --csv my.csv --target label   (custom dataset)
"""
import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------
def load_data(name: str, csv: str = None, target: str = None):
    if csv:
        df = pd.read_csv(csv)
        return df.drop(columns=[target]), df[target]

    if name == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
        # sklearn encodes malignant=0, benign=1 -> flip so 1 = disease (malignant)
        return d.data, (1 - d.target).rename("malignant")

    if name == "diabetes":
        d = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        X, y = d.data.copy(), d.target
        # Physiologically impossible zeros are missing values in the Pima dataset
        for c in ["plas", "pres", "skin", "insu", "mass"]:
            X[c] = X[c].replace(0, np.nan)
        return X, y

    if name == "heart":
        d = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser="auto")
        return d.data, d.target

    raise ValueError(f"Unknown dataset: {name}")


def encode_target(y: pd.Series):
    """Encode string labels, ensuring the 'disease present' class is 1."""
    if y.dtype.kind in "iu" and set(y.unique()) <= {0, 1}:
        return y.values, [0, 1]
    positives = {"tested_positive", "present", "malignant", "yes", "true", "1", "m"}
    le = LabelEncoder().fit(y.astype(str))
    classes = list(le.classes_)
    yy = le.transform(y.astype(str))
    pos = [i for i, c in enumerate(classes) if c.lower() in positives]
    if pos and pos[0] == 0:  # flip so positive class == 1
        yy = 1 - yy
        classes = classes[::-1]
    return yy, classes


# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
def run_eda(X: pd.DataFrame, y: np.ndarray, tag: str):
    print(f"\nShape: {X.shape} | Missing values: {int(X.isna().sum().sum())}")
    print("Class balance:", dict(zip(*np.unique(y, return_counts=True))))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(x=y, ax=ax[0])
    ax[0].set_title("Class distribution (1 = disease)")
    num = X.select_dtypes("number")
    corr = num.assign(target=y).corr()["target"].drop("target").abs().sort_values(ascending=False)
    top = corr.head(12).index
    sns.heatmap(num[top].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1], cbar=False,
                annot_kws={"size": 7})
    ax[1].set_title("Correlation of top-12 target-related features")
    plt.tight_layout()
    plt.savefig(OUT / f"{tag}_eda.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# 3. Models + hyperparameter search spaces
# ----------------------------------------------------------------------------
def build_models(pos_weight: float):
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
            {"clf__C": np.logspace(-3, 2, 20), "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]},
        ),
        "SVM": (
            SVC(class_weight="balanced", random_state=SEED),
            {"clf__C": np.logspace(-1, 2, 20), "clf__gamma": ["scale", "auto", 0.01, 0.1],
             "clf__kernel": ["rbf", "linear"]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [None, 4, 6, 8, 12],
             "clf__min_samples_split": [2, 5, 10], "clf__min_samples_leaf": [1, 2, 4],
             "clf__max_features": ["sqrt", "log2", 0.5]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1,
                          scale_pos_weight=pos_weight, tree_method="hist"),
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [2, 3, 4, 6],
             "clf__learning_rate": [0.01, 0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__reg_lambda": [1, 5, 10]},
        )
    else:
        print("[!] xgboost not installed -> skipping XGBoost (pip install xgboost)")
    return models


def make_pipeline(clf, X: pd.DataFrame):
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", __import__("sklearn.preprocessing", fromlist=["OneHotEncoder"])
                           .OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


# ----------------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------------
def evaluate(model, X_te, y_te):
    # SVC uses decision_function: Platt-scaled predict_proba can invert on small datasets
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
    else:
        proba = model.decision_function(X_te)
        pred = (proba > 0).astype(int)
    return {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }, pred, proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="breast_cancer", choices=["breast_cancer", "diabetes", "heart"])
    ap.add_argument("--csv")
    ap.add_argument("--target")
    ap.add_argument("--iters", type=int, default=25, help="RandomizedSearch iterations per model")
    args = ap.parse_args()
    tag = Path(args.csv).stem if args.csv else args.dataset

    X, y_raw = load_data(args.dataset, args.csv, args.target)
    y, classes = encode_target(pd.Series(y_raw))
    run_eda(X, y, tag)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted, probas = {}, {}, {}
    for name, (clf, space) in build_models(pos_weight).items():
        print(f"\n=== Tuning {name} ===")
        # SVM: tune on F1 (ROC-AUC alone can pick a tiny C whose decision threshold predicts one class only)
        scoring = "f1" if name == "SVM" else "roc_auc"
        search = RandomizedSearchCV(make_pipeline(clf, X), space, n_iter=args.iters, scoring=scoring,
                                    cv=cv, random_state=SEED, n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        metrics, pred, proba = evaluate(search.best_estimator_, X_te, y_te)
        metrics["CV score"] = search.best_score_  # SVM: CV F1, others: CV ROC-AUC
        results[name], fitted[name], probas[name] = metrics, search.best_estimator_, proba
        print("Best params:", {k.replace("clf__", ""): v for k, v in search.best_params_.items()})
        print(classification_report(y_te, pred, target_names=[str(c) for c in classes]))

    res = pd.DataFrame(results).T.round(4).sort_values("ROC-AUC", ascending=False)
    print("\n================ MODEL COMPARISON ================\n", res)
    res.to_csv(OUT / f"{tag}_results.csv")

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for n, m in fitted.items():
        RocCurveDisplay.from_estimator(m, X_te, y_te, name=n, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_title(f"ROC curves - {tag}")
    plt.tight_layout()
    plt.savefig(OUT / f"{tag}_roc.png", dpi=150)
    plt.close()

    # Best model: confusion matrix + permutation importance + persistence
    best_name = res.index[0]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, ax=ax[0], cmap="Blues",
                                          display_labels=[str(c) for c in classes])
    ax[0].set_title(f"Confusion matrix - {best_name}")
    pi = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=20,
                                random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values().tail(12)
    imp.plot.barh(ax=ax[1])
    ax[1].set_title("Permutation importance (top 12)")
    plt.tight_layout()
    plt.savefig(OUT / f"{tag}_best_model.png", dpi=150)
    plt.close()

    joblib.dump({"model": best, "features": list(X.columns), "classes": classes},
                OUT / f"{tag}_best_model.joblib")
    print(f"Artifacts saved in ./{OUT}/")


if __name__ == "__main__":
    main()