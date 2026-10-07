"""
Disease Prediction - Prediction / Testing Script

Works with the model produced by disease_prediction.py

Examples:
    python predict_disease.py --dataset breast_cancer
    python predict_disease.py --dataset diabetes
    python predict_disease.py --dataset heart

Or specify the model directly:
    python predict_disease.py --model outputs/breast_cancer_best_model.joblib
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
def load_model(model_path):
    print(f"\nLoading model: {model_path}")

    artifact = joblib.load(model_path)

    model = artifact["model"]
    features = artifact["features"]
    classes = artifact["classes"]

    print("Model loaded successfully.")

    return model, features, classes


# ---------------------------------------------------------
# Get values from user
# ---------------------------------------------------------
def get_patient_data(features):
    print("\n" + "=" * 60)
    print("ENTER PATIENT DATA")
    print("=" * 60)

    data = {}

    for feature in features:

        while True:
            value = input(f"{feature}: ").strip()

            try:
                data[feature] = float(value)
                break
            except ValueError:
                print("Please enter a numeric value.")

    return pd.DataFrame([data], columns=features)


# ---------------------------------------------------------
# Make prediction
# ---------------------------------------------------------
def predict(model, patient_data, classes):

    # Prediction
    prediction = model.predict(patient_data)[0]

    # Probability, if available
    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient_data)[0]
        probability = float(np.max(probabilities))

    return prediction, probability


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------
def display_result(prediction, probability, classes):

    print("\n")
    print("=" * 60)
    print("                 PREDICTION RESULT")
    print("=" * 60)

    print(f"Predicted class : {prediction}")

    if probability is not None:
        print(f"Confidence      : {probability * 100:.2f}%")

    print("-" * 60)

    # Your training code tries to make 1 = disease.
    if str(prediction).lower() in [
        "1",
        "malignant",
        "present",
        "tested_positive",
        "yes",
        "true",
        "m"
    ]:
        print("Result          : DISEASE DETECTED")
    else:
        print("Result          : NO DISEASE DETECTED")

    print("=" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Predict disease using a trained model."
    )

    parser.add_argument(
        "--dataset",
        choices=["breast_cancer", "diabetes", "heart"],
        help="Dataset/model to use"
    )

    parser.add_argument(
        "--model",
        help="Path to a .joblib model"
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Determine model path
    # ---------------------------------------------
    if args.model:
        model_path = Path(args.model)

    elif args.dataset:
        model_path = Path(
            f"outputs/{args.dataset}_best_model.joblib"
        )

    else:
        print("Please specify --dataset or --model")
        return

    # Check model exists
    if not model_path.exists():
        print(f"\nERROR: Model not found:")
        print(model_path)
        return

    # ---------------------------------------------
    # Load model
    # ---------------------------------------------
    model, features, classes = load_model(model_path)

    # Show expected features
    print("\nModel expects these features:")
    for i, feature in enumerate(features, start=1):
        print(f"{i}. {feature}")

    # ---------------------------------------------
    # Get patient information
    # ---------------------------------------------
    patient_data = get_patient_data(features)

    # ---------------------------------------------
    # Prediction
    # ---------------------------------------------
    prediction, probability = predict(
        model,
        patient_data,
        classes
    )

    # ---------------------------------------------
    # Display result
    # ---------------------------------------------
    display_result(
        prediction,
        probability,
        classes
    )


if __name__ == "__main__":
    main()
