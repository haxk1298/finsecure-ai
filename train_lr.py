from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "splitted_DS"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# Load datasets
# ============================================================

train_path = DATA_DIR / "train.parquet"
val_path = DATA_DIR / "val.parquet"
test_path = DATA_DIR / "test.parquet"

train = pd.read_parquet(train_path)
val = pd.read_parquet(val_path)
test = pd.read_parquet(test_path)


# ============================================================
# Features
# ============================================================

features = [
    "amount",
    "log_amount",
    "recency_hours",
    "txn_count_24h",
    "is_dest_new",
    "hours_day",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]


# ============================================================
# Training data
# ============================================================

X_train = train[features]
y_train = train["isFraud"]

X_val = val[features]
y_val = val["isFraud"]

X_test = test[features]
y_test = test["isFraud"]


# ============================================================
# Handle class imbalance
# ============================================================

scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)


# ============================================================
# XGBoost model
# ============================================================

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)


# ============================================================
# Train model
# ============================================================

print("Training XGBoost model...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# Evaluation
# ============================================================

def evaluate(X, y, name):

    y_prob = model.predict_proba(X)[:, 1]

    y_pred = (
        y_prob >= 0.8
    ).astype(int)

    print(f"\n{name}:")
    print(
        f"  AUC: "
        f"{roc_auc_score(y, y_prob):.4f}"
    )

    print(
        f"  Precision: "
        f"{precision_score(y, y_pred, zero_division=0):.4f}"
    )

    print(
        f"  Recall: "
        f"{recall_score(y, y_pred, zero_division=0):.4f}"
    )


evaluate(
    X_train,
    y_train,
    "Train"
)

evaluate(
    X_val,
    y_val,
    "Validation"
)

evaluate(
    X_test,
    y_test,
    "Test"
)


# ============================================================
# Save model
# ============================================================

MODEL_PATH = MODEL_DIR / "xboost_model.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved successfully!")
print(f"Location: {MODEL_PATH}")


# ============================================================
# Test saved model
# ============================================================

print("\nTesting saved model...")

loaded_model = joblib.load(
    MODEL_PATH
)

test_prob = loaded_model.predict_proba(
    X_test.head(1)
)

print(
    f"Sample prediction: {test_prob}"
)

print(
    f"Actual label: {y_test.iloc[0]}"
)