from pathlib import Path
import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "fraud_detection.csv"
OUTPUT_PATH = BASE_DIR / "prepared_txn"


# ============================================================
# Columns required for fraud detection
# ============================================================

USE_COLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    usecols=USE_COLS
)


# ============================================================
# Remove duplicates and sort by time
# ============================================================

df = (
    df
    .drop_duplicates()
    .sort_values("step")
    .reset_index(drop=True)
)


# ============================================================
# Clean balance columns
# ============================================================

balance_columns = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

for column in balance_columns:
    if column in df.columns:
        df[column] = df[column].clip(lower=0)


# ============================================================
# Display basic information
# ============================================================

print(df.head())

print(
    f"\nFraud rate: {df['isFraud'].mean():.4f}"
)


# ============================================================
# Save prepared dataset
# ============================================================

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved prepared dataset -> {OUTPUT_PATH}"
)

print(
    f"Rows: {len(df)}"
)