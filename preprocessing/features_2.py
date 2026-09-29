from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "prepared_txn"
OUTPUT_PATH = BASE_DIR / "features.parquet"


# ============================================================
# Feature engineering
# ============================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Log transformed transaction amount
    # --------------------------------------------------------

    df["log_amount"] = np.log1p(
        df["amount"]
    )


    # --------------------------------------------------------
    # Time since previous transaction
    # --------------------------------------------------------

    df["prev_step_user"] = (
        df.groupby("nameOrig")["step"]
        .shift(1)
    )

    df["recency_hours"] = (
        df["step"] - df["prev_step_user"]
    ).fillna(1e6)


    # --------------------------------------------------------
    # Transaction activity
    # --------------------------------------------------------

    df["txn_count_24h"] = (
        df.groupby("nameOrig")["step"]
        .diff()
        .lt(24)
        .groupby(df["nameOrig"])
        .cumsum()
        .fillna(0)
    )


    # --------------------------------------------------------
    # Destination history
    # --------------------------------------------------------

    df["user_dest_count"] = (
        df.groupby(
            ["nameOrig", "nameDest"]
        ).cumcount()
    )

    df["is_dest_new"] = (
        df["user_dest_count"] == 0
    ).astype(int)

    df = df.drop(
        columns=["user_dest_count"]
    )


    # --------------------------------------------------------
    # Hour of day
    # --------------------------------------------------------

    df["hours_day"] = (
        df["step"] % 24
    )


    # --------------------------------------------------------
    # Transaction type one-hot encoding
    # --------------------------------------------------------

    dummies = pd.get_dummies(
        df["type"],
        prefix="type"
    )

    df = pd.concat(
        [df, dummies],
        axis=1
    )


    # --------------------------------------------------------
    # Final feature columns
    # --------------------------------------------------------

    label = "isFraud"

    new_cols = [
        "step",
        "nameOrig",
        "nameDest",
        "amount",
        "log_amount",
        "recency_hours",
        "txn_count_24h",
        "is_dest_new",
        "hours_day",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ] + list(dummies.columns) + [label]


    return df[new_cols]


# ============================================================
# Load prepared data
# ============================================================

df = pd.read_parquet(
    INPUT_PATH
)


# ============================================================
# Build features
# ============================================================

df_feat = build_features(df)


# ============================================================
# Handle missing values
# ============================================================

df_feat = df_feat.fillna(0)


# ============================================================
# Save feature dataset
# ============================================================

df_feat.to_parquet(
    OUTPUT_PATH,
    index=False
)

print(
    f"Saved -> {OUTPUT_PATH}"
)

print(
    f"Rows: {len(df_feat)}"
)