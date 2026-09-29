from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "features.parquet"

OUTPUT_DIR = BASE_DIR / "splitted_DS"

OUTPUT_DIR.mkdir(
    exist_ok=True
)

TRAIN_PATH = OUTPUT_DIR / "train.parquet"
VAL_PATH = OUTPUT_DIR / "val.parquet"
TEST_PATH = OUTPUT_DIR / "test.parquet"


# ============================================================
# Load feature dataset
# ============================================================

df = pd.read_parquet(
    INPUT_PATH
)


# ============================================================
# Determine time range
# ============================================================

min_step = int(
    df["step"].min()
)

max_step = int(
    df["step"].max()
)


# ============================================================
# Time-based split
# ============================================================

train_cut = (
    min_step
    + int(
        0.70 * (max_step - min_step)
    )
)

val_cut = (
    min_step
    + int(
        0.85 * (max_step - min_step)
    )
)


# ============================================================
# Create datasets
# ============================================================

train = df[
    df["step"] <= train_cut
]

val = df[
    (df["step"] > train_cut)
    & (df["step"] <= val_cut)
]

test = df[
    df["step"] > val_cut
]


# ============================================================
# Save datasets
# ============================================================

train.to_parquet(
    TRAIN_PATH,
    index=False
)

val.to_parquet(
    VAL_PATH,
    index=False
)

test.to_parquet(
    TEST_PATH,
    index=False
)


# ============================================================
# Display split information
# ============================================================

print("\nDataset split completed:")

print(
    f"Train: "
    f"[{train['step'].min()}.."
    f"{train['step'].max()}] "
    f"n={len(train)}"
)

print(
    f"Validation: "
    f"[{val['step'].min()}.."
    f"{val['step'].max()}] "
    f"n={len(val)}"
)

print(
    f"Test: "
    f"[{test['step'].min()}.."
    f"{test['step'].max()}] "
    f"n={len(test)}"
)

print(
    f"\nSaved datasets to: {OUTPUT_DIR}"
)