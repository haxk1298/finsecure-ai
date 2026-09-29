from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = BASE_DIR / "features.parquet"


df = pd.read_parquet(
    FEATURES_PATH
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())