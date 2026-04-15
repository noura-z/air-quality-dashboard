"""
preprocessing.py — Air Quality Data Pipeline
=============================================
Handles cleaning, feature engineering, AQI labeling,
and train/test splitting for the Air Quality Dashboard.

Dataset: OpenAQ + AirGradient (Ben Guerir / Marrakech area)
Columns: pm1, pm25, relativehumidity, temperature, um003, lat/lon, timestamps
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# WHO PM2.5 Thresholds (24h mean, µg/m³)
# ─────────────────────────────────────────────
WHO_THRESHOLDS = [
    (0,    5,   "Good",          0),
    (5,    15,  "Moderate",      1),
    (15,   25,  "Unhealthy (S)", 2),
    (25,   50,  "Unhealthy",     3),
    (50,   75,  "Very Unhealthy",4),
    (75,   float("inf"), "Hazardous", 5),
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and parse datetime columns."""
    df = pd.read_csv(filepath)
    df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"], utc=True, format="mixed")
    df["datetimeLocal"] = pd.to_datetime(df["datetimeLocal"], utc=True, format="mixed")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates, drop always-null columns,
    fix sensor outliers, and sort by time.
    """
    # Drop fully empty columns (country_iso, isMobile, isMonitor)
    df = df.drop(columns=["country_iso", "isMobile", "isMonitor"], errors="ignore")

    # Remove duplicate rows (same location + timestamp)
    df = df.drop_duplicates(subset=["location_id", "datetimeUtc"])

    # Sort chronologically
    df = df.sort_values("datetimeUtc").reset_index(drop=True)

    # Clamp obviously impossible sensor readings
    df["pm25"] = df["pm25"].clip(lower=0, upper=500)
    df["pm1"]  = df["pm1"].clip(lower=0, upper=500)
    df["relativehumidity"] = df["relativehumidity"].clip(0, 100)
    df["temperature"] = df["temperature"].clip(-10, 60)
    df["um003"] = df["um003"].clip(lower=0)

    # Drop rows where the target (pm25) is NaN
    df = df.dropna(subset=["pm25"])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical and categorical time features from UTC datetime."""
    dt = df["datetimeUtc"].dt

    df["hour"]       = dt.hour
    df["dayofweek"]  = dt.dayofweek          # 0=Mon, 6=Sun
    df["month"]      = dt.month
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # Cyclical encoding (keeps hour 23 close to hour 0)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-location rolling averages and lag features for PM2.5.
    Windows: 1h, 6h, 24h. Lags: 1h, 3h, 6h, 24h.
    Assumes data is already sorted by datetimeUtc.
    """
    df = df.copy()

    for loc_id, grp in df.groupby("location_id"):
        idx = grp.index

        # Rolling windows (min_periods allows partial windows at edges)
        for window, label in [(1, "1h"), (6, "6h"), (24, "24h")]:
            df.loc[idx, f"pm25_roll_{label}"] = (
                grp["pm25"].rolling(window, min_periods=1).mean().values
            )
            df.loc[idx, f"pm1_roll_{label}"] = (
                grp["pm1"].rolling(window, min_periods=1).mean().values
            )
            df.loc[idx, f"rh_roll_{label}"] = (
                grp["relativehumidity"].rolling(window, min_periods=1).mean().values
            )

        # Lag features
        for lag, label in [(1, "1h"), (3, "3h"), (6, "6h"), (24, "24h")]:
            df.loc[idx, f"pm25_lag_{label}"] = grp["pm25"].shift(lag).values

    return df


def add_aqi_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add WHO-based AQI category columns:
      - aqi_label : human-readable string
      - aqi_level : integer 0–5
    """
    def _classify(pm25_val):
        for lo, hi, label, level in WHO_THRESHOLDS:
            if lo <= pm25_val < hi:
                return label, level
        return "Hazardous", 5

    labels, levels = zip(*df["pm25"].apply(_classify))
    df["aqi_label"] = labels
    df["aqi_level"] = levels
    return df


def get_feature_columns() -> list:
    """Return the ordered list of feature columns used by ML models."""
    return [
        # Raw sensors
        "pm1", "relativehumidity", "temperature", "um003",
        # Rolling averages
        "pm25_roll_1h", "pm25_roll_6h", "pm25_roll_24h",
        "pm1_roll_1h",  "pm1_roll_6h",  "pm1_roll_24h",
        "rh_roll_1h",   "rh_roll_6h",   "rh_roll_24h",
        # Lags
        "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_24h",
        # Cyclical time
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Categorical time
        "hour", "dayofweek", "month", "is_weekend",
        # Spatial
        "latitude", "longitude",
    ]


def prepare_features(df: pd.DataFrame):
    """
    Full pipeline: clean → time features → rolling features → AQI labels.
    Returns (X, y, df_clean).
    """
    df = clean_data(df)
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_aqi_label(df)

    feature_cols = get_feature_columns()

    # Drop rows with NaN in features (from lag/rolling at start of series)
    df_model = df.dropna(subset=feature_cols + ["pm25"]).copy()

    X = df_model[feature_cols]
    y = df_model["pm25"]

    return X, y, df_model


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Chronological train/test split (shuffle=False preserves time order)."""
    return train_test_split(X, y, test_size=test_size, shuffle=False)


# ─────────────────────────────────────────────
# Quick sanity check when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/openaq_merged.csv"

    print(f"Loading {filepath} ...")
    df_raw = load_data(filepath)
    print(f"  Raw shape: {df_raw.shape}")

    X, y, df_clean = prepare_features(df_raw)
    print(f"  Clean shape: {df_clean.shape}")
    print(f"  Features: {X.shape[1]} columns")
    print(f"  Target — PM2.5 mean: {y.mean():.2f} µg/m³")
    print(f"\n  AQI distribution:\n{df_clean['aqi_label'].value_counts()}")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\n  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print("Done ✓")
