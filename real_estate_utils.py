"""Shared helpers for the Real Estate Investment Advisor project.

Used by BOTH the Jupyter notebook (training) and the Streamlit app (inference),
so the features are always built identically.
"""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CURRENT_YEAR = 2025
HORIZON_YEARS = 5
BASE_GROWTH_RATE = 0.08            # fixed-rate baseline: Future = Current * (1 + r) ** t
GOOD_INVESTMENT_THRESHOLD = 4      # multi-factor score (0-7) >= threshold -> Good Investment
TRANSPORT_MAP = {"low": 1, "medium": 2, "high": 3}
NEG_VALUES = {"no", "none", "n", "false", "0", "0.0", "nan", "", "na", "null"}

NUM_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property",
    "Floor_No", "Total_Floors", "Floor_Ratio", "Nearby_Schools", "Nearby_Hospitals",
    "Parking_Flag", "Security_Flag", "Amenity_Count", "Transport_Score", "Is_Ready",
    "School_Density_Score", "Hospital_Score", "Infrastructure_Score",
    "City_Median_PPSF", "PPSF_vs_City",
]
CAT_FEATURES = [
    "State", "City", "Property_Type", "Furnished_Status", "Facing",
    "Owner_Type", "Availability_Status",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

RAW_NUMERIC_COLS = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
]


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def to_binary(s: pd.Series) -> pd.Series:
    """Turn Yes/No, True/False, 0/1 or counts (>0) into a 0/1 flag."""
    s = pd.Series(s)
    num = pd.to_numeric(s, errors="coerce")
    text_flag = ~s.astype(str).str.strip().str.lower().isin(NEG_VALUES)
    out = np.where(num.notna(), num > 0, text_flag)
    return pd.Series(out.astype(int), index=s.index)


def count_amenities(x) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return len([a for a in str(x).split(",") if a.strip()])


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text, drop duplicates, fix types, impute missing values."""
    d = df.copy()
    d.columns = d.columns.str.strip()

    text_cols = [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]
    for c in text_cols:
        d[c] = d[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # duplicates (ignore the ID column, which is unique by construction)
    subset = [c for c in d.columns if c != "ID"]
    d = d.drop_duplicates(subset=subset).reset_index(drop=True)

    for c in RAW_NUMERIC_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # price and size are essential: drop rows where they are missing / invalid
    d = d.dropna(subset=["Size_in_SqFt", "Price_in_Lakhs"])
    d = d[(d["Size_in_SqFt"] > 0) & (d["Price_in_Lakhs"] > 0)].reset_index(drop=True)

    # impute: numeric -> median, categorical -> mode
    for c in RAW_NUMERIC_COLS:
        if c in d.columns and d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())
    for c in text_cols:
        if c != "ID" and d[c].isna().any():
            mode = d[c].mode(dropna=True)
            d[c] = d[c].fillna(mode.iloc[0] if len(mode) else "Unknown")

    # consistency fixes
    d["Age_of_Property"] = (CURRENT_YEAR - d["Year_Built"]).clip(lower=0)
    d["Total_Floors"] = np.maximum(d["Total_Floors"], d["Floor_No"])
    d["Price_per_SqFt"] = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]   # Rs per sq ft
    return d


def iqr_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def cap_outliers(df: pd.DataFrame, cols=("Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt"), k: float = 3.0):
    """Winsorise extreme values with an IQR rule (keeps all rows)."""
    d = df.copy()
    bounds = {}
    for c in cols:
        lo, hi = iqr_bounds(d[c], k)
        lo = max(lo, d[c].min())
        bounds[c] = (float(lo), float(hi))
        d[c] = d[c].clip(lo, hi)
    d["Price_per_SqFt"] = d["Price_in_Lakhs"] * 1e5 / d["Size_in_SqFt"]
    return d, bounds


# --------------------------------------------------------------------------- #
# Meta (statistics learned from training data, reused at inference time)
# --------------------------------------------------------------------------- #
def fit_meta(df: pd.DataFrame) -> dict:
    d = df
    city_ppsf = d.groupby("City")["Price_per_SqFt"].median()
    rank_pct = city_ppsf.rank(pct=True)
    amen_counts = d["Amenities"].apply(count_amenities)
    amen_items = (
        d["Amenities"].dropna().astype(str).str.split(",").explode().str.strip()
    )
    meta = {
        "current_year": CURRENT_YEAR,
        "global_median_ppsf": float(d["Price_per_SqFt"].median()),
        "global_median_price": float(d["Price_in_Lakhs"].median()),
        "city_median_ppsf": {k: float(v) for k, v in city_ppsf.items()},
        "city_type_median_price": {
            f"{c}|{t}": float(v)
            for (c, t), v in d.groupby(["City", "Property_Type"])["Price_in_Lakhs"].median().items()
        },
        # costlier cities get slightly higher growth, cheaper ones slightly lower (+/-1%)
        "city_growth_adj": {k: float((v - 0.5) * 0.02) for k, v in rank_pct.items()},
        "schools_max": float(max(d["Nearby_Schools"].max(), 1)),
        "hospitals_max": float(max(d["Nearby_Hospitals"].max(), 1)),
        "amenity_max": float(max(amen_counts.max(), 1)),
        "city_state": d.drop_duplicates("City").set_index("City")["State"].to_dict(),
        "amenity_options": sorted(amen_items[amen_items != ""].unique().tolist()),
        "transport_options": sorted(d["Public_Transport_Accessibility"].dropna().unique().tolist()),
        "categories": {c: sorted(d[c].dropna().unique().tolist()) for c in CAT_FEATURES},
        "ranges": {
            "size": [float(d["Size_in_SqFt"].min()), float(d["Size_in_SqFt"].max())],
            "price": [float(d["Price_in_Lakhs"].min()), float(d["Price_in_Lakhs"].max())],
            "year_built": [int(d["Year_Built"].min()), int(d["Year_Built"].max())],
            "bhk": [int(d["BHK"].min()), int(d["BHK"].max())],
            "floors": [int(d["Total_Floors"].max())],
            "schools_median": float(d["Nearby_Schools"].median()),
            "hospitals_median": float(d["Nearby_Hospitals"].median()),
        },
    }
    return meta


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    d = df.copy()
    d["Transport_Score"] = (
        d["Public_Transport_Accessibility"].astype(str).str.strip().str.lower().map(TRANSPORT_MAP).fillna(2)
    )
    d["Parking_Flag"] = to_binary(d["Parking_Space"])
    d["Security_Flag"] = to_binary(d["Security"])
    d["Amenity_Count"] = d["Amenities"].apply(count_amenities)
    d["Is_Ready"] = d["Availability_Status"].astype(str).str.lower().str.contains("ready").astype(int)
    d["Floor_Ratio"] = d["Floor_No"] / d["Total_Floors"].replace(0, np.nan)
    d["Floor_Ratio"] = d["Floor_Ratio"].fillna(0)

    d["School_Density_Score"] = (d["Nearby_Schools"] / meta["schools_max"]).clip(0, 1)
    d["Hospital_Score"] = (d["Nearby_Hospitals"] / meta["hospitals_max"]).clip(0, 1)
    d["Infrastructure_Score"] = (
        d["School_Density_Score"]
        + d["Hospital_Score"]
        + d["Transport_Score"] / 3
        + (d["Amenity_Count"] / meta["amenity_max"]).clip(0, 1)
        + d["Parking_Flag"]
        + d["Security_Flag"]
    ) / 6

    d["City_Median_PPSF"] = d["City"].map(meta["city_median_ppsf"]).fillna(meta["global_median_ppsf"])
    d["PPSF_vs_City"] = d["Price_per_SqFt"] / d["City_Median_PPSF"]
    return d


# --------------------------------------------------------------------------- #
# Target creation (domain rules)
# --------------------------------------------------------------------------- #
def make_targets(df: pd.DataFrame, meta: dict, seed: int = 42) -> pd.DataFrame:
    """Create 'Good_Investment' (classification) and 'Future_Price_5Y_Lakhs' (regression)."""
    d = df.copy()
    rng = np.random.default_rng(seed)

    # ---- Classification: multi-factor score (max 7) ------------------------
    meta["infra_median"] = float(d["Infrastructure_Score"].median())
    city_type_med = np.array([
        meta["city_type_median_price"].get(f"{c}|{t}", meta["global_median_price"])
        for c, t in zip(d["City"], d["Property_Type"])
    ])
    d["Investment_Score"] = (
        2 * (d["Price_per_SqFt"] <= d["City_Median_PPSF"]).astype(int)    # cheaper per sqft than city median
        + (d["Price_in_Lakhs"] <= city_type_med).astype(int)             # cheaper than city/type median price
        + (d["BHK"] >= 3).astype(int)
        + d["Is_Ready"]
        + (d["Transport_Score"] >= 2).astype(int)
        + (d["Infrastructure_Score"] >= meta["infra_median"]).astype(int)
    )
    d["Good_Investment"] = (d["Investment_Score"] >= GOOD_INVESTMENT_THRESHOLD).astype(int)

    # ---- Regression: feature-based annual growth rate ----------------------
    city_adj = d["City"].map(meta["city_growth_adj"]).fillna(0.0)
    rate = (
        BASE_GROWTH_RATE
        + 0.010 * (d["Transport_Score"] - 2)
        + 0.030 * (d["Infrastructure_Score"] - 0.5)
        + 0.005 * d["Is_Ready"]
        + city_adj
        + rng.normal(0, 0.004, len(d))
    ).clip(0.03, 0.15)
    d["Annual_Growth_Rate"] = rate
    d["Future_Price_5Y_Lakhs"] = d["Price_in_Lakhs"] * (1 + rate) ** HORIZON_YEARS
    return d
