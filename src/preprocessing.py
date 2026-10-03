"""Feature engineering + preprocessing, packaged as a scikit-learn Pipeline.

The *whole* fitted pipeline (feature creation -> scaling/encoding -> classifier) is saved
with joblib, so the API runs exactly the same transformations that were used in training.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age_years", "height", "weight", "bmi", "systolic_bp", "diastolic_bp"]
CATEGORICAL_FEATURES = ["cholesterol", "glucose"]  # ordinal codes 1/2/3, one-hot encoded
BINARY_FEATURES = ["sex", "smoking", "alcohol", "active"]


def add_bmi(X: pd.DataFrame) -> pd.DataFrame:
    """Engineer BMI = weight (kg) / height (m)^2. Defined at module level so it can be pickled."""
    X = X.copy()
    X["bmi"] = X["weight"] / (X["height"] / 100.0) ** 2
    return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(categories=[[1, 2, 3], [1, 2, 3]], handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """Full pipeline: raw input frame -> BMI -> preprocess -> classifier."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_bmi)),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
