"""Central configuration: paths, constants and column names shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT / "data" / "raw" / "cardio_train.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "cleaned.csv"
MODEL_PATH = ROOT / "models" / "pulsepredict_pipeline.joblib"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"
DEFAULT_DB_PATH = ROOT / "data" / "pulsepredict.db"

MODEL_VERSION = "1.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Probability at or above which a profile is labelled "Higher-risk".
RISK_THRESHOLD = 0.5

TARGET = "target"

# The columns the *deployed* model expects as raw input (BMI is derived inside the pipeline).
MODEL_INPUT_COLUMNS = [
    "age_years",
    "sex",
    "height",
    "weight",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "glucose",
    "smoking",
    "alcohol",
    "active",
]
