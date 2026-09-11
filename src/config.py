from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


# Dataset
DATA_PATH = RAW_DATA_DIR / "city_day.csv"


# Core pollutant features
POLLUTANT_FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
]


# Target
TARGET = "AQI"


# Reproducibility
RANDOM_STATE = 42


# K-Means
K_RANGE = range(2, 8)


# Regression time split
REGRESSION_SPLIT_DATE = "2019-01-01"