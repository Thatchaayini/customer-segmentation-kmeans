"""Shared settings used across the project."""

RANDOM_STATE = 42
N_INIT = 10
INIT_METHOD = "k-means++"
DEFAULT_K = 3

DATA_PATH = "data/customers.csv"
SEGMENTED_PATH = "data/customers_segmented.csv"

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "AnnualIncome",
    "PurchaseFrequency",
    "AvgOrderValue",
]
GENDER_MAP = {"Male": 0, "Female": 1}
LINKAGE = "ward"
DBSCAN_EPS = 1.2
DBSCAN_MIN_SAMPLES = 5
CACHE_DIR = ".cache"