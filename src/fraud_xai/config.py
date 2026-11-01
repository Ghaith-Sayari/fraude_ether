"""Global constants and paths (single source of truth)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_BUNDLE = ARTIFACTS_DIR / "latest_bundle.joblib"

RANDOM_STATE = 42
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train / validation / test

TARGET_CANDIDATES = (
    "flag", "is_fraud", "isfraud", "fraud", "fraud_bool", "is fraudulent",
    "is_fraudulent", "fraudfound_p", "fraud_reported", "potentialfraud",
    "is_laundering", "class", "label", "target",
)
TIME_CANDIDATES = ("time", "date", "timestamp", "trans_date", "step", "month", "datetime")
AMOUNT_CANDIDATES = ("amt", "amount", "transaction amount", "transactionamt", "value")
