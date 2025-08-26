from pathlib import Path
import pandas as pd

FEATURES = [
    "logins_per_week",
    "avg_session_time",
    "feedback_score",
    "is_premium",
    "tenure_weeks",
]
TARGET = "churned"

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return pd.read_csv(p)

def split_Xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)
