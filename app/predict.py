from pathlib import Path
import joblib
import numpy as np
from app.utils import FEATURES

MODEL_PATH = Path("models/xgb_churn.pkl")
EXPLAINER_PATH = Path("models/shap_explainer.pkl")
BASELINE_X_PATH = Path("models/baseline_X_sample.pkl")

class ChurnModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Train first (/train).")
        self.model = joblib.load(MODEL_PATH)
        self.explainer = joblib.load(EXPLAINER_PATH) if EXPLAINER_PATH.exists() else None

    def predict_one(self, payload: dict) -> dict:
        x = np.array([[payload.get(k) for k in FEATURES]], dtype=float)
        proba = float(self.model.predict_proba(x)[0, 1])
        pred = int(proba >= 0.5)

        explanation = None
        if self.explainer is not None:
            shap_vals = self.explainer.shap_values(x)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            explanation = {f: float(v) for f, v in zip(FEATURES, shap_vals[0])}

        return {"prediction": pred, "probability": proba, "explanation": explanation}
