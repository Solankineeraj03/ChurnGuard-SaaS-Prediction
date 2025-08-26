from pathlib import Path
import json
import joblib
import shap
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from app.utils import load_csv, split_Xy, ensure_dir, FEATURES

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODEL_PATH = MODELS_DIR / "xgb_churn.pkl"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"
BASELINE_X_PATH = MODELS_DIR / "baseline_X_sample.pkl"

def _plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_shap_summary(explainer, X, out_path: Path):
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(6,3.5))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def train(dataset_path: str = "data/sample_data/churn.csv", test_size: float = 0.2, seed: int = 42):
    ensure_dir(MODELS_DIR); ensure_dir(REPORTS_DIR)
    df = load_csv(dataset_path)
    X, y = split_Xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    mlflow.set_experiment("churnguard")
    with mlflow.start_run(run_name="xgb_baseline"):
        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=seed, n_jobs=-1, tree_method="hist",
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.08)

        report_text = classification_report(y_test, preds, digits=3)
        (REPORTS_DIR / "classification_report.txt").write_text(report_text)

        # save artifacts
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, EXPLAINER_PATH)
        mlflow.log_artifact(str(EXPLAINER_PATH))

        baseline_X = X_train.sample(min(200, len(X_train)), random_state=seed)
        joblib.dump(baseline_X, BASELINE_X_PATH)
        mlflow.log_artifact(str(BASELINE_X_PATH))

        # plots
        _plot_confusion_matrix(y_test, preds, REPORTS_DIR / "confusion_matrix.png")
        mlflow.log_artifact(str(REPORTS_DIR / "confusion_matrix.png"))

        try:
            _plot_shap_summary(explainer, baseline_X, REPORTS_DIR / "shap_summary.png")
            mlflow.log_artifact(str(REPORTS_DIR / "shap_summary.png"))
        except Exception:
            pass  # fallback if SHAP plotting breaks in headless env

        summary = {
            "roc_auc": float(auc),
            "n_features": len(FEATURES),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURES,
        }
        (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(REPORTS_DIR / "summary.json"))

        return summary

if __name__ == "__main__":
    s = train()
    print(s)
