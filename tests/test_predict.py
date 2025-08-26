from app.train import train
from app.predict import ChurnModel

def test_end_to_end_train_and_predict():
    # Train once on sample data
    summary = train("data/sample_data/churn.csv")
    assert "roc_auc" in summary

    # Load model and predict
    model = ChurnModel()
    payload = {
        "logins_per_week": 5,
        "avg_session_time": 18.2,
        "feedback_score": 3.0,
        "is_premium": 0,
        "tenure_weeks": 22,
    }
    out = model.predict_one(payload)
    assert "prediction" in out and "probability" in out
