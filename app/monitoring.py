from prometheus_client import Counter, Histogram

PREDICTIONS = Counter("churn_predictions_total", "Total prediction calls")
TRAIN_RUNS = Counter("churn_train_runs_total", "Total training runs")

INFER_LATENCY = Histogram("churn_infer_latency_seconds", "Prediction latency")
