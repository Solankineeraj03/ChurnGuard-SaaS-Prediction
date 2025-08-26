# 🚀 ChurnGuard: SaaS Churn Prediction System  

Production-ready, end-to-end ML system for predicting SaaS customer churn with real-time serving, monitoring, and explainability.  

---

## 📊 Overview  
ChurnGuard is an ML pipeline + API for churn detection in SaaS platforms.  
- **Model**: XGBoost trained on behavioral + engagement metrics.  
- **Serving**: FastAPI with REST endpoints (`/train`, `/predict`).  
- **Explainability**: SHAP per-request feature attributions.  
- **Monitoring**: Prometheus metrics + Grafana dashboards.  
- **Experiment Tracking**: MLflow integration for versioning & metrics.  
- **Deployment**: Docker container, CI/CD ready.  

---

## 📂 Dataset  
Sample synthetic dataset at: `data/sample_data/churn.csv`  

| user_id | logins_per_week | avg_session_time | feedback_score | is_premium | tenure_weeks | churned |
|---------|-----------------|------------------|----------------|------------|--------------|---------|
| 1       | 10              | 32.5             | 4.3            | 1          | 80           | 0       |
| 2       | 3               | 12.0             | 2.1            | 0          | 10           | 1       |

---

## 🔑 Features  
- End-to-end training + serving pipeline  
- SHAP explainability for model predictions  
- Prometheus metrics for monitoring inference latency & usage  
- MLflow logging for experiments  
- Automated tests with Pytest  
- Dockerized for portability  

---

## ⚡ Quickstart  

### 1. Install dependencies
```bash
pip install -r requirements.txt

