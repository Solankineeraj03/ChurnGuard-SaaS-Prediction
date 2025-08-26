
cat > README.md << 'EOF'
Production-style, end-to-end ML system:
- Train **XGBoost** on SaaS usage metrics
- Serve predictions via **FastAPI**
- Per-request **SHAP explanations**
- **Prometheus** `/metrics` for monitoring
- **MLflow** experiment tracking
- **Pytest** + GitHub **CI**
- Containerized with **Docker**
