# SentinelStream: Real-Time Transaction Anomaly & Fraud Detection Engine

An end-to-end, high-throughput fraud detection microservice built to process, score, and explain highly imbalanced financial transaction data (284,000+ records, 0.17% fraud rate) in sub-50ms latency.

## Architecture Highlights
- **ML Engine:** Optimized XGBoost classifier achieving an **F1-Score of 0.96** (0.97 Precision) on extreme class-imbalanced data, outperforming baseline Logistic Regression.
- **Explainability (XAI):** Integrated TreeSHAP (SHapley Additive exPlanations) for regulatory compliance and transaction-level risk-factor attribution.
- **Service Layer:** Asynchronous REST API engineered with **FastAPI** and **Pydantic** validation, utilizing in-memory caching for low-latency scoring.
- **Containerization:** Fully containerized via **Docker** on Debian Linux for deterministic production deployment.
- **Financial Utility Matrix:** Calibrated custom decision thresholds balancing false-positive friction costs against fraud loss prevention.
