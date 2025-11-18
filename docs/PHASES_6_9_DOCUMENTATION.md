# Phases 6-9 Implementation Documentation

Complete documentation for advanced MLOps, data management, reporting, and explainability features.

## Table of Contents

- [Phase 6: MLOps & Automation](#phase-6-mlops--automation)
- [Phase 7: Data Management](#phase-7-data-management)
- [Phase 8: Documentation & Reporting](#phase-8-documentation--reporting)
- [Phase 9: Advanced Features](#phase-9-advanced-features)
- [Getting Started](#getting-started)
- [Examples](#examples)

---

## Phase 6: MLOps & Automation

### Overview

Automated model lifecycle management with retraining, monitoring, and hyperparameter optimization.

### Components

#### 1. Automated Retraining Pipeline (`src/mlops/retraining.py`)

**Features:**
- Scheduled model retraining based on performance degradation or time
- Automatic model promotion if new model performs better
- Rollback capability for underperforming models
- Training history tracking

**Usage:**

```python
from src.mlops.retraining import AutomatedRetrainingPipeline

# Initialize pipeline
pipeline = AutomatedRetrainingPipeline()

# Run retraining (checks criteria automatically)
results = pipeline.retrain_pipeline(force=False)

# Force retraining
results = pipeline.retrain_pipeline(force=True)

# Retrain specific models only
results = pipeline.retrain_pipeline(
    models_to_train=["random_forest", "xgboost"]
)
```

**Configuration:**

```yaml
# In configs/config.yaml
retraining:
  performance_threshold: 0.05  # 5% performance drop triggers retraining
  min_samples_required: 100    # Minimum samples for retraining
  interval_days: 30            # Retrain every 30 days
```

**Retraining Triggers:**
- Performance degradation > 5%
- 30 days since last training
- Data drift detected
- Manual trigger

#### 2. Data Drift Detection (`src/mlops/drift_detection.py`)

**Features:**
- Multiple drift detection methods:
  - Kolmogorov-Smirnov test
  - Chi-square test
  - Population Stability Index (PSI)
  - Jensen-Shannon divergence
- Numerical and categorical feature support
- Drift reporting and history

**Usage:**

```python
from src.mlops.drift_detection import DriftDetector

# Initialize detector
detector = DriftDetector(significance_level=0.05)

# Detect drift
results = detector.detect_drift(
    reference_data=reference_df,
    current_data=current_df
)

# Check results
print(f"Drift detected: {results['overall_drift']}")
print(f"Drifted features: {results['drifted_features']}")
print(f"Drift score: {results['drift_score']:.2%}")

# Get drift summary
summary = detector.get_drift_summary()
```

**Interpretation:**
- **PSI < 0.1:** No significant drift
- **PSI 0.1-0.2:** Moderate drift - monitor
- **PSI > 0.2:** Significant drift - consider retraining

#### 3. Hyperparameter Tuning (`src/mlops/hyperparameter_tuning.py`)

**Features:**
- Bayesian optimization with Optuna
- Automatic pruning of unpromising trials
- Study persistence and resumption
- Visualization of optimization history

**Usage:**

```python
from src.mlops.hyperparameter_tuning import HyperparameterTuner

# Initialize tuner
tuner = HyperparameterTuner(n_trials=100, cv_folds=5)

# Tune single model
results = tuner.tune_model(
    model_name="random_forest",
    X=X_train,
    y=y_train,
    metric="accuracy"
)

print(f"Best accuracy: {results['best_value']:.4f}")
print(f"Best params: {results['best_params']}")

# Tune multiple models
all_results = tuner.tune_all_models(X_train, y_train)

# Visualize optimization
tuner.visualize_optimization("random_forest")
```

**Supported Models:**
- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM
- KNN
- Decision Tree
- XGBoost, LightGBM, CatBoost (extend as needed)

---

## Phase 7: Data Management

### Overview

Production-grade data management with prediction storage, feature store, and data versioning.

### Components

#### 1. Prediction Store (`src/mlops/prediction_store.py`)

**Features:**
- SQLite database for predictions
- Batch insertion support
- Performance tracking
- Drift event logging

**Database Schema:**

```sql
predictions (
    id, timestamp, model_name, model_version,
    prediction, prediction_label, confidence,
    probability_no_disease, probability_disease,
    age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak,
    actual_label, correct, inference_time_ms,
    user_id, session_id, metadata
)

model_performance (
    id, timestamp, model_name, model_version,
    metric_name, metric_value, dataset, samples_count
)

drift_events (
    id, timestamp, feature_name, drift_method, drift_score,
    drift_detected, reference_dates, current_dates
)
```

**Usage:**

```python
from src.mlops.prediction_store import PredictionStore

# Initialize store
store = PredictionStore()

# Store prediction
pred_id = store.store_prediction(
    prediction=1,
    prediction_label="Disease",
    model_name="random_forest",
    input_data={
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3
    },
    probability={"no_disease": 0.23, "disease": 0.77},
    confidence=0.77
)

# Get recent predictions
recent = store.get_recent_predictions(limit=100, model_name="random_forest")

# Get statistics
stats = store.get_prediction_stats()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Accuracy: {stats['accuracy']:.2%}")

# Export to CSV
store.export_to_csv("predictions_export.csv")
```

#### 2. Feature Store (`src/mlops/feature_store.py`)

**Features:**
- Feature registration and metadata
- Feature versioning
- Feature group management
- Feature statistics computation

**Usage:**

```python
from src.mlops.feature_store import FeatureStore

# Initialize store
store = FeatureStore()

# Register feature
store.register_feature(
    name="age",
    dtype="int",
    description="Patient age in years",
    feature_group="raw_features",
    tags=["demographic", "raw"]
)

# Register feature group
store.register_feature_group(
    name="vital_signs",
    features=["trestbps", "chol", "thalach"],
    description="Vital sign measurements"
)

# Store computed features
feature_set_id = store.store_features(
    data=features_df,
    feature_group="engineered_features"
)

# Retrieve features
features = store.get_features(
    feature_group="engineered_features",
    latest=True
)

# Compute statistics
stats = store.compute_feature_statistics("raw_features")
```

#### 3. Data Versioning with DVC

**Setup:**

```bash
# Initialize DVC
dvc init

# Track data
dvc add data/raw/cleveland.csv
dvc add data/raw/hungarian.csv

# Commit DVC files
git add data/raw/*.dvc .dvc/config
git commit -m "Track data with DVC"

# Configure remote storage (e.g., S3)
dvc remote add -d myremote s3://mybucket/dvcstore
dvc push

# Pull data
dvc pull
```

**Benefits:**
- Version control for datasets
- Efficient storage of large files
- Reproducible data pipelines
- Collaboration support

---

## Phase 8: Documentation & Reporting

### Overview

Automated documentation and reporting for model transparency and compliance.

### Components

#### 1. Report Generator (`src/reporting/report_generator.py`)

**Features:**
- HTML model evaluation reports
- Drift detection reports
- Performance comparison reports
- Automated recommendations

**Usage:**

```python
from src.reporting.report_generator import ReportGenerator

# Initialize generator
generator = ReportGenerator()

# Generate model evaluation report
results = {
    "models": {
        "random_forest": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87,
            "f1": 0.85,
            "roc_auc": 0.90
        },
        "xgboost": {
            "accuracy": 0.87,
            "precision": 0.86,
            "recall": 0.88,
            "f1": 0.87,
            "roc_auc": 0.92
        }
    }
}

report_path = generator.generate_model_evaluation_report(results)

# Generate drift report
drift_report = generator.generate_drift_report(drift_results)
```

**Report Contents:**
- Model performance metrics
- Best model recommendations
- Performance trends
- Actionable insights
- Visualizations

#### 2. Model Cards (`src/reporting/model_card.py`)

**Features:**
- Standardized model documentation
- Intended use and limitations
- Ethical considerations
- Performance characteristics

**Usage:**

```python
from src.reporting.model_card import generate_heart_disease_model_card

# Generate model card
metrics = {
    "accuracy": "0.8500",
    "precision": "0.8300",
    "recall": "0.8700",
    "f1": "0.8500",
    "roc_auc": "0.9000"
}

card_path = generate_heart_disease_model_card("random_forest", metrics)
```

**Model Card Sections:**
- Model Details
- Intended Use
- Factors
- Metrics
- Training Data
- Evaluation Data
- Ethical Considerations
- Caveats and Recommendations

---

## Phase 9: Advanced Features

### Overview

Model explainability, monitoring, and advanced analytics.

### Components

#### 1. Model Explainability (`src/explainability/explainer.py`)

**Features:**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance visualization
- Individual prediction explanations

**Usage:**

```python
from src.explainability.explainer import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model, feature_names=X_train.columns.tolist())

# Initialize SHAP
explainer.initialize_shap(X_train[:100])

# Explain predictions
shap_values = explainer.explain_prediction_shap(X_test[:10], save_plot=True)

# Plot feature importance
explainer.plot_feature_importance_shap(X_test[:100], save_plot=True)

# Get feature contributions for single prediction
contributions = explainer.get_feature_contributions(X_test, instance_idx=0)

for feature, value in list(contributions.items())[:5]:
    direction = "→ Disease" if value > 0 else "→ No Disease"
    print(f"{feature}: {value:+.4f} {direction}")

# Generate explanation report
explainer.generate_explanation_report(X_test, instance_idx=0)
```

**Initialize LIME:**

```python
# Initialize LIME
explainer.initialize_lime(X_train)

# Explain with LIME
lime_exp = explainer.explain_prediction_lime(
    X_test.values,
    instance_idx=0,
    num_features=10,
    save_plot=True
)
```

**Interpretation:**
- **Positive SHAP values:** Push prediction towards "Disease"
- **Negative SHAP values:** Push prediction towards "No Disease"
- **Magnitude:** Importance of the contribution

#### 2. Advanced Monitoring (`src/mlops/monitoring.py`)

**Features:**
- Performance monitoring with alerts
- Drift detection monitoring
- Prediction quality tracking
- Custom alert handlers

**Usage:**

```python
from src.mlops.monitoring import ModelMonitor, email_alert_handler

# Initialize monitor
monitor = ModelMonitor()

# Add custom alert handler
monitor.add_alert_handler(
    lambda alert: print(f"CUSTOM ALERT: {alert['message']}")
)

# Check performance
alerts = monitor.check_performance(
    current_metrics={"accuracy": 0.80, "f1": 0.79},
    baseline_metrics={"accuracy": 0.87, "f1": 0.85}
)

# Check drift
drift_alerts = monitor.check_drift(drift_results)

# Check prediction quality
pred_alerts = monitor.check_prediction_quality(
    predictions=predictions_list,
    window_hours=24
)

# Get alert summary
summary = monitor.get_alert_summary(hours=24)
print(f"Total alerts: {summary['total_alerts']}")
print(f"By severity: {summary['alerts_by_severity']}")
```

**Alert Types:**
- `performance_degradation`: Model accuracy dropped
- `data_drift`: Significant feature drift detected
- `high_error_rate`: Error rate exceeds threshold
- `high_latency`: Inference time too high
- `prediction_distribution`: Unusual prediction patterns

**Alert Handlers:**

```python
# Email alerts
smtp_config = {
    "host": "smtp.gmail.com",
    "port": 587,
    "username": "your@email.com",
    "password": "password",
    "from": "alerts@ml.com",
    "to": "team@ml.com"
}

monitor.add_alert_handler(
    lambda alert: email_alert_handler(alert, smtp_config)
)

# Slack alerts
webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
monitor.add_alert_handler(
    lambda alert: slack_alert_handler(alert, webhook_url)
)
```

---

## Getting Started

### Installation

All required dependencies are in `requirements.txt`:

```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU support
pip install -r requirements-gpu.txt
```

### Quick Start

#### 1. Run Automated Retraining

```bash
# Run retraining check
python -m src.mlops.retraining

# Force retraining
python -c "from src.mlops.retraining import AutomatedRetrainingPipeline; pipeline = AutomatedRetrainingPipeline(); pipeline.retrain_pipeline(force=True)"
```

#### 2. Detect Data Drift

```bash
python -m src.mlops.drift_detection
```

#### 3. Tune Hyperparameters

```bash
python -m src.mlops.hyperparameter_tuning
```

#### 4. Generate Reports

```bash
python -m src.reporting.report_generator
```

#### 5. Explain Models

```bash
python -m src.explainability.explainer
```

---

## Examples

### Complete MLOps Workflow

```python
# 1. Train initial model
from src.models.train import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)
trainer.save_all_models()

# 2. Set up prediction tracking
from src.mlops.prediction_store import PredictionStore

store = PredictionStore()

# Make predictions and store them
for idx, row in X_test.iterrows():
    pred = model.predict([row])[0]
    proba = model.predict_proba([row])[0]

    store.store_prediction(
        prediction=pred,
        prediction_label="Disease" if pred == 1 else "No Disease",
        model_name="random_forest",
        input_data=row.to_dict(),
        probability={"no_disease": proba[0], "disease": proba[1]},
        confidence=max(proba)
    )

# 3. Monitor performance
from src.mlops.monitoring import ModelMonitor

monitor = ModelMonitor()
alerts = monitor.check_performance(current_metrics, baseline_metrics)

# 4. Detect drift
from src.mlops.drift_detection import DriftDetector

detector = DriftDetector()
drift_results = detector.detect_drift(reference_data, current_data)

# 5. Auto-retrain if needed
from src.mlops.retraining import AutomatedRetrainingPipeline

pipeline = AutomatedRetrainingPipeline()
results = pipeline.retrain_pipeline()

# 6. Generate reports
from src.reporting.report_generator import ReportGenerator

generator = ReportGenerator()
generator.generate_model_evaluation_report(results)
generator.generate_drift_report(drift_results)

# 7. Explain predictions
from src.explainability.explainer import ModelExplainer

explainer = ModelExplainer(model, feature_names=X_train.columns.tolist())
explainer.initialize_shap(X_train[:100])
contributions = explainer.get_feature_contributions(X_test, instance_idx=0)
```

### Scheduled Retraining

Use cron for scheduled retraining:

```bash
# Add to crontab (weekly retraining on Sunday at 2 AM)
0 2 * * 0 cd /path/to/project && python -m src.mlops.retraining >> /var/log/ml_retraining.log 2>&1
```

### Continuous Monitoring

```python
import time
from src.mlops.monitoring import ModelMonitor
from src.mlops.prediction_store import PredictionStore

monitor = ModelMonitor()
store = PredictionStore()

while True:
    # Get recent predictions
    recent_preds = store.get_recent_predictions(limit=1000)

    # Convert to dict format
    predictions = recent_preds.to_dict('records')

    # Check quality
    alerts = monitor.check_prediction_quality(predictions, window_hours=1)

    if alerts:
        print(f"Generated {len(alerts)} alerts")

    # Wait 1 hour
    time.sleep(3600)
```

---

## Best Practices

### 1. Retraining

- Monitor model performance continuously
- Set appropriate retraining thresholds based on your use case
- Always validate new models before promotion
- Keep backup of previous model versions
- Document retraining triggers and results

### 2. Drift Detection

- Run drift detection regularly (e.g., weekly)
- Investigate drifted features to understand causes
- Consider retraining when significant drift detected
- Track drift over time to identify trends

### 3. Monitoring

- Set up automated alerts for critical metrics
- Review alert summaries regularly
- Investigate root causes of alerts
- Adjust thresholds based on operational experience

### 4. Explainability

- Generate explanations for high-stakes predictions
- Use SHAP for global feature importance
- Use LIME for local explanations
- Document explanation methodology in model cards

### 5. Documentation

- Generate model cards for all production models
- Update documentation when models are retrained
- Include ethical considerations and limitations
- Maintain change log of model versions

---

## Troubleshooting

### SHAP Installation Issues

```bash
# If SHAP fails to install
pip install shap --no-build-isolation

# For specific Python versions
pip install shap==0.41.0
```

### SQLite Locked Database

```bash
# Increase timeout in prediction_store.py
with sqlite3.connect(self.db_path, timeout=30.0) as conn:
    ...
```

### Memory Issues with SHAP

```python
# Use smaller background dataset
explainer.initialize_shap(X_train[:50])  # Instead of 100+

# Use KernelExplainer for large models
explainer.initialize_shap(X_background, model_type="kernel")
```

---

## Support

For issues or questions:
- Check the [main README](../README.md)
- Review [API documentation](API_DOCUMENTATION.md)
- Open an issue on GitHub

---

**Last Updated:** 2024-01-XX
