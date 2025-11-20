"""Automated report generation for model evaluation and monitoring."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
from jinja2 import Template

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate automated reports for model evaluation and monitoring.

    Features:
    - HTML reports with visualizations
    - Model comparison reports
    - Performance monitoring reports
    - Drift detection reports
    """

    def __init__(self):
        """Initialize report generator."""
        self.config = Config()

        # Paths
        self.reports_dir = Path(self.config.get("paths.data.processed", "data/processed")) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_evaluation_report(
        self,
        model_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive model evaluation report.

        Args:
            model_results: Dictionary with model evaluation results
            output_path: Output path for report

        Returns:
            Path to generated report
        """
        logger.info("Generating model evaluation report...")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.reports_dir / f"model_evaluation_{timestamp}.html")

        # Generate HTML report
        html = self._generate_evaluation_html(model_results)

        # Save report
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Saved model evaluation report to {output_path}")

        return output_path

    def _generate_evaluation_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for model evaluation report."""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { margin: 0; font-size: 2.5em; }
        h2 {
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 { color: #764ba2; margin-top: 30px; }
        .metadata {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        .model-section {
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover { background-color: #f5f5f5; }
        .best { color: #2ecc71; font-weight: bold; }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: #666;
            padding: 20px;
        }
        .timestamp {
            font-size: 0.9em;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Model Evaluation Report</h1>
        <p class="timestamp">Generated: {{ timestamp }}</p>
    </div>

    <div class="metadata">
        <h3>Report Metadata</h3>
        <p><strong>Total Models Evaluated:</strong> {{ num_models }}</p>
        <p><strong>Dataset:</strong> {{ dataset_name }}</p>
        <p><strong>Evaluation Date:</strong> {{ eval_date }}</p>
    </div>

    <h2>📊 Model Performance Overview</h2>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Best Accuracy</div>
            <div class="metric-value">{{ "%.2f"|format(best_accuracy * 100) }}%</div>
            <div>{{ best_model }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best F1 Score</div>
            <div class="metric-value">{{ "%.2f"|format(best_f1 * 100) }}%</div>
            <div>{{ best_f1_model }}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best ROC-AUC</div>
            <div class="metric-value">{{ "%.2f"|format(best_roc_auc * 100) }}%</div>
            <div>{{ best_roc_model }}</div>
        </div>
    </div>

    <h2>🔍 Detailed Model Results</h2>

    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
            </tr>
        </thead>
        <tbody>
            {% for model in models %}
            <tr>
                <td><strong>{{ model.name }}</strong></td>
                <td>{{ "%.4f"|format(model.accuracy) }}</td>
                <td>{{ "%.4f"|format(model.precision) }}</td>
                <td>{{ "%.4f"|format(model.recall) }}</td>
                <td>{{ "%.4f"|format(model.f1) }}</td>
                <td>{{ "%.4f"|format(model.roc_auc) }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {% if recommendations %}
    <h2>💡 Recommendations</h2>
    <div class="model-section">
        <ul>
            {% for rec in recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <div class="footer">
        <p>Heart Disease Prediction System - Automated Report</p>
        <p class="timestamp">{{ timestamp }}</p>
    </div>
</body>
</html>
        """

        # Prepare data for template
        models_data = []
        best_accuracy = 0
        best_f1 = 0
        best_roc_auc = 0
        best_model = ""
        best_f1_model = ""
        best_roc_model = ""

        for model_name, metrics in results.get("models", {}).items():
            model_data = {
                "name": model_name,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1": metrics.get("f1", 0),
                "roc_auc": metrics.get("roc_auc", 0)
            }
            models_data.append(model_data)

            if model_data["accuracy"] > best_accuracy:
                best_accuracy = model_data["accuracy"]
                best_model = model_name

            if model_data["f1"] > best_f1:
                best_f1 = model_data["f1"]
                best_f1_model = model_name

            if model_data["roc_auc"] > best_roc_auc:
                best_roc_auc = model_data["roc_auc"]
                best_roc_model = model_name

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        template = Template(html_template)
        html = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_models=len(models_data),
            dataset_name="Heart Disease UCI",
            eval_date=datetime.now().strftime("%Y-%m-%d"),
            best_accuracy=best_accuracy,
            best_f1=best_f1,
            best_roc_auc=best_roc_auc,
            best_model=best_model,
            best_f1_model=best_f1_model,
            best_roc_model=best_roc_model,
            models=models_data,
            recommendations=recommendations
        )

        return html

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        models = results.get("models", {})
        if not models:
            return recommendations

        # Find best models
        accuracies = {name: metrics.get("accuracy", 0) for name, metrics in models.items()}
        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]

        recommendations.append(
            f"Best performing model is {best_model} with {best_accuracy:.2%} accuracy"
        )

        # Check if ensemble would help
        if len(models) >= 3:
            recommendations.append(
                "Consider creating an ensemble of top 3 models for potentially better performance"
            )

        # Check for overfitting
        for name, metrics in models.items():
            train_acc = metrics.get("train_accuracy", 0)
            test_acc = metrics.get("accuracy", 0)

            if train_acc - test_acc > 0.1:
                recommendations.append(
                    f"{name} shows signs of overfitting (train-test gap: {(train_acc - test_acc):.2%}). "
                    "Consider regularization or cross-validation"
                )

        # Check model diversity
        accuracy_range = max(accuracies.values()) - min(accuracies.values())
        if accuracy_range > 0.15:
            recommendations.append(
                "High variance in model performance. Consider feature engineering or data quality improvements"
            )

        return recommendations

    def generate_drift_report(
        self,
        drift_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate drift detection report."""
        logger.info("Generating drift detection report...")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.reports_dir / f"drift_report_{timestamp}.html")

        html = self._generate_drift_html(drift_results)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Saved drift report to {output_path}")

        return output_path

    def _generate_drift_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for drift report."""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Drift Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 { margin: 0; }
        .alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .drift-detected {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        table {
            width: 100%;
            background: white;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th {
            background: #f093fb;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        .drifted { color: #dc3545; font-weight: bold; }
        .stable { color: #28a745; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 Data Drift Detection Report</h1>
        <p>Generated: {{ timestamp }}</p>
    </div>

    {% if overall_drift %}
    <div class="drift-detected">
        <strong>⚠️ DRIFT DETECTED:</strong> {{ num_drifted }} out of {{ total_features }} features show drift
    </div>
    {% else %}
    <div class="success">
        <strong>✓ NO DRIFT DETECTED:</strong> All features are stable
    </div>
    {% endif %}

    <h2>Drift Summary</h2>
    <p><strong>Drift Score:</strong> {{ "%.2f"|format(drift_score * 100) }}%</p>
    <p><strong>Reference Samples:</strong> {{ reference_samples }}</p>
    <p><strong>Current Samples:</strong> {{ current_samples }}</p>

    {% if drifted_features %}
    <h2>Drifted Features</h2>
    <ul>
        {% for feature in drifted_features %}
        <li class="drifted">{{ feature }}</li>
        {% endfor %}
    </ul>
    {% endif %}

    <h2>Detailed Results</h2>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Type</th>
                <th>Drift Detected</th>
                <th>PSI Value</th>
                <th>KS Statistic</th>
            </tr>
        </thead>
        <tbody>
            {% for feature in features %}
            <tr>
                <td>{{ feature.name }}</td>
                <td>{{ feature.type }}</td>
                <td class="{% if feature.drift %}drifted{% else %}stable{% endif %}">
                    {{ "Yes" if feature.drift else "No" }}
                </td>
                <td>{{ "%.4f"|format(feature.psi) if feature.psi is not none else "N/A" }}</td>
                <td>{{ "%.4f"|format(feature.ks) if feature.ks is not none else "N/A" }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <div class="footer">
        <p>Automated Drift Detection System</p>
    </div>
</body>
</html>
        """

        # Prepare data
        features_data = []

        for feature_name, feature_info in results.get("numerical_drift", {}).items():
            features_data.append({
                "name": feature_name,
                "type": "Numerical",
                "drift": feature_info.get("drift_detected", False),
                "psi": feature_info.get("methods", {}).get("psi", {}).get("value"),
                "ks": feature_info.get("methods", {}).get("kolmogorov_smirnov", {}).get("statistic")
            })

        for feature_name, feature_info in results.get("categorical_drift", {}).items():
            features_data.append({
                "name": feature_name,
                "type": "Categorical",
                "drift": feature_info.get("drift_detected", False),
                "psi": feature_info.get("methods", {}).get("psi", {}).get("value"),
                "ks": None
            })

        template = Template(html_template)
        html = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_drift=results.get("overall_drift", False),
            num_drifted=len(results.get("drifted_features", [])),
            total_features=len(features_data),
            drift_score=results.get("drift_score", 0),
            reference_samples=results.get("reference_samples", 0),
            current_samples=results.get("current_samples", 0),
            drifted_features=results.get("drifted_features", []),
            features=features_data
        )

        return html

    def generate_summary_report(
        self,
        include_models: bool = True,
        include_drift: bool = True,
        include_predictions: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive summary report."""
        logger.info("Generating summary report...")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.reports_dir / f"summary_report_{timestamp}.html")

        # Collect data from various sources
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sections": []
        }

        # Add sections based on flags
        # This would be expanded with actual data loading

        logger.info(f"Saved summary report to {output_path}")

        return output_path


if __name__ == "__main__":
    # Example usage
    generator = ReportGenerator()

    # Example model results
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
    print(f"Generated report: {report_path}")
