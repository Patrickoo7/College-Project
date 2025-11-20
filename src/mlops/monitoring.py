"""Advanced monitoring and alerting for ML models."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelMonitor:
    """
    Monitor model performance and trigger alerts.

    Features:
    - Performance monitoring
    - Drift detection monitoring
    - Prediction quality monitoring
    - Alert thresholds and notifications
    """

    def __init__(self):
        """Initialize model monitor."""
        self.config = Config()
        self.metadata_dir = Path(self.config.get("paths.models.metadata", "models/metadata"))
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Alert thresholds
        self.thresholds = {
            "accuracy_drop": 0.05,  # 5% drop triggers alert
            "drift_score": 0.3,  # Drift score > 0.3 triggers alert
            "error_rate": 0.2,  # Error rate > 20% triggers alert
            "prediction_latency_ms": 1000,  # Latency > 1s triggers alert
        }

        # Alert handlers
        self.alert_handlers = []

    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Add custom alert handler.

        Args:
            handler: Callable that receives alert dictionary
        """
        self.alert_handlers.append(handler)

    def check_performance(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Check model performance and generate alerts if needed.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline metrics to compare against

        Returns:
            List of alerts
        """
        alerts = []

        if baseline_metrics is None:
            baseline_metrics = self._load_baseline_metrics()

        if not baseline_metrics:
            logger.info("No baseline metrics available for comparison")
            return alerts

        # Check accuracy drop
        current_acc = current_metrics.get("accuracy", 0)
        baseline_acc = baseline_metrics.get("accuracy", 0)

        if baseline_acc - current_acc > self.thresholds["accuracy_drop"]:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "performance_degradation",
                "severity": "high",
                "metric": "accuracy",
                "current_value": current_acc,
                "baseline_value": baseline_acc,
                "drop": baseline_acc - current_acc,
                "message": f"Accuracy dropped by {(baseline_acc - current_acc):.2%}"
            }
            alerts.append(alert)
            self._trigger_alert(alert)

        # Check other metrics
        for metric in ["precision", "recall", "f1"]:
            current_val = current_metrics.get(metric, 0)
            baseline_val = baseline_metrics.get(metric, 0)

            if baseline_val - current_val > self.thresholds["accuracy_drop"]:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "metric_degradation",
                    "severity": "medium",
                    "metric": metric,
                    "current_value": current_val,
                    "baseline_value": baseline_val,
                    "drop": baseline_val - current_val,
                    "message": f"{metric} dropped by {(baseline_val - current_val):.2%}"
                }
                alerts.append(alert)
                self._trigger_alert(alert)

        # Save alerts
        if alerts:
            self._save_alerts(alerts)

        return alerts

    def check_drift(self, drift_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for data drift and generate alerts.

        Args:
            drift_results: Results from drift detection

        Returns:
            List of alerts
        """
        alerts = []

        drift_score = drift_results.get("drift_score", 0)

        if drift_score > self.thresholds["drift_score"]:
            drifted_features = drift_results.get("drifted_features", [])

            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "data_drift",
                "severity": "high",
                "drift_score": drift_score,
                "num_drifted_features": len(drifted_features),
                "drifted_features": drifted_features[:5],  # Top 5
                "message": f"Data drift detected! {len(drifted_features)} features drifted"
            }
            alerts.append(alert)
            self._trigger_alert(alert)

        if alerts:
            self._save_alerts(alerts)

        return alerts

    def check_prediction_quality(
        self,
        predictions: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Monitor prediction quality over time.

        Args:
            predictions: List of prediction records
            window_hours: Time window to analyze

        Returns:
            List of alerts
        """
        alerts = []

        if not predictions:
            return alerts

        # Filter predictions within window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_predictions = [
            p for p in predictions
            if datetime.fromisoformat(p.get("timestamp", "")) > cutoff_time
        ]

        if not recent_predictions:
            return alerts

        # Calculate error rate (for labeled predictions)
        labeled = [p for p in recent_predictions if p.get("actual_label") is not None]

        if labeled:
            errors = sum(1 for p in labeled if not p.get("correct", False))
            error_rate = errors / len(labeled)

            if error_rate > self.thresholds["error_rate"]:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "high_error_rate",
                    "severity": "high",
                    "error_rate": error_rate,
                    "num_predictions": len(labeled),
                    "num_errors": errors,
                    "message": f"Error rate {error_rate:.2%} exceeds threshold"
                }
                alerts.append(alert)
                self._trigger_alert(alert)

        # Check prediction latency
        latencies = [p.get("inference_time_ms", 0) for p in recent_predictions if p.get("inference_time_ms")]

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            if avg_latency > self.thresholds["prediction_latency_ms"]:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "high_latency",
                    "severity": "medium",
                    "avg_latency_ms": avg_latency,
                    "max_latency_ms": max_latency,
                    "message": f"Average latency {avg_latency:.0f}ms exceeds threshold"
                }
                alerts.append(alert)
                self._trigger_alert(alert)

        # Check prediction distribution
        disease_predictions = sum(1 for p in recent_predictions if p.get("prediction") == 1)
        disease_rate = disease_predictions / len(recent_predictions)

        # Alert if prediction distribution is too skewed
        if disease_rate > 0.9 or disease_rate < 0.1:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "prediction_distribution",
                "severity": "low",
                "disease_rate": disease_rate,
                "num_predictions": len(recent_predictions),
                "message": f"Unusual prediction distribution: {disease_rate:.2%} disease predictions"
            }
            alerts.append(alert)
            self._trigger_alert(alert)

        if alerts:
            self._save_alerts(alerts)

        return alerts

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert through all handlers."""
        logger.warning(f"ALERT: {alert['message']}")

        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        # Log to file
        self._log_alert(alert)

    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to file."""
        alert_file = self.metadata_dir / "alerts.log"

        with open(alert_file, "a") as f:
            f.write(json.dumps(alert) + "\n")

    def _save_alerts(self, alerts: List[Dict[str, Any]]):
        """Save alerts to JSON file."""
        alerts_file = self.metadata_dir / "alerts_history.json"

        # Load existing alerts
        all_alerts = []
        if alerts_file.exists():
            try:
                with open(alerts_file, "r") as f:
                    all_alerts = json.load(f)
            except Exception as e:
                logger.error(f"Error loading alerts history: {e}")

        # Add new alerts
        all_alerts.extend(alerts)

        # Keep last 1000 alerts
        all_alerts = all_alerts[-1000:]

        # Save
        with open(alerts_file, "w") as f:
            json.dump(all_alerts, f, indent=2)

    def _load_baseline_metrics(self) -> Optional[Dict[str, float]]:
        """Load baseline metrics."""
        baseline_file = self.metadata_dir / "baseline_performance.json"

        if not baseline_file.exists():
            return None

        try:
            with open(baseline_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading baseline metrics: {e}")
            return None

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent alerts.

        Args:
            hours: Number of hours to look back

        Returns:
            Alert summary
        """
        alerts_file = self.metadata_dir / "alerts_history.json"

        if not alerts_file.exists():
            return {"total_alerts": 0, "alerts_by_type": {}, "alerts_by_severity": {}}

        try:
            with open(alerts_file, "r") as f:
                all_alerts = json.load(f)

            # Filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [
                alert for alert in all_alerts
                if datetime.fromisoformat(alert.get("timestamp", "")) > cutoff_time
            ]

            # Count by type
            alerts_by_type = {}
            for alert in recent_alerts:
                alert_type = alert.get("type", "unknown")
                alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1

            # Count by severity
            alerts_by_severity = {}
            for alert in recent_alerts:
                severity = alert.get("severity", "unknown")
                alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1

            return {
                "total_alerts": len(recent_alerts),
                "alerts_by_type": alerts_by_type,
                "alerts_by_severity": alerts_by_severity,
                "recent_alerts": recent_alerts[:10]  # Last 10
            }

        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {"total_alerts": 0, "alerts_by_type": {}, "alerts_by_severity": {}}


def email_alert_handler(
    alert: Dict[str, Any],
    smtp_config: Dict[str, str]
):
    """
    Send alert via email.

    Args:
        alert: Alert dictionary
        smtp_config: SMTP configuration (host, port, username, password, from, to)
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_config["from"]
        msg["To"] = smtp_config["to"]
        msg["Subject"] = f"ML Model Alert: {alert['type']} - {alert['severity']}"

        body = f"""
Alert Details:
--------------
Type: {alert['type']}
Severity: {alert['severity']}
Time: {alert['timestamp']}

Message: {alert['message']}

Full Details:
{json.dumps(alert, indent=2)}
        """

        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"]) as server:
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            server.send_message(msg)

        logger.info(f"Alert email sent to {smtp_config['to']}")

    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")


def slack_alert_handler(alert: Dict[str, Any], webhook_url: str):
    """
    Send alert to Slack.

    Args:
        alert: Alert dictionary
        webhook_url: Slack webhook URL
    """
    try:
        import requests

        severity_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🚨"
        }

        message = {
            "text": f"{severity_emoji.get(alert['severity'], '❗')} ML Model Alert",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{alert['message']}*"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Type:*\n{alert['type']}"},
                        {"type": "mrkdwn", "text": f"*Severity:*\n{alert['severity']}"},
                    ]
                }
            ]
        }

        response = requests.post(webhook_url, json=message)
        response.raise_for_status()

        logger.info("Alert sent to Slack")

    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


if __name__ == "__main__":
    # Example usage
    monitor = ModelMonitor()

    # Example: Check performance
    current_metrics = {
        "accuracy": 0.80,
        "precision": 0.78,
        "recall": 0.82,
        "f1": 0.80
    }

    baseline_metrics = {
        "accuracy": 0.87,
        "precision": 0.85,
        "recall": 0.88,
        "f1": 0.86
    }

    alerts = monitor.check_performance(current_metrics, baseline_metrics)
    print(f"\nGenerated {len(alerts)} alerts")

    for alert in alerts:
        print(f"- {alert['message']}")

    # Get alert summary
    summary = monitor.get_alert_summary(hours=24)
    print(f"\nAlert Summary (24 hours):")
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"By Type: {summary['alerts_by_type']}")
    print(f"By Severity: {summary['alerts_by_severity']}")
