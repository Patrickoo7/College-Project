"""Database storage for predictions and model serving history."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import sqlite3
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionStore:
    """
    Store and retrieve predictions with metadata.

    Features:
    - SQLite database for predictions
    - Batch insertion support
    - Query interface for analysis
    - Automatic schema management
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize prediction store.

        Args:
            db_path: Path to SQLite database (None = default)
        """
        self.config = Config()

        if db_path is None:
            db_dir = Path(self.config.get("paths.data.processed", "data/processed"))
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "predictions.db")

        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    prediction INTEGER NOT NULL,
                    prediction_label TEXT NOT NULL,
                    confidence REAL,
                    probability_no_disease REAL,
                    probability_disease REAL,
                    age INTEGER,
                    sex INTEGER,
                    cp INTEGER,
                    trestbps INTEGER,
                    chol INTEGER,
                    restecg INTEGER,
                    thalach INTEGER,
                    exang INTEGER,
                    oldpeak REAL,
                    actual_label INTEGER,
                    correct BOOLEAN,
                    inference_time_ms REAL,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT
                )
            """)

            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    dataset TEXT,
                    samples_count INTEGER
                )
            """)

            # Drift detection table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    feature_name TEXT NOT NULL,
                    drift_method TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    drift_detected BOOLEAN NOT NULL,
                    reference_start_date DATETIME,
                    reference_end_date DATETIME,
                    current_start_date DATETIME,
                    current_end_date DATETIME
                )
            """)

            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                ON predictions(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_model
                ON predictions(model_name, timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_model
                ON model_performance(model_name, timestamp)
            """)

            conn.commit()

        logger.info(f"Initialized prediction store at {self.db_path}")

    def store_prediction(
        self,
        prediction: int,
        prediction_label: str,
        model_name: str,
        input_data: Dict[str, Any],
        probability: Optional[Dict[str, float]] = None,
        confidence: Optional[float] = None,
        model_version: Optional[str] = None,
        actual_label: Optional[int] = None,
        inference_time_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store a prediction.

        Args:
            prediction: Predicted class (0 or 1)
            prediction_label: Prediction label
            model_name: Name of model used
            input_data: Input features
            probability: Prediction probabilities
            confidence: Confidence score
            model_version: Model version
            actual_label: Actual label (if known)
            inference_time_ms: Inference time in milliseconds
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata

        Returns:
            Prediction ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Extract probabilities
            prob_no_disease = None
            prob_disease = None
            if probability:
                prob_no_disease = probability.get("no_disease")
                prob_disease = probability.get("disease")

            # Determine correctness
            correct = None
            if actual_label is not None:
                correct = prediction == actual_label

            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, model_name, model_version, prediction, prediction_label,
                    confidence, probability_no_disease, probability_disease,
                    age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak,
                    actual_label, correct, inference_time_ms,
                    user_id, session_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                model_name,
                model_version,
                prediction,
                prediction_label,
                confidence,
                prob_no_disease,
                prob_disease,
                input_data.get("age"),
                input_data.get("sex"),
                input_data.get("cp"),
                input_data.get("trestbps"),
                input_data.get("chol"),
                input_data.get("restecg"),
                input_data.get("thalach"),
                input_data.get("exang"),
                input_data.get("oldpeak"),
                actual_label,
                correct,
                inference_time_ms,
                user_id,
                session_id,
                json.dumps(metadata) if metadata else None
            ))

            conn.commit()
            prediction_id = cursor.lastrowid

        return prediction_id

    def store_batch_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Store multiple predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            List of prediction IDs
        """
        prediction_ids = []

        for pred in predictions:
            pred_id = self.store_prediction(**pred)
            prediction_ids.append(pred_id)

        logger.info(f"Stored {len(prediction_ids)} predictions")

        return prediction_ids

    def get_recent_predictions(
        self,
        limit: int = 100,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get recent predictions.

        Args:
            limit: Maximum number of predictions
            model_name: Filter by model name

        Returns:
            DataFrame of predictions
        """
        # Input validation
        if limit <= 0 or limit > 10000:
            raise ValueError("Limit must be between 1 and 10000")

        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM predictions"
            params = []

            if model_name:
                # Use parameterized query to prevent SQL injection
                query += " WHERE model_name = ?"
                params.append(model_name)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)

        return df

    def get_model_accuracy(
        self,
        model_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[float]:
        """
        Calculate model accuracy from stored predictions.

        Args:
            model_name: Name of model
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Accuracy score or None if insufficient data
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT AVG(CAST(correct AS FLOAT)) as accuracy
                FROM predictions
                WHERE model_name = ? AND actual_label IS NOT NULL
            """

            params = [model_name]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            cursor = conn.cursor()
            result = cursor.execute(query, params).fetchone()

            if result and result[0] is not None:
                return result[0]

        return None

    def get_prediction_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get prediction statistics.

        Args:
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Dictionary of statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as disease_predictions,
                    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as no_disease_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(inference_time_ms) as avg_inference_time,
                    COUNT(DISTINCT model_name) as models_used,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(CASE WHEN actual_label IS NOT NULL THEN 1 END) as labeled_predictions,
                    AVG(CASE WHEN actual_label IS NOT NULL THEN CAST(correct AS FLOAT) END) as accuracy
                FROM predictions
                WHERE 1=1
            """

            params = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            cursor = conn.cursor()
            result = cursor.execute(query, params).fetchone()

            stats = {
                "total_predictions": result[0] or 0,
                "disease_predictions": result[1] or 0,
                "no_disease_predictions": result[2] or 0,
                "avg_confidence": result[3],
                "avg_inference_time_ms": result[4],
                "models_used": result[5] or 0,
                "unique_users": result[6] or 0,
                "labeled_predictions": result[7] or 0,
                "accuracy": result[8]
            }

        return stats

    def store_model_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        model_version: Optional[str] = None,
        dataset: Optional[str] = None,
        samples_count: Optional[int] = None
    ):
        """
        Store model performance metrics.

        Args:
            model_name: Name of model
            metrics: Dictionary of metric names and values
            model_version: Model version
            dataset: Dataset name
            samples_count: Number of samples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO model_performance (
                        timestamp, model_name, model_version, metric_name, metric_value,
                        dataset, samples_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    model_name,
                    model_version,
                    metric_name,
                    metric_value,
                    dataset,
                    samples_count
                ))

            conn.commit()

        logger.info(f"Stored performance metrics for {model_name}")

    def store_drift_event(
        self,
        feature_name: str,
        drift_method: str,
        drift_score: float,
        drift_detected: bool,
        reference_dates: Optional[tuple] = None,
        current_dates: Optional[tuple] = None
    ):
        """
        Store drift detection event.

        Args:
            feature_name: Name of feature
            drift_method: Detection method used
            drift_score: Drift score
            drift_detected: Whether drift was detected
            reference_dates: Tuple of (start, end) for reference period
            current_dates: Tuple of (start, end) for current period
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            ref_start = ref_end = cur_start = cur_end = None
            if reference_dates:
                ref_start, ref_end = reference_dates
            if current_dates:
                cur_start, cur_end = current_dates

            cursor.execute("""
                INSERT INTO drift_events (
                    timestamp, feature_name, drift_method, drift_score, drift_detected,
                    reference_start_date, reference_end_date,
                    current_start_date, current_end_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                feature_name,
                drift_method,
                drift_score,
                drift_detected,
                ref_start,
                ref_end,
                cur_start,
                cur_end
            ))

            conn.commit()

    def export_to_csv(
        self,
        output_path: str,
        table: str = "predictions",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Export database table to CSV.

        Args:
            output_path: Path to output CSV file
            table: Table name to export
            start_date: Start date filter
            end_date: End date filter
        """
        # Whitelist allowed table names to prevent SQL injection
        ALLOWED_TABLES = {'predictions', 'model_performance', 'drift_events'}
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table}. Allowed tables: {ALLOWED_TABLES}")

        with sqlite3.connect(self.db_path) as conn:
            # Safe to use f-string since table is whitelisted
            query = f"SELECT * FROM {table}"
            params = []

            if start_date or end_date:
                query += " WHERE 1=1"

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} rows to {output_path}")


if __name__ == "__main__":
    # Example usage
    store = PredictionStore()

    # Store a prediction
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

    print(f"Stored prediction with ID: {pred_id}")

    # Get statistics
    stats = store.get_prediction_stats()
    print(f"\nPrediction Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
