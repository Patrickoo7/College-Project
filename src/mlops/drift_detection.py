"""Data drift detection for monitoring feature and target distributions."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Detect statistical drift in features and target variable.

    Implements multiple drift detection methods:
    - Kolmogorov-Smirnov test
    - Chi-square test
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize drift detector.

        Args:
            significance_level: Significance level for statistical tests
        """
        self.config = Config()
        self.significance_level = significance_level
        self.metadata_dir = Path(self.config.get("paths.models.metadata", "models/metadata"))
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.

        Args:
            reference_data: Reference/baseline data
            current_data: Current/production data
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names

        Returns:
            Dictionary containing drift detection results
        """
        logger.info("Starting drift detection...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "reference_samples": len(reference_data),
            "current_samples": len(current_data),
            "numerical_drift": {},
            "categorical_drift": {},
            "overall_drift": False,
            "drifted_features": []
        }

        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = reference_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        if categorical_features is None:
            categorical_features = reference_data.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()

        # Detect drift in numerical features
        for feature in numerical_features:
            if feature in current_data.columns:
                drift_result = self._detect_numerical_drift(
                    reference_data[feature],
                    current_data[feature],
                    feature
                )
                results["numerical_drift"][feature] = drift_result

                if drift_result["drift_detected"]:
                    results["drifted_features"].append(feature)

        # Detect drift in categorical features
        for feature in categorical_features:
            if feature in current_data.columns:
                drift_result = self._detect_categorical_drift(
                    reference_data[feature],
                    current_data[feature],
                    feature
                )
                results["categorical_drift"][feature] = drift_result

                if drift_result["drift_detected"]:
                    results["drifted_features"].append(feature)

        # Determine overall drift
        results["overall_drift"] = len(results["drifted_features"]) > 0

        # Calculate drift score with division by zero protection
        total_features = len(numerical_features) + len(categorical_features)
        if total_features == 0:
            results["drift_score"] = 0.0
            logger.warning("No features available for drift detection")
        else:
            results["drift_score"] = len(results["drifted_features"]) / total_features

        # Save drift report
        self._save_drift_report(results)

        logger.info(
            f"Drift detection completed. "
            f"Drifted features: {len(results['drifted_features'])} / "
            f"{len(numerical_features) + len(categorical_features)}"
        )

        return results

    def _detect_numerical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Detect drift in numerical feature using multiple methods.

        Args:
            reference: Reference feature data
            current: Current feature data
            feature_name: Name of the feature

        Returns:
            Dictionary with drift detection results
        """
        result = {
            "feature": feature_name,
            "type": "numerical",
            "drift_detected": False,
            "methods": {}
        }

        # Remove NaN values
        ref_clean = reference.dropna()
        cur_clean = current.dropna()

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            result["error"] = "Insufficient non-null values"
            return result

        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_clean, cur_clean)
        result["methods"]["kolmogorov_smirnov"] = {
            "statistic": float(ks_statistic),
            "p_value": float(ks_pvalue),
            "drift": ks_pvalue < self.significance_level
        }

        # Population Stability Index (PSI)
        psi_value = self._calculate_psi(ref_clean, cur_clean)
        result["methods"]["psi"] = {
            "value": float(psi_value),
            "drift": psi_value > 0.2  # PSI > 0.2 indicates significant drift
        }

        # Statistical moments comparison
        result["statistics"] = {
            "reference": {
                "mean": float(ref_clean.mean()),
                "std": float(ref_clean.std()),
                "median": float(ref_clean.median()),
                "min": float(ref_clean.min()),
                "max": float(ref_clean.max())
            },
            "current": {
                "mean": float(cur_clean.mean()),
                "std": float(cur_clean.std()),
                "median": float(cur_clean.median()),
                "min": float(cur_clean.min()),
                "max": float(cur_clean.max())
            }
        }

        # Overall drift decision (if any method detects drift)
        result["drift_detected"] = (
            result["methods"]["kolmogorov_smirnov"]["drift"] or
            result["methods"]["psi"]["drift"]
        )

        return result

    def _detect_categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
        feature_name: str
    ) -> Dict[str, Any]:
        """
        Detect drift in categorical feature.

        Args:
            reference: Reference feature data
            current: Current feature data
            feature_name: Name of the feature

        Returns:
            Dictionary with drift detection results
        """
        result = {
            "feature": feature_name,
            "type": "categorical",
            "drift_detected": False,
            "methods": {}
        }

        # Get value counts
        ref_counts = reference.value_counts(normalize=True, dropna=False)
        cur_counts = current.value_counts(normalize=True, dropna=False)

        # Align indices
        all_categories = set(ref_counts.index) | set(cur_counts.index)

        ref_dist = pd.Series([ref_counts.get(cat, 0) for cat in all_categories])
        cur_dist = pd.Series([cur_counts.get(cat, 0) for cat in all_categories])

        # Chi-square test
        try:
            # Create contingency table
            ref_total = len(reference)
            cur_total = len(current)

            observed = pd.concat([
                ref_counts * ref_total,
                cur_counts * cur_total
            ], axis=1).fillna(0)

            # Check if we have enough categories for chi-square test
            if observed.shape[0] < 2:
                logger.warning(f"Insufficient categories for chi-square test on {feature_name}")
                result["methods"]["chi_square"] = {
                    "error": "Insufficient categories (need at least 2)"
                }
            else:
                chi2_statistic, chi2_pvalue = stats.chi2_contingency(observed.T)[:2]

                result["methods"]["chi_square"] = {
                    "statistic": float(chi2_statistic),
                    "p_value": float(chi2_pvalue),
                    "drift": chi2_pvalue < self.significance_level
                }
        except Exception as e:
            logger.warning(f"Chi-square test failed for {feature_name}: {e}")
            result["methods"]["chi_square"] = {"error": str(e)}

        # Population Stability Index (PSI) for categorical
        psi_value = self._calculate_psi_categorical(ref_dist, cur_dist)
        result["methods"]["psi"] = {
            "value": float(psi_value),
            "drift": psi_value > 0.2
        }

        # Category distribution comparison
        result["distributions"] = {
            "reference": ref_counts.to_dict(),
            "current": cur_counts.to_dict()
        }

        # Overall drift decision
        drift_flags = [
            method.get("drift", False)
            for method in result["methods"].values()
            if "drift" in method
        ]
        result["drift_detected"] = any(drift_flags)

        return result

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI) for numerical features.

        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate distributions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalize to percentages
        ref_percents = ref_counts / len(reference)
        cur_percents = cur_counts / len(current)

        # Add small constant to avoid division by zero
        epsilon = 1e-10
        ref_percents = ref_percents + epsilon
        cur_percents = cur_percents + epsilon

        # Calculate PSI
        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))

        return psi

    def _calculate_psi_categorical(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Calculate PSI for categorical features.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            PSI value
        """
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        ref_percents = reference + epsilon
        cur_percents = current + epsilon

        # Calculate PSI
        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))

        return psi

    def _save_drift_report(self, results: Dict[str, any]):
        """Save drift detection report."""
        report_path = self.metadata_dir / "drift_reports.json"

        # Load existing reports
        reports = []
        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    reports = json.load(f)
            except Exception as e:
                logger.error(f"Error loading drift reports: {e}")

        # Add new report
        reports.append(results)

        # Keep only last 100 reports
        reports = reports[-100:]

        # Save updated reports
        with open(report_path, "w") as f:
            json.dump(reports, f, indent=2)

        logger.info(f"Saved drift report to {report_path}")

    def get_drift_summary(self) -> Optional[Dict[str, any]]:
        """Get summary of recent drift reports."""
        report_path = self.metadata_dir / "drift_reports.json"

        if not report_path.exists():
            return None

        try:
            with open(report_path, "r") as f:
                reports = json.load(f)

            if not reports:
                return None

            # Get last 10 reports
            recent_reports = reports[-10:]

            summary = {
                "total_reports": len(reports),
                "recent_drift_detected": sum(
                    1 for r in recent_reports if r.get("overall_drift", False)
                ),
                "frequently_drifted_features": self._get_frequently_drifted_features(reports),
                "latest_report": recent_reports[-1]
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting drift summary: {e}")
            return None

    def _get_frequently_drifted_features(
        self,
        reports: List[Dict],
        top_n: int = 5
    ) -> List[Tuple[str, int]]:
        """Get features that drift most frequently."""
        feature_drift_counts = {}

        for report in reports:
            for feature in report.get("drifted_features", []):
                feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1

        # Sort by count
        sorted_features = sorted(
            feature_drift_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:top_n]


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    datasets = loader.load_all_datasets()
    df = loader.combine_datasets(datasets)

    # Split into reference and current
    split_point = int(len(df) * 0.8)
    reference_data = df[:split_point]
    current_data = df[split_point:]

    # Detect drift
    detector = DriftDetector()
    results = detector.detect_drift(reference_data, current_data)

    print("\nDrift Detection Results:")
    print(f"Overall Drift: {results['overall_drift']}")
    print(f"Drift Score: {results['drift_score']:.2%}")
    print(f"Drifted Features: {results['drifted_features']}")
