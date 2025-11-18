"""Feature store for managing and serving features."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Feature store for managing computed features.

    Features:
    - Feature registration and metadata
    - Feature caching and retrieval
    - Feature versioning
    - Feature lineage tracking
    """

    def __init__(self):
        """Initialize feature store."""
        self.config = Config()

        # Paths
        self.features_dir = Path(self.config.get("paths.data.processed", "data/processed")) / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.features_dir / "feature_registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load feature registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "features": {},
                "feature_groups": {},
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            self._save_registry()

    def _save_registry(self):
        """Save feature registry."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_feature(
        self,
        name: str,
        dtype: str,
        description: str,
        feature_group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Register a feature.

        Args:
            name: Feature name
            dtype: Data type
            description: Feature description
            feature_group: Feature group name
            tags: Feature tags
            metadata: Additional metadata
        """
        feature_info = {
            "name": name,
            "dtype": dtype,
            "description": description,
            "feature_group": feature_group,
            "tags": tags or [],
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "version": 1
        }

        if name in self.registry["features"]:
            # Increment version
            feature_info["version"] = self.registry["features"][name]["version"] + 1
            logger.info(f"Updated feature {name} to version {feature_info['version']}")
        else:
            logger.info(f"Registered new feature: {name}")

        self.registry["features"][name] = feature_info
        self._save_registry()

    def register_feature_group(
        self,
        name: str,
        features: List[str],
        description: str,
        metadata: Optional[Dict] = None
    ):
        """
        Register a feature group.

        Args:
            name: Feature group name
            features: List of feature names
            description: Group description
            metadata: Additional metadata
        """
        group_info = {
            "name": name,
            "features": features,
            "description": description,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat()
        }

        self.registry["feature_groups"][name] = group_info
        self._save_registry()

        logger.info(f"Registered feature group: {name} with {len(features)} features")

    def store_features(
        self,
        data: pd.DataFrame,
        feature_group: str,
        entity_id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None
    ) -> str:
        """
        Store computed features.

        Args:
            data: DataFrame with features
            feature_group: Feature group name
            entity_id_column: Column name for entity ID
            timestamp_column: Column name for timestamp

        Returns:
            Feature set ID
        """
        # Generate feature set ID
        timestamp = datetime.now().isoformat()
        feature_set_id = self._generate_feature_set_id(feature_group, timestamp)

        # Add metadata columns
        if timestamp_column is None:
            data["_timestamp"] = timestamp

        if entity_id_column is None:
            data["_entity_id"] = range(len(data))

        # Save features
        feature_path = self.features_dir / f"{feature_set_id}.parquet"
        data.to_parquet(feature_path, index=False)

        # Update registry with feature set info
        if "feature_sets" not in self.registry:
            self.registry["feature_sets"] = {}

        self.registry["feature_sets"][feature_set_id] = {
            "feature_group": feature_group,
            "timestamp": timestamp,
            "path": str(feature_path),
            "num_rows": len(data),
            "num_features": len(data.columns),
            "features": list(data.columns)
        }

        self._save_registry()

        logger.info(f"Stored feature set {feature_set_id} with {len(data)} rows")

        return feature_set_id

    def get_features(
        self,
        feature_group: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        entity_ids: Optional[List] = None,
        latest: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve features.

        Args:
            feature_group: Feature group name
            feature_names: Specific feature names to retrieve
            entity_ids: Specific entity IDs
            latest: Whether to get latest feature set

        Returns:
            DataFrame with features or None
        """
        if "feature_sets" not in self.registry or not self.registry["feature_sets"]:
            logger.warning("No feature sets available")
            return None

        # Find matching feature sets
        matching_sets = []

        for feature_set_id, info in self.registry["feature_sets"].items():
            if feature_group and info["feature_group"] != feature_group:
                continue
            matching_sets.append((feature_set_id, info))

        if not matching_sets:
            logger.warning(f"No feature sets found for group: {feature_group}")
            return None

        # Sort by timestamp and get latest if requested
        matching_sets.sort(key=lambda x: x[1]["timestamp"], reverse=True)

        if latest:
            feature_set_id, info = matching_sets[0]
        else:
            # Get all and combine
            dfs = []
            for feature_set_id, info in matching_sets:
                df = pd.read_parquet(info["path"])
                dfs.append(df)
            data = pd.concat(dfs, ignore_index=True)
            return self._filter_features(data, feature_names, entity_ids)

        # Load features
        data = pd.read_parquet(info["path"])

        # Filter
        return self._filter_features(data, feature_names, entity_ids)

    def _filter_features(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]],
        entity_ids: Optional[List]
    ) -> pd.DataFrame:
        """Filter features by names and entity IDs."""
        # Filter by feature names
        if feature_names:
            # Keep metadata columns
            metadata_cols = [col for col in data.columns if col.startswith("_")]
            selected_cols = metadata_cols + [col for col in feature_names if col in data.columns]
            data = data[selected_cols]

        # Filter by entity IDs
        if entity_ids and "_entity_id" in data.columns:
            data = data[data["_entity_id"].isin(entity_ids)]

        return data

    def _generate_feature_set_id(self, feature_group: str, timestamp: str) -> str:
        """Generate unique feature set ID."""
        content = f"{feature_group}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        return f"{feature_group}_{hash_obj.hexdigest()[:8]}"

    def get_feature_info(self, feature_name: str) -> Optional[Dict]:
        """Get feature metadata."""
        return self.registry["features"].get(feature_name)

    def get_feature_group_info(self, group_name: str) -> Optional[Dict]:
        """Get feature group metadata."""
        return self.registry["feature_groups"].get(group_name)

    def list_features(
        self,
        feature_group: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        List available features.

        Args:
            feature_group: Filter by feature group
            tags: Filter by tags

        Returns:
            List of feature names
        """
        features = []

        for name, info in self.registry["features"].items():
            # Filter by feature group
            if feature_group and info.get("feature_group") != feature_group:
                continue

            # Filter by tags
            if tags:
                feature_tags = set(info.get("tags", []))
                if not feature_tags.intersection(tags):
                    continue

            features.append(name)

        return features

    def list_feature_groups(self) -> List[str]:
        """List available feature groups."""
        return list(self.registry["feature_groups"].keys())

    def compute_feature_statistics(
        self,
        feature_group: str
    ) -> Optional[Dict[str, Dict]]:
        """
        Compute statistics for features in a group.

        Args:
            feature_group: Feature group name

        Returns:
            Dictionary of feature statistics
        """
        data = self.get_features(feature_group=feature_group)

        if data is None:
            return None

        stats = {}

        # Compute statistics for each feature
        for col in data.columns:
            if col.startswith("_"):
                continue  # Skip metadata columns

            col_stats = {}

            if pd.api.types.is_numeric_dtype(data[col]):
                col_stats = {
                    "type": "numeric",
                    "count": int(data[col].count()),
                    "missing": int(data[col].isna().sum()),
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "median": float(data[col].median()),
                    "q25": float(data[col].quantile(0.25)),
                    "q75": float(data[col].quantile(0.75))
                }
            else:
                value_counts = data[col].value_counts()
                col_stats = {
                    "type": "categorical",
                    "count": int(data[col].count()),
                    "missing": int(data[col].isna().sum()),
                    "unique": int(data[col].nunique()),
                    "top": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                }

            stats[col] = col_stats

        return stats

    def delete_feature_set(self, feature_set_id: str):
        """Delete a feature set."""
        if "feature_sets" not in self.registry:
            return

        if feature_set_id in self.registry["feature_sets"]:
            info = self.registry["feature_sets"][feature_set_id]

            # Delete file
            feature_path = Path(info["path"])
            if feature_path.exists():
                feature_path.unlink()

            # Remove from registry
            del self.registry["feature_sets"][feature_set_id]
            self._save_registry()

            logger.info(f"Deleted feature set: {feature_set_id}")


# Initialize feature registry with common heart disease features
def initialize_heart_disease_features():
    """Initialize feature registry with heart disease features."""
    store = FeatureStore()

    # Register raw features
    raw_features = [
        ("age", "int", "Age in years"),
        ("sex", "int", "Sex (0: Female, 1: Male)"),
        ("cp", "int", "Chest pain type (0-3)"),
        ("trestbps", "int", "Resting blood pressure (mm Hg)"),
        ("chol", "int", "Serum cholesterol (mg/dl)"),
        ("restecg", "int", "Resting ECG results (0-2)"),
        ("thalach", "int", "Maximum heart rate achieved"),
        ("exang", "int", "Exercise induced angina (0: No, 1: Yes)"),
        ("oldpeak", "float", "ST depression induced by exercise")
    ]

    for name, dtype, desc in raw_features:
        store.register_feature(name, dtype, desc, feature_group="raw", tags=["raw"])

    # Register engineered features
    engineered_features = [
        ("age_group", "str", "Age group category"),
        ("bmi_category", "str", "BMI category"),
        ("heart_rate_category", "str", "Heart rate category"),
        ("cholesterol_ratio", "float", "Cholesterol to age ratio"),
        ("exercise_capacity", "float", "Exercise capacity score")
    ]

    for name, dtype, desc in engineered_features:
        store.register_feature(name, dtype, desc, feature_group="engineered", tags=["engineered"])

    # Register feature groups
    store.register_feature_group(
        "raw_features",
        [f[0] for f in raw_features],
        "Raw input features from heart disease dataset"
    )

    store.register_feature_group(
        "engineered_features",
        [f[0] for f in engineered_features],
        "Engineered features for improved predictions"
    )

    logger.info("Initialized heart disease feature registry")


if __name__ == "__main__":
    # Initialize feature store
    initialize_heart_disease_features()

    # Example usage
    store = FeatureStore()

    print("\nAvailable Features:")
    for feature in store.list_features():
        info = store.get_feature_info(feature)
        print(f"  {feature}: {info['description']}")

    print("\nFeature Groups:")
    for group in store.list_feature_groups():
        info = store.get_feature_group_info(group)
        print(f"  {group}: {len(info['features'])} features")
