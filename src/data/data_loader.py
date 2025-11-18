"""Data loading module for heart disease datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..utils.config import get_config
from ..utils.exceptions import DataLoadError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and combine heart disease datasets from multiple sources."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config = get_config(config_path)
        self.data_path = self.config.get_path("paths.data.raw")

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a single dataset by name.

        Args:
            dataset_name: Name of the dataset (from config)

        Returns:
            DataFrame containing the dataset

        Raises:
            DataLoadError: If dataset cannot be loaded
        """
        # Get filename from config
        filename = self.config.get(f"datasets.{dataset_name}")
        if filename is None:
            raise DataLoadError(f"Dataset '{dataset_name}' not found in configuration")

        file_path = self.data_path / filename

        if not file_path.exists():
            raise DataLoadError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading dataset: {dataset_name} from {file_path}")

        try:
            # Handle different file formats
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                raise DataLoadError(f"Unsupported file format: {file_path.suffix}")

            logger.info(f"Loaded {len(df)} records from {dataset_name}")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load dataset '{dataset_name}': {str(e)}")

    def load_cleveland(self) -> pd.DataFrame:
        """
        Load Cleveland dataset.

        Returns:
            DataFrame with Cleveland data
        """
        return self.load_dataset("cleveland")

    def load_hungarian(self) -> pd.DataFrame:
        """
        Load Hungarian dataset.

        Returns:
            DataFrame with Hungarian data
        """
        return self.load_dataset("hungarian")

    def load_switzerland(self) -> pd.DataFrame:
        """
        Load Switzerland dataset.

        Returns:
            DataFrame with Switzerland data
        """
        return self.load_dataset("switzerland")

    def load_va(self) -> pd.DataFrame:
        """
        Load VA (Long Beach) dataset.

        Returns:
            DataFrame with VA data
        """
        return self.load_dataset("va")

    def load_heart(self) -> pd.DataFrame:
        """
        Load heart dataset (combined/processed).

        Returns:
            DataFrame with heart data
        """
        return self.load_dataset("heart")

    def load_all_uci_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 4 UCI heart disease datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = ["cleveland", "hungarian", "switzerland", "va"]
        data_dict = {}

        for dataset in datasets:
            try:
                data_dict[dataset] = self.load_dataset(dataset)
            except DataLoadError as e:
                logger.warning(f"Could not load {dataset}: {str(e)}")

        logger.info(f"Loaded {len(data_dict)} UCI datasets")
        return data_dict

    def combine_datasets(
        self,
        dataset_names: List[str],
        add_source_column: bool = True
    ) -> pd.DataFrame:
        """
        Combine multiple datasets into one.

        Args:
            dataset_names: List of dataset names to combine
            add_source_column: If True, adds a 'source' column indicating origin

        Returns:
            Combined DataFrame

        Raises:
            DataLoadError: If combining fails
        """
        logger.info(f"Combining datasets: {dataset_names}")

        dfs = []
        for name in dataset_names:
            try:
                df = self.load_dataset(name)
                if add_source_column:
                    df["source"] = name
                dfs.append(df)
            except DataLoadError as e:
                logger.warning(f"Skipping {name}: {str(e)}")

        if not dfs:
            raise DataLoadError("No datasets could be loaded for combining")

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} datasets into {len(combined_df)} total records")

        return combined_df

    def load_and_combine_uci(self) -> pd.DataFrame:
        """
        Load and combine all 4 UCI datasets.

        Returns:
            Combined DataFrame with all UCI data
        """
        dataset_names = ["cleveland", "hungarian", "switzerland", "va"]
        return self.combine_datasets(dataset_names, add_source_column=True)

    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """
        Get information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing dataset information
        """
        df = self.load_dataset(dataset_name)

        info = {
            "name": dataset_name,
            "shape": df.shape,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }

        return info

    def get_all_datasets_info(self) -> Dict[str, Dict]:
        """
        Get information about all configured datasets.

        Returns:
            Dictionary mapping dataset names to their info
        """
        datasets = self.config.get("datasets", {})
        info_dict = {}

        for dataset_name in datasets.keys():
            try:
                info_dict[dataset_name] = self.get_dataset_info(dataset_name)
            except DataLoadError as e:
                logger.warning(f"Could not get info for {dataset_name}: {str(e)}")

        return info_dict

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = "csv"
    ) -> Path:
        """
        Save processed data to the processed data directory.

        Args:
            df: DataFrame to save
            filename: Name of the file (without extension)
            format: File format ('csv', 'parquet', 'pickle')

        Returns:
            Path to saved file

        Raises:
            DataLoadError: If saving fails
        """
        processed_path = self.config.get_path("paths.data.processed")
        processed_path.mkdir(parents=True, exist_ok=True)

        try:
            if format == "csv":
                file_path = processed_path / f"{filename}.csv"
                df.to_csv(file_path, index=False)
            elif format == "parquet":
                file_path = processed_path / f"{filename}.parquet"
                df.to_parquet(file_path, index=False)
            elif format == "pickle":
                file_path = processed_path / f"{filename}.pkl"
                df.to_pickle(file_path)
            else:
                raise DataLoadError(f"Unsupported format: {format}")

            logger.info(f"Saved processed data to {file_path}")
            return file_path

        except Exception as e:
            raise DataLoadError(f"Failed to save processed data: {str(e)}")

    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from the processed data directory.

        Args:
            filename: Name of the file (with extension)

        Returns:
            DataFrame containing processed data

        Raises:
            DataLoadError: If loading fails
        """
        processed_path = self.config.get_path("paths.data.processed")
        file_path = processed_path / filename

        if not file_path.exists():
            raise DataLoadError(f"Processed data file not found: {file_path}")

        try:
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix == ".pkl":
                df = pd.read_pickle(file_path)
            else:
                raise DataLoadError(f"Unsupported format: {file_path.suffix}")

            logger.info(f"Loaded processed data from {file_path}")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load processed data: {str(e)}")
