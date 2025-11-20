"""Data loading module for heart disease datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from ..config import get_config
from ..config.schemas import DataConfig
from ..core.interfaces import IDataLoader
from ..utils.exceptions import DataLoadError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader(IDataLoader):
    """Load and combine heart disease datasets from multiple sources.

    This implementation uses configuration to determine dataset locations,
    supported formats, and default datasets to load.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """Initialize DataLoader.

        Args:
            config: Data configuration. If None, loads from global config.
        """
        if config is None:
            app_config = get_config()
            config = app_config.data

        self.config = config
        self.raw_data_path = Path(config.raw_data_dir)
        self.processed_data_path = Path(config.processed_data_dir)
        self.interim_data_path = Path(config.interim_data_dir)

        logger.debug(f"DataLoader initialized with raw_data_path={self.raw_data_path}")

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset by name.

        Args:
            dataset_name: Name of the dataset (e.g., 'cleveland', 'hungarian')

        Returns:
            DataFrame containing the dataset

        Raises:
            DataLoadError: If dataset cannot be loaded
        """
        # Construct file path - try dataset_name.csv
        file_path = self.raw_data_path / f"{dataset_name}.csv"

        if not file_path.exists():
            # Try without extension in case it's specified
            file_path = self.raw_data_path / dataset_name
            if not file_path.exists():
                raise DataLoadError(
                    f"Dataset file not found: {file_path}. "
                    f"Searched in {self.raw_data_path}"
                )

        logger.info(f"Loading dataset: {dataset_name} from {file_path}")

        try:
            # Handle different file formats
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix == ".pkl":
                df = pd.read_pickle(file_path)
            else:
                # Default to CSV if no extension
                df = pd.read_csv(file_path)

            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns from {dataset_name}")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load dataset '{dataset_name}': {str(e)}")

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames

        Example:
            >>> loader = DataLoader()
            >>> datasets = loader.load_all_datasets()
            >>> print(datasets.keys())
            dict_keys(['cleveland', 'hungarian', 'switzerland', 'va'])
        """
        dataset_names = self.config.datasets
        data_dict = {}

        logger.info(f"Loading {len(dataset_names)} datasets: {dataset_names}")

        for dataset_name in dataset_names:
            try:
                data_dict[dataset_name] = self.load_dataset(dataset_name)
            except DataLoadError as e:
                logger.warning(f"Could not load {dataset_name}: {str(e)}")

        logger.info(f"Successfully loaded {len(data_dict)}/{len(dataset_names)} datasets")
        return data_dict

    def combine_datasets(
        self,
        dataset_names: List[str],
        add_source_column: bool = True
    ) -> pd.DataFrame:
        """Combine multiple datasets into one.

        Args:
            dataset_names: List of dataset names to combine
            add_source_column: If True, adds a 'source' column indicating origin

        Returns:
            Combined DataFrame

        Raises:
            DataLoadError: If combining fails or no datasets could be loaded

        Example:
            >>> loader = DataLoader()
            >>> combined = loader.combine_datasets(['cleveland', 'hungarian'])
            >>> print(len(combined))
            920
        """
        logger.info(f"Combining {len(dataset_names)} datasets: {dataset_names}")

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
            raise DataLoadError(
                f"No datasets could be loaded for combining. "
                f"Tried: {dataset_names}"
            )

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(
            f"Combined {len(dfs)} datasets into {len(combined_df)} total records "
            f"with {len(combined_df.columns)} columns"
        )

        return combined_df

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing dataset information

        Example:
            >>> loader = DataLoader()
            >>> info = loader.get_dataset_info('cleveland')
            >>> print(info['rows'], info['columns'])
            303 14
        """
        df = self.load_dataset(dataset_name)

        info = {
            "name": dataset_name,
            "shape": df.shape,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

        return info

    def get_all_datasets_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all configured datasets.

        Returns:
            Dictionary mapping dataset names to their info
        """
        dataset_names = self.config.datasets
        info_dict = {}

        for dataset_name in dataset_names:
            try:
                info_dict[dataset_name] = self.get_dataset_info(dataset_name)
            except DataLoadError as e:
                logger.warning(f"Could not get info for {dataset_name}: {str(e)}")

        return info_dict

    def save_processed_data(
        self,
        df: pd.DataFrame,
        filename: str,
        format: str = "csv",
        destination: str = "processed"
    ) -> Path:
        """Save processed data to the specified directory.

        Args:
            df: DataFrame to save
            filename: Name of the file (without extension)
            format: File format ('csv', 'parquet', 'pickle')
            destination: Destination directory ('processed' or 'interim')

        Returns:
            Path to saved file

        Raises:
            DataLoadError: If saving fails
        """
        # Select destination path
        if destination == "processed":
            dest_path = self.processed_data_path
        elif destination == "interim":
            dest_path = self.interim_data_path
        else:
            raise DataLoadError(f"Invalid destination: {destination}")

        dest_path.mkdir(parents=True, exist_ok=True)

        try:
            if format == "csv":
                file_path = dest_path / f"{filename}.csv"
                df.to_csv(file_path, index=False)
            elif format == "parquet":
                file_path = dest_path / f"{filename}.parquet"
                df.to_parquet(file_path, index=False)
            elif format == "pickle":
                file_path = dest_path / f"{filename}.pkl"
                df.to_pickle(file_path)
            else:
                raise DataLoadError(f"Unsupported format: {format}")

            logger.info(f"Saved processed data to {file_path} ({format} format)")
            return file_path

        except Exception as e:
            raise DataLoadError(f"Failed to save processed data: {str(e)}")

    def load_processed_data(self, filename: str, source: str = "processed") -> pd.DataFrame:
        """Load processed data from the specified directory.

        Args:
            filename: Name of the file (with extension)
            source: Source directory ('processed' or 'interim')

        Returns:
            DataFrame containing processed data

        Raises:
            DataLoadError: If loading fails
        """
        # Select source path
        if source == "processed":
            source_path = self.processed_data_path
        elif source == "interim":
            source_path = self.interim_data_path
        else:
            raise DataLoadError(f"Invalid source: {source}")

        file_path = source_path / filename

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

    # Convenience methods for specific datasets (config-driven)

    def load_cleveland(self) -> pd.DataFrame:
        """Load Cleveland dataset."""
        return self.load_dataset("cleveland")

    def load_hungarian(self) -> pd.DataFrame:
        """Load Hungarian dataset."""
        return self.load_dataset("hungarian")

    def load_switzerland(self) -> pd.DataFrame:
        """Load Switzerland dataset."""
        return self.load_dataset("switzerland")

    def load_va(self) -> pd.DataFrame:
        """Load VA (Long Beach) dataset."""
        return self.load_dataset("va")

    def load_all_uci_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all 4 UCI heart disease datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        # Get UCI datasets from config, fallback to defaults
        uci_datasets = ["cleveland", "hungarian", "switzerland", "va"]
        data_dict = {}

        for dataset in uci_datasets:
            try:
                data_dict[dataset] = self.load_dataset(dataset)
            except DataLoadError as e:
                logger.warning(f"Could not load {dataset}: {str(e)}")

        logger.info(f"Loaded {len(data_dict)} UCI datasets")
        return data_dict

    def load_and_combine_uci(self) -> pd.DataFrame:
        """Load and combine all 4 UCI datasets.

        Returns:
            Combined DataFrame with all UCI data
        """
        dataset_names = ["cleveland", "hungarian", "switzerland", "va"]
        return self.combine_datasets(dataset_names, add_source_column=True)
