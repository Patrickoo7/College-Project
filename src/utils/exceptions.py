"""Custom exceptions for the heart disease prediction system."""


class HeartDiseasePredictionError(Exception):
    """Base exception for heart disease prediction system."""
    pass


class ConfigurationError(HeartDiseasePredictionError):
    """Raised when there's an issue with configuration."""
    pass


class DataLoadError(HeartDiseasePredictionError):
    """Raised when data loading fails."""
    pass


class DataValidationError(HeartDiseasePredictionError):
    """Raised when data validation fails."""
    pass


class PreprocessingError(HeartDiseasePredictionError):
    """Raised when data preprocessing fails."""
    pass


class FeatureEngineeringError(HeartDiseasePredictionError):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(HeartDiseasePredictionError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(HeartDiseasePredictionError):
    """Raised when model prediction fails."""
    pass


class ModelLoadError(HeartDiseasePredictionError):
    """Raised when model loading fails."""
    pass


class ModelSaveError(HeartDiseasePredictionError):
    """Raised when model saving fails."""
    pass


class EvaluationError(HeartDiseasePredictionError):
    """Raised when model evaluation fails."""
    pass


class APIError(HeartDiseasePredictionError):
    """Raised when API operations fail."""
    pass


class InvalidInputError(HeartDiseasePredictionError):
    """Raised when input data is invalid."""
    pass
