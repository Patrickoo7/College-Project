"""Pydantic schemas for type-safe, validated configuration."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class Environment(str, Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ScalingMethod(str, Enum):
    """Supported scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class ImputationStrategy(str, Enum):
    """Supported imputation strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    KNN = "knn"
    ITERATIVE = "iterative"


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

class FeatureRange(BaseModel):
    """Range validation for a single feature."""
    min: float = Field(..., description="Minimum allowed value")
    max: float = Field(..., description="Maximum allowed value")

    @validator('max')
    def max_greater_than_min(cls, v, values):
        if 'min' in values and v <= values['min']:
            raise ValueError('max must be greater than min')
        return v


class ValidationConfig(BaseModel):
    """Configuration for data validation rules."""

    # Feature value ranges
    ranges: Dict[str, FeatureRange] = Field(
        default_factory=lambda: {
            "age": FeatureRange(min=0, max=120),
            "sex": FeatureRange(min=0, max=1),
            "cp": FeatureRange(min=0, max=4),
            "trestbps": FeatureRange(min=50, max=250),
            "chol": FeatureRange(min=100, max=600),
            "fbs": FeatureRange(min=0, max=1),
            "restecg": FeatureRange(min=0, max=2),
            "thalach": FeatureRange(min=50, max=250),
            "exang": FeatureRange(min=0, max=1),
            "oldpeak": FeatureRange(min=0, max=10),
            "slope": FeatureRange(min=0, max=3),
            "ca": FeatureRange(min=0, max=4),
            "thal": FeatureRange(min=0, max=7),
            "target": FeatureRange(min=0, max=4),
        },
        description="Valid ranges for each feature"
    )

    # Validation thresholds
    missing_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Max proportion of missing values allowed per feature"
    )

    min_class_percentage: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Minimum percentage for a class to be considered valid"
    )

    # Schema validation
    required_features: List[str] = Field(
        default_factory=lambda: [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        description="Features required in input data"
    )

    target_column: str = Field(default="target", description="Name of target column")


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

class DomainRulesConfig(BaseModel):
    """Configuration for domain-specific feature engineering rules."""

    age_risk_threshold: int = Field(default=55, ge=0, le=120)
    high_chol_threshold: int = Field(default=240, ge=100, le=600)
    high_bp_threshold: int = Field(default=140, ge=50, le=250)
    max_hr_formula: str = Field(default="220-age", description="Formula for max heart rate")
    low_hr_percentage_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering parameters."""

    # Polynomial features
    polynomial_degree: int = Field(default=2, ge=1, le=3, description="Degree for polynomial features")
    include_polynomial: bool = Field(default=False, description="Whether to generate polynomial features")

    # Feature interactions
    interaction_pairs: List[Tuple[str, str]] = Field(
        default_factory=lambda: [
            ("age", "chol"),
            ("age", "thalach"),
            ("trestbps", "chol"),
            ("exang", "oldpeak"),
            ("cp", "thalach"),
        ],
        description="Pairs of features to create interactions for"
    )

    # Feature binning
    age_bins: List[int] = Field(
        default_factory=lambda: [0, 40, 50, 60, 70, 120],
        description="Bins for age categorization"
    )

    chol_bins: List[int] = Field(
        default_factory=lambda: [0, 200, 240, 280, 600],
        description="Bins for cholesterol categorization"
    )

    bp_bins: List[int] = Field(
        default_factory=lambda: [0, 120, 140, 180, 300],
        description="Bins for blood pressure categorization"
    )

    # Domain-specific rules
    domain_rules: DomainRulesConfig = Field(default_factory=DomainRulesConfig)

    # Feature selection
    k_best_features: int = Field(default=10, ge=1, le=50, description="Number of best features to select")
    feature_selection_method: str = Field(
        default="mutual_info",
        description="Method for feature selection (mutual_info, f_classif, chi2)"
    )


# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""

    # Scaling
    scaling_method: ScalingMethod = Field(default=ScalingMethod.STANDARD)

    # Imputation
    imputation_strategy: ImputationStrategy = Field(default=ImputationStrategy.KNN)
    knn_neighbors: int = Field(default=5, ge=1, le=20, description="Neighbors for KNN imputation")

    # Outlier detection
    zscore_threshold: float = Field(default=3.0, ge=1.0, le=5.0, description="Z-score threshold for outlier detection")
    iqr_multiplier: float = Field(default=1.5, ge=1.0, le=3.0, description="IQR multiplier for outlier detection")
    outlier_detection_method: str = Field(
        default="iqr",
        description="Method for outlier detection (zscore, iqr, isolation_forest)"
    )

    # Random state
    random_state: int = Field(default=42, description="Random seed for reproducibility")


# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION
# ============================================================================

class HPSearchSpace(BaseModel):
    """Hyperparameter search space for a model."""
    param_ranges: Dict[str, Any] = Field(..., description="Parameter ranges for hyperparameter search")


class HyperparameterConfig(BaseModel):
    """Configuration for hyperparameter tuning."""

    # General tuning settings
    default_trials: int = Field(default=100, ge=10, le=500, description="Default number of Optuna trials")
    cv_folds: int = Field(default=5, ge=3, le=10, description="Cross-validation folds")
    startup_trials: int = Field(default=10, ge=5, le=50, description="Optuna startup trials")
    warmup_steps: int = Field(default=5, ge=3, le=20, description="Optuna warmup steps")

    # Model-specific search spaces
    random_forest: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": {"low": 50, "high": 500, "step": 50},
            "max_depth": {"low": 3, "high": 30},
            "min_samples_split": {"low": 2, "high": 20},
            "min_samples_leaf": {"low": 1, "high": 10},
        }
    )

    gradient_boosting: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": {"low": 50, "high": 300, "step": 50},
            "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"low": 3, "high": 15},
            "subsample": {"low": 0.6, "high": 1.0},
        }
    )

    logistic_regression: Dict[str, Any] = Field(
        default_factory=lambda: {
            "C": {"low": 1e-3, "high": 100, "log": True},
            "penalty": {"choices": ["l1", "l2", "elasticnet"]},
            "solver": {"choices": ["liblinear", "saga"]},
            "max_iter": {"value": 1000},
        }
    )

    svm: Dict[str, Any] = Field(
        default_factory=lambda: {
            "C": {"low": 0.1, "high": 100, "log": True},
            "kernel": {"choices": ["linear", "rbf", "poly"]},
            "gamma": {"choices": ["scale", "auto"]},
        }
    )

    knn: Dict[str, Any] = Field(
        default_factory=lambda: {
            "n_neighbors": {"low": 3, "high": 50},
            "weights": {"choices": ["uniform", "distance"]},
            "metric": {"choices": ["euclidean", "manhattan", "minkowski"]},
        }
    )

    decision_tree: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_depth": {"low": 3, "high": 30},
            "min_samples_split": {"low": 2, "high": 20},
            "min_samples_leaf": {"low": 1, "high": 10},
            "criterion": {"choices": ["gini", "entropy"]},
        }
    )


# ============================================================================
# MLOPS CONFIGURATION
# ============================================================================

class DriftDetectionConfig(BaseModel):
    """Configuration for data drift detection."""

    significance_level: float = Field(default=0.05, ge=0.01, le=0.1, description="Significance level for statistical tests")
    psi_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="PSI threshold for drift detection")
    psi_bins: int = Field(default=10, ge=5, le=20, description="Number of bins for PSI calculation")
    epsilon: float = Field(default=1e-10, description="Small value for numerical stability")
    max_reports: int = Field(default=100, ge=10, le=1000, description="Max drift reports to keep in history")


class MonitoringConfig(BaseModel):
    """Configuration for model monitoring and alerting."""

    accuracy_drop_threshold: float = Field(default=0.05, ge=0.01, le=0.2, description="Threshold for accuracy drop alert")
    drift_score_threshold: float = Field(default=0.3, ge=0.1, le=0.5, description="Threshold for drift score alert")
    error_rate_threshold: float = Field(default=0.2, ge=0.05, le=0.5, description="Threshold for error rate alert")
    prediction_latency_ms: int = Field(default=1000, ge=100, le=5000, description="Max prediction latency in milliseconds")

    distribution_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "disease_rate_high": 0.9,
            "disease_rate_low": 0.1,
        },
        description="Thresholds for prediction distribution monitoring"
    )

    max_alerts_history: int = Field(default=1000, ge=100, le=10000, description="Max alerts to keep in history")


class RetrainingConfig(BaseModel):
    """Configuration for automated model retraining."""

    performance_threshold: float = Field(default=0.05, ge=0.01, le=0.2, description="Performance drop threshold for retraining")
    min_samples_required: int = Field(default=100, ge=50, le=1000, description="Min samples required for retraining")
    interval_days: int = Field(default=30, ge=1, le=365, description="Max days before retraining")
    promotion_tolerance: float = Field(default=0.01, ge=0.0, le=0.1, description="Tolerance for model promotion")
    cron_schedule: str = Field(default="0 2 * * 0", description="Cron expression for scheduled retraining")


class MLOpsConfig(BaseModel):
    """Configuration for MLOps components."""

    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)

    # Database
    max_query_limit: int = Field(default=10000, ge=100, le=100000, description="Max rows to return in queries")


# ============================================================================
# API CONFIGURATION
# ============================================================================

class CORSConfig(BaseModel):
    """CORS configuration."""

    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class APIConfig(BaseModel):
    """Configuration for API settings."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API port")

    # API metadata
    title: str = Field(default="Heart Disease Prediction API", description="API title")
    version: str = Field(default="1.0.0", description="API version")
    description: str = Field(default="ML-powered heart disease prediction service", description="API description")
    prefix: str = Field(default="/api/v1", description="API route prefix")

    # CORS
    cors: CORSConfig = Field(default_factory=CORSConfig)

    # File upload limits
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Max file upload size in MB")
    max_batch_rows: int = Field(default=10000, ge=100, le=100000, description="Max rows in batch predictions")

    # Model settings
    default_model: str = Field(default="random_forest", description="Default model for predictions")

    # Security
    enable_auth: bool = Field(default=False, description="Enable JWT authentication")
    secret_key: Optional[str] = Field(default=None, description="JWT secret key")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440, description="Access token expiration time")


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class DataConfig(BaseModel):
    """Configuration for data loading and splitting."""

    # Data directories
    raw_data_dir: str = Field(default="data/raw", description="Directory for raw data")
    processed_data_dir: str = Field(default="data/processed", description="Directory for processed data")
    interim_data_dir: str = Field(default="data/interim", description="Directory for interim data")

    # Dataset sources
    datasets: List[str] = Field(
        default_factory=lambda: ["cleveland", "hungarian", "switzerland", "va"],
        description="List of dataset names to load"
    )

    # Train/test split
    test_size: float = Field(default=0.25, ge=0.1, le=0.5, description="Test set size")
    validation_size: float = Field(default=0.15, ge=0.0, le=0.3, description="Validation set size")
    stratify: bool = Field(default=True, description="Whether to stratify split by target")

    # Class mapping
    binary_classification: bool = Field(default=True, description="Convert multi-class to binary")


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class EnsembleConfig(BaseModel):
    """Configuration for ensemble methods."""

    stacking_cv: int = Field(default=5, ge=3, le=10, description="CV folds for stacking ensemble")
    voting_weights: Optional[List[float]] = Field(default=None, description="Weights for voting ensemble")


class ModelConfig(BaseModel):
    """Configuration for model training and evaluation."""

    # Model selection
    models_to_train: List[str] = Field(
        default_factory=lambda: [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "svm",
            "knn",
            "decision_tree",
        ],
        description="List of models to train"
    )

    # Training settings
    use_gpu: bool = Field(default=False, description="Use GPU for training if available")
    device_id: int = Field(default=0, ge=0, le=7, description="GPU device ID")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")

    # Cross-validation
    cv_folds: int = Field(default=5, ge=3, le=10, description="Cross-validation folds")
    cv_metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc"],
        description="Metrics to compute during cross-validation"
    )

    # Ensemble
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    # Model directories
    model_dir: str = Field(default="models", description="Directory to save models")
    backup_dir: str = Field(default="models/backups", description="Directory for model backups")


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

class VisualizationConfig(BaseModel):
    """Configuration for visualization settings."""

    # Figure settings
    figure_size: Tuple[int, int] = Field(default=(8, 6), description="Default figure size (width, height)")
    dpi: int = Field(default=300, ge=72, le=600, description="Figure DPI for saving")

    # Colors
    cm_colormap: str = Field(default="Blues", description="Colormap for confusion matrix")
    roc_color: str = Field(default="darkorange", description="ROC curve color")
    baseline_color: str = Field(default="navy", description="Baseline color in plots")

    # Class labels
    class_labels: List[str] = Field(
        default_factory=lambda: ["No Disease", "Disease"],
        description="Display labels for classes"
    )

    # Plot settings
    show_plots: bool = Field(default=False, description="Whether to display plots (use False for servers)")
    save_plots: bool = Field(default=True, description="Whether to save plots to files")
    plots_dir: str = Field(default="reports/figures", description="Directory to save plots")


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    log_dir: str = Field(default="logs", description="Directory for log files")
    log_file: str = Field(default="app.log", description="Log file name")
    max_bytes: int = Field(default=10485760, ge=1048576, le=104857600, description="Max log file size in bytes (10MB)")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )


# ============================================================================
# MAIN APPLICATION CONFIGURATION
# ============================================================================

class AppConfig(BaseModel):
    """Main application configuration combining all sub-configurations."""

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Current environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Sub-configurations
    api: APIConfig = Field(default_factory=APIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    hyperparameters: HyperparameterConfig = Field(default_factory=HyperparameterConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # MLflow tracking
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking server URI")
    mlflow_experiment_name: str = Field(default="heart_disease_prediction", description="MLflow experiment name")

    class Config:
        """Pydantic config."""
        use_enum_values = True
        validate_assignment = True
