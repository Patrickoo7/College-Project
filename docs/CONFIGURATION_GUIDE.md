# Configuration Guide

## Overview

The Heart Disease Prediction system is now **100% config-driven** with a modular architecture based on dependency injection. This guide explains how to configure and customize every aspect of the system without changing code.

## Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Configuration Files](#configuration-files)
3. [Environment-Specific Configurations](#environment-specific-configurations)
4. [Environment Variables](#environment-variables)
5. [Configuration Schema](#configuration-schema)
6. [Using Configuration in Code](#using-configuration-in-code)
7. [Dependency Injection](#dependency-injection)
8. [Examples](#examples)

---

## Configuration Architecture

### Design Principles

The configuration system follows these principles:

1. **Type-Safe**: All configurations use Pydantic models for validation
2. **Layered**: Base config + environment overrides + env variables
3. **Modular**: Organized by domain (API, Data, MLOps, etc.)
4. **Validated**: Automatic validation with clear error messages
5. **Hot-Reloadable**: Configuration can be reloaded without restart (dev mode)

### Configuration Loading Order

```
1. Base Configuration (app_config.yaml)
   ↓
2. Environment Override (development_config.yaml, staging_config.yaml, production_config.yaml)
   ↓
3. Environment Variables (APP_* prefix)
   ↓
4. Final Validated Configuration
```

---

## Configuration Files

### Directory Structure

```
configs/
├── app_config.yaml              # Base configuration (all environments)
├── development_config.yaml       # Development overrides
├── staging_config.yaml          # Staging overrides
├── production_config.yaml       # Production overrides
└── [custom_config.yaml]         # Custom configurations
```

### app_config.yaml

The base configuration file containing default values for all settings:

```yaml
environment: "development"
debug: false

api:
  host: "0.0.0.0"
  port: 8000
  title: "Heart Disease Prediction API"
  version: "1.0.0"
  default_model: "random_forest"
  max_file_size_mb: 10
  max_batch_rows: 10000

  cors:
    allowed_origins:
      - "http://localhost:3000"
    allow_credentials: true

data:
  raw_data_dir: "data/raw"
  test_size: 0.25
  datasets:
    - cleveland
    - hungarian

preprocessing:
  scaling_method: "standard"
  imputation_strategy: "knn"
  knn_neighbors: 5

# ... and many more sections
```

---

## Environment-Specific Configurations

### Development (development_config.yaml)

Optimized for fast iteration:

```yaml
environment: "development"
debug: true

api:
  cors:
    allowed_origins: ["*"]  # Allow all origins
  enable_auth: false        # No auth required

hyperparameters:
  default_trials: 20        # Reduced for faster tuning
  cv_folds: 3               # Fewer folds

logging:
  level: "DEBUG"
```

### Staging (staging_config.yaml)

Mirrors production with some relaxed settings:

```yaml
environment: "staging"
debug: false

api:
  cors:
    allowed_origins:
      - "https://staging.example.com"
  enable_auth: true

mlops:
  monitoring:
    accuracy_drop_threshold: 0.05

logging:
  level: "INFO"
```

### Production (production_config.yaml)

Strict settings for production:

```yaml
environment: "production"
debug: false

api:
  cors:
    allowed_origins:
      - "https://example.com"
  enable_auth: true        # Auth required
  max_file_size_mb: 5      # Stricter limits

mlops:
  monitoring:
    accuracy_drop_threshold: 0.03  # More sensitive
    prediction_latency_ms: 500     # Stricter SLA

logging:
  level: "WARNING"         # Less verbose
```

---

## Environment Variables

Environment variables override configuration file values using the `APP_` prefix and double underscores for nesting.

### Format

```bash
APP_<SECTION>__<SUBSECTION>__<KEY>=value
```

### Examples

```bash
# Select environment (development, staging, production)
export ENV=production

# Override API settings
export APP_API__HOST=0.0.0.0
export APP_API__PORT=9000

# Override CORS origins
export APP_API__CORS__ALLOWED_ORIGINS=https://example.com,https://app.example.com

# Enable debug mode
export APP_DEBUG=true

# Set JWT secret (REQUIRED in production if auth enabled)
export APP_API__SECRET_KEY=your-super-secret-key-here

# Override data directories
export APP_DATA__RAW_DATA_DIR=/mnt/data/raw
export APP_DATA__PROCESSED_DATA_DIR=/mnt/data/processed

# Override MLOps thresholds
export APP_MLOPS__MONITORING__ACCURACY_DROP_THRESHOLD=0.03
export APP_MLOPS__DRIFT_DETECTION__PSI_THRESHOLD=0.15

# GPU settings
export APP_MODEL__USE_GPU=true
export APP_MODEL__DEVICE_ID=0

# MLflow configuration
export APP_MLFLOW_TRACKING_URI=https://mlflow.example.com
```

### Using .env Files

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values:
   ```bash
   ENV=development
   APP_API__PORT=8000
   APP_API__SECRET_KEY=dev-secret-key
   ```

3. Use python-dotenv (automatically loaded):
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Loads .env file
   ```

---

## Configuration Schema

### Complete Configuration Hierarchy

```
AppConfig
├── environment: str
├── debug: bool
├── api: APIConfig
│   ├── host: str
│   ├── port: int
│   ├── title: str
│   ├── version: str
│   ├── prefix: str
│   ├── cors: CORSConfig
│   │   ├── allowed_origins: List[str]
│   │   ├── allow_credentials: bool
│   │   ├── allow_methods: List[str]
│   │   └── allow_headers: List[str]
│   ├── max_file_size_mb: int
│   ├── max_batch_rows: int
│   ├── default_model: str
│   ├── enable_auth: bool
│   ├── secret_key: Optional[str]
│   └── access_token_expire_minutes: int
├── data: DataConfig
│   ├── raw_data_dir: str
│   ├── processed_data_dir: str
│   ├── datasets: List[str]
│   ├── test_size: float
│   ├── validation_size: float
│   └── stratify: bool
├── preprocessing: PreprocessingConfig
│   ├── scaling_method: str
│   ├── imputation_strategy: str
│   ├── knn_neighbors: int
│   ├── zscore_threshold: float
│   └── outlier_detection_method: str
├── validation: ValidationConfig
│   ├── ranges: Dict[str, FeatureRange]
│   ├── missing_threshold: float
│   └── required_features: List[str]
├── feature_engineering: FeatureEngineeringConfig
│   ├── polynomial_degree: int
│   ├── interaction_pairs: List[Tuple[str, str]]
│   ├── age_bins: List[int]
│   └── domain_rules: DomainRulesConfig
├── hyperparameters: HyperparameterConfig
│   ├── default_trials: int
│   ├── cv_folds: int
│   └── [model_name]: Dict[str, Any]
├── model: ModelConfig
│   ├── models_to_train: List[str]
│   ├── use_gpu: bool
│   ├── n_jobs: int
│   └── cv_metrics: List[str]
├── mlops: MLOpsConfig
│   ├── drift_detection: DriftDetectionConfig
│   │   ├── significance_level: float
│   │   ├── psi_threshold: float
│   │   └── psi_bins: int
│   ├── monitoring: MonitoringConfig
│   │   ├── accuracy_drop_threshold: float
│   │   ├── drift_score_threshold: float
│   │   └── error_rate_threshold: float
│   └── retraining: RetrainingConfig
│       ├── performance_threshold: float
│       └── interval_days: int
├── visualization: VisualizationConfig
│   ├── figure_size: Tuple[int, int]
│   ├── dpi: int
│   └── class_labels: List[str]
└── logging: LoggingConfig
    ├── level: str
    ├── log_dir: str
    └── max_bytes: int
```

---

## Using Configuration in Code

### Basic Usage

```python
from src.config import get_config

# Get configuration
config = get_config()

# Access values with type safety and autocomplete
api_port = config.api.port
db_host = config.database.host
psi_threshold = config.mlops.drift_detection.psi_threshold
```

### Accessing Nested Values

```python
# Dot notation (recommended)
cors_origins = config.api.cors.allowed_origins

# Alternative: using get method with default
threshold = config.mlops.monitoring.accuracy_drop_threshold
```

### Loading Different Environments

```python
from src.config import load_config

# Load specific environment
dev_config = load_config(environment="development")
prod_config = load_config(environment="production")
```

### Reloading Configuration (Development)

```python
from src.config import reload_config

# Reload config after changes
config = reload_config()
```

### Using ConfigManager

```python
from src.config import ConfigManager

# Get singleton instance
config_mgr = ConfigManager.get_instance()

# Access config
config = config_mgr.config

# Get specific value by path
port = config_mgr.get('api.port', default=8000)

# Update value (in-memory only, not persistent)
config_mgr.update('api.port', 9000)

# Save current config to file
config_mgr.save_current('runtime_config.yaml')
```

---

## Dependency Injection

### Using the Container

```python
from src.core import Container, IDataLoader, IModelTrainer
from src.data.data_loader import DataLoader
from src.models.train import ModelTrainer

# Get container
container = Container()

# Register services
container.register_singleton(IDataLoader, DataLoader)
container.register_transient(IModelTrainer, ModelTrainer)

# Resolve services
data_loader = container.resolve(IDataLoader)
model_trainer = container.resolve(IModelTrainer)
```

### Using Dependency Injection Decorator

```python
from src.core import inject, IDataLoader

@inject(IDataLoader)
def process_data(data_loader: IDataLoader):
    """Function with auto-injected dependency."""
    data = data_loader.load_all_datasets()
    # Process data...

# Call without arguments - data_loader is injected
process_data()
```

---

## Examples

### Example 1: Running with Different Environments

```bash
# Development (default)
python -m src.api.app

# Staging
ENV=staging python -m src.api.app

# Production
ENV=production python -m src.api.app
```

### Example 2: Override Configuration via Environment

```bash
# Run on different port
APP_API__PORT=9000 python -m src.api.app

# Use different data directory
APP_DATA__RAW_DATA_DIR=/mnt/data python -m src.train.py

# Enable GPU
APP_MODEL__USE_GPU=true python -m src.train.py
```

### Example 3: Custom Configuration File

```python
from src.config import ConfigLoader

loader = ConfigLoader()
config = loader.load(config_file="custom_config.yaml")
```

### Example 4: Programmatic Configuration

```python
from src.config import ConfigManager

config_mgr = ConfigManager.get_instance()

# Change hyperparameter tuning trials
config_mgr.update('hyperparameters.default_trials', 50)

# Change monitoring threshold
config_mgr.update('mlops.monitoring.accuracy_drop_threshold', 0.03)
```

### Example 5: Validation Ranges

```python
from src.config import get_config

config = get_config()

# Get age validation range
age_range = config.validation.ranges['age']
print(f"Age must be between {age_range.min} and {age_range.max}")

# Access all feature ranges
for feature, range_config in config.validation.ranges.items():
    print(f"{feature}: {range_config.min} - {range_config.max}")
```

---

## Best Practices

### 1. Never Hardcode Values

❌ **Bad:**
```python
max_file_size = 10 * 1024 * 1024  # Hardcoded
```

✅ **Good:**
```python
from src.config import get_config
config = get_config()
max_file_size = config.api.max_file_size_mb * 1024 * 1024
```

### 2. Use Environment Variables for Secrets

❌ **Bad:**
```yaml
api:
  secret_key: "hardcoded-secret"  # In config file
```

✅ **Good:**
```bash
export APP_API__SECRET_KEY="secret-from-vault"
```

### 3. Environment-Specific Overrides

❌ **Bad:**
```python
if environment == "production":
    trials = 100
else:
    trials = 20
```

✅ **Good:**
```yaml
# production_config.yaml
hyperparameters:
  default_trials: 100

# development_config.yaml
hyperparameters:
  default_trials: 20
```

### 4. Type Safety

Always use the Pydantic models for type checking:

```python
from src.config import get_config

config = get_config()

# This gives you autocomplete and type checking
port: int = config.api.port  # Type is known
```

---

## Troubleshooting

### Configuration Validation Errors

If you see validation errors:

```python
pydantic.error_wrappers.ValidationError: 1 validation error for AppConfig
api -> port
  value is not a valid integer (type=type_error.integer)
```

**Solution:** Check your config file or environment variable has the correct type.

### Missing Configuration File

```python
FileNotFoundError: Config directory not found: /path/to/configs
```

**Solution:** Ensure you're running from the project root or set `config_dir` explicitly.

### Environment Not Loading

If environment-specific config isn't loading:

1. Check ENV environment variable:
   ```bash
   echo $ENV
   ```

2. Verify file naming (must be `{environment}_config.yaml`):
   - `development_config.yaml`
   - `staging_config.yaml`
   - `production_config.yaml`

---

## Migration from Old Config

If you're upgrading from the old `configs/config.yaml`:

1. Old config is still supported for backward compatibility
2. New config provides:
   - Type validation
   - Environment-specific overrides
   - Environment variable support
   - Better organization

To migrate:

1. Review your current `configs/config.yaml`
2. Map values to new `configs/app_config.yaml` structure
3. Test with `ENV=development python -m src.api.app`
4. Gradually migrate code to use new config system

---

## Summary

The new configuration system provides:

✅ **100% Config-Driven** - No hardcoded values
✅ **Type-Safe** - Pydantic validation
✅ **Environment-Aware** - Dev/Staging/Prod configs
✅ **Flexible** - Override with environment variables
✅ **Modular** - Organized by domain
✅ **Testable** - Easy to mock in tests
✅ **Production-Ready** - Validated and secure

For questions or issues, refer to the Pydantic schemas in `src/config/schemas.py` or check the example configs in the `configs/` directory.
