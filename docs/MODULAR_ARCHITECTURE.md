# Modular Architecture Guide

## Overview

The Heart Disease Prediction system has been refactored to follow a **modular, loosely-coupled architecture** based on **Dependency Injection** and **Interface-Based Design**. This document explains the architectural patterns and how to work with the modular codebase.

## Table of Contents

1. [Architecture Principles](#architecture-principles)
2. [Core Concepts](#core-concepts)
3. [Directory Structure](#directory-structure)
4. [Interfaces](#interfaces)
5. [Dependency Injection](#dependency-injection)
6. [Component Modules](#component-modules)
7. [Adding New Components](#adding-new-components)
8. [Testing](#testing)
9. [Best Practices](#best-practices)

---

## Architecture Principles

### SOLID Principles

The architecture follows SOLID principles:

1. **S**ingle Responsibility - Each module has one clear purpose
2. **O**pen/Closed - Open for extension, closed for modification
3. **L**iskov Substitution - Interfaces can be swapped seamlessly
4. **I**nterface Segregation - Small, focused interfaces
5. **D**ependency Inversion - Depend on abstractions, not concretions

### Key Benefits

✅ **Loosely Coupled** - Components don't depend on each other directly
✅ **Testable** - Easy to mock dependencies for unit tests
✅ **Maintainable** - Changes in one module don't affect others
✅ **Scalable** - New components can be added without breaking existing code
✅ **Flexible** - Implementations can be swapped without code changes

---

## Core Concepts

### 1. Interfaces (Abstract Base Classes)

Interfaces define contracts that implementations must follow:

```python
from abc import ABC, abstractmethod

class IDataLoader(ABC):
    """Interface for data loading."""

    @abstractmethod
    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a dataset."""
        pass
```

### 2. Implementations

Concrete classes implement the interfaces:

```python
class DataLoader(IDataLoader):
    """CSV-based data loader implementation."""

    def __init__(self, config: DataConfig):
        self.config = config

    def load_dataset(self, name: str) -> pd.DataFrame:
        path = self.config.raw_data_dir / f"{name}.csv"
        return pd.read_csv(path)
```

### 3. Dependency Injection

Dependencies are injected rather than created:

```python
# ❌ Bad: Tightly coupled
class ModelTrainer:
    def __init__(self):
        self.data_loader = DataLoader()  # Hard dependency

# ✅ Good: Loosely coupled
class ModelTrainer:
    def __init__(self, data_loader: IDataLoader):
        self.data_loader = data_loader  # Injected dependency
```

---

## Directory Structure

```
src/
├── config/                    # Configuration system
│   ├── __init__.py
│   ├── schemas.py            # Pydantic configuration models
│   ├── loader.py             # Configuration loader
│   └── manager.py            # Configuration manager
│
├── core/                      # Core infrastructure
│   ├── __init__.py
│   ├── interfaces.py         # Abstract base classes
│   └── container.py          # Dependency injection container
│
├── data/                      # Data processing modules
│   ├── __init__.py
│   ├── data_loader.py        # IDataLoader implementation
│   ├── data_validator.py     # IDataValidator implementation
│   └── data_preprocessor.py  # IDataPreprocessor implementation
│
├── features/                  # Feature engineering
│   ├── __init__.py
│   └── feature_engineering.py # IFeatureEngineer implementation
│
├── models/                    # Model training and prediction
│   ├── __init__.py
│   ├── train.py              # IModelTrainer implementation
│   └── predict.py            # IPredictor implementation
│
├── mlops/                     # MLOps components
│   ├── __init__.py
│   ├── drift_detection.py    # IDriftDetector implementation
│   ├── monitoring.py         # IMonitor implementation
│   └── retraining.py         # IRetrainer implementation
│
└── api/                       # API endpoints
    ├── __init__.py
    ├── app.py                # FastAPI application
    └── routes.py             # API routes
```

---

## Interfaces

### Core Interfaces

Located in `src/core/interfaces.py`:

| Interface | Purpose | Key Methods |
|-----------|---------|-------------|
| `IDataLoader` | Load datasets | `load_dataset()`, `load_all_datasets()`, `combine_datasets()` |
| `IDataValidator` | Validate data | `validate_schema()`, `validate_ranges()`, `validate_all()` |
| `IDataPreprocessor` | Preprocess data | `fit()`, `transform()`, `fit_transform()` |
| `IFeatureEngineer` | Engineer features | `create_features()`, `select_features()` |
| `IModelTrainer` | Train models | `train()`, `evaluate()`, `save_model()`, `load_model()` |
| `IPredictor` | Make predictions | `predict()`, `predict_proba()`, `explain_prediction()` |
| `IDriftDetector` | Detect drift | `detect_drift()`, `calculate_psi()` |
| `IMonitor` | Monitor models | `log_prediction()`, `check_model_performance()`, `create_alert()` |
| `IRetrainer` | Retrain models | `should_retrain()`, `retrain()`, `promote_model()` |

### Interface Example

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class IDataLoader(ABC):
    """Interface for data loading components."""

    @abstractmethod
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a single dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Loaded dataset as DataFrame
        """
        pass

    @abstractmethod
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets."""
        pass
```

---

## Dependency Injection

### The Container

The `Container` class manages service registration and resolution:

```python
from src.core import Container

# Create container
container = Container()

# Register services
container.register_singleton(IDataLoader, DataLoader)
container.register_transient(IDataValidator, DataValidator)
```

### Registration Types

#### 1. Singleton (Shared Instance)

```python
# Register class (instantiated on first resolve)
container.register_singleton(IDataLoader, DataLoader)

# Register instance (already created)
loader = DataLoader(config)
container.register_singleton(IDataLoader, instance=loader)

# Register factory (function that creates instance)
container.register_singleton(
    IDataLoader,
    factory=lambda: DataLoader(get_config())
)
```

#### 2. Transient (New Instance Each Time)

```python
# New instance created on every resolve()
container.register_transient(IDataValidator, DataValidator)
```

### Resolution

```python
# Resolve a service
data_loader = container.resolve(IDataLoader)

# Resolve with arguments (for transients)
validator = container.resolve(IDataValidator, config=my_config)
```

### Automatic Injection

Use the `@inject` decorator for automatic dependency injection:

```python
from src.core import inject, IDataLoader

@inject(IDataLoader)
def train_model(data_loader: IDataLoader):
    """Function with auto-injected dependency."""
    data = data_loader.load_all_datasets()
    # Train model...

# Call without arguments - data_loader is injected
train_model()
```

---

## Component Modules

### Data Module

#### DataLoader

```python
from src.data import DataLoader
from src.config import get_config

config = get_config()
loader = DataLoader(config.data)

# Load single dataset
df = loader.load_dataset("cleveland")

# Load all datasets
datasets = loader.load_all_datasets()

# Combine datasets
combined = loader.combine_datasets(["cleveland", "hungarian"])
```

#### DataValidator

```python
from src.data import DataValidator

validator = DataValidator(config.validation)

# Validate schema
is_valid, errors = validator.validate_schema(df)

# Validate ranges
is_valid, errors = validator.validate_ranges(df)

# Run all validations
results = validator.validate_all(df)
```

#### DataPreprocessor

```python
from src.data import DataPreprocessor

preprocessor = DataPreprocessor(config.preprocessing)

# Fit on training data
preprocessor.fit(X_train, y_train)

# Transform test data
X_test_scaled = preprocessor.transform(X_test)

# Fit and transform in one step
X_train_scaled = preprocessor.fit_transform(X_train, y_train)
```

### Features Module

#### FeatureEngineer

```python
from src.features import FeatureEngineer

engineer = FeatureEngineer(config.feature_engineering)

# Create engineered features
df_with_features = engineer.create_features(df)

# Select best features
X_selected, feature_names = engineer.select_features(X, y)
```

### Models Module

#### ModelTrainer

```python
from src.models import ModelTrainer

trainer = ModelTrainer(config.model, config.hyperparameters)

# Train models
results = trainer.train(X_train, y_train)

# Evaluate model
metrics = trainer.evaluate(model, X_test, y_test)

# Save model
trainer.save_model(model, "models/rf_model.pkl", metadata={"version": "1.0"})

# Load model
model, metadata = trainer.load_model("models/rf_model.pkl")
```

#### Predictor

```python
from src.models import Predictor

predictor = Predictor(config.model)

# Make prediction
result = predictor.predict(X, model_name="random_forest")

# Get probabilities
probas = predictor.predict_proba(X)

# Explain prediction
explanation = predictor.explain_prediction(X)
```

### MLOps Module

#### DriftDetector

```python
from src.mlops import DriftDetector

detector = DriftDetector(config.mlops.drift_detection)

# Detect drift
drift_results = detector.detect_drift(
    reference_data=baseline_data,
    current_data=new_data,
    feature_names=feature_names
)

# Calculate PSI
psi = detector.calculate_psi(baseline_array, current_array)
```

#### Monitor

```python
from src.mlops import Monitor

monitor = Monitor(config.mlops.monitoring)

# Log prediction
monitor.log_prediction(
    model_name="random_forest",
    features={"age": 55, "chol": 240},
    prediction=1,
    probability=0.85,
    latency_ms=45.2
)

# Check performance
performance = monitor.check_model_performance("random_forest")

# Create alert
monitor.create_alert(
    alert_type="performance",
    severity="warning",
    message="Accuracy dropped by 6%",
    metadata={"current_accuracy": 0.82, "baseline": 0.88}
)
```

---

## Adding New Components

### Step 1: Define Interface

Add to `src/core/interfaces.py`:

```python
class INewComponent(ABC):
    """Interface for new component."""

    @abstractmethod
    def do_something(self, input: Any) -> Any:
        """Do something useful."""
        pass
```

### Step 2: Create Implementation

Create `src/module/new_component.py`:

```python
from src.core.interfaces import INewComponent
from src.config import get_config

class NewComponent(INewComponent):
    """Implementation of INewComponent."""

    def __init__(self, config):
        self.config = config

    def do_something(self, input: Any) -> Any:
        """Implementation of the method."""
        # Your logic here
        return result
```

### Step 3: Register in Container

In your bootstrap/initialization code:

```python
from src.core import get_container
from src.module.new_component import NewComponent

container = get_container()
container.register_singleton(INewComponent, NewComponent)
```

### Step 4: Use the Component

```python
from src.core import get_container, INewComponent

container = get_container()
component = container.resolve(INewComponent)
result = component.do_something(input_data)
```

---

## Testing

### Mocking Dependencies

The modular architecture makes testing easy:

```python
import pytest
from unittest.mock import Mock
from src.models.train import ModelTrainer

def test_model_trainer():
    # Mock the data loader
    mock_loader = Mock(spec=IDataLoader)
    mock_loader.load_dataset.return_value = test_df

    # Inject mock
    trainer = ModelTrainer(data_loader=mock_loader)

    # Test
    result = trainer.train(X_train, y_train)
    assert result is not None
    mock_loader.load_dataset.assert_called_once()
```

### Test Fixtures

```python
import pytest
from src.config import load_config

@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return load_config(environment="development")

@pytest.fixture
def data_loader(test_config):
    """Fixture providing data loader."""
    return DataLoader(test_config.data)

def test_load_dataset(data_loader):
    df = data_loader.load_dataset("cleveland")
    assert len(df) > 0
```

---

## Best Practices

### 1. Program to Interfaces

✅ **Good:**
```python
def train_model(loader: IDataLoader):  # Depend on interface
    data = loader.load_dataset("cleveland")
```

❌ **Bad:**
```python
def train_model(loader: DataLoader):  # Depend on concrete class
    data = loader.load_dataset("cleveland")
```

### 2. Inject Dependencies

✅ **Good:**
```python
class ModelTrainer:
    def __init__(self, loader: IDataLoader, config: ModelConfig):
        self.loader = loader  # Injected
        self.config = config  # Injected
```

❌ **Bad:**
```python
class ModelTrainer:
    def __init__(self):
        self.loader = DataLoader()  # Created internally
        self.config = get_config()   # Global state
```

### 3. Use Configuration

✅ **Good:**
```python
trials = config.hyperparameters.default_trials
```

❌ **Bad:**
```python
trials = 100  # Hardcoded
```

### 4. Keep Modules Focused

Each module should have a single, clear responsibility:

- `data/` - Data loading and preprocessing
- `features/` - Feature engineering
- `models/` - Model training and prediction
- `mlops/` - Monitoring and operations
- `api/` - API endpoints

### 5. Type Hints

Always use type hints for better IDE support:

```python
def load_dataset(self, name: str) -> pd.DataFrame:
    ...
```

---

## Migration Guide

### From Old Code to Modular

**Before:**
```python
# Tightly coupled, hardcoded
class Trainer:
    def train(self):
        loader = DataLoader()
        data = loader.load_dataset("cleveland")
        trials = 100  # Hardcoded
        # ...
```

**After:**
```python
# Loosely coupled, config-driven
class Trainer:
    def __init__(self, loader: IDataLoader, config: ModelConfig):
        self.loader = loader
        self.config = config

    def train(self):
        data = self.loader.load_dataset("cleveland")
        trials = self.config.hyperparameters.default_trials
        # ...
```

---

## Summary

The modular architecture provides:

✅ **Separation of Concerns** - Each module has one responsibility
✅ **Dependency Injection** - Loose coupling between components
✅ **Interface-Based** - Program to contracts, not implementations
✅ **Config-Driven** - No hardcoded values
✅ **Testable** - Easy to mock and test
✅ **Maintainable** - Changes don't ripple through the codebase
✅ **Extensible** - New components can be added easily

For detailed configuration options, see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).
