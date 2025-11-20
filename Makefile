.PHONY: help install install-dev clean lint format test test-cov train evaluate predict mlflow-ui docs setup

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
FLAKE8 := $(PYTHON) -m flake8
ISORT := $(PYTHON) -m isort
MYPY := $(PYTHON) -m mypy

# Directories
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
MODELS_DIR := models
LOGS_DIR := logs
MLRUNS_DIR := mlruns

help:  ## Show this help message
	@echo "Heart Disease Prediction - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev:  ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

setup:  ## Initial project setup
	@echo "Setting up project directories..."
	mkdir -p $(DATA_DIR)/{raw,processed,external}
	mkdir -p $(MODELS_DIR)/{artifacts,scaler,metadata}
	mkdir -p $(LOGS_DIR)
	mkdir -p notebooks
	mkdir -p tests
	@echo "Setup complete!"

# Code Quality
lint:  ## Run linting checks
	@echo "Running flake8..."
	$(FLAKE8) $(SRC_DIR) --max-line-length=100 --exclude=__pycache__
	@echo "Running mypy..."
	$(MYPY) $(SRC_DIR) --ignore-missing-imports

format:  ## Format code with black and isort
	@echo "Formatting with black..."
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@echo "Sorting imports with isort..."
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

format-check:  ## Check code formatting without modifying
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

# Testing
test:  ## Run tests
	$(PYTEST) $(TEST_DIR) -v

test-cov:  ## Run tests with coverage
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term

test-fast:  ## Run tests without coverage (fast)
	$(PYTEST) $(TEST_DIR) -v -x

# Data Processing
data-validate:  ## Validate raw data
	$(PYTHON) -c "from src.data.data_loader import DataLoader; from src.data.data_validator import DataValidator; loader = DataLoader(); validator = DataValidator(); df = loader.load_cleveland(); validator.validate_all(df)"

data-process:  ## Process raw data
	$(PYTHON) -c "from src.data.data_loader import DataLoader; from src.data.data_preprocessor import DataPreprocessor; loader = DataLoader(); preprocessor = DataPreprocessor(); df = loader.load_cleveland(); df_clean = preprocessor.preprocess_pipeline(df); loader.save_processed_data(df_clean, 'cleveland_processed')"

# Model Training
train:  ## Train all models
	@echo "Training all models..."
	$(PYTHON) -m src.models.train

train-single:  ## Train a single model (usage: make train-single MODEL=logistic_regression)
	$(PYTHON) -m src.models.train --model $(MODEL)

# Evaluation
evaluate:  ## Evaluate all trained models
	@echo "Evaluating models..."
	$(PYTHON) -m src.models.evaluate

# Prediction
predict:  ## Make predictions (usage: make predict INPUT=data.csv)
	$(PYTHON) -m src.models.predict --input $(INPUT)

# MLflow
mlflow-ui:  ## Start MLflow UI
	mlflow ui --backend-store-uri $(MLRUNS_DIR) --port 5000

mlflow-clean:  ## Clean MLflow runs
	rm -rf $(MLRUNS_DIR)

# Documentation
docs:  ## Build documentation
	mkdocs build

docs-serve:  ## Serve documentation locally
	mkdocs serve

# Cleaning
clean:  ## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Cleaned temporary files!"

clean-all: clean mlflow-clean  ## Clean everything including MLflow runs
	rm -rf $(LOGS_DIR)/*
	@echo "Cleaned everything!"

clean-models:  ## Clean saved models
	rm -rf $(MODELS_DIR)/artifacts/*
	rm -rf $(MODELS_DIR)/scaler/*
	@echo "Cleaned model artifacts!"

# Docker
docker-build:  ## Build Docker image
	docker build -t heart-disease-prediction:latest .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 heart-disease-prediction:latest

# Git
git-status:  ## Show git status
	git status

git-add-all:  ## Add all changes to git
	git add .

# Complete Pipeline
pipeline:  ## Run complete ML pipeline
	@echo "Running complete ML pipeline..."
	$(MAKE) data-validate
	$(MAKE) data-process
	$(MAKE) train
	$(MAKE) evaluate
	@echo "Pipeline complete!"

# Development
dev-setup: install-dev setup  ## Complete development setup
	@echo "Development environment ready!"

# Info
info:  ## Show project information
	@echo "Project: Heart Disease Prediction"
	@echo "Python Version: $$($(PYTHON) --version)"
	@echo "Pip Version: $$($(PIP) --version)"
	@echo "Project Directory: $$(pwd)"
