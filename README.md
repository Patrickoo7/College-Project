# Heart Disease Prediction - ML System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-ready machine learning system that predicts heart disease risk using patient data and multiple ML algorithms.**

---

## 📖 What is This Project?

This is a **complete machine learning project** that predicts whether a person has heart disease based on medical data like age, blood pressure, cholesterol levels, and other health indicators.

### Why This Project?

- **For Healthcare**: Helps doctors identify high-risk patients early
- **For Learning**: Demonstrates professional ML development practices
- **For Production**: Ready to deploy with API, web interface, and monitoring

### What You'll Get

✅ Trained ML models that predict heart disease with ~85% accuracy
✅ REST API to make predictions from any application
✅ Web interface for easy interaction
✅ Complete MLOps pipeline (tracking, monitoring, retraining)
✅ Production-ready code with testing and documentation

---

## 🎯 Quick Start (For Absolute Beginners)

### Step 1: Prerequisites

**What you need installed on your computer:**

1. **Python 3.8 or higher** ([Download here](https://www.python.org/downloads/))
   - Check: Open terminal and run `python --version`

2. **Git** ([Download here](https://git-scm.com/downloads))
   - Check: Run `git --version`

3. **pip** (comes with Python)
   - Check: Run `pip --version`

**Optional (for faster training):**
- NVIDIA GPU with CUDA for 5-15x speedup (see [GPU Setup](#gpu-setup-optional))

### Step 2: Download the Project

```bash
# Open terminal/command prompt and run:

# 1. Go to where you want the project
cd ~/Desktop  # or wherever you want

# 2. Clone (download) this repository
git clone https://github.com/Patrickoo7/College-Project.git

# 3. Enter the project folder
cd College-Project
```

### Step 3: Install Dependencies

```bash
# Create a virtual environment (isolated Python environment)
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Install required packages (this will take 5-10 minutes)
pip install --upgrade pip
pip install -r requirements.txt
```

**You'll see your terminal prompt change to show `(venv)` - this means you're in the virtual environment.**

### Step 4: Train Your First Model

```bash
# Train a simple Random Forest model
python -m src.models.train

# This will:
# - Load and validate data
# - Train multiple ML models
# - Save trained models to models/ folder
# - Show accuracy results

# Wait 5-15 minutes for training to complete
```

### Step 5: Start the API (Make Predictions)

```bash
# Start the web API
python -m src.api.app

# You should see:
# "Uvicorn running on http://0.0.0.0:8000"
```

**Open your browser and visit:**
- http://localhost:8000 - API home
- http://localhost:8000/docs - Interactive API documentation

### Step 6: Make Your First Prediction

**Option A: Using the web interface (Interactive Docs)**

1. Go to http://localhost:8000/docs
2. Click on `POST /api/v1/predict`
3. Click "Try it out"
4. Paste this example:
   ```json
   {
     "age": 55,
     "sex": 1,
     "cp": 0,
     "trestbps": 140,
     "chol": 250,
     "fbs": 0,
     "restecg": 0,
     "thalach": 150,
     "exang": 0,
     "oldpeak": 1.0,
     "slope": 1,
     "ca": 0,
     "thal": 2
   }
   ```
5. Click "Execute"
6. See the prediction result!

**Option B: Using curl (command line)**

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 0, "trestbps": 140,
    "chol": 250, "fbs": 0, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
  }'
```

**You'll get a response like:**
```json
{
  "prediction": 1,
  "prediction_label": "Disease",
  "confidence": 0.75,
  "probability": {
    "disease": 0.75,
    "no_disease": 0.25
  }
}
```

🎉 **Congratulations! You just made your first heart disease prediction!**

---

## 📊 Datasets Used

This project uses **4 different heart disease datasets** from the UCI Machine Learning Repository. These are real medical datasets used in research.

### Dataset Details

| Dataset | Records | Location | Description |
|---------|---------|----------|-------------|
| **Cleveland** | 303 | Cleveland Clinic | Most commonly used, highest quality |
| **Hungarian** | 294 | Hungarian Institute of Cardiology | European patients |
| **Switzerland** | 123 | University Hospital, Zurich | Swiss patients |
| **Long Beach VA** | 200 | VA Medical Center, Long Beach | US veterans |

**Total**: 920 patient records from different populations

### What's in the Data?

Each patient record contains **13 medical features**:

1. **age**: Age in years (29-77)
2. **sex**: Male (1) or Female (0)
3. **cp**: Chest pain type (0-3)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (94-200 mm Hg)
5. **chol**: Cholesterol level (126-564 mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1=true, 0=false)
7. **restecg**: Resting ECG results (0-2)
8. **thalach**: Maximum heart rate achieved (71-202)
9. **exang**: Exercise induced angina (1=yes, 0=no)
10. **oldpeak**: ST depression induced by exercise (0-6.2)
11. **slope**: Slope of peak exercise ST segment (0-2)
12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **thal**: Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)

**Target**: **target** (0 = No disease, 1 = Has disease)

### Where is the Data?

```
data/raw/
├── cleveland.csv      # Main dataset
├── hungarian.csv      # Hungarian data
├── switzerland.csv    # Swiss data
└── va.csv            # Long Beach VA data
```

---

## 🏗️ Project Structure (Simplified)

```
College-Project/
│
├── 📁 src/                    # Source code
│   ├── 📁 config/             # Configuration system
│   ├── 📁 data/               # Data loading & processing
│   ├── 📁 features/           # Feature engineering
│   ├── 📁 models/             # ML model training & prediction
│   ├── 📁 api/                # REST API
│   └── 📁 mlops/              # MLOps (monitoring, drift detection)
│
├── 📁 data/                   # Datasets
│   ├── raw/                   # Original data
│   └── processed/             # Cleaned data
│
├── 📁 models/                 # Trained models (created after training)
│
├── 📁 configs/                # Configuration files
│   ├── app_config.yaml        # Main config
│   └── production_config.yaml # Production settings
│
├── 📁 tests/                  # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
│
├── 📁 notebooks/              # Jupyter notebooks (exploratory)
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Configuration

The project uses **YAML configuration files** to control all settings - no need to change code!

### Main Configuration File

**File**: `configs/app_config.yaml`

```yaml
# Example: Change basic settings
environment: development  # development, staging, or production
debug: true              # Enable debug mode

api:
  host: 0.0.0.0
  port: 8000             # Change API port
  enable_auth: false     # Enable JWT authentication

model:
  use_gpu: true          # Use GPU if available
  n_jobs: -1             # CPU cores (-1 = use all)

data:
  datasets:              # Which datasets to use
    - cleveland
    - hungarian
```

### Environment-Specific Configuration

```bash
# Development (default)
ENV=development python -m src.api.app

# Production
ENV=production python -m src.api.app

# Custom settings via environment variables
APP_API__PORT=9000 python -m src.api.app
```

**Environment variable format**: `APP_<SECTION>__<KEY>=value`

Examples:
```bash
APP_API__PORT=9000              # Set API port to 9000
APP_MODEL__USE_GPU=false        # Disable GPU
APP_API__ENABLE_AUTH=true       # Enable authentication
```

---

## 🚀 How to Use

### 1. Train Models

```bash
# Train all models (takes 10-20 minutes)
python -m src.models.train

# Train specific model
python -m src.models.train --model random_forest

# Models are saved to: models/
```

**What happens during training:**
- Loads data from `data/raw/`
- Validates data quality
- Preprocesses (cleaning, scaling)
- Engineers features (creates new features)
- Trains 8+ different ML algorithms
- Evaluates with cross-validation
- Saves best models
- Logs everything to MLflow

### 2. Make Predictions

#### Option A: Python API

```python
from src.models.predict import HeartDiseasePredictor

# Load trained model
predictor = HeartDiseasePredictor(model_path="models/random_forest.pkl")

# Predict for one patient
result = predictor.predict_single(
    age=55, sex=1, cp=0, trestbps=140, chol=250,
    fbs=0, restecg=0, thalach=150, exang=0,
    oldpeak=1.0, slope=1, ca=0, thal=2
)

print(result)
# Output: {'prediction': 1, 'prediction_label': 'Disease', 'probability': {...}}
```

#### Option B: REST API

**Start the API:**
```bash
python -m src.api.app
# API runs on http://localhost:8000
```

**Make prediction via HTTP:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": 1, "cp": 0, ...}'
```

### 3. View Experiment Tracking

```bash
# Start MLflow UI
mlflow ui

# Open browser: http://localhost:5000
# See all training runs, metrics, and models
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report: open htmlcov/index.html
```

---

## 🔧 Advanced Features

### GPU Setup (Optional)

**For 5-15x faster training on NVIDIA GPUs:**

**Requirements:**
- NVIDIA GPU (GTX 1060 or better)
- CUDA Toolkit 11.8+ ([Download](https://developer.nvidia.com/cuda-downloads))

**Installation:**
```bash
# Install GPU-accelerated packages
pip install xgboost lightgbm catboost

# For PyTorch (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "from src.utils.gpu_utils import get_gpu_manager; get_gpu_manager().print_gpu_summary()"
```

**Auto-detection**: The system automatically uses GPU if available, falls back to CPU if not.

### Authentication (Optional)

**Enable JWT authentication for API:**

```bash
# 1. Generate secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. Set environment variables
export APP_API__ENABLE_AUTH=true
export APP_API__SECRET_KEY="your-secret-key-here"

# 3. Start API
python -m src.api.app

# 4. Login to get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=secret"

# 5. Use token in requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/predict -d '{...}'
```

**Default credentials**: `admin` / `secret` (change in `src/api/auth.py`)

### Docker Deployment

**Run with Docker (no Python installation needed):**

```bash
# Build and run
docker-compose up --build

# API available at: http://localhost:8000
# Web app at: http://localhost:8501
```

### Monitoring & Drift Detection

```bash
# Enable model monitoring
python -m src.mlops.monitoring

# Detect data drift
python -m src.mlops.drift_detection

# Automatic retraining if drift detected
python -m src.mlops.retraining
```

---

## 🤖 Machine Learning Models

The system trains and compares **8+ algorithms**:

1. **Logistic Regression** - Simple, interpretable baseline
2. **Random Forest** - Ensemble of decision trees (usually best)
3. **XGBoost** - Gradient boosting (high performance)
4. **LightGBM** - Fast gradient boosting
5. **CatBoost** - Handles categorical features well
6. **SVM** - Support Vector Machine
7. **KNN** - K-Nearest Neighbors
8. **Decision Tree** - Simple tree-based model

**Plus ensemble methods**: Voting and Stacking classifiers

### Model Performance

**Expected accuracy**: 82-88% (varies by dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 85.2% | 86.1% | 84.3% | 85.2% |
| XGBoost | 84.8% | 85.7% | 83.9% | 84.8% |
| LightGBM | 84.5% | 85.3% | 83.7% | 84.5% |

*(Actual results depend on data split and hyperparameters)*

---

## 📝 API Endpoints

**Base URL**: `http://localhost:8000/api/v1`

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API health |
| `/predict` | POST | Predict single patient |
| `/predict/batch` | POST | Predict multiple patients |
| `/predict/upload` | POST | Upload CSV file for batch predictions |
| `/models` | GET | List available models |
| `/models/{name}` | GET | Get model info |
| `/models/{name}/use` | POST | Switch active model |
| `/auth/login` | POST | Get JWT token (if auth enabled) |
| `/auth/status` | GET | Check auth status |

### Example: Batch Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"age": 55, "sex": 1, "cp": 0, ...},
      {"age": 60, "sex": 0, "cp": 1, ...}
    ]
  }'
```

### Example: Upload CSV

```bash
curl -X POST http://localhost:8000/api/v1/predict/upload \
  -F "file=@patients.csv"

# Returns CSV with predictions
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"

```bash
# Solution: Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Problem: "Port 8000 already in use"

```bash
# Solution: Change port in config
APP_API__PORT=9000 python -m src.api.app

# Or kill the process using port 8000
# Mac/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Problem: "CUDA out of memory" (GPU)

```bash
# Solution: Disable GPU or reduce batch size
export APP_MODEL__USE_GPU=false
python -m src.models.train
```

### Problem: "Data files not found"

```bash
# Solution: Make sure you're in project root directory
cd /path/to/College-Project

# Check data exists
ls data/raw/
# Should see: cleveland.csv, hungarian.csv, etc.
```

### Problem: "Permission denied"

```bash
# Solution: Run with proper permissions
# Mac/Linux:
chmod +x script.sh
./script.sh

# Or use sudo (not recommended for Python scripts)
```

### Problem: Training is too slow

**Solutions:**
1. **Use GPU** (see [GPU Setup](#gpu-setup-optional))
2. **Reduce number of models**: Edit `configs/app_config.yaml`
   ```yaml
   model:
     models_to_train:
       - random_forest  # Only train this one
   ```
3. **Use fewer datasets**: Edit config
   ```yaml
   data:
     datasets:
       - cleveland  # Only use Cleveland data
   ```

---

## 📚 Learning Resources

### Understanding the Code

**Start here if you're new to ML:**

1. **Data Loading**: `src/data/data_loader.py`
   - How data is loaded from CSV files

2. **Data Validation**: `src/data/data_validator.py`
   - How we check data quality

3. **Feature Engineering**: `src/features/feature_engineering.py`
   - How we create new features

4. **Model Training**: `src/models/train.py`
   - How models are trained and saved

5. **API**: `src/api/app.py`, `src/api/routes.py`
   - How the REST API works

### Architecture

**Modern design patterns used:**

- **Dependency Injection**: Loose coupling between components
- **Configuration-Driven**: All settings in YAML files
- **Interface-Based**: Abstract interfaces for testability
- **Type-Safe**: Pydantic models for validation

### Testing

```bash
# Run tests to learn how components work
pytest tests/unit/ -v

# Look at test files:
# - tests/unit/test_data_loader.py
# - tests/unit/test_data_validator.py
# - tests/integration/test_api_integration.py
```

---

## 🔄 Common Workflows

### 1. Training a New Model

```bash
# 1. Make changes to config
nano configs/app_config.yaml

# 2. Train
python -m src.models.train

# 3. View results
mlflow ui
```

### 2. Deploying to Production

```bash
# 1. Set production config
export ENV=production
export APP_API__ENABLE_AUTH=true
export APP_API__SECRET_KEY="your-secret"

# 2. Start API
python -m src.api.app

# Or use Docker
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Updating the Model

```bash
# 1. Retrain with new data
python -m src.models.train

# 2. Restart API to load new model
# (or use the /models/{name}/use endpoint)
```

### 4. Monitoring in Production

```bash
# Start monitoring
python -m src.mlops.monitoring

# Check for drift
python -m src.mlops.drift_detection

# Auto-retrain if needed
python -m src.mlops.retraining
```

---

## 🎓 For Students & Developers

### What You Can Learn

- ✅ **Machine Learning**: Classification, evaluation, ensemble methods
- ✅ **MLOps**: Experiment tracking, model versioning, monitoring
- ✅ **API Development**: FastAPI, REST principles, authentication
- ✅ **Software Engineering**: Design patterns, testing, configuration
- ✅ **DevOps**: Docker, CI/CD, deployment

### Extending the Project

**Ideas for improvements:**

1. **Add more models**: Neural networks, ensemble methods
2. **Feature selection**: Implement automated feature selection
3. **Hyperparameter tuning**: Add automated tuning with Optuna
4. **Web interface**: Build React/Vue.js frontend
5. **Mobile app**: Create mobile app using the API
6. **Real-time predictions**: WebSocket support for live predictions
7. **A/B testing**: Compare different models in production

### Using for Your Portfolio

This project demonstrates:

- ✅ Production-ready ML pipeline
- ✅ RESTful API design
- ✅ MLOps best practices
- ✅ Clean code architecture
- ✅ Testing and documentation
- ✅ Docker deployment

---

## 📞 Support

### Getting Help

1. **Check logs**: `logs/app.log`
2. **Run tests**: `pytest -v`
3. **Check configuration**: `configs/app_config.yaml`
4. **Review error messages** carefully

### Common Questions

**Q: Can I use my own dataset?**
A: Yes! Place your CSV in `data/raw/` and update `configs/app_config.yaml`:
```yaml
data:
  datasets:
    - your_dataset_name  # your_dataset_name.csv
```

**Q: How do I change the model?**
A: Use the API endpoint:
```bash
curl -X POST http://localhost:8000/api/v1/models/xgboost/use
```

**Q: Can I deploy to cloud?**
A: Yes! See `azure/` folder for Azure deployment, or use Docker on AWS/GCP.

**Q: How do I add authentication?**
A: Set `APP_API__ENABLE_AUTH=true` and provide secret key (see [Authentication](#authentication-optional))

---

## 📄 License

MIT License - Feel free to use this project for learning or commercial purposes.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the heart disease datasets
- **FastAPI** for the excellent web framework
- **MLflow** for experiment tracking
- **Scikit-learn** for ML algorithms

---

## 📧 Contact

**Author**: Patrickoo7
**Email**: awateprateek@gmail.com
**GitHub**: [Patrickoo7/College-Project](https://github.com/Patrickoo7/College-Project)

---

**Made with ❤️ for learning and healthcare**

---

## Quick Command Reference

```bash
# Installation
pip install -r requirements.txt

# Training
python -m src.models.train

# Start API
python -m src.api.app

# Run tests
pytest

# View experiments
mlflow ui

# Docker
docker-compose up

# Check config
cat configs/app_config.yaml
```

---

**Last Updated**: November 2025
**Version**: 1.0.0
