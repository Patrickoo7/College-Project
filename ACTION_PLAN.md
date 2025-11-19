# Prioritized Action Plan - Heart Disease Prediction ML Project

**Total Recommendations:** 87 specific improvements  
**Estimated Total Effort:** 4-5 weeks for critical + high priority  
**Expected Improvement:** 5.5/10 → 7.5/10 maturity score

---

## CRITICAL PRIORITY (Do This Week!)

### CR1: Fix CORS Security Vulnerability (5 minutes)
**File:** `/home/user/College-Project/src/api/app.py` Line 33  
**Current Issue:** `allow_origins=["*"]` allows attacks from any domain

```python
# BEFORE (INSECURE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AFTER (SECURE)
import os
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,https://app.example.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

**PR Template:** "Security: Restrict CORS to specific domains"

---

### CR2: Make CI/CD Tests Blocking (5 minutes)
**File:** `/home/user/College-Project/.github/workflows/azure-deploy.yml` Line 54  
**Current Issue:** Tests fail but code still deploys to production

```yaml
# BEFORE (ALLOWS BROKEN CODE)
- name: Run tests
  run: |
    pytest tests/ -v --cov=src --cov-report=xml --cov-report=term
  continue-on-error: true  # <-- PROBLEM

# AFTER (BLOCKS BROKEN CODE)
- name: Run tests
  run: |
    pytest tests/ -v --cov=src --cov-report=xml --cov-report=term
  continue-on-error: false  # <-- FIXED
```

**PR Template:** "CI/CD: Make test failures block deployments"

---

### CR3: Add Pre-commit Hooks (1 day)
**File:** Create `/home/user/College-Project/.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.10
        args: ["--line-length=100"]

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]

  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ["-c", ".bandit", "-r", "src/"]

  - repo: https://github.com/hadialqattan/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
```

**Installation:**
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

**PR Template:** "DevOps: Add pre-commit hooks for code quality"

---

### CR4: Create .env.example (1 hour)
**File:** Create `/home/user/College-Project/.env.example`

```bash
# API Configuration
API_KEY=your-api-key-here
ALLOWED_ORIGINS=http://localhost:3000,https://app.example.com

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=sqlite:///./data.db
PREDICTION_STORE_PATH=data/predictions.db

# Monitoring
LOG_LEVEL=INFO
ENABLE_AUDIT_LOGGING=true
REDACT_PII=true
PROMETHEUS_ENABLED=true

# Deployment
ENVIRONMENT=development
DEBUG=false

# Alerts
SLACK_WEBHOOK_URL=your-webhook-url
EMAIL_ALERTS_ENABLED=false
```

**PR Template:** "Docs: Add environment configuration template"

---

## HIGH PRIORITY (Next 2 Weeks)

### HP1: Create Comprehensive Test Suite (2-3 weeks)

#### HP1a: Data Module Tests (3-4 days)
**File:** Create `/home/user/College-Project/tests/test_data_loader.py`  
**Target:** 40+ test functions

```python
# Key tests needed:
# - test_load_cleveland()
# - test_load_all_uci_datasets()
# - test_combine_datasets()
# - test_missing_data_handling()
# - test_data_type_validation()
# - test_duplicate_row_handling()
# - test_file_not_found_error()
# - test_empty_file_handling()
```

#### HP1b: Model Tests (3-4 days)
**File:** Create `/home/user/College-Project/tests/test_models.py`  
**Target:** 60+ test functions

```python
# Key tests needed:
# - test_trainer_initialization()
# - test_train_single_model()
# - test_train_all_models()
# - test_model_persistence()
# - test_model_loading()
# - test_prediction_output_shape()
# - test_batch_prediction()
# - test_model_evaluation_metrics()
# - test_cross_validation()
# - test_ensemble_creation()
```

#### HP1c: MLOps Tests (2-3 days)
**File:** Create `/home/user/College-Project/tests/test_mlops.py`  
**Target:** 50+ test functions

```python
# Key tests needed:
# - test_drift_detection()
# - test_feature_store_operations()
# - test_retraining_trigger()
# - test_monitoring_alerts()
# - test_prediction_store()
# - test_hyperparameter_tuning()
```

**Impact:** 3% → 40%+ code coverage

---

### HP2: API Authentication & Rate Limiting (2-3 days)

**File:** Create `/home/user/College-Project/src/api/security.py`

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
import jwt
import os
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
```

**File:** Create `/home/user/College-Project/src/api/rate_limiter.py`

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Usage in routes:
# @router.post("/predict")
# @limiter.limit("10/minute")
# async def predict(request: Request, patient: PatientInput):
#     ...
```

**Update:** `/home/user/College-Project/src/api/routes.py`
- Add `@limiter.limit("10/minute")` to `/predict` endpoint
- Add `@limiter.limit("5/minute")` to `/predict/batch` endpoint
- Add `@limiter.limit("2/minute")` to `/predict/upload` endpoint

**PR Template:** "Security: Add API authentication and rate limiting"

---

### HP3: Monitoring & Observability Setup (2-3 days)

**File:** Create `/home/user/College-Project/src/api/metrics.py`

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
predictions_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_name', 'prediction_label']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0]
)

active_models = Gauge(
    'active_models',
    'Number of loaded models'
)

api_request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

accuracy_gauge = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_name']
)
```

**File:** Create `/home/user/College-Project/docker-compose-monitoring.yml`

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

**Create:** `/home/user/College-Project/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'heart-disease-api'
    static_configs:
      - targets: ['localhost:8000']
```

**PR Template:** "Observability: Add Prometheus metrics and Grafana setup"

---

### HP4: MLOps Infrastructure (3-4 days)

#### HP4a: Configure DVC (1 day)
**File:** Update `/home/user/College-Project/.dvc/config`

```ini
['remote "myremote"']
    url = s3://my-bucket/dvc-storage
    
[core]
    remote = myremote
    autostage = true
```

#### HP4b: Create A/B Testing Framework (2-3 days)
**File:** Create `/home/user/College-Project/src/mlops/ab_testing.py`

```python
class ABTester:
    """A/B testing framework for safe model deployment."""
    
    def __init__(self):
        self.control_model = None
        self.candidate_model = None
        self.results = {}
    
    def run_test(self, test_data, test_labels, traffic_split=0.5):
        """Run A/B test with traffic split."""
        pass
    
    def calculate_statistical_significance(self):
        """Calculate if differences are statistically significant."""
        pass
    
    def get_winner(self, confidence_level=0.95):
        """Determine winning model."""
        pass
```

**PR Template:** "MLOps: Implement A/B testing framework"

---

## MEDIUM PRIORITY (Weeks 3-4)

### MP1: Database Optimization (1-2 days)
- Add indexes to prediction_store table
- Implement query result caching
- Add connection pooling
- Optimize batch operations

### MP2: Code Refactoring (3-4 days)
- Split `src/reporting/report_generator.py` (595 lines)
- Split `src/mlops/prediction_store.py` (549 lines)
- Split `src/models/evaluate.py` (524 lines)
- Add complexity analysis with radon

### MP3: Data Quality Framework (2-3 days)
- Integrate Great Expectations
- Create Pandera schemas
- Add data profiling with ydata-profiling
- Implement anomaly detection

### MP4: Frontend Enhancements (2-3 days)
- Add prediction history dashboard
- Add batch prediction interface
- Display SHAP explanations
- Mobile optimization

---

## LOW PRIORITY (Later)

### LP1: Documentation (2-3 days)
- Add architecture diagrams (Mermaid)
- Create API examples in multiple languages
- Write troubleshooting guide
- Record video tutorials

### LP2: Advanced Performance (2-3 days)
- Model quantization
- ONNX conversion
- SHAP value caching
- Query result caching

### LP3: Model Explainability (2-3 days)
- Add confidence intervals for explanations
- Interactive explanation dashboard
- Fairness/bias analysis
- Adversarial robustness testing

---

## QUICK WINS (Can Do Today)

1. **Fix CORS** (5 min) - 90% risk reduction
2. **Make tests blocking** (5 min) - Prevent broken deployments
3. **Create .env.example** (15 min) - Documentation
4. **Add basic monitoring endpoints** (30 min) - Visibility
5. **Update README with security note** (15 min) - User awareness

**Total Time:** Less than 2 hours for huge risk reduction!

---

## Key Metrics to Track

### Before & After Comparison

| Metric | Before | After (Goal) | Time |
|--------|--------|--------------|------|
| Test Coverage | 3% | 40%+ | 2-3 weeks |
| Code Coverage Score | 2/10 | 8/10 | 2-3 weeks |
| Security Score | 4/10 | 9/10 | 1 week |
| Monitoring Score | 3/10 | 8/10 | 2-3 days |
| API Latency | Unknown | <100ms | 2-3 days |
| Overall Maturity | 5.5/10 | 7.5/10 | 4-5 weeks |

---

## Sprint Planning Template

### Sprint 1 (Week 1): Critical Fixes
- [ ] CR1: Fix CORS (5 min)
- [ ] CR2: Make tests blocking (5 min)
- [ ] CR3: Add pre-commit hooks (1 day)
- [ ] CR4: Create .env.example (1 hour)
- [ ] Setup basic Prometheus metrics (1 day)

**Deliverables:** 3 critical security fixes, 1 monitoring endpoint

### Sprint 2 (Weeks 2-3): Testing Foundation
- [ ] HP1a: Data tests (3-4 days)
- [ ] HP1b: Model tests (3-4 days)
- [ ] HP1c: MLOps tests (2-3 days)
- [ ] Setup coverage reporting (1 day)

**Deliverables:** 150+ tests, 40%+ coverage

### Sprint 3 (Week 4): API Hardening
- [ ] HP2: API auth & rate limiting (2-3 days)
- [ ] HP3: Observability (2-3 days)

**Deliverables:** Secure API, production monitoring

### Sprint 4 (Week 5): MLOps Maturity
- [ ] HP4a: Configure DVC (1 day)
- [ ] HP4b: A/B testing framework (2-3 days)

**Deliverables:** Safe experimentation framework

---

## Checklist for Success

### Week 1 Done?
- [ ] CORS fixed
- [ ] CI/CD tests blocking
- [ ] Pre-commit hooks installed
- [ ] .env.example created
- [ ] Basic monitoring in place

### End of Month Done?
- [ ] 40%+ test coverage
- [ ] API authentication working
- [ ] Prometheus/Grafana running
- [ ] DVC configured
- [ ] A/B testing framework ready

### Security Checklist
- [ ] CORS restricted
- [ ] API keys/JWT implemented
- [ ] Rate limiting active
- [ ] Audit logging enabled
- [ ] PII redaction in logs
- [ ] Dependencies scanned

### Monitoring Checklist
- [ ] Prometheus metrics exposed
- [ ] Grafana dashboards created
- [ ] Alert handlers activated
- [ ] Health checks working
- [ ] Request tracing enabled
- [ ] Log aggregation setup

---

## Resources Needed

### Development Tools
- `prometheus_client` (already in requirements)
- `slowapi` (add to requirements)
- `python-jose[cryptography]` (add to requirements)
- `pre-commit` (add to requirements-dev)

### Infrastructure
- Prometheus instance (Docker)
- Grafana instance (Docker)
- Optional: Loki for log aggregation
- Optional: Jaeger for distributed tracing

### Team Effort
- 1 senior dev: 4-5 weeks
- 2 mid-level devs: 2-3 weeks
- 3+ junior devs: 1-2 weeks (with guidance)

---

## Success Criteria

Project will be considered "improved" when:

1. **Testing:** ≥40% code coverage, all critical modules tested
2. **Security:** OAuth2/JWT auth, rate limiting, no CORS wildcard
3. **Monitoring:** Prometheus metrics, Grafana dashboards, active alerts
4. **MLOps:** DVC configured, A/B testing framework, model registry
5. **CI/CD:** Tests blocking, pre-commit hooks, security scanning
6. **Code Quality:** <10 cyclomatic complexity per function, type hints

---

## Questions & Support

For each area, reach out to:
- **Testing Framework:** Contact QA lead
- **Security Hardening:** Security team review
- **MLOps Setup:** ML platform team
- **Monitoring:** DevOps/SRE team
- **API Design:** Backend lead

Good luck with the improvements!
