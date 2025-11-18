# Security Guidelines

## Overview

This document outlines the security measures, best practices, and guidelines for the Heart Disease Prediction System.

---

## 🔒 Security Measures Implemented

### 1. Input Validation

#### API Endpoints
- **File uploads**: Type, size, and row limits enforced
- **Model names**: Alphanumeric validation to prevent path traversal
- **Patient data**: Pydantic schema validation

#### Database Queries
- **Parameterized queries**: All SQL uses placeholders
- **Table name whitelist**: Only approved tables accessible
- **Input sanitization**: All user inputs validated

### 2. Authentication & Authorization

**Current State:** No authentication (development mode)

**Production Recommendations:**
```python
# Add API key authentication
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@router.post("/predict")
async def predict(patient: PatientInput, api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    ...
```

**Or use OAuth2:**
```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/predict")
async def predict(patient: PatientInput, token: str = Depends(oauth2_scheme)):
    user = verify_token(token)
    ...
```

### 3. Rate Limiting

**Not Implemented** - Recommended for production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, patient: PatientInput):
    ...
```

---

## 🛡️ Security Best Practices

### For Developers

1. **Never trust user input**
   - Validate all inputs
   - Sanitize data before processing
   - Use type hints and Pydantic models

2. **Use parameterized queries**
   ```python
   # ✅ GOOD
   cursor.execute("SELECT * FROM table WHERE id = ?", (user_id,))

   # ❌ BAD
   cursor.execute(f"SELECT * FROM table WHERE id = {user_id}")
   ```

3. **Validate file uploads**
   - Check file type
   - Limit file size
   - Scan for malicious content
   - Use antivirus for production

4. **Secure file operations**
   ```python
   # ✅ GOOD - Validate path
   if not re.match(r'^[a-zA-Z0-9_-]+$', filename):
       raise ValueError("Invalid filename")
   path = safe_dir / filename

   # ❌ BAD - Path traversal vulnerability
   path = base_dir / user_filename
   ```

5. **Error handling**
   ```python
   # ✅ GOOD - Generic error message
   except Exception as e:
       logger.error(f"Error: {e}")
       raise HTTPException(500, "Internal server error")

   # ❌ BAD - Exposes stack trace
   except Exception as e:
       raise HTTPException(500, str(e))
   ```

### For Deployment

1. **Environment Variables**
   ```bash
   # .env file (never commit!)
   API_KEY=your-secret-key-here
   DATABASE_URL=postgresql://user:pass@localhost/db
   SECRET_KEY=your-secret-key
   ```

2. **HTTPS Only**
   - Use TLS/SSL certificates
   - Redirect HTTP to HTTPS
   - Set HSTS headers

3. **CORS Configuration**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],  # Not "*"
       allow_credentials=True,
       allow_methods=["GET", "POST"],
       allow_headers=["*"],
   )
   ```

4. **Security Headers**
   ```python
   @app.middleware("http")
   async def add_security_headers(request: Request, call_next):
       response = await call_next(request)
       response.headers["X-Content-Type-Options"] = "nosniff"
       response.headers["X-Frame-Options"] = "DENY"
       response.headers["X-XSS-Protection"] = "1; mode=block"
       response.headers["Strict-Transport-Security"] = "max-age=31536000"
       return response
   ```

---

## 🔐 Data Protection

### PHI/PII Handling

**Important:** This system processes Protected Health Information (PHI).

#### HIPAA Compliance Requirements

1. **Data Encryption**
   - Encrypt data at rest
   - Encrypt data in transit (HTTPS)
   - Use encrypted database connections

2. **Access Control**
   - Role-based access control (RBAC)
   - Audit logging
   - Minimum necessary access

3. **Data Retention**
   - Define retention policies
   - Secure deletion procedures
   - Backup encryption

#### Implementation

```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

key = os.getenv("ENCRYPTION_KEY")
cipher = Fernet(key)

# Encrypt before storing
encrypted_data = cipher.encrypt(patient_data.encode())

# Decrypt when retrieving
decrypted_data = cipher.decrypt(encrypted_data).decode()
```

### Database Security

```python
# Use environment variables for credentials
DATABASE_URL = os.getenv("DATABASE_URL")

# Enable SSL for database connections
engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"}
)

# Use read-only connections when possible
readonly_engine = create_engine(
    DATABASE_URL,
    connect_args={"options": "-c default_transaction_read_only=on"}
)
```

---

## 🚨 Incident Response

### Security Incident Checklist

1. **Immediate Actions**
   - [ ] Isolate affected systems
   - [ ] Preserve logs and evidence
   - [ ] Notify security team
   - [ ] Document timeline

2. **Investigation**
   - [ ] Identify attack vector
   - [ ] Assess data exposure
   - [ ] Review access logs
   - [ ] Check for backdoors

3. **Remediation**
   - [ ] Patch vulnerabilities
   - [ ] Reset compromised credentials
   - [ ] Update firewall rules
   - [ ] Deploy fixes

4. **Post-Incident**
   - [ ] Conduct post-mortem
   - [ ] Update security procedures
   - [ ] Train team members
   - [ ] Improve monitoring

### Logging for Security

```python
import logging

# Security-specific logger
security_logger = logging.getLogger("security")

# Log authentication attempts
security_logger.info(f"Login attempt: user={username}, ip={ip_address}")

# Log data access
security_logger.info(f"Patient data accessed: id={patient_id}, user={user_id}")

# Log suspicious activity
security_logger.warning(f"Multiple failed login attempts: ip={ip_address}")

# Log security events
security_logger.error(f"SQL injection attempt detected: query={query}")
```

---

## 🔍 Security Monitoring

### Metrics to Monitor

1. **Failed Authentication Attempts**
   - Track failed logins
   - Alert on brute force patterns
   - Block after threshold

2. **Unusual API Usage**
   - Spike in requests
   - Unusual endpoints accessed
   - Large file uploads

3. **Database Activity**
   - Failed queries
   - Slow queries (possible DoS)
   - Unusual data access patterns

### Alerting Rules

```python
# Example: Alert on suspicious activity
if failed_login_attempts > 5:
    send_alert("Multiple failed login attempts", level="high")

if request_rate > 100:
    send_alert("High request rate detected", level="medium")

if upload_size > 50_000_000:
    send_alert("Large file upload", level="medium")
```

---

## 📋 Security Checklist

### Development
- [ ] Input validation on all endpoints
- [ ] Parameterized SQL queries
- [ ] Error handling without information leakage
- [ ] Secure file operations
- [ ] Type hints and validation
- [ ] Security code review

### Testing
- [ ] Penetration testing
- [ ] SQL injection testing
- [ ] XSS testing
- [ ] CSRF testing
- [ ] File upload security testing
- [ ] Authentication bypass testing

### Deployment
- [ ] HTTPS enabled
- [ ] Security headers configured
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] API authentication enabled
- [ ] Database encryption enabled
- [ ] Secrets in environment variables
- [ ] Audit logging enabled
- [ ] Monitoring configured
- [ ] Backup procedures tested

### Maintenance
- [ ] Regular security updates
- [ ] Dependency vulnerability scanning
- [ ] Log review
- [ ] Access control review
- [ ] Incident response plan tested
- [ ] Security training for team

---

## 🔧 Tools & Resources

### Security Scanning

```bash
# Dependency vulnerability scanning
pip install safety
safety check

# Code security analysis
pip install bandit
bandit -r src/

# SAST scanning
pip install semgrep
semgrep --config=auto src/
```

### Secrets Detection

```bash
# Detect secrets in code
pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline

# GitGuardian for git history
pip install ggshield
ggshield secret scan repo .
```

### Penetration Testing

```bash
# OWASP ZAP
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t http://localhost:8000

# SQLMap for SQL injection testing
sqlmap -u "http://localhost:8000/api/v1/predict?model=test" --batch

# Burp Suite Community Edition
# Download from: https://portswigger.net/burp/communitydownload
```

---

## 📞 Contact & Reporting

### Security Issues

**DO NOT** create public GitHub issues for security vulnerabilities.

Instead:
1. Email: security@yourproject.com
2. Use GitHub Security Advisories
3. Contact project maintainer directly

### Bug Bounty (If Applicable)

We welcome responsible disclosure:
- Severity levels: Critical, High, Medium, Low
- Response time: 24-48 hours
- Remediation timeline: Based on severity

---

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE Top 25](https://cwe.mitre.org/top25/archive/2023/2023_top25_list.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

**Last Updated:** 2024-01-18
**Version:** 1.0
**Maintained By:** Security Team
