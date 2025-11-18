"""Model card generation for ML model documentation."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCard:
    """
    Generate model cards following best practices for ML documentation.

    Model cards provide transparency about model development, intended use,
    limitations, and performance characteristics.
    """

    def __init__(self):
        """Initialize model card generator."""
        self.config = Config()
        self.docs_dir = Path("docs/model_cards")
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_card(
        self,
        model_name: str,
        model_details: Dict[str, Any],
        intended_use: Dict[str, Any],
        factors: Dict[str, Any],
        metrics: Dict[str, Any],
        evaluation_data: Dict[str, Any],
        training_data: Dict[str, Any],
        ethical_considerations: Optional[Dict[str, Any]] = None,
        caveats_recommendations: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive model card.

        Args:
            model_name: Name of the model
            model_details: Technical details about the model
            intended_use: Intended use cases
            factors: Relevant factors for model performance
            metrics: Performance metrics
            evaluation_data: Evaluation dataset information
            training_data: Training dataset information
            ethical_considerations: Ethical considerations
            caveats_recommendations: Caveats and recommendations

        Returns:
            Path to generated model card
        """
        logger.info(f"Generating model card for {model_name}...")

        card_content = self._build_model_card_content(
            model_name=model_name,
            model_details=model_details,
            intended_use=intended_use,
            factors=factors,
            metrics=metrics,
            evaluation_data=evaluation_data,
            training_data=training_data,
            ethical_considerations=ethical_considerations or {},
            caveats_recommendations=caveats_recommendations or {}
        )

        # Save as Markdown
        output_path = self.docs_dir / f"{model_name}_model_card.md"
        with open(output_path, "w") as f:
            f.write(card_content)

        # Save as JSON
        json_path = self.docs_dir / f"{model_name}_model_card.json"
        card_dict = {
            "model_name": model_name,
            "model_details": model_details,
            "intended_use": intended_use,
            "factors": factors,
            "metrics": metrics,
            "evaluation_data": evaluation_data,
            "training_data": training_data,
            "ethical_considerations": ethical_considerations,
            "caveats_recommendations": caveats_recommendations,
            "generated_date": datetime.now().isoformat()
        }

        with open(json_path, "w") as f:
            json.dump(card_dict, f, indent=2)

        logger.info(f"Saved model card to {output_path}")

        return str(output_path)

    def _build_model_card_content(
        self,
        model_name: str,
        model_details: Dict,
        intended_use: Dict,
        factors: Dict,
        metrics: Dict,
        evaluation_data: Dict,
        training_data: Dict,
        ethical_considerations: Dict,
        caveats_recommendations: Dict
    ) -> str:
        """Build model card content in Markdown format."""

        content = f"""# Model Card: {model_name}

## Model Details

**Model Name:** {model_name}
**Model Type:** {model_details.get('type', 'N/A')}
**Model Version:** {model_details.get('version', '1.0')}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Developer:** {model_details.get('developer', 'Heart Disease Prediction Team')}
**License:** {model_details.get('license', 'MIT')}

### Model Description

{model_details.get('description', 'Machine learning model for heart disease prediction.')}

### Model Architecture

- **Algorithm:** {model_details.get('algorithm', 'N/A')}
- **Framework:** {model_details.get('framework', 'scikit-learn')}
- **Parameters:** {model_details.get('num_parameters', 'N/A')}

### Training Information

- **Training Date:** {model_details.get('training_date', 'N/A')}
- **Training Duration:** {model_details.get('training_duration', 'N/A')}
- **Hardware:** {model_details.get('hardware', 'CPU/GPU')}

## Intended Use

### Primary Intended Uses

{intended_use.get('primary_uses', 'Heart disease risk assessment and prediction.')}

### Primary Intended Users

{intended_use.get('intended_users', 'Healthcare professionals, researchers, and medical decision support systems.')}

### Out-of-Scope Uses

{intended_use.get('out_of_scope', """
- Definitive diagnosis without medical professional oversight
- Treatment decisions without additional clinical evaluation
- Use on populations significantly different from training data
""")}

## Factors

### Relevant Factors

{factors.get('relevant_factors', """
- **Age:** Patient age affects model predictions
- **Sex:** Gender-specific patterns in heart disease
- **Medical History:** Prior conditions and measurements
- **Geographic Location:** Dataset includes multiple regions
""")}

### Evaluation Factors

{factors.get('evaluation_factors', """
Models evaluated across:
- Age groups (young, middle-aged, elderly)
- Gender (male, female)
- Dataset source (Cleveland, Hungarian, Switzerland, VA Long Beach)
""")}

## Metrics

### Model Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A')} |
| Precision | {metrics.get('precision', 'N/A')} |
| Recall | {metrics.get('recall', 'N/A')} |
| F1 Score | {metrics.get('f1', 'N/A')} |
| ROC-AUC | {metrics.get('roc_auc', 'N/A')} |
| Specificity | {metrics.get('specificity', 'N/A')} |
| Sensitivity | {metrics.get('sensitivity', 'N/A')} |

### Decision Thresholds

- **Default Threshold:** {metrics.get('threshold', 0.5)}
- **High Sensitivity Threshold:** {metrics.get('high_sensitivity_threshold', 0.3)}
- **High Specificity Threshold:** {metrics.get('high_specificity_threshold', 0.7)}

## Training Data

### Datasets

{training_data.get('datasets', """
- **Cleveland:** Heart disease database from Cleveland Clinic Foundation
- **Hungarian:** Institute of Cardiology, University Hospital Zurich, Switzerland
- **Switzerland:** University Hospital, Zurich, Switzerland
- **VA Long Beach:** V.A. Medical Center, Long Beach, CA
""")}

### Data Preprocessing

{training_data.get('preprocessing', """
- Missing value imputation
- Outlier detection and handling
- Feature scaling and normalization
- Categorical encoding
""")}

### Dataset Size

- **Total Samples:** {training_data.get('total_samples', 'N/A')}
- **Training Samples:** {training_data.get('training_samples', 'N/A')}
- **Validation Samples:** {training_data.get('validation_samples', 'N/A')}
- **Test Samples:** {training_data.get('test_samples', 'N/A')}

## Evaluation Data

### Dataset

{evaluation_data.get('dataset', 'Hold-out test set from UCI Heart Disease datasets')}

### Motivation

{evaluation_data.get('motivation', 'Evaluate model performance on unseen data from the same distribution')}

### Preprocessing

{evaluation_data.get('preprocessing', 'Same preprocessing steps as training data')}

## Ethical Considerations

{ethical_considerations.get('considerations', """
### Fairness

- Model performance should be monitored across different demographic groups
- Regular audits for bias in predictions across age, sex, and ethnicity

### Privacy

- Model trained on de-identified medical data
- No patient identifiable information stored or used
- Predictions should be handled according to HIPAA guidelines

### Use Cases

- Model is a decision support tool, not a replacement for medical diagnosis
- Should be used in conjunction with clinical judgment
- Not suitable for emergency medical decisions

### Transparency

- Model architecture and performance metrics are publicly documented
- Limitations and failure modes are clearly communicated
""")}

## Caveats and Recommendations

### Known Limitations

{caveats_recommendations.get('limitations', """
- Model trained on historical data from specific medical centers
- Performance may vary on populations not represented in training data
- Does not account for all possible risk factors
- Limited to features available in UCI dataset
""")}

### Recommendations

{caveats_recommendations.get('recommendations', """
- Use as part of comprehensive medical assessment
- Regular model retraining with updated data
- Monitor model performance in production
- Validate predictions with clinical expertise
- Consider local population characteristics when deploying
""")}

### Failure Modes

{caveats_recommendations.get('failure_modes', """
- May underperform on edge cases or rare conditions
- Sensitive to data quality and missing values
- May not generalize to significantly different populations
- Limited by feature set available in training data
""")}

## Model Maintenance

### Monitoring

- Continuous monitoring of prediction accuracy
- Data drift detection on input features
- Performance degradation alerts

### Retraining Schedule

- Scheduled retraining: {model_details.get('retraining_schedule', 'Monthly')}
- Triggered retraining: When performance drops below {model_details.get('retraining_threshold', '85%')} accuracy

### Version Control

- Model versions tracked with MLflow
- All experiments logged and reproducible
- Previous versions maintained for rollback

## Contact Information

**Maintainer:** {model_details.get('maintainer', 'ML Engineering Team')}
**Email:** {model_details.get('contact_email', 'ml-team@example.com')}
**Repository:** {model_details.get('repository', 'https://github.com/user/heart-disease-prediction')}

## References

{model_details.get('references', """
1. UCI Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease
2. Model Cards for Model Reporting: https://arxiv.org/abs/1810.03993
""")}

---

*This model card was automatically generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return content


def generate_heart_disease_model_card(model_name: str, metrics: Dict[str, float]) -> str:
    """Generate model card for heart disease prediction model."""

    generator = ModelCard()

    model_details = {
        "type": "Binary Classification",
        "version": "1.0",
        "developer": "Heart Disease Prediction Team",
        "license": "MIT",
        "description": f"""
This {model_name} model predicts the presence of heart disease based on patient medical data.
It uses 9 clinical features including age, sex, chest pain type, blood pressure, cholesterol,
ECG results, maximum heart rate, exercise-induced angina, and ST depression.
        """,
        "algorithm": model_name,
        "framework": "scikit-learn / XGBoost / LightGBM / CatBoost",
        "training_date": datetime.now().strftime('%Y-%m-%d'),
        "hardware": "CPU/GPU with automatic detection",
        "retraining_schedule": "Monthly or when accuracy drops below 85%",
        "retraining_threshold": "85%",
        "maintainer": "ML Engineering Team",
        "repository": "https://github.com/user/College-Project"
    }

    intended_use = {
        "primary_uses": """
- Clinical decision support for heart disease risk assessment
- Screening tool for identifying high-risk patients
- Research and educational purposes
        """,
        "intended_users": """
- Healthcare professionals (doctors, nurses, medical staff)
- Medical researchers
- Healthcare administrators
- Clinical decision support systems
        """,
        "out_of_scope": """
- Definitive medical diagnosis without professional oversight
- Emergency medical decisions
- Treatment planning without additional clinical evaluation
- Use on pediatric populations (model trained on adult data)
- Populations significantly different from training data demographics
        """
    }

    factors = {
        "relevant_factors": """
- **Age:** Strong predictor, model calibrated for ages 29-77
- **Sex:** Gender-specific patterns considered
- **Medical Measurements:** Blood pressure, cholesterol, heart rate
- **Clinical Tests:** ECG results, exercise stress tests
- **Symptoms:** Chest pain type, exercise-induced angina
        """,
        "evaluation_factors": """
- Age groups: <40, 40-60, >60
- Gender: Male, Female
- Dataset source: Cleveland, Hungarian, Switzerland, VA
- Disease severity: Binary (presence/absence)
        """
    }

    evaluation_data = {
        "dataset": "UCI Heart Disease Dataset - Hold-out Test Set",
        "motivation": "Evaluate generalization on unseen data from same distribution",
        "preprocessing": "Consistent with training data preprocessing"
    }

    training_data = {
        "datasets": """
Combined UCI Heart Disease datasets:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- University Hospital Zurich, Switzerland
- V.A. Medical Center, Long Beach, CA
        """,
        "preprocessing": """
- Missing value imputation using median/mode
- Outlier detection using IQR method
- Feature scaling with StandardScaler
- One-hot encoding for categorical variables
- Feature engineering for domain-specific features
        """,
        "total_samples": "920 (approximate)",
        "training_samples": "736 (80%)",
        "validation_samples": "92 (10%)",
        "test_samples": "92 (10%)"
    }

    ethical_considerations = {
        "considerations": """
### Fairness and Bias

- Model evaluated across demographic groups
- Performance monitored for disparate impact
- Regular bias audits conducted
- Fairness metrics computed across protected attributes

### Privacy and Security

- Trained on de-identified patient data
- No PHI (Protected Health Information) stored
- HIPAA compliance considerations in deployment
- Secure model serving infrastructure

### Clinical Decision Support

- Explicitly NOT a diagnostic tool
- Requires medical professional oversight
- Part of comprehensive patient assessment
- Transparent about confidence and uncertainty

### Societal Impact

- May improve early detection of heart disease
- Could reduce healthcare disparities with proper deployment
- Potential for automation bias must be mitigated
- Requires ongoing validation in diverse populations
        """
    }

    caveats_recommendations = {
        "limitations": """
- Historical data may not reflect current medical practices
- Limited to 9 clinical features from UCI dataset
- Does not capture all heart disease risk factors
- Performance may vary across different populations
- Lacks information on medication history, family history, lifestyle factors
        """,
        "recommendations": """
- Always use with clinical judgment
- Validate on local population before deployment
- Monitor performance continuously in production
- Retrain periodically with updated data
- Use appropriate decision thresholds for use case
- Combine with other clinical assessments
- Document all prediction decisions
        """,
        "failure_modes": """
- Missing or invalid input features
- Out-of-distribution inputs (ages, values outside training range)
- Data quality issues (measurement errors, data entry mistakes)
- Concept drift (changing patient populations or medical practices)
- Adversarial inputs or data poisoning
        """
    }

    card_path = generator.generate_model_card(
        model_name=model_name,
        model_details=model_details,
        intended_use=intended_use,
        factors=factors,
        metrics=metrics,
        evaluation_data=evaluation_data,
        training_data=training_data,
        ethical_considerations=ethical_considerations,
        caveats_recommendations=caveats_recommendations
    )

    return card_path


if __name__ == "__main__":
    # Example: Generate model card for Random Forest
    metrics = {
        "accuracy": "0.8500",
        "precision": "0.8300",
        "recall": "0.8700",
        "f1": "0.8500",
        "roc_auc": "0.9000",
        "specificity": "0.8200",
        "sensitivity": "0.8700"
    }

    card_path = generate_heart_disease_model_card("random_forest", metrics)
    print(f"Generated model card: {card_path}")
