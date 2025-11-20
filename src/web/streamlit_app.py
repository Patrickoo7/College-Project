"""Streamlit web application for heart disease prediction."""

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.predict import HeartDiseasePredictor
from src.utils.config import get_config
from src.utils.gpu_utils import get_gpu_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-positive {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .prediction-negative {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name="random_forest"):
    """Load and cache the prediction model."""
    try:
        models_path = config.get_path("paths.models.artifacts")
        model_path = models_path / f"{model_name}.pkl"

        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None

        predictor = HeartDiseasePredictor(model_path=model_path)
        return predictor
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


def get_available_models():
    """Get list of available models."""
    try:
        models_path = config.get_path("paths.models.artifacts")
        if not models_path.exists():
            return []

        model_files = list(models_path.glob("*.pkl"))
        return [f.stem for f in model_files]
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        return []


def create_gauge_chart(probability, title="Prediction Confidence"):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model selection
        available_models = get_available_models()

        if not available_models:
            st.error("No trained models found. Please train models first.")
            st.stop()

        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0 if "random_forest" not in available_models else available_models.index("random_forest")
        )

        # GPU info
        st.markdown("---")
        st.header("💻 System Info")
        gpu_manager = get_gpu_manager()
        gpu_available, gpu_info = gpu_manager.detect_gpu()

        if gpu_available:
            st.success(f"✅ GPU Available: {gpu_info.get('device_count', 0)} device(s)")
        else:
            st.info("💻 Running on CPU")

        # About section
        st.markdown("---")
        st.header("ℹ️ About")
        st.info(
            "This system uses machine learning to predict the likelihood of heart disease "
            "based on various clinical parameters."
        )

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🏥 Single Prediction", "📊 Batch Prediction", "📈 Model Info"])

    # Load model
    predictor = load_model(selected_model)

    if predictor is None:
        st.error("Failed to load model. Please check the logs.")
        return

    # Tab 1: Single Prediction
    with tab1:
        st.markdown('<h2 class="sub-header">Enter Patient Information</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, step=1)
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
            cp = st.selectbox(
                "Chest Pain Type",
                options=[
                    (0, "Typical Angina"),
                    (1, "Atypical Angina"),
                    (2, "Non-anginal Pain"),
                    (3, "Asymptomatic")
                ],
                format_func=lambda x: f"{x[0]}: {x[1]}"
            )

        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
            restecg = st.selectbox(
                "Resting ECG",
                options=[
                    (0, "Normal"),
                    (1, "ST-T Wave Abnormality"),
                    (2, "Left Ventricular Hypertrophy")
                ],
                format_func=lambda x: f"{x[0]}: {x[1]}"
            )

        with col3:
            thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150, step=1)
            exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        # Predict button
        if st.button("🔍 Predict", type="primary", use_container_width=True):
            try:
                # Make prediction
                with st.spinner("Making prediction..."):
                    result = predictor.predict_single(
                        age=age,
                        sex=sex[1],
                        cp=cp[0],
                        trestbps=trestbps,
                        chol=chol,
                        restecg=restecg[0],
                        thalach=thalach,
                        exang=exang[1],
                        oldpeak=oldpeak
                    )

                # Display results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)

                # Prediction result
                col1, col2 = st.columns([1, 2])

                with col1:
                    if result["prediction"] == 1:
                        st.markdown(
                            '<div class="prediction-box prediction-positive">'
                            '<h3 style="color: #d32f2f;">⚠️ Heart Disease Detected</h3>'
                            '<p>The model predicts a high likelihood of heart disease.</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div class="prediction-box prediction-negative">'
                            '<h3 style="color: #388e3c;">✅ No Heart Disease</h3>'
                            '<p>The model predicts a low likelihood of heart disease.</p>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                with col2:
                    # Probability gauge
                    if result.get("probability"):
                        disease_prob = result["probability"]["disease"]
                        fig = create_gauge_chart(disease_prob, "Disease Probability")
                        st.plotly_chart(fig, use_container_width=True)

                # Detailed probabilities
                if result.get("probability"):
                    st.markdown("### Detailed Probabilities")
                    prob_df = pd.DataFrame([
                        {"Class": "No Disease", "Probability": result["probability"]["no_disease"]},
                        {"Class": "Disease", "Probability": result["probability"]["disease"]}
                    ])

                    fig = px.bar(
                        prob_df,
                        x="Class",
                        y="Probability",
                        color="Class",
                        color_discrete_map={"No Disease": "#66bb6a", "Disease": "#ef5350"}
                    )
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)

                # Input summary
                with st.expander("📋 Input Data Summary"):
                    input_df = pd.DataFrame([result["input_data"]])
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")

    # Tab 2: Batch Prediction
    with tab2:
        st.markdown('<h2 class="sub-header">Upload CSV for Batch Prediction</h2>', unsafe_allow_html=True)

        st.info(
            "Upload a CSV file with the following columns: "
            "age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak"
        )

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)

                st.success(f"✅ Loaded {len(df)} records")

                # Show preview
                with st.expander("📄 Data Preview"):
                    st.dataframe(df.head(10), use_container_width=True)

                # Predict button
                if st.button("🔍 Predict All", type="primary"):
                    with st.spinner("Processing batch prediction..."):
                        # Make predictions
                        results = predictor.predict_batch(df, include_probabilities=True)

                        # Display results
                        st.markdown("---")
                        st.markdown('<h2 class="sub-header">Batch Prediction Results</h2>', unsafe_allow_html=True)

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)

                        disease_count = (results["prediction"] == 1).sum()
                        no_disease_count = (results["prediction"] == 0).sum()
                        total = len(results)

                        col1.metric("Total Patients", total)
                        col2.metric("Disease Detected", disease_count)
                        col3.metric("No Disease", no_disease_count)

                        # Results table
                        st.markdown("### Detailed Results")
                        st.dataframe(results, use_container_width=True)

                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        # Visualization
                        col1, col2 = st.columns(2)

                        with col1:
                            # Pie chart
                            fig = px.pie(
                                values=[disease_count, no_disease_count],
                                names=["Disease", "No Disease"],
                                title="Prediction Distribution",
                                color_discrete_sequence=["#ef5350", "#66bb6a"]
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Probability distribution
                            if "probability_disease" in results.columns:
                                fig = px.histogram(
                                    results,
                                    x="probability_disease",
                                    nbins=20,
                                    title="Disease Probability Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Batch prediction failed: {str(e)}")
                logger.error(f"Batch prediction error: {str(e)}")

    # Tab 3: Model Info
    with tab3:
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)

        try:
            model_info = predictor.get_model_info()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Model Details")
                st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                st.write(f"**Features:** {model_info.get('n_features', 'Unknown')}")
                st.write(f"**Supports Probabilities:** {'Yes' if model_info.get('supports_probabilities') else 'No'}")

            with col2:
                st.markdown("### System Information")
                st.write(f"**Selected Model:** {selected_model}")
                st.write(f"**GPU Enabled:** {'Yes' if config.get('gpu.enabled') else 'No'}")
                st.write(f"**GPU Available:** {'Yes' if gpu_available else 'No'}")

            # Feature importance (if available)
            if hasattr(predictor.model, 'feature_importances_'):
                st.markdown("### Feature Importance")

                importance = predictor.model.feature_importances_
                features = model_info.get("feature_names", [f"Feature {i}" for i in range(len(importance))])

                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)

                fig = px.bar(
                    importance_df.head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Most Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    main()
