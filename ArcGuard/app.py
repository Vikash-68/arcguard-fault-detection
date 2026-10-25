import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st

# Ensure LocationExtractor is available on __main__ for unpickling joblib pipeline
try:
    from train_model import LocationExtractor
    setattr(sys.modules['__main__'], 'LocationExtractor', LocationExtractor)
except ImportError:
    pass

st.set_page_config(
    page_title="ArcGuard - Power-System Fault Classification Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean engineering aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .advisory-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .stAlert {
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load model and metadata
@st.cache_resource
def load_model_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'arcguard_fault_model.joblib')
    metadata_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metadata.json')

    if not os.path.exists(model_path):
        return None, None, f"Model file not found at: {model_path}"
    if not os.path.exists(metadata_path):
        return None, None, f"Metadata file not found at: {metadata_path}"

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return None, None, f"Error deserializing model pipeline: {str(e)}"

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        return None, None, f"Error reading metadata JSON: {str(e)}"

    return model, metadata, None


def main():
    st.title("⚡ ArcGuard Telemetry & Fault Classification Dashboard")
    st.caption("Power Transmission & Distribution Monitoring System • Baseline Classification Interface")

    # Mandatory Engineering Advisory & Diagnostic Limitation
    st.markdown("""
    <div class="advisory-box">
        <h4 style="margin-top:0; color: #b45309;">⚠️ EXPERIMENTAL MODEL NOTICE & OPERATIONAL DISCLAIMER</h4>
        <p style="margin-bottom:6px; font-size:0.95rem; color:#92400e;">
            <strong>Test Performance Notice:</strong> The deployed Random Forest model achieved a test accuracy of 
            <strong>34.31%</strong> and a macro F1-score of <strong>0.3443</strong> on three balanced classes. 
            This performance is near chance (~33.3%). Predictions are experimental baseline outputs and 
            <strong>must NOT</strong> be utilized for operational relaying, safety interlocking, or field dispatch.
        </p>
        <p style="margin-bottom:0; font-size:0.9rem; color:#92400e;">
            <strong>Scope & Sensor Capabilities:</strong> This model operates exclusively on tabular steady-state telemetry. 
            It is <strong>NOT</strong> a partial-discharge (PD) detector and does not capture high-frequency electromagnetic, 
            acoustic, or transient waveform anomalies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load Model and Metadata
    model, metadata, load_error = load_model_artifacts()

    if load_error:
        st.error(f"❌ Failed to initialize dashboard components: {load_error}")
        st.info("Please verify that `train_model.py` was executed and required files exist in `models/`.")
        return

    # Sidebar Navigation & Model Info
    st.sidebar.header("System Configuration")
    st.sidebar.markdown(f"**Candidate Model:** `{metadata.get('selected_model_type', 'Random Forest')}`")
    st.sidebar.markdown(f"**Target Task:** `{metadata.get('target_column', 'Fault Type')}`")
    st.sidebar.markdown(f"**Random Seed:** `{metadata.get('random_seed', 42)}`")
    st.sidebar.markdown(f"**Training Samples:** `{metadata.get('training_samples', 404)}`")
    st.sidebar.markdown(f"**Test Samples:** `{metadata.get('test_samples', 102)}`")

    st.sidebar.divider()
    st.sidebar.markdown("### Target Categories")
    for cls in metadata.get('class_labels', []):
        st.sidebar.markdown(f"- `{cls}`")

    st.sidebar.divider()
    st.sidebar.markdown("### Feature Verification")
    st.sidebar.caption("Required input columns:")
    for f in metadata.get('input_features_raw', []):
        st.sidebar.markdown(f"- `{f}`")

    # Tabs for Workflow
    tab_predict, tab_benchmark, tab_schema = st.tabs([
        "📊 Telemetry Ingestion & Prediction",
        "📈 Model Benchmarks & Confusion Matrices",
        "📋 Data Schema & Diagnostics"
    ])

    # ---------------- TAB 1: Prediction & Ingestion ----------------
    with tab_predict:
        st.subheader("1. Ingest Telemetry Data for Classification")
        st.markdown("Upload a CSV file containing pre-fault transmission telemetry to generate predicted fault categories.")

        uploaded_file = st.file_uploader(
            "Upload Telemetry CSV",
            type=["csv"],
            help="Upload a CSV with the verified input feature columns."
        )

        col_sample, _ = st.columns([1, 4])
        use_sample = col_sample.button("Load Verification Sample Batch")

        df_to_predict = None

        if uploaded_file is not None:
            try:
                # Attempt to parse CSV with UTF-8 first, fallback to latin-1
                try:
                    df_to_predict = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_to_predict = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                st.error(f"❌ Failed to parse uploaded CSV file: {str(e)}")
        elif use_sample:
            sample_path = os.path.join(os.path.dirname(__file__), 'dataset', 'your_actual_dataset_file.csv')
            if os.path.exists(sample_path):
                try:
                    df_sample = pd.read_csv(sample_path, encoding='utf-8')
                    # Take first 10 rows as verification sample
                    df_to_predict = df_sample.head(10).copy()
                    st.success("Loaded 10-row verification sample from dataset.")
                except Exception as e:
                    st.error(f"Failed to load sample dataset: {str(e)}")
            else:
                st.warning("Sample dataset file not found in `dataset/` directory.")

        if df_to_predict is not None:
            st.markdown("#### Input Telemetry Preview")
            st.dataframe(df_to_predict.head(10), use_container_width=True)

            # Validate Columns against metadata schema
            required_inputs = metadata.get('input_features_raw', [])
            
            # Check for column alignment (including common encoding variations for °C)
            col_mapping = {}
            for col in df_to_predict.columns:
                if col.strip() == 'Temperature (C)' and 'Temperature (°C)' in required_inputs and 'Temperature (°C)' not in df_to_predict.columns:
                    col_mapping[col] = 'Temperature (°C)'
            if col_mapping:
                df_to_predict = df_to_predict.rename(columns=col_mapping)
                st.info(f"Mapped column `{list(col_mapping.keys())[0]}` to `{list(col_mapping.values())[0]}`.")

            missing_required = [c for c in required_inputs if c not in df_to_predict.columns]

            if missing_required:
                st.error("❌ Schema Validation Failed: The uploaded CSV does not match the model's required input features.")
                st.markdown(f"**Missing Columns:** `{missing_required}`")
                st.markdown(f"**Uploaded Columns:** `{df_to_predict.columns.tolist()}`")
                st.markdown(f"**Required Columns:** `{required_inputs}`")
            else:
                st.success("✅ All required input features are present and validated.")

                # Extract input feature subset
                X_input = df_to_predict[required_inputs].copy()

                # Generate Predictions
                try:
                    with st.spinner("Executing pipeline preprocessing and inference..."):
                        preds = model.predict(X_input)

                    # Append predictions
                    df_results = df_to_predict.copy()
                    df_results['Predicted Fault Type'] = preds

                    st.markdown("#### Classification Results")
                    st.dataframe(df_results, use_container_width=True)

                    # Summary of predictions
                    st.markdown("##### Prediction Distribution Summary")
                    pred_counts = pd.Series(preds).value_counts().reset_index()
                    pred_counts.columns = ['Predicted Category', 'Count']
                    st.table(pred_counts)

                    # CSV Download Button
                    csv_export = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv_export,
                        file_name="arcguard_predictions.csv",
                        mime="text/csv",
                        key="download_preds"
                    )

                except Exception as e:
                    st.error(f"❌ Error during model execution: {str(e)}")
                    st.exception(e)

    # ---------------- TAB 2: Model Benchmarks & Metrics ----------------
    with tab_benchmark:
        st.subheader("2. Evaluation Metrics & Benchmark Performance")
        st.markdown("Independent evaluation performed on the untouched 20% test fold (102 records).")

        metrics_summary = metadata.get('metrics_summary', {})
        rf_metrics = metrics_summary.get('Random Forest', {})
        dt_metrics = metrics_summary.get('Decision Tree', {})

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Test Accuracy (RF)</div>
            <div class="metric-val">{rf_metrics.get('accuracy', 0.3431):.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Macro F1-Score (RF)</div>
            <div class="metric-val">{rf_metrics.get('macro_f1', 0.3443):.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Macro Precision (RF)</div>
            <div class="metric-val">{rf_metrics.get('macro_precision', 0.3462):.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Macro Recall (RF)</div>
            <div class="metric-val">{rf_metrics.get('macro_recall', 0.3433):.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Candidate Model Comparison Table")
        comparison_data = [
            {
                "Model Architecture": "Random Forest (Selected)",
                "Accuracy": f"{rf_metrics.get('accuracy', 0.3431):.4f}",
                "Macro Precision": f"{rf_metrics.get('macro_precision', 0.3462):.4f}",
                "Macro Recall": f"{rf_metrics.get('macro_recall', 0.3433):.4f}",
                "Macro F1-Score": f"{rf_metrics.get('macro_f1', 0.3443):.4f}",
                "Status": "Candidate Model Saved"
            },
            {
                "Model Architecture": "Decision Tree",
                "Accuracy": f"{dt_metrics.get('accuracy', 0.3235):.4f}",
                "Macro Precision": f"{dt_metrics.get('macro_precision', 0.3278):.4f}",
                "Macro Recall": f"{dt_metrics.get('macro_recall', 0.3249):.4f}",
                "Macro F1-Score": f"{dt_metrics.get('macro_f1', 0.3244):.4f}",
                "Status": "Evaluated Baseline"
            }
        ]
        st.table(pd.DataFrame(comparison_data))

        st.markdown("#### Test Confusion Matrices")
        cm_combined_path = os.path.join(os.path.dirname(__file__), 'results', 'combined_confusion_matrix.png')
        cm_rf_path = os.path.join(os.path.dirname(__file__), 'results', 'random_forest_confusion_matrix.png')

        if os.path.exists(cm_combined_path):
            st.image(cm_combined_path, caption="Combined Test Confusion Matrices (Decision Tree vs Random Forest)", use_container_width=True)
        elif os.path.exists(cm_rf_path):
            st.image(cm_rf_path, caption="Random Forest Test Confusion Matrix", use_container_width=True)
        else:
            st.warning("Confusion matrix plot file not found in `results/`.")

        st.markdown("#### Full Evaluation Report")
        report_file_path = os.path.join(os.path.dirname(__file__), 'results', 'model_report.txt')
        if os.path.exists(report_file_path):
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.text_area("model_report.txt", report_content, height=280)
        else:
            st.info("Full evaluation report not found at `results/model_report.txt`.")

    # ---------------- TAB 3: Data Schema & Diagnostics ----------------
    with tab_schema:
        st.subheader("3. Engineering Diagnostics & Data Schema")
        st.markdown("Overview of feature pipeline configurations, data leakage controls, and physical limitations.")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Pre-Fault Input Features (Accepted)")
            st.markdown("""
            - `Voltage (V)`: Continuous steady-state line voltage
            - `Current (A)`: Continuous conductor current
            - `Power Load (MW)`: Feeder active power consumption
            - `Temperature (°C)`: Local ambient temperature
            - `Wind Speed (km/h)`: Local ambient wind velocity
            - `Weather Condition`: Categorical (`Clear`, `Rainy`, `Snowy`, `Thunderstorm`, `Windstorm`)
            - `Fault Location (Latitude, Longitude)`: Geographic coordinate pair string `(lat, lon)`
            """)

            st.markdown("##### Pipeline Preprocessing Steps")
            st.markdown("""
            1. **Numerical Transform**: `SimpleImputer(median)` followed by `StandardScaler`
            2. **Categorical Transform**: `SimpleImputer(most_frequent)` followed by `OneHotEncoder(handle_unknown='ignore')`
            3. **Coordinate Transform**: `LocationExtractor` (extracts floats) followed by `SimpleImputer(median)` and `StandardScaler`
            """)

        with col_right:
            st.markdown("##### Excluded Columns (Strict Leakage Prevention)")
            st.markdown("""
            - `Fault ID`: Excluded to avoid memorization of arbitrary index records.
            - `Duration of Fault (hrs)`: Excluded because fault duration is measurable only **after** clearance.
            - `Down time (hrs)`: Excluded because outage downtime is recorded **after** grid restoration.
            - `Maintenance Status`: Excluded because repair logs are updated **post-incident**.
            - `Component Health`: Excluded because inspection ratings reflect post-fault component state.
            """)

            st.markdown("##### Physical & Engineering Limitations")
            st.markdown("""
            - **Tabular Steady-State vs Transient Events**: Steady-state power values (RMS V, RMS I, MW) exhibit nearly 
              identical mean levels across different fault types without microsecond-scale transient waveforms.
            - **No Partial Discharge Signal**: Partial discharge occurs at high frequencies (MHz/GHz) or as ultrasonic 
              acoustic impulses. The current dataset contains no high-frequency or acoustic channels.
            - **Splitting Structure**: While stratified random splitting preserves class balance, field deployments 
              require chronological or substation-clustered splits to account for seasonal and topological shifts.
            """)

if __name__ == '__main__':
    main()
