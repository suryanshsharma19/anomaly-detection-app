# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support
import joblib
import os
import datetime
import json
import logging
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import yaml
from typing import Dict, List, Optional, Union
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('anomaly_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('AnomalyApp')

# --- Configuration Management ---
class Config:
    def __init__(self):
        self.config_path = Path('config.yaml')
        self.default_config = {
            'model': {
                'train_points': 100,
                'contamination': 'auto',
                'model_prefix': 'iforest_model_gui',
                'n_estimators': 100,
                'random_state': 42
            },
            'ui': {
                'theme': 'dark',
                'chart_height': 500,
                'max_display_points': 1000
            },
            'data': {
                'timestamp_col': 'timestamp',
                'supported_formats': ['.csv', '.xlsx', '.parquet']
            }
        }
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self.default_config

    def save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

config = Config()

# --- Helper Functions ---
def save_model(model, filename: str) -> bool:
    """Saves the trained Isolation Forest model with metadata."""
    try:
        model_metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'features': st.session_state.feature_names,
            'config': config.config['model']
        }
        
        save_data = {
            'model': model,
            'metadata': model_metadata
        }
        
        joblib.dump(save_data, filename)
        logger.info(f"Model saved to '{filename}'")
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False

def load_model(filename: str) -> Optional[tuple]:
    """Loads a pre-trained Isolation Forest model with metadata."""
    if os.path.exists(filename):
        try:
            save_data = joblib.load(filename)
            model = save_data['model']
            metadata = save_data['metadata']
            logger.info(f"Loaded model from '{filename}' trained on {metadata['timestamp']}")
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    return None

def format_features(features_dict: Dict) -> str:
    """Formats features dictionary for display with improved formatting."""
    if not isinstance(features_dict, dict):
        return str(features_dict)
    
    items = []
    for k, v in sorted(features_dict.items()):
        if isinstance(v, float):
            items.append(f"{k}: {v:.4f}")
        elif isinstance(v, (int, str)):
            items.append(f"{k}: {v}")
        else:
            items.append(f"{k}: {str(v)}")
    return ", ".join(items)

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generates a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

def plot_anomalies(df: pd.DataFrame, anomalies: List[Dict], feature_cols: List[str]) -> None:
    """Creates interactive visualizations of anomalies."""
    if not anomalies:
        st.warning("No anomalies to visualize")
        return

    # Create a copy of the dataframe for visualization
    viz_df = df.copy()
    viz_df['is_anomaly'] = False
    
    # Mark anomalies in the dataframe
    for anomaly in anomalies:
        timestamp = anomaly['Timestamp']
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        mask = (viz_df[config.config['data']['timestamp_col']] == timestamp)
        viz_df.loc[mask, 'is_anomaly'] = True

    # Time series plot
    st.subheader("Time Series Analysis")
    fig = go.Figure()
    
    for col in feature_cols:
        fig.add_trace(go.Scatter(
            x=viz_df[config.config['data']['timestamp_col']],
            y=viz_df[col],
            name=col,
            mode='lines'
        ))
    
    # Add anomaly markers
    anomaly_df = viz_df[viz_df['is_anomaly']]
    fig.add_trace(go.Scatter(
        x=anomaly_df[config.config['data']['timestamp_col']],
        y=[max(viz_df[col].max() for col in feature_cols)] * len(anomaly_df),
        mode='markers',
        name='Anomalies',
        marker=dict(
            color='red',
            size=10,
            symbol='x'
        )
    ))
    
    fig.update_layout(
        height=config.config['ui']['chart_height'],
        title="Feature Values Over Time with Anomaly Markers",
        xaxis_title="Time",
        yaxis_title="Value"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature distribution plot
    st.subheader("Feature Distributions")
    for col in feature_cols:
        fig = px.histogram(
            viz_df,
            x=col,
            color='is_anomaly',
            title=f"Distribution of {col}",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Main Anomaly Detection Logic ---
def detect_anomalies_in_df(df: pd.DataFrame, feature_columns: List[str], 
                          timestamp_col: str, contamination: Union[str, float],
                          train_points: int, model_filename_prefix: str) -> List[Dict]:
    """Enhanced anomaly detection with performance metrics and visualization."""
    # Initialize session state
    if 'iforest_model' not in st.session_state:
        st.session_state.iforest_model = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = []
    if 'points_processed' not in st.session_state:
        st.session_state.points_processed = 0
    if 'detected_anomalies' not in st.session_state:
        st.session_state.detected_anomalies = []
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}

    # Preprocess the data
    df_processed = df[feature_columns].copy()
    # Normalize the features
    for col in feature_columns:
        mean = df_processed[col].mean()
        std = df_processed[col].std()
        if std > 0:  # Avoid division by zero
            df_processed[col] = (df_processed[col] - mean) / std

    model_filename = f"{model_filename_prefix}_{'_'.join(sorted(feature_columns))}.joblib"

    # Attempt to load model
    if st.session_state.iforest_model is None and st.session_state.points_processed == 0:
        loaded_data = load_model(model_filename)
        if loaded_data:
            st.session_state.iforest_model, metadata = loaded_data
            st.session_state.feature_names = metadata['features']
            st.success("Using pre-trained model from " + metadata['timestamp'])

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.session_state.detected_anomalies = []

    # Process data
    for index, row in df_processed.iterrows():
        current_progress = (index + 1) / len(df_processed)
        progress_bar.progress(current_progress)

        # Feature extraction
        timestamp = df.iloc[index].get(timestamp_col, datetime.datetime.now().isoformat())
        features = {}
        valid_row = True
        
        for col in feature_columns:
            try:
                features[col] = float(row[col])
            except (ValueError, TypeError):
                status_text.warning(f"Row {index+1}: Non-numeric data in '{col}'. Skipping.")
                valid_row = False
                break

        if not valid_row:
            continue

        current_features_vector = [features[name] for name in feature_columns]

        # Training or Prediction
        if st.session_state.iforest_model is None:
            status_text.text(f"Training: {st.session_state.points_processed+1}/{train_points}")
            st.session_state.training_data.append(current_features_vector)
            st.session_state.points_processed += 1

            if st.session_state.points_processed >= train_points:
                status_text.text("Training model...")
                try:
                    # Use a more conservative contamination rate
                    contam_val = 0.1  # Default to 10% if not specified
                    if contamination != 'auto':
                        try:
                            contam_val = float(contamination)
                            if not (0 < contam_val < 0.5):
                                st.warning("Invalid contamination. Using 0.1.")
                                contam_val = 0.1
                        except ValueError:
                            st.warning("Invalid contamination. Using 0.1.")
                            contam_val = 0.1

                    trained_model = IsolationForest(
                        n_estimators=config.config['model']['n_estimators'],
                        contamination=contam_val,
                        random_state=config.config['model']['random_state']
                    ).fit(np.array(st.session_state.training_data))

                    st.session_state.iforest_model = trained_model
                    st.session_state.feature_names = feature_columns
                    save_model(trained_model, model_filename)
                    st.session_state.training_data = []
                except Exception as e:
                    status_text.error(f"Training failed: {e}")
                    st.session_state.points_processed = 0
                    st.session_state.training_data = []
                    st.session_state.iforest_model = None
                    break
        else:
            status_text.text(f"Analyzing: {index+1}/{len(df_processed)}")
            try:
                value_array = np.array([current_features_vector])
                prediction = st.session_state.iforest_model.predict(value_array)
                score = st.session_state.iforest_model.score_samples(value_array)[0]

                if prediction[0] == -1:
                    anomaly_info = {
                        "Timestamp": timestamp,
                        "IForest Score": score,
                        "Features": features,
                        "Row Index": index
                    }
                    st.session_state.detected_anomalies.append(anomaly_info)
            except Exception as e:
                status_text.error(f"Prediction failed: {e}")

    # Calculate performance metrics
    if st.session_state.iforest_model is not None:
        y_true = [1 if i in [a['Row Index'] for a in st.session_state.detected_anomalies] else 0 
                 for i in range(len(df_processed))]
        y_pred = st.session_state.iforest_model.predict(df_processed[feature_columns])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        st.session_state.performance_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return st.session_state.detected_anomalies

# --- Streamlit UI ---
st.set_page_config(
    page_title="Advanced Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 Anomaly Detection")
    st.markdown("""
    ### About
    This advanced anomaly detection system uses Isolation Forest algorithm to identify unusual patterns in your data.
    
    ### Features
    - Real-time anomaly detection
    - Interactive visualizations
    - Model performance metrics
    - Export capabilities
    - Configurable parameters
    """)

# Main content
st.title("Advanced Anomaly Detection System")
st.markdown("""
    Upload your data file and configure the detection parameters to identify anomalies in your dataset.
    The system supports CSV, Excel, and Parquet file formats.
""")

# File upload
uploaded_file = st.file_uploader(
    "Choose a data file",
    type=config.config['data']['supported_formats']
)

if uploaded_file is not None:
    try:
        # Read file based on extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == '.xlsx':
            df = pd.read_excel(uploaded_file)
        elif file_extension == '.parquet':
            df = pd.read_parquet(uploaded_file)
        
        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        available_columns = list(df.columns)

        # Configuration
        st.subheader("Configuration")
        col1, col2 = st.columns(2)

        with col1:
            feature_cols = st.multiselect(
                "Select Feature Columns",
                options=available_columns,
                help="Choose numerical columns for anomaly detection"
            )
            
            ts_col_options = ["(Generate Timestamps)"] + available_columns
            selected_ts_col = st.selectbox(
                "Timestamp Column",
                options=ts_col_options,
                index=0
            )
            timestamp_col_name = selected_ts_col if selected_ts_col != "(Generate Timestamps)" else None

        with col2:
            train_points = st.number_input(
                "Training Points",
                min_value=20,
                value=config.config['model']['train_points'],
                step=10
            )
            
            contamination_str = st.text_input(
                "Contamination Rate",
                value=str(config.config['model']['contamination']),
                help="Expected proportion of anomalies (0.0 to 0.5 or 'auto')"
            )

        # Analysis execution
        if st.button("🚀 Start Analysis", key="start_analysis"):
            if not feature_cols:
                st.warning("Please select at least one feature column")
            else:
                with st.spinner("Analyzing data..."):
                    anomalies = detect_anomalies_in_df(
                        df,
                        feature_cols,
                        timestamp_col_name,
                        contamination_str,
                        train_points,
                        config.config['model']['model_prefix']
                    )

                # Results display
                st.subheader("Results")
                
                # Performance metrics
                if st.session_state.performance_metrics:
                    metrics_df = pd.DataFrame([st.session_state.performance_metrics])
                    st.write("Model Performance Metrics:")
                    st.dataframe(metrics_df.style.format("{:.3f}"))

                # Anomalies table
                if anomalies:
                    st.write(f"Found {len(anomalies)} anomalies")
                    display_data = []
                    for anom in anomalies:
                        row_data = {
                            "Timestamp": anom["Timestamp"],
                            "IForest Score": anom["IForest Score"],
                            "Features": format_features(anom["Features"])
                        }
                        display_data.append(row_data)

                    st.dataframe(pd.DataFrame(display_data), use_container_width=True)
                    
                    # Export options
                    st.subheader("Export Results")
                    export_df = pd.DataFrame(display_data)
                    st.markdown(get_download_link(export_df, "anomalies.csv"), unsafe_allow_html=True)
                    
                    # Visualizations
                    plot_anomalies(df, anomalies, feature_cols)
                else:
                    st.info("No anomalies detected with current settings")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {str(e)}", exc_info=True)