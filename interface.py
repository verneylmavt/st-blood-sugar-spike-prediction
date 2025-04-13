import streamlit as st
import pandas as pd
import numpy as np
import torch
from models import BrisT1DLSTM, BrisT1DDTransformer, BrisT1DTCN
import joblib

# -----------------------
# Streamlit configuration
# -----------------------
st.set_page_config(page_title="Glucose Prediction Dashboard", layout="centered")

# -----------------------
# Resource loading functions
# -----------------------
@st.cache_resource
def load_model(name):
    if name == "LSTM":
        model = BrisT1DLSTM()
        model.load_state_dict(torch.load(".export/checkpoint-aug-lstm.pt", map_location="cpu", weights_only=True))
    elif name == "Transformer":
        model = BrisT1DDTransformer()
        model.load_state_dict(torch.load(".export/checkpoint-aug-tf.pt", map_location="cpu", weights_only=True))
    else:
        model = BrisT1DTCN(num_inputs=6, num_channels=[64, 32])
        model.load_state_dict(torch.load(".export/checkpoint-aug-tcn.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_data
def load_data():
    return pd.read_csv(".data/train_test_aug.csv")

@st.cache_data
def load_feature_scaler():
    return joblib.load(".data/feature_scaler_aug.pkl")

@st.cache_data
def load_target_scaler():
    return joblib.load(".data/target_scaler_aug.pkl")

# -----------------------
# Data preprocessing functions
# -----------------------
def prepare_display_data(df_scaled, feature_scaler, target_scaler, input_features):
    """
    Returns a copy of the dataframe with the input features and target inverse transformed 
    for display purposes.
    """
    df_display = df_scaled.copy()
    df_display[input_features] = feature_scaler.inverse_transform(df_display[input_features])
    df_display["bg+1:00"] = target_scaler.inverse_transform(
        df_display["bg+1:00"].values.reshape(-1, 1)
    ).ravel()
    return df_display

# -----------------------
# Model inference function
# -----------------------
def infer_glucose(model, row, input_features, target_scaler):
    """
    Runs inference on a row (with scaled features) and returns the prediction 
    in original units.
    """
    # Reshape the input to shape (1, 24, 6)
    x_scaled = row[input_features].astype(np.float32).values.reshape(1, 24, 6)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        pred_scaled = model(x_tensor).item()
    # Convert the scaled prediction to original units for display and error computation
    pred_original = target_scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred_original

# -----------------------
# Error metrics and UI rendering functions
# -----------------------
def compute_error_metrics(pred, true):
    """
    Computes absolute, relative, and percentage errors on original scale.
    """
    error = abs(pred - true)
    relative_error = error / abs(true) if true != 0 else float('inf')
    percent_error = (abs(pred - true) / abs(true)) * 100 if true != 0 else float('inf')
    return error, relative_error, percent_error

def render_prediction_metrics(error, relative_error, percent_error):
    """
    Renders error metrics using Streamlit columns and styled HTML.
    """
    st.markdown("### Prediction Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold;">Absolute Error</div>
            <div style="font-size: 2rem; margin-top: 0.5rem;">{error:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold;">Relative Error</div>
            <div style="font-size: 2rem; margin-top: 0.5rem;">{relative_error:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold;">% Error</div>
            <div style="font-size: 2rem; margin-top: 0.5rem;">{percent_error:.2f}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.title("🩸 Glucose Prediction")

    # -----------------------
    # Model selection
    # -----------------------
    model_name = st.selectbox("Select a Model", ["LSTM", "Transformer", "TCN"], index=0)
    st.divider()
    
    # -----------------------
    # Constants and feature definitions
    # -----------------------
    feature_order = ["bg", "insulin", "carbs", "cals", "hr", "steps"]
    time_steps = [f"{i//60}:{i%60:02d}" for i in range(115, -5, -5)]  # Generates sequence from 1:55 to 0:00
    input_features = [f"{f}-{t}" for t in time_steps for f in feature_order]

    # -----------------------
    # Load resources
    # -----------------------
    model = load_model(model_name)
    # df_scaled holds the scaled data that the model was trained on
    df_scaled = load_data()
    feature_scaler = load_feature_scaler()
    target_scaler = load_target_scaler()

    # Prepare display data (inverse transformed only for presentation)
    df_display = prepare_display_data(df_scaled, feature_scaler, target_scaler, input_features)

    # -----------------------
    # UI: Data preview and row selection
    # -----------------------
    st.markdown("### Choose a Row to Predict From")

    preview_limit = 10
    show_all = st.checkbox("Show full dataset", value=False)
    df_preview = df_display if show_all else df_display.head(preview_limit)
    st.dataframe(df_preview[[*input_features, "bg+1:00"]])

    row_idx = st.selectbox(
        "Select a Row Index",
        options=df_preview.index.tolist(),
        format_func=lambda i: f"Row {i} (bg+1:00: {df_display.loc[i, 'bg+1:00']:.2f})"
    )
    st.divider()

    # -----------------------
    # On-demand prediction execution
    # -----------------------
    if st.button("Run Prediction on Selected Row"):
        # Use the scaled data for model prediction
        row_scaled = df_scaled.iloc[row_idx]
        # Run inference using the modularized function
        pred_bg_original = infer_glucose(model, row_scaled, input_features, target_scaler)
        # Get the inverse transformed ground truth for comparison from the display dataframe
        true_bg_original = df_display.loc[row_idx, "bg+1:00"]
        
        st.markdown(f"### Predicted Glucose (bg+1:00): {pred_bg_original:.2f}")
        st.markdown(f"### Real Glucose (bg+1:00): {true_bg_original:.2f}")
        
        # Compute and display error metrics on original scale
        error, relative_error, percent_error = compute_error_metrics(pred_bg_original, true_bg_original)
        render_prediction_metrics(error, relative_error, percent_error)

if __name__ == "__main__":
    main()