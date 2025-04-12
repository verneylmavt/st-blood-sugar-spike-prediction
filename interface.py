import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
from models import BrisT1DLSTM, BrisT1DDTransformer, BrisT1DTCN

# Streamlit config
st.set_page_config(page_title="Glucose Prediction Dashboard", layout="centered")
st.title("📈 Glucose Prediction Model Comparison")

# Model selection
model_name = st.selectbox("Select a Model", ["LSTM", "Transformer", "TCN"], index=0)

@st.cache_resource
def load_model(name):
    if name == "LSTM":
        model = BrisT1DLSTM()
        model.load_state_dict(torch.load(".export/checkpoint-aug-lstm.pt", map_location="cpu"))
    elif name == "Transformer":
        model = BrisT1DDTransformer()
        model.load_state_dict(torch.load(".export/checkpoint-aug-tf.pt", map_location="cpu"))
    else:
        model = BrisT1DTCN(num_inputs=6, num_channels=[64, 32])
        model.load_state_dict(torch.load(".export/checkpoint-aug-tcn.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_data():
    return pd.read_csv("train_test_aug.csv")

# Load resources
model = load_model(model_name)
df_scaled = load_data()

# Feature info
feature_order = ["bg", "insulin", "carbs", "cals", "hr", "steps"]
time_steps = [f"{i//60}:{i%60:02d}" for i in range(115, -5, -5)]  # 1:55 to 0:00
input_features = [f"{f}-{t}" for t in time_steps for f in feature_order]


# Row Selection Section with Data Preview

st.markdown("### 📄 Choose a Row to Predict From")

preview_limit = 10
show_all = st.checkbox("🔍 Show full dataset", value=False) # If user wants to see entire dataset
df_preview = df_scaled if show_all else df_scaled.head(preview_limit)

# Display preview
st.dataframe(df_preview[[*input_features, "bg+1:00"]])

# Let user select row from preview
row_idx = st.selectbox(
    "Select a Row Index",
    options=df_preview.index.tolist(),
    format_func=lambda i: f"Row {i} (bg+1:00: {df_scaled.loc[i, 'bg+1:00']:.2f})"
)

if st.button("Run Prediction on Selected Row"):
    row = df_scaled.iloc[row_idx]

    # Extract input features for 24x6 sequence
    x_scaled = row[input_features].astype(float).values.astype(np.float32).reshape(1, 24, 6)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Ground truth
    true_bg = row["bg+1:00"]

    # Predict
    with torch.no_grad():
        pred_bg = model(x_tensor).item()

    # Display results
    st.markdown(f"### 📍 Predicted Glucose (bg+1:00): {pred_bg:.2f} (scaled)")
    st.markdown(f"### 🎯 Ground Truth Glucose (bg+1:00): {true_bg:.2f} (scaled)")

    # Error metrics
    error = abs(pred_bg - true_bg)
    relative_error = error / abs(true_bg) if true_bg != 0 else float('inf')
    percent_error = ((pred_bg - true_bg)/ true_bg) * 100 if true_bg != 0 else float('inf')


    # Display metrics in cards
    st.markdown("### 📊 Prediction Metrics")
    col1, col2, col3 = st.columns(3)

    # Styling done to show metrics
    with col1:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold;">Absolute Error</div>
            <div style="font-size: 2rem; color: white; margin-top: 0.5rem;">{error:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold;">Relative Error</div>
            <div style="font-size: 2rem; color: white; margin-top: 0.5rem;">{relative_error:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="border: 2px solid #444; border-radius: 10px; padding: 1.2rem; text-align: center;">
            <div style="font-size: 1.2rem; font-weight: bold;">% Error</div>
            <div style="font-size: 2rem; color: white; margin-top: 0.5rem;">{percent_error:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

