# SUTD 50.039 Deep Learning | Blood Sugar Spikes Prediction

This repository contains Group 20 project implementation for predicting future blood glucose levels in diabetic patients using deep learning techniques. Developed for the BrisT1D Blood Glucose Prediction Competition on Kaggle, our goal is to forecast blood glucose readings one hour ahead by leveraging past time-series data from blood glucose, insulin intake, carbohydrate intake, heart rate, steps walked, and calories burnt.

---

## Project Overview

Diabetic patients require accurate and timely blood glucose predictions to manage their condition effectively. In this project, we built and compared three deep learning models: LSTM, Transformer, and Temporal Convolutional Network (TCN) to forecast blood sugar spikes. The models were trained using historical measurements aggregated at five-minute intervals and focus on the most recent 2 hours (24 time steps) of data.

Key aspects of our project include:

- **Data Preprocessing:**  
  Handling missing values through interpolation, zero-filling, and forward/backward-fill; applying Gaussian noise for data augmentation; and scaling features with StandardScaler.
  
- **Model Architectures:**  
  - **LSTM:** Two-layer LSTM with dropout and dense layers, capturing sequential dependencies effectively.  
  - **Transformer:** Utilizes input projection, positional encoding, and multi-head self-attention to capture long-range dependencies.  
  - **TCN:** Employs dilated causal convolutions to achieve a stable convergence and robust predictions, especially on unseen participants.
  
- **Evaluation:**  
  Comprehensive evaluation with regression metrics (RMSE, MAE, R², CCC) on internal test splits, along with external validation via Kaggle submissions.

- **Interactive Interface:**  
  A Streamlit-based interface that enables real-time inference.

---

## Directory Structure

```
│   data_analysis.ipynb
│   interface.py
│   LSTM.ipynb
│   models.py
│   requirements.txt
│   TCN.ipynb
│   Transformer.ipynb
│   
├───.data
│       feature_scaler_aug.pkl
│       target_scaler_aug.pkl
│       train_aug_mean.csv
│       train_test_aug.csv
│
└───.export
        checkpoint-aug-lstm.pt
        checkpoint-aug-tcn.pt
        checkpoint-aug-tf.pt
        test_submission_aug_lstm.csv
        test_submission_aug_lstm.png
        test_submission_aug_tcn.csv
        test_submission_aug_tcn.png
        test_submission_aug_tf.csv
        test_submission_aug_tf.png
```

**Root-level Files:**

- **`data_analysis.ipynb`**  
  Notebook for exploring and visualizing the dataset, inspecting missing values, and understanding temporal trends.

- **`LSTM.ipynb`, `Transformer.ipynb`, `TCN.ipynb`**  
  Notebooks that detail the training process for each deep learning model, including architecture, training steps, and evaluation.

- **`interface.py`**  
  Implements the interactive Streamlit dashboard for real-time inference.

**Folders:**

- **`.data`**  
  Contains scaler objects and processed dataset CSV files for training and testing.

- **`.export`**  
  Holds model checkpoint files (pre-trained models) and generated test submission files (CSV) for each model.

---

## Environment Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/verneylmavt/st-blood-sugar-spike-prediction.git
   cd st-blood-sugar-spike-prediction
   ```

2. **Set Up a Virtual Environment**

   Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # For macOS/Linux
   venv\Scripts\activate       # For Windows
   ```

3. **Install Dependencies**

   Upgrade pip and install the required packages:

   ```bash
   pip install -U pip wheel setuptools
   pip install -r requirements.txt
   ```

---

## Contributors

**Group 20**  
- **Elvern Neylmav Tanny (1006203)**
- **Vanya Jalan (1006190)**  