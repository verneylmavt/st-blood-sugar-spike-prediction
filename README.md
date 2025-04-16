# Predicting Blood Sugar Spikes for Diabetic Patients

This repository contains our project implementation for predicting future blood glucose levels in diabetic patients using deep learning techniques. Developed for the BrisT1D Blood Glucose Prediction Competition on Kaggle, our goal is to forecast blood glucose readings one hour ahead by leveraging multi-modal data from continuous glucose monitors (CGM), insulin pumps, and smartwatches.

---

## Project Overview

Diabetic patients require accurate and timely blood glucose predictions to manage their condition effectively. In this project, we built and compared three deep learning models—LSTM, Transformer, and Temporal Convolutional Network (TCN)—to forecast blood sugar spikes. The models were trained using historical measurements aggregated at five-minute intervals and focus on the most recent 2 hours (24 time steps) of data, including blood glucose, insulin, carbohydrate intake, heart rate, steps, and calories.

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
  A Streamlit-based interface that enables real-time inference and visualization of predictions compared with ground truth values.

---

## Directory Structure

```
│   .gitattributes
│   .gitignore
│   data_analysis.ipynb
│   interface.py
│   LSTM.ipynb
│   models.py
│   requirements.txt
│   TCN.ipynb
│   Transformer.ipynb
│   
├───.data
│       feature_scaler_aug.joblib
│       feature_scaler_aug.pkl
│       target_scaler_aug.joblib
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

- **`.gitattributes`, `.gitignore`**  
  Git configuration files.

- **`data_analysis.ipynb`**  
  Notebook for exploring and visualizing the dataset, inspecting missing values, and understanding temporal trends.

- **`LSTM.ipynb`, `Transformer.ipynb`, `TCN.ipynb`**  
  Notebooks that detail the training process for each deep learning model, including architecture, training steps, and evaluation.

- **`models.py`**  
  Contains the PyTorch implementations of our LSTM, Transformer, and TCN models.

- **`interface.py`**  
  Implements the interactive Streamlit dashboard for real-time inference and performance evaluation.

- **`requirements.txt`**  
  Lists all dependencies required to run the codebase.

**Folders:**

- **`.data`**  
  Contains scaler objects and processed dataset CSV files for training and testing.

- **`.export`**  
  Holds model checkpoint files (pre-trained models) and generated test submission files (CSV and PNG) for each model.

---

## Environment Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
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

## Notebooks and Workflow

1. **Data Analysis**

   - **`data_analysis.ipynb`**  
     Start here to perform exploratory data analysis, assess data quality, and visualize trends. This notebook also details the preprocessing steps applied to the raw data.

2. **Model Training and Evaluation**

   - **`LSTM.ipynb`**  
     Implements a two-layer LSTM network to capture sequential dependencies with dropout regularization and dense layers for final prediction.

   - **`Transformer.ipynb`**  
     Applies a Transformer-based architecture featuring input projection, positional encoding, and self-attention mechanisms to learn from long-range dependencies.

   - **`TCN.ipynb`**  
     Utilizes a Temporal Convolutional Network with dilated convolutions to achieve robust performance on the test set, especially on unseen participants.

   Choose the model you want to explore; each notebook contains training routines, loss visualization, and evaluation metrics.

3. **Interactive Interface**

   - **`interface.py`**  
     Run this script to launch the Streamlit dashboard for interactive evaluation. The dashboard enables you to select a model, view predictions, compare against ground truth, and inspect error metrics in real time.

---

## Training and Evaluation Process

- **Training:**  
  All models are trained using the Mean Squared Error (MSE) loss and the Adam optimizer. Early stopping based on validation loss is implemented to prevent overfitting. Checkpoints for each model are saved in the `.export` folder.

- **Evaluation:**  
  The models are evaluated using internal metrics (RMSE, MAE, R², CCC) and through Kaggle submissions. While the LSTM and Transformer models excel in internal evaluations, the TCN model provided the best generalization on unseen data, achieving a Kaggle score closest to the competition top score.

- **Submission:**  
  Pre-trained checkpoints and sample test submission files are provided in the `.export` directory for reproducibility and further analysis.

---

## Future Directions

- **Hybrid and Ensemble Strategies:**  
  Combine deep learning outputs with traditional machine learning techniques (e.g., LightGBM) to boost predictive performance.

- **Advanced Architectures:**  
  Explore state-of-the-art approaches such as Temporal Fusion Transformer, Autoformer, or N-BEATS for extended forecasting capabilities.

- **Real-Time Deployment:**  
  Optimize models for streaming data to support real-time monitoring and intervention in clinical settings.

- **Personalization:**  
  Develop patient-specific models to cater to individual physiological differences, potentially enhancing prediction accuracy.

---

## Contributors

**Group 20**  
- **Vanya Jalan (1006190)**  
- **Elvern Neylmav Tanny (1006203)**

---

## References

- [BrisT1D Blood Glucose Prediction Competition](https://www.kaggle.com/c/bris-t1d-blood-glucose-prediction)
- Salinas, D., Flunkert, V., & Gasthaus, J. (2017). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*. arXiv.  
- Oreshkin, B. N., et al. (2019). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*. arXiv.  
- Lim, B., et al. (2019). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. arXiv.  
- Zhou, H., et al. (2020). *Informer: Beyond efficient transformer for long sequence time-series forecasting*. arXiv.  
- Wu, H., et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*. arXiv.

---

## Contact

For further information or collaboration inquiries, please open an [issue](https://github.com/yourusername/yourrepo/issues) or reach out directly to the project maintainers.

---

This README provides a complete guide—from environment setup to training, evaluation, and future improvements—ensuring transparency and reproducibility in our deep learning approach to blood glucose prediction.