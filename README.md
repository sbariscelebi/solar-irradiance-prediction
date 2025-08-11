# Solar Irradiance Prediction with CNN-LSTM and Wavelet Transform

This repository contains the implementation of a Master's thesis project focused on predicting solar irradiance using a hybrid CNN-LSTM model enhanced with wavelet transform and SHAP-based feature selection. The project leverages advanced machine learning techniques to forecast solar energy (derived from Global Horizontal Irradiance, GHI) with high accuracy, incorporating explainability through SHAP analysis and hyperparameter optimization using Optuna.

## 📑 Project Overview

The goal of this project is to develop a robust model for predicting solar irradiance, which is critical for optimizing solar energy systems. The methodology combines:
- **Wavelet Transform**: For time-frequency decomposition of input features.
- **CNN-LSTM Architecture**: To capture both spatial and temporal patterns in the data.
- **SHAP-based Feature Selection**: To identify the most impactful features for prediction.
- **Optuna Hyperparameter Optimization**: To fine-tune model performance.
- **Ablation Study**: To evaluate the contribution of each component to the model's performance.

The dataset used includes meteorological variables such as GHI, DNI, DHI, air temperature, relative humidity, and wind speed, sampled at 15-minute intervals.

## 📊 Key Features

- **Data Preprocessing**: Handles missing values, applies wavelet transform (e.g., `db4` wavelet), and engineers temporal features.
- **Model Architecture**: A hybrid CNN-LSTM model with configurable hyperparameters (e.g., learning rate, filters, LSTM units).
- **Feature Selection**: SHAP (SHapley Additive exPlanations) is used to select the top 80% of features based on their contribution to predictions.
- **Evaluation Metrics**: RMSE, R², MAE, and uncertainty quantification for model performance.
- **Visualization**: Includes wavelet component plots, SHAP summary plots, time-series predictions, and hourly RMSE analysis.
- **Checkpointing**: Saves intermediate results to ensure reproducibility and scalability.

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Required libraries:
  ```bash
  pip install tensorflow numpy pandas matplotlib seaborn pywt shap optuna scikit-learn scipy
