import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
import numpy as np
import scipy.stats
import gc
import logging
from utils import save_checkpoint, load_checkpoint  # Import from utils

def build_model(input_shape, learning_rate, filters1, filters2, lstm_units1, lstm_units2, dropout1, dropout2, dense_units, use_lstm=True, use_dropout=True):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Add Input layer
    model.add(Conv1D(filters=filters1, kernel_size=3, activation='relu'))  # Remove input_shape from Conv1D
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=filters2, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    if use_lstm:
        model.add(LSTM(lstm_units1, return_sequences=True))
        if use_dropout:
            model.add(Dropout(dropout1))
        model.add(LSTM(lstm_units2))
        if use_dropout:
            model.add(Dropout(dropout2))
    else:
        model.add(Flatten())
        model.add(Dense(lstm_units1, activation='relu'))
        if use_dropout:
            model.add(Dropout(dropout1))
        model.add(Dense(lstm_units2, activation='relu'))
        if use_dropout:
            model.add(Dropout(dropout2))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    logging.info("Model built")
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs: int, patience: int, fold: int, initial_epoch: int = 0, batch_size: int = 64):
    """ Trains the model; can continue from initial epoch. """
    callbacks = [
        EarlyStopping(monitor="val_mae", patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=f"saved_model/best_model_fold_{fold}.keras", save_best_only=True, monitor="val_mae")
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, initial_epoch=initial_epoch, batch_size=batch_size,
        callbacks=callbacks, verbose=0
    )
    logging.info("Fold %d trained (final epoch: %d)", fold, history.epoch[-1])
    return history, model

def inverse_scale(y_scaled, scaler):
    """ Converts 1D y_scaled vector back to original scale. """
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

def calculate_metrics(y_true, y_pred):
    base = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        'MedAE': median_absolute_error(y_true, y_pred),
        'EVS': explained_variance_score(y_true, y_pred),
        'CVRMSE': (np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else np.inf
    }
    base.update(calculate_additional_metrics(y_true, y_pred))
    return base

def calculate_additional_metrics(y_true, y_pred):
    """ Returns additional error metrics for y_true and y_pred vectors. 
    Return: dict
    """
    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)
    # Symmetric MAPE
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    # MASE (scaled MAE) — for non-seasonal data
    mae = mean_absolute_error(y_true, y_pred)
    scale = mean_absolute_error(y_true[1:], y_true[:-1])  # naive forecast error
    mase = mae / (scale + 1e-8)
    return {
        'MBE': mbe,
        'sMAPE': smape,
        'MASE': mase,
    }

def evaluate_model(model, X_train, y_train, X_test, y_test, idx_test, scaler_y):
    # 1) Predictions (scaled)
    y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
    y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()
    # 2) Convert to original scale
    y_train_true = inverse_scale(y_train, scaler_y)
    y_test_true = inverse_scale(y_test, scaler_y)
    y_train_pred = inverse_scale(y_train_pred_scaled, scaler_y)
    y_test_pred = inverse_scale(y_test_pred_scaled, scaler_y)
    # 3) Calculate metrics
    train_scaled = calculate_metrics(y_train, y_train_pred_scaled)
    test_scaled = calculate_metrics(y_test, y_test_pred_scaled)
    train_orig = calculate_metrics(y_train_true, y_train_pred)
    test_orig = calculate_metrics(y_test_true, y_test_pred)
    # 4) Flatten with suffixes
    train_metrics = {f"{k}_scaled": v for k, v in train_scaled.items()}
    train_metrics.update({f"{k}_orig": v for k, v in train_orig.items()})
    test_metrics = {f"{k}_scaled": v for k, v in test_scaled.items()}
    test_metrics.update({f"{k}_orig": v for k, v in test_orig.items()})
    logging.info(f"Train metrics(all): {train_metrics}")
    logging.info(f"Test metrics(all): {test_metrics}")
    return train_metrics, test_metrics, idx_test, y_test_true, y_test_pred, y_test, y_test_pred_scaled

def predict_with_uncertainty(model, X, n_iter=100):
    preds = np.array([model(X, training=True).numpy().flatten() for _ in range(n_iter)])
    return preds.mean(axis=0), preds.std(axis=0)

def evaluate_uncertainty(model, X_test, y_test_scaled, scaler_y, n_iter=100, alpha=0.95):
    """ y_test_scaled : scaled (standardized) true values
    scaler_y : same scaler; used for inverse_scale
    """
    # MC Dropout predictions (scaled)
    mean_pred_scaled, std_pred_scaled = predict_with_uncertainty(model, X_test, n_iter)
    # Reverse scaling
    y_test = inverse_scale(y_test_scaled, scaler_y)
    mean_pred = inverse_scale(mean_pred_scaled, scaler_y)
    std_factor = (getattr(scaler_y, "scale_", None) or getattr(scaler_y, "scale", None) or np.array([1.0]))[0]
    std_pred = std_pred_scaled * std_factor  # std scales linearly
    # Metrics on raw values
    metrics = calculate_metrics(y_test, mean_pred)
    # PICP
    z = scipy.stats.norm.ppf((1 + alpha) / 2)
    lower, upper = mean_pred - z * std_pred, mean_pred + z * std_pred
    metrics['PICP'] = np.mean((y_test >= lower) & (y_test <= upper))
    logging.info(f"Uncertainty metrics: {metrics}, Std: {np.mean(std_pred):.4f}")
    return metrics, mean_pred, std_pred