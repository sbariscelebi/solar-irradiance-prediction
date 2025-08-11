import pandas as pd
import numpy as np
import pywt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import re
import logging
import os
from utils import load_checkpoint, save_checkpoint  # Import from utils

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, sep=';')
    cols = ['period_end', 'ghi', 'dni', 'dhi', 'clearsky_ghi', 'clearsky_dni', 'clearsky_dhi',
            'air_temp', 'dewpoint_temp', 'relative_humidity', 'wind_speed_10m', 'cloud_opacity',
            'zenith', 'azimuth', 'surface_pressure']
    if not all(col in df.columns for col in cols):
        raise ValueError("Missing required columns")
    df = df[cols].rename(columns={'air_temp': 'air_temperature', 'dewpoint_temp': 'dew_point', 'wind_speed_10m': 'wind_speed'})
    df['datetime'] = pd.to_datetime(df['period_end'])
    df.set_index('datetime', inplace=True)
    df.drop('period_end', axis=1, inplace=True)
    logging.info("Data loaded")
    return df

def preprocess_data(df):
    df = df.copy()
    df['solar_energy'] = df['ghi'] * 0.18
    initial_rows = len(df)
    df = df.dropna()
    logging.info(f"Dropped {initial_rows - len(df)} rows due to missing values")
    return df

def strip_time_suffix(col_name):
    try:
        return re.sub(r'_t\d+$', '', col_name)
    except Exception as e:
        logging.error(f"Error in strip_time_suffix: {str(e)}")
        return None

def apply_wavelet_single_column(col_data, col_name, wavelet, wavelet_level, include_cA, include_cD1, include_cD2):
    wav_data = {}
    coeffs = pywt.wavedec(col_data, wavelet, level=wavelet_level, mode='periodization')
    if include_cA:
        wav_data[f'{col_name}_cA'] = pywt.waverec([coeffs[0]] + [None] * wavelet_level, wavelet, mode='periodization')[:len(col_data)]
    if include_cD1 and wavelet_level >= 1:
        wav_data[f'{col_name}_cD1'] = pywt.waverec([None, coeffs[1]] + [None] * (wavelet_level - 1), wavelet, mode='periodization')[:len(col_data)]
    if include_cD2 and wavelet_level >= 2:
        wav_data[f'{col_name}_cD2'] = pywt.waverec([None] * 2 + [coeffs[2]] + [None] * (wavelet_level - 2), wavelet, mode='periodization')[:len(col_data)]
    return wav_data

def apply_wavelet(df, wavelet, wavelet_level, include_cA=True, include_cD1=True, include_cD2=False):
    # Only numerical columns excluding target
    input_features = [col for col in df.select_dtypes(include=np.number).columns if col != 'solar_energy']
    wav_data_list = Parallel(n_jobs=-1, backend='loky')(
        delayed(apply_wavelet_single_column)(
            df[col].values, col, wavelet, wavelet_level, include_cA, include_cD1, include_cD2
        ) for col in input_features
    )
    wav_data = {}
    for d in wav_data_list:
        wav_data.update(d)
    df_wav = pd.concat([df, pd.DataFrame(wav_data, index=df.index)], axis=1)
    wav_cols = list(wav_data.keys())
    print(f"Wavelet components: {wav_cols}")
    return df_wav, wav_cols

def build_selected_dataframe(df_fe: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """ Builds a subset dataframe from df_fe based on features selected by SHAP.
    If selected_features is empty, uses all columns except target.
    """
    cols = {strip_time_suffix(f) for f in selected_features if strip_time_suffix(f) in df_fe.columns}
    if not cols:
        cols = set(df_fe.columns) - {'solar_energy'}
    df_selected = df_fe[list(cols) + ['solar_energy']].dropna()
    return df_selected

def engineer_features(df):
    """ Safe feature engineering.
    Checks if required columns exist before calculating each new feature.
    Skips the step if columns are missing to avoid KeyError.
    """
    df = df.copy()
    # Apparent temperature
    if {'air_temperature', 'wind_speed'}.issubset(df.columns):
        df['apparent_temp'] = df['air_temperature'] - 0.7 * df['wind_speed']
    # DNI / GHI ratio
    if {'dni', 'ghi'}.issubset(df.columns):
        df['dni_ghi_ratio'] = df['dni'] / (df['ghi'] + 1e-6)
    # DHI / GHI ratio
    if {'dhi', 'ghi'}.issubset(df.columns):
        df['dhi_ghi_ratio'] = df['dhi'] / (df['ghi'] + 1e-6)
    # Clean unnecessary NaNs
    df = df.dropna()
    logging.info("Features engineered (safe mode)")
    return df

def create_sequences(X, y, time_steps, indices):
    Xs, ys, idxs = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
        idxs.append(indices[i + time_steps])  # Index of target value
    return np.array(Xs), np.array(ys), np.array(idxs)

def split_scale_data(df, target_col, test_size, time_steps):
    try:
        X, y = df.drop(target_col, axis=1), df[target_col]
        indices = df.index  # Datetime index
        tscv = TimeSeriesSplit(n_splits=5)
        fold_data = []
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X, y, indices, test_size=test_size, shuffle=False
        )
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_temp)):
            X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
            y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
            idx_train, idx_val = idx_temp[train_idx], idx_temp[val_idx]
            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
            y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
            X_train_seq, y_train_seq, idx_train_seq = create_sequences(
                X_train_scaled, y_train_scaled, time_steps, idx_train
            )
            X_val_seq, y_val_seq, idx_val_seq = create_sequences(
                X_val_scaled, y_val_scaled, time_steps, idx_val
            )
            X_test_seq, y_test_seq, idx_test_seq = create_sequences(
                X_test_scaled, y_test_scaled, time_steps, idx_test
            )
            fold_data.append({
                'fold': fold + 1,
                'X_train_seq': X_train_seq, 'X_val_seq': X_val_seq, 'X_test_seq': X_test_seq,
                'y_train_seq': y_train_seq, 'y_val_seq': y_val_seq, 'y_test_seq': y_test_seq,
                'idx_train_seq': idx_train_seq, 'idx_val_seq': idx_val_seq, 'idx_test_seq': idx_test_seq,
                'scaler_X': scaler_X, 'scaler_y': scaler_y,
                'feature_names': X.columns.tolist()
            })
        logging.info("Data split and scaled with indices")
        return fold_data
    except Exception as e:
        logging.error(f"Error in split_scale_data: {str(e)}")
        raise