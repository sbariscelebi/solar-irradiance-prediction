import os
import time
import logging
import pickle
import re
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import shap
import gc
import optuna
from pathlib import Path

try:
    import plotly
    import optuna.visualization as vis
    _plotly_available = True
except ImportError:
    vis = None
    _plotly_available = False

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import scipy.stats
import matplotlib as mpl
import matplotlib.cm as cm
from tensorflow.keras.layers import Input
from joblib import Parallel, delayed

tf.keras.utils.set_random_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
warnings.filterwarnings("ignore", category=UserWarning)
import seaborn as sns

from utils import init_plot_style, save_checkpoint, load_checkpoint, save_fig_multi
from preprocessing import load_data, preprocess_data, apply_wavelet, engineer_features, split_scale_data, build_selected_dataframe, strip_time_suffix
from models import build_model, train_model, evaluate_model, evaluate_uncertainty
from analysis import compute_shap_values, select_shap_features, plot_shap_summary, analyze_pca, plot_wavelet_components, plot_wavelet_energy, plot_time_series_prediction, plot_results, plot_hourly_rmse, ablation_study, plot_ablation_barplot, optimize_params, optimize_wavelet_params, plot_fold_comparison, prettify_metric, plot_uncertainty_band, plot_optuna_history

# Constants
SHAP_SAMPLES = 10  # Remember to make 20
MODEL_NAME = "CNN_LSTM_Wavelet_ShapPercentile"
TARGET_COL = "solar_energy"
PLOTS_DIR = "plots"
N_TRIALS = 20  # Remember to make 20  # OPTUNA search space
BATCH_SIZE = 64
MODEL_DIR = "saved_model"

MANUAL_PARAMS = {
    "learning_rate": 0.0001, "filters1": 64, "filters2": 64, "lstm_units1": 128, "lstm_units2": 64,
    "dropout1": 0.3, "dropout2": 0.2, "dense_units": 32, "wavelet": "db4", "wavelet_level": 2,
    "include_cA": True, "include_cD1": True, "include_cD2": False, "test_size": 0.2,
    "time_steps": 24, "batch_size": 64, "epochs": 100, "patience": 10, "shap_percentile": 0.8
}
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main(rng: np.random.Generator = np.random.default_rng(42)):
    init_plot_style()
    # Create folder for checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    timing = {}

    try:
        # Data loading and preprocessing
        start = time.time()
        df = load_checkpoint('checkpoints/df_preprocessed.pkl')
        if df is None:
            df = load_data(MAIN_PATH)
            df = preprocess_data(df)
            save_checkpoint(df, 'checkpoints/df_preprocessed.pkl')
        timing['Data_Load'] = time.time() - start

        # Pre-data split (for wavelet optimization)
        start = time.time()
        fold_data_raw = load_checkpoint('checkpoints/fold_data_raw.pkl')
        if fold_data_raw is None:
            fold_data_raw = split_scale_data(df, TARGET_COL, MANUAL_PARAMS['test_size'], MANUAL_PARAMS['time_steps'])
            save_checkpoint(fold_data_raw, 'checkpoints/fold_data_raw.pkl')
        timing['Data_Split_Raw'] = time.time() - start

        # Wavelet and feature engineering
        start = time.time()
        df_fe = load_checkpoint('checkpoints/df_fe.pkl')
        wav_cols = load_checkpoint('checkpoints/wav_cols.pkl')
        if df_fe is None or wav_cols is None:
            # First and only wavelet transformation
            best_wavelet_params = optimize_wavelet_params(df, TARGET_COL, fold_data_raw[0])

            df_wav, wav_cols = apply_wavelet(
                df,
                wavelet=best_wavelet_params['wavelet'],
                wavelet_level=best_wavelet_params['wavelet_level'],
                include_cA=MANUAL_PARAMS['include_cA'],
                include_cD1=MANUAL_PARAMS['include_cD1'],
                include_cD2=MANUAL_PARAMS['include_cD2']
            )

            level = best_wavelet_params['wavelet_level']
            best_components = ["ghi_cA"] + [f"ghi_cD{i}" for i in range(1, level+1)]
            best_components = [x for x in best_components if x in wav_cols]

            plot_wavelet_components(
                df_wav, wav_cols, sample_col="ghi", best_components=best_components
            )

            print("Original Features:")
            original_cols = df.columns.tolist()

            wavelet_cols = df_wav.columns.tolist()
            added_wavelet_cols = [col for col in wavelet_cols if col not in original_cols]

            print("Original Features:", original_cols)
            print("\nAll Features (After Wavelet):", wavelet_cols)
            print("\nOnly Added by Wavelet:", added_wavelet_cols)

            # Save to Excel
            with pd.ExcelWriter(os.path.join(PLOTS_DIR, "feature_lists.xlsx")) as writer:
                pd.DataFrame({'Original Features': original_cols}).to_excel(writer, sheet_name="Original", index=False)
                pd.DataFrame({'All Features': wavelet_cols}).to_excel(writer, sheet_name="All_After_Wavelet", index=False)
                pd.DataFrame({'Wavelet Features': added_wavelet_cols}).to_excel(writer, sheet_name="Only_Wavelet", index=False)

            logging.info("Feature lists saved to Excel: plots/feature_lists.xlsx")

            plot_wavelet_components(df_wav, wav_cols, sample_col="ghi")
            plot_wavelet_energy(df_wav, wav_cols)
            df_fe = engineer_features(df_wav)
            save_checkpoint(df_fe, 'checkpoints/df_fe.pkl')
            save_checkpoint(wav_cols, 'checkpoints/wav_cols.pkl')
        timing['Preprocessing'] = time.time() - start

        # Data split (after feature engineering)
        start = time.time()
        fold_data = load_checkpoint('checkpoints/fold_data.pkl')
        if fold_data is None:
            fold_data = split_scale_data(df_fe, TARGET_COL, MANUAL_PARAMS['test_size'], MANUAL_PARAMS['time_steps'])
            save_checkpoint(fold_data, 'checkpoints/fold_data.pkl')
        timing['Data_Split'] = time.time() - start

        # SHAP analysis
        start = time.time()
        fold_info = fold_data[0]
        shap_values = load_checkpoint('checkpoints/shap_values.pkl')
        selected_features = load_checkpoint('checkpoints/selected_features.pkl')
        if selected_features is None:
            selected_features = []

        # if shap_values is None or selected_features is None:
        model = build_model(
            (fold_info['X_train_seq'].shape[1], fold_info['X_train_seq'].shape[2]),
            **{k: MANUAL_PARAMS[k] for k in [
                'learning_rate', 'filters1', 'filters2',
                'lstm_units1', 'lstm_units2',
                'dropout1', 'dropout2', 'dense_units'
            ]}
        )
        history, trained_model = train_model(
            model,
            fold_info['X_train_seq'],
            fold_info['y_train_seq'],
            fold_info['X_val_seq'],
            fold_info['y_val_seq'],
            MANUAL_PARAMS['epochs'],
            MANUAL_PARAMS['patience'],
            fold_info['fold']
        )
        # shap_values = compute_shap_values(trained_model, fold_info, SHAP_SAMPLES)
        save_checkpoint(shap_values, 'checkpoints/shap_values.pkl')

        feature_names = [
            f"{fn}_t{t}"
            for t in range(MANUAL_PARAMS['time_steps'])
            for fn in fold_info['feature_names']
        ]

        # Flatten time dimension
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1)
        X_sample_flat = fold_info['X_test_seq'][:SHAP_SAMPLES].reshape(SHAP_SAMPLES, -1)  # <-- added

        selected_features = select_shap_features(
            shap_values_flat,
            feature_names,
            MANUAL_PARAMS['shap_percentile']
        )
        save_checkpoint(selected_features, 'checkpoints/selected_features.pkl')

        selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

        excluded_feats = plot_shap_summary(
            shap_values_flat[:, selected_indices],
            X_sample_flat[:, selected_indices],
            [feature_names[i] for i in selected_indices],
            plots_dir=PLOTS_DIR,
            model_name=MODEL_NAME,
            percentile=0.20,
            max_display=20,
            plot_type="dot",
            verbose=True
        )

        timing['SHAP'] = time.time() - start

        # PCA analysis
        start = time.time()
        analyze_pca(df_fe, selected_features)
        timing['PCA'] = time.time() - start

        # Update data with selected features
        start = time.time()
        fold_data_selected = load_checkpoint('checkpoints/fold_data_selected.pkl')
        if fold_data_selected is None:
            selected_cols = {strip_time_suffix(f) for f in selected_features if strip_time_suffix(f) in df_fe.columns}
            if not selected_cols:
                selected_cols = set(df_fe.columns) - {TARGET_COL}
            df_selected = df_fe[list(selected_cols) + [TARGET_COL]].dropna()
            fold_data_selected = split_scale_data(df_selected, TARGET_COL, MANUAL_PARAMS['test_size'], MANUAL_PARAMS['time_steps'])
            save_checkpoint(fold_data_selected, 'checkpoints/fold_data_selected.pkl')
        timing['Data_Split_Selected'] = time.time() - start
        logging.info("Selected fold data prepared and saved to checkpoint")

        # Ablation study
        # ---- CREATE / LOAD DF_SELECTED ------------------------------------------
        start = time.time()
        df_selected = load_checkpoint('checkpoints/df_selected.pkl')
        if df_selected is None:
            df_selected = build_selected_dataframe(df_fe, selected_features)
            save_checkpoint(df_selected, 'checkpoints/df_selected.pkl')
        timing['Data_Selected_Build'] = time.time() - start

        # ---- ABLATION STUDY ----------------------------------------------------
        start = time.time()
        ablation_df = ablation_study(df_selected, selected_features, timing)

        plot_ablation_barplot(ablation_df, metric="Test_RMSE_orig")
        plot_ablation_barplot(ablation_df, metric="Test_R2_orig")
        plot_ablation_barplot(ablation_df, metric="Test_Uncertainty")

        timing['Ablation'] = time.time() - start

        # Training and evaluation
        results = []
        for fold_info in fold_data_selected:
            start = time.time()
            fold_id = fold_info["fold"]
            model_path = f"{MODEL_DIR}/{MODEL_NAME}_fold_{fold_id}.keras"
            metrics_csv = f"{PLOTS_DIR}/metrics_fold_{fold_id}.csv"

            # Skip if model and metrics already exist
            if Path(model_path).exists() and Path(metrics_csv).exists():
                logging.info("Fold %d already trained, skipping.", fold_id)
                continue

            # Continue hyper-parameter search (Optuna) from where left off
            best_params = optimize_params(fold_info, rng)

            # Build model and load possible checkpoint
            model = build_model(
                (fold_info["X_train_seq"].shape[1], fold_info["X_train_seq"].shape[2]),
                **best_params
            )
            ckpt_dir = f"checkpoints/fold_{fold_id}"
            os.makedirs(ckpt_dir, exist_ok=True)
            latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if latest_ckpt:
                model.load_weights(latest_ckpt)
                last_epoch = int(re.search(r"epoch_(\d+)", latest_ckpt).group(1))
                initial_epoch = last_epoch + 1
                logging.info("Fold %d continuing from (epoch %d).", fold_id, initial_epoch)
            else:
                initial_epoch = 0
                logging.info("Fold %d starting from scratch.", fold_id)

            # Train model
            history, trained_model = train_model(
                model,
                fold_info["X_train_seq"], fold_info["y_train_seq"],
                fold_info["X_val_seq"], fold_info["y_val_seq"],
                MANUAL_PARAMS["epochs"], MANUAL_PARAMS["patience"],
                fold_id,
                initial_epoch=initial_epoch
            )

            # SHAP analysis for Fold 1
            if fold_id == 1:  # Only for Fold 1
                shap_values_fold = compute_shap_values(trained_model, fold_info, SHAP_SAMPLES)
                feature_names_fold = [
                    f"{fn}_t{t}"
                    for t in range(MANUAL_PARAMS['time_steps'])
                    for fn in fold_info['feature_names']
                ]
                shap_values_fold_flat = shap_values_fold.reshape(shap_values_fold.shape[0], -1)
                X_sample_fold_flat = fold_info['X_test_seq'][:SHAP_SAMPLES].reshape(SHAP_SAMPLES, -1)

                # Limited indices with selected_features only
                selected_indices = [i for i, f in enumerate(feature_names_fold) if f in selected_features]
                selected_feature_names = [feature_names_fold[i] for i in selected_indices]

                # Reduce SHAP values and X_sample to selected features only
                shap_values_selected = shap_values_fold_flat[:, selected_indices]
                X_sample_selected = X_sample_fold_flat[:, selected_indices]

                # Show maximum 20 features (or less, depending on selected feature count)
                top_n = min(len(selected_feature_names), 20)

                logging.info(f"SHAP plot for selected feature count: {len(selected_feature_names)}")
                excluded_feats_fold = plot_shap_summary(
                    shap_values_selected,
                    X_sample_selected,
                    selected_feature_names,
                    plots_dir=PLOTS_DIR,
                    model_name=f"{MODEL_NAME}_fold_{fold_id}_selected",
                    percentile=0.20,
                    max_display=20,
                    plot_type="dot",
                    verbose=True
                )

                logging.info(f"SHAP summary plot generated for fold {fold_id} with {len(selected_feature_names)} selected features")

            # Calculate performance and uncertainty
            train_metrics, test_metrics, idx_test, y_test_true, y_test_pred, y_test_scaled, y_test_pred_scaled = evaluate_model(
                trained_model,
                fold_info["X_train_seq"], fold_info["y_train_seq"],
                fold_info["X_test_seq"], fold_info["y_test_seq"],
                fold_info["idx_test_seq"],
                fold_info["scaler_y"]
            )
            unc_metrics, mean_pred, std_pred = evaluate_uncertainty(
                trained_model,
                fold_info["X_test_seq"],
                fold_info["y_test_seq"],
                fold_info["scaler_y"]
            )

            # Save test results to Excel file
            test_results_df = pd.DataFrame({
                'Date': idx_test,
                'Actual': y_test_true,
                'Predicted': y_test_pred,
                'Normalized_Actual': y_test_scaled,
                'Normalized_Predicted': y_test_pred_scaled
            })

            if test_results_df['Date'].dt.tz is not None:
                test_results_df['Date'] = test_results_df['Date'].dt.tz_localize(None)

            excel_path = f"{PLOTS_DIR}/test_results_fold_{fold_id}.xlsx"
            test_results_df.to_excel(excel_path, index=False)
            logging.info(f"Test results for fold {fold_id} saved to {excel_path}")

            # Calculate datetime index and draw hourly RMSE graph
            start_idx = len(df_selected) - len(y_test_true)
            datetime_index = df_selected.index[start_idx:start_idx + len(y_test_true)]
            plot_hourly_rmse(datetime_index, y_test_true, y_test_pred, fold_info["fold"])

            # Other graphs
            plot_time_series_prediction(df_selected, y_test_true, y_test_pred, fold_id)
            plot_uncertainty_band(datetime_index, y_test_true, mean_pred, std_pred, fold_id)
            plot_results(history, y_test_true, y_test_pred, fold_id)

            # Save results
            trained_model.save(model_path)
            all_metrics = {
                **{f"Train_{k}": v for k, v in train_metrics.items()},
                **{f"Test_{k}": v for k, v in test_metrics.items()},
                **{f"Unc_{k}": v for k, v in unc_metrics.items()},
                "Unc_StdMean": float(np.mean(std_pred)),
                "Test_Uncertainty": float(np.mean(std_pred))
            }
            pd.DataFrame([all_metrics]).to_csv(metrics_csv, index=False)
            results.append(all_metrics)

            timing[f"Fold_{fold_id}"] = time.time() - start
            tf.keras.backend.clear_session()
            gc.collect()

        # Fold summary files and graphs
        results_df = pd.DataFrame(results)
        if "Configuration" not in results_df.columns:
            results_df["Configuration"] = "OptimizedModel"
        results_df.to_csv(f"{PLOTS_DIR}/all_fold_results.csv", index=False)
        plot_fold_comparison(results_df)

        pd.DataFrame(list(timing.items()),
                     columns=["Stage", "Time (s)"]).to_csv(
                         f"{PLOTS_DIR}/timing_results.csv", index=False)
        print("Project COMPLETED")
        logging.info("Pipeline completed")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()