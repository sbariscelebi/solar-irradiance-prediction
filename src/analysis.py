import shap
import optuna
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging
from utils import save_fig_multi, PLOTS_DIR, MODEL_NAME  # Import from utils

def plot_wavelet_components(
    df_original: pd.DataFrame,
    wav_cols: list,
    sample_col: str = "ghi",
    plots_dir: str = PLOTS_DIR,
    wavelet: str = "db4",
    best_components: list = None  # Best wavelet component names as parameter
) -> None:
    """ Plots wavelet components and original in controlled order.
    - best_components: ['ghi_cA', 'ghi_cD1', ...] Determined by Optuna.
    """
    # If best_components not provided, use all existing wavelet components
    if best_components is None:
        comp_cols = [c for c in wav_cols if c.startswith(f"{sample_col}_")]
        # Default: cA, then Original, then cD1, cD2, ...
        order = sorted(comp_cols)
    else:
        # Only selected components in order
        order = best_components

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.set_facecolor("white")

    # 1. (back) Component-1: (e.g. cA)
    if len(order) > 0:
        ax.plot(
            df_original.index,
            df_original[order[0]],
            label=order[0],
            linewidth=2,
            color="#E69F00",  # orange
            zorder=1
        )

    # 2. Middle: Original data
    ax.plot(
        df_original.index,
        df_original[sample_col],
        label="Original",
        linewidth=2.5,
        color="#1f77b4",
        zorder=2
    )

    # 3. Top: Component-2 (e.g. cD1 or determined by Optuna)
    if len(order) > 1:
        ax.plot(
            df_original.index,
            df_original[order[1]],
            label=order[1],
            linewidth=2,
            color="#9B59B6",  # purple
            zorder=3
        )

    ax.set_title(f"Wavelet Components", fontweight="bold", fontsize=20, pad=14, color="black")
    ax.set_xlabel("Time", fontsize=16, color="black")
    ax.set_ylabel("Irradiance (W/m²)", fontsize=16, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")
    ax.grid(True, color="lightblue", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=14)
    fig.tight_layout(pad=2)

    path_no_ext = os.path.join(plots_dir, f"wavelet_components_{sample_col}")
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    logging.info(f"Wavelet components plot saved: {path_no_ext}.png and {path_no_ext}.svg")

def plot_wavelet_energy(
    df_wave: pd.DataFrame,
    wav_cols: list,
    plots_dir: str = PLOTS_DIR
) -> None:
    """ Visualizes the total energy (∑x²) of wavelet bands using a bar plot.
    The plot is saved in both PNG and SVG formats at 300 dpi.
    """
    # ────────── Calculate energy ──────────────────────────────────────────
    band_energy = (
        df_wave[wav_cols]
        .apply(lambda col: np.sum(np.square(col.values)))
        .groupby(lambda c: c.split("_")[-1])  # cA, cD1…
        .sum()
        .sort_index()
    )

    # ────────── Create plot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_facecolor("white")  # White background
    colors = sns.color_palette("viridis", len(band_energy))  # Improved color palette

    ax.bar(
        band_energy.index,
        band_energy.values,
        color=colors,
        edgecolor="black",
        linewidth=1.0  # Slightly thicker edges for better definition
    )

    # Set title and labels
    ax.set_title(
        "Wavelet Band Energy (∑x²)",
        fontweight="bold",
        fontsize=20,
        pad=14,
        color="black"  # Bold black title
    )
    ax.set_xlabel("Band", fontsize=18, color="black")   # 14 pt axis label
    ax.set_ylabel("Energy", fontsize=18, color="black") # 14 pt axis label
    ax.tick_params(axis="both", labelsize=16, colors="black")  # Black tick labels
    ax.grid(True, color="lightblue", alpha=0.5)  # Light blue grid
    ax.spines["top"].set_visible(False)   # Remove top spine
    ax.spines["right"].set_visible(False) # Remove right spine

    # Legend settings (for potential future use)
    if ax.get_legend():
        ax.get_legend().set_frame_on(True)
        ax.get_legend().set_facecolor("white")    # White legend background
        ax.get_legend().set_edgecolor("black")    # Black legend border
        ax.get_legend().set_fontsize(14)          # 12 pt legend text

    fig.tight_layout(pad=2)

    # ────────── Save plot (PNG + SVG, 300 dpi) ────────────────────────────
    path_no_ext = os.path.join(plots_dir, "wavelet_band_energy")
    save_fig_multi(fig, path_no_ext, dpi=300)  # .png & .svg
    plt.close(fig)

    logging.info(
        f"Wavelet band energy plot saved: {path_no_ext}.png and {path_no_ext}.svg"
    )

def plot_time_series_prediction(
    df_original: pd.DataFrame,
    y_test:      np.ndarray,
    y_pred:      np.ndarray,
    fold:        int,
    model_name:  str = MODEL_NAME,
    plots_dir:   str = PLOTS_DIR,
    days:        int = 20          # Added: first 20 days
) -> None:
    """ Plots the actual and predicted time series values for the first `days` of the test set.
    Limits y-axis to [0, 300] for clear visualization.
    """
    try:
        offset         = 24  # MANUAL_PARAMS["time_steps"]
        start_idx      = len(df_original) - len(y_test) - offset
        datetime_index = df_original.index[start_idx + offset : start_idx + offset + len(y_test)]

        # Find first 20 days
        first_date = datetime_index[0]
        day_end = first_date + pd.Timedelta(days=days)
        # Mask: first 20 days
        mask = datetime_index < day_end
        datetime_index_20d = datetime_index[mask]
        y_test_20d = y_test[mask]
        y_pred_20d = y_pred[mask]

        fig, ax = plt.subplots(figsize=(15, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        ax.plot(
            datetime_index_20d,
            y_test_20d,
            label="Actual",
            linewidth=2,
            color="#000000"
        )
        ax.plot(
            datetime_index_20d,
            y_pred_20d,
            label=model_name,
            linewidth=2,
            linestyle="--",
            color="#00CED1"
        )

        # Title and axes
        ax.set_title(
            f"Time Series Prediction Performance – First {days} Days (Fold {fold})",
            fontweight="bold",
            fontsize=20,
            pad=14,
            color='black'
        )
        ax.set_xlabel("Date", fontsize=18, color='black')
        ax.set_ylabel("GHI (W/m²)", fontsize=18, color='black')
        ax.set_ylim(0, 300)  # Y axis limited to 0–300

        legend = ax.legend(loc='upper left', frameon=True, fontsize=14)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        ax.tick_params(axis="x", rotation=45, labelsize=16, colors='black')
        ax.tick_params(axis="y", labelsize=16, colors='black')
        ax.set_yticks(np.arange(0, 301, 50))  # 0, 50, 100, ... 300
        ax.grid(True, color='lightblue')
        fig.tight_layout(pad=2)

        path_no_ext = os.path.join(
            plots_dir, f"{model_name}_timeplot_fold_{fold}_first{days}days"
        )
        save_fig_multi(fig, path_no_ext, dpi=300)
        plt.close(fig)

        logging.info(
            f"Time-series prediction plot (first {days} days) saved: {path_no_ext}.png and {path_no_ext}.svg"
        )
    except Exception as e:
        logging.error(f"Error in plot_time_series_prediction: {e}")

def compute_shap_values(model, fold_info, n_back=10):
    """ Computes SHAP values using GradientExplainer (np.ndarray, shape = [N, F]).
    Args:
        model: Trained Keras model.
        fold_info (dict): Fold data (X_train_seq, X_test_seq, etc.).
        n_back (int): Number of samples for SHAP.
    Returns:
        np.ndarray: SHAP values (shape: [n_back, time_steps, num_features]).
    """
    try:
        # Check data sizes
        if len(fold_info['X_train_seq']) < n_back or len(fold_info['X_test_seq']) < n_back:
            n_back = min(len(fold_info['X_train_seq']), len(fold_info['X_test_seq']))
            logging.warning(f"n_back reduced to {n_back} due to insufficient data size.")

        # Mixed-precision errors prevention with float32
        background = fold_info['X_train_seq'][:n_back].astype("float32")
        test_data = fold_info['X_test_seq'][:n_back].astype("float32")

        explainer = shap.GradientExplainer(model, background)
        shap_vals = explainer(test_data)  # >=0.44 API

        # Output may be list or Explanation obj → convert to ndarray
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]  # Assume single output model
        elif hasattr(shap_vals, "values"):
            shap_vals = shap_vals.values

        # Check SHAP values shape
        logging.info(f"SHAP values shape: {shap_vals.shape}")
        return shap_vals
    except Exception as e:
        logging.error(f"Error in compute_shap_values: {str(e)}")
        raise

def select_shap_features(shap_values, feature_names, percentile):
    mean_impact = np.abs(shap_values).mean(axis=0)
    threshold = np.percentile(mean_impact, 100 * percentile)
    selected = [name for name, impact in zip(feature_names, mean_impact) if impact >= threshold]
    logging.info(f"SHAP selected features: {selected}")
    return selected

def visualize_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME
) -> None:
    """ Plots the top `top_n` features by mean absolute SHAP value in a horizontal bar chart.
    The plot is saved in both PNG and SVG formats at 300 dpi, and the data is also exported as a CSV.
    Saved files:
    plots/<model_name>_shap_barplot.png
    plots/<model_name>_shap_barplot.svg
    plots/<model_name>_top_shap_features.csv
    """
    # ────────── Data preparation ──────────────────────────────────────────
    shap_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Mean_SHAP": np.abs(shap_values).mean(axis=0)
        })
        .sort_values("Mean_SHAP", ascending=False)
        .head(top_n)
    )

    # ────────── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor("white")  # Background white

    sns.barplot(
        data=shap_df,
        x="Mean_SHAP",
        y="Feature",
        palette="flare",
        edgecolor="black",
        linewidth=0.8,
        ax=ax
    )

    ax.set_title(
        f"Top {top_n} SHAP Features",
        fontweight="bold",
        fontsize=20,
        pad=14,
        color="black"
    )
    ax.set_xlabel("Mean |SHAP|", fontsize=18, color="black")
    ax.set_ylabel("Feature", fontsize=18, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")
    ax.grid(True, color="lightblue", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2)

    # ────────── Save (PNG + SVG, 300 dpi) ────────────────────────────────
    path_no_ext = os.path.join(plots_dir, f"{model_name}_shap_barplot")
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    # ────────── CSV export ───────────────────────────────────────────────
    csv_path = os.path.join(plots_dir, f"{model_name}_top_shap_features.csv")
    shap_df.to_csv(csv_path, index=False)

    # ────────── Logging ──────────────────────────────────────────────────
    logging.info(
        "SHAP bar plot saved: %s.png, %s.svg and CSV exported: %s",
        path_no_ext, path_no_ext, csv_path
    )

def analyze_pca(
    df: pd.DataFrame,
    selected_features: list[str],
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME
) -> None:
    """ Applies PCA on features selected by SHAP; plots the cumulative explained variance curve 
    and saves it in both PNG and SVG formats at 300 dpi.
    """
    # Determine columns
    norm_cols = {
        col.lower().replace("temperature", "temp").replace("dew_point", "dew"): col
        for col in df.columns
    }
    selected_cols = {
        norm_cols.get(strip_time_suffix(f).lower(), None)
        for f in selected_features if strip_time_suffix(f)
    }
    selected_cols = {c for c in selected_cols if c in df.columns}
    if not selected_cols:
        selected_cols = set(df.columns) - {'solar_energy'}

    # PCA
    X_scaled = StandardScaler().fit_transform(df[list(selected_cols)].dropna())
    pca = PCA().fit(X_scaled)
    explained_var = np.cumsum(pca.explained_variance_ratio_) * 100

    # Plot
    fig = plt.figure(figsize=(9, 5))
    fig.set_facecolor("white")
    ax = plt.gca()

    # Cumulative variance
    ax.plot(
        range(1, len(explained_var) + 1),
        explained_var,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Cumulative Variance"
    )

    # 95% threshold line
    ax.axhline(
        y=95,
        color="#FF0000",  # e.g. #FF0000 for bright red (ideal for SCI)
        linestyle="--",
        linewidth=2,
        label="95% Threshold"
    )

    ax.set_title(
        "PCA – Cumulative Explained Variance",
        fontweight="bold",
        fontsize=20,
        pad=14,
        color="black"
    )
    ax.set_xlabel("Number of Components", fontsize=18, color="black")
    ax.set_ylabel("Cumulative Variance (%)", fontsize=18, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")
    ax.grid(True, color="lightblue", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend (white background, black edge, 12 pt)
    legend = ax.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=14)
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout(pad=2)

    # Save
    path_no_ext = os.path.join(plots_dir, f"{model_name}_pca_variance")
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    logging.info(
        "PCA cumulative explained variance plot saved: %s.png and %s.svg",
        path_no_ext, path_no_ext
    )

def plot_results(
    history: "tensorflow.python.keras.callbacks.History",
    y_test: np.ndarray,
    y_pred: np.ndarray,
    fold: int,
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME
) -> None:
    """ 
    • Panel-1: Training and validation loss
    • Panel-2: Actual vs. prediction scatter (bright blue points + full red regression line)
    • Panel-3: Residual distribution (histogram + KDE)
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.set_facecolor("white")  # Background white

    # Panel-1: Loss curves
    axs[0].plot(history.history["loss"], label="Training", linewidth=2)
    axs[0].plot(history.history["val_loss"], label="Validation", linewidth=2, linestyle="--")
    axs[0].set_title("Loss Curves", fontweight="bold", fontsize=20, pad=10, color="black")
    axs[0].set_xlabel("Epoch", fontsize=18, color="black")
    axs[0].set_ylabel("Loss", fontsize=18, color="black")
    axs[0].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=14)

    # Panel-2: Prediction vs. Actual
    axs[1].scatter(
        y_test,
        y_pred,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
        color="#00BFFF",  # Bright blue
        label="Predictions"
    )
    min_v, max_v = y_test.min(), y_test.max()
    axs[1].plot(
        [min_v, max_v],
        [min_v, max_v],
        linestyle="--",
        color="red",  # Full red regression line
        linewidth=2,
        label="y = x"
    )
    axs[1].set_title("Prediction vs. Actual", fontweight="bold", fontsize=20, pad=10, color="black")
    axs[1].set_xlabel("Actual", fontsize=18, color="black")
    axs[1].set_ylabel("Prediction", fontsize=18, color="black")
    axs[1].legend(frameon=True, facecolor="white", edgecolor="black", fontsize=14)

    # Panel-3: Residual Distribution
    sns.histplot(y_test - y_pred, kde=True, ax=axs[2], edgecolor="black", linewidth=0.4, color="#00BFFF")
    axs[2].set_title("Residual Distribution", fontweight="bold", fontsize=20, pad=10, color="black")
    axs[2].set_xlabel("Residual", fontsize=18, color="black")
    axs[2].set_ylabel("Frequency", fontsize=18, color="black")

    # Common style settings
    for ax in axs:
        ax.tick_params(axis="both", labelsize=16, colors="black")
        ax.grid(True, color="lightblue", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Model Results – Fold {fold}", fontweight="bold", fontsize=20, y=1.03, color="black")
    fig.tight_layout(pad=2)

    # Save (PNG + SVG, 300 dpi)
    path_no_ext = os.path.join(plots_dir, f"{model_name}_fold_{fold}_results")
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    logging.info(
        "Fold %d results plotted: %s.png and %s.svg", fold, path_no_ext, path_no_ext
    )

def plot_hourly_rmse(datetime_index, y_true, y_pred, fold, plots_dir=PLOTS_DIR):
    # Convert data to DataFrame
    df_test = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "hour": datetime_index.hour
    })

    # Hourly RMSE calculation
    rmse_per_hour = df_test.groupby("hour").apply(
        lambda d: np.sqrt(mean_squared_error(d["y_true"], d["y_pred"]))
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(rmse_per_hour.index, rmse_per_hour.values, color="skyblue", edgecolor="black")
    ax.set_xlabel("Hour of Day", fontsize=18, color="black")
    ax.set_ylabel("RMSE", fontsize=18, color="black")
    ax.set_title(f"Hourly RMSE Analysis – Fold {fold}", fontweight="bold", fontsize=18, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")
    ax.grid(True, color="lightgray", alpha=0.5)
    fig.tight_layout(pad=2)

    # Save plot
    path_no_ext = os.path.join(plots_dir, f"hourly_rmse_fold_{fold}")
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    logging.info(f"Hourly RMSE plot saved: {path_no_ext}.png and {path_no_ext}.svg")

def ablation_fold_worker(fold_info, config):
    model = build_model(
        (fold_info['X_train_seq'].shape[1], fold_info['X_train_seq'].shape[2]),
        **{k: 0.0001 for k in ['learning_rate', 'filters1', 'filters2', 'lstm_units1', 'lstm_units2', 'dropout1', 'dropout2', 'dense_units']},
        use_lstm=config['use_lstm'],
        use_dropout=config['use_dropout']
    )
    history, trained_model = train_model(
        model, fold_info['X_train_seq'], fold_info['y_train_seq'],
        fold_info['X_val_seq'], fold_info['y_val_seq'],
        100, 10, fold_info['fold']
    )
    # train_metrics, test_metrics, *_ = evaluate_model(
    #     trained_model,
    #     fold_info['X_train_seq'], fold_info['y_train_seq'],
    #     fold_info['X_test_seq'], fold_info['y_test_seq'],
    #     fold_info['idx_test_seq'],  # ← Added as idx_test
    #     fold_info['scaler_y']  # ← Now correctly passed as scaler_y
    # )
    # _, _, std_pred = evaluate_uncertainty(trained_model, fold_info['X_test_seq'], fold_info['y_test_seq'])
    _, _, std_pred = evaluate_uncertainty(
        trained_model, fold_info['X_test_seq'], fold_info['y_test_seq'], fold_info['scaler_y']
    )
    metrics = {f"Train_{k}": v for k, v in train_metrics.items()}
    metrics.update({f"Test_{k}": v for k, v in test_metrics.items()})
    metrics['Test_Uncertainty'] = np.mean(std_pred)
    metrics['fold'] = fold_info['fold']
    tf.keras.backend.clear_session()
    gc.collect()
    return metrics

def ablation_study(df, selected_features, timing):
    configs = [
        {"name": "Baseline", "use_wavelet": True, "use_shap": True, "use_lstm": True, "use_dropout": True},
        {"name": "No_Wavelet", "use_wavelet": False, "use_shap": True, "use_lstm": True, "use_dropout": True},
        {"name": "No_SHAP", "use_wavelet": True, "use_shap": False, "use_lstm": True, "use_dropout": True},
        {"name": "No_LSTM", "use_wavelet": True, "use_shap": True, "use_lstm": False, "use_dropout": True},
        {"name": "No_Dropout", "use_wavelet": True, "use_shap": True, "use_lstm": True, "use_dropout": False},
    ]
    results = []
    start = time.time()
    for config in configs:
        df_proc = df.copy()
        if config['use_wavelet']:
            df_proc, _ = apply_wavelet(df_proc, wavelet="db4", wavelet_level=2, include_cA=True, include_cD1=True, include_cD2=False)
        df_proc = engineer_features(df_proc)
        if config['use_shap'] and selected_features:
            sel_cols = {base for f in selected_features if (base := strip_time_suffix(f)) in df_proc.columns}
            if not sel_cols:
                logging.warning("No valid columns selected by SHAP for ablation study. Using all features.")
                sel_cols = set(df_proc.columns) - {'solar_energy'}
            df_proc = df_proc[list(sel_cols) + ['solar_energy']].dropna()
            if df_proc.empty:
                raise ValueError("Processed DataFrame is empty after feature selection and dropna.")
            logging.info(f"Ablation study using columns: {sel_cols}")

        # 1) Re-split dataset for each configuration
        fold_data = split_scale_data(
            df_proc, 'solar_energy', 0.2, 24
        )

        # 2) Train folds in GPU-friendly serial manner
        fold_results = []
        for fold_info in fold_data:
            fold_results.append(ablation_fold_worker(fold_info, config))

        for fold_result in fold_results:
            fold_result['Configuration'] = config['name']
            results.append(fold_result)

    timing['Ablation_Study'] = time.time() - start
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{PLOTS_DIR}/ablation_study_results.csv')
    logging.info("Ablation study completed")
    return results_df

def plot_ablation_barplot(
    ablation_df: pd.DataFrame,
    metric: str = "Test_RMSE",
    plots_dir: str = "plots",
    model_name: str = "Model"
) -> None:
    """ Visualizes ablation study results as a bar plot with ±1 SD error bars.
    Saves the figure as PNG and SVG with 300 dpi resolution.
    Saved files
    -----------
    plots/<model_name>_ablation_<metric>.png
    plots/<model_name>_ablation_<metric>.svg
    """
    # Average and standard deviation calculation
    mean_vals = ablation_df.groupby("Configuration")[metric].mean()
    std_vals = ablation_df.groupby("Configuration")[metric].std()
    order = mean_vals.index  # Fixed order

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')  # Figure background white
    ax.set_facecolor('white')  # Axis background white

    bars = ax.bar(
        x=np.arange(len(order)),
        height=mean_vals.loc[order].values,
        yerr=std_vals.loc[order].values,
        capsize=6,
        color=sns.color_palette("tab10", len(order)),  # Nicer colors
        edgecolor="black",
        linewidth=0.9
    )

    ax.set_title(
        f"Ablation Study – {metric} (±1 SD)",
        fontweight="bold",
        fontsize=20,
        pad=14,
        color='black'  # Title dark black
    )
    ax.set_xlabel("Configuration", fontsize=18, color='black')  # X axis 14 pt, black
    ax.set_ylabel(metric, fontsize=18, color='black')  # Y axis 14 pt, black
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=16, color='black')
    ax.tick_params(axis="y", labelsize=16, colors='black')  # Y tick labels black
    ax.grid(axis="y", color='lightblue', alpha=0.5)  # Grid light blue
    fig.tight_layout(pad=2)

    # Save (PNG + SVG, 300 dpi)
    path_no_ext = os.path.join(
        plots_dir, f"{model_name}_ablation_{metric}"
    )
    save_fig_multi(fig, path_no_ext, dpi=300)  # .png & .svg
    plt.close(fig)

    logging.info(
        "Ablation barplot saved: %s.png and %s.svg", path_no_ext, path_no_ext
    )

def get_or_create_study(study_name: str, rng: np.random.Generator) -> optuna.Study:
    """ Prepares or continues a persistent (SQLite) Optuna study. """
    storage_uri = f"sqlite:///checkpoints/{study_name}.db"
    return optuna.create_study(
        study_name = study_name,
        storage = storage_uri,
        direction = "minimize",
        sampler = optuna.samplers.TPESampler(seed=int(rng.integers(0, 2**32))),
        load_if_exists = True  # <-- critical!
    )

def optimize_params(fold_info, rng) -> dict:
    """ Runs hyper-parameter search for a single fold persistently. """
    study_name = f"optuna_fold_{fold_info['fold']}"
    study = get_or_create_study(study_name, rng)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "filters1": trial.suggest_categorical("filters1", [32, 64, 128]),
            "filters2": trial.suggest_categorical("filters2", [32, 64, 128]),
            "lstm_units1": trial.suggest_categorical("lstm_units1", [64, 128, 256]),
            "lstm_units2": trial.suggest_categorical("lstm_units2", [32, 64, 128]),
            "dropout1": trial.suggest_float("dropout1", 0.1, 0.5),
            "dropout2": trial.suggest_float("dropout2", 0.1, 0.5),
            "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
        }
        # — Train model —
        model = build_model(
            (fold_info["X_train_seq"].shape[1], fold_info["X_train_seq"].shape[2]),
            **params
        )
        history, _ = train_model(
            model, fold_info["X_train_seq"], fold_info["y_train_seq"],
            fold_info["X_val_seq"], fold_info["y_val_seq"],
            100, 10, fold_info["fold"]
        )
        y_pred = model.predict(fold_info["X_test_seq"], verbose=0).flatten()
        tf.keras.backend.clear_session()
        del model, history
        gc.collect()
        # — Value to optimize —
        return np.sqrt(mean_squared_error(fold_info["y_test_seq"], y_pred))

    # Continue from where left off; previous trials already in study.db
    study.optimize(objective, n_trials=20, n_jobs=1)
    plot_optuna_history(study, fold_info["fold"])
    logging.info("Optuna best params: %s", study.best_params)
    return study.best_params

# Metric names suitable for literature
def prettify_metric(metric):
    table = {
        "Test_RMSE_orig": "Test RMSE (Original)",
        "Test_MAE_orig": "Test MAE (Original)",
        "Test_R2_orig": "Test R² (Original)",  # R²
        "Test_MAPE_orig": "Test MAPE (Original)",
        "Test_Uncertainty": "Test Uncertainty (MC Dropout)",
        "Test_RMSE_scaled": "Test RMSE (Scaled)",
        "Test_MAE_scaled": "Test MAE (Scaled)",
        "Test_R2_scaled": "Test R² (Scaled)",
        "Test_MAPE_scaled": "Test MAPE (Scaled)",
        # Add more if needed
    }
    return table.get(metric, metric.replace("_", " ").title())

def plot_fold_comparison(
    results_df: pd.DataFrame,
    metrics: list[str] | None = None,
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME
) -> None:
    """ Compares test metrics for each fold and configuration using a bar plot.
    Plots are saved as both PNG and SVG with 300 dpi resolution.
    """
    if metrics is None:
        metrics = ["Test_RMSE_orig", "Test_MAE_orig", "Test_R2_orig", "Test_MAPE_orig", "Test_Uncertainty"]

    # Fold label + combined key (in English)
    results_df = results_df.copy()
    results_df["Fold"] = [f"Fold {i+1}" for i in range(len(results_df))]
    results_df["FoldCfg"] = results_df["Fold"] + " | " + results_df["Configuration"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')

        mean_vals = results_df.groupby("FoldCfg")[metric].mean()
        std_vals = results_df.groupby("FoldCfg")[metric].std()
        order = mean_vals.index

        sns.barplot(
            data=results_df,
            x="FoldCfg",
            y=metric,
            order=order,
            palette="tab10",
            ci=None,
            ax=ax,
            edgecolor="black",
            linewidth=0.8
        )

        x_pos = np.arange(len(order))
        ax.errorbar(
            x=x_pos,
            y=mean_vals.loc[order].values,
            yerr=std_vals.loc[order].values,
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1
        )

        # --- Proper metric names for title and axes ---
        pretty_metric = prettify_metric(metric)
        ax.set_title(
            f"{pretty_metric} Comparison Across Folds",
            fontweight="bold",
            fontsize=20,
            pad=14,
            color='black'
        )
        ax.set_xlabel("Fold | Configuration", fontsize=18, color='black')
        ax.set_ylabel(pretty_metric, fontsize=18, color='black')
        ax.set_xticklabels(order, rotation=25, ha="right", fontsize=16, color='black')
        ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()], fontsize=16, color='black')
        ax.grid(color='lightblue', linestyle='-', linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax.get_legend():
            legend = ax.get_legend()
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            for text in legend.get_texts():
                text.set_color('black')
                text.set_fontsize(14)

        fig.tight_layout(pad=2)

        path_no_ext = os.path.join(
            plots_dir, f"{model_name}_{metric}_fold_comparison"
        )
        save_fig_multi(fig, path_no_ext, dpi=300)
        plt.close(fig)

        logging.info(
            "%s fold comparison plot saved: %s.png and %s.svg",
            pretty_metric, path_no_ext, path_no_ext
        )

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME,
    percentile: float = 0.20,  # Top 20%
    max_display: int = 20,  # Max features to show in plot
    plot_type: str = "dot",
    verbose: bool = True
) -> list[str]:
    """ Evaluates only the top 20% features.
    - If this group > max_display, only first max_display features shown in graph.
    - Features not in graph (but in top 20%) are printed.
    - Function returns list of *features not in graph but selected*.
    """
    # 1) Importance scores
    mean_imp = np.abs(shap_values).mean(axis=0)

    # 2) 20% threshold
    threshold = np.percentile(mean_imp, 100 * (1 - percentile))  # Highest 20%
    selected_mask = mean_imp >= threshold
    selected_idx_all = np.where(selected_mask)[0]

    # 3) Sort by importance
    order_all = np.argsort(mean_imp[selected_idx_all])[::-1]
    selected_idx_sorted = selected_idx_all[order_all]

    # 4) To show and exclude
    show_idx = selected_idx_sorted[:max_display]
    excluded_idx = selected_idx_sorted[max_display:]
    show_features = [feature_names[i] for i in show_idx]
    excluded_features = [feature_names[i] for i in excluded_idx]

    # 5) Plot settings
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'font.family': 'Times New Roman',
        'font.size': 16,
    })

    shap.summary_plot(
        shap_values[:, show_idx],
        features=X_sample[:, show_idx],
        feature_names=show_features,
        plot_type=plot_type,
        show=False,
        max_display=len(show_features),
        cmap='icefire'
    )

    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(12, min(0.35 * len(show_features) + 2, 15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlabel("SHAP Value", fontsize=18, color="black")
    ax.set_ylabel("Feature", fontsize=18, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")

    fig.suptitle(
        f"SHAP Summary Plot ({plot_type.capitalize()}) – Top {percentile*100:.0f}% (max {max_display})",
        fontweight="bold",
        fontsize=20,
        color="black",
        y=1.03
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Color bar
    if len(fig.axes) > 1:
        try:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.set_label("Feature Value", fontsize=16, color="black")
                cbar.ax.tick_params(labelsize=16, colors="black")
                cbar.outline.set_edgecolor("black")
                cbar.outline.set_linewidth(1)
                fig.axes[1].set_facecolor("white")
        except Exception:
            pass

    fig.tight_layout(pad=2)

    path_no_ext = os.path.join(plots_dir, f"{model_name}_shap_summary_{plot_type}")
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    logging.info(
        "SHAP summary plot saved: %s.png and %s.svg", path_no_ext, path_no_ext
    )

    # 6) Print excluded
    if verbose and excluded_features:
        print("\n[INFO] Features not in SHAP dot graph (but in top 20%):")
        for f in excluded_features:
            print(f" - {f}")

    return excluded_features

def plot_uncertainty_band(
    datetime_index: pd.DatetimeIndex,
    y_test: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    fold: int,
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME,
    y_max: int = 300,
    days: int = 20
) -> None:
    """ MC-Dropout mean and ±1σ uncertainty band (first `days` days).
    Uncertainty band drawn more prominently.
    """
    # Only first N days:
    first_date = datetime_index[0]
    last_date = first_date + pd.Timedelta(days=days)
    mask = datetime_index < last_date
    datetime_index = datetime_index[mask]
    y_test = y_test[mask]
    mean_pred = mean_pred[mask]
    std_pred = std_pred[mask]

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # True values (black line)
    ax.plot(
        datetime_index,
        y_test,
        label="Actual",
        linewidth=2,
        color="#000000"
    )

    # Mean prediction (dark blue line)
    ax.plot(
        datetime_index,
        mean_pred,
        label="Prediction (Mean)",
        linewidth=2.2,
        color="#0072B2"
    )

    # More prominent uncertainty band (darker and opaque)
    ax.fill_between(
        datetime_index,
        mean_pred - std_pred,
        mean_pred + std_pred,
        color="#F75C5C",  # More saturated red/pink
        alpha=0.45,  # More opaque and prominent
        label="±1σ Uncertainty",
        zorder=1  # Band below
    )

    ax.set_title(
        f"MC-Dropout Uncertainty Band – Fold {fold} (First {days} Days)",
        fontweight="bold",
        fontsize=20,
        pad=14,
        color='black'
    )
    ax.set_xlabel("Date", fontsize=18, color='black')
    ax.set_ylabel("GHI (W/m²)", fontsize=18, color='black')
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max+1, 50))
    legend = ax.legend(loc='upper left', frameon=True, fontsize=14)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    ax.tick_params(axis='x', rotation=45, labelsize=16, colors='black')
    ax.tick_params(axis='y', labelsize=16, colors='black')
    ax.grid(True, color='lightblue')
    fig.tight_layout(pad=2)

    path_no_ext = os.path.join(
        plots_dir, f"{model_name}_uncertainty_band_fold_{fold}_first{days}days"
    )
    save_fig_multi(fig, path_no_ext, dpi=300)
    plt.close(fig)

    logging.info(
        f"Uncertainty band plot (first {days} days) saved: {path_no_ext}.png and {path_no_ext}.svg"
    )

def optimize_wavelet_params(df, target_col, fold_info):
    study = optuna.create_study(direction='minimize')
    def objective(trial):
        wavelet = trial.suggest_categorical('wavelet', ['db4', 'sym4', 'coif1'])
        level = trial.suggest_int('wavelet_level', 1, 3)
        df_wav, _ = apply_wavelet(df, wavelet=wavelet, wavelet_level=level, include_cA=True, include_cD1=True, include_cD2=False)
        df_fe = engineer_features(df_wav)
        fold_data = split_scale_data(df_fe, target_col, 0.2, 24)[0]
        model = build_model(
            (fold_data['X_train_seq'].shape[1], fold_data['X_train_seq'].shape[2]),
            **{k: 0.0001 for k in ['learning_rate', 'filters1', 'filters2', 'lstm_units1', 'lstm_units2', 'dropout1', 'dropout2', 'dense_units']}
        )
        history, _ = train_model(model, fold_data['X_train_seq'], fold_data['y_train_seq'], fold_data['X_val_seq'], fold_data['y_val_seq'], 100, 10, fold_data['fold'])
        y_pred = model.predict(fold_data['X_test_seq'], verbose=0).flatten()
        return np.sqrt(mean_squared_error(fold_data['y_test_seq'], y_pred))

    study.optimize(objective, n_trials=20)
    # Save all trials as dataframe
    trials_df = study.trials_dataframe()
    csv_path = os.path.join(PLOTS_DIR, "optuna_wavelet_trials.csv")
    trials_df.to_csv(csv_path, index=False)
    return study.best_params

def plot_optuna_history(
    study: "optuna.study.Study",
    fold: int,
    plots_dir: str = PLOTS_DIR,
    model_name: str = MODEL_NAME
) -> None:
    """ Plots the Optuna optimization history (trial number vs. objective value) and saves the figure in both PNG and SVG format at 300 dpi.
    The "best value" is highlighted with a bright red line.
    All styles (background, font size, legend) are suitable for high-quality scientific publication.
    """
    import optuna.visualization.matplotlib as vis
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import os, logging

    # Create the optimization history plot
    plot_obj = vis.plot_optimization_history(study)

    # Optuna ≥ v3 returns Axes, < v3 returns Figure
    if isinstance(plot_obj, mpl.axes.Axes):
        ax = plot_obj
        fig = ax.get_figure()
    else:
        fig = plot_obj
        ax = fig.axes[0]

    fig.set_size_inches(10, 6)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Style: black text, white bg, axis/tick/label/legend as specified
    ax.set_title(
        f"Optuna Optimization History – Fold {fold}",
        fontweight="bold",
        fontsize=20,
        pad=14,
        color="black"
    )
    ax.set_xlabel("Trial Number", fontsize=18, color="black")
    ax.set_ylabel("Objective Value", fontsize=18, color="black")
    ax.tick_params(axis="both", labelsize=16, colors="black")
    ax.grid(True, color="lightblue", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Highlight best value with a bright red horizontal line
    # Find the lowest y value (best objective) among all data points
    try:
        ydata = []
        xdata = []
        # Scan all line2D in axes
        for line in ax.lines:
            if line.get_label() != "Best Value":
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                break
        if len(xdata) > 0 and len(ydata) > 0:
            best_idx = np.argmin(ydata)
            best_x = xdata[best_idx]
            best_y = ydata[best_idx]
            # Draw a bright red horizontal line at best_y
            ax.axhline(y=best_y, color="#FF0000", linestyle="--", linewidth=3, label="Best Value")
            # Optional: highlight best point
            ax.scatter([best_x], [best_y], color="#FF0000", edgecolor="black", zorder=5)
    except Exception as e:
        logging.warning(f"Could not add best value line: {e}")

    # Legend: white background, black edge, 12pt font
    legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', fontsize=14)
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout(pad=2)

    # Save as PNG and SVG with 300 dpi
    path_no_ext = os.path.join(
        plots_dir, f"{model_name}_optuna_history_fold_{fold}"
    )
    fig.savefig(f"{path_no_ext}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_no_ext}.svg", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logging.info(
        "Optuna optimization history plot saved: %s.png and %s.svg", path_no_ext, path_no_ext
    )