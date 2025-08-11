import os
import time
import logging
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import gc
from pathlib import Path

# Logging setup
logging.basicConfig(filename='plots/pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_plot_style(
    grid_style: str = "whitegrid",
    palette: str = "mako"
) -> None:
    """ Universal typography and color settings: 
        - Background white
        - Texts full black
        - Axis labels: 16 pt, black
        - Title: 20 pt, bold, black
        - Legend: 12 pt, inside white, edge full black
        - Font: Times New Roman
    """
    # Seaborn theme + axes/figure facecolor override!
    sns.set_theme(
        style=grid_style,
        palette=palette,
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "text.color": "black",
            "grid.color": "lightgrey",
        }
    )
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "legend.frameon": True,
        "legend.loc": "best",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "patch.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

mpl.rcParams["font.family"] = "DejaVu Sans"  # or "Arial" or "Liberation Sans"

def save_fig_multi(
    fig: "matplotlib.figure.Figure",
    path_no_ext: str,
    dpi: int = 300
) -> None:
    """ Saves the given figure as both PNG and SVG.
    Args:
        fig        : matplotlib Figure object
        path_no_ext: File path without extension (e.g. 'plots/myplot')
        dpi        : Resolution (for PNG)
    """
    # PNG
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    # SVG (dpi parameter is ineffective in vector format but kept for consistency)
    fig.savefig(f"{path_no_ext}.svg", dpi=dpi, bbox_inches="tight")

def save_checkpoint(data, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Checkpoint saved: {path}")
    except Exception as e:
        logging.error(f"Checkpoint save error: {str(e)}")
        raise

def load_checkpoint(path):
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        logging.error(f"Checkpoint load error: {str(e)}")
        return None