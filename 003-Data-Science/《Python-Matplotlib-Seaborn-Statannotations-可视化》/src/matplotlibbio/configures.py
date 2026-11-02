from loguru import logger
import matplotlib.pyplot as plt

from matplotlib import patches 
from matplotlib import colors


base_config: dict = {
    # === Font ===
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    # === Export ===
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # === Figure ===
    'figure.layout.constrained': True,
    # === Axes ===
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    # === Legend ===
    'legend.fontsize': 6,
}


colors: dict = {
    'blue': '#1f77b4',  
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
}