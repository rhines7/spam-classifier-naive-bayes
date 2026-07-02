"""Project paths and reproducibility settings."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "spamassassin"
FIGURES_DIR = PROJECT_ROOT / "figures"

np.random.seed(42)
