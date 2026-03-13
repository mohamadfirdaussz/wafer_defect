# -*- coding: utf-8 -*-
"""
config.py
────────────────────────────────────────────────────────────────────────
Centralized Configuration for WM-811K Wafer Defect Classification Pipeline
────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import logging
from pathlib import Path

# ───────────────────────────────────────────────
# 1️⃣ PATH CONFIGURATION
# ───────────────────────────────────────────────

# Base directory (The directory containing this script)
BASE_DIR = Path(__file__).resolve().parent

# Project Root (One level up from ml_flow)
PROJECT_ROOT = BASE_DIR.parent

# Input Data Path
# NOTE: Update this path if your dataset is in a different location.
# Currently pointing to the user's external OneDrive folder as per original code.
# Ideally, move 'LSWMD.pkl' to PROJECT_ROOT/datasets/ for portability.
RAW_DATA_PATH = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1\datasets\LSWMD.pkl"

# Output Directories (Relative to Project Root for better organization)
# Or keep them in OneDrive if strictly required, but here we default to relative
# for reproducibility.
# Uncomment the OneDrive paths below if you prefer the old location.

# RESULTS_ROOT = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1" 
RESULTS_ROOT = PROJECT_ROOT  # Defaulting to inside the project

DATA_LOADER_RESULTS_DIR = RESULTS_ROOT / "data_loader_results"
FEATURE_ENGINEERING_DIR = RESULTS_ROOT / "Feature_engineering_results"
PREPROCESSING_DIR = RESULTS_ROOT / "preprocessing_results"
FEATURE_SELECTION_DIR = RESULTS_ROOT / "feature_selection_results"
MODEL_ARTIFACTS_DIR = RESULTS_ROOT / "model_artifacts"

# Result Files
CLEANED_DATA_FILE = DATA_LOADER_RESULTS_DIR / "cleaned_full_wm811k.npz"
FEATURES_FILE_CSV = FEATURE_ENGINEERING_DIR / "features_dataset.csv"
FEATURES_FILE_PARQUET = FEATURE_ENGINEERING_DIR / "features_dataset.parquet"
SCALER_FILE = PREPROCESSING_DIR / "standard_scaler.joblib"
MODEL_READY_DATA_FILE = PREPROCESSING_DIR / "model_ready_data.npz"
EXPANDED_DATA_FILE = FEATURE_SELECTION_DIR / "data_track_4E_Full_Expansion_expanded.npz"

# ───────────────────────────────────────────────
# 2️⃣ HYPERPARAMETERS
# ───────────────────────────────────────────────

# General
RANDOM_SEED = 42

# Stage 1: Data Loading
TARGET_SIZE = (64, 64)

# Stage 2: Feature Engineering
N_RADON_THETA = 72
RADON_OUTPUT_POINTS = 20
N_JOBS = -1

# Stage 3: Preprocessing
TEST_SPLIT_SIZE = 0.2
TARGET_SAMPLES_PER_CLASS = 500

# Stage 3.5: Feature Expansion
COMBINATION_DEGREE = 2
INCLUDE_BIAS = False

# Stage 4: Feature Selection
N_PREFILTER = 1000
N_FEATURES_RFE = 25
N_FEATURES_RF = 25

# Stage 5: Model Tuning
N_FOLDS = 3

# ───────────────────────────────────────────────
# 3️⃣ LOGGING CONFIGURATION
# ───────────────────────────────────────────────

def configure_logging(name):
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)
