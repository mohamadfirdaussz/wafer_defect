# """
# helper.py
# ────────────────────────────────────────────────────────────────────────
# WM-811K Pipeline Utilities & Configuration

# ### 🎯 PURPOSE
# Centralizes configuration (file paths, constants) and shared utility 
# functions (logging, timers) used across the orchestration pipeline.

# ### ⚙️ CONTENTS
# 1. CONFIG: Dictionary containing all absolute paths.
# 2. setup_logging: Standardizes console and file logging.
# 3. ExecutionTimer: Context manager to track how long steps take.
# ────────────────────────────────────────────────────────────────────────
# """

# import os
# import logging
# import time
# from contextlib import contextmanager

# # ───────────────────────────────────────────────
# # 1. GLOBAL CONFIGURATION
# # ───────────────────────────────────────────────
# # Update these paths to match your local environment exactly
# BASE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1"

# CONFIG = {
#     "PATHS": {
#         "RAW_DATA": os.path.join(BASE_DIR, "datasets", "LSWMD.pkl"),
#         "DATA_LOADER_OUT": os.path.join(BASE_DIR, "data_loader_results"),
#         "FEATURE_OUT": os.path.join(BASE_DIR, "Feature_engineering_results"),
#         "PREPROC_OUT": os.path.join(BASE_DIR, "preprocessing_results"),
#         "SELECTION_OUT": os.path.join(BASE_DIR, "feature_selection_results"),
#         "MODEL_OUT": os.path.join(BASE_DIR, "model_artifacts"),
#     },
#     "FILES": {
#         "CLEAN_DATA": "cleaned_full_wm811k.npz",
#         "FEATURES_CSV": "features_dataset.csv",
#         "MODEL_READY": "model_ready_data.npz",
#         "EXPANDED_DATA": "data_track_4E_Full_Expansion_expanded.npz",
#         "MASTER_RESULTS": "master_model_comparison.csv"
#     },
#     "PARAMS": {
#         "IMG_SIZE": (64, 64),
#         "TEST_SIZE": 0.2,
#         "RANDOM_SEED": 42
#     }
# }

# # ───────────────────────────────────────────────
# # 2. LOGGING UTILITY
# # ───────────────────────────────────────────────
# def setup_logging(name: str, log_file: str = "pipeline.log") -> logging.Logger:
#     """
#     Sets up a standardized logger that writes to both console and a file.
    
#     **Log Format:**
#     - File: `Timestamp - LoggerName - Level - Message`
#     - Console: `>> Message` (Simplified for readability)

#     Args:
#         name (str): Name of the logger (usually `__name__`).
#         log_file (str): Filename for the log output (e.g., 'pipeline.log').

#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
    
#     # Create handlers if they don't exist
#     if not logger.handlers:
#         # File Handler
#         fh = logging.FileHandler(log_file)
#         fh.setLevel(logging.INFO)
#         fh_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         fh.setFormatter(fh_fmt)
        
#         # Console Handler
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)
#         ch_fmt = logging.Formatter('>> %(message)s')
#         ch.setFormatter(ch_fmt)
        
#         logger.addHandler(fh)
#         logger.addHandler(ch)
        
#     return logger

# # ───────────────────────────────────────────────
# # 3. TIMING UTILITY
# # ───────────────────────────────────────────────
# @contextmanager
# def execution_timer(task_name: str, logger: logging.Logger = None):
#     """
#     Context manager to measure execution time of a block of code.
    
#     Wraps code blocks to automatically print "Starting..." and "Finished in X seconds"
#     messages. Useful for profiling long-running ML stages.

#     Usage:
#         ```python
#         with execution_timer("Data Loading", logger):
#             load_data()
#         ```

#     Args:
#         task_name (str): Human-readable name of the task.
#         logger (logging.Logger, optional): Logger to use. If None, prints to stdout.
#     """
#     start_time = time.time()
#     if logger:
#         logger.info(f"⏳ Starting: {task_name}...")
#     else:
#         print(f"⏳ Starting: {task_name}...")
        
#     try:
#         yield
#     finally:
#         elapsed = time.time() - start_time
#         msg = f"✅ Finished: {task_name} in {elapsed:.2f} seconds."
#         if logger:
#             logger.info(msg)
#         else:
#             print(msg)


"""
helper.py
────────────────────────────────────────────────────────────────────────
WM-811K Pipeline Utilities & Configuration

### 🎯 PURPOSE
Centralizes configuration (file paths, constants) and shared utility 
functions (logging, timers) used across the orchestration pipeline.

### ⚙️ CONTENTS
1. CONFIG: Dictionary containing all absolute paths.
2. setup_logging: Standardizes console and file logging.
3. ExecutionTimer: Context manager to track how long steps take.
────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import time
from contextlib import contextmanager

# ───────────────────────────────────────────────
# 1. GLOBAL CONFIGURATION
# ───────────────────────────────────────────────
# Update these paths to match your local environment exactly
BASE_DIR = r"C:\Users\user\OneDrive - ums.edu.my\FYP 1"

CONFIG = {
    "PATHS": {
        "RAW_DATA": os.path.join(BASE_DIR, "datasets", "LSWMD.pkl"),
        "DATA_LOADER_OUT": os.path.join(BASE_DIR, "data_loader_results"),
        "FEATURE_OUT": os.path.join(BASE_DIR, "Feature_engineering_results"),
        "PREPROC_OUT": os.path.join(BASE_DIR, "preprocessing_results"),
        "SELECTION_OUT": os.path.join(BASE_DIR, "feature_selection_results"),
        "MODEL_OUT": os.path.join(BASE_DIR, "model_artifacts"),
    },
    "FILES": {
        "CLEAN_DATA": "cleaned_full_wm811k.npz",
        "FEATURES_CSV": "features_dataset.csv",
        "MODEL_READY": "model_ready_data.npz",
        "EXPANDED_DATA": "data_track_4E_Full_Expansion_expanded.npz",
        "MASTER_RESULTS": "master_model_comparison.csv"
    },
    "PARAMS": {
        "IMG_SIZE": (64, 64),
        "TEST_SIZE": 0.2,
        "RANDOM_SEED": 42
    }
}

# ───────────────────────────────────────────────
# 2. LOGGING UTILITY
# ───────────────────────────────────────────────
def setup_logging(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Sets up a logger that writes to both console and a file.
    
    Args:
        name (str): Name of the logger (usually __name__).
        log_file (str): Filename for the log output.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers if they don't exist
    if not logger.handlers:
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_fmt)
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_fmt = logging.Formatter('>> %(message)s')
        ch.setFormatter(ch_fmt)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

# ───────────────────────────────────────────────
# 3. TIMING UTILITY
# ───────────────────────────────────────────────
@contextmanager
def execution_timer(task_name: str, logger: logging.Logger = None):
    """
    Context manager to measure execution time of a block of code.
    
    Usage:
        with execution_timer("Data Loading", logger):
            load_data()
    """
    start_time = time.time()
    if logger:
        logger.info(f"⏳ Starting: {task_name}...")
    else:
        print(f"⏳ Starting: {task_name}...")
        
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        msg = f"✅ Finished: {task_name} in {elapsed:.2f} seconds."
        if logger:
            logger.info(msg)
        else:
            print(msg)