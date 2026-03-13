# """
# data_loader.py(Stage 1)
# ────────────────────────────────────────────────────────────────────────────────
# WM-811K Wafer Map Preprocessing (Stage 1: Cleaning & Resizing)

# ### PURPOSE
# This script serves as the "Entry Point" for the Machine Learning pipeline.
# It transforms the raw, messy WM-811K dataset into a clean, standardized format
# suitable for feature engineering.

# ### KEY OPERATIONS
# 1. Load Data: Reads the raw pickle file (`LSWMD.pkl`).
# 2. Data Cleaning: 
#    - Fixes column name typos.
#    - Removes wafers with missing labels.
#    - Drops specific noise classes (e.g., 'Near-full').
#    - Removes tiny wafers (< 5x5 pixels).
# 3. Preprocessing:
#    - Denoise: Applies a 2x2 Median Filter.
#    - Resize: Resizes all maps to 64x64 using Nearest Neighbor interpolation.
# 4. Output:
#    - Saves a compressed `.npz` file containing the full, clean dataset.

# ### ⚠️ NOTE
# This stage does **NOT** perform class balancing or train/test splitting. 
# Those steps are handled in later stages to prevent data leakage.
# ────────────────────────────────────────────────────────────────────────────────
# """

# import os
# import random
# import warnings
# import numpy as np
# import pandas as pd
# from scipy import ndimage
# from tqdm import tqdm
# from typing import Tuple, List, Optional

# # Register tqdm with pandas to use .progress_apply()
# tqdm.pandas()

# # Suppress warnings for cleaner output
# warnings.filterwarnings("ignore")

# # ──────────────────────────────────────────────────────────────────────────────
# # 1️⃣ LOAD DATA
# # ──────────────────────────────────────────────────────────────────────────────

# def load_dataset(pickle_path: str) -> pd.DataFrame:
#     """
#     Loads the main WM-811K dataset from a pickle file.
    
#     This dataset typically contains ~811,457 wafer maps. The file is expected to be a
#     pandas DataFrame serialized with `pd.to_pickle`.

#     Args:
#         pickle_path (str): The absolute path to the .pkl file.

#     Returns:
#         pd.DataFrame: The raw dataframe loaded from the file.

#     Raises:
#         FileNotFoundError: If the file does not exist at the specified path.
#     """
#     print(f"🔹 Loading pickle file from: {pickle_path}")
    
#     if not os.path.exists(pickle_path):
#         raise FileNotFoundError(f"File not found: {pickle_path}")
        
#     df = pd.read_pickle(pickle_path)

#     # Fix known typo in original dataset: 'trianTestLabel' -> 'trainTestLabel'
#     if "trianTestLabel" in df.columns:
#         df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

#     # Drop 'waferIndex' as it is metadata and not a predictive feature
#     df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    
#     return df


# def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates and adds a 'waferMapDim' column (height, width).
    
#     Args:
#         df (pd.DataFrame): Dataframe containing a 'waferMap' column.

#     Returns:
#         pd.DataFrame: Dataframe with the new 'waferMapDim' column.
#     """
#     df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
#     return df


# # ──────────────────────────────────────────────────────────────────────────────
# # 2️⃣ CLEAN LABELS & FILTER DATA
# # ──────────────────────────────────────────────────────────────────────────────

# def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Cleans label columns and removes unlabeled data.

#     Operations:
#     - Extracts the string from nested lists (e.g., [['Center']] -> 'Center').
#     - Drops rows where 'failureType' is NaN.
#     - Converts 'failureType' to a categorical format for memory efficiency.

#     Args:
#         df (pd.DataFrame): The raw dataframe.

#     Returns:
#         pd.DataFrame: The cleaned dataframe with valid labels only.
#     """
#     df = df.copy()
    
#     # Un-nest the labels
#     df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
#     df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    
#     # CRITICAL: Drop any wafer that doesn't have a label.
#     df.dropna(subset=["failureType"], inplace=True)
    
#     # Convert to category type
#     df["failureType"] = df["failureType"].astype("category")
    
#     return df


# def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Removes noisy, small, or irrelevant wafers from the dataset.

#     Filters applied:
#     - Removes wafers smaller than 5x5 pixels.
#     - Removes the 'Near-full' class (often considered noise/irrelevant).

#     Args:
#         df (pd.DataFrame): The dataframe with valid labels.

#     Returns:
#         pd.DataFrame: The filtered dataframe.
#     """
#     # Remove wafers with dimensions 5x5 or smaller
#     df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
    
#     # Remove 'Near-full' class
#     df = df[df["failureType"] != "Near-full"]
    
#     # Clean up unused categories
#     df["failureType"] = df["failureType"].cat.remove_unused_categories()
    
#     return df


# # ──────────────────────────────────────────────────────────────────────────────
# # 3️⃣ PREPROCESSING (DENOISE & RESIZE)
# # ──────────────────────────────────────────────────────────────────────────────

# def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Applies a 2x2 Median Filter to remove high-frequency 'salt-and-pepper' noise 
#     from wafer maps while preserving edge fidelity.

#     Methodology:
#     The Median Filter replaces each pixel's value with the median value of its 
#     neighboring pixels within a 2x2 window. 

#     Why Median Filter?
#     1. Noise Reduction: Wafer maps often contain random, isolated single-pixel 
#        noise (value 2) caused by sensor glitches or dust. These 'lonely dots' 
#        confuse classifiers. A median filter effectively deletes these outliers 
#        because they are statistically different from their neighbors.
    
#     2. Edge Preservation: Unlike a Gaussian or Mean filter, which blurs images, 
#        a Median filter preserves sharp edges. This is critical for semiconductor 
#        defects like 'Scratches' or 'Rings', where the sharp boundary defines 
#        the defect type. Blurring would destroy the linearity of a scratch.

    

#     Args:
#         df (pd.DataFrame): Input Dataframe containing the 'waferMap' column. 
#                            Each entry must be a 2D numpy array representing a wafer.

#     Returns:
#         pd.DataFrame: A copy of the dataframe with the 'waferMap' column updated 
#                       to contain the denoised, cleaner 2D arrays.
#     """
#     df = df.copy()
#     print("🔹 Applying denoise (median filter)...")
#     df["waferMap"] = df["waferMap"].progress_apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
#     return df

# def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Maps string labels to integer classes (0-7).

#     Mapping:
#     - Center: 0, Donut: 1, Edge-Loc: 2, Edge-Ring: 3, 
#     - Loc: 4, Random: 5, Scratch: 6, none: 7

#     Args:
#         df (pd.DataFrame): Dataframe with 'failureType' column.

#     Returns:
#         pd.DataFrame: Dataframe with new 'failureNum' column.
#     """
#     mapping_type = {
#         "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
#         "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
#     }
#     df["failureNum"] = df["failureType"].map(mapping_type)
#     return df


# def resize_wafer_map(w: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
#     """
#     Resizes a single wafer map using Nearest Neighbor interpolation.

#     **Why use Nearest Neighbor?**
#     The wafer map contains discrete class labels (0=Background, 1=Wafer, 2=Defect).
#     Using Bilinear or Bicubic interpolation would introduce float values (e.g., 1.5)
#     which ruins the discrete nature of the data. Nearest Neighbor preserves the exact
#     categorical values.

#     **Why 64x64?**
#     It is the "Sweet Spot":
#     1.  Small enough for fast feature extraction (thousands of wafers processed per minute).
#     2.  Large enough to preserve the morphology of defects (Scratches/Rings are still visible).
#     3.  Uniformity is required for vectorized numpy operations later in the pipeline.

#     Args:
#         w (np.ndarray): The input wafer map array (variable dimensions).
#         target_size (Tuple[int, int]): Desired (height, width). Defaults to (64, 64).

#     Returns:
#         np.ndarray: The resized 64x64 wafer map.
#     """
#     return ndimage.zoom(
#         w, 
#         (target_size[0] / w.shape[0], target_size[1] / w.shape[1]),
#         order=0 
#     )


# def apply_resize(df: pd.DataFrame, size: Tuple[int, int] = (64, 64)) -> pd.DataFrame:
#     """
#     Applies the resize function to the entire dataframe.

#     Args:
#         df (pd.DataFrame): Dataframe containing 'waferMap'.
#         size (Tuple[int, int]): Target dimensions.

#     Returns:
#         pd.DataFrame: Dataframe with resized wafer maps.
#     """
#     df = df.copy()
#     print(f"🔹 Applying resize to {size}...")
#     df["waferMap"] = df["waferMap"].progress_apply(lambda x: resize_wafer_map(x, size))
#     return df


# # ──────────────────────────────────────────────────────────────────────────────
# # 4️⃣ SAVE CLEAN DATA
# # ──────────────────────────────────────────────────────────────────────────────

# def save_cleaned_data(df: pd.DataFrame, save_path: str):
#     """
#     Saves the processed wafer maps and labels to a compressed .npz file.

#     The .npz file will contain:
#     - 'waferMap': A 3D numpy array of shape (N, H, W).
#     - 'labels': A 1D numpy array of shape (N,).

#     Args:
#         df (pd.DataFrame): The final processed dataframe.
#         save_path (str): Destination path for the .npz file.
#     """
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     # Convert dataframe column of arrays into a single 3D numpy array
#     # This is much faster for loading in the next stage
#     wafer_maps_array = np.stack(df["waferMap"].values)
#     labels_array = df["failureNum"].to_numpy()

#     np.savez_compressed(
#         save_path,
#         waferMap=wafer_maps_array,  
#         labels=labels_array         
#     )
#     print(f"💾 Cleaned wafer maps saved to: {save_path}")
#     print(f"   Saved array shapes: waferMap={wafer_maps_array.shape}, labels={labels_array.shape}")


# # ──────────────────────────────────────────────────────────────────────────────
# # 5️⃣ MAIN PIPELINE CONTROLLER
# # ──────────────────────────────────────────────────────────────────────────────

# def load_and_preprocess(
#     pickle_path: str, 
#     save_path: str,
#     target_size: Tuple[int, int],
#     seed: int
# ) -> pd.DataFrame:
#     """
#     Orchestrates the full data loading and cleaning pipeline.

#     Args:
#         pickle_path (str): Path to raw input file.
#         save_path (str): Path to save processed output.
#         target_size (Tuple[int, int]): Dimension to resize wafers to.
#         seed (int): Random seed for reproducibility.

#     Returns:
#         pd.DataFrame: The final cleaned dataframe.
#     """
#     # Set global seed
#     random.seed(seed)
#     np.random.seed(seed)
    
#     print("\n" + "="*50)
#     print("🚀 STAGE 1: DATA LOADING & CLEANING")
#     print("="*50)
    
#     # 1. Load
#     df = load_dataset(pickle_path)

#     # 2. Clean & Filter
#     df = add_wafer_dimensions(df)
#     df = clean_labels(df)
#     df = filter_wafers(df)
    
#     # 3. Preprocess
#     df = apply_denoise(df)
#     df = encode_labels(df)

#     # 4. Resize
#     # Note: We do NOT balance here. We pass the full dataset to Stage 2/3.
#     df = apply_resize(df, size=target_size)

#     print(f"✅ Final cleaned dataset size: {len(df)} wafers")

#     # 5. Save
#     save_cleaned_data(df, save_path)

#     print("="*50)
#     print("✅ STAGE 1 COMPLETE")
#     return df


# # ──────────────────────────────────────────────────────────────────────────────
# # 6️⃣ EXECUTION ENTRY POINT
# # ──────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
    
#     # Import centralized config
#     try:
#         from config import RAW_DATA_PATH, CLEANED_DATA_FILE, TARGET_SIZE, RANDOM_SEED
#     except ImportError:
#         # Fallback if running directly without package logical setup
#         import sys
#         sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#         from config import RAW_DATA_PATH, CLEANED_DATA_FILE, TARGET_SIZE, RANDOM_SEED

#     # --- RUN PIPELINE ---
#     # Ensure raw path string
#     input_path = str(RAW_DATA_PATH)
#     output_path = str(CLEANED_DATA_FILE)

#     df = load_and_preprocess(
#         pickle_path=input_path,
#         save_path=output_path,
#         target_size=TARGET_SIZE,
#         seed=RANDOM_SEED 
#     )
    
#     print("\n🎯 Next Step: Run 'feature_engineering.py' to generate features.")










"""
data_loader.py(Stage 1)
────────────────────────────────────────────────────────────────────────────────
WM-811K Wafer Map Preprocessing (Stage 1: Cleaning & Resizing)

### PURPOSE
This script serves as the "Entry Point" for the Machine Learning pipeline.
It transforms the raw, messy WM-811K dataset into a clean, standardized format
suitable for feature engineering.

### KEY OPERATIONS
1. Load Data: Reads the raw pickle file (`LSWMD.pkl`).
2. Data Cleaning: 
   - Fixes column name typos.
   - Removes wafers with missing labels.
   - Drops specific noise classes (e.g., 'Near-full').
   - Removes tiny wafers (< 5x5 pixels).
3. Preprocessing:
   - Denoise: Applies a 2x2 Median Filter.
   - Resize: Resizes all maps to 64x64 using Nearest Neighbor interpolation.
4. Output:
   - Saves a compressed `.npz` file containing the full, clean dataset.

### ⚠️ NOTE
This stage does **NOT** perform class balancing or train/test splitting. 
Those steps are handled in later stages to prevent data leakage.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
from typing import Tuple, List, Optional

# Register tqdm with pandas to use .progress_apply()
tqdm.pandas()

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(pickle_path: str) -> pd.DataFrame:
    """
    Loads the main WM-811K dataset from a pickle file.

    Args:
        pickle_path (str): The absolute path to the .pkl file.

    Returns:
        pd.DataFrame: The raw dataframe loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
    """
    print(f"🔹 Loading pickle file from: {pickle_path}")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
        
    df = pd.read_pickle(pickle_path)

    # Fix known typo in original dataset: 'trianTestLabel' -> 'trainTestLabel'
    if "trianTestLabel" in df.columns:
        df.rename(columns={"trianTestLabel": "trainTestLabel"}, inplace=True)

    # Drop 'waferIndex' as it is metadata and not a predictive feature
    df.drop(["waferIndex"], axis=1, inplace=True, errors="ignore")
    
    return df


def add_wafer_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds a 'waferMapDim' column (height, width).
    
    Args:
        df (pd.DataFrame): Dataframe containing a 'waferMap' column.

    Returns:
        pd.DataFrame: Dataframe with the new 'waferMapDim' column.
    """
    df["waferMapDim"] = df["waferMap"].apply(lambda x: (x.shape[0], x.shape[1]))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ CLEAN LABELS & FILTER DATA
# ──────────────────────────────────────────────────────────────────────────────

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans label columns and removes unlabeled data.

    Operations:
    - Extracts the string from nested lists (e.g., [['Center']] -> 'Center').
    - Drops rows where 'failureType' is NaN.
    - Converts 'failureType' to a categorical format for memory efficiency.

    Args:
        df (pd.DataFrame): The raw dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe with valid labels only.
    """
    df = df.copy()
    
    # Un-nest the labels
    df["failureType"] = df["failureType"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    df["trainTestLabel"] = df["trainTestLabel"].apply(lambda x: x[0][0] if len(x) > 0 else np.nan)
    
    # CRITICAL: Drop any wafer that doesn't have a label.
    df.dropna(subset=["failureType"], inplace=True)
    
    # Convert to category type
    df["failureType"] = df["failureType"].astype("category")
    
    return df


def filter_wafers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes noisy, small, or irrelevant wafers from the dataset.

    Filters applied:
    - Removes wafers smaller than 5x5 pixels.
    - Removes the 'Near-full' class (often considered noise/irrelevant).

    Args:
        df (pd.DataFrame): The dataframe with valid labels.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    # Remove wafers with dimensions 5x5 or smaller
    df = df[df["waferMapDim"].apply(lambda x: all(np.greater(x, (5, 5))))]
    
    # Remove 'Near-full' class
    df = df[df["failureType"] != "Near-full"]
    
    # Clean up unused categories
    df["failureType"] = df["failureType"].cat.remove_unused_categories()
    
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ PREPROCESSING (DENOISE & RESIZE)
# ──────────────────────────────────────────────────────────────────────────────

def apply_denoise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a 2x2 Median Filter to remove high-frequency 'salt-and-pepper' noise 
    from wafer maps while preserving edge fidelity.

    Methodology:
    The Median Filter replaces each pixel's value with the median value of its 
    neighboring pixels within a 2x2 window. 

    Why Median Filter?
    1. Noise Reduction: Wafer maps often contain random, isolated single-pixel 
       noise (value 2) caused by sensor glitches or dust. These 'lonely dots' 
       confuse classifiers. A median filter effectively deletes these outliers 
       because they are statistically different from their neighbors.
    
    2. Edge Preservation: Unlike a Gaussian or Mean filter, which blurs images, 
       a Median filter preserves sharp edges. This is critical for semiconductor 
       defects like 'Scratches' or 'Rings', where the sharp boundary defines 
       the defect type. Blurring would destroy the linearity of a scratch.

    

    Args:
        df (pd.DataFrame): Input Dataframe containing the 'waferMap' column. 
                           Each entry must be a 2D numpy array representing a wafer.

    Returns:
        pd.DataFrame: A copy of the dataframe with the 'waferMap' column updated 
                      to contain the denoised, cleaner 2D arrays.
    """
    df = df.copy()
    print("🔹 Applying denoise (median filter)...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: ndimage.median_filter(x, size=(2, 2)))
    return df

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps string labels to integer classes (0-7).

    Mapping:
    - Center: 0, Donut: 1, Edge-Loc: 2, Edge-Ring: 3, 
    - Loc: 4, Random: 5, Scratch: 6, none: 7

    Args:
        df (pd.DataFrame): Dataframe with 'failureType' column.

    Returns:
        pd.DataFrame: Dataframe with new 'failureNum' column.
    """
    mapping_type = {
        "Center": 0, "Donut": 1, "Edge-Loc": 2, "Edge-Ring": 3,
        "Loc": 4, "Random": 5, "Scratch": 6, "none": 7
    }
    df["failureNum"] = df["failureType"].map(mapping_type)
    return df


def resize_wafer_map(w: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Resizes a single wafer map using Nearest Neighbor interpolation.

    Note: Nearest Neighbor (order=0) is used to preserve the discrete 
    pixel values (0, 1, 2) and avoid creating artifacts like 0.5.

    Args:
        w (np.ndarray): The input wafer map array.
        target_size (Tuple[int, int]): Desired (height, width).

    Returns:
        np.ndarray: The resized wafer map.
        
    **We resize to 64x64 because it is the "Sweet Spot":
    1.Small enough to be fast.
    2.Large enough to still see the defects.
    3.Uniform so the math code doesn't crash.
    """
    return ndimage.zoom(
        w, 
        (target_size[0] / w.shape[0], target_size[1] / w.shape[1]),
        order=0 
    )


def apply_resize(df: pd.DataFrame, size: Tuple[int, int] = (64, 64)) -> pd.DataFrame:
    """
    Applies the resize function to the entire dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing 'waferMap'.
        size (Tuple[int, int]): Target dimensions.

    Returns:
        pd.DataFrame: Dataframe with resized wafer maps.
    """
    df = df.copy()
    print(f"🔹 Applying resize to {size}...")
    df["waferMap"] = df["waferMap"].progress_apply(lambda x: resize_wafer_map(x, size))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ SAVE CLEAN DATA
# ──────────────────────────────────────────────────────────────────────────────

def save_cleaned_data(df: pd.DataFrame, save_path: str):
    """
    Saves the processed wafer maps and labels to a compressed .npz file.

    The .npz file will contain:
    - 'waferMap': A 3D numpy array of shape (N, H, W).
    - 'labels': A 1D numpy array of shape (N,).

    Args:
        df (pd.DataFrame): The final processed dataframe.
        save_path (str): Destination path for the .npz file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert dataframe column of arrays into a single 3D numpy array
    # This is much faster for loading in the next stage
    wafer_maps_array = np.stack(df["waferMap"].values)
    labels_array = df["failureNum"].to_numpy()

    np.savez_compressed(
        save_path,
        waferMap=wafer_maps_array,  
        labels=labels_array         
    )
    print(f"💾 Cleaned wafer maps saved to: {save_path}")
    print(f"   Saved array shapes: waferMap={wafer_maps_array.shape}, labels={labels_array.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣ MAIN PIPELINE CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(
    pickle_path: str, 
    save_path: str,
    target_size: Tuple[int, int],
    seed: int
) -> pd.DataFrame:
    """
    Orchestrates the full data loading and cleaning pipeline.

    Args:
        pickle_path (str): Path to raw input file.
        save_path (str): Path to save processed output.
        target_size (Tuple[int, int]): Dimension to resize wafers to.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: The final cleaned dataframe.
    """
    # Set global seed
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*50)
    print("🚀 STAGE 1: DATA LOADING & CLEANING")
    print("="*50)
    
    # 1. Load
    df = load_dataset(pickle_path)

    # 2. Clean & Filter
    df = add_wafer_dimensions(df)
    df = clean_labels(df)
    df = filter_wafers(df)
    
    # 3. Preprocess
    df = apply_denoise(df)
    df = encode_labels(df)

    # 4. Resize
    # Note: We do NOT balance here. We pass the full dataset to Stage 2/3.
    df = apply_resize(df, size=target_size)

    print(f"✅ Final cleaned dataset size: {len(df)} wafers")

    # 5. Save
    save_cleaned_data(df, save_path)

    print("="*50)
    print("✅ STAGE 1 COMPLETE")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣ EXECUTION ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Import centralized config
    try:
        from config import RAW_DATA_PATH, CLEANED_DATA_FILE, TARGET_SIZE, RANDOM_SEED
    except ImportError:
        # Fallback if running directly without package logical setup
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from config import RAW_DATA_PATH, CLEANED_DATA_FILE, TARGET_SIZE, RANDOM_SEED

    # --- RUN PIPELINE ---
    # Ensure raw path string
    input_path = str(RAW_DATA_PATH)
    output_path = str(CLEANED_DATA_FILE)

    df = load_and_preprocess(
        pickle_path=input_path,
        save_path=output_path,
        target_size=TARGET_SIZE,
        seed=RANDOM_SEED 
    )
    
    print("\n🎯 Next Step: Run 'feature_engineering.py' to generate features.")
