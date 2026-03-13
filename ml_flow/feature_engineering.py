# """
# feature_engineering.py (Stage 2)
# ────────────────────────────────────────────────────────────────────────────────
# WM-811K Feature Extraction (Stage 2)

# ### 🎯 PURPOSE
# This script serves as the "Translation Layer" of the pipeline. It converts raw 
# 64x64 wafer map images into a structured dataset of 66 numerical features. 
# Raw pixel data is often too high-dimensional and noisy for traditional classifiers; 
# therefore, we extract high-level "descriptors" that quantify specific visual properties.

# ### ⚙️ FEATURE GROUPS EXPLAINED (66 Total Features)

# 1. 📍 Density Features (13 Features)
#    - **Logic:** The wafer is divided into 13 spatial zones:
#      - 9 Inner blocks (3x3 grid) to detect 'Center' or 'Loc' defects.
#      - 4 Edge strips (Top, Bottom, Left, Right) to detect 'Edge-Loc' or 'Edge-Ring'.
#    - **Calculation:** Density = (Defect Pixels / Total Pixels) * 100 in each zone.
#    - **Why?** Defects like 'Edge-Ring' are defined entirely by *where* they appear. 
     

# 2. 🌀 Radon Transform Features (40 Features)
#    - **Logic:** The Radon transform projects the image along specific angles (0° to 180°). 
#      Think of it as taking an X-ray of the wafer from multiple sides.
#    - **Calculation:** We compute the Mean and Standard Deviation of the projections 
#      (Sinogram) to capture the intensity profile.
#    - **Why?** Standard density checks fail to see lines. The Radon transform creates 
#      massive spikes in the data when it aligns with a straight line, making it the 
#      perfect tool for detecting **'Scratch'** defects.
     

# 3. 📐 Geometry Features (7 Features)
#    - **Logic:** We treat the defect pixels as a "blob" and measure its shape using 
#      connected component analysis (RegionProps).
#    - **Key Metrics:**
#      - `Eccentricity`: 0 = Circle ('Center'), 1 = Line ('Scratch').
#      - `Solidity`: How solid the object is. Low solidity = 'Random' noise or 'Loc'.
#      - `Num_Regions`: Count of separate blobs. High count = 'Random' or 'Donut'.
#    - **Why?** This helps distinguishing shapes (e.g., a 'Loc' blob vs a 'Scratch' line).
     

# 4. 📊 Statistical Features (6 Features)
#    - **Logic:** Simple statistical moments of the flattened pixel array.
#    - **Metrics:** Mean, Variance, Skewness, Kurtosis, Median, Standard Deviation.
#    - **Why?** Capture the global "noise level" and distribution "tail." 
#      High variance often indicates a significant defect presence vs. a clean wafer.

# ### 📦 OUTPUT
# - `features_dataset.csv`: A tabular dataset ready for Machine Learning.
#   - Rows: Individual Wafers (Samples)
#   - Columns: 66 Features + 1 Target Label
# ────────────────────────────────────────────────────────────────────────────────
# """

# import os
# import logging
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
# from scipy import ndimage, interpolate
# from scipy.stats import skew, kurtosis
# from skimage.transform import radon
# from skimage import measure
# from tqdm import tqdm
# from typing import List, Tuple, Union

# # ──────────────────────────────────────────────────────────────────────────────
# # 📝 CONFIGURATION
# # ──────────────────────────────────────────────────────────────────────────────

# try:
#     from config import CLEANED_DATA_FILE, FEATURES_FILE_CSV, FEATURES_FILE_PARQUET, N_RADON_THETA, RADON_OUTPUT_POINTS, N_JOBS, configure_logging, FEATURE_ENGINEERING_DIR
# except ImportError:
#     import sys
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#     from config import CLEANED_DATA_FILE, FEATURES_FILE_CSV, FEATURES_FILE_PARQUET, N_RADON_THETA, RADON_OUTPUT_POINTS, N_JOBS, configure_logging, FEATURE_ENGINEERING_DIR

# # Logging Setup
# logger = configure_logging(__name__)


# # ──────────────────────────────────────────────────────────────────────────────
# # 1️⃣ HELPER FUNCTIONS
# # ──────────────────────────────────────────────────────────────────────────────

# def _validate_image(img: np.ndarray) -> np.ndarray:
#     """
#     Ensures the image is a valid 2D array and handles non-finite values.

#     Args:
#         img (np.ndarray): Input image array.

#     Returns:
#         np.ndarray: Validated 2D float32 image.
    
#     Raises:
#         ValueError: If image is not 2D.
#     """
#     if img.ndim != 2:
#         raise ValueError(f"Expected 2D image, got shape {img.shape}")
    
#     if not np.isfinite(img).all():
#         img = np.nan_to_num(img, nan=0.0)
        
#     return img.astype(np.float32)


# def cal_den(region: np.ndarray) -> float:
#     """
#     Calculates defect density percentage in a specific region.
#     Target value '2' represents a defect.
#     """
#     if region.size == 0:
#         return 0.0
#     return 100.0 * (np.count_nonzero(region == 2) / region.size)


# def find_regions(img: np.ndarray) -> List[float]:
#     """
#     Divides the wafer map into 13 spatial zones to capture defect location patterns.
    
#     This feature is critical for distinguishing location-based defects:
#     - **Edge-Ring:** High density in the outer 4 strips.
#     - **Center:** High density in the center block (Inner 5).
#     - **Loc:** High density in one specific block.

#     **Zone Mapping:**
#     - Zones 0-3: Top, Right, Bottom, Left edge strips (outer frame).
#     - Zones 4-12: 3x3 Grid covering the inner 80% of the wafer.

#     Args:
#         img (np.ndarray): 64x64 binary or float wafer map.

#     Returns:
#         List[float]: A list of 13 density values (0.0 to 100.0).
#     """
#     rows, cols = img.shape
    
#     # Safety check for tiny images
#     if rows < 5 or cols < 5: 
#         return [0.0] * 13

#     # Define boundaries (approx 1/5th cuts)
#     r_edges = np.unique(np.linspace(0, rows, 6, dtype=int))
#     c_edges = np.unique(np.linspace(0, cols, 6, dtype=int))
    
#     # Extract slices
#     regions = [
#         img[r_edges[0]:r_edges[1], :],                      # Top Edge (0)
#         img[:, c_edges[4]:c_edges[5]],                      # Right Edge (1)
#         img[r_edges[4]:r_edges[5], :],                      # Bottom Edge (2)
#         img[:, c_edges[0]:c_edges[1]],                      # Left Edge (3)
#         img[r_edges[1]:r_edges[2], c_edges[1]:c_edges[2]],  # Inner 1 (4)
#         img[r_edges[1]:r_edges[2], c_edges[2]:c_edges[3]],  # Inner 2 (5)
#         img[r_edges[1]:r_edges[2], c_edges[3]:c_edges[4]],  # Inner 3 (6)
#         img[r_edges[2]:r_edges[3], c_edges[1]:c_edges[2]],  # Inner 4 (7)
#         img[r_edges[2]:r_edges[3], c_edges[2]:c_edges[3]],  # Inner 5 (Center) (8)
#         img[r_edges[2]:r_edges[3], c_edges[3]:c_edges[4]],  # Inner 6 (9)
#         img[r_edges[3]:r_edges[4], c_edges[1]:c_edges[2]],  # Inner 7 (10)
#         img[r_edges[3]:r_edges[4], c_edges[2]:c_edges[3]],  # Inner 8 (11)
#         img[r_edges[3]:r_edges[4], c_edges[3]:c_edges[4]]   # Inner 9 (12)
#     ]
#     return [cal_den(r) for r in regions]


# def _safe_radon(img: np.ndarray, n_theta: int) -> np.ndarray:
#     """
#     Computes the Radon transform (Sinogram) of an image.
    
#     **Why Radon?**
#     The Radon transform sums pixel intensities along straight lines at various angles.
#     This creates strong peaks in the sinogram when the projection angle aligns with
#     a linear structure on the wafer. It is the single most effective technique for
#     detecting **'Scratch'** defects, which are defined by their linearity.

#     Args:
#         img (np.ndarray): 2D image array (Defects only, no background).
#         n_theta (int): Number of projection angles (0 to 180 degrees).

#     Returns:
#         np.ndarray: The sinogram (rows=image_height, cols=n_theta).
#     """
#     theta = np.linspace(0., 180., n_theta, endpoint=False)
#     try:
#         return radon(img, theta=theta, circle=False)
#     except Exception:
#         # Fallback for empty or corrupted images
#         return np.zeros((img.shape[0], n_theta))


# def cubic_inter_features(sinogram: np.ndarray, output_points: int) -> np.ndarray:
#     """
#     Compresses the 2D Radon Sinogram into a 1D feature vector.
    
#     Method:
#     1. Calculate Mean and Std Dev profiles along the projection axis.
#     2. Interpolate these profiles to a fixed length (output_points).
#     """
#     # Existing check
#     if sinogram.size == 0: 
#         return np.zeros(output_points * 2)

#     # NEW: Check for zero variance (flat signal) to prevent interpolation errors
#     if np.max(sinogram) == np.min(sinogram):
#          return np.zeros(output_points * 2)

#     mean_profile = np.mean(sinogram, axis=1)
#     std_profile = np.std(sinogram, axis=1)
    
#     # Create interpolation functions
#     x = np.arange(len(mean_profile))
#     f_mean = interpolate.interp1d(x, mean_profile, kind='linear')
#     f_std = interpolate.interp1d(x, std_profile, kind='linear')
    
#     # Sample at fixed points
#     xnew = np.linspace(0, len(mean_profile)-1, output_points)
    
#     return np.concatenate([f_mean(xnew), f_std(xnew)])


# def fea_geom(img: np.ndarray) -> List[float]:
#     """
#     Extracts geometric properties of the *largest* defect cluster.
    
#     Features: Area, Perimeter, Major/Minor Axis, Eccentricity, Solidity.
#     Plus: Number of distinct defect regions.
#     """
#     # Create binary mask (Defect=1, Background/Pass=0)
#     binary_img = (img == 2).astype(int)
    
#     # Label connected components
#     labels = measure.label(binary_img, connectivity=1)
    
#     # Case: No defects found
#     if labels.max() == 0: 
#         return [0.0] * 7 

#     # Get properties of all regions
#     props = measure.regionprops(labels)
    
#     # Select the largest region (by area)
#     region = max(props, key=lambda r: r.area)
    
#     return [
#         region.area,
#         region.perimeter,
#         region.major_axis_length,
#         region.minor_axis_length,
#         region.eccentricity,
#         region.solidity,
#         float(len(props))  # Count of distinct regions
#     ]


# def fea_stats(img: np.ndarray) -> List[float]:
#     """
#     Extracts global statistical features from the image pixels.
#     Includes checks for flat images to prevent NaN in skew/kurtosis.
#     """
#     pixels = img.flatten()
    
#     variance = np.var(pixels)
    
#     # If image is completely flat (all 0s or all 1s), skew/kurt are undefined.
#     if variance == 0:
#         return [float(np.mean(pixels)), 0.0, 0.0, 0.0, 0.0, float(np.median(pixels))]
    
#     return [
#         float(np.mean(pixels)),
#         float(np.std(pixels)),
#         float(variance),
#         float(skew(pixels, nan_policy='omit')),
#         float(kurtosis(pixels, nan_policy='omit')),
#         float(np.median(pixels))
#     ]


# # ──────────────────────────────────────────────────────────────────────────────
# # 2️⃣ MAIN EXTRACTION PIPELINE
# # ──────────────────────────────────────────────────────────────────────────────

# def process_single_wafer(img: np.ndarray) -> np.ndarray:
#     """
#     Worker function to process a single wafer map.
#     Calls all feature extractors and concatenates results.
#     """
#     # Ensure clean input
#     img = _validate_image(img)
    
#     # 1. Density (13 features)
#     dens = find_regions(img)
    
#     # 2. Radon (40 features)
#     # Convert '1' (Pass) to '0' (Background) so we only project Defects (2)
#     img_clean = img.copy()
#     img_clean[img_clean == 1] = 0
#     sinogram = _safe_radon(img_clean, n_theta=N_RADON_THETA)
#     radon_feats = cubic_inter_features(sinogram, output_points=RADON_OUTPUT_POINTS)
    
#     # 3. Geometry (7 features)
#     geom = fea_geom(img)
    
#     # 4. Statistics (6 features)
#     stats = fea_stats(img)
    
#     # Flatten and combine
#     return np.concatenate([dens, radon_feats, geom, stats])


# def extract_and_save():
#     """
#     Orchestrator function:
#     1. Loads the cleaned .npz file.
#     2. Runs feature extraction in parallel (using all CPU cores).
#     3. Saves the result as a CSV and Parquet file for the next stage.
#     """
#     # Convert path objects to str if needed
#     input_file = str(CLEANED_DATA_FILE)
    
#     if not os.path.exists(input_file):
#         logger.error(f"Input file not found: {input_file}")
#         return

#     logger.info(f"Loading data from {input_file}...")
#     data = np.load(input_file, allow_pickle=True)
#     X_imgs = data['waferMap']
#     y_labels = data['labels']
    
#     logger.info(f"Detected {len(X_imgs)} wafers.")
#     logger.info(f"Extracting features (Jobs: {N_JOBS}). This may take a while...")
    
#     # Parallel Processing with Progress Bar
#     X_features = Parallel(n_jobs=N_JOBS)(
#         delayed(process_single_wafer)(img) for img in tqdm(X_imgs, unit="wafer")
#     )
#     X_features = np.array(X_features)
    
#     # Define Column Names for clarity
#     feature_names = (
#         [f"density_{i+1}" for i in range(13)] +
#         [f"radon_mean_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
#         [f"radon_std_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
#         # Updated Geometry columns to include num_regions
#         ["geom_area", "geom_perimeter", "geom_major_axis", "geom_minor_axis", 
#          "geom_eccentricity", "geom_solidity", "geom_num_regions"] +
#         ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
#     )
    
#     # Create DataFrame
#     df = pd.DataFrame(X_features, columns=feature_names)
#     df['target'] = y_labels # Append target column
    
#     # Save
#     os.makedirs(FEATURE_ENGINEERING_DIR, exist_ok=True)
    
#     # 1. CSV Save
#     csv_path = str(FEATURES_FILE_CSV)
#     df.to_csv(csv_path, index=False)
    
#     # 2. Parquet Save (New)
#     parquet_path = str(FEATURES_FILE_PARQUET)
#     df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
#     logger.info(f"✅ Success! Features saved to:")
#     logger.info(f"   CSV:     {csv_path}")
#     logger.info(f"   Parquet: {parquet_path}")
#     logger.info(f"   Shape: {df.shape}")

# if __name__ == "__main__":
#     extract_and_save()



















"""
feature_engineering.py (Stage 2)
────────────────────────────────────────────────────────────────────────────────
WM-811K Feature Extraction (Stage 2)

### 🎯 PURPOSE
This script serves as the "Translation Layer" of the pipeline. It converts raw 
64x64 wafer map images into a structured dataset of 66 numerical features. 
Raw pixel data is often too high-dimensional and noisy for traditional classifiers; 
therefore, we extract high-level "descriptors" that quantify specific visual properties.

### ⚙️ FEATURE GROUPS EXPLAINED (66 Total Features)

1. 📍 Density Features (13 Features)
   - **Logic:** The wafer is divided into 13 spatial zones:
     - 9 Inner blocks (3x3 grid) to detect 'Center' or 'Loc' defects.
     - 4 Edge strips (Top, Bottom, Left, Right) to detect 'Edge-Loc' or 'Edge-Ring'.
   - **Calculation:** Density = (Defect Pixels / Total Pixels) * 100 in each zone.
   - **Why?** Defects like 'Edge-Ring' are defined entirely by *where* they appear. 
     

2. 🌀 Radon Transform Features (40 Features)
   - **Logic:** The Radon transform projects the image along specific angles (0° to 180°). 
     Think of it as taking an X-ray of the wafer from multiple sides.
   - **Calculation:** We compute the Mean and Standard Deviation of the projections 
     (Sinogram) to capture the intensity profile.
   - **Why?** Standard density checks fail to see lines. The Radon transform creates 
     massive spikes in the data when it aligns with a straight line, making it the 
     perfect tool for detecting **'Scratch'** defects.
     

3. 📐 Geometry Features (7 Features)
   - **Logic:** We treat the defect pixels as a "blob" and measure its shape using 
     connected component analysis (RegionProps).
   - **Key Metrics:**
     - `Eccentricity`: 0 = Circle ('Center'), 1 = Line ('Scratch').
     - `Solidity`: How solid the object is. Low solidity = 'Random' noise or 'Loc'.
     - `Num_Regions`: Count of separate blobs. High count = 'Random' or 'Donut'.
   - **Why?** This helps distinguishing shapes (e.g., a 'Loc' blob vs a 'Scratch' line).
     

4. 📊 Statistical Features (6 Features)
   - **Logic:** Simple statistical moments of the flattened pixel array.
   - **Metrics:** Mean, Variance, Skewness, Kurtosis, Median, Standard Deviation.
   - **Why?** Capture the global "noise level" and distribution "tail." 
     High variance often indicates a significant defect presence vs. a clean wafer.

### 📦 OUTPUT
- `features_dataset.csv`: A tabular dataset ready for Machine Learning.
  - Rows: Individual Wafers (Samples)
  - Columns: 66 Features + 1 Target Label
────────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage, interpolate
from scipy.stats import skew, kurtosis
from skimage.transform import radon
from skimage import measure
from tqdm import tqdm
from typing import List, Tuple, Union

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

try:
    from config import CLEANED_DATA_FILE, FEATURES_FILE_CSV, FEATURES_FILE_PARQUET, N_RADON_THETA, RADON_OUTPUT_POINTS, N_JOBS, configure_logging, FEATURE_ENGINEERING_DIR
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import CLEANED_DATA_FILE, FEATURES_FILE_CSV, FEATURES_FILE_PARQUET, N_RADON_THETA, RADON_OUTPUT_POINTS, N_JOBS, configure_logging, FEATURE_ENGINEERING_DIR

# Logging Setup
logger = configure_logging(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _validate_image(img: np.ndarray) -> np.ndarray:
    """
    Ensures the image is a valid 2D array and handles non-finite values.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")
    
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0)
        
    return img.astype(np.float32)


def cal_den(region: np.ndarray) -> float:
    """
    Calculates defect density percentage in a specific region.
    Target value '2' represents a defect.
    """
    if region.size == 0:
        return 0.0
    return 100.0 * (np.count_nonzero(region == 2) / region.size)


def find_regions(img: np.ndarray) -> List[float]:
    """
    Divides the wafer map into 13 spatial zones to capture defect location.
    
    Zones logic:
    - 4 Edge Strips (Top, Bottom, Left, Right)
    - 9 Inner Grid Blocks (3x3 grid in the center)
    
    Args:
        img (np.ndarray): 64x64 wafer map.

    Returns:
        List[float]: A list of 13 density values.
    """
    rows, cols = img.shape
    
    # Safety check for tiny images
    if rows < 5 or cols < 5: 
        return [0.0] * 13

    # Define boundaries (approx 1/5th cuts)
    r_edges = np.unique(np.linspace(0, rows, 6, dtype=int))
    c_edges = np.unique(np.linspace(0, cols, 6, dtype=int))
    
    # Extract slices
    regions = [
        img[r_edges[0]:r_edges[1], :],                      # Top Edge
        img[:, c_edges[4]:c_edges[5]],                      # Right Edge
        img[r_edges[4]:r_edges[5], :],                      # Bottom Edge
        img[:, c_edges[0]:c_edges[1]],                      # Left Edge
        img[r_edges[1]:r_edges[2], c_edges[1]:c_edges[2]],  # Inner 1
        img[r_edges[1]:r_edges[2], c_edges[2]:c_edges[3]],  # Inner 2
        img[r_edges[1]:r_edges[2], c_edges[3]:c_edges[4]],  # Inner 3
        img[r_edges[2]:r_edges[3], c_edges[1]:c_edges[2]],  # Inner 4
        img[r_edges[2]:r_edges[3], c_edges[2]:c_edges[3]],  # Inner 5
        img[r_edges[2]:r_edges[3], c_edges[3]:c_edges[4]],  # Inner 6
        img[r_edges[3]:r_edges[4], c_edges[1]:c_edges[2]],  # Inner 7
        img[r_edges[3]:r_edges[4], c_edges[2]:c_edges[3]],  # Inner 8
        img[r_edges[3]:r_edges[4], c_edges[3]:c_edges[4]]   # Inner 9
    ]
    return [cal_den(r) for r in regions]


def _safe_radon(img: np.ndarray, n_theta: int) -> np.ndarray:
    """
    Computes the Radon transform (Sinogram).
    
    Why: Radon transform sums pixel intensities along straight lines.
    It creates strong peaks for linear defects (Scratches), which are
    hard to detect with simple density checks.
    """
    theta = np.linspace(0., 180., n_theta, endpoint=False)
    try:
        return radon(img, theta=theta, circle=False)
    except Exception:
        # Fallback for empty or corrupted images
        return np.zeros((img.shape[0], n_theta))


def cubic_inter_features(sinogram: np.ndarray, output_points: int) -> np.ndarray:
    """
    Compresses the 2D Radon Sinogram into a 1D feature vector.
    
    Method:
    1. Calculate Mean and Std Dev profiles along the projection axis.
    2. Interpolate these profiles to a fixed length (output_points).
    """
    if sinogram.size == 0: 
        return np.zeros(output_points * 2)

    mean_profile = np.mean(sinogram, axis=1)
    std_profile = np.std(sinogram, axis=1)
    
    # Create interpolation functions
    x = np.arange(len(mean_profile))
    f_mean = interpolate.interp1d(x, mean_profile, kind='linear')
    f_std = interpolate.interp1d(x, std_profile, kind='linear')
    
    # Sample at fixed points
    xnew = np.linspace(0, len(mean_profile)-1, output_points)
    
    return np.concatenate([f_mean(xnew), f_std(xnew)])


def fea_geom(img: np.ndarray) -> List[float]:
    """
    Extracts geometric properties of the *largest* defect cluster.
    
    Features: Area, Perimeter, Major/Minor Axis, Eccentricity, Solidity.
    Plus: Number of distinct defect regions.
    """
    # Create binary mask (Defect=1, Background/Pass=0)
    binary_img = (img == 2).astype(int)
    
    # Label connected components
    labels = measure.label(binary_img, connectivity=1)
    
    # Case: No defects found
    if labels.max() == 0: 
        return [0.0] * 7 

    # Get properties of all regions
    props = measure.regionprops(labels)
    
    # Select the largest region (by area)
    region = max(props, key=lambda r: r.area)
    
    return [
        region.area,
        region.perimeter,
        region.major_axis_length,
        region.minor_axis_length,
        region.eccentricity,
        region.solidity,
        float(len(props))  # Count of distinct regions
    ]


def fea_stats(img: np.ndarray) -> List[float]:
    """
    Extracts global statistical features from the image pixels.
    Includes checks for flat images to prevent NaN in skew/kurtosis.
    """
    pixels = img.flatten()
    
    variance = np.var(pixels)
    
    # If image is completely flat (all 0s or all 1s), skew/kurt are undefined.
    if variance == 0:
        return [float(np.mean(pixels)), 0.0, 0.0, 0.0, 0.0, float(np.median(pixels))]
    
    return [
        float(np.mean(pixels)),
        float(np.std(pixels)),
        float(variance),
        float(skew(pixels, nan_policy='omit')),
        float(kurtosis(pixels, nan_policy='omit')),
        float(np.median(pixels))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ MAIN EXTRACTION PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def process_single_wafer(img: np.ndarray) -> np.ndarray:
    """
    Worker function to process a single wafer map.
    Calls all feature extractors and concatenates results.
    """
    # Ensure clean input
    img = _validate_image(img)
    
    # 1. Density (13 features)
    dens = find_regions(img)
    
    # 2. Radon (40 features)
    # Convert '1' (Pass) to '0' (Background) so we only project Defects (2)
    img_clean = img.copy()
    img_clean[img_clean == 1] = 0
    sinogram = _safe_radon(img_clean, n_theta=N_RADON_THETA)
    radon_feats = cubic_inter_features(sinogram, output_points=RADON_OUTPUT_POINTS)
    
    # 3. Geometry (7 features)
    geom = fea_geom(img)
    
    # 4. Statistics (6 features)
    stats = fea_stats(img)
    
    # Flatten and combine
    return np.concatenate([dens, radon_feats, geom, stats])


def extract_and_save():
    """
    Orchestrator function:
    1. Loads the cleaned .npz file.
    2. Runs feature extraction in parallel (using all CPU cores).
    3. Saves the result as a CSV and Parquet file for the next stage.
    """
    # Convert path objects to str if needed
    input_file = str(CLEANED_DATA_FILE)
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading data from {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    X_imgs = data['waferMap']
    y_labels = data['labels']
    
    logger.info(f"Detected {len(X_imgs)} wafers.")
    logger.info(f"Extracting features (Jobs: {N_JOBS}). This may take a while...")
    
    # Parallel Processing with Progress Bar
    X_features = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_wafer)(img) for img in tqdm(X_imgs, unit="wafer")
    )
    X_features = np.array(X_features)
    
    # Define Column Names for clarity
    feature_names = (
        [f"density_{i+1}" for i in range(13)] +
        [f"radon_mean_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        [f"radon_std_{i+1}" for i in range(RADON_OUTPUT_POINTS)] +
        # Updated Geometry columns to include num_regions
        ["geom_area", "geom_perimeter", "geom_major_axis", "geom_minor_axis", 
         "geom_eccentricity", "geom_solidity", "geom_num_regions"] +
        ["stat_mean", "stat_std", "stat_var", "stat_skew", "stat_kurt", "stat_median"]
    )
    
    # Create DataFrame
    df = pd.DataFrame(X_features, columns=feature_names)
    df['target'] = y_labels # Append target column
    
    # Save
    os.makedirs(FEATURE_ENGINEERING_DIR, exist_ok=True)
    
    # 1. CSV Save
    csv_path = str(FEATURES_FILE_CSV)
    df.to_csv(csv_path, index=False)
    
    # 2. Parquet Save (New)
    parquet_path = str(FEATURES_FILE_PARQUET)
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    
    logger.info(f"✅ Success! Features saved to:")
    logger.info(f"   CSV:     {csv_path}")
    logger.info(f"   Parquet: {parquet_path}")
    logger.info(f"   Shape: {df.shape}")

if __name__ == "__main__":
    extract_and_save()
