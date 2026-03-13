# # -*- coding: utf-8 -*-
# """
# 📜 data_preprocessor.py (Stage 3: Split, Scale, Balance)
# ────────────────────────────────────────────────────────────────────────────────
# WM-811K Data Preparation & Hybrid Balancing

# ### 🎯 PURPOSE
# This script serves as the "Gatekeeper" between raw feature extraction and model training. 
# It addresses the three fundamental challenges of the WM-811K dataset: 
# 1. **Data Leakage:** Ensuring the Test set is never seen during scaling or balancing.
# 2. **Feature Scaling:** Normalizing features with vastly different ranges (e.g., Area vs Density).
# 3. **Extreme Imbalance:** 'None' class has ~147k samples, while 'Scratch' has ~500.

# ### ⚙️ THE PIPELINE (Strict Execution Order)

# 1. **Stratified Split (The "Leak-Proof" Wall)**
#    - We split the data into Training (80%) and Testing (20%) sets immediately.
#    - **Critical:** We use `stratify=y` to ensure that rare defects (like 'Scratch') 
#      are present in both sets in the exact same proportion as the original data. 
#      Without this, the Test set might end up with zero Scratches.

# 2. **Standardization (Z-Score Scaling)**
#    - Formula: z = (x - u) / s
#    - **Anti-Leakage Logic:** The `StandardScaler` computes the Mean (u) and Std Dev (s) 
#      using **ONLY the Training Set**. These Training stats are then applied to scale the Test set. 
#    - *Why?* If we calculated the mean of the whole dataset, information from the 
#      Test set would "leak" into the Training set, inflating our accuracy scores.

# 3. **Hybrid Balancing (The "Goldilocks" Strategy)**
#    - Training on raw data causes the model to ignore defects (96% accuracy by guessing 'None').
#    - We enforce a target of **500 samples per class** for the Training set:
#      - **Majority Class ('None'):** 📉 Undersampled from ~147k -> 500. 
#        (Reduces noise and training time).
#      - **Minority Classes ('Scratch'):** 📈 Oversampled (SMOTE) from ~100 -> 500.
#        (Creates synthetic points between neighbors to boost signal).
#    - **Result:** A perfectly balanced training set where the model treats every class equally.

# ### 📦 OUTPUT (`model_ready_data.npz`)
# A compressed dictionary containing:
# 1. `X_train_balanced`: Scaled & SMOTE-augmented (Use for training models).
# 2. `X_test`: Scaled real-world data (Use for final evaluation).
# 3. `scaler`: The saved scaler object (Required to preprocess new wafers in deployment).
# ────────────────────────────────────────────────────────────────────────────────
# """

# import os
# import joblib
# import warnings
# import numpy as np
# import pandas as pd
# from typing import Optional, Tuple, Dict

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler # use StandardScaler to make sure features with big numbers don't accidentally overpower features with small numbers. It creates a "level playing field."
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline as ImbPipeline

# # Suppress warnings from imblearn regarding class chunks
# warnings.filterwarnings("ignore")

# # ──────────────────────────────────────────────────────────────────────────────
# # 📝 CONFIGURATION
# # ──────────────────────────────────────────────────────────────────────────────

# try:
#     from config import FEATURES_FILE_CSV, PREPROCESSING_DIR, TEST_SPLIT_SIZE, RANDOM_SEED, TARGET_SAMPLES_PER_CLASS, MODEL_READY_DATA_FILE, SCALER_FILE
# except ImportError:
#     import sys
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#     from config import FEATURES_FILE_CSV, PREPROCESSING_DIR, TEST_SPLIT_SIZE, RANDOM_SEED, TARGET_SAMPLES_PER_CLASS, MODEL_READY_DATA_FILE, SCALER_FILE

# # ──────────────────────────────────────────────────────────────────────────────
# # 1️⃣ MAIN FUNCTION
# # ──────────────────────────────────────────────────────────────────────────────

# def prepare_data_for_modeling(
#     feature_csv_path: str, 
#     output_dir: str, 
#     test_size: float, 
#     seed: int
# ) -> None:
#     """
#     Loads features, splits data, scales it, and applies hybrid balancing.

#     **The Leak-Proof Logic:**
#     1.  **Split:** Data is split *strictly* before processing to ensure the test set
#         remains a true "unseen" simulation. Stratification ensures all defect types
#         are present in the test set.
#     2.  **Scale:** The Scaler is fit *only* on the Training data. This prevents "looking
#         ahead" at the test data distribution.
#     3.  **Balance:** Balancing (SMOTE) is applied *only* to the Training data. The Test
#         set must remain imbalanced to reflect the real-world deployment scenario.

#     Args:
#         feature_csv_path (str): Path to the input CSV from Stage 2.
#         output_dir (str): Folder to save the resulting .npz and scaler.
#         test_size (float): Proportion of dataset to include in the test split.
#         seed (int): Random seed for reproducibility.

#     Returns:
#         None: Saves files to disk but returns nothing.
#     """
#     print("\n" + "="*50)
#     print("🚀 STAGE 3: DATA PREPARATION & BALANCING")
#     print("="*50)

#     # --- 1. Load Data ---
#     print(f"📂 Loading features from: {feature_csv_path}")
#     if not os.path.exists(feature_csv_path):
#         raise FileNotFoundError(f"Input file not found: {feature_csv_path}")

#     df = pd.read_csv(feature_csv_path)
#     print(f"   Original Shape: {df.shape}")

#     # --- 2. Identify Target Column ---
#     # Handles variations if previous script named it 'label' or 'target'
#     if "label" in df.columns:
#         target_col = "label"
#     elif "target" in df.columns:
#         target_col = "target"
#     else:
#         raise KeyError("Dataset missing 'label' or 'target' column.")

#     X = df.drop(columns=[target_col])
#     y = df[target_col]
#     feature_names = X.columns.to_list()

#     # --- 3. CRITICAL: Stratified Train-Test Split ---
#     # We split BEFORE balancing to ensure the test set is 100% real data.
#     print(f"✂️  Splitting data (Test Size={test_size}, Stratified)...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, 
#         test_size=test_size, 
#         stratify=y, 
#         random_state=seed
#     )
#     print(f"   Train set (Imbalanced): {X_train.shape[0]} samples")
#     print(f"   Test set (Locked):      {X_test.shape[0]} samples")

#     # --- 4. CRITICAL: Scaling ---
#     # Fit on Train, Transform on Test. Prevents leakage.
#     print("📏 Scaling features (StandardScaler)...")
#     scaler = StandardScaler()
    
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test) # Note: use transform(), not fit_transform()

#     # Save point: Imbalanced data (useful for Cross-Validation later)
#     X_train_imbalanced = X_train_scaled.copy()
#     y_train_imbalanced = y_train.copy()

#     # --- 5. CRITICAL: Hybrid Balancing ---
#     print(f"⚖️  Applying Hybrid Balancing (Target: {TARGET_SAMPLES_PER_CLASS} samples/class)...")
    
#     # 5a. Analyze current distribution
#     unique, counts = np.unique(y_train, return_counts=True)
#     class_counts = dict(zip(unique, counts))
    
#     # 5b. Define Strategies
#     # - Undersample classes that have TOO MANY (> 500)
#     # - SMOTE classes that have TOO FEW (< 500)
#     under_strategy = {k: TARGET_SAMPLES_PER_CLASS for k, v in class_counts.items() if v > TARGET_SAMPLES_PER_CLASS}
#     over_strategy  = {k: TARGET_SAMPLES_PER_CLASS for k, v in class_counts.items() if v < TARGET_SAMPLES_PER_CLASS}
    
#     # 5c. Build Pipeline
#     steps = []
#     if under_strategy:
#         steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)))
    
#     if over_strategy:
#         # k_neighbors=3 is safer for very small classes than the default 5
#         steps.append(('over', SMOTE(sampling_strategy=over_strategy, random_state=seed, k_neighbors=3)))
        
#     balancer = ImbPipeline(steps)
    
#     # 5d. Execute Balancing
#     try:
#         X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
#         print(f"   ✅ Balancing Complete.")
#         print(f"   Balanced Train Shape: {X_train_balanced.shape}")
#     except ValueError as e:
#         print(f"⚠️ Balancing Warning: {e}")
#         print("   (Likely a class has fewer than 4 samples. Using Imbalanced data as fallback.)")
#         X_train_balanced, y_train_balanced = X_train_scaled, y_train

#     # --- 6. Save Outputs ---
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save Scaler (for deployment)
#     scaler_path = str(SCALER_FILE)
#     joblib.dump(scaler, scaler_path)
    
#     # Save Data
#     npz_path = str(MODEL_READY_DATA_FILE)
#     np.savez_compressed(
#         npz_path,
#         X_train_imbalanced=X_train_imbalanced,
#         y_train_imbalanced=y_train_imbalanced.to_numpy(),
#         X_train_balanced=X_train_balanced,
#         y_train_balanced=y_train_balanced.to_numpy(),
#         X_test=X_test_scaled,
#         y_test=y_test.to_numpy(),
#         feature_names=np.array(feature_names)
#     )
    
#     print("="*50)
#     print(f"💾 RESULTS SAVED:")
#     print(f"   Scaler: {scaler_path}")
#     print(f"   Data:   {npz_path}")
#     print("✅ STAGE 3 COMPLETE")


# # ──────────────────────────────────────────────────────────────────────────────
# # 2️⃣ EXECUTION ENTRY POINT
# # ──────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
    
#     input_csv = str(FEATURES_FILE_CSV)
#     output_dir = str(PREPROCESSING_DIR)

#     prepare_data_for_modeling(
#         feature_csv_path=input_csv, 
#         output_dir=output_dir,
#         test_size=TEST_SPLIT_SIZE,
#         seed=RANDOM_SEED
#     )











# -*- coding: utf-8 -*-
"""
📜 data_preprocessor.py (Stage 3: Split, Scale, Balance)
────────────────────────────────────────────────────────────────────────────────
WM-811K Data Preparation & Hybrid Balancing

### 🎯 PURPOSE
This script serves as the "Gatekeeper" between raw feature extraction and model training. 
It addresses the three fundamental challenges of the WM-811K dataset: 
1. **Data Leakage:** Ensuring the Test set is never seen during scaling or balancing.
2. **Feature Scaling:** Normalizing features with vastly different ranges (e.g., Area vs Density).
3. **Extreme Imbalance:** 'None' class has ~147k samples, while 'Scratch' has ~500.

### ⚙️ THE PIPELINE (Strict Execution Order)

1. **Stratified Split (The "Leak-Proof" Wall)**
   - We split the data into Training (80%) and Testing (20%) sets immediately.
   - **Critical:** We use `stratify=y` to ensure that rare defects (like 'Scratch') 
     are present in both sets in the exact same proportion as the original data. 
     Without this, the Test set might end up with zero Scratches.

2. **Standardization (Z-Score Scaling)**
   - Formula: z = (x - u) / s
   - **Anti-Leakage Logic:** The `StandardScaler` computes the Mean (u) and Std Dev (s) 
     using **ONLY the Training Set**. These Training stats are then applied to scale the Test set. 
   - *Why?* If we calculated the mean of the whole dataset, information from the 
     Test set would "leak" into the Training set, inflating our accuracy scores.

3. **Hybrid Balancing (The "Goldilocks" Strategy)**
   - Training on raw data causes the model to ignore defects (96% accuracy by guessing 'None').
   - We enforce a target of **500 samples per class** for the Training set:
     - **Majority Class ('None'):** 📉 Undersampled from ~147k -> 500. 
       (Reduces noise and training time).
     - **Minority Classes ('Scratch'):** 📈 Oversampled (SMOTE) from ~100 -> 500.
       (Creates synthetic points between neighbors to boost signal).
   - **Result:** A perfectly balanced training set where the model treats every class equally.

### 📦 OUTPUT (`model_ready_data.npz`)
A compressed dictionary containing:
1. `X_train_balanced`: Scaled & SMOTE-augmented (Use for training models).
2. `X_test`: Scaled real-world data (Use for final evaluation).
3. `scaler`: The saved scaler object (Required to preprocess new wafers in deployment).
────────────────────────────────────────────────────────────────────────────────
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # use StandardScaler to make sure features with big numbers don't accidentally overpower features with small numbers. It creates a "level playing field."
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress warnings from imblearn regarding class chunks
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

try:
    from config import FEATURES_FILE_CSV, PREPROCESSING_DIR, TEST_SPLIT_SIZE, RANDOM_SEED, TARGET_SAMPLES_PER_CLASS, MODEL_READY_DATA_FILE, SCALER_FILE
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import FEATURES_FILE_CSV, PREPROCESSING_DIR, TEST_SPLIT_SIZE, RANDOM_SEED, TARGET_SAMPLES_PER_CLASS, MODEL_READY_DATA_FILE, SCALER_FILE

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data_for_modeling(
    feature_csv_path: str, 
    output_dir: str, 
    test_size: float, 
    seed: int
) -> None:
    """
    Loads features, splits data, scales it, and applies hybrid balancing.
    
    Args:
        feature_csv_path (str): Path to the input CSV from Stage 2.
        output_dir (str): Folder to save the resulting .npz and scaler.
        test_size (float): Proportion of dataset to include in the test split.
        seed (int): Random seed for reproducibility.
    """
    print("\n" + "="*50)
    print("🚀 STAGE 3: DATA PREPARATION & BALANCING")
    print("="*50)

    # --- 1. Load Data ---
    print(f"📂 Loading features from: {feature_csv_path}")
    if not os.path.exists(feature_csv_path):
        raise FileNotFoundError(f"Input file not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path)
    print(f"   Original Shape: {df.shape}")

    # --- 2. Identify Target Column ---
    # Handles variations if previous script named it 'label' or 'target'
    if "label" in df.columns:
        target_col = "label"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise KeyError("Dataset missing 'label' or 'target' column.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.to_list()

    # --- 3. CRITICAL: Stratified Train-Test Split ---
    # We split BEFORE balancing to ensure the test set is 100% real data.
    print(f"✂️  Splitting data (Test Size={test_size}, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=seed
    )
    print(f"   Train set (Imbalanced): {X_train.shape[0]} samples")
    print(f"   Test set (Locked):      {X_test.shape[0]} samples")

    # --- 4. CRITICAL: Scaling ---
    # Fit on Train, Transform on Test. Prevents leakage.
    print("📏 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Note: use transform(), not fit_transform()

    # Save point: Imbalanced data (useful for Cross-Validation later)
    X_train_imbalanced = X_train_scaled.copy()
    y_train_imbalanced = y_train.copy()

    # --- 5. CRITICAL: Hybrid Balancing ---
    print(f"⚖️  Applying Hybrid Balancing (Target: {TARGET_SAMPLES_PER_CLASS} samples/class)...")
    
    # 5a. Analyze current distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # 5b. Define Strategies
    # - Undersample classes that have TOO MANY (> 500)
    # - SMOTE classes that have TOO FEW (< 500)
    under_strategy = {k: TARGET_SAMPLES_PER_CLASS for k, v in class_counts.items() if v > TARGET_SAMPLES_PER_CLASS}
    over_strategy  = {k: TARGET_SAMPLES_PER_CLASS for k, v in class_counts.items() if v < TARGET_SAMPLES_PER_CLASS}
    
    # 5c. Build Pipeline
    steps = []
    if under_strategy:
        steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=seed)))
    
    if over_strategy:
        # k_neighbors=3 is safer for very small classes than the default 5
        steps.append(('over', SMOTE(sampling_strategy=over_strategy, random_state=seed, k_neighbors=3)))
        
    balancer = ImbPipeline(steps)
    
    # 5d. Execute Balancing
    try:
        X_train_balanced, y_train_balanced = balancer.fit_resample(X_train_scaled, y_train)
        print(f"   ✅ Balancing Complete.")
        print(f"   Balanced Train Shape: {X_train_balanced.shape}")
    except ValueError as e:
        print(f"⚠️ Balancing Warning: {e}")
        print("   (Likely a class has fewer than 4 samples. Using Imbalanced data as fallback.)")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    # --- 6. Save Outputs ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Scaler (for deployment)
    scaler_path = str(SCALER_FILE)
    joblib.dump(scaler, scaler_path)
    
    # Save Data
    npz_path = str(MODEL_READY_DATA_FILE)
    np.savez_compressed(
        npz_path,
        X_train_imbalanced=X_train_imbalanced,
        y_train_imbalanced=y_train_imbalanced.to_numpy(),
        X_train_balanced=X_train_balanced,
        y_train_balanced=y_train_balanced.to_numpy(),
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(),
        feature_names=np.array(feature_names)
    )
    
    print("="*50)
    print(f"💾 RESULTS SAVED:")
    print(f"   Scaler: {scaler_path}")
    print(f"   Data:   {npz_path}")
    print("✅ STAGE 3 COMPLETE")


# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ EXECUTION ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    input_csv = str(FEATURES_FILE_CSV)
    output_dir = str(PREPROCESSING_DIR)

    prepare_data_for_modeling(
        feature_csv_path=input_csv, 
        output_dir=output_dir,
        test_size=TEST_SPLIT_SIZE,
        seed=RANDOM_SEED
    )
