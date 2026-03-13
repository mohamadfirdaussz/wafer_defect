# # -*- coding: utf-8 -*-
# """
# 📜 feature_selection.py (Stage 4: The Selection Funnel)
# ────────────────────────────────────────────────────────────────────────────────
# WM-811K Feature Optimization Pipeline

# ### 🎯 PURPOSE
# This script addresses the **Curse of Dimensionality**.
# Stage 3.5 generated ~6,500+ features. While these capture complex patterns, 
# using all of them would crash complex models (like SVM) and lead to severe overfitting.



# ### ⚙️ THE STRATEGY: "The Funnel"
# We use a two-stage scientific approach to reduce features while keeping the signal:

# 1. **Stage 1: Global Pre-Filtering (The Sieve)**
#    - **Method:** ANOVA (Analysis of Variance) F-value.
#    - **Logic:** Fast statistical test. Discards features that have statistically 
#      zero correlation with the defect label.
#    - **Reduction:** ~6,500 → 1,000 features.

# 2. **Stage 2: Fine Selection (The Magnifying Glass)**
#    - We run 3 parallel "Tracks" to find the best subset. Each uses a different mathematical logic:
#      - **Track 4B (Wrapper - RFE):** Recursively removes the weakest feature until 25 remain.
#      - **Track 4C (Embedded - Random Forest):** Keeps features that best split the decision trees.
#      - **Track 4D (Embedded - Lasso):** Uses L1 regularization to mathematically force weak coefficients to zero.

# ### 📦 OUTPUT
# Saves three optimized datasets (`.npz`) to `feature_selection_results/`.
# These "Golden Subsets" will compete in the final Model Tuning stage.
# ────────────────────────────────────────────────────────────────────────────────
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# import warnings
# import joblib
# from typing import List, Tuple

# # ML Imports
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE, SelectKBest, f_classif
# from sklearn.linear_model import LogisticRegression

# # Suppress convergence warnings for cleaner output
# warnings.filterwarnings("ignore")

# # ──────────────────────────────────────────────────────────────────────────────
# # 📝 CONFIGURATION
# # ──────────────────────────────────────────────────────────────────────────────

# # ──────────────────────────────────────────────────────────────────────────────
# # 📝 CONFIGURATION
# # ──────────────────────────────────────────────────────────────────────────────

# try:
#     from config import EXPANDED_DATA_FILE, FEATURE_SELECTION_DIR, N_PREFILTER, N_FEATURES_RFE, N_FEATURES_RF
# except ImportError:
#     import sys
#     sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#     from config import EXPANDED_DATA_FILE, FEATURE_SELECTION_DIR, N_PREFILTER, N_FEATURES_RFE, N_FEATURES_RF


# # ──────────────────────────────────────────────────────────────────────────────
# # 1️⃣ HELPER FUNCTIONS
# # ──────────────────────────────────────────────────────────────────────────────

# def save_track_data(
#     output_dir: str, 
#     track_name: str, 
#     X_train: np.ndarray, 
#     X_test: np.ndarray, 
#     y_train: np.ndarray, 
#     y_test: np.ndarray, 
#     features: List[str]
# ):
#     """
#     Saves a 'Golden Subset' of features to a compressed .npz file.

#     Args:
#         output_dir (str): Directory where the file will be saved.
#         track_name (str): Identifier for the selection method (e.g., '4B_RFE').
#         X_train (np.ndarray): Reduced training data.
#         X_test (np.ndarray): Reduced testing data.
#         y_train (np.ndarray): Training labels.
#         y_test (np.ndarray): Testing labels.
#         features (List[str]): List of the selected feature names.

#     Returns:
#         None: Saves file to disk.
#     """
#     file_path = os.path.join(output_dir, f"data_track_{track_name}.npz")
    
#     np.savez_compressed(
#         file_path,
#         X_train=X_train,
#         y_train=y_train,
#         X_test=X_test,
#         y_test=y_test,
#         feature_names=np.array(features)
#     )
#     print(f"✅ Saved Track {track_name}: {X_train.shape[1]} features")
#     print(f"   Path: {file_path}")


# def run_feature_selection(input_file_path: str, output_dir: str):
#     """
#     Orchestrates the Multi-Track Feature Selection Pipeline.

#     **The Funnel Architecture:**
#     1.  **Stage 1: ANOVA Pre-filtering (The Sieve)**
#         -   Fast statistical test (F-value) to discard features with zero correlation.
#         -   Reduces ~6,500 features to ~1,000.
#         -   Why? Computationally expensive wrappers like RFE would take days on 6,500 features.
    
#     2.  **Stage 2: Fine Selection (The Tracks)**
#         -   **Track 4B (RFE):** Wrapper method. Recursively kills the weakest feature. Best accuracy.
#         -   **Track 4C (Random Forest):** Embedded method. Uses tree splits to find importance. Good for non-linear.
#         -   **Track 4D (Lasso):** Regularization method. Forces weak coefficients to zero. Best for interpretability.

#     Args:
#         input_file_path (str): Path to the expanded feature set (.npz).
#         output_dir (str): Directory to save the resulting 3 datasets.
#     """
#     print(f"\n" + "="*50)
#     print(f"🏃 STARTING FEATURE SELECTION FUNNEL")
#     print(f"   Input: {input_file_path}")
#     print("="*50)
    
#     # --- 1. Load Data ---
#     if not os.path.exists(input_file_path):
#         print(f"❌ ERROR: File not found at {input_file_path}")
#         return

#     print(f"📂 Loading High-Dimensional Data...")
#     try:
#         data = np.load(input_file_path, allow_pickle=True)
#         X_train = data['X_train']
#         y_train = data['y_train']
#         X_test = data['X_test']
#         y_test = data['y_test']
#         feature_names = data['feature_names']
        
#         print(f"   Loaded Train: {X_train.shape} | Test: {X_test.shape}")
#         print(f"   Total Features: {len(feature_names)}")
#     except KeyError as e:
#         print(f"❌ Error loading NPZ keys: {e}")
#         return

#     # ──────────────────────────────────────────────────────────────────────────
#     # ⚡ STAGE 1: GLOBAL PRE-FILTERING (ANOVA)
#     # ──────────────────────────────────────────────────────────────────────────
#     # Goal: Quickly cut 6,500 -> 1,000 using simple statistics.
#     # Why: Running RFE on 6,500 features would take days. ANOVA takes seconds.
    
#     if X_train.shape[1] > N_PREFILTER:
#         print(f"\n⚡ STAGE 1: Pre-filtering (ANOVA F-value)...")
#         print(f"   Reducing {X_train.shape[1]} -> {N_PREFILTER} features.")
        
#         pre_selector = SelectKBest(score_func=f_classif, k=N_PREFILTER)
#         X_train_filtered = pre_selector.fit_transform(X_train, y_train)
        
#         # Get mask of survivors
#         filter_mask = pre_selector.get_support()
#         filtered_feature_names = feature_names[filter_mask]
        
#         print(f"   ✅ Pre-filter complete.")
#     else:
#         print("\n⚡ SKIPPING STAGE 1 (Feature count already low).")
#         X_train_filtered = X_train
#         filtered_feature_names = feature_names

#     # ──────────────────────────────────────────────────────────────────────────
#     # 🔄 TRACK 4B: WRAPPER METHOD (RFE)
#     # ──────────────────────────────────────────────────────────────────────────
#     print(f"\n--- Track 4B: Recursive Feature Elimination (RFE) ---")
#     print(f"   Target: {N_FEATURES_RFE} features")
    
#     # We use Logistic Regression as the 'estimator' because it is linear and fast.
#     # step=50: Drops 50 weakest features per iteration (speeds up process).
#     model_rfe = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
#     rfe = RFE(model_rfe, n_features_to_select=N_FEATURES_RFE, step=50, verbose=1)
    
#     print("   Running RFE (this may take a moment)...")
 
#     try:
#         rfe.fit(X_train_filtered, y_train)
        
#         # Extract survivors
#         rfe_names = filtered_feature_names[rfe.support_]
        
#         # Map back to original X_train
#         final_mask = np.isin(feature_names, rfe_names)
#         X_train_4B = X_train[:, final_mask]
#         X_test_4B = X_test[:, final_mask]
        
#         save_track_data(output_dir, "4B_RFE", X_train_4B, X_test_4B, y_train, y_test, rfe_names)

#     except Exception as e:
#         print(f"\n❌ CRITICAL ERROR in Track 4B (RFE): {e}")
#         print("   Tips for debugging:")
#         print("   - MemoryError: Reduce N_PREFILTER (currently {N_PREFILTER}) or check system RAM.")
#         print("   - ValueError: Check for NaNs or infinite values in input data.")
#         print("   - RuntimeWarning: Convergence issues. Increase max_iter in LogisticRegression.")
#         print("   ⚠️ Skipping Track 4B and continuing...\n")

#     # ──────────────────────────────────────────────────────────────────────────
#     # 🌲 TRACK 4C: EMBEDDED METHOD (Random Forest)
#     # ──────────────────────────────────────────────────────────────────────────
#     print(f"\n--- Track 4C: Random Forest Importance ---")
#     print(f"   Target: {N_FEATURES_RF} features")
    
#     # RF is robust against noise, so we can feed it the original X_train (not filtered)
#     # if we have enough RAM, but using filtered is safer for speed.
#     rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train) # Using full set to capture non-linear interactions ANOVA missed
    
#     importances = rf.feature_importances_
    
#     # Get indices of top K importance scores
#     indices = np.argsort(importances)[-N_FEATURES_RF:]
    
#     rf_names = feature_names[indices]
#     X_train_4C = X_train[:, indices]
#     X_test_4C = X_test[:, indices]
    
#     save_track_data(output_dir, "4C_RF_Importance", X_train_4C, X_test_4C, y_train, y_test, rf_names)
    
#     # Save CSV Ranking for Thesis
#     ranking_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#     ranking_df = ranking_df.sort_values(by='Importance', ascending=False)
#     ranking_csv = os.path.join(output_dir, "RF_Feature_Importance_Ranking.csv")
#     ranking_df.to_csv(ranking_csv, index=False)
#     print(f"   📄 Feature Ranking saved to {ranking_csv}")

#     # ──────────────────────────────────────────────────────────────────────────
#     # 📉 TRACK 4D: EMBEDDED METHOD (Lasso L1)
#     # ──────────────────────────────────────────────────────────────────────────
#     print("\n--- Track 4D: Lasso Regularization (L1) ---")
#     print(f"   Applying sparsity penalty...")
    
#     # C=0.005 is a strong penalty (Smaller C = More features removed).
#     l1_model = LogisticRegression(
#         penalty='l1', solver='liblinear', C=0.005, random_state=42, max_iter=2000
#     )
    
#     l1_model.fit(X_train_filtered, y_train)
    
#     # Select features where coefficient is NOT zero
#     l1_support = np.any(np.abs(l1_model.coef_) > 1e-5, axis=0)
#     lasso_names = filtered_feature_names[l1_support]
    
#     print(f"   Lasso selected {len(lasso_names)} features.")
    
#     # Safety: If Lasso kills everything, fallback to Top 25 from ANOVA
#     if len(lasso_names) < 2:
#         print("⚠️ WARNING: Lasso selected too few features. Falling back to Top 25 ANOVA.")
#         lasso_names = filtered_feature_names[:25]
    
#     final_mask = np.isin(feature_names, lasso_names)
#     X_train_4D = X_train[:, final_mask]
#     X_test_4D = X_test[:, final_mask]
    
#     save_track_data(output_dir, "4D_Lasso", X_train_4D, X_test_4D, y_train, y_test, lasso_names)


# if __name__ == "__main__":
    
#     output_dir = str(FEATURE_SELECTION_DIR)
#     os.makedirs(output_dir, exist_ok=True)
    
#     input_file_path = str(EXPANDED_DATA_FILE)
    
#     run_feature_selection(input_file_path, output_dir)
    
#     print("\n" + "="*50)
#     print("✅ FEATURE SELECTION COMPLETE")
#     print("Ready for Stage 5: Model Tuning.")








# -*- coding: utf-8 -*-
"""
📜 feature_selection.py (Stage 4: The Selection Funnel)
────────────────────────────────────────────────────────────────────────────────
WM-811K Feature Optimization Pipeline

### 🎯 PURPOSE
This script addresses the **Curse of Dimensionality**.
Stage 3.5 generated ~6,500+ features. While these capture complex patterns, 
using all of them would crash complex models (like SVM) and lead to severe overfitting.



### ⚙️ THE STRATEGY: "The Funnel"
We use a two-stage scientific approach to reduce features while keeping the signal:

1. **Stage 1: Global Pre-Filtering (The Sieve)**
   - **Method:** ANOVA (Analysis of Variance) F-value.
   - **Logic:** Fast statistical test. Discards features that have statistically 
     zero correlation with the defect label.
   - **Reduction:** ~6,500 → 1,000 features.

2. **Stage 2: Fine Selection (The Magnifying Glass)**
   - We run 3 parallel "Tracks" to find the best subset. Each uses a different mathematical logic:
     - **Track 4B (Wrapper - RFE):** Recursively removes the weakest feature until 25 remain.
     - **Track 4C (Embedded - Random Forest):** Keeps features that best split the decision trees.
     - **Track 4D (Embedded - Lasso):** Uses L1 regularization to mathematically force weak coefficients to zero.

### 📦 OUTPUT
Saves three optimized datasets (`.npz`) to `feature_selection_results/`.
These "Golden Subsets" will compete in the final Model Tuning stage.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import joblib
from typing import List, Tuple

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 📝 CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

try:
    from config import EXPANDED_DATA_FILE, FEATURE_SELECTION_DIR, N_PREFILTER, N_FEATURES_RFE, N_FEATURES_RF
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import EXPANDED_DATA_FILE, FEATURE_SELECTION_DIR, N_PREFILTER, N_FEATURES_RFE, N_FEATURES_RF


# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def save_track_data(
    output_dir: str, 
    track_name: str, 
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray, 
    features: List[str]
):
    """
    Saves a 'Golden Subset' of features to a compressed .npz file.
    """
    file_path = os.path.join(output_dir, f"data_track_{track_name}.npz")
    
    np.savez_compressed(
        file_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(features)
    )
    print(f"✅ Saved Track {track_name}: {X_train.shape[1]} features")
    print(f"   Path: {file_path}")


def run_feature_selection(input_file_path: str, output_dir: str):
    """
    Main execution flow for the Feature Selection Funnel.
    """
    print(f"\n" + "="*50)
    print(f"🏃 STARTING FEATURE SELECTION FUNNEL")
    print(f"   Input: {input_file_path}")
    print("="*50)
    
    # --- 1. Load Data ---
    if not os.path.exists(input_file_path):
        print(f"❌ ERROR: File not found at {input_file_path}")
        return

    print(f"📂 Loading High-Dimensional Data...")
    try:
        data = np.load(input_file_path, allow_pickle=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        feature_names = data['feature_names']
        
        print(f"   Loaded Train: {X_train.shape} | Test: {X_test.shape}")
        print(f"   Total Features: {len(feature_names)}")
    except KeyError as e:
        print(f"❌ Error loading NPZ keys: {e}")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # ⚡ STAGE 1: GLOBAL PRE-FILTERING (ANOVA)
    # ──────────────────────────────────────────────────────────────────────────
    # Goal: Quickly cut 6,500 -> 1,000 using simple statistics.
    # Why: Running RFE on 6,500 features would take days. ANOVA takes seconds.
    
    if X_train.shape[1] > N_PREFILTER:
        print(f"\n⚡ STAGE 1: Pre-filtering (ANOVA F-value)...")
        print(f"   Reducing {X_train.shape[1]} -> {N_PREFILTER} features.")
        
        pre_selector = SelectKBest(score_func=f_classif, k=N_PREFILTER)
        X_train_filtered = pre_selector.fit_transform(X_train, y_train)
        
        # Get mask of survivors
        filter_mask = pre_selector.get_support()
        filtered_feature_names = feature_names[filter_mask]
        
        print(f"   ✅ Pre-filter complete.")
    else:
        print("\n⚡ SKIPPING STAGE 1 (Feature count already low).")
        X_train_filtered = X_train
        filtered_feature_names = feature_names

    # ──────────────────────────────────────────────────────────────────────────
    # 🔄 TRACK 4B: WRAPPER METHOD (RFE)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n--- Track 4B: Recursive Feature Elimination (RFE) ---")
    print(f"   Target: {N_FEATURES_RFE} features")
    
    # We use Logistic Regression as the 'estimator' because it is linear and fast.
    # step=50: Drops 50 weakest features per iteration (speeds up process).
    model_rfe = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

    
    rfe = RFE(model_rfe, n_features_to_select=N_FEATURES_RFE, step=50, verbose=1)
    
    print("   Running RFE (this may take a moment)...")
<<<<<<< HEAD
    rfe.fit(X_train_filtered, y_train)
    
    # Extract survivors
    rfe_names = filtered_feature_names[rfe.support_]
    
    # Map back to original X_train
    final_mask = np.isin(feature_names, rfe_names)
    X_train_4B = X_train[:, final_mask]
    X_test_4B = X_test[:, final_mask]
    
    save_track_data(output_dir, "4B_RFE", X_train_4B, X_test_4B, y_train, y_test, rfe_names)

=======
    print("   Running RFE (this may take a moment)...")
    try:
        rfe.fit(X_train_filtered, y_train)
        
        # Extract survivors
        rfe_names = filtered_feature_names[rfe.support_]
        
        # Map back to original X_train
        final_mask = np.isin(feature_names, rfe_names)
        X_train_4B = X_train[:, final_mask]
        X_test_4B = X_test[:, final_mask]
        
        save_track_data(output_dir, "4B_RFE", X_train_4B, X_test_4B, y_train, y_test, rfe_names)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in Track 4B (RFE): {e}")
        print("   Tips for debugging:")
        print("   - MemoryError: Reduce N_PREFILTER (currently {N_PREFILTER}) or check system RAM.")
        print("   - ValueError: Check for NaNs or infinite values in input data.")
        print("   - RuntimeWarning: Convergence issues. Increase max_iter in LogisticRegression.")
        print("   ⚠️ Skipping Track 4B and continuing...\n")
    
>>>>>>> 7ddbfddb6b368d04f8e6eb739eca30893b562401
    # ──────────────────────────────────────────────────────────────────────────
    # 🌲 TRACK 4C: EMBEDDED METHOD (Random Forest)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n--- Track 4C: Random Forest Importance ---")
    print(f"   Target: {N_FEATURES_RF} features")
    
    # RF is robust against noise, so we can feed it the original X_train (not filtered)
    # if we have enough RAM, but using filtered is safer for speed.
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train) # Using full set to capture non-linear interactions ANOVA missed
    
    importances = rf.feature_importances_
    
    # Get indices of top K importance scores
    indices = np.argsort(importances)[-N_FEATURES_RF:]
    
    rf_names = feature_names[indices]
    X_train_4C = X_train[:, indices]
    X_test_4C = X_test[:, indices]
    
    save_track_data(output_dir, "4C_RF_Importance", X_train_4C, X_test_4C, y_train, y_test, rf_names)
    
    # Save CSV Ranking for Thesis
    ranking_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    ranking_df = ranking_df.sort_values(by='Importance', ascending=False)
    ranking_csv = os.path.join(output_dir, "RF_Feature_Importance_Ranking.csv")
    ranking_df.to_csv(ranking_csv, index=False)
    print(f"   📄 Feature Ranking saved to {ranking_csv}")

    # ──────────────────────────────────────────────────────────────────────────
    # 📉 TRACK 4D: EMBEDDED METHOD (Lasso L1)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Track 4D: Lasso Regularization (L1) ---")
    print(f"   Applying sparsity penalty...")
    
    # C=0.005 is a strong penalty (Smaller C = More features removed).
    l1_model = LogisticRegression(
        penalty='l1', solver='liblinear', C=0.005, random_state=42, max_iter=2000
    )
    
    l1_model.fit(X_train_filtered, y_train)
    
    # Select features where coefficient is NOT zero
    l1_support = np.any(np.abs(l1_model.coef_) > 1e-5, axis=0)
    lasso_names = filtered_feature_names[l1_support]
    
    print(f"   Lasso selected {len(lasso_names)} features.")
    
    # Safety: If Lasso kills everything, fallback to Top 25 from ANOVA
    if len(lasso_names) < 2:
        print("⚠️ WARNING: Lasso selected too few features. Falling back to Top 25 ANOVA.")
        lasso_names = filtered_feature_names[:25]
    
    final_mask = np.isin(feature_names, lasso_names)
    X_train_4D = X_train[:, final_mask]
    X_test_4D = X_test[:, final_mask]
    
    save_track_data(output_dir, "4D_Lasso", X_train_4D, X_test_4D, y_train, y_test, lasso_names)


if __name__ == "__main__":
    
    output_dir = str(FEATURE_SELECTION_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    input_file_path = str(EXPANDED_DATA_FILE)
    
    run_feature_selection(input_file_path, output_dir)
    
    print("\n" + "="*50)
    print("✅ FEATURE SELECTION COMPLETE")
    print("Ready for Stage 5: Model Tuning.")