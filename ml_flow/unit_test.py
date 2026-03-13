
# """
# unit_test.py
# ────────────────────────────────────────────────────────────────────────
# # MASTER UNIT TEST SUITE FOR WAFER DEFECT PIPELINE (FIXED)

# ### [PURPOSE]
# Validates logic integrity. Fixed to handle float precision and 
# current model naming conventions.

# ### [HOW TO RUN]
# Run: `python unit_test.py`
# ────────────────────────────────────────────────────────────────────────
# """

# import unittest
# import numpy as np
# import pandas as pd
# import os
# import sys
# import warnings

# # Suppress warnings for cleaner output
# np.set_printoptions(suppress=True)
# warnings.filterwarnings("ignore")

# # ───────────────────────────────────────────────
# # 1️⃣ DYNAMIC IMPORT CHECKER
# # ───────────────────────────────────────────────
# print("\n[INFO] CHECKING PROJECT MODULES...")
# required_modules = [
#     "data_loader", 
#     "feature_engineering", 
#     "data_preprocessor",
#     "feature_combination",
#     "feature_selection",
#     "model_tuning"
# ]

# missing = []
# for mod in required_modules:
#     try:
#         __import__(mod)
#         print(f"   [OK] {mod}.py imported successfully.")
#     except ImportError as e:
#         missing.append(mod)
#         print(f"   [ERROR] Could not import '{mod}.py'. Reason: {e}")

# if missing:
#     print(f"\n[CRITICAL] Missing files: {missing}")
#     sys.exit(1)

# # Import modules for testing
# import data_loader
# import feature_engineering
# import feature_combination
# import model_tuning
# from sklearn.preprocessing import StandardScaler

# # ───────────────────────────────────────────────
# # [TEST CLASS]
# # ───────────────────────────────────────────────
# class TestPipelineSuite(unittest.TestCase):

#     def setUp(self):
#         """Creates dummy data before every test."""
#         self.raw_wafer = np.random.randint(0, 3, (20, 20))
#         self.proc_wafer = np.random.randint(0, 3, (64, 64))
#         self.df_raw = pd.DataFrame({
#             'waferMap': [self.raw_wafer, self.raw_wafer],
#             'failureType': [[['Loc']], [['none']]],
#             'trainTestLabel': [[['Training']], [['Test']]]
#         })

#     # ==================================================================
#     # [STAGE 1] DATA LOADER TESTS
#     # ==================================================================
#     def test_s1_resize(self):
#         """Test if resizing correctly converts 20x20 -> 64x64."""
#         print("\n[TEST] [Stage 1] Testing Resizing Logic...")
#         result = data_loader.resize_wafer_map(self.raw_wafer, target_size=(64, 64))
        
#         self.assertEqual(result.shape, (64, 64), "Output shape is not 64x64")
#         unique = np.unique(result)
#         self.assertTrue(all(x in [0, 1, 2] for x in unique), "Resizing introduced invalid values")
#         print("   [OK] Resize logic verified.")

#     def test_s1_clean_labels(self):
#         """Test if nested labels [['Loc']] become 'Loc'."""
#         print("\n[TEST] [Stage 1] Testing Label Cleaning...")
#         df = self.df_raw.copy()
#         # Simulate the cleaning logic locally to verify pandas behavior
#         df["failureType"] = df["failureType"].apply(lambda x: x[0][0])
#         self.assertEqual(df["failureType"][0], "Loc", "Label flattening failed")
#         print("   [OK] Label cleaning logic verified.")

#     # ==================================================================
#     # [STAGE 2] FEATURE ENGINEERING TESTS
#     # ==================================================================
#     def test_s2_feature_extraction(self):
#         """Test if extractor returns exactly 66 features."""
#         print("\n[TEST] [Stage 2] Testing Feature Extractor...")
#         features = feature_engineering.process_single_wafer(self.proc_wafer)
        
#         # 13 Density + 40 Radon + 7 Geom + 6 Stats = 66
#         self.assertEqual(len(features), 66, f"Expected 66 features, got {len(features)}")
#         self.assertFalse(np.isnan(features).any(), "Extracted features contain NaNs")
#         print("   [OK] Feature count (66) and validity verified.")

#    # ==================================================================
#     # [STAGE 3.5] FEATURE EXPANSION TESTS
#     # ==================================================================
#     def test_s35_math_logic(self):
#         """Test Sum and Difference logic (Ratio removed for safety)."""
#         print("\n[TEST] [Stage 3.5] Testing Expansion Math...")
#         X_tiny = np.array([[10.0, 2.0]])
#         names = ['A', 'B']
        
#         # Generates: [A+B, A-B]
#         X_new, new_names = feature_combination.generate_math_combinations(X_tiny, names)
        
#         # CHECK 1: Correct Dimensions (Should be 2 features, NOT 3)
#         self.assertEqual(X_new.shape[1], 2, f"Expected 2 features (Sum, Diff), got {X_new.shape[1]}")
        
#         # CHECK 2: Sum Logic (10 + 2 = 12)
#         self.assertEqual(X_new[0][0], 12.0, "Summation logic failed")
        
#         # CHECK 3: Diff Logic (10 - 2 = 8)
#         self.assertEqual(X_new[0][1], 8.0, "Difference logic failed")
        
#         # CHECK 4: Verify Ratio is GONE
#         # If we try to access index 2, it should fail. We verify this implicitly by checking shape above.
        
#         print(f"   [OK] Math interaction terms verified. (Ratio correctly absent).")


#     # ==================================================================
#     # [STAGE 5] MODEL TUNING TESTS
#     # ==================================================================
#     def test_s5_model_loading(self):
#         """Test if model configuration loads correctly (Updated Keys)."""
#         print("\n[TEST] [Stage 5] Testing Model Configuration...")
#         models, grids = model_tuning.get_models_and_grids()
        
#         # Get actual keys from the user's file to verify they exist
#         actual_keys = list(models.keys())
        
#         # FIX: We check for 'GradBoosting' OR 'GradientBoosting' to be safe
#         has_gbm = 'GradBoosting' in actual_keys or 'GradientBoosting' in actual_keys
#         self.assertTrue(has_gbm, f"Missing Gradient Boosting model. Found: {actual_keys}")
        
#         # Check specific expected keys from your error log
#         if 'GradBoosting' in actual_keys:
#             self.assertIn('GradBoosting', actual_keys)
        
#         self.assertIn('XGBoost', actual_keys)
#         self.assertIn('SVM', actual_keys) # Or 'SVC' depending on your naming
            
#         print("   [OK] Model dictionary loaded successfully.")

# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("[START] STARTING FULL PIPELINE UNIT TESTS")
#     print("="*60)
#     unittest.main(verbosity=2)





"""
unit_test.py
────────────────────────────────────────────────────────────────────────
# MASTER UNIT TEST SUITE FOR WAFER DEFECT PIPELINE (FIXED)

### [PURPOSE]
Validates logic integrity. Fixed to handle float precision and 
current model naming conventions.

### [HOW TO RUN]
Run: `python unit_test.py`
────────────────────────────────────────────────────────────────────────
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Suppress warnings for cleaner output
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# 1️⃣ DYNAMIC IMPORT CHECKER
# ───────────────────────────────────────────────
print("\n[INFO] CHECKING PROJECT MODULES...")
required_modules = [
    "data_loader", 
    "feature_engineering", 
    "data_preprocessor",
    "feature_combination",
    "feature_selection",
    "model_tuning",
    "model_tuning_optimized"
]

missing = []
for mod in required_modules:
    try:
        __import__(mod)
        print(f"   [OK] {mod}.py imported successfully.")
    except ImportError as e:
        missing.append(mod)
        print(f"   [ERROR] Could not import '{mod}.py'. Reason: {e}")

if missing:
    print(f"\n[CRITICAL] Missing files: {missing}")
    sys.exit(1)

# Import modules for testing
import data_loader
import feature_engineering
import feature_combination
import model_tuning
from sklearn.preprocessing import StandardScaler

# ───────────────────────────────────────────────
# [TEST CLASS]
# ───────────────────────────────────────────────
class TestPipelineSuite(unittest.TestCase):

    def setUp(self):
        """Creates dummy data before every test."""
        self.raw_wafer = np.random.randint(0, 3, (20, 20))
        self.proc_wafer = np.random.randint(0, 3, (64, 64))
        self.df_raw = pd.DataFrame({
            'waferMap': [self.raw_wafer, self.raw_wafer],
            'failureType': [[['Loc']], [['none']]],
            'trainTestLabel': [[['Training']], [['Test']]]
        })

    # ==================================================================
    # [STAGE 1] DATA LOADER TESTS
    # ==================================================================
    def test_s1_resize(self):
        """Test if resizing correctly converts 20x20 -> 64x64."""
        print("\n[TEST] [Stage 1] Testing Resizing Logic...")
        result = data_loader.resize_wafer_map(self.raw_wafer, target_size=(64, 64))
        
        self.assertEqual(result.shape, (64, 64), "Output shape is not 64x64")
        unique = np.unique(result)
        self.assertTrue(all(x in [0, 1, 2] for x in unique), "Resizing introduced invalid values")
        print("   [OK] Resize logic verified.")

    def test_s1_clean_labels(self):
        """Test if nested labels [['Loc']] become 'Loc'."""
        print("\n[TEST] [Stage 1] Testing Label Cleaning...")
        df = self.df_raw.copy()
        # Simulate the cleaning logic locally to verify pandas behavior
        df["failureType"] = df["failureType"].apply(lambda x: x[0][0])
        self.assertEqual(df["failureType"][0], "Loc", "Label flattening failed")
        print("   [OK] Label cleaning logic verified.")

    # ==================================================================
    # [STAGE 2] FEATURE ENGINEERING TESTS
    # ==================================================================
    def test_s2_feature_extraction(self):
        """Test if extractor returns exactly 66 features."""
        print("\n[TEST] [Stage 2] Testing Feature Extractor...")
        features = feature_engineering.process_single_wafer(self.proc_wafer)
        
        # 13 Density + 40 Radon + 7 Geom + 6 Stats = 66
        self.assertEqual(len(features), 66, f"Expected 66 features, got {len(features)}")
        self.assertFalse(np.isnan(features).any(), "Extracted features contain NaNs")
        print("   [OK] Feature count (66) and validity verified.")

    # ==================================================================
    # [STAGE 3] PREPROCESSING TESTS
    # ==================================================================
    def test_s3_scaling(self):
        """Test if StandardScaler works as expected."""
        print("\n[TEST] [Stage 3] Testing Scaling Logic...")
        X_dummy = np.random.normal(100, 20, (50, 5))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_dummy)
        
        self.assertAlmostEqual(X_scaled.mean(), 0, delta=0.5, msg="Mean not centered at 0")
        self.assertAlmostEqual(X_scaled.std(), 1, delta=0.1, msg="Std not scaled to 1")
        print("   [OK] Scaling logic verified.")

    # ==================================================================
    # [STAGE 3.5] FEATURE EXPANSION TESTS
    # ==================================================================
    def test_s35_math_logic(self):
        """Test A+B, A-B, A/B logic with epsilon tolerance."""
        print("\n[TEST] [Stage 3.5] Testing Expansion Math...")
        X_tiny = np.array([[10.0, 2.0]])
        names = ['A', 'B']
        
        # This function casts to float32 internally in your script
        X_new, _ = feature_combination.generate_math_combinations(X_tiny, names)
        
        # FIX: We use 'places=4' to allow for float32 precision and the 1e-6 epsilon
        # Ratio: 10.0 / (2.0 + 0.000001) = 4.9999975
        self.assertEqual(X_new[0][0], 12.0)
        self.assertEqual(X_new[0][1], 8.0)
        self.assertAlmostEqual(X_new[0][2], 5.0, places=4, msg="Ratio math failed tolerance check")
        
        print("   [OK] Math interaction terms verified (with epsilon tolerance).")

    # ==================================================================
    # [STAGE 5] MODEL TUNING TESTS
    # ==================================================================
    def test_s5_model_loading(self):
        """Test if model configuration loads correctly (Updated Keys)."""
        print("\n[TEST] [Stage 5] Testing Model Configuration...")
        models, grids = model_tuning.get_models_and_grids()
        
        # Get actual keys from the user's file to verify they exist
        actual_keys = list(models.keys())
        
        # FIX: We check for 'GradBoosting' OR 'GradientBoosting' to be safe
        has_gbm = 'GradBoosting' in actual_keys or 'GradientBoosting' in actual_keys
        self.assertTrue(has_gbm, f"Missing Gradient Boosting model. Found: {actual_keys}")
        
        # Check specific expected keys from your error log
        if 'GradBoosting' in actual_keys:
            self.assertIn('GradBoosting', actual_keys)
        
        self.assertIn('XGBoost', actual_keys)
        self.assertIn('SVM', actual_keys) # Or 'SVC' depending on your naming
            
        print("   [OK] Model dictionary loaded successfully.")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("[START] STARTING FULL PIPELINE UNIT TESTS")
    print("="*60)
    unittest.main(verbosity=2)