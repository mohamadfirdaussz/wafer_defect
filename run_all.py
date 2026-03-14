"""
run_all.py
────────────────────────────────────────────────────────────────────────
WM-811K Wafer Defect Classification - One-Click Execution Script
────────────────────────────────────────────────────────────────────────

🎯 PURPOSE:
This is the master entry point for the entire project. It automates:
  1. Python version verification
  2. Dependency installation check
  3. Dataset verification
  4. Full pipeline execution

💻 USAGE:
Simply run:
    python run_all.py

────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import subprocess
from pathlib import Path

# ───────────────────────────────────────────────
# 1️⃣ COLOR OUTPUT (Cross-platform)
# ───────────────────────────────────────────────

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message, color=Colors.OKBLUE):
    """Print colored message with fallback for Windows"""
    try:
        print(f"{color}{message}{Colors.ENDC}")
    except:
        print(message)

def print_header(message):
    """Print a formatted header"""
    print("\n" + "="*70)
    print_colored(f"  {message}", Colors.BOLD + Colors.HEADER)
    print("="*70)

def print_success(message):
    """Print success message"""
    print_colored(f"✅ {message}", Colors.OKGREEN)

def print_error(message):
    """Print error message"""
    print_colored(f"❌ {message}", Colors.FAIL)

def print_warning(message):
    """Print warning message"""
    print_colored(f"⚠️  {message}", Colors.WARNING)

def print_info(message):
    """Print info message"""
    print_colored(f"ℹ️  {message}", Colors.OKCYAN)

# ───────────────────────────────────────────────
# 2️⃣ ENVIRONMENT CHECKS
# ───────────────────────────────────────────────

def check_python_version():
    """Verify Python version is compatible"""
    print_header("1️⃣  Checking Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Detected Python {version_str}")
    print_info(f"Interpreter: {sys.executable}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error("Python 3.9 or higher is required!")
        print_info("Please install Python 3.9, 3.10, or 3.11")
        return False
    
    if version.major == 3 and version.minor >= 12:
        print_warning("Python 3.12+ detected. Some ML libraries may have compatibility issues.")
        print_warning("Recommended: Python 3.9 - 3.11")
        print_info("Continuing anyway...")
    
    print_success("Python version is compatible")
    return True

def check_dataset():
    """Verify the dataset file exists"""
    print_header("2️⃣  Checking Dataset")
    
    # Define possible dataset locations
    dataset_paths = [
        Path("ml_flow/datasets/LSWMD.pkl"),
        Path("datasets/LSWMD.pkl"),
        Path("LSWMD.pkl")
    ]
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
            print_success(f"Dataset found: {dataset_path.absolute()}")
            print_info(f"Size: {file_size_mb:.1f} MB")
            return True
    
    print_error("Dataset not found!")
    print("")
    print_info("Expected location: ml_flow/datasets/LSWMD.pkl")
    print("")
    print_warning("To download the dataset, run ONE of these commands:")
    print("  • bash scripts/kaggle_setup_and_download.sh  (if you uploaded kaggle.json)")
    print("  • bash scripts/kaggle_from_secret.sh         (if you set KAGGLE_JSON secret)")
    print("")
    return False

def check_dependencies():
    """Verify critical dependencies are installed"""
    print_header("3️⃣  Checking Dependencies")
    
    critical_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'xgboost'
    ]
    
    missing = []
    for pkg in critical_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        print_info("Installing dependencies from ml_flow/requirement.txt...")
        
        req_file = Path("ml_flow/requirement.txt")
        if req_file.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
                    check=True
                )
                print_success("Dependencies installed successfully")
            except subprocess.CalledProcessError:
                print_error("Failed to install dependencies")
                return False
        else:
            print_error("requirement.txt not found")
            return False
    else:
        print_success("All critical dependencies are installed")
    
    return True

def run_pipeline():
    """Execute the main ML pipeline"""
    print_header("4️⃣  Running ML Pipeline")
    
    pipeline_script = Path("ml_flow/main.py")
    
    if not pipeline_script.exists():
        print_error("Pipeline script not found: ml_flow/main.py")
        return False
    
    print_info("Starting pipeline execution...")
    print_info("This may take 10-30 minutes depending on your hardware.")
    print("")
    
    try:
        # Run the pipeline in the same process to show live output
        subprocess.run(
            [sys.executable, str(pipeline_script)],
            check=True,
            cwd=str(Path.cwd())
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Pipeline execution failed (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print("")
        print_warning("Pipeline interrupted by user (Ctrl+C)")
        return False

# ───────────────────────────────────────────────
# 3️⃣ MAIN EXECUTION
# ───────────────────────────────────────────────

def main():
    """Main execution flow"""
    print("")
    print_colored("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║        🏭  WM-811K WAFER DEFECT CLASSIFICATION PIPELINE           ║
    ║                      One-Click Execution                          ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """, Colors.BOLD + Colors.HEADER)
    
    print_info(f"Project Directory: {Path.cwd()}")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check dataset
    if not check_dataset():
        sys.exit(1)
    
    # Step 3: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 4: Run pipeline
    print("")
    success = run_pipeline()
    
    # Final summary
    print("")
    print_header("📊 Execution Summary")
    
    if success:
        print_success("Pipeline completed successfully! 🎉")
        print("")
        print_info("Results are available in:")
        print("  📂 data_loader_results/")
        print("  📂 Feature_engineering_results/")
        print("  📂 preprocessing_results/")
        print("  📂 feature_selection_results/")
        print("  📂 model_artifacts/")
        print("")
        print_info("View final model performance:")
        print("  📄 model_artifacts/master_model_comparison.csv")
        print("")
    else:
        print_error("Pipeline execution failed.")
        print_info("Check the error messages above for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("")
        print_warning("\n🛑 Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("")
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)