import sys
import os
import subprocess
from pathlib import Path

# ───────────────────────────────────────────────
# 1️⃣ COLOR OUTPUT
# ───────────────────────────────────────────────
class Colors:
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
    try:
        print(f"{color}{message}{Colors.ENDC}")
    except:
        print(message)

def print_header(message):
    print("\n" + "="*70)
    print_colored(f"  {message}", Colors.BOLD + Colors.HEADER)
    print("="*70)

def print_success(message):
    print_colored(f"✅ {message}", Colors.OKGREEN)

def print_error(message):
    print_colored(f"❌ {message}", Colors.FAIL)

def print_warning(message):
    print_colored(f"⚠️  {message}", Colors.WARNING)

def print_info(message):
    print_colored(f"ℹ️  {message}", Colors.OKCYAN)

# ───────────────────────────────────────────────
# 2️⃣ PYTHON 3.10 CHECK & VENV SETUP
# ───────────────────────────────────────────────
PYTHON_BIN = "python3.10"
VENV_DIR = Path(".venv")

def check_python310():
    try:
        version_output = subprocess.check_output([PYTHON_BIN, "--version"]).decode()
        print_info(f"Detected {version_output.strip()}")
        return True
    except Exception:
        print_warning("Python 3.10.x not detected")
        return False

def create_venv():
    if not VENV_DIR.exists():
        print_info("Creating virtual environment with Python 3.10...")
        try:
            subprocess.run([PYTHON_BIN, "-m", "venv", str(VENV_DIR)], check=True)
            print_success("Virtual environment created")
        except subprocess.CalledProcessError:
            print_error("Failed to create virtual environment")
            return False
    else:
        print_info("Virtual environment already exists")
    return True

def install_requirements():
    pip_bin = VENV_DIR / "bin" / "pip"
    req_file = Path("ml_flow/requirement.txt")
    if not req_file.exists():
        print_error("requirement.txt not found")
        return False
    print_info("Installing dependencies in virtual environment...")
    try:
        subprocess.run([str(pip_bin), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_bin), "install", "-r", str(req_file)], check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to install dependencies")
        return False

# ───────────────────────────────────────────────
# 3️⃣ DATASET CHECK & PIPELINE
# ───────────────────────────────────────────────
def check_dataset():
    dataset_paths = [
        Path("ml_flow/datasets/LSWMD.pkl"),
        Path("datasets/LSWMD.pkl"),
        Path("LSWMD.pkl")
    ]
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            size_mb = dataset_path.stat().st_size / (1024*1024)
            print_success(f"Dataset found: {dataset_path.absolute()}")
            print_info(f"Size: {size_mb:.1f} MB")
            return True
    print_error("Dataset not found!")
    print_warning("To download, run Kaggle scripts or set KAGGLE_JSON secret")
    return False

def run_pipeline():
    pipeline_script = Path("ml_flow/main.py")
    if not pipeline_script.exists():
        print_error("Pipeline script not found: ml_flow/main.py")
        return False
    python_bin = VENV_DIR / "bin" / "python"
    try:
        subprocess.run([str(python_bin), str(pipeline_script)], check=True, cwd=str(Path.cwd()))
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Pipeline execution failed (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print_warning("Pipeline interrupted by user")
        return False

# ───────────────────────────────────────────────
# 4️⃣ MAIN EXECUTION
# ───────────────────────────────────────────────
def main():
    print_colored("\n🏭  WM-811K Wafer Defect Classification - One-Click Execution\n",
                  Colors.BOLD + Colors.HEADER)

    # Step 0: Ensure Python 3.10
    if not check_python310():
        print_error("Python 3.10 is required. Please install it manually.")
        sys.exit(1)

    if not create_venv():
        sys.exit(1)

    if not install_requirements():
        sys.exit(1)

    # Step 1: Check dataset
    if not check_dataset():
        sys.exit(1)

    # Step 2: Run pipeline
    success = run_pipeline()

    print_header("📊 Execution Summary")
    if success:
        print_success("Pipeline completed successfully! 🎉")
    else:
        print_error("Pipeline execution failed.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)