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
# 2️⃣ PYTHON 3.10 CHECK
# ───────────────────────────────────────────────
def check_python310():
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Detected Python {version_str} ({sys.executable})")
    if version.major == 3 and version.minor == 10:
        print_success("Python 3.10.x detected")
        return True
    print_warning("Python 3.10.x not detected")
    return False

def setup_python310():
    setup_script = Path("scripts/setup_python310.sh")
    if not setup_script.exists():
        print_error("setup_python310.sh not found!")
        return False
    try:
        print_info("🔧 Installing Python 3.10.8...")
        subprocess.run(["bash", str(setup_script)], check=True)
        print_success("Python 3.10.8 installation completed")
        return True
    except subprocess.CalledProcessError:
        print_error("Python 3.10 installation failed")
        return False

# ───────────────────────────────────────────────
# 3️⃣ EXISTING ENVIRONMENT & PIPELINE CHECKS
# ───────────────────────────────────────────────
def check_python_version():
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
    print_header("2️⃣  Checking Dataset")
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

def check_dependencies():
    print_header("3️⃣  Checking Dependencies")
    critical_packages = ['numpy','pandas','sklearn','xgboost']
    missing = []
    for pkg in critical_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print_warning(f"Missing packages: {', '.join(missing)}")
        req_file = Path("ml_flow/requirement.txt")
        if req_file.exists():
            try:
                subprocess.run([sys.executable,"-m","pip","install","-r",str(req_file),"--quiet"], check=True)
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
    print_header("4️⃣  Running ML Pipeline")
    pipeline_script = Path("ml_flow/main.py")
    if not pipeline_script.exists():
        print_error("Pipeline script not found: ml_flow/main.py")
        return False
    try:
        subprocess.run([sys.executable,str(pipeline_script)], check=True, cwd=str(Path.cwd()))
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
    print_colored("\n🏭  WM-811K Wafer Defect Classification - One-Click Execution\n", Colors.BOLD + Colors.HEADER)

    # Step 0: Ensure Python 3.10
    if not check_python310():
        if not setup_python310():
            sys.exit(1)

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
    success = run_pipeline()

    print_header("📊 Execution Summary")
    if success:
        print_success("Pipeline completed successfully! 🎉")
    else:
        print_error("Pipeline execution failed.")
        print_info("Check the error messages above for details.")
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