# #!/bin/bash

# # ───────────────────────────────
# # Setup Python 3.10.8 in Codespace
# # ───────────────────────────────

# set -e  # Exit on any error

# echo "📌 Updating system packages..."
# sudo apt update -y
# sudo apt upgrade -y
# sudo apt install -y software-properties-common build-essential wget curl git

# echo "📌 Adding deadsnakes PPA..."
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt update -y

# echo "📌 Installing Python 3.10.8 and venv dependencies..."
# sudo apt install -y python3.10=3.10.8-1~22.04 python3.10-venv python3.10-distutils python3.10-dev

# echo "📌 Verifying Python installation..."
# python3.10 --version

# # ───────────────────────────────
# # Create project virtual environment
# # ───────────────────────────────
# VENV_DIR="project.venv"

# echo "📌 Creating virtual environment at $VENV_DIR..."
# python3.10 -m venv $VENV_DIR

# echo "📌 Activating virtual environment..."
# source $VENV_DIR/bin/activate

# echo "📌 Upgrading pip inside venv..."
# pip install --upgrade pip

# # ───────────────────────────────
# # Optional: Install requirements if requirements.txt exists
# # ───────────────────────────────
# if [ -f "requirements.txt" ]; then
#     echo "📌 Installing Python dependencies from requirements.txt..."
#     pip install -r requirements.txt
# else
#     echo "⚠️ No requirements.txt found, skipping dependency installation."
# fi

# echo "✅ Python 3.10.8 setup complete!"
# echo "💡 To activate venv in the future, run: source $VENV_DIR/bin/activate"
# python --version




#!/bin/bash
# setup_python310.sh
# ───────────────────────────────────────────────
# Minimal Python 3.10 setup for WM-811K project
# ───────────────────────────────────────────────

set -e

echo "📌 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y wget build-essential libssl-dev zlib1g-dev \
libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
libffi-dev libbz2-dev liblzma-dev uuid-dev tk-dev xz-utils

PYTHON_VERSION=3.10.8
PYTHON_TAR=Python-$PYTHON_VERSION.tgz

if [ ! -f $PYTHON_TAR ]; then
    echo "📥 Downloading Python $PYTHON_VERSION..."
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/$PYTHON_TAR
fi

echo "⚙️  Extracting Python..."
tar -xf $PYTHON_TAR

cd Python-$PYTHON_VERSION
echo "⚙️  Configuring build..."
./configure --enable-optimizations --with-ensurepip=install
echo "⚙️  Building Python (this may take a few minutes)..."
make -j$(nproc)
echo "⚙️  Installing Python 3.10 locally..."
sudo make altinstall

cd ..
echo "✅ Python 3.10 installation complete!"
python3.10 --version