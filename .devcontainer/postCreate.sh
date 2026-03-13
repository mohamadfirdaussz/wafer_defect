#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo "🚀 Setting up Wafer Defect Classification"
echo "==============================================="

echo "==> Upgrading pip"
python -m pip install --upgrade pip setuptools wheel --quiet

# Install from ml_flow/requirement.txt
if [ -f ml_flow/requirement.txt ]; then
  echo "==> Installing Python dependencies from ml_flow/requirement.txt"
  pip install -r ml_flow/requirement.txt --quiet
  echo "✅ All dependencies installed"
else
  echo "⚠️  ml_flow/requirement.txt not found!"
  echo "==> Installing baseline dependencies"
  pip install kaggle numpy pandas scikit-learn --quiet
fi

# Create expected folders
echo "==> Creating project directories"
mkdir -p ml_flow/datasets
mkdir -p data_loader_results
mkdir -p Feature_engineering_results
mkdir -p preprocessing_results
mkdir -p feature_selection_results
mkdir -p model_artifacts

echo "✅ postCreate complete - environment ready!"
