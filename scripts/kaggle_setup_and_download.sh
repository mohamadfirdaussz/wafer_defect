#!/usr/bin/env bash
set -euo pipefail

# Configuration
DATASET="qingyi/wm811k-wafer-map"
OUTDIR="ml_flow/datasets"
ZIPNAME="wm811k-wafer-map.zip"
TARGET_PKL="$OUTDIR/LSWMD.pkl"

echo "==============================================="
echo "📥 Kaggle Dataset Download (Upload Method)"
echo "==============================================="

# Check for kaggle.json in repo root
if [ ! -f kaggle.json ]; then
  echo "❌ kaggle.json not found in repo root."
  echo ""
  echo "Please:"
  echo "  1. Download kaggle.json from https://www.kaggle.com/settings (API section)"
  echo "  2. Upload it to the Codespace file explorer (drag & drop into repo root)"
  echo "  3. Re-run this script"
  echo ""
  exit 1
fi

echo "✅ Found kaggle.json"
echo "==> Setting up Kaggle credentials"
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo "==> Ensuring Kaggle CLI is installed"
pip install -q kaggle

echo "==> Creating dataset directory: $OUTDIR"
mkdir -p "$OUTDIR"

echo "==> Downloading dataset: $DATASET"
echo "    (This may take 1-2 minutes, ~150 MB)"
kaggle datasets download -d "$DATASET" -p "$OUTDIR" --force

echo "==> Extracting dataset"
unzip -o "$OUTDIR/$ZIPNAME" -d "$OUTDIR/"
rm -f "$OUTDIR/$ZIPNAME"

echo "==> Verifying dataset"
if [ -f "$TARGET_PKL" ]; then
  echo "✅ Dataset ready!"
  ls -lh "$TARGET_PKL"
  echo ""
  echo "Next step: Run the pipeline"
  echo "    python run_all.py"
  echo ""
else
  echo "❌ Expected file not found: $TARGET_PKL"
  echo ""
  echo "Contents of $OUTDIR:"
  ls -lah "$OUTDIR"
  exit 1
fi
