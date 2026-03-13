#!/usr/bin/env bash
set -euo pipefail

# Configuration
DATASET="qingyi/wm811k-wafer-map"
OUTDIR="ml_flow/datasets"
ZIPNAME="wm811k-wafer-map.zip"
TARGET_PKL="$OUTDIR/LSWMD.pkl"

echo "==============================================="
echo "📥 Kaggle Dataset Download"
echo "==============================================="

# Check if dataset already exists
if [ -f "$TARGET_PKL" ]; then
  echo "✅ Dataset already exists: $TARGET_PKL"
  ls -lh "$TARGET_PKL"
  echo ""
  echo "To re-download, delete the file first:"
  echo "  rm $TARGET_PKL"
  exit 0
fi

# Try Method 1: Check for KAGGLE_JSON environment variable (Codespaces secret)
if [ -n "${KAGGLE_JSON:-}" ]; then
  echo "🔐 Using KAGGLE_JSON secret"
  mkdir -p ~/.kaggle
  printf "%s" "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json

# Try Method 2: Check for kaggle.json in repo root (uploaded file)
elif [ -f kaggle.json ]; then
  echo "📁 Using kaggle.json from repo root"
  mkdir -p ~/.kaggle
  cp kaggle.json ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json

# Try Method 3: Check if ~/.kaggle/kaggle.json already exists
elif [ -f ~/.kaggle/kaggle.json ]; then
  echo "✅ Using existing kaggle.json from ~/.kaggle/"

# No credentials found
else
  echo "❌ No Kaggle credentials found!"
  echo ""
  echo "Please choose ONE method:"
  echo ""
  echo "  Method A: Upload kaggle.json"
  echo "    1. Download from https://www.kaggle.com/settings (API section)"
  echo "    2. Upload to this Codespace (drag & drop into repo root)"
  echo "    3. Re-run this script"
  echo ""
  echo "  Method B: Use Codespace Secret"
  echo "    1. Go to GitHub repo → Settings → Secrets → Codespaces"
  echo "    2. Create secret: KAGGLE_JSON = <contents of kaggle.json>"
  echo "    3. Rebuild Codespace"
  echo ""
  exit 1
fi

# Install Kaggle CLI if needed
echo "==> Ensuring Kaggle CLI is installed"
pip install -q kaggle

# Create directory
echo "==> Creating dataset directory: $OUTDIR"
mkdir -p "$OUTDIR"

# Download
echo "==> Downloading dataset: $DATASET"
echo "    (This may take 1-2 minutes, ~150 MB)"
kaggle datasets download -d "$DATASET" -p "$OUTDIR" --force

# Extract
echo "==> Extracting dataset"
unzip -o "$OUTDIR/$ZIPNAME" -d "$OUTDIR/"
rm -f "$OUTDIR/$ZIPNAME"

# Verify
echo "==> Verifying dataset"
if [ -f "$TARGET_PKL" ]; then
  echo ""
  echo "✅ Dataset ready!"
  ls -lh "$TARGET_PKL"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Next step: Run the pipeline"
  echo "    python run_all.py"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
else
  echo "❌ Expected file not found: $TARGET_PKL"
  echo ""
  echo "Contents of $OUTDIR:"
  ls -lah "$OUTDIR"
  exit 1
fi
