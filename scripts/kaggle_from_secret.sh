#!/usr/bin/env bash
set -euo pipefail

# Configuration
DATASET="qingyi/wm811k-wafer-map"
OUTDIR="ml_flow/datasets"
ZIPNAME="wm811k-wafer-map.zip"
TARGET_PKL="$OUTDIR/LSWMD.pkl"

echo "==============================================="
echo "📥 Kaggle Dataset Download (Secret Method)"
echo "==============================================="

# Check for KAGGLE_JSON environment variable
if [ -z "${KAGGLE_JSON:-}" ]; then
  echo "❌ KAGGLE_JSON environment variable not set."
  echo ""
  echo "To set up a Codespace secret:"
  echo "  1. Go to your GitHub repo → Settings → Secrets → Codespaces"
  echo "  2. Click 'New repository secret'"
  echo "  3. Name: KAGGLE_JSON"
  echo "  4. Value: Paste entire contents of your kaggle.json file"
  echo "  5. Rebuild your Codespace"
  echo ""
  exit 1
fi

echo "✅ Found KAGGLE_JSON secret"
echo "==> Writing Kaggle credentials from secret"
mkdir -p ~/.kaggle
printf "%s" "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
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
