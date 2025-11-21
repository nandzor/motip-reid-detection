#!/bin/bash
# Script to copy ONNX model from parent directory to web-client/public/models/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_SOURCE="$PARENT_DIR/models/yolov8n.onnx"
MODEL_DEST="$SCRIPT_DIR/public/models/yolov8n.onnx"

echo "🔍 Looking for ONNX model..."

if [ -f "$MODEL_SOURCE" ]; then
    echo "✓ Found model at: $MODEL_SOURCE"
    mkdir -p "$(dirname "$MODEL_DEST")"
    cp "$MODEL_SOURCE" "$MODEL_DEST"
    echo "✅ Model copied to: $MODEL_DEST"
else
    echo "⚠️  Model not found at: $MODEL_SOURCE"
    echo ""
    echo "Please ensure the ONNX model exists, or export it using:"
    echo "  cd $PARENT_DIR"
    echo "  python setup_model.py"
    echo ""
    echo "Or manually place yolov8n.onnx in:"
    echo "  $SCRIPT_DIR/public/models/"
    exit 1
fi
