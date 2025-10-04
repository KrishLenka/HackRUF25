#!/bin/bash

# Example script showing how to set up and train your skin classification model
# 
# Before running this script:
# 1. Replace the paths below with your actual image directories
# 2. Make sure you have downloaded your healthy and unhealthy skin images
# 3. Install dependencies: pip install -r ../requirements.txt

echo "==========================================="
echo "Skin Classification Model - Setup Example"
echo "==========================================="

# Step 1: Organize your downloaded dataset
echo ""
echo "Step 1: Organizing dataset..."
echo "Replace these paths with your actual image directories:"
echo ""

# CHANGE THESE PATHS TO YOUR ACTUAL DIRECTORIES
HEALTHY_DIR="$HOME/Downloads/healthy_skin_images"
UNHEALTHY_DIR="$HOME/Downloads/unhealthy_skin_images"

# Check if directories exist
if [ ! -d "$HEALTHY_DIR" ]; then
    echo "❌ Error: Healthy images directory not found: $HEALTHY_DIR"
    echo ""
    echo "Please update HEALTHY_DIR in this script to point to your healthy images folder"
    echo "Example: HEALTHY_DIR=\"/Users/yourusername/Downloads/healthy_images\""
    exit 1
fi

if [ ! -d "$UNHEALTHY_DIR" ]; then
    echo "❌ Error: Unhealthy images directory not found: $UNHEALTHY_DIR"
    echo ""
    echo "Please update UNHEALTHY_DIR in this script to point to your unhealthy images folder"
    echo "Example: UNHEALTHY_DIR=\"/Users/yourusername/Downloads/unhealthy_images\""
    exit 1
fi

echo "✓ Found healthy images directory: $HEALTHY_DIR"
echo "✓ Found unhealthy images directory: $UNHEALTHY_DIR"
echo ""

# Organize the data
python prepare_data.py \
  --mode binary \
  --healthy_dir "$HEALTHY_DIR" \
  --unhealthy_dir "$UNHEALTHY_DIR" \
  --target_dir data \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Error organizing dataset"
    exit 1
fi

echo ""
echo "✓ Dataset organized successfully!"
echo ""

# Step 2: Train the model
echo "Step 2: Training the model..."
echo ""

python train.py \
  --data_dir data \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0001 \
  --model efficientnet_b0 \
  --num_classes 2

if [ $? -ne 0 ]; then
    echo "❌ Error during training"
    exit 1
fi

echo ""
echo "✓ Training completed!"
echo ""

# Step 3: Evaluate the model
echo "Step 3: Evaluating the model on test set..."
echo ""

python evaluate.py

if [ $? -ne 0 ]; then
    echo "❌ Error during evaluation"
    exit 1
fi

echo ""
echo "✓ Evaluation completed!"
echo ""

# Step 4: Show results
echo "==========================================="
echo "Setup and training completed successfully!"
echo "==========================================="
echo ""
echo "Your trained model is saved at:"
echo "  - models/pt_skin_model.pth (for app.py)"
echo ""
echo "Training logs and plots are in:"
echo "  - logs/"
echo ""
echo "Evaluation results are in:"
echo "  - evaluation/"
echo ""
echo "To start the Flask API server:"
echo "  python app.py"
echo ""
echo "To test a prediction:"
echo "  curl -X POST http://localhost:5000/predict -F \"file=@path/to/image.jpg\""
echo ""
echo "==========================================="
