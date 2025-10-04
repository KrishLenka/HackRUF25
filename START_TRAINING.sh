#!/bin/bash

# Quick start script for training the 25-class skin condition model
# Make sure you're in the project root directory

echo "=================================="
echo "Starting Skin Condition Training"
echo "=================================="
echo ""
echo "Dataset: 25 classes"
echo "Train: 14,122 images"
echo "Val:    2,514 images"
echo "Test:   4,194 images"
echo ""
echo "Training will begin in 3 seconds..."
sleep 3

cd backend
python train.py --num_classes 25 --epochs 50

echo ""
echo "Training complete! Check backend/logs/ for results."
