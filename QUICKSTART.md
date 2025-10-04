# Quick Start Guide - Training Your Skin Classification Model

## 📦 What You Have

You've downloaded datasets with healthy and unhealthy skin images. Here's how to get started training your model.

## ⚡ Fast Setup (3 Steps)

### Step 1: Install Dependencies
```bash
# From the project root directory
pip install -r requirements.txt
```

This installs PyTorch, TensorFlow, Flask, and all necessary libraries.

### Step 2: Organize Your Downloaded Images

You need to tell the script where your downloaded images are located. 

**Example:** If you have:
- Healthy images in: `/Users/klenka/Downloads/healthy_skin/`
- Unhealthy images in: `/Users/klenka/Downloads/unhealthy_skin/`

Run this command:
```bash
cd backend

python prepare_data.py \
  --mode binary \
  --healthy_dir /Users/klenka/Downloads/healthy_skin \
  --unhealthy_dir /Users/klenka/Downloads/unhealthy_skin \
  --target_dir data
```

**Replace the paths above with YOUR actual directories!**

This will automatically:
- Copy your images into the correct structure
- Split into 70% training, 15% validation, 15% testing
- Show you the distribution of images

Expected output:
```
Processing healthy images...
Found 1000 healthy images
  Copying 700 images to train/healthy...
  Copying 150 images to val/healthy...
  Copying 150 images to test/healthy...

Processing unhealthy images...
Found 1000 unhealthy images
...

TRAIN:
  Healthy: 700
  Unhealthy: 700
  Total: 1400
```

### Step 3: Train the Model
```bash
# Still in the backend directory
python train.py --num_classes 2 --epochs 50
```

Training will:
- Use GPU automatically if available (otherwise CPU)
- Show progress bars for each epoch
- Save the best model automatically
- Generate training plots

You'll see output like:
```
Device: cuda
Model: efficientnet_b0
Number of classes: 2

Epoch 1/50
Training: 100%|████████| 44/44 [00:15<00:00, 2.89it/s]
Train Loss: 0.6234 | Train Acc: 68.00%
Validation: 100%|████████| 10/10 [00:02<00:00, 4.12it/s]
Val Loss: 0.5123 | Val Acc: 74.00%
✓ Saved best model
```

Training takes:
- **CPU**: 2-5 hours for 50 epochs (depending on dataset size)
- **GPU**: 20-40 minutes for 50 epochs

You can stop training anytime with `Ctrl+C` and resume later.

## 🎯 What Happens Next

After training completes, you'll have:

✅ **Trained Model**: `backend/models/pt_skin_model.pth`
- This is ready to use with your Flask API

✅ **Training Logs**: `backend/logs/`
- Loss and accuracy plots
- Confusion matrix
- Classification report

✅ **Checkpoints**: `backend/checkpoints/`
- Full model checkpoint for resuming training

## 🧪 Test Your Model

### Evaluate on Test Set
```bash
python evaluate.py
```

This generates detailed metrics and visualizations in `evaluation/` folder.

### Start the API Server
```bash
python app.py
```

Server runs at `http://localhost:5000`

### Test with an Image
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/your/test/image.jpg"
```

Response:
```json
{
  "prediction": "unhealthy",
  "final_confidence": 0.89,
  "advice": {
    "short": "Suspicious lesion detected...",
    "urgency": "high"
  }
}
```

## 🔍 Where Are Your Downloaded Images?

If you're not sure where you downloaded your images, try:

```bash
# Search in Downloads folder
find ~/Downloads -type d -name "*skin*" -o -name "*healthy*" -o -name "*unhealthy*" 2>/dev/null

# Or search in common locations
find ~ -name "*.jpg" -path "*/skin*" 2>/dev/null | head -10
```

## 📊 Directory Structure After Setup

```
HackRUF25/
└── backend/
    ├── app.py                 # Your Flask API (already exists)
    ├── train.py              # Training script (new)
    ├── evaluate.py           # Evaluation script (new)
    ├── prepare_data.py       # Data organization (new)
    │
    ├── data/                 # Your organized dataset (created by prepare_data.py)
    │   ├── train/
    │   │   ├── healthy/      # Training healthy images
    │   │   └── unhealthy/    # Training unhealthy images
    │   ├── val/              # Same structure for validation
    │   └── test/             # Same structure for testing
    │
    ├── models/               # Trained models saved here
    │   └── pt_skin_model.pth
    │
    ├── checkpoints/          # Training checkpoints
    │   └── best_model.pth
    │
    └── logs/                 # Training logs and plots
        ├── loss_plot.png
        ├── accuracy_plot.png
        └── confusion_matrix_best.png
```

## ❓ Common Issues

### "No module named 'torch'"
```bash
pip install -r requirements.txt
```

### "Training data directory not found"
Make sure you ran `prepare_data.py` first and check that `backend/data/train/` exists.

### "CUDA out of memory"
Reduce batch size:
```bash
python train.py --num_classes 2 --batch_size 16
```

### "No images found"
Check your image directory paths are correct and contain `.jpg`, `.png`, or `.jpeg` files.

## 💡 Pro Tips

1. **Start Small**: Try with 100 images per class first to test the pipeline
2. **Monitor Training**: Watch the terminal output - validation accuracy should increase
3. **Early Stopping**: Training stops automatically if not improving (saves time!)
4. **GPU Recommended**: Training is 10x faster with a GPU
5. **Image Quality**: Better quality images = better model accuracy

## 📞 Need Help?

Check these files for more details:
- **TRAINING_SETUP.md** - Comprehensive training guide
- **backend/train.py** - See all training options
- **backend/EXAMPLE_SETUP.sh** - Automated setup script

Good luck! 🚀
