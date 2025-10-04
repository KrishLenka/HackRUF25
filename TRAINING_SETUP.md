# Skin Classification Model - Training Setup Guide

This guide will help you set up and train your skin lesion classification model.

## 📁 Expected Dataset Structure

Your dataset should be organized as follows:

```
data/
├── train/
│   ├── healthy/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── unhealthy/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── healthy/
│   │   └── ...
│   └── unhealthy/
│       └── ...
└── test/
    ├── healthy/
    │   └── ...
    └── unhealthy/
        └── ...
```

## 🚀 Quick Start

### 1. Install Dependencies

First, install all required packages:

```bash
pip install -r requirements.txt
```

### 2. Organize Your Dataset

If you have downloaded separate folders of healthy and unhealthy images, use the `prepare_data.py` script to organize them:

```bash
cd backend

python prepare_data.py \
  --mode binary \
  --healthy_dir /path/to/your/healthy/images \
  --unhealthy_dir /path/to/your/unhealthy/images \
  --target_dir data \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15
```

**Example:**
```bash
python prepare_data.py \
  --mode binary \
  --healthy_dir ~/Downloads/healthy_skin \
  --unhealthy_dir ~/Downloads/unhealthy_skin \
  --target_dir data
```

This will:
- Create the `data/` directory with `train/`, `val/`, and `test/` subdirectories
- Split your images into 70% training, 15% validation, 15% testing
- Organize them by class (healthy/unhealthy)

### 3. Train the Model

Once your data is organized, start training:

```bash
python train.py --num_classes 2
```

**With custom parameters:**
```bash
python train.py \
  --data_dir data \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0001 \
  --model efficientnet_b0 \
  --num_classes 2
```

### 4. Monitor Training

During training, you'll see:
- Real-time progress bars
- Training and validation metrics
- Best model checkpoints saved automatically

The training will:
- Use data augmentation (flips, rotations, color jitter)
- Apply early stopping if validation doesn't improve
- Save the best model based on validation accuracy
- Generate training plots and logs

### 5. Evaluate the Model

After training, evaluate on the test set:

```bash
python evaluate.py
```

This generates:
- Confusion matrix
- Classification report
- ROC curves
- Precision-recall curves
- Detailed metrics in JSON format

## 📊 Output Files

After training, you'll find:

```
backend/
├── models/
│   ├── pt_skin_model.pth          # Best model (for app.py)
│   ├── pt_skin_model_final.pth    # Final epoch model
│   └── model_config.json           # Model configuration
├── checkpoints/
│   └── best_model.pth              # Full checkpoint with optimizer
├── logs/
│   ├── training_history.json       # Training metrics
│   ├── loss_plot.png               # Loss curves
│   ├── accuracy_plot.png           # Accuracy curves
│   ├── confusion_matrix_best.png   # Confusion matrix
│   └── classification_report.txt   # Detailed metrics
└── evaluation/                      # Test set results
    ├── confusion_matrix_test.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── test_classification_report.txt
    └── test_metrics.json
```

## 🔧 Advanced Options

### Custom Model Architecture

Change the backbone architecture:
```bash
python train.py --model efficientnet_b3  # or resnet50, densenet121, etc.
```

Available models from `timm`:
- `efficientnet_b0` (default, fastest)
- `efficientnet_b3` (more accurate, slower)
- `resnet50`
- `densenet121`
- `mobilenetv3_large_100`

### Multi-class Classification

If you have more than 2 classes (e.g., different types of skin conditions):

1. Organize data:
```
data/
├── train/
│   ├── melanoma/
│   ├── nevus/
│   ├── bcc/
│   └── seborrheic_keratosis/
...
```

2. Train with the correct number of classes:
```bash
python train.py --num_classes 4
```

### Adjust Training Parameters

```bash
python train.py \
  --batch_size 64 \        # Larger batch (needs more GPU memory)
  --epochs 100 \           # Train longer
  --lr 0.0001 \            # Learning rate
  --data_dir /path/to/data # Custom data directory
```

## 🧪 Testing the Trained Model

### With the Flask API

1. Start the Flask server:
```bash
cd backend
python app.py
```

2. Test with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/test/image.jpg"
```

### Quick Inference Script

Create `test_inference.py`:
```python
import torch
from PIL import Image
from torchvision import transforms
from train import SkinClassificationModel

# Load model
model = SkinClassificationModel(num_classes=2)
model.load_state_dict(torch.load('models/pt_skin_model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('test_image.jpg').convert('RGB')
img_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs, dim=1)

print(f"Prediction: {'Unhealthy' if pred.item() == 1 else 'Healthy'}")
print(f"Confidence: {probs[0][pred].item():.2%}")
```

## 💡 Tips for Better Results

1. **More Data**: Aim for at least 500-1000 images per class
2. **Balanced Classes**: Try to have similar numbers of healthy and unhealthy images
3. **Image Quality**: Use clear, well-lit images
4. **Data Augmentation**: Already enabled by default
5. **Transfer Learning**: The model uses ImageNet pretrained weights (enabled by default)
6. **Early Stopping**: Training stops if validation doesn't improve for 10 epochs
7. **Learning Rate**: If training is unstable, try reducing `--lr 0.00005`

## 🐛 Troubleshooting

### Out of Memory Error
Reduce batch size:
```bash
python train.py --batch_size 16
```

### Training Too Slow
Use a smaller model:
```bash
python train.py --model mobilenetv3_large_100
```

### Low Accuracy
- Check if data is organized correctly
- Increase training epochs
- Ensure images are of good quality
- Try a larger model like `efficientnet_b3`
- Check class balance

### CUDA Not Available
The code automatically uses CPU if CUDA is not available. To use GPU:
1. Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
2. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

## 📈 Expected Results

With a good dataset (1000+ images per class), you should see:
- Training accuracy: 90-95%
- Validation accuracy: 85-90%
- Test accuracy: 85-90%

If results are much lower:
- Check data quality and labeling
- Ensure classes are balanced
- Train for more epochs
- Try different model architectures

## 🔄 Resume Training

To resume from a checkpoint:
```python
# Modify train.py to load checkpoint:
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 📝 Notes

- The model uses EfficientNet-B0 by default (lightweight and accurate)
- Training uses AdamW optimizer with learning rate scheduling
- Images are resized to 224x224 pixels
- Data augmentation includes flips, rotations, and color jitter
- Early stopping prevents overfitting
- All training metrics are logged and visualized

## 🎯 Next Steps

1. Train your model with your dataset
2. Evaluate on test set
3. Integrate with the Flask API (model is already compatible)
4. Build a frontend for user interaction
5. Consider deploying to cloud (AWS, GCP, Azure)

Good luck with your training! 🚀
