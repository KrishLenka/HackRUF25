# HackRUF25 - Skin Classification Project

AI-powered skin lesion classification using deep learning.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
See [TRAINING_SETUP.md](TRAINING_SETUP.md) for detailed instructions on organizing and training your model.

Quick command to organize your downloaded images:
```bash
cd backend
python prepare_data.py --mode binary \
  --healthy_dir /path/to/healthy/images \
  --unhealthy_dir /path/to/unhealthy/images \
  --target_dir data
```

### 3. Train the Model
```bash
cd backend
python train.py --num_classes 2 --epochs 50
```

### 4. Run the API
```bash
cd backend
python app.py
```

## 📚 Documentation
- **[TRAINING_SETUP.md](TRAINING_SETUP.md)**: Complete guide for dataset preparation and model training
- **backend/train.py**: Training script with data augmentation and early stopping
- **backend/evaluate.py**: Model evaluation with detailed metrics
- **backend/app.py**: Flask API for inference

## 🏗️ Project Structure
```
HackRUF25/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── prepare_data.py     # Data organization utility
│   ├── models/             # Trained models
│   ├── checkpoints/        # Training checkpoints
│   ├── logs/               # Training logs and plots
│   └── data/               # Dataset (train/val/test)
├── requirements.txt        # Python dependencies
└── TRAINING_SETUP.md       # Training guide
```

## 🔧 Features
- Binary and multi-class skin lesion classification
- Transfer learning with EfficientNet, ResNet, and more
- Data augmentation for better generalization
- Early stopping and learning rate scheduling
- Comprehensive evaluation metrics
- Multi-model ensemble inference (TensorFlow, PyTorch, ONNX)
- REST API for predictions