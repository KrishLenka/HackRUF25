# 🎮 GPU Support Guide

## ✅ Universal GPU Support

Your code now **automatically detects and uses the best available hardware**:

1. **NVIDIA GPU (CUDA)** - Preferred for NVIDIA cards
2. **Apple Silicon (MPS)** - For M1/M2/M3 Macs
3. **CPU** - Fallback if no GPU available

**No code changes needed!** The system automatically selects the best device.

## 🔍 Device Detection Priority

The code checks in this order:

```
1. Is NVIDIA CUDA available? → Use CUDA
2. Is Apple Silicon MPS available? → Use MPS
3. Otherwise → Use CPU
```

## 🖥️ Current System

Your current machine:
```
✗ CUDA Available: NO
✓ MPS Available: YES (Apple Silicon GPU)
Selected Device: MPS
PyTorch Version: 2.8.0
```

## 🚀 Training on Different Systems

### On Your Mac (Apple Silicon)
```bash
cd backend
python train.py --num_classes 25 --epochs 50
```

**Output:**
```
============================================================
TRAINING CONFIGURATION
============================================================
Device: MPS
GPU: Apple Silicon (Metal Performance Shaders)

Model: efficientnet_b0
Number of classes: 25
Batch size: 32
...
```

### On NVIDIA GPU System (Linux/Windows)
Same command works automatically:
```bash
cd backend
python train.py --num_classes 25 --epochs 50
```

**Output:**
```
============================================================
TRAINING CONFIGURATION
============================================================
Device: CUDA
GPU: NVIDIA GeForce RTX 3090
GPU Memory: 24.0 GB

Model: efficientnet_b0
Number of classes: 25
Batch size: 32
...
```

### On CPU-only System
Same command, no changes:
```bash
cd backend
python train.py --num_classes 25 --epochs 50
```

**Output:**
```
============================================================
TRAINING CONFIGURATION
============================================================
Device: CPU
GPU: Not available - using CPU

Model: efficientnet_b0
Number of classes: 25
Batch size: 32
...
```

## ⚙️ What Changed

### 1. `train.py` - Automatic Device Detection
```python
# Automatically detects: NVIDIA CUDA, Apple Silicon MPS, or CPU
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
    gpu_name = "Apple Silicon"
else:
    device = "cpu"
    gpu_name = "CPU only"
```

### 2. `app.py` - Flask API Device Detection
```python
# Automatically detect best available device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
```

### 3. `evaluate.py` - Evaluation Device Detection
Same automatic detection for model evaluation.

### 4. Smart Pin Memory
```python
# Pin memory only for CUDA (improves NVIDIA GPU performance)
use_pin_memory = (config.device == "cuda")
```

Pin memory is only enabled for CUDA because it's most beneficial there.

## 🔧 PyTorch Installation for Different GPUs

### For NVIDIA GPU (CUDA)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Or visit: https://pytorch.org/get-started/locally/

### For Apple Silicon (Already Installed)
```bash
# Standard PyTorch includes MPS support
pip install torch torchvision
```

### For CPU Only
```bash
# Standard installation
pip install torch torchvision
```

## 📊 Training Speed Comparison

For 50 epochs with 14,122 training images:

| Hardware | Time | Speed |
|----------|------|-------|
| **NVIDIA RTX 3090** | ~15-20 min | 🚀🚀🚀 Fastest |
| **NVIDIA RTX 3060** | ~25-35 min | 🚀🚀 Fast |
| **Apple M1/M2 Pro** | ~40-60 min | 🚀 Good |
| **CPU (16 cores)** | ~4-6 hours | 🐌 Slow |

## 🧪 Test Your GPU

### Quick Test Script
```bash
cd backend
python3 -c "
import torch

print('GPU Detection:')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

print(f'  MPS Available: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()}')
"
```

### Full Device Info
```bash
python3 -c "
import torch

if torch.cuda.is_available():
    print('NVIDIA CUDA')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'    Compute: {props.major}.{props.minor}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple Silicon MPS')
else:
    print('CPU only')
"
```

## 💡 Optimization Tips

### For NVIDIA GPU (CUDA)
```bash
# Increase batch size for better GPU utilization
python train.py --num_classes 25 --batch_size 64 --epochs 50

# Use mixed precision training (faster)
# (Not implemented yet, but easy to add with torch.cuda.amp)
```

### For Apple Silicon (MPS)
```bash
# Default batch size is good
python train.py --num_classes 25 --batch_size 32 --epochs 50

# MPS may have issues with some operations, fallback:
# Edit train.py, line 57: device = "cpu"
```

### For CPU
```bash
# Reduce batch size to avoid memory issues
python train.py --num_classes 25 --batch_size 8 --epochs 50

# Reduce workers
python train.py --num_classes 25 --batch_size 8 --num_workers 2 --epochs 50
```

## 🌍 Running on Different Computers

### Scenario 1: Train on NVIDIA GPU, Deploy on Mac

**On NVIDIA Computer:**
```bash
cd backend
python train.py --num_classes 25 --epochs 50
# → Creates: models/pt_skin_model.pth
```

**Transfer model to Mac:**
```bash
# Copy the model file
scp backend/models/pt_skin_model.pth user@mac:~/HackRUF25/backend/models/
```

**On Mac:**
```bash
cd backend
python app.py  # Automatically uses MPS or CPU
```

The model file works on **any device** - trained on CUDA, runs on MPS or CPU!

### Scenario 2: Train on Mac, Deploy on NVIDIA

Same process - the `.pth` model files are **device-independent**.

## 🔍 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --num_classes 25 --batch_size 16 --epochs 50
```

### MPS Fallback to CPU
If MPS has issues:
```python
# Edit backend/train.py, line 57:
device = "cpu"
```

### Check CUDA Installation
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
nvcc --version  # Should match PyTorch CUDA version
```

### Reinstall PyTorch with CUDA
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Expected Performance

### Training Time (50 epochs, 14K images)
- **RTX 3090**: 15-20 min
- **RTX 3080**: 20-25 min
- **RTX 3060**: 25-35 min
- **Apple M2 Pro**: 40-60 min
- **Apple M1**: 50-70 min
- **CPU (i7)**: 4-6 hours

### Memory Usage
- **Batch 32**: ~4-6 GB GPU memory
- **Batch 64**: ~8-12 GB GPU memory
- **Batch 16**: ~2-3 GB GPU memory

## ✅ Summary

✨ **Your code is now universal!**

- ✅ Works on **NVIDIA GPUs** (CUDA)
- ✅ Works on **Apple Silicon** (MPS)
- ✅ Works on **any CPU**
- ✅ Automatically detects best device
- ✅ No manual configuration needed
- ✅ Models are **device-independent**

Just run:
```bash
python train.py --num_classes 25 --epochs 50
```

It will automatically use the best available hardware! 🚀

---

## 📝 Quick Reference

```bash
# Check what device will be used
python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else ('MPS' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'CPU'))"

# Train (automatic device selection)
python train.py --num_classes 25 --epochs 50

# Force CPU (if needed)
# Edit train.py line 57: device = "cpu"
```

Your training will work on **any computer** now! 🎉
