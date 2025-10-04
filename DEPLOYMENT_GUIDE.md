# 🚀 Deployment Guide - Training on Another Computer

This guide shows you how to set up and train your model on a different computer (e.g., one with an NVIDIA GPU).

## 📋 Overview

Your repository is configured to:
- ✅ Include all code and scripts
- ✅ Include directory structure
- ✅ Exclude large files (data, models)
- ✅ Work on any computer with GPU or CPU

## 🔄 Step-by-Step Setup

### 1. Push Your Code to Git

**On your Mac:**
```bash
cd /Users/klenka/HackRUF25

# Add all code files
git add .

# Commit
git commit -m "Add training scripts with universal GPU support"

# Push to your remote (GitHub, GitLab, etc.)
git push origin main
```

### 2. Pull on the New Computer

**On NVIDIA computer (or any other computer):**
```bash
# Clone the repository
git clone <your-repo-url> HackRUF25
cd HackRUF25

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-training.txt
```

### 3. Transfer the Dataset

The dataset is **NOT in git** (too large). You need to transfer it manually.

#### Option A: Direct Transfer (Fastest)

**On your Mac:**
```bash
cd backend
tar -czf skin_data.tar.gz data/train data/val data/test
scp skin_data.tar.gz user@nvidia-computer:~/
```

**On NVIDIA computer:**
```bash
cd HackRUF25/backend
tar -xzf ~/skin_data.tar.gz
rm ~/skin_data.tar.gz
```

#### Option B: Cloud Storage

**On your Mac:**
```bash
# Upload to Google Drive, Dropbox, or similar
# Upload the entire backend/data/ folder
```

**On NVIDIA computer:**
```bash
# Download from cloud storage
# Extract to HackRUF25/backend/data/
```

#### Option C: Rsync (Incremental)

**From your Mac:**
```bash
rsync -avz --progress backend/data/ user@nvidia-computer:~/HackRUF25/backend/data/
```

### 4. Verify Setup

**On NVIDIA computer:**
```bash
cd HackRUF25/backend

# Check GPU detection
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ NVIDIA GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('✗ No CUDA GPU detected')
"

# Check dataset
python3 -c "
from pathlib import Path
train = len(list(Path('data/train').rglob('*.jpg')))
val = len(list(Path('data/val').rglob('*.jpg')))
test = len(list(Path('data/test').rglob('*.jpg')))
print(f'✓ Train: {train} images')
print(f'✓ Val: {val} images')
print(f'✓ Test: {test} images')
"
```

Expected output:
```
✓ NVIDIA GPU: NVIDIA GeForce RTX 3090
  Memory: 24.0 GB

✓ Train: 14122 images
✓ Val: 2514 images
✓ Test: 4194 images
```

### 5. Start Training

**On NVIDIA computer:**
```bash
cd backend
python train.py --num_classes 25 --epochs 50
```

The script will **automatically detect and use the NVIDIA GPU**! 🎉

Training time with NVIDIA GPU: ~15-30 minutes (vs 40-60 min on Mac)

## 📊 What's in Git vs. What's Not

### ✅ Tracked in Git (Small Files)
- All Python scripts (`.py` files)
- Requirements files
- Documentation (`.md` files)
- Configuration files
- Directory structure (via `.gitkeep`)

### ❌ NOT in Git (Large Files)
- Dataset images (`data/train/`, `data/val/`, `data/test/`)
- Trained models (`models/*.pth`)
- Training outputs (`logs/`, `checkpoints/`, `evaluation/`)
- Virtual environment (`venv/`)

## 🔄 After Training on NVIDIA Computer

### Transfer Model Back to Mac

**On NVIDIA computer:**
```bash
cd backend
scp models/pt_skin_model.pth user@mac:~/HackRUF25/backend/models/
scp models/model_config.json user@mac:~/HackRUF25/backend/models/
```

**On your Mac:**
```bash
cd backend
python app.py  # Use the model trained on NVIDIA GPU!
```

The model works on **any device** - train on NVIDIA, deploy on Mac! ✅

## 🌐 Cloud Training Options

### Google Colab
```python
# In a Colab notebook
!git clone <your-repo-url>
%cd HackRUF25
!pip install -r requirements-training.txt

# Upload dataset or mount Google Drive
# Then train
!cd backend && python train.py --num_classes 25 --epochs 50
```

### AWS / GCP / Azure
1. Launch GPU instance (e.g., AWS p3.2xlarge with V100)
2. SSH into instance
3. Follow "Pull on the New Computer" steps above
4. Train with CUDA acceleration

## 📝 Quick Reference

### Push Code from Mac
```bash
git add .
git commit -m "Update training code"
git push origin main
```

### Pull and Setup on New Computer
```bash
git clone <repo-url> HackRUF25
cd HackRUF25
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-training.txt
# Transfer dataset
cd backend
python train.py --num_classes 25 --epochs 50
```

### Transfer Dataset
```bash
# Quick transfer (from Mac)
cd backend
tar -czf skin_data.tar.gz data/train data/val data/test
scp skin_data.tar.gz user@other-computer:~/

# Extract (on other computer)
cd HackRUF25/backend
tar -xzf ~/skin_data.tar.gz
```

### Transfer Trained Model Back
```bash
# From NVIDIA computer to Mac
scp backend/models/pt_skin_model.pth user@mac:~/HackRUF25/backend/models/
```

## 💡 Tips

### 1. Use Git for Code, Not Data
- ✅ Code, scripts, configs → Git
- ❌ Data, models, outputs → Manual transfer or cloud storage

### 2. Compress Dataset
Dataset compression reduces transfer time:
```bash
tar -czf data.tar.gz data/  # ~1-2 GB compressed from ~5 GB
```

### 3. Incremental Sync
Use `rsync` for incremental updates:
```bash
rsync -avz --progress backend/data/ user@server:~/HackRUF25/backend/data/
```

### 4. Git LFS for Teams
If working in a team, consider Git LFS for model files:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

### 5. Cloud Storage
For frequent transfers, use cloud storage:
- Google Drive
- Dropbox
- AWS S3
- Azure Blob Storage

## 🎯 Complete Workflow Example

### Scenario: Train on NVIDIA Server, Deploy on Mac

**Day 1 - Setup:**
```bash
# On Mac: Push code
git push origin main

# On NVIDIA Server: Pull and setup
git clone <repo> && cd HackRUF25
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-training.txt

# Transfer dataset
scp -r mac-user@mac:~/HackRUF25/backend/data backend/
```

**Day 2 - Train:**
```bash
# On NVIDIA Server
cd backend
python train.py --num_classes 25 --epochs 50
# Training time: 20 minutes (vs 60 min on Mac!)
```

**Day 3 - Deploy:**
```bash
# Transfer model back to Mac
scp backend/models/pt_skin_model.pth mac-user@mac:~/HackRUF25/backend/models/

# On Mac: Start API
cd backend
python app.py
```

✨ **Result:** Fast training on NVIDIA GPU, model works perfectly on Mac!

## 🆘 Troubleshooting

### "No module named 'torch'"
```bash
source venv/bin/activate  # Activate venv first!
pip install -r requirements-training.txt
```

### "No such file or directory: data/train"
```bash
# Dataset not transferred yet
# Follow step 3: Transfer the Dataset
```

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --num_classes 25 --batch_size 16 --epochs 50
```

### Git shows "modified" for large files
```bash
# Check .gitignore is working
git check-ignore -v backend/data/train/*
# Should show: .gitignore:... backend/data/train/*
```

## ✅ Summary

1. **Code in Git** ✅ (push/pull normally)
2. **Data NOT in Git** ❌ (transfer manually)
3. **Models NOT in Git** ❌ (transfer after training)
4. **Universal GPU support** ✅ (works on any computer)
5. **Directory structure preserved** ✅ (via .gitkeep)

Your workflow:
```
Mac → Push code → NVIDIA Server
      Pull code ← 
      Transfer data →
      Train (fast!) →
      Transfer model ←
Mac → Use model
```

🎉 **You're ready to train on any computer!**
