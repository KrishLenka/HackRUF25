# Data Directory

This directory contains your training dataset (25 skin condition classes).

## Expected Structure

```
data/
├── train/          # Training images (14,122 images)
│   ├── Acne and Rosacea Photos/
│   ├── Healthy_Skin/
│   ├── vitiligo/
│   └── ... (25 classes total)
├── val/            # Validation images (2,514 images)
│   └── (same 25 classes)
└── test/           # Test images (4,194 images)
    └── (same 25 classes)
```

## Note on Git

⚠️ **Dataset images are NOT tracked in git** - they're too large (~several GB).

## Setting Up Data on a New Computer

When you pull this repo on another computer, the data directories will be empty.

### Option 1: Transfer Dataset Manually

**From your Mac:**
```bash
# Compress the dataset
cd backend
tar -czf skin_data.tar.gz data/train data/val data/test

# Transfer to other computer
scp skin_data.tar.gz user@nvidia-computer:~/
```

**On NVIDIA computer:**
```bash
cd HackRUF25/backend
tar -xzf ~/skin_data.tar.gz
```

### Option 2: Cloud Storage

1. Upload `backend/data/` to Google Drive, Dropbox, etc.
2. Download on the new computer to `backend/data/`

### Option 3: Use Original Sources

If you have access to the original dataset downloads:
1. Download the dataset on the new computer
2. Run `python organize_dataset.py` to reorganize it

## Verify Dataset

After setting up data on the new computer:
```bash
cd backend
python3 -c "
from pathlib import Path
print(f\"Train images: {len(list(Path('data/train').rglob('*.jpg')))}\")
print(f\"Val images: {len(list(Path('data/val').rglob('*.jpg')))}\")
print(f\"Test images: {len(list(Path('data/test').rglob('*.jpg')))}\")
"
```

Expected output:
```
Train images: 14122
Val images: 2514
Test images: 4194
```

## Size Warning

The complete dataset is approximately **3-5 GB**.
Make sure you have enough disk space and bandwidth for transfer.
